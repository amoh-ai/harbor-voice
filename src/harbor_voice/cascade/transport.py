"""The reference cascade, as a transport.

STT -> LLM -> TTS, composed in-process, so harbor-voice owns every seam and
can mark all seven stages. A speech-to-speech provider exposes two.

Turn boundaries are detected by input-side tail silence: once no speech has
arrived for ``input_idle_s`` (a driver's ``filler`` frames do not count), the
accumulated audio is a turn and the pipeline
runs. That is deliberately the same rule the output side uses
(:func:`~harbor_voice.transport.drain.drain_until_idle`) — a turn ends when
somebody stops talking, on both sides.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import override

from harbor_voice.audio.clock import Clock
from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.audio.stages import StageMark
from harbor_voice.providers import LLMProvider
from harbor_voice.providers import STTProvider
from harbor_voice.providers import Transcript
from harbor_voice.providers import TTSProvider
from harbor_voice.providers.base import ProviderRef
from harbor_voice.providers.tools import ToolCallingLLM
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.ids import PartyId
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport

DEFAULT_INPUT_IDLE_S = 0.08


class CascadeError(RuntimeError):
    """The cascade was composed in a way that cannot serve the task."""


class CascadeTransport(VoiceTransport):
    """A composed STT -> LLM -> TTS agent behind the transport interface.

    Args:
        stt: Recognition slot. Swap this, hold the LLM, and you isolate
            mishearing from misreasoning.
        llm: Reasoning slot.
        tts: Synthesis slot.
        party: Who this agent speaks as.
        input_idle_s: Tail silence on the input side that ends a turn.
        emit_partials: Whether the STT's partials are surfaced. Must match the
            capability declaration, which the conformance suite checks.
    """

    def __init__(
        self,
        *,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        party: PartyId,
        clock: SessionClock | None = None,
        input_idle_s: float = DEFAULT_INPUT_IDLE_S,
        emit_partials: bool = False,
        tools: ToolRouter | None = None,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._party = party
        self._clock = clock or SessionClock()
        self._input_idle_s = input_idle_s
        self._emit_partials = emit_partials
        self._tools = tools

        self._open = False
        self._inbound: list[AudioFrame] = []
        self._outbound: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._watcher: asyncio.Task[None] | None = None
        self._last_input_at: float | None = None
        self._seq = 0
        self._history: list[str] = []
        self._stages: list[StageMark] = []
        self._transcripts: list[Transcript] = []

    # ------------------------------------------------------------ declared

    @property
    @override
    def capabilities(self) -> Capabilities:
        """A cascade sees every seam, so it declares every stage it emits."""
        stages = {
            Stage.VAD_SPEECH_START,
            Stage.VAD_SPEECH_END,
            Stage.ASR_FINAL,
            Stage.LLM_FIRST_TOKEN,
            Stage.LLM_DONE,
            Stage.TTS_FIRST_BYTE,
            Stage.TTS_DONE,
            Stage.RESPONSE_START,
            Stage.RESPONSE_DONE,
        }
        if self._emit_partials:
            stages.add(Stage.ASR_PARTIAL)
        return Capabilities(
            duplex=frozenset({Duplex.HALF, Duplex.FULL}),
            pipeline=Pipeline.CASCADE,
            turn_control=TurnControl.HARNESS_VAD,
            exposes_asr=True,
            exposes_partials=self._emit_partials,
            exposes_stages=frozenset(stages),
            can_be_interrupted=True,
            audio_in=CANONICAL,
            audio_out=self._tts.audio_format,
        )

    @property
    @override
    def is_open(self) -> bool:
        return self._open

    @property
    def stages(self) -> tuple[StageMark, ...]:
        """Stage marks emitted so far. Read by the adapter per turn."""
        return tuple(self._stages)

    @property
    def transcripts(self) -> tuple[Transcript, ...]:
        """Every final transcript, in order. The HEARD side."""
        return tuple(self._transcripts)

    def take_stages(self) -> tuple[StageMark, ...]:
        """Drain and return the stage marks accumulated since the last call."""
        marks = tuple(self._stages)
        self._stages.clear()
        return marks

    # ----------------------------------------------------------- lifecycle

    @override
    async def open(self, config: SessionConfig) -> None:
        self._open = True
        self._outbound = asyncio.Queue()
        self._inbound = []
        self._last_input_at = None
        self._watcher = asyncio.create_task(self._watch_for_turn_end())

    @override
    async def close(self) -> None:
        self._open = False
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
            self._watcher = None
        await self._outbound.put(None)
        # Each slot keeps one connection for the call; this is where it ends.
        for slot in (self._stt, self._llm, self._tts):
            if (aclose := getattr(slot, "aclose", None)) is not None:
                with contextlib.suppress(Exception):
                    await aclose()

    @override
    async def tx(self, frame: AudioFrame) -> None:
        self.require_open()
        # The driver's line silence between turns is not the caller talking;
        # the tail silence below is measured from speech.
        if frame.filler:
            return
        if not self._inbound:
            self._mark(Stage.VAD_SPEECH_START, self._clock.now())
        self._inbound.append(frame)
        self._last_input_at = self._clock.now()

    @override
    def rx(self) -> AsyncIterator[AudioFrame]:
        async def _gen() -> AsyncIterator[AudioFrame]:
            while True:
                frame = await self._outbound.get()
                if frame is None:
                    return
                yield frame

        return _gen()

    # -------------------------------------------------------------- engine

    async def _watch_for_turn_end(self) -> None:
        """Run the pipeline once the input side goes quiet."""
        while True:
            await asyncio.sleep(self._input_idle_s / 2)
            if not self._inbound or self._last_input_at is None:
                continue
            if self._clock.now() - self._last_input_at < self._input_idle_s:
                continue
            turn_audio = list(self._inbound)
            self._inbound = []
            self._last_input_at = None
            await self._run_pipeline(turn_audio)

    async def _run_pipeline(self, audio: list[AudioFrame]) -> None:
        """STT -> LLM -> TTS, marking each seam as it is crossed."""
        self._mark(Stage.VAD_SPEECH_END, self._clock.now())
        heard = await self._hear(audio)

        said_parts: list[str] = []
        first_token = True
        async for chunk in self._reply(heard):
            if first_token:
                self._mark(Stage.LLM_FIRST_TOKEN, self._clock.now())
                first_token = False
            said_parts.append(chunk)
        # Concatenated, not space-joined. A streaming LLM yields raw text
        # fragments ("Sure", ", I'll", " reschedule"), so inserting spaces
        # between them mangles the text: "Sure , I'll  reschedule". The
        # offline EchoLLM yields whole words, which made a space-join look
        # correct for every test until a real provider arrived.
        said = "".join(said_parts).strip()
        self._mark(Stage.LLM_DONE, self._clock.now(), text=said)
        self._history.extend((heard, said))

        async for frame in self._synthesize(said):
            await self._outbound.put(frame)

    async def _hear(self, audio: list[AudioFrame]) -> str:
        """The STT's final transcript of one utterance, marking ASR."""
        heard = ""
        async for partial in self._stt.transcribe(audio):
            if partial.is_final:
                heard = partial.text
                self._transcripts.append(partial)
                self._mark(Stage.ASR_FINAL, self._clock.now(), text=partial.text)
            elif self._emit_partials:
                self._mark(Stage.ASR_PARTIAL, self._clock.now(), text=partial.text)
        return heard

    async def _synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        """The TTS's audio for ``text`` as frames, marking TTS and the response."""
        first_byte = True
        emitted = 0
        async for pcm in self._tts.synthesize(text, party=self._party):
            now = self._clock.now()
            if first_byte:
                self._mark(Stage.TTS_FIRST_BYTE, now)
                self._mark(Stage.RESPONSE_START, now)
                first_byte = False
            frame = AudioFrame(
                pcm=pcm,
                seq=self._seq,
                party=self._party,
                t_produced=now,
                fmt=self._tts.audio_format,
            ).stamp(Clock.RX, now)
            self._seq += 1
            emitted += 1
            yield frame

        done_at = self._clock.now()
        self._mark(Stage.TTS_DONE, done_at)
        if emitted:
            self._mark(Stage.RESPONSE_DONE, done_at)

    def _reply(self, heard: str) -> AsyncIterator[str]:
        """The LLM's answer, with a tool loop when the task configured one.

        Routed here rather than branched inline so the stage marks above
        stay identical either way: ``llm_first_token`` means the first token
        of the reply, whether or not a tool ran on the way to it.

        A provider that cannot call tools is not silently given a tool-free
        turn. An undeclared optional raises rather than degrading,
        because a benchmark that is *entirely* tool-calling scored against an
        agent that was never offered its tools is a number about nothing.
        """
        if self._tools is None:
            return self._llm.respond(heard, self._history)
        if not isinstance(self._llm, ToolCallingLLM):
            raise CascadeError(
                f"this task configured tools but the llm slot "
                f"{self._llm.name!r} does not implement ToolCallingLLM; it "
                f"would answer without ever being offered them"
            )
        return self._llm.respond_with_tools(heard, self._history, router=self._tools)

    @property
    def ref(self) -> ProviderRef:
        """The three slots as one agent: provider ``cascade``, and as model
        each slot's ``provider/model`` joined by ``+``, the form tau-voice's
        job config names a cascade in."""
        return ProviderRef(
            provider="cascade",
            model="+".join(
                _slot_name(slot) for slot in (self._stt, self._llm, self._tts)
            ),
        )

    @property
    def tools(self) -> ToolRouter | None:
        """The tool router, if the task configured one."""
        return self._tools

    @property
    def heard(self) -> list[str]:
        """What this cascade's STT made of the caller's audio, per turn.

        The HEARD side of said-vs-heard. A cascade is the only
        pipeline that can serve this from the agent itself — an S2S model
        exposes no seam — so a cascade that kept its own transcript private
        would make the one metric it is uniquely able to support
        unavailable.

        Distinct from the reference STT, which answers a different question:
        not "what did this agent hear" but "what would a competent recogniser
        have got from the same degraded audio".
        """
        return self._history[0::2]

    @property
    def said(self) -> list[str]:
        """What the LLM produced and the TTS was asked to speak, per turn."""
        return self._history[1::2]

    def _mark(self, stage: Stage, t: float, **meta: object) -> None:
        self._stages.append(
            StageMark(
                stage=stage,
                t=t,
                party=self._party,
                clock=Clock.PRODUCED,
                meta=dict(meta),
            )
        )


def _slot_name(slot: object) -> str:
    """``provider/model``, or the provider alone when it has no model."""
    name = str(getattr(slot, "name", type(slot).__name__))
    model = getattr(slot, "model", None)
    return f"{name}/{model}" if isinstance(model, str) and model else name
