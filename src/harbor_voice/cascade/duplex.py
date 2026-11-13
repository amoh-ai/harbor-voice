"""The cascade on an open line: it finds the turns, as tau2's cascaded agent does.

:class:`~harbor_voice.cascade.transport.CascadeTransport` is told where a turn
ends (input stops arriving). On a full-duplex call the caller's line never
stops, silence included, so this transport listens to the line itself and
runs the turn-taking of tau2's ``CascadedVoiceProvider``
(``tau2/voice/audio_native/livekit/provider.py`` at tau2 ``2174a60``), with
harbor-voice's slots in place of LiveKit's plugins:

- **Turn end.** Speech followed by 350 ms of non-speech
  (:mod:`~harbor_voice.cascade.endpointing`, Deepgram's ``endpointing_ms``)
  ends an utterance; it is transcribed, and its text added to what the
  caller has said since the agent last answered. If that is at least
  ``min_transcript_chars`` long, the LLM is called with it.
- **The conversation** is the agent's own and keeps everything, as tau2's
  ``ConversationContext`` does: what was heard, every reply in full, every
  tool call and result. A reply the caller cut off stays in it whole: tau2
  never tells its LLM what was not heard.
- **Tool calls.** A round with tool calls is not spoken, even if it has text;
  the calls go to the tools, and their results start the next round.
- **Barge-in** is not this transport's to handle. tau2's provider never
  cancels a reply (its ``interrupt()`` is never called); the tick adapter
  drops the agent audio still unplayed when the caller starts speaking. So
  this transport reports the caller's speech and transcripts on the event
  stream and keeps generating; the consumer decides what is played
  (tau-voice's ``CascadeAgentStepper``).
- **Overlap.** A turn that ends while a reply is still being generated starts
  its own reply alongside, as tau2's per-tick tasks do.

Events go out on one ordered stream (:data:`OptionalOp.EVENT_STREAM`), in
tau2's order: ``InputSpeech`` for the caller's speech starting and ending,
``InputTranscript`` for each final transcript (before the end it belongs
to, as Deepgram reports them), ``OutputTranscript`` with a reply's whole text
when the LLM returns it, and ``OutputAudio`` as the TTS produces it. Replies
are tagged ``utt_<n>``; ``n`` moves on when a reply has been synthesised,
exactly as tau2's adapter counts them.

Differences from tau2 that follow from our slots:

- Speech edges come from a level detector, not Deepgram's VAD model, and
  arrive without a network round trip.
- The recogniser hears a whole utterance once it has ended, so there are no
  interim transcripts, and tau2's ``utterance_end_ms`` fallback (a final
  transcript waiting for an end of speech that never came) cannot arise: the
  final arrives with the end.
- The LLM answers in one call (:class:`~harbor_voice.providers.tools.ChatLLM`)
  rather than streaming; tau2 waits for the whole reply before speaking too.
- An assistant turn with tool calls is one message carrying its text and its
  calls, which is how LiveKit sends tau2's context to the model.
- A slot that fails is recorded in :attr:`errors` and the call goes on, as
  tau2 logs a failed stage and carries on, when the failure can clear by
  itself (a rate limit, a timeout). One that cannot (a rejected key, a
  request the provider refuses) would fail every turn the same way, leaving
  the caller talking to silence for the rest of the call. That ends the
  call: the next audio sent raises it, so the trial errors rather than
  scoring the silence.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from collections.abc import Coroutine
from typing import Any
from typing import override

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import OptionalOp
from harbor_voice.cascade.endpointing import DEFAULT_ENDPOINTING_MS
from harbor_voice.cascade.endpointing import Endpointer
from harbor_voice.cascade.endpointing import SpeechEdge
from harbor_voice.cascade.transport import CascadeError
from harbor_voice.cascade.transport import CascadeTransport
from harbor_voice.ids import PartyId
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers import LLMProvider
from harbor_voice.providers import STTProvider
from harbor_voice.providers import TTSProvider
from harbor_voice.providers.tools import ChatLLM
from harbor_voice.providers.tools import ChatRound
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.providers.tools import ToolSpec
from harbor_voice.audio.clock import SessionClock
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import InputTranscript
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import StreamEvent

LINE = AudioFormat(sample_rate_hz=16_000)
"""What the recogniser is fed: tau2's ``DEFAULT_PCM_SAMPLE_RATE`` PCM16."""


class DuplexCascadeTransport(CascadeTransport):
    """STT -> LLM -> TTS on a continuous line, with tau2's cascaded turn-taking.

    Args:
        stt: Recognition slot; hears one finished utterance at a time.
        llm: Reasoning slot. Must implement
            :class:`~harbor_voice.providers.tools.ChatLLM`: the conversation,
            tool results included, is kept here and sent whole.
        tts: Synthesis slot.
        party: Who this agent speaks as.
        system_prompt: The conversation's system message.
        tools: The task's tools, if it has any.
        endpointing_ms: Non-speech that ends the caller's utterance.
        min_transcript_chars: tau2's ``TurnTakingConfig.min_transcript_chars``:
            less heard than this does not start a reply.
        audio_in: Of the caller's line; PCM16 mono.
    """

    def __init__(
        self,
        *,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        party: PartyId,
        system_prompt: str = "",
        tools: ToolRouter | None = None,
        clock: SessionClock | None = None,
        endpointing_ms: int = DEFAULT_ENDPOINTING_MS,
        min_transcript_chars: int = 1,
        audio_in: AudioFormat = LINE,
    ) -> None:
        if not isinstance(llm, ChatLLM):
            raise CascadeError(
                f"the llm slot {llm.name!r} cannot hold a conversation (it does "
                f"not implement ChatLLM), so a duplex cascade would forget every "
                f"tool result after the turn it came back in. openai.llm, "
                f"selfhosted.llm, litellm.llm and the offline echo LLMs can"
            )
        super().__init__(
            stt=stt,
            llm=llm,
            tts=tts,
            party=party,
            clock=clock,
            tools=tools,
        )
        self._chat = llm
        self._system_prompt = system_prompt
        self._endpointing_ms = endpointing_ms
        self._min_chars = min_transcript_chars
        self._audio_in = audio_in
        self._endpointer = Endpointer(audio_in, endpointing_ms=endpointing_ms)
        self._edges: asyncio.Queue[SpeechEdge] = asyncio.Queue()
        self._inbox: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._listener: asyncio.Task[None] | None = None
        self._replies: set[asyncio.Task[None]] = set()
        self._messages: list[dict[str, Any]] = []
        self._unanswered = ""
        self._utterance = 0
        self._caller = PartyId("caller")
        self.errors: list[str] = []
        """Every slot failure, as ``"<slot>: <error>"``."""
        self.failure: ProviderError | None = None
        """The first failure that cannot clear by itself; it ends the call."""

    @property
    @override
    def capabilities(self) -> Capabilities:
        """The cascade's, on a full-duplex line, with the ordered event stream."""
        return super().capabilities.model_copy(
            update={
                "duplex": frozenset({Duplex.FULL}),
                "optional_ops": frozenset({OptionalOp.EVENT_STREAM}),
                "audio_in": self._audio_in,
            }
        )

    @property
    def messages(self) -> tuple[dict[str, Any], ...]:
        """The conversation as the LLM sees it, in OpenAI chat format."""
        return tuple(self._messages)

    # ----------------------------------------------------------- lifecycle

    @override
    async def open(self, config: SessionConfig) -> None:
        if config.audio_in != self._audio_in:
            raise ValueError(
                f"this cascade listens to {self._audio_in}, not {config.audio_in}"
            )
        self._open = True
        self._endpointer = Endpointer(
            self._audio_in, endpointing_ms=self._endpointing_ms
        )
        self._edges = asyncio.Queue()
        self._inbox = asyncio.Queue()
        self._messages = (
            [{"role": "system", "content": self._system_prompt}]
            if self._system_prompt
            else []
        )
        self._unanswered = ""
        self._utterance = 0
        self._listener = asyncio.create_task(self._listen())

    @override
    async def close(self) -> None:
        tasks = [*self._replies, *([self._listener] if self._listener else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._replies.clear()
        self._listener = None
        self._inbox.put_nowait(None)
        await super().close()

    @override
    async def tx(self, frame: AudioFrame) -> None:
        self.require_open()
        if self.failure is not None:
            raise self.failure
        if frame.fmt != self._audio_in:
            raise ValueError(
                f"this cascade listens to {self._audio_in}, got a {frame.fmt} frame"
            )
        self._caller = frame.party
        for edge in self._endpointer.feed(frame.pcm):
            self._edges.put_nowait(edge)

    @override
    async def events(self) -> AsyncIterator[StreamEvent]:
        while (event := await self._inbox.get()) is not None:
            yield event

    @override
    async def rx(self) -> AsyncIterator[AudioFrame]:
        """The agent's audio alone, from the same stream as :meth:`events`."""
        seq = 0
        async for event in self.events():
            if isinstance(event, OutputAudio):
                yield AudioFrame(
                    pcm=event.pcm,
                    seq=seq,
                    party=self._party,
                    t_produced=event.t,
                    fmt=self._tts.audio_format,
                )
                seq += 1

    # -------------------------------------------------------------- engine

    async def _listen(self) -> None:
        """tau2's ``process_audio``: the caller's speech, in the order heard.

        One edge at a time, so a start never overtakes the end before it
        while that utterance is still being transcribed.
        """
        while True:  # cancelled on close
            edge = await self._edges.get()
            if edge.started:
                self._mark(Stage.VAD_SPEECH_START, self._clock.now())
                self._emit(InputSpeech(True, edge.at_ms, self._clock.now()))
                continue
            self._mark(Stage.VAD_SPEECH_END, self._clock.now())
            heard = await self._transcribe(edge.audio)
            if heard:
                self._emit(InputTranscript(None, heard, self._clock.now()))
                self._unanswered = f"{self._unanswered} {heard}".strip()
            self._emit(InputSpeech(False, edge.at_ms, self._clock.now()))
            # tau2's _should_trigger_llm: enough heard, and speech has ended.
            if len(self._unanswered) >= self._min_chars:
                said, self._unanswered = self._unanswered, ""
                self._start(self._respond(said))

    async def _transcribe(self, pcm: bytes) -> str:
        frame = AudioFrame(
            pcm=pcm,
            seq=len(self._transcripts),
            party=self._caller,
            t_produced=self._clock.now(),
            fmt=self._audio_in,
        )
        try:
            return (await self._hear([frame])).strip()
        except Exception as exc:  # noqa: BLE001 - recorded; tau2 carries on
            self._failed("stt", exc)
            return ""

    async def _respond(self, said: str) -> None:
        """tau2's ``_process_llm``, then ``_continue_after_tool`` per tool round."""
        self._messages.append({"role": "user", "content": said})
        while True:
            round_ = await self._chat.chat(list(self._messages), self._specs())
            now = self._clock.now()
            self._mark(Stage.LLM_FIRST_TOKEN, now)
            self._mark(Stage.LLM_DONE, now, text=round_.text)
            if round_.text:
                # tau2's adapter takes a reply's text on LLM_COMPLETED, even
                # from a round that only calls tools and is never spoken.
                self._emit(OutputTranscript(self._utterance_id(), round_.text, now))
            if round_.text or round_.calls:
                self._messages.append(_assistant(round_))
            if not round_.calls:
                break
            # All of a round's calls at once: they share a step, as tau2's do.
            results = await asyncio.gather(*map(self._run_tool, round_.calls))
            self._messages.extend(results)
        if round_.text:
            async for frame in self._synthesize(round_.text):
                self._emit(
                    OutputAudio(frame.pcm, self._utterance_id(), frame.t_produced)
                )
            self._utterance += 1

    async def _run_tool(self, call: ToolInvocation) -> dict[str, Any]:
        if self._tools is None:
            raise CascadeError(
                f"the model called {call.name!r}; no tools are configured"
            )
        result = await self._tools.invoke(call, at_s=self._clock.now())
        return {
            "role": "tool",
            "tool_call_id": call.tool_use_id,
            "content": result.as_text(),
        }

    def _specs(self) -> list[ToolSpec]:
        return list(self._tools.specs) if self._tools is not None else []

    def _utterance_id(self) -> str:
        """tau2's ``utt_<n>``, read when each event goes out, as tau2 reads it."""
        return f"utt_{self._utterance}"

    def _start(self, reply: Coroutine[Any, Any, None]) -> None:
        async def run() -> None:
            try:
                await reply
            except Exception as exc:  # noqa: BLE001 - recorded; tau2 carries on
                self._failed("reply", exc)

        task = asyncio.create_task(run())
        self._replies.add(task)
        task.add_done_callback(self._replies.discard)

    def _failed(self, slot: str, exc: Exception) -> None:
        self.errors.append(f"{slot}: {type(exc).__name__}: {exc}")
        if (
            isinstance(exc, ProviderError)
            and not exc.is_retryable
            and self.failure is None
        ):
            self.failure = exc

    def _emit(self, event: StreamEvent) -> None:
        self._inbox.put_nowait(event)


def _assistant(round_: ChatRound) -> dict[str, Any]:
    """The model's turn as an OpenAI chat message: its text and its calls."""
    message: dict[str, Any] = {"role": "assistant", "content": round_.text or None}
    if round_.calls:
        message["tool_calls"] = [
            {
                "id": call.tool_use_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in round_.calls
        ]
    return message
