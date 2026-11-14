"""The cascade on an open line: Deepgram finds the turns, as in tau2's cascaded agent.

:class:`~harbor_voice.cascade.transport.CascadeTransport` is told where a turn
ends (input stops arriving). On a full-duplex call the caller's line never
stops, silence included, so this transport streams the whole line to
Deepgram and runs the turn-taking of tau2's ``CascadedVoiceProvider``
(``tau2/voice/audio_native/livekit/provider.py`` at tau2 ``2174a60``), with
harbor-voice's slots in place of LiveKit's plugins:

- **Hearing.** One Deepgram stream per call
  (:class:`~harbor_voice.providers.deepgram.DeepgramCallStream`, with tau2's
  settings: ``nova-3``, ``endpointing`` 350 ms, VAD events, interim results)
  reports where the caller's speech starts and ends, and what was said. Turn
  detection is Deepgram's, as in tau2, so the STT slot must be
  ``deepgram.stt``; any other is refused when the cascade is built.
- **Turn end.** Each final transcript is added to what the caller has said
  since the agent last answered. When Deepgram ends the speech
  (``speech_final``) and that is at least ``min_transcript_chars`` long, the
  LLM is called with it. If a final transcript is left waiting
  ``utterance_end_ms`` (2 s) with no transcript after it, because the end of
  speech never came (noise keeps Deepgram's VAD open), the LLM is called
  anyway, unless a reply is being generated or spoken: tau2's fallback,
  checked as tau2 checks it, whenever caller audio arrives.
- **The conversation** is the agent's own and keeps everything, as tau2's
  ``ConversationContext`` does: what was heard, every reply in full, every
  tool call and result. A reply the caller cut off stays in it whole: tau2
  never tells its LLM what was not heard.
- **Tool calls.** A round with tool calls is not spoken, even if it has text;
  the calls go to the tools, and their results start the next round.
- **Barge-in** is not this transport's to handle. tau2's provider never
  cancels a reply (its ``interrupt()`` is never called); the tick adapter
  drops the agent audio still unplayed when the caller starts speaking or a
  transcript arrives. So this transport reports the caller's speech and
  transcripts on the event stream and keeps generating; the consumer decides
  what is played (tau-voice's ``CascadeAgentStepper``).
- **Overlap.** A turn that ends while a reply is still being generated starts
  its own reply alongside, as tau2's per-tick tasks do.

Events go out on one ordered stream (:data:`OptionalOp.EVENT_STREAM`), in
tau2's order: ``InputSpeech`` for the caller's speech starting and ending,
``InputTranscript`` for each interim (``is_final=False``) and final
transcript (a final comes before the end it belongs to, as Deepgram reports
them), ``OutputTranscript`` with a reply's whole text when the LLM returns
it, and ``OutputAudio`` as the TTS produces it. Replies are tagged
``utt_<n>``; ``n`` moves on when a reply has been synthesised, exactly as
tau2's adapter counts them.

Usage is metered per slot, as tau2's adapter meters it: seconds of audio
streamed to Deepgram, the LLM's tokens, and the characters sent to the TTS
(:meth:`DuplexCascadeTransport.usage_records`).

Differences from tau2 that follow from our slots:

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
  scoring the silence. A Deepgram stream that drops ends the call the same
  way, whatever the reason: LiveKit would reconnect, but a call with a hole
  in what the agent heard is not one to score.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any
from typing import override

from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import OptionalOp
from harbor_voice.cascade.transport import CascadeError
from harbor_voice.cascade.transport import CascadeTransport
from harbor_voice.ids import PartyId
from harbor_voice.providers import LLMProvider
from harbor_voice.providers import STTProvider
from harbor_voice.providers import Transcript
from harbor_voice.providers import TTSProvider
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.deepgram.stt import NAME as DEEPGRAM_STT
from harbor_voice.providers.deepgram.stt import TAU2_LISTENING
from harbor_voice.providers.deepgram.stt import DeepgramCallStream
from harbor_voice.providers.deepgram.stt import DeepgramSTT
from harbor_voice.providers.deepgram.stt import Heard
from harbor_voice.providers.deepgram.stt import ListenEvent
from harbor_voice.providers.deepgram.stt import ListenSettings
from harbor_voice.providers.tools import ChatLLM
from harbor_voice.providers.tools import ChatRound
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.providers.tools import ToolSpec
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import InputTranscript
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import StreamEvent

LINE = AudioFormat(sample_rate_hz=16_000)
"""What Deepgram is fed: tau2's ``DEFAULT_PCM_SAMPLE_RATE`` PCM16."""


def stt_refusal(stt: str) -> str | None:
    """Why a duplex cascade cannot take ``stt`` (a spec or a provider name),
    or ``None`` when it can."""
    if stt.split("/", 1)[0] == DEEPGRAM_STT:
        return None
    return (
        f"a duplex cascade's stt slot must be {DEEPGRAM_STT}, not {stt!r}: "
        f"turn detection is Deepgram's, as in tau2's cascaded agent. One "
        f"Deepgram stream per call reports where the caller's speech starts "
        f"and ends as well as what was said, and no other recogniser here "
        f"does both"
    )


@dataclass(frozen=True, slots=True)
class _Sent:
    """Caller audio went to Deepgram, which is when tau2 checks its
    utterance-end fallback."""


class DuplexCascadeTransport(CascadeTransport):
    """STT -> LLM -> TTS on a continuous line, with tau2's cascaded turn-taking.

    Args:
        stt: Recognition slot: ``deepgram.stt``, which hears the whole call
            on one stream and finds the turns.
        llm: Reasoning slot. Must implement
            :class:`~harbor_voice.providers.tools.ChatLLM`: the conversation,
            tool results included, is kept here and sent whole.
        tts: Synthesis slot.
        party: Who this agent speaks as.
        system_prompt: The conversation's system message.
        tools: The task's tools, if it has any.
        listening: The Deepgram stream's settings; tau2's by default.
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
        listening: ListenSettings = TAU2_LISTENING,
        min_transcript_chars: int = 1,
        audio_in: AudioFormat = LINE,
    ) -> None:
        if not isinstance(stt, DeepgramSTT):
            raise CascadeError(
                stt_refusal(stt.name)
                or f"the stt slot is a {type(stt).__name__}, not a DeepgramSTT"
            )
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
            emit_partials=True,
        )
        self._deepgram = stt
        self._chat = llm
        self._system_prompt = system_prompt
        self._listening = listening
        self._min_chars = min_transcript_chars
        self._audio_in = audio_in
        self._stream: DeepgramCallStream | None = None
        self._heard: asyncio.Queue[ListenEvent | _Sent | ProviderError] = (
            asyncio.Queue()
        )
        self._inbox: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._pump: asyncio.Task[None] | None = None
        self._listener: asyncio.Task[None] | None = None
        self._replies: set[asyncio.Task[None]] = set()
        self._messages: list[dict[str, Any]] = []
        self._unanswered = ""
        self._speaking = False
        self._last_heard: float | None = None
        self._busy = 0
        """Replies being generated or spoken: tau2's PROCESSING and SPEAKING."""
        self._utterance = 0
        self._stt_seconds = 0.0
        self._llm_requests = 0
        self._llm_tokens = {"prompt": 0, "cached": 0, "completion": 0}
        self._llm_metered = False
        """Whether the LLM reported tokens at all; if not, they stay ``None``."""
        self._tts_characters = 0
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

    def usage_records(self, component: str = "agent") -> list[UsageRecord]:
        """What each slot consumed so far, in the unit it is billed in.

        As tau2's cascaded adapter meters them: ``<component>.stt`` is the
        seconds of caller audio streamed to Deepgram, ``<component>.llm`` the
        LLM's calls and tokens (``None`` if it never reported any),
        ``<component>.tts`` the characters sent to the TTS (all of a reply's
        text, even if the caller cut it off).
        """
        seconds = self._stt_seconds
        if self._stream is not None:
            seconds += self._stream.audio_seconds
        tokens: dict[str, int | None] = (
            dict(self._llm_tokens)
            if self._llm_metered
            else dict.fromkeys(self._llm_tokens)
        )
        return [
            UsageRecord(
                component=f"{component}.stt",
                model=self._deepgram.model,
                seconds=round(seconds, 3),
            ),
            UsageRecord(
                component=f"{component}.llm",
                model=_model(self._llm),
                requests=self._llm_requests,
                input_tokens=tokens["prompt"],
                cached_input_tokens=tokens["cached"],
                output_tokens=tokens["completion"],
            ),
            UsageRecord(
                component=f"{component}.tts",
                model=_model(self._tts),
                characters=self._tts_characters,
            ),
        ]

    # ----------------------------------------------------------- lifecycle

    @override
    async def open(self, config: SessionConfig) -> None:
        if config.audio_in != self._audio_in:
            raise ValueError(
                f"this cascade listens to {self._audio_in}, not {config.audio_in}"
            )
        stream = self._deepgram.listen(self._audio_in, self._listening)
        await stream.open()
        self._stream = stream
        self._open = True
        self._heard = asyncio.Queue()
        self._inbox = asyncio.Queue()
        self._messages = (
            [{"role": "system", "content": self._system_prompt}]
            if self._system_prompt
            else []
        )
        self._unanswered = ""
        self._speaking = False
        self._last_heard = None
        self._utterance = 0
        self._pump = asyncio.create_task(self._read(stream))
        self._listener = asyncio.create_task(self._listen())

    @override
    async def close(self) -> None:
        tasks = [*self._replies, self._listener, self._pump]
        running = [task for task in tasks if task is not None]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        self._replies.clear()
        self._listener = self._pump = None
        if self._stream is not None:
            self._stt_seconds += self._stream.audio_seconds
            await self._stream.close()
            self._stream = None
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
        assert self._stream is not None
        try:
            await self._stream.send(frame.pcm)
        except ProviderError as exc:
            self._lost(exc)
            raise
        self._heard.put_nowait(_Sent())

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

    async def _read(self, stream: DeepgramCallStream) -> None:
        """Deepgram's events into the queue :meth:`_listen` takes in order."""
        try:
            async for event in stream.events():
                self._heard.put_nowait(event)
        except ProviderError as exc:
            self._heard.put_nowait(exc)

    async def _listen(self) -> None:
        """tau2's ``process_audio``: what Deepgram heard, in the order heard,
        with the fallback checked each time caller audio has gone out."""
        while True:  # cancelled on close
            item = await self._heard.get()
            if isinstance(item, ProviderError):
                self._lost(item)
                return
            if isinstance(item, _Sent):
                self._check_utterance_end()
            else:
                self._on_heard(item)

    def _on_heard(self, event: ListenEvent) -> None:
        now = self._clock.now()
        if event.kind is Heard.SPEECH_STARTED:
            self._speaking = True
            self._mark(Stage.VAD_SPEECH_START, now)
            self._emit(InputSpeech(True, event.audio_ms, now))
        elif event.kind is Heard.INTERIM:
            self._last_heard = now
            self._mark(Stage.ASR_PARTIAL, now, text=event.text)
            self._emit(InputTranscript(None, event.text, now, is_final=False))
        elif event.kind is Heard.FINAL:
            self._last_heard = now
            self._transcripts.append(Transcript(text=event.text, is_final=True))
            self._mark(Stage.ASR_FINAL, now, text=event.text)
            self._emit(InputTranscript(None, event.text, now))
            self._unanswered = f"{self._unanswered} {event.text}".strip()
        else:
            self._speaking = False
            self._mark(Stage.VAD_SPEECH_END, now)
            self._emit(InputSpeech(False, event.audio_ms, now))
            # tau2's _should_trigger_llm: enough heard, and speech has ended.
            if len(self._unanswered) >= self._min_chars:
                self._answer()

    def _check_utterance_end(self) -> None:
        """tau2's fallback for an end of speech that never came: a final
        transcript and ``utterance_end_ms`` with no transcript since, while
        no reply is being generated or spoken."""
        wait_ms = self._listening.utterance_end_ms
        if not self._unanswered or self._last_heard is None or wait_ms <= 0:
            return
        if (self._clock.now() - self._last_heard) * 1000 < wait_ms or self._busy:
            return
        self._speaking = False
        self._answer()

    def _answer(self) -> None:
        """tau2's ``_trigger_llm``: the reply starts with all that was heard."""
        said, self._unanswered = self._unanswered, ""
        self._last_heard = None
        self._start(self._respond(said))

    async def _respond(self, said: str) -> None:
        """tau2's ``_process_llm``, then ``_continue_after_tool`` per tool round."""
        self._messages.append({"role": "user", "content": said})
        while True:
            self._busy += 1
            try:
                round_ = await self._chat.chat(list(self._messages), self._specs())
            finally:
                self._busy -= 1
            self._count(round_)
            now = self._clock.now()
            self._mark(Stage.LLM_FIRST_TOKEN, now)
            self._mark(Stage.LLM_DONE, now, text=round_.text)
            if round_.text:
                # tau2's adapter takes a reply's text on LLM_COMPLETED, even
                # from a round that only calls tools and is never spoken.
                self._emit(OutputTranscript(self._utterance_id(), round_.text, now))
            if not round_.calls:
                if round_.text:
                    self._messages.append(_assistant(round_))
                break
            # All of a round's calls at once: they share a step, as tau2's do.
            results = await asyncio.gather(*map(self._run_tool, round_.calls))
            # The calls and their results join the conversation together: a
            # reply to the caller's next words can start while these tools
            # run, and a call without its result is a request OpenAI refuses.
            self._messages.append(_assistant(round_))
            self._messages.extend(results)
        if round_.text:
            # Billed on submission, as tau2 counts it, even if cut off.
            self._tts_characters += len(round_.text)
            self._busy += 1
            try:
                async for frame in self._synthesize(round_.text):
                    self._emit(
                        OutputAudio(frame.pcm, self._utterance_id(), frame.t_produced)
                    )
            finally:
                self._busy -= 1
            self._utterance += 1

    def _count(self, round_: ChatRound) -> None:
        usage = round_.usage
        self._llm_metered = self._llm_metered or bool(usage)
        details = usage.get("prompt_tokens_details") or {}
        self._llm_requests += 1
        self._llm_tokens["prompt"] += int(usage.get("prompt_tokens") or 0)
        self._llm_tokens["cached"] += int(details.get("cached_tokens") or 0)
        self._llm_tokens["completion"] += int(usage.get("completion_tokens") or 0)

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

    def _lost(self, exc: ProviderError) -> None:
        """The Deepgram stream died. Nothing is heard after this, so the call
        ends whatever the failure, retryable or not."""
        self.errors.append(f"stt: {type(exc).__name__}: {exc}")
        if self.failure is None:
            self.failure = exc

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


def _model(slot: object) -> str | None:
    model = getattr(slot, "model", None)
    return model if isinstance(model, str) else None
