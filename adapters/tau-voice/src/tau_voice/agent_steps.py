"""What the caller hears of the agent, one step at a time.

A port of tau2's discrete-time tick template (``DiscreteTimeAdapter.
_async_run_tick`` with ``DiscreteTimeOpenAIAdapter._process_event`` and the
helpers in ``tick_result.py``), restated over provider-neutral
:mod:`~harbor_voice.transport.stream` events so any transport declaring
``EVENT_STREAM`` gets the same treatment. Per step:

- agent audio is capped at one step; the excess carries into the next step;
- the agent's transcript is handed out in proportion to the audio played;
- when the provider reports the caller starting to speak over agent audio,
  the unplayed rest of that utterance is dropped and the provider is told
  where it was cut.

:class:`LiveAgentStepper` is tau2's OpenAI Live tick,
:class:`GeminiAgentStepper` its Gemini Live tick and
:class:`CascadeAgentStepper` its cascaded (LiveKit) tick, each the same
template with that adapter's differences.

Ported rather than reimplemented: tau-voice's numbers depend on these rules,
including their quirks, and ``tests/adapters/tau-voice/fixtures/tau2_*_steps.json``
hold outputs recorded from tau2's own adapters that this module must match
byte for byte. One quirk is kept on purpose: the truncation point reported to
the provider is the length of the *padded* step, so it is always one full
step (200 ms) — that is what tau2 tells OpenAI, and the agent's view of the
conversation should match it.
"""

from __future__ import annotations

import audioop
import dataclasses
import math
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import override

from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import InputTranscript
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import ResponseDone
from harbor_voice.transport.stream import StreamEvent

MULAW_SILENCE = b"\x7f"
"""tau2's ``TELEPHONY_ULAW_SILENCE``."""

Chunk = tuple[bytes, str | None]


@dataclass(slots=True)
class _Utterance:
    """tau2's ``UtteranceTranscript``."""

    audio_received: int = 0
    transcript: str = ""
    audio_played: int = 0
    chars_shown: int = 0

    def text_for(self, played: int) -> str:
        self.audio_played += played
        if self.audio_received == 0 or not self.transcript:
            return ""
        ratio = min(1.0, self.audio_played / self.audio_received)
        end = max(self.chars_shown, math.ceil(ratio * len(self.transcript)))
        text = self.transcript[self.chars_shown : end]
        self.chars_shown = end
        return text


@dataclass(frozen=True, slots=True)
class Truncation:
    """Tell the provider this much of ``utterance`` was heard."""

    utterance: str
    audio_end_ms: int


@dataclass(frozen=True, slots=True)
class AgentStep:
    """The agent's side of one step, as the caller experiences it."""

    played: bytes
    """Exactly one step of audio, silence-padded."""
    raw_bytes: int
    """Agent audio actually present this step, before padding."""
    transcript: str
    was_truncated: bool
    truncated_bytes: int
    utterances: tuple[str, ...]
    vad: tuple[str, ...] = ()
    interruption_ms: int | None = None
    """Where in the caller's audio the barge-in was detected, if one was."""
    user_ms_at_start: int = 0
    """Caller audio sent to the agent before this step (ms)."""
    speech: bool | None = None
    """Speech activity, where the stepper measures it (tau2's
    ``TickResult.contains_speech``); ``None`` falls back to any audio."""

    @property
    def contains_speech(self) -> bool:
        """tau2's rule when no provider reports speech activity: any audio."""
        return self.speech if self.speech is not None else self.raw_bytes > 0

    def raw_data(self) -> dict[str, Any]:
        """tau2's ``TickResult`` fields as its agent stores them in ``raw_data``.

        The ones tau2's interaction metrics read: timing, VAD events, the
        transcript share and the truncation record.
        """
        return {
            "vad_events": list(self.vad),
            "proportional_transcript": self.transcript,
            "was_truncated": self.was_truncated,
            "truncated_audio_bytes": self.truncated_bytes,
            "interruption_audio_start_ms": self.interruption_ms,
            "cumulative_user_audio_at_tick_start_ms": self.user_ms_at_start,
        }


@dataclass(slots=True)
class _Step:
    chunks: list[Chunk]
    skip: str | None
    user_ms_at_start: int
    was_truncated: bool = False
    truncated_bytes: int = 0
    interruption_ms: int | None = None
    vad: list[str] = field(default_factory=list)


class AgentStepper:
    """Turns the agent's ordered output into per-step playback.

    Args:
        bytes_per_step: Agent audio per step in the session format.
        bytes_per_second: Of that format.
        silence: One sample of silence in that format.
    """

    truncates = True
    """Whether a barge-in cuts the agent off (and the provider is told)."""
    sends_whole = False
    """With ``convert_input``: send each step at once, not in paced packets."""

    def __init__(
        self,
        *,
        bytes_per_step: int,
        bytes_per_second: int,
        silence: bytes = MULAW_SILENCE,
    ) -> None:
        if bytes_per_step <= 0 or bytes_per_second <= 0:
            raise ValueError("bytes_per_step and bytes_per_second must be positive")
        self.bytes_per_step = bytes_per_step
        self.bytes_per_second = bytes_per_second
        self.silence = silence
        self._buffered: list[Chunk] = []
        self._skip: str | None = None
        self._user_ms = 0
        self._utterances: dict[str, _Utterance] = {}
        self._step: _Step | None = None

    # -------------------------------------------------------------- per step

    def begin(self) -> None:
        """Start a step: carry buffered audio and any skip from the last one."""
        if self._step is not None:
            raise RuntimeError("begin() called twice without end()")
        self._step = _Step(
            chunks=self._buffered,
            skip=self._skip,
            user_ms_at_start=self._user_ms,
        )
        self._buffered = []

    def feed(self, event: StreamEvent) -> Truncation | None:
        """Apply one event. Returns a truncation to send to the provider now."""
        step = self._current()
        if isinstance(event, OutputAudio):
            if step.skip is not None:
                if event.utterance == step.skip:
                    step.truncated_bytes += len(event.pcm)
                    return None
                step.skip = None
            step.chunks.append((event.pcm, event.utterance))
            if event.utterance:
                self._utterance(event.utterance).audio_received += len(event.pcm)
            return None
        if isinstance(event, OutputTranscript):
            if event.utterance and event.delta:
                self._utterance(event.utterance).transcript += event.delta
            return None
        if isinstance(event, InputSpeech):
            if not event.started:
                step.vad.append("speech_stopped")
                return None
            step.vad.append("speech_started")
            return self._barge_in(step, event.audio_ms)
        raise TypeError(f"not a stream event: {event!r}")

    def end(self, *, user_bytes: int) -> AgentStep:
        """Close the step. ``user_bytes``: caller audio sent this step."""
        step = self._current()
        self._buffered = self._cap(step)
        transcript = self._transcript(step.chunks)
        self._skip = step.skip
        self._user_ms += int(user_bytes / self.bytes_per_second * 1000)
        self._step = None
        seen: dict[str, None] = {}
        for _, utterance in step.chunks:
            if utterance:
                seen.setdefault(utterance)
        return AgentStep(
            played=self._played(step),
            raw_bytes=sum(len(data) for data, _ in step.chunks),
            transcript=transcript,
            was_truncated=step.was_truncated,
            truncated_bytes=step.truncated_bytes,
            utterances=tuple(seen),
            vad=tuple(step.vad),
            interruption_ms=step.interruption_ms,
            user_ms_at_start=step.user_ms_at_start,
        )

    # ------------------------------------------------------------- internals

    def _current(self) -> _Step:
        if self._step is None:
            raise RuntimeError("no step in progress; call begin() first")
        return self._step

    def _utterance(self, key: str) -> _Utterance:
        found = self._utterances.get(key)
        if found is None:
            found = self._utterances[key] = _Utterance()
        return found

    def _barge_in(self, step: _Step, audio_ms: int | None) -> Truncation | None:
        # tau2 also consults its carried buffer here; it is always empty at
        # this point because begin() moved it into the step. Kept for fidelity.
        if not (step.chunks or self._buffered):
            return None
        last = step.chunks[-1][1] if step.chunks else self._buffered[-1][1]
        if self._buffered:
            step.truncated_bytes += sum(len(data) for data, _ in self._buffered)
            self._buffered.clear()
        step.was_truncated = True
        step.skip = last
        step.interruption_ms = audio_ms if audio_ms is not None else 0
        received = sum(len(data) for data, _ in step.chunks)
        # Assignment, not +=: tau2's truncate_agent_audio overwrites here.
        step.truncated_bytes = max(0, received - len(self._played(step)))
        if last is None:
            return None
        played_ms = int(len(self._played(step)) / self.bytes_per_second * 1000)
        return Truncation(utterance=last, audio_end_ms=played_ms)

    def _played(self, step: _Step) -> bytes:
        """tau2's ``get_played_agent_audio``: exactly one step, padded."""
        size = self.bytes_per_step
        if step.was_truncated and step.interruption_ms is not None:
            within = step.interruption_ms - step.user_ms_at_start
            step_ms = size / self.bytes_per_second * 1000
            within = max(0, min(within, step_ms))
            limit = int(within * self.bytes_per_second / 1000)
            kept: list[bytes] = []
            total = 0
            for data, _ in step.chunks:
                if total >= limit:
                    break
                piece = data[: limit - total]
                kept.append(piece)
                total += len(piece)
            audio = b"".join(kept)
            return (
                audio + self.silence * (size - len(audio))
                if len(audio) < size
                else audio
            )
        raw = b"".join(data for data, _ in step.chunks)
        if not raw:
            return self.silence * size
        if len(raw) >= size:
            return raw[:size]
        return raw + self.silence * (size - len(raw))

    def _cap(self, step: _Step) -> list[Chunk]:
        """tau2's ``buffer_excess_audio``: keep one step, carry or drop the rest."""
        size = self.bytes_per_step
        keep: list[Chunk] = []
        total = 0
        if step.was_truncated:
            dropped = 0
            for data, utterance in step.chunks:
                if total + len(data) <= size:
                    keep.append((data, utterance))
                    total += len(data)
                else:
                    room = size - total
                    if room > 0:
                        keep.append((data[:room], utterance))
                        dropped += len(data) - room
                    else:
                        dropped += len(data)
                    total = size
            step.chunks = keep
            step.truncated_bytes += dropped
            return []
        carry: list[Chunk] = []
        for index, (data, utterance) in enumerate(step.chunks):
            if total + len(data) <= size:
                keep.append((data, utterance))
                total += len(data)
                continue
            room = size - total
            if room > 0:
                keep.append((data[:room], utterance))
                carry.append((data[room:], utterance))
            else:
                carry.append((data, utterance))
            carry.extend(step.chunks[index + 1 :])
            break
        step.chunks = keep
        return carry

    def _transcript(self, chunks: list[Chunk]) -> str:
        """tau2's ``get_proportional_transcript``."""
        per_utterance: dict[str, int] = {}
        for data, utterance in chunks:
            if utterance:
                per_utterance[utterance] = per_utterance.get(utterance, 0) + len(data)
        parts: list[str] = []
        for utterance, played in per_utterance.items():
            tracked = self._utterances.get(utterance)
            if tracked is not None:
                text = tracked.text_for(played)
                if text:
                    parts.append(text)
        return " ".join(parts)


class _PCMStepper(AgentStepper):
    """A tick for a provider that speaks PCM rather than the phone line's μ-law.

    tau2's ``StreamingTelephonyConverter`` both ways, one instance for the
    call: the caller's μ-law step into the provider's PCM
    (:meth:`convert_input`), the provider's PCM into μ-law as it is fed.
    Neither side cuts an utterance short in the provider, so there is
    nothing to truncate.
    """

    truncates = False

    def __init__(
        self,
        *,
        bytes_per_step: int,
        bytes_per_second: int,
        silence: bytes = MULAW_SILENCE,
        input_rate: int,
        output_rate: int,
    ) -> None:
        super().__init__(
            bytes_per_step=bytes_per_step,
            bytes_per_second=bytes_per_second,
            silence=silence,
        )
        from tau_voice.tau2_audio.audio_converter import StreamingTelephonyConverter

        self._converter = StreamingTelephonyConverter(
            input_sample_rate=input_rate, output_sample_rate=output_rate
        )

    def convert_input(self, ulaw: bytes) -> bytes:
        """The caller's step, as the provider takes it (tau2's ``convert_input``)."""
        return self._converter.convert_input(ulaw)


class LiveAgentStepper(_PCMStepper):
    """tau2's ``DiscreteTimeOpenAILiveAdapter`` tick, for OpenAI Live.

    The Realtime tick (:class:`AgentStepper`) with tau2's four Live changes:

    - agent audio arrives as 24 kHz PCM16 and goes through tau2's streaming
      telephony converter to μ-law; the same converter turns the caller's
      μ-law into the provider's PCM (:meth:`convert_input`);
    - the transcript is released by audio position: a delta counts as spoken
      once the audio played so far reaches the point where it arrived;
    - speech is measured, not inferred from audio presence: WebRTC carries
      silence too, so a step speaks if its audio peaks above 32;
    - the caller's speech (Live's input transcript) is only recorded as a VAD
      event. Live never truncates, so there is nothing to cut and nothing to
      tell the provider.
    """

    sends_whole = True

    def __init__(
        self,
        *,
        bytes_per_step: int,
        bytes_per_second: int,
        silence: bytes = MULAW_SILENCE,
        provider_rate: int = 24_000,
    ) -> None:
        super().__init__(
            bytes_per_step=bytes_per_step,
            bytes_per_second=bytes_per_second,
            silence=silence,
            input_rate=provider_rate,
            output_rate=provider_rate,
        )
        self.provider_rate = provider_rate
        self._deltas: deque[tuple[int, str]] = deque()
        self._played_bytes = 0

    @override
    def feed(self, event: StreamEvent) -> Truncation | None:
        if isinstance(event, InputSpeech):
            if event.started:
                self._current().vad.append("speech_started")
            return None
        if isinstance(event, OutputTranscript):
            if event.audio_position is not None:
                self._deltas.append((event.audio_position, event.delta))
            return None
        if isinstance(event, OutputAudio):
            ulaw = self._converter.convert_output(event.pcm)
            return super().feed(OutputAudio(ulaw, event.utterance, event.t))
        return super().feed(event)

    @override
    def end(self, *, user_bytes: int) -> AgentStep:
        step = super().end(user_bytes=user_bytes)
        # What actually played this step, before padding (``agent_audio_data``).
        audio = step.played[: step.raw_bytes]
        self._played_bytes += len(audio)
        played_pcm = (
            self._played_bytes * self.provider_rate * 2 // self.bytes_per_second
        )
        text = []
        while self._deltas and self._deltas[0][0] <= played_pcm:
            text.append(self._deltas.popleft()[1])
        pcm = audioop.ulaw2lin(audio, 2)
        return dataclasses.replace(
            step,
            transcript="".join(text),
            speech=bool(pcm) and audioop.max(pcm, 2) > 32,
        )


class CascadeAgentStepper(_PCMStepper):
    """tau2's ``LiveKitCascadedAdapter`` tick, for a duplex cascade.

    The Realtime tick's capping, carry-over and proportional transcript, with
    the cascaded adapter's own barge-in in place of truncation:

    - a reply's whole text arrives when its LLM call returns, and is handed
      out in proportion to the reply's audio as it plays;
    - when the caller starts speaking, or the recogniser reports a transcript,
      while agent audio is in this step or carried from the last, all of it is
      dropped, including what would have played this step, and every
      utterance's transcript is forgotten (``_check_barge_in``,
      ``_clear_agent_audio``). Nothing is said to the cascade: tau2 never
      cancels a reply;
    - audio of a forgotten utterance that arrives later still plays, without
      transcript: tau2 counts audio only for utterances it is tracking. Text
      that arrives later for it (a tool round's continuation) is tracked again.
    """

    sends_whole = True

    @override
    def feed(self, event: StreamEvent) -> Truncation | None:
        step = self._current()
        if isinstance(event, OutputAudio):
            ulaw = self._converter.convert_output(event.pcm)
            step.chunks.append((ulaw, event.utterance))
            tracked = self._utterances.get(event.utterance or "")
            if tracked is not None:
                tracked.audio_received += len(ulaw)
            return None
        if isinstance(event, OutputTranscript):
            if event.utterance and event.delta:
                self._utterance(event.utterance).transcript += event.delta
            return None
        if isinstance(event, InputSpeech):
            if not event.started:
                step.vad.append("speech_stopped")
                return None
            step.vad.append("speech_started")
        elif not isinstance(event, InputTranscript):
            raise TypeError(f"not a cascade stream event: {event!r}")
        if step.chunks or self._buffered:
            step.vad.append("interrupted")
            step.chunks.clear()
            self._buffered.clear()
            self._utterances.clear()
        return None


class GeminiAgentStepper(_PCMStepper):
    """tau2's ``DiscreteTimeGeminiAdapter`` tick, for Gemini Live.

    Gemini has no truncate call: it hears the caller start speaking, stops
    by itself and says so (``serverContent.interrupted``, which the transport
    passes on as a cancelled :class:`ResponseDone`). So, as in tau2:

    - the caller's μ-law goes through tau2's streaming converter to 16 kHz
      PCM16, and the agent's 24 kHz PCM16 back to μ-law;
    - on the interruption, the rest of the cut-off utterance is dropped and
      the step is marked truncated, with nothing sent to the provider;
    - the caller's own speech events are not recorded (tau2 ignores them);
    - each step goes out in paced 20 ms packets (tau2's default send).

    Audio is converted at the end of the step, after every event is applied,
    because tau2 does it in that order: an interruption resets the converter
    before that step's audio goes through it. Byte counts for the transcript
    share and for dropped audio use tau2's estimate (the PCM length scaled
    to μ-law), not the converter's exact output.
    """

    def __init__(
        self,
        *,
        bytes_per_step: int,
        bytes_per_second: int,
        silence: bytes = MULAW_SILENCE,
        input_rate: int = 16_000,
        output_rate: int = 24_000,
    ) -> None:
        super().__init__(
            bytes_per_step=bytes_per_step,
            bytes_per_second=bytes_per_second,
            silence=silence,
            input_rate=input_rate,
            output_rate=output_rate,
        )
        self.output_rate = output_rate
        self._speaking: str | None = None
        self._pending: list[Chunk] = []

    def _as_ulaw_bytes(self, pcm: bytes) -> int:
        return int(len(pcm) * self.bytes_per_second / (self.output_rate * 2))

    @override
    def feed(self, event: StreamEvent) -> Truncation | None:
        step = self._current()
        if isinstance(event, OutputAudio):
            if step.skip is not None and event.utterance == step.skip:
                step.truncated_bytes += self._as_ulaw_bytes(event.pcm)
                return None
            self._pending.append((event.pcm, event.utterance))
            if event.utterance:
                self._speaking = event.utterance
                self._utterance(event.utterance).audio_received += self._as_ulaw_bytes(
                    event.pcm
                )
            return None
        if isinstance(event, ResponseDone):
            if event.status == "cancelled":
                step.vad.append("interrupted")
                if self._buffered:
                    step.truncated_bytes += sum(len(d) for d, _ in self._buffered)
                    self._buffered.clear()
                step.was_truncated = True
                step.skip = self._speaking
                self._converter.reset()
            return None
        if isinstance(event, InputSpeech):
            return None
        if isinstance(event, OutputTranscript):
            return super().feed(event)
        raise TypeError(f"not a Gemini stream event: {event!r}")

    @override
    def end(self, *, user_bytes: int) -> AgentStep:
        step = self._current()
        for pcm, utterance in self._pending:
            step.chunks.append((self._converter.convert_output(pcm), utterance))
        self._pending.clear()
        return super().end(user_bytes=user_bytes)
