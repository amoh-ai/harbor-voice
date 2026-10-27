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

Ported rather than reimplemented: tau-voice's numbers depend on these rules,
including their quirks, and ``tests/unit/voice/fixtures/tau2_agent_steps.json``
holds outputs recorded from tau2's own adapter that this module must match
byte for byte. One quirk is kept on purpose: the truncation point reported to
the provider is the length of the *padded* step, so it is always one full
step (200 ms) — that is what tau2 tells OpenAI, and the agent's view of the
conversation should match it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field

from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
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

    @property
    def contains_speech(self) -> bool:
        """tau2's rule when no provider reports speech activity: any audio."""
        return self.raw_bytes > 0


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
