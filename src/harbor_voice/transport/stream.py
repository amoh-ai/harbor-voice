"""The agent's output as one ordered stream — :data:`OptionalOp.EVENT_STREAM`.

``rx()`` yields audio and ``control()`` yields everything else, on two
iterators. That split is right for a free-running call and wrong for a
stepped one, because the step logic that tau-voice measures depends on the
*order* in which things arrived: an agent audio chunk that lands after the
provider reported the caller barging in belongs to a cancelled utterance and
must not be played. Two iterators cannot promise that order; one can.

A transport declaring ``EVENT_STREAM`` yields, in arrival order:

- :class:`OutputAudio` — agent audio, tagged with the utterance it belongs to;
- :class:`OutputTranscript` — the agent's own transcript of an utterance, as
  deltas. tau-voice's caller reads *this*, not a recognition of the audio;
- :class:`InputSpeech` — the provider's VAD opening or closing on our audio,
  with its position in the input audio buffer.

**Consume ``events()`` or ``rx()``, not both** — they share one inbox.

:data:`OptionalOp.TRUNCATE` is the other half: telling the provider how much
of an utterance was actually heard before the caller cut in, so its own
history matches what the caller heard.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutputAudio:
    """Agent audio in the session's ``audio_out`` format."""

    pcm: bytes
    utterance: str | None
    t: float
    """Session time it arrived."""


@dataclass(frozen=True, slots=True)
class OutputTranscript:
    """A transcript delta for one of the agent's own utterances."""

    utterance: str
    delta: str
    t: float


@dataclass(frozen=True, slots=True)
class InputSpeech:
    """The provider's VAD opened (``started``) or closed on our audio.

    ``audio_ms`` is the position in the audio *we sent* where it happened,
    as the provider counts it — which is what lets a stepped caller work out
    how much agent audio had played when the caller cut in.
    """

    started: bool
    audio_ms: int | None
    t: float


StreamEvent = OutputAudio | OutputTranscript | InputSpeech
