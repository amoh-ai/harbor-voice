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

A transport may also yield the **session events** below, but only when its
caller asks for them (OpenAI Realtime: ``RealtimeSettings.extended_events``).
A stepped consumer rejects a kind it does not know — deliberately, so nothing
is dropped unseen — so these never reach one that did not opt in:

- :class:`InputTranscript` — the provider's own transcript of our audio;
- :class:`OutputTranscriptDone` — an utterance's final transcript;
- :class:`ResponseDone` — a response finished, and how;
- :class:`ToolCall` — a tool call the transport answered, with its result.

A recorder that writes a benchmark's own log from a live call (EVA's audit
log is written from exactly these) needs them in this same order, which is
why they share the stream rather than arriving on a side channel.

**Consume ``events()`` or ``rx()``, not both** — they share one inbox.

:data:`OptionalOp.TRUNCATE` is the other half: telling the provider how much
of an utterance was actually heard before the caller cut in, so its own
history matches what the caller heard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    utterance: str | None
    delta: str
    t: float
    audio_position: int | None = None
    """Bytes of agent audio (``audio_out``) received before this delta, for a
    provider that places its transcript on the audio timeline rather than per
    utterance (OpenAI Live): the delta is spoken once playback reaches it."""


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


@dataclass(frozen=True, slots=True)
class InputTranscript:
    """The provider's transcript of a stretch of our audio, once final.

    Realtime: ``conversation.item.input_audio_transcription.completed``, or
    ``.failed`` with :attr:`error` set and empty text. ``text`` is exactly as
    the provider sent it — unstripped, possibly empty — because what a
    recorder skips is the benchmark's rule, not the transport's.
    """

    item_id: str | None
    text: str
    t: float
    error: str | None = None
    """Why transcription failed, when it did. A recorder that keeps the turn
    anyway (EVA writes a placeholder) needs to know it happened."""


@dataclass(frozen=True, slots=True)
class OutputTranscriptDone:
    """An agent utterance's whole transcript, as the provider closed it.

    Realtime: ``response.output_audio_transcript.done``. The deltas already
    went out as :class:`OutputTranscript`; this is the provider's final text,
    which can differ from their concatenation.
    """

    utterance: str | None
    text: str
    t: float


@dataclass(frozen=True, slots=True)
class ResponseDone:
    """A response finished. Realtime: ``response.done``.

    ``status`` is the provider's (``completed``, ``cancelled``, ``incomplete``,
    ``failed``); a cancelled response is one the caller cut off.
    ``has_function_calls`` says whether any output item was a tool call, and
    ``text`` is the text carried by the response's own output items — both
    computed exactly as EVA's ``openai_realtime_server`` computes them
    (``_response_has_function_calls``, ``_extract_response_text``), since its
    recorder falls back on that text when no transcript arrived.
    """

    response_id: str | None
    status: str
    has_function_calls: bool
    text: str
    t: float
    input_tokens: int | None = None
    """The response's own usage, when it reported any (``None`` otherwise).
    The transport also sums usage for the whole session."""
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call the transport ran for the agent and answered.

    Put on the stream once the answer is back and **before** it is sent to
    the model, so in a transport that runs calls in its receive loop it sits
    exactly where the call happened among the other events.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    """What the tools returned (an error is a result too; see ``is_error``)."""
    is_error: bool
    started: float
    """Session time the call went to the tools."""
    t: float
    """Session time the answer came back."""


StreamEvent = (
    OutputAudio
    | OutputTranscript
    | InputSpeech
    | InputTranscript
    | OutputTranscriptDone
    | ResponseDone
    | ToolCall
)
