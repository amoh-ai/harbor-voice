"""The slot protocols — STT, LLM, TTS.

Three protocols, one per seam of a cascade. They live here rather than inside
``cascade/`` because the cascade is only one of their consumers:

===========================  =========  ====================================
Consumer                     Needs      Role
===========================  =========  ====================================
``CascadeTransport``         all three  the system under test
``UniversalSimulator``       TTS        the caller's voice — the other party
metrics (said-vs-heard)      STT        the measuring device
audio judge                  LLM        grading, after the call is over
===========================  =========  ====================================

The distinction that matters: the metrics STT is held fixed while the
agent's STT varies. That is the reference-cascade ablation — hold the LLM fixed, swap the
STT, and *misheard the drug name* separates from *does not know the drug*. It
only works if the two are different objects, which is why a module named after
one consumer could not hold them.

No speech-to-speech provider permits that ablation, because it exposes no seam
between hearing and reasoning.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from typing import runtime_checkable

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId


@dataclass(frozen=True, slots=True)
class Transcript:
    """What an STT heard. The HEARD side of said-vs-heard."""

    text: str
    is_final: bool = True
    confidence: float | None = None
    audio_duration_s: float = 0.0

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")


@runtime_checkable
class STTProvider(Protocol):
    """Audio in, text out. The seam an S2S model does not expose."""

    @property
    def name(self) -> str: ...

    def transcribe(self, frames: Sequence[AudioFrame]) -> AsyncIterator[Transcript]:
        """Yield zero or more partials, then exactly one final.

        Partials are what gate after-words interruption, so a provider that
        cannot emit them must not declare ``exposes_partials``.
        """
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Text in, streamed text out."""

    @property
    def name(self) -> str: ...

    def respond(self, prompt: str, history: Sequence[str]) -> AsyncIterator[str]:
        """Yield response fragments. The first marks ``llm_first_token``.

        Fragments concatenate: ``"".join(chunks)`` must be the reply.
        They are not words, and a consumer must not insert separators. Real
        streaming APIs emit arbitrary text deltas — ``"Sure"``, ``", I'll"``,
        ``" reschedule"`` — so a space-join produces ``"Sure , I'll
        reschedule"``. Any provider yielding word-sized pieces carries its own
        spacing.
        """
        ...


@runtime_checkable
class TTSProvider(Protocol):
    """Text in, audio out."""

    @property
    def name(self) -> str: ...

    @property
    def audio_format(self) -> AudioFormat: ...

    def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        """Yield PCM chunks. The first marks ``tts_first_byte``."""
        ...
