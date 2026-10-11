"""Where the caller's words come from.

A three-way choice, not an assumption:

* **Script** — fixed turns. Cheapest and most reproducible, which is what RL
  rollouts and regression suites want.
* **Corpus** — real recorded human audio. Full-Duplex-Bench v3 uses this for
  disfluency because *TTS-generated "um" is not what a hesitation sounds
  like*. A simulator that can only speak synthesised text cannot port FDB, so
  corpus is a first-class source rather than an escape hatch.
* **LLM** — a persona driven by a model. Most realistic, least reproducible,
  and the most expensive per rollout.

Whatever the source, the text it yields is the **SAID** side of said-vs-heard:
the other party's own words, against which the system's
transcription is the hypothesis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Any
from typing import Protocol
from typing import runtime_checkable


class SourceKind(StrEnum):
    """Which of the three modes a source is. Recorded in the trajectory."""

    SCRIPT = "script"
    CORPUS = "corpus"
    LLM = "llm"


@dataclass(frozen=True, slots=True)
class Content:
    """One thing the caller is about to say.

    Attributes:
        text: The SAID side. For a corpus clip this is the annotation, which
            is still the reference for WER even though no TTS produced it.
        audio: Pre-recorded PCM, when the source is a corpus. When present,
            synthesis is skipped — you cannot synthesise a real hesitation.
        intent: What this turn is for, carried into the trajectory so a grader
            can ask whether it was addressed.
        metadata: Source-specific extras, e.g. a corpus clip's disfluency type.
    """

    text: str
    audio: bytes | None = None
    intent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_prerecorded(self) -> bool:
        return self.audio is not None


@runtime_checkable
class ContentSource(Protocol):
    """Yields what the caller says next, or ``None`` when the call is over."""

    @property
    def kind(self) -> SourceKind: ...

    @property
    def is_deterministic(self) -> bool:
        """Whether replaying with the same seed reproduces the same words."""
        ...

    async def next(self, heard: Sequence[str]) -> Content | None:
        """Produce the next utterance, given what the agent has said so far."""
        ...

    def reset(self) -> None:
        """Rewind for a new run."""
        ...


@dataclass(slots=True)
class ScriptContent:
    """Fixed turns, in order. Deterministic by construction."""

    turns: Sequence[str]
    intents: Sequence[str | None] = ()
    _index: int = field(default=0, init=False)

    @property
    def kind(self) -> SourceKind:
        return SourceKind.SCRIPT

    @property
    def is_deterministic(self) -> bool:
        return True

    async def next(self, heard: Sequence[str]) -> Content | None:
        if self._index >= len(self.turns):
            return None
        text = self.turns[self._index]
        intent = self.intents[self._index] if self._index < len(self.intents) else None
        self._index += 1
        return Content(text=text, intent=intent)

    def reset(self) -> None:
        self._index = 0


@dataclass(frozen=True, slots=True)
class CorpusClip:
    """One recorded utterance and its annotation."""

    text: str
    audio: bytes
    intent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CorpusContent:
    """Real recorded audio, played in order.

    The reason this exists: FDB v3's disfluency corpus is annotated human
    recordings, because a synthesised "um" is not a hesitation. Treat
    disfluency as a **corpus property, not a knob**.
    """

    clips: Sequence[CorpusClip]
    _index: int = field(default=0, init=False)

    @property
    def kind(self) -> SourceKind:
        return SourceKind.CORPUS

    @property
    def is_deterministic(self) -> bool:
        return True

    async def next(self, heard: Sequence[str]) -> Content | None:
        if self._index >= len(self.clips):
            return None
        clip = self.clips[self._index]
        self._index += 1
        return Content(
            text=clip.text,
            audio=clip.audio,
            intent=clip.intent,
            metadata=dict(clip.metadata),
        )

    def reset(self) -> None:
        self._index = 0


@dataclass(slots=True)
class PersonaContent:
    """An LLM speaking as a persona.

    The slot is defined here; a real model goes behind
    :class:`~harbor_voice.providers.LLMProvider`. Marked
    non-deterministic so anything that depends on reproducibility — RL
    rollouts, regression runs — can refuse it rather than discover the problem
    in the variance.
    """

    brief: str
    llm: Any
    max_turns: int = 20
    _spoken: int = field(default=0, init=False)

    @property
    def kind(self) -> SourceKind:
        return SourceKind.LLM

    @property
    def is_deterministic(self) -> bool:
        return False

    async def next(self, heard: Sequence[str]) -> Content | None:
        if self._spoken >= self.max_turns:
            return None
        last = heard[-1] if heard else ""
        chunks = [chunk async for chunk in self.llm.respond(last, list(heard))]
        self._spoken += 1
        text = " ".join(chunks).strip()
        return Content(text=text, intent=None) if text else None

    def reset(self) -> None:
        self._spoken = 0
