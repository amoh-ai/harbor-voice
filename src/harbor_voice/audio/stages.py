"""Stage marks — the cascade / S2S observability asymmetry, made explicit.

A cascade pipeline (STT -> LLM -> TTS) exposes seven seams. A speech-to-speech
model exposes two: audio went in, audio came out. That asymmetry is a property
of the system under test, and it must be **declared and recorded**, never
inferred from which marks happened to show up.

::

    stage                 cascade   s2s
    ────────────────────────────────────────────────
    vad_speech_start         ✓       ✓  if reported
    vad_speech_end           ✓       ✓  if reported
    asr_partial              ✓       —
    asr_final                ✓       —
    llm_first_token          ✓       —
    llm_done                 ✓       —
    tts_first_byte           ✓       —
    tts_done                 ✓       —
    response_start           ✓       ✓
    response_done            ✓       ✓
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from typing import Mapping
from typing import override

from harbor_voice.audio.clock import Clock
from harbor_voice.ids import PartyId

_EMPTY_META: Mapping[str, Any] = MappingProxyType({})


class Stage(StrEnum):
    """A named seam in a party's processing of a turn."""

    VAD_SPEECH_START = "vad_speech_start"
    VAD_SPEECH_END = "vad_speech_end"
    ASR_PARTIAL = "asr_partial"
    ASR_FINAL = "asr_final"
    LLM_FIRST_TOKEN = "llm_first_token"
    LLM_DONE = "llm_done"
    TTS_FIRST_BYTE = "tts_first_byte"
    TTS_DONE = "tts_done"
    RESPONSE_START = "response_start"
    RESPONSE_DONE = "response_done"


CASCADE_STAGES: frozenset[Stage] = frozenset(Stage)
"""Every stage is observable in a cascade — harbor-voice owns the seams."""

S2S_REQUIRED_STAGES: frozenset[Stage] = frozenset(
    {Stage.RESPONSE_START, Stage.RESPONSE_DONE}
)
"""All a speech-to-speech provider must expose."""

S2S_OPTIONAL_STAGES: frozenset[Stage] = frozenset(
    {Stage.VAD_SPEECH_START, Stage.VAD_SPEECH_END, Stage.ASR_FINAL}
)
"""Exposed by an S2S provider only when it volunteers them.

``ASR_FINAL`` is here because **Amazon Nova Sonic emits the transcript of its
own input** — a ``textOutput`` event with ``role: USER`` — and the original
assumption that a speech-to-speech model exposes no recognised text turned out
to be false for it.

The distinction that survives, and the one that actually matters, is between
*emitting a transcript* and *exposing a seam*. Nova Sonic hands you what it
heard; it does not let you replace the recogniser that heard it. So
said-vs-heard works against an S2S party, while the ablation — hold the LLM
fixed, swap the STT — still does not. ``CASCADE_ONLY_STAGES``
keeps the five seams that genuinely require harbor-voice to have composed the
pipeline itself.

``ASR_PARTIAL`` is deliberately **not** here: a provider that streams partials
is making a stronger claim, and no S2S provider has been verified to do so.
"""

S2S_STAGES: frozenset[Stage] = S2S_REQUIRED_STAGES | S2S_OPTIONAL_STAGES
"""The most a speech-to-speech provider can expose."""

CASCADE_ONLY_STAGES: frozenset[Stage] = CASCADE_STAGES - S2S_STAGES
"""Stages that exist only because harbor-voice composed the pipeline itself."""


@dataclass(frozen=True, slots=True)
class StageMark:
    """A stage, when it happened, and in whose clock domain.

    ``clock`` is carried because a provider-reported stage lands in a different
    domain from one we observed locally, and a latency built across the two
    must say so (see :class:`~harbor_voice.audio.clock.Interval`).
    """

    stage: Stage
    t: float
    party: PartyId
    clock: Clock = Clock.RX
    meta: Mapping[str, Any] = field(default=_EMPTY_META)

    def __post_init__(self) -> None:
        if self.t < 0:
            raise ValueError(f"stage {self.stage.value} has negative time {self.t}")

    @override
    def __repr__(self) -> str:
        return (
            f"StageMark({self.stage.value} @ {self.t:.3f}s "
            f"{self.party!r} [{self.clock.value}])"
        )
