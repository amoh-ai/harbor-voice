"""What a provider can do.

This is :mod:`harbor_voice.capabilities` applied one level down. A *party*
declares what it can do in a conversation; a *provider* declares what it can do
with audio or text, and the consequence is the same: **a missing capability
yields** ``not_applicable``, **never a zero**.

The concrete case that motivates it. ``tts_first_byte`` is one of the ten
cascade stages. Give the cascade a one-shot TTS — text in, finished blob out —
and ``tts_first_byte`` equals ``tts_complete``. The stage still reports a
number, the number is wrong, and nothing notices. Declaring ``streams=False``
turns that silent fiction into an explicit abstention.

**Capabilities belong to the class, not the instance.** ``doctor`` must
print what will be N/A before a container starts and without credentials, so a
provider states its limits through :meth:`ProviderSpec.capabilities` and never
by connecting first. A provider that can only describe itself once connected
cannot be preflighted, which defeats the most valuable command, ``doctor``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from harbor_voice.capabilities import Applicability
from harbor_voice.capabilities import Verdict


class Modality(StrEnum):
    """What a provider converts, which decides the slot it can fill."""

    STT = "stt"
    """Audio in, text out."""

    LLM = "llm"
    """Text in, text out."""

    TTS = "tts"
    """Text in, audio out."""

    S2S = "s2s"
    """Audio in, audio out — a transport, not a slot."""

    @property
    def is_slot(self) -> bool:
        """Whether this modality composes into a cascade.

        ``s2s`` does not: it is a :class:`~harbor_voice.transport.VoiceTransport`
        in its own right, exposing four stages instead of ten. Forcing it into
        a slot interface would mean inventing a request/response shape for
        something that has none.
        """
        return self is not Modality.S2S

    @property
    def consumes_audio(self) -> bool:
        return self in (Modality.STT, Modality.S2S)

    @property
    def produces_audio(self) -> bool:
        return self in (Modality.TTS, Modality.S2S)


class ProviderCapability(StrEnum):
    """A claim a provider can make, and the conformance suite can verify."""

    STREAMS = "streams"
    EMITS_PARTIALS = "emits_partials"
    WORD_TIMINGS = "word_timings"
    DIARIZES = "diarizes"


class ProviderCapabilityError(ValueError):
    """A provider declared a capability its modality cannot have."""


class ProviderCapabilities(BaseModel):
    """A provider's declared limits.

    Note:
        Consistency failures raise :class:`ProviderCapabilityError` inside the
        validator, which pydantic re-raises wrapped in ``ValidationError``.
        Catch ``ValidationError``; the message is preserved inside it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    modality: Modality

    streams: bool = Field(
        default=False,
        description=(
            "Produces output incrementally. Gates every *_first_byte and "
            "*_first_token stage: without it those marks are undefined, not 0."
        ),
    )
    emits_partials: bool = Field(
        default=False,
        description=(
            "Emits non-final transcripts. Gates after-words interruption and "
            "the backchannel selectivity metrics."
        ),
    )
    word_timings: bool = Field(
        default=False,
        description=(
            "Reports per-word time offsets. Valid for STT (word-level results) "
            "and for TTS -- Polly's speech marks are exactly this, and they are "
            "what makes per-word alignment of agent output possible without "
            "interpolating positions from byte offsets."
        ),
    )
    diarizes: bool = Field(
        default=False,
        description="Attributes audio to speakers. Needed for multi-party.",
    )

    attested: frozenset[ProviderCapability] = Field(
        default=frozenset(),
        description=(
            "Claims a conformance run has verified. A claim absent here "
            "is usable but cannot gate a published number -- rule 3."
        ),
    )

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> Self:
        if self.emits_partials and not self.modality.consumes_audio:
            raise ProviderCapabilityError(
                f"modality={self.modality.value} cannot emit partial "
                f"transcripts; it does not consume audio"
            )
        if self.emits_partials and not self.streams:
            raise ProviderCapabilityError(
                "emits_partials=True requires streams=True — a partial is an "
                "incremental result, so a one-shot provider cannot have one"
            )
        if self.diarizes and not self.modality.consumes_audio:
            raise ProviderCapabilityError(
                f"modality={self.modality.value} cannot diarize; it does not "
                f"consume audio"
            )
        if self.word_timings and self.modality is Modality.LLM:
            raise ProviderCapabilityError(
                "modality=llm cannot report word timings; it produces no audio "
                "for the words to be timed against"
            )

        claimed = {
            ProviderCapability.STREAMS: self.streams,
            ProviderCapability.EMITS_PARTIALS: self.emits_partials,
            ProviderCapability.WORD_TIMINGS: self.word_timings,
            ProviderCapability.DIARIZES: self.diarizes,
        }
        overclaimed = sorted(c.value for c in self.attested if not claimed[c])
        if overclaimed:
            raise ProviderCapabilityError(
                f"cannot attest {overclaimed}: not declared. Attesting a "
                f"capability the provider does not claim is how a conformance "
                f"report comes to certify something that was never run"
            )
        return self

    def has(self, capability: ProviderCapability) -> bool:
        """Whether this provider claims ``capability`` (declared, not probed)."""
        match capability:
            case ProviderCapability.STREAMS:
                return self.streams
            case ProviderCapability.EMITS_PARTIALS:
                return self.emits_partials
            case ProviderCapability.WORD_TIMINGS:
                return self.word_timings
            case ProviderCapability.DIARIZES:
                return self.diarizes

    def is_attested(self, capability: ProviderCapability) -> bool:
        """Whether a conformance run has verified ``capability`` (rule 3)."""
        return capability in self.attested

    def evaluate(self, *required: ProviderCapability) -> Applicability:
        """Decide whether something depending on ``required`` may report.

        Returns the same :class:`~harbor_voice.capabilities.Applicability` the
        party-level model returns, so a metric does not care which level made
        it inapplicable — only that it did, and why.

        Returns:
            ``NOT_APPLICABLE`` with a reason per missing capability;
            ``UNATTESTED`` when every capability is claimed but at least one is
            unverified; ``SATISFIED`` otherwise.
        """
        missing = [c for c in required if not self.has(c)]
        if missing:
            return Applicability(
                verdict=Verdict.NOT_APPLICABLE,
                reasons=tuple(f"provider does not support {c.value}" for c in missing),
            )
        unattested = [c for c in required if not self.is_attested(c)]
        if unattested:
            return Applicability(
                verdict=Verdict.UNATTESTED,
                reasons=tuple(
                    f"{c.value} is declared but no conformance run verified it"
                    for c in unattested
                ),
            )
        return Applicability(verdict=Verdict.SATISFIED)
