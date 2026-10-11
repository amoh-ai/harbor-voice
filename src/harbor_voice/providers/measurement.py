"""The measuring devices.

A benchmark needs instruments that are **not** the thing being measured. Two
of them, and both are providers:

``reference_stt``
    Transcribes audio the harness controls, to say what was *hearable*. For a
    speech-to-speech agent that exposes no ASR there is otherwise no "heard"
    side at all — and even for a cascade it separates *the line was
    bad* from *the agent was bad*, because it answers what a competent
    recogniser would have got from the same degraded audio.
``judge``
    Declared here so the independence rule covers it from the start.

**The rule this module exists to enforce: a measuring device must not be the
system under test.** Transcribe the agent's input with the agent's own STT and
the divergence is zero by construction — a number that is always perfect and
always meaningless. It is an easy mistake to make from a config file, where
both are just a spec string, and an expensive one to notice later, because the
metric does not crash. It reports success.
"""

from __future__ import annotations

from dataclasses import dataclass

from harbor_voice.capabilities import Applicability
from harbor_voice.capabilities import Verdict
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapability
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.providers.registry import parse_spec


class MeasurementIndependenceError(ValueError):
    """A measuring device is the same thing as the system under test."""


@dataclass(frozen=True, slots=True)
class ReferenceStack:
    """Instruments held fixed while the agent varies.

    Fixed is the operative word. Vary the reference STT between conditions and
    every cross-condition comparison silently stops being like-for-like, which
    is the same failure as changing a ruler between measurements.
    """

    reference_stt: str | None = None
    judge: str | None = None

    def check_independence(self, *, agent_stt: str | None) -> None:
        """Refuse a reference that is the agent's own recogniser.

        Compared at **provider** granularity, not provider-and-model: two
        models from one vendor share an acoustic frontend and most of their
        failure modes, so ``aws.transcribe/en-US`` is not an independent check
        on ``aws.transcribe/en-GB``. Correlated errors would cancel and the
        divergence would understate.

        Raises:
            MeasurementIndependenceError: If the two share a provider.
        """
        if self.reference_stt is None or agent_stt is None:
            return
        reference_provider, _ = parse_spec(self.reference_stt, default_model="_")
        agent_provider, _ = parse_spec(agent_stt, default_model="_")
        if reference_provider == agent_provider:
            raise MeasurementIndependenceError(
                f"reference STT and agent STT are both {reference_provider!r}; "
                f"the divergence would be measuring a recogniser against "
                f"itself and would report near-zero regardless of the audio. "
                f"Pick a different vendor for the reference."
            )

    def applicability(self, registry: ProviderRegistry) -> Applicability:
        """Whether the reference STT can support a divergence metric.

        A reference that cannot stream is still usable — the measurement runs
        offline over stored audio, where first-byte latency is meaningless.
        What it must do is produce a final transcript, which every STT does,
        so the only thing worth gating on here is that one was configured.
        """
        if self.reference_stt is None:
            return Applicability(
                Verdict.NOT_APPLICABLE,
                ("no reference STT configured; set --metrics-stt",),
            )
        entry, _ = registry.resolve(self.reference_stt)
        if entry.spec.modality is not Modality.STT:
            return Applicability(
                Verdict.NOT_APPLICABLE,
                (
                    f"reference_stt {self.reference_stt!r} is a "
                    f"{entry.spec.modality.value} provider",
                ),
            )
        return Applicability(Verdict.SATISFIED)

    def word_timing_applicability(self, registry: ProviderRegistry) -> Applicability:
        """Whether per-word alignment can be reported off the reference STT."""
        if self.reference_stt is None:
            return Applicability(
                Verdict.NOT_APPLICABLE, ("no reference STT configured",)
            )
        return registry.capabilities(self.reference_stt).evaluate(
            ProviderCapability.WORD_TIMINGS
        )
