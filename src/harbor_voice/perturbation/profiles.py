"""Named conditions: a fixed set of perturbation layers under one name.

A profile is a **reproducible condition**. ``background_noise`` means the
same layers at the same parameters in every run, so two runs under it are
comparable, and ``summarise`` says which metrics it moves. "Some background
noise at about 15 dB" does not survive contact with a reviewer; a name does.

**Core's profiles are generic**, named for what they do. A benchmark's own
conditions — EVA's condition B, FDB's overlap set — belong to that
benchmark's adapter, which registers them (:func:`register_profile`) with the
benchmark's exact parameters, built from core layers. An adapter that
reproduces its benchmark another way registers none: tau-voice runs tau2's
own speech complexities through tau2's own effects code.

A profile may be **declared before it can be built** — a layer not
implemented yet, a corpus not available. It still lists, ``doctor`` reports
it, and only construction fails: "not yet" and "not supported" are different
answers, and a condition quietly missing from the list would give the wrong
one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field

from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import LayerKey
from harbor_voice.ids import PlanId
from harbor_voice.perturbation.base import SeedSet
from harbor_voice.perturbation.channel import GilbertElliottLoss
from harbor_voice.perturbation.channel import MuLawCodec
from harbor_voice.perturbation.conversation import ConversationPlan
from harbor_voice.perturbation.conversation import ProbeKind
from harbor_voice.perturbation.environment import BackgroundBed
from harbor_voice.perturbation.environment import synthetic_bed
from harbor_voice.perturbation.persona import Accent
from harbor_voice.perturbation.persona import PersonaPlan
from harbor_voice.perturbation.persona import VoiceSelection
from harbor_voice.perturbation.plan import PerturbationPipeline

Builder = Callable[[SeedSet], PerturbationPipeline]


class UnknownProfile(LookupError):
    """No profile is registered under that name."""


class DuplicateProfile(ValueError):
    """A profile is already registered under that name."""


class ProfileNotBuildable(NotImplementedError):
    """The profile is declared but needs a layer that does not exist yet."""


@dataclass(frozen=True, slots=True)
class Profile:
    """A named condition and the layers it switches on."""

    name: str
    layers: tuple[LayerKey, ...]
    description: str
    """What the condition is, in a line (``harbor-voice profiles`` shows it)."""
    build: Builder | None = None
    note: str = ""

    @property
    def is_buildable(self) -> bool:
        return self.build is not None

    @property
    def families(self) -> frozenset[Layer]:
        """Layer families this profile touches, for perturbation–metric pairing."""
        return frozenset(key.layer for key in self.layers)

    @property
    def is_clean(self) -> bool:
        """Whether this is the baseline every degraded run is compared to."""
        return not self.layers

    def pipeline(self, seeds: SeedSet) -> PerturbationPipeline:
        """Build the pipeline, or say which layer is missing.

        Raises:
            ProfileNotBuildable: Saying what is actually missing, because "not
                supported" and "not yet" are different answers and the caller
                deserves the second one.

                The reason comes from :attr:`note` when there is one. Inferring
                it from the layer list is right while a layer is unbuilt and
                wrong once it is not: a condition can need a recorded corpus
                although every layer it names exists.
        """
        if self.build is None:
            reason = self.note or (
                f"it needs {sorted({k.layer.value for k in self.layers})}, "
                f"which harbor-voice does not implement yet"
            )
            raise ProfileNotBuildable(
                f"profile {self.name!r} cannot be built: {reason}. Its layers "
                f"are declared so 'doctor' and 'profiles' can report on it"
            )
        return self.build(seeds)


def _clean(seeds: SeedSet) -> PerturbationPipeline:
    return PerturbationPipeline(seeds=seeds, plan_id=PlanId("clean"))


def _telephony(seeds: SeedSet) -> PerturbationPipeline:
    return PerturbationPipeline(
        [MuLawCodec(), GilbertElliottLoss(rate=0.02, burst_duration_ms=200.0)],
        seeds=seeds,
        plan_id=PlanId("telephony"),
    )


def _bed() -> BackgroundBed:
    """A room at 15 dB SNR that drifts ±3 dB, slowly."""
    return BackgroundBed(
        source=synthetic_bed(seconds=10.0, seed=7),
        target_snr_db=15.0,
        snr_drift_db=3.0,
        variation_speed=0.1,
    )


def _background_noise(seeds: SeedSet) -> PerturbationPipeline:
    return PerturbationPipeline(
        [_bed()], seeds=seeds, plan_id=PlanId("background_noise")
    )


def _overlap(seeds: SeedSet) -> PerturbationPipeline:
    """L5 only — the line is clean and the *conversation* is the perturbation.

    Nothing here degrades a sample, which is the point: an agent that handles
    a pristine line perfectly can still talk over its caller, and that
    failure is invisible to every signal-level condition.
    """
    return PerturbationPipeline(
        seeds=seeds,
        plan_id=PlanId("overlap"),
        conversation=ConversationPlan.of(
            ProbeKind.INTERRUPTION, ProbeKind.BACKCHANNEL, seeds=seeds
        ),
    )


def _selectivity_probes(seeds: SeedSet) -> PerturbationPipeline:
    """The three things an agent should hear and keep talking through: a
    backchannel, a vocal tic, and speech not directed at it.

    No interruption layer, deliberately. Mixing one in would mean every turn
    carried both a probe the agent should answer and probes it should ignore,
    and a failure would not say which instinct was wrong.
    """
    return PerturbationPipeline(
        seeds=seeds,
        plan_id=PlanId("selectivity_probes"),
        conversation=ConversationPlan.of(
            ProbeKind.BACKCHANNEL,
            ProbeKind.VOCAL_TIC,
            ProbeKind.NON_DIRECTED,
            seeds=seeds,
        ),
    )


ACCENT = Accent.INDIAN
"""The accent of the generic accented conditions: one fixed choice, so the
condition means the same in every run. Another accent is another condition —
register it as a profile of its own."""


def _accented(seeds: SeedSet) -> PerturbationPipeline:
    """An accented caller on an otherwise clean line: accent only, no change
    of behaviour, so a difference is the accent's."""
    return PerturbationPipeline(
        seeds=seeds,
        plan_id=PlanId("accented"),
        persona=PersonaPlan(voice=VoiceSelection(accent=ACCENT)),
    )


def _accented_noisy(seeds: SeedSet) -> PerturbationPipeline:
    """The accented caller, now in a noisy room."""
    return PerturbationPipeline(
        [_bed()],
        seeds=seeds,
        plan_id=PlanId("accented_noisy"),
        persona=PersonaPlan(voice=VoiceSelection(accent=ACCENT)),
    )


def _noisy_line(seeds: SeedSet) -> PerturbationPipeline:
    """L3 + L4 together — a noisy room over a lossy phone line."""
    return PerturbationPipeline(
        [
            BackgroundBed(
                source=synthetic_bed(seconds=10.0, seed=7),
                target_snr_db=10.0,
                snr_drift_db=2.0,
                variation_speed=0.1,
            ),
            MuLawCodec(),
            GilbertElliottLoss(rate=0.05, burst_duration_ms=200.0),
        ],
        seeds=seeds,
        plan_id=PlanId("noisy_line"),
    )


def _keys(*raw: str) -> tuple[LayerKey, ...]:
    return tuple(LayerKey.parse(r) for r in raw)


PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        Profile(
            name="clean",
            layers=(),
            description="no perturbation: the baseline",
            build=_clean,
            note="the reference every robustness delta is measured against",
        ),
        Profile(
            name="telephony",
            layers=_keys("L4.codec", "L4.frame_loss"),
            description="a phone line: mu-law 8 kHz with burst loss",
            build=_telephony,
        ),
        Profile(
            name="background_noise",
            layers=_keys("L3.background_bed"),
            description="a noisy room: a background bed at 15 dB SNR",
            build=_background_noise,
        ),
        Profile(
            name="noisy_line",
            layers=_keys("L3.background_bed", "L4.codec", "L4.frame_loss"),
            description="a call from a noisy room over a bad line",
            build=_noisy_line,
        ),
        Profile(
            name="accented",
            layers=_keys("L1.accent"),
            description=f"an accented caller ({ACCENT.value}) on a clean line",
            build=_accented,
        ),
        Profile(
            name="accented_noisy",
            layers=_keys("L1.accent", "L3.background_bed"),
            description=f"an accented caller ({ACCENT.value}) in a noisy room",
            build=_accented_noisy,
        ),
        Profile(
            name="overlap",
            layers=_keys("L5.interruption", "L5.backchannel"),
            description="a caller who interrupts and backchannels",
            build=_overlap,
            note="a clean line; the conversation is the perturbation",
        ),
        Profile(
            name="selectivity_probes",
            layers=_keys("L5.backchannel", "L5.vocal_tic", "L5.non_directed"),
            description="sounds the agent should hear and keep talking through",
            build=_selectivity_probes,
            note="a backchannel, a vocal tic, speech not directed at the agent",
        ),
    )
}
"""Core's generic conditions; adapters add their own with :func:`register_profile`."""


def register_profile(profile: Profile, *, replace: bool = False) -> Profile:
    """Make a condition available by name — an adapter's, at import.

    A benchmark adapter registers its benchmark's conditions here, built from
    core layers at the benchmark's parameters, so ``voice.toml``'s
    ``profile``, ``doctor`` and ``harbor-voice profiles`` know them like core's.

    Raises:
        DuplicateProfile: The name is taken and ``replace`` is not set; two
            conditions under one name would make results under it ambiguous.
    """
    if profile.name in PROFILES and not replace:
        raise DuplicateProfile(f"a profile named {profile.name!r} is registered")
    PROFILES[profile.name] = profile
    return profile


def get_profile(name: str) -> Profile:
    """Look a profile up by name.

    Raises:
        UnknownProfile: Listing what exists, because a typo in ``voice.toml``
            should not read as an unsupported condition.
    """
    try:
        return PROFILES[name]
    except KeyError:
        raise UnknownProfile(
            f"unknown profile {name!r}; available: {', '.join(sorted(PROFILES))}"
        ) from None


def profile_names(*, buildable_only: bool = False) -> list[str]:
    return sorted(
        name
        for name, profile in PROFILES.items()
        if profile.is_buildable or not buildable_only
    )


@dataclass(frozen=True, slots=True)
class ProfileSummary:
    """One row of ``harbor-voice profiles``."""

    profile: Profile
    paired_metrics: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def unpaired(self) -> tuple[str, ...]:
        """Active layers no registered metric reads, at profile level."""
        return tuple(
            str(key)
            for key in self.profile.layers
            if not self.paired_metrics.get(str(key))
        )


def summarise(profile: Profile) -> ProfileSummary:
    """Pair a profile's layers with the metrics that read them.

    Answers the question a task author actually has: *if I switch this
    profile on, which numbers move?*
    """
    from harbor_voice.metrics.base import REGISTRY

    paired: dict[str, tuple[str, ...]] = {}
    for key in profile.layers:
        needed = f"layer:{key}"
        paired[str(key)] = tuple(
            name
            for name in REGISTRY.names()
            if needed in REGISTRY.get(name).requires
            or key.layer in REGISTRY.get(name).sensitive_to
        )
    return ProfileSummary(profile=profile, paired_metrics=paired)
