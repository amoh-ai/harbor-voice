"""Named conditions, so a port can reproduce its paper.

A profile is a **published name for a set of perturbation layers**. The point
is comparability: ``eva_b`` has to mean the same thing in our run and in the
paper's, and "we added some background noise at about 15 dB" does not survive
contact with a reviewer.

Several profiles name layers that do not exist yet (L1 persona, L2 speech,
L5 conversation, L6 world). They are **declared
anyway**, and refuse to build, following the same rule as an unimplemented
provider: the name is real, its layer list is real, ``doctor`` and ``profiles``
can report on it, and only construction fails. A profile quietly missing from
the list would be read as "that condition is not supported", which is a
different and wrong claim from "not yet".
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
from harbor_voice.perturbation.speech import SpeechPlan
from harbor_voice.perturbation.speech import VocalTics

Builder = Callable[[SeedSet], PerturbationPipeline]


class UnknownProfile(LookupError):
    """No profile is registered under that name."""


class ProfileNotBuildable(NotImplementedError):
    """The profile is declared but needs a layer that does not exist yet."""


@dataclass(frozen=True, slots=True)
class Profile:
    """A named condition and the layers it switches on."""

    name: str
    layers: tuple[LayerKey, ...]
    reproduces: str
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
                it from the layer list was right while whole layers were
                unbuilt and became wrong the moment one was not: ``fdb_disfluent``
                declares L2 and L5, both of which now exist, and is unbuildable
                because nobody has the recorded corpus.
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


def _eva_b(seeds: SeedSet) -> PerturbationPipeline:
    return PerturbationPipeline(
        [
            BackgroundBed(
                source=synthetic_bed(seconds=10.0, seed=7),
                target_snr_db=15.0,
                snr_drift_db=3.0,
                variation_speed=0.1,
            )
        ],
        seeds=seeds,
        plan_id=PlanId("eva_b"),
    )


def _fdb_overlap(seeds: SeedSet) -> PerturbationPipeline:
    """L5 only — the line is clean and the *conversation* is the perturbation.

    Nothing here degrades a sample, which is the point: an agent that handles
    a pristine line perfectly can still talk over its caller, and FDB v1.5
    exists because that failure is invisible to every signal-level condition.
    """
    return PerturbationPipeline(
        seeds=seeds,
        plan_id=PlanId("fdb_overlap"),
        conversation=ConversationPlan.of(
            ProbeKind.INTERRUPTION, ProbeKind.BACKCHANNEL, seeds=seeds
        ),
    )


def _selectivity_probes(seeds: SeedSet) -> PerturbationPipeline:
    """The three things an agent should hear and ignore — tau's S_BC/S_VT/S_ND.

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


def _eva_accent(seeds: SeedSet) -> PerturbationPipeline:
    """EVA condition A — an accented caller on an otherwise clean line.

    Pins ``accent`` and not ``behaviour``, which is EVA's own constraint:
    ``PerturbationConfig`` makes the two mutually exclusive because each maps
    onto a pre-provisioned ElevenLabs agent ID and a call gets one ID. A
    behaviour variant is therefore a *different* EVA condition, not a second
    axis of this one.

    Note:
        Which accent EVA's published condition A uses is **unconfirmed** —
        their four are french, indian, spanish and chinese. It must be read
        off the paper before this profile is used for a parity claim.
    """
    return PerturbationPipeline(
        seeds=seeds,
        plan_id=PlanId("eva_a"),
        persona=PersonaPlan(voice=VoiceSelection(accent=Accent.INDIAN)),
    )


def _eva_accent_plus_bed(seeds: SeedSet) -> PerturbationPipeline:
    """EVA condition A+B — the accented caller, now in a noisy room."""
    return PerturbationPipeline(
        [
            BackgroundBed(
                source=synthetic_bed(seconds=10.0, seed=7),
                target_snr_db=15.0,
                snr_drift_db=3.0,
                variation_speed=0.1,
            )
        ],
        seeds=seeds,
        plan_id=PlanId("eva_a_plus_b"),
        persona=PersonaPlan(voice=VoiceSelection(accent=Accent.INDIAN)),
    )


def _tau_voice_default(seeds: SeedSet) -> PerturbationPipeline:
    """tau-voice's default condition — tics, a room, and a telephone line."""
    return PerturbationPipeline(
        [
            BackgroundBed(
                source=synthetic_bed(seconds=10.0, seed=7),
                target_snr_db=15.0,
                snr_drift_db=2.0,
                variation_speed=0.1,
            ),
            MuLawCodec(),
            GilbertElliottLoss(rate=0.02, burst_duration_ms=200.0),
        ],
        seeds=seeds,
        plan_id=PlanId("tau_voice_default"),
        speech=SpeechPlan([VocalTics()], seeds=seeds),
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
            reproduces="every baseline",
            build=_clean,
            note="the reference every robustness delta is measured against",
        ),
        Profile(
            name="telephony",
            layers=_keys("L4.codec", "L4.frame_loss"),
            reproduces="real-world PSTN",
            build=_telephony,
            note="mu-law 8 kHz plus burst loss",
        ),
        Profile(
            name="eva_b",
            layers=_keys("L3.background_bed"),
            reproduces="EVA condition B",
            build=_eva_b,
        ),
        Profile(
            name="noisy_line",
            layers=_keys("L3.background_bed", "L4.codec", "L4.frame_loss"),
            reproduces="a call from a noisy room over a bad line",
            build=_noisy_line,
        ),
        Profile(
            name="eva_a",
            layers=_keys("L1.accent"),
            reproduces="EVA condition A",
            build=_eva_accent,
            note="accent only; EVA makes accent and behaviour exclusive, and "
            "which accent condition A pins is unconfirmed",
        ),
        Profile(
            name="eva_a_plus_b",
            layers=_keys("L1.accent", "L3.background_bed"),
            reproduces="EVA condition A+B",
            build=_eva_accent_plus_bed,
        ),
        Profile(
            name="tau_voice_default",
            layers=_keys(
                "L2.vocal_tics", "L3.background_bed", "L4.codec", "L4.frame_loss"
            ),
            reproduces="tau-voice",
            build=_tau_voice_default,
        ),
        Profile(
            name="fdb_overlap",
            layers=_keys("L5.interruption", "L5.backchannel"),
            reproduces="FDB v1.5",
            build=_fdb_overlap,
            note="a clean line; the conversation is the perturbation",
        ),
        Profile(
            name="selectivity_probes",
            layers=_keys("L5.backchannel", "L5.vocal_tic", "L5.non_directed"),
            reproduces="tau-voice selectivity (S_BC, S_VT, S_ND)",
            build=_selectivity_probes,
            note="things the agent should hear and keep talking through",
        ),
        Profile(
            name="fdb_disfluent",
            layers=_keys("L2.disfluency", "L5.interruption"),
            reproduces="FDB v3",
            note="needs FDB-v3's recorded disfluency corpus, which is "
            "CC BY-NC 4.0. The L2 text layers exist; their docs are "
            "explicit that a synthesised 'um' is not a substitute",
        ),
    )
}


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
