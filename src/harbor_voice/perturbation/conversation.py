"""L5 — when they speak.

Every other layer transforms a signal: L2 changes how a voice sounds, L3 puts
a room around it, L4 puts it down a wire. L5 transforms **nothing**. It decides
*when* the caller speaks relative to the agent, which is why it is
orthogonal: it drives the floor rather than the audio.

That is also why nothing here subclasses
:class:`~harbor_voice.perturbation.base.Perturbation`, whose contract is
``apply(pcm, ctx) -> bytes`` with an enforced equal length. An L5 layer with an
identity ``apply`` would type-check and would put a lie in the plan record: a
reader would see a layer in the signal chain that never touched a sample.

### Four probes, one mechanism

A probe is a short utterance injected into the agent's turn. All four are the
same machinery; they differ in one bit — whether the caller is claiming the
floor (:class:`~harbor_voice.floor.Bid`):

===============  ====================  ===========================  ===========
Probe            Bid                   The agent is right if it     Family
===============  ====================  ===========================  ===========
``interruption`` ``FLOOR``             **stops**, quickly           turns
``backchannel``  ``BACKCHANNEL``       keeps going                  selectivity
``vocal_tic``    ``BACKCHANNEL``       keeps going                  selectivity
``non_directed`` ``BACKCHANNEL``       keeps going                  selectivity
===============  ====================  ===========================  ===========

tau-voice's selectivity family exists *only because* its simulator injects
these, which is the perturbation–metric pairing exactly: the perturbation is the probe that makes the
measurement possible. Without L5, ``selectivity.*``, ``turns.overlap_s``,
``turns.barge_in_success_rate`` and ``timing.yield_latency_s`` are all
correctly inert, which is the state ``doctor`` reported before this module
existed.

### Scheduling is decided before the agent replies

Offsets are drawn from the ``policy`` seed stream at the start of a turn, as
offsets from **the agent's first frame**. A probe whose offset lands past the
end of the reply never fires, and is recorded as such rather than
dropped — "the agent ignored the interruption" and "the interruption never
happened" are different findings, and the second one is about the harness.

Deciding up front rather than reacting to the agent's pace is what keeps a run
reproducible from its seed: a slower agent gets the same probes at the same
offsets, and only *whether they landed* changes.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Any
from typing import override

from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import LayerKey
from harbor_voice.floor import Bid
from harbor_voice.perturbation.base import SeedSet

_POLICY_STREAM = "policy"


class ConversationError(ValueError):
    """An L5 layer was configured in a way that cannot produce a measurement."""


class ProbeKind(StrEnum):
    """What kind of thing the caller interjects, and what it asks for.

    The value doubles as the layer name (``L5.backchannel``) and as the
    ``meta["probe"]`` tag the selectivity metrics match on. One string, so the
    layer a task activates and the probe a metric counts cannot drift apart.
    """

    INTERRUPTION = "interruption"
    BACKCHANNEL = "backchannel"
    VOCAL_TIC = "vocal_tic"
    NON_DIRECTED = "non_directed"

    @property
    def bid(self) -> Bid:
        """Whether this interjection claims the floor."""
        return Bid.FLOOR if self is ProbeKind.INTERRUPTION else Bid.BACKCHANNEL

    @property
    def expects_yield(self) -> bool:
        """Whether a correct agent stops when this arrives."""
        return self.bid is Bid.FLOOR

    @property
    def key(self) -> LayerKey:
        """The ``L5.backchannel``-style key a task activates it by."""
        return LayerKey(layer=Layer.CONVERSATION, name=self.value)


@dataclass(frozen=True, slots=True)
class Probe:
    """One scheduled interjection."""

    kind: ProbeKind
    text: str
    at_s: float
    """Offset from the agent's first frame, in seconds."""

    index: int = 0
    turn_index: int = 0

    def __post_init__(self) -> None:
        if self.at_s < 0:
            raise ConversationError(f"probe offset cannot be negative: {self.at_s}")
        if not self.text.strip():
            raise ConversationError("a probe needs something to say")

    @property
    def bid(self) -> Bid:
        return self.kind.bid

    @property
    def expects_yield(self) -> bool:
        return self.kind.expects_yield

    @property
    def key(self) -> LayerKey:
        return self.kind.key


class Landing(StrEnum):
    """Whether a probe landed somewhere it could be measured.

    The key is recording *whether the interruption fired*. Without it a barge-in metric cannot tell "the agent
    handled it" from "it never happened", and the second one is a harness
    finding wearing an agent's score.
    """

    DURING_SPEECH = "during_speech"
    """It fired while the agent was talking. The only measurable case."""

    REPLY_ALREADY_DONE = "reply_already_done"
    """The reply ended before the probe's offset; it never went out."""

    NO_REPLY = "no_reply"
    """The agent produced no audio at all, so there was nothing to interrupt."""

    INDISTINGUISHABLE = "indistinguishable"
    """It fired, and not one more frame of agent audio arrived.

    Yielding takes time — an agent has to notice the speech, stop generating
    and stop sending, and in a real-time call that is always some trailing
    audio. Zero trailing audio therefore means the reply had already finished
    generating and the probe landed in the gap. Scoring that as a perfect
    yield would hand short-winded agents a free pass on every barge-in, so it
    is excluded from the denominator and counted separately.

    The residual ambiguity — a reply that *would* have ended a moment later —
    cannot be settled from one trajectory, because yielding is what shortens
    the reply. Settling it needs the same input replayed without the probe,
    which is the robustness family's paired clean run.
    """

    @property
    def is_measurable(self) -> bool:
        return self is Landing.DURING_SPEECH


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """What happened to one probe. The record both families read."""

    probe: Probe
    landing: Landing
    at_reply_s: float | None = None
    """How far into the agent's reply audio the probe landed.

    Reply-audio seconds, not wall seconds. A barge-in "one second into the
    reply" means one second of the agent's *speech*, which is what a caller
    experiences and what stays comparable when a harness runs faster than
    real time. In a real-time call the two coincide.
    """

    reply_total_s: float | None = None
    """Total agent audio for the turn, or ``None`` if the reply never ended."""

    t_injected: float | None = None
    """Session time the probe went on the wire, for the timeline."""

    deadline_s: float = 0.0

    @property
    def fired(self) -> bool:
        return self.t_injected is not None

    @property
    def is_measurable(self) -> bool:
        return self.landing.is_measurable

    @property
    def yield_latency_s(self) -> float | None:
        """Agent audio that kept arriving after the probe landed.

        ``None`` if the reply never ended — the drain gave up first, so we do
        not know that it ever stopped, and a stand-in would be invented data.
        """
        if self.at_reply_s is None or self.reply_total_s is None:
            return None
        return max(0.0, self.reply_total_s - self.at_reply_s)

    @property
    def yielded_in_time(self) -> bool | None:
        """``None`` means *never stopped*, not *unknown*."""
        latency = self.yield_latency_s
        return None if latency is None else latency <= self.deadline_s

    @property
    def ignored(self) -> bool | None:
        """Whether the agent kept talking through it.

        The same observation as :attr:`yielded_in_time`, read the other way
        up: for a backchannel, carrying on *is* the correct behaviour. ``None``
        when the probe did not land somewhere measurable.
        """
        if not self.is_measurable:
            return None
        return self.yielded_in_time is not True

    @property
    def correct(self) -> bool | None:
        """Whether the agent did the right thing, whichever right thing it was."""
        if not self.is_measurable:
            return None
        if self.probe.expects_yield:
            return self.yielded_in_time is True
        return self.ignored is True


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    """Defaults for one probe kind, kept apart from the injector.

    Separate so ``params()`` writes the *requested* settings into the plan
    without having to reconstruct them from an object's private state.
    """

    kind: ProbeKind
    phrases: tuple[str, ...]
    first_at_s: float = 1.0
    every_s: float | None = None
    jitter_s: float = 0.0
    max_per_turn: int = 1
    probability: float = 1.0

    def __post_init__(self) -> None:
        if not self.phrases:
            raise ConversationError(f"{self.kind.value} needs at least one phrase")
        if self.first_at_s < 0:
            raise ConversationError(f"first_at_s cannot be negative: {self.first_at_s}")
        if self.every_s is not None and self.every_s <= 0:
            raise ConversationError(f"every_s must be positive: {self.every_s}")
        if self.max_per_turn < 1:
            raise ConversationError(f"max_per_turn must be >= 1: {self.max_per_turn}")
        if not 0.0 <= self.probability <= 1.0:
            raise ConversationError(f"probability out of range: {self.probability}")
        if self.jitter_s < 0:
            raise ConversationError(f"jitter_s cannot be negative: {self.jitter_s}")


class ProbeInjector:
    """Schedules one kind of interjection, seeded on the ``policy`` stream.

    One class rather than four subclasses: the kinds differ in their defaults
    and in :attr:`ProbeKind.bid`, not in behaviour, and four near-identical
    subclasses would be four places to fix the same bug.
    """

    def __init__(self, spec: ProbeSpec, *, seeds: SeedSet | None = None) -> None:
        self._spec = spec
        self._rng: Any = None
        self._scheduled = 0
        self._fired = 0
        self._measurable = 0
        self._correct = 0
        self._drift_s = 0.0
        self.reset(seeds or SeedSet.derive(0))

    # ------------------------------------------------------------- identity

    @property
    def kind(self) -> ProbeKind:
        return self._spec.kind

    @property
    def layer(self) -> Layer:
        return Layer.CONVERSATION

    @property
    def key(self) -> LayerKey:
        return self._spec.kind.key

    @property
    def spec(self) -> ProbeSpec:
        return self._spec

    # -------------------------------------------------------------- running

    def reset(self, seeds: SeedSet) -> None:
        """Re-seed and clear per-run counters."""
        self._rng = seeds.rng(_POLICY_STREAM)
        self._scheduled = self._fired = self._measurable = self._correct = 0
        self._drift_s = 0.0

    def schedule(self, turn_index: int) -> tuple[Probe, ...]:
        """Draw this turn's probes, as offsets from the agent's first frame."""
        spec = self._spec
        probes: list[Probe] = []
        at = spec.first_at_s
        for _ in range(spec.max_per_turn):
            if spec.probability < 1.0 and float(self._rng.random()) > spec.probability:
                at = _advance(at, spec.every_s)
                continue
            offset = at
            if spec.jitter_s:
                drift = float(self._rng.uniform(-1.0, 1.0)) * spec.jitter_s
                offset = max(0.0, at + drift)
            phrase = spec.phrases[int(self._rng.integers(len(spec.phrases)))]
            probes.append(
                Probe(
                    kind=spec.kind,
                    text=phrase,
                    at_s=offset,
                    index=self._scheduled + len(probes),
                    turn_index=turn_index,
                )
            )
            at = _advance(at, spec.every_s)
            if spec.every_s is None:
                break
        self._scheduled += len(probes)
        return tuple(probes)

    def observe(self, result: ProbeResult) -> None:
        """Count what actually happened: achieved, not requested, for L5.

        A plan that records only what it *asked* for cannot tell you that two
        thirds of its interruptions landed after the agent had stopped talking.
        """
        if result.fired:
            self._fired += 1
            if result.at_reply_s is not None:
                self._drift_s += result.at_reply_s - result.probe.at_s
        if result.is_measurable:
            self._measurable += 1
            if result.correct:
                self._correct += 1

    # --------------------------------------------------------------- record

    def params(self) -> dict[str, Any]:
        spec = self._spec
        return {
            "phrases": list(spec.phrases),
            "first_at_s": spec.first_at_s,
            "every_s": spec.every_s,
            "jitter_s": spec.jitter_s,
            "max_per_turn": spec.max_per_turn,
            "probability": spec.probability,
            "bid": spec.kind.bid.value,
            "expects_yield": spec.kind.expects_yield,
        }

    def achieved(self) -> dict[str, float]:
        """What landed, not what was asked for — the SNR rule applied to a layer
        with no SNR.

        ``offset_drift_s`` is the one that catches a quiet problem: a probe
        can only land on a frame boundary, so an agent that streams its reply
        in half-second lumps turns ``at_s=0.8`` into an injection at 1.3.
        The run is still valid, but a reader comparing two agents should know
        that one of them was probed later than the other.
        """
        record = {
            "scheduled": float(self._scheduled),
            "fired": float(self._fired),
            "measurable": float(self._measurable),
            "correct": float(self._correct),
        }
        if self._fired:
            record["offset_drift_s"] = self._drift_s / self._fired
        return record

    @override
    def __repr__(self) -> str:
        return f"ProbeInjector({self.key}, scheduled={self._scheduled})"


def _advance(at: float, every_s: float | None) -> float:
    return at if every_s is None else at + every_s


# ------------------------------------------------------------------ factories


def interruption(
    *,
    phrases: Sequence[str] = ("Sorry, wait.", "Hold on.", "No, stop."),
    first_at_s: float = 1.0,
    jitter_s: float = 0.0,
) -> ProbeInjector:
    """``L5.interruption`` — a genuine claim on the floor.

    The agent is expected to stop. How fast it does is
    ``timing.yield_latency_s``; whether it did at all is
    ``turns.barge_in_success_rate``.
    """
    return ProbeInjector(
        ProbeSpec(
            kind=ProbeKind.INTERRUPTION,
            phrases=tuple(phrases),
            first_at_s=first_at_s,
            jitter_s=jitter_s,
        )
    )


def backchannel(
    *,
    phrases: Sequence[str] = ("mm-hm", "right", "okay", "yeah"),
    first_at_s: float = 0.8,
    every_s: float | None = 2.0,
    max_per_turn: int = 2,
) -> ProbeInjector:
    """``L5.backchannel`` — continuer tokens. The agent should talk through them.

    The commonest false positive in deployed voice agents: an "mm-hm" trips
    the VAD, the agent stops mid-sentence, and the caller has to start again.
    ``selectivity.backchannel`` is the number that catches it.
    """
    return ProbeInjector(
        ProbeSpec(
            kind=ProbeKind.BACKCHANNEL,
            phrases=tuple(phrases),
            first_at_s=first_at_s,
            every_s=every_s,
            max_per_turn=max_per_turn,
        )
    )


def vocal_tic(
    *,
    phrases: Sequence[str] = ("um", "uh", "er"),
    first_at_s: float = 0.6,
    probability: float = 0.6,
) -> ProbeInjector:
    """``L5.vocal_tic`` — a filled pause while the agent holds the floor.

    Distinct from ``L2.vocal_tics``, which decorates the caller's *own* turn.
    Same sound, different layer, because what makes this one a probe is when
    it arrives, not how it is pronounced.
    """
    return ProbeInjector(
        ProbeSpec(
            kind=ProbeKind.VOCAL_TIC,
            phrases=tuple(phrases),
            first_at_s=first_at_s,
            probability=probability,
        )
    )


def non_directed(
    *,
    phrases: Sequence[str] = ("no, not you", "one second", "it is the office"),
    first_at_s: float = 1.2,
) -> ProbeInjector:
    """``L5.non_directed`` — speech the agent hears but was not addressed to it."""
    return ProbeInjector(
        ProbeSpec(
            kind=ProbeKind.NON_DIRECTED,
            phrases=tuple(phrases),
            first_at_s=first_at_s,
        )
    )


_FACTORIES = {
    ProbeKind.INTERRUPTION: interruption,
    ProbeKind.BACKCHANNEL: backchannel,
    ProbeKind.VOCAL_TIC: vocal_tic,
    ProbeKind.NON_DIRECTED: non_directed,
}


def injector_for(kind: ProbeKind) -> ProbeInjector:
    """A default injector for one kind, for profiles that want the standard one."""
    return _FACTORIES[kind]()


@dataclass(slots=True)
class ConversationPlan:
    """The active L5 layers, their schedule, and what landed.

    Lives beside the signal chain in
    :class:`~harbor_voice.perturbation.plan.PerturbationPipeline` rather than
    inside it: it contributes to ``active_layers`` and to the written plan, and
    never to ``apply``.
    """

    injectors: tuple[ProbeInjector, ...] = ()
    seeds: SeedSet = field(default_factory=lambda: SeedSet.derive(0))

    def __post_init__(self) -> None:
        kinds = [i.kind for i in self.injectors]
        duplicates = {k for k in kinds if kinds.count(k) > 1}
        if duplicates:
            raise ConversationError(
                f"two injectors for {sorted(d.value for d in duplicates)}; "
                f"one layer, one injector — otherwise the plan record cannot "
                f"say which set of parameters produced a probe"
            )
        self.reset(self.seeds)

    @classmethod
    def of(cls, *kinds: ProbeKind, seeds: SeedSet | None = None) -> ConversationPlan:
        """A plan with the default injector for each named kind."""
        return cls(
            injectors=tuple(injector_for(k) for k in kinds),
            seeds=seeds or SeedSet.derive(0),
        )

    @property
    def is_empty(self) -> bool:
        return not self.injectors

    @property
    def active_layers(self) -> frozenset[LayerKey]:
        """What a task has switched on. Feeds metric applicability."""
        return frozenset(i.key for i in self.injectors)

    @property
    def expects_yield(self) -> bool:
        """Whether any active layer claims the floor — i.e. needs barge-in."""
        return any(i.kind.expects_yield for i in self.injectors)

    def reset(self, seeds: SeedSet | None = None) -> None:
        """Re-seed every injector for a new run."""
        if seeds is not None:
            self.seeds = seeds
        for injector in self.injectors:
            injector.reset(self.seeds)

    def schedule(self, turn_index: int) -> tuple[Probe, ...]:
        """Every layer's probes for one agent turn, in the order they fire."""
        probes: list[Probe] = []
        for injector in self.injectors:
            probes.extend(injector.schedule(turn_index))
        return tuple(sorted(probes, key=lambda p: (p.at_s, p.kind.value)))

    def observe(self, results: Iterable[ProbeResult]) -> None:
        """Feed back what landed, per layer."""
        by_kind = {i.kind: i for i in self.injectors}
        for result in results:
            injector = by_kind.get(result.probe.kind)
            if injector is not None:
                injector.observe(result)

    def as_record(self) -> dict[str, dict[str, Any]]:
        """The L5 half of the written plan."""
        return {
            str(i.key): {"params": i.params(), "achieved": i.achieved()}
            for i in self.injectors
        }

    @override
    def __repr__(self) -> str:
        layers = ", ".join(str(i.key) for i in self.injectors) or "none"
        return f"ConversationPlan({layers})"
