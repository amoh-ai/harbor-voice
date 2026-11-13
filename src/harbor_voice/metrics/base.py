"""Metrics that know when they do not apply.

Two rules shape everything here.

A metric reads the trajectory and never touches the live run. That
is what makes ``regrade`` possible: fix a definition, re-score every historical
trial for the price of CPU, and apply a metric invented a year later to data
collected today.

A metric declares its preconditions and abstains outside them. The
enforcement is stronger than returning a sentinel — a metric whose
preconditions fail is never called at all. A function that runs in conditions
it was not designed for produces a number, and a number gets published. The
only reliable way to prevent that is to not reach the function.

What a metric may still do, once it has been reached, is abstain for *data*
reasons: no turn had both sides, no barge-in occurred. That is
:class:`NotApplicable` returned from the body, and it is a different statement
from "this configuration cannot support this metric" — the first says the run
had nothing to measure, the second says the measurement was never meaningful.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any

from harbor_voice.capabilities import Applicability
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import LayerKey
from harbor_voice.capabilities import Precondition
from harbor_voice.capabilities import Verdict
from harbor_voice.capabilities import combine
from harbor_voice.capabilities import evaluate as evaluate_preconditions
from harbor_voice.trajectory.model import PartyRole
from harbor_voice.trajectory.model import VoiceTrajectory

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


class Direction(StrEnum):
    """Which way is better. Needed to aggregate and to rank, never assumed.

    ``NEUTRAL`` is not a hedge. Turn-overlap ratio is the standing example:
    its polarity is task-dependent. Overlap is rudeness in an intake
    call and attentiveness in a crisis line, so a metric that declared one
    direction globally would rank half its use cases backwards.
    """

    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"
    NEUTRAL = "neutral"


class Family(StrEnum):
    """The eight metric families. A metric must belong to exactly one."""

    TASK = "task"
    DIVERGENCE = "divergence"
    TIMING = "timing"
    TURNS = "turns"
    SELECTIVITY = "selectivity"
    SPEECH = "speech"
    ROBUSTNESS = "robustness"
    VALIDITY = "validity"


@dataclass(frozen=True, slots=True)
class Value:
    """A measured number, with what it was measured over."""

    value: float
    unit: str | None = None
    n: int | None = None
    """Sample size behind the number. A latency over one turn and one over
    forty are different claims, and the aggregate cannot tell them apart
    afterwards."""

    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NotApplicable:
    """The run had nothing to measure.

    Distinct from a configuration that cannot support the metric: this one was
    reached, looked, and found no data. "No barge-in occurred" is a fact about
    the call; "this agent cannot be interrupted" is a fact about the setup.
    """

    reasons: tuple[str, ...] = ()

    @classmethod
    def because(cls, *reasons: str) -> NotApplicable:
        return cls(reasons=reasons)


Outcome = Value | NotApplicable


@dataclass(frozen=True, slots=True)
class MetricContext:
    """Everything a metric is allowed to see.

    Deliberately narrow. A metric that could reach a live transport would stop
    being re-runnable, and ``regrade`` would quietly become a different
    computation from the original scoring.
    """

    trajectory: VoiceTrajectory
    clean_baseline: VoiceTrajectory | None = None
    """The matched-seed clean run, for the robustness family."""

    provider_applicability: Applicability | None = None
    """Verdict from the provider level, folded in by :func:`combine`."""

    @property
    def agent(self) -> Capabilities | None:
        """Capabilities of the party under test."""
        for party in self.trajectory.parties:
            if (
                party.role is PartyRole.SYSTEM_UNDER_TEST
                and party.capabilities is not None
            ):
                return party.capabilities
        return None

    @property
    def active_layers(self) -> frozenset[LayerKey]:
        """Perturbations actually switched on, read from the written plan.

        From the plan rather than the request: what a run *did* is
        the only thing a metric may condition on.
        """
        layers = self.trajectory.perturbation_plan.get("layers") or {}
        keys: set[LayerKey] = set()
        for raw in layers:
            try:
                keys.add(LayerKey.parse(str(raw)))
            except ValueError:
                continue
        return frozenset(keys)

    @property
    def floor(self) -> Duplex | None:
        matrix = self.trajectory.config_matrix
        if matrix is None:
            return None
        try:
            return Duplex(matrix.floor_policy)
        except ValueError:
            return None


MetricFn = Callable[[MetricContext], Outcome]


@dataclass(frozen=True, slots=True)
class MetricDef:
    """A metric, its preconditions, and how to read its number."""

    name: str
    family: Family
    direction: Direction
    fn: MetricFn
    requires: tuple[str, ...] = ()
    sensitive_to: tuple[Layer, ...] = ()
    """Perturbation layers this metric *responds to*.

    There are two ways a metric can pair with a layer, and conflating them
    makes the pairing rule either toothless or wrong.

    Precondition pairing (``requires=["layer:L5.backchannel"]``): the
    metric exists *only because* the perturbation does. Selectivity is the
    whole family — there is no backchannel score without a backchannel.

    Sensitivity (this field): the metric is meaningful anyway, and the
    layer moves it. Said-vs-heard WER is defined on a clean line; switching
    on a codec and burst loss is what makes it *interesting*. Demanding a
    precondition here would be wrong — it would make WER inapplicable to
    the clean baseline, which is the thing every degraded run is compared
    against.

    Declared at layer granularity rather than per perturbation: a metric
    sensitive to the channel is sensitive to every channel perturbation,
    and enumerating them would go stale the first time one is added.
    """

    unit: str | None = None
    doc: str = ""
    benchmark: str | None = None
    """Adapter that supplied this metric, for a ``task.*`` one.

    Carried so ``measurement.json`` can say which port produced a number,
    and a parity disagreement has somewhere to point.
    """

    def applicability(self, context: MetricContext) -> Applicability:
        """Whether this metric may run at all, across both levels.

        Returns ``NOT_APPLICABLE`` when the agent declared no capabilities and
        the metric needs some — an undeclared party is not a permissive one.
        """
        if not self.requires:
            return combine(
                context.provider_applicability or Applicability(Verdict.SATISFIED)
            )
        agent = context.agent
        if agent is None:
            return Applicability(
                Verdict.NOT_APPLICABLE,
                (
                    f"{self.name} requires {list(self.requires)} but no agent "
                    f"party declared capabilities",
                ),
            )
        party_verdict = evaluate_preconditions(
            tuple(Precondition.parse(r) for r in self.requires),
            agent,
            active_layers=context.active_layers,
            floor=context.floor,
        )
        if context.provider_applicability is None:
            return party_verdict
        return combine(party_verdict, context.provider_applicability)


@dataclass(frozen=True, slots=True)
class MetricResult:
    """What a metric reported, and whether it may be published.

    Four columns, which is the whole point of ``measurement.json``: the value,
    whether it applies, whether it is verified, and why not.
    """

    name: str
    family: Family
    direction: Direction
    verdict: Verdict
    value: float | None = None
    unit: str | None = None
    n: int | None = None
    reasons: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)
    benchmark: str | None = None

    @property
    def reported(self) -> bool:
        """Whether there is a number at all."""
        return self.value is not None

    @property
    def publishable(self) -> bool:
        """Whether this number may gate a published claim — rule 3.

        An unattested capability yields a number that is usable for debugging
        and not for a paper. Keeping the two apart is the entire reason
        ``UNATTESTED`` exists as a third verdict.
        """
        return self.value is not None and self.verdict is Verdict.SATISFIED

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "family": self.family.value,
            "direction": self.direction.value,
            "verdict": self.verdict.value,
            "value": self.value,
        }
        if self.unit is not None:
            out["unit"] = self.unit
        if self.n is not None:
            out["n"] = self.n
        if self.reasons:
            out["reasons"] = list(self.reasons)
        if self.detail:
            out["detail"] = self.detail
        if self.benchmark is not None:
            out["benchmark"] = self.benchmark
        return out


def evaluate_metric(defn: MetricDef, context: MetricContext) -> MetricResult:
    """Run one metric, or decline to.

    The declining is the interesting half. When preconditions fail the
    function is not called — not called and discarded, not called and
    overridden. A metric body that executes outside its design conditions
    produces a plausible number, and a plausible number is indistinguishable
    from a real one by the time it reaches a table.
    """
    verdict = defn.applicability(context)
    if verdict.verdict is Verdict.NOT_APPLICABLE:
        return MetricResult(
            name=defn.name,
            family=defn.family,
            direction=defn.direction,
            verdict=Verdict.NOT_APPLICABLE,
            unit=defn.unit,
            reasons=verdict.reasons,
            benchmark=defn.benchmark,
        )

    outcome = defn.fn(context)
    if isinstance(outcome, NotApplicable):
        return MetricResult(
            name=defn.name,
            family=defn.family,
            direction=defn.direction,
            verdict=Verdict.NOT_APPLICABLE,
            unit=defn.unit,
            reasons=outcome.reasons,
            benchmark=defn.benchmark,
        )
    return MetricResult(
        name=defn.name,
        family=defn.family,
        direction=defn.direction,
        verdict=verdict.verdict,
        value=outcome.value,
        unit=outcome.unit or defn.unit,
        n=outcome.n,
        reasons=verdict.reasons,
        detail=outcome.detail,
        benchmark=defn.benchmark,
    )


# ------------------------------------------------------------------ registry


class MetricRegistry:
    """Named metrics. One global default, plus private ones for tests."""

    def __init__(self) -> None:
        self._defs: dict[str, MetricDef] = {}

    def add(self, defn: MetricDef) -> None:
        if defn.name in self._defs:
            raise ValueError(f"metric {defn.name!r} is already registered")
        self._defs[defn.name] = defn

    def __contains__(self, name: object) -> bool:
        return name in self._defs

    def __len__(self) -> int:
        return len(self._defs)

    def get(self, name: str) -> MetricDef:
        try:
            return self._defs[name]
        except KeyError:
            known = ", ".join(sorted(self._defs)) or "(none)"
            raise KeyError(f"unknown metric {name!r}; registered: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._defs)

    def evaluate(
        self, names: Sequence[str], context: MetricContext
    ) -> list[MetricResult]:
        """Score a selection, in the order asked for."""
        return [evaluate_metric(self.get(name), context) for name in names]

    def evaluate_all(self, context: MetricContext) -> list[MetricResult]:
        return self.evaluate(self.names(), context)


REGISTRY = MetricRegistry()
"""The default registry. Families register into it on import."""


def metric(
    *,
    name: str,
    family: Family,
    direction: Direction,
    requires: Sequence[str] = (),
    sensitive_to: Sequence[Layer] = (),
    unit: str | None = None,
    registry: MetricRegistry | None = None,
) -> Callable[[MetricFn], MetricFn]:
    """Register a metric.

    Preconditions are parsed at decoration time, so a typo in
    ``requires=["duplex:ful"]`` is an import error rather than a metric that
    silently never applies. A metric that is quietly always N/A is worse than
    one that crashes: the column is simply absent and nobody asks why.

    The ``task.`` namespace is refused here: task success is
    the benchmark's definition, and the harness must not be able to quietly
    start supplying one. Adapters use
    :func:`~harbor_voice.metrics.task.task_metric`.

    Raises:
        ValueError: If ``name`` is in the reserved ``task.`` namespace.
    """
    if name.startswith("task."):
        raise ValueError(
            f"metric {name!r} is in the 'task.' namespace, which only an "
            f"adapter may fill — use task_metric() from harbor_voice.metrics.task"
        )
    parsed = tuple(Precondition.parse(r) for r in requires)
    # `registry or REGISTRY` would be wrong: MetricRegistry defines
    # __len__, so an empty one is falsy and a caller's private registry
    # would silently receive nothing while the global filled up.
    target = REGISTRY if registry is None else registry

    def decorate(fn: MetricFn) -> MetricFn:
        target.add(
            MetricDef(
                name=name,
                family=family,
                direction=direction,
                fn=fn,
                requires=tuple(str(p) for p in parsed),
                sensitive_to=tuple(sensitive_to),
                unit=unit,
                doc=(fn.__doc__ or "").strip(),
            )
        )
        return fn

    return decorate
