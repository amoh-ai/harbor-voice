"""The adapter seam.

Task success is the **benchmark's** definition, not ours. tau-voice decides
what a completed booking is; FDB decides what its labels mean; we decide
nothing about either, and porting that decision faithfully is what the parity
gate checks.

Everything else harbor-voice measures — divergence, timing, turn-taking,
selectivity, speech, validity — is a property of the **call** rather than the
task, and is ours precisely so that it is identical across ports. If each
adapter reimplemented said-vs-heard WER, no two voice benchmarks would be
comparable, and the harness would have no reason to exist.

The boundary between those two is the ``task.`` prefix, and this module makes
it a mechanism rather than a convention.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from harbor_voice.capabilities import LayerKey
from harbor_voice.metrics.base import REGISTRY
from harbor_voice.metrics.base import Direction
from harbor_voice.metrics.base import Family
from harbor_voice.metrics.base import MetricDef
from harbor_voice.metrics.base import MetricFn
from harbor_voice.metrics.base import MetricRegistry
from harbor_voice.capabilities import Precondition

TASK_PREFIX = "task."
"""Reserved for metrics an adapter supplies.

A reader of a results table can tell, without asking, which numbers came from
the benchmark and which from the harness.
"""


class TaskNamespaceError(ValueError):
    """Something tried to register on the wrong side of the boundary."""


def task_metric(
    *,
    name: str,
    direction: Direction,
    benchmark: str,
    requires: Sequence[str] = (),
    unit: str | None = None,
    registry: MetricRegistry | None = None,
) -> Callable[[MetricFn], MetricFn]:
    """Register a benchmark's own scoring, supplied by its adapter.

    Args:
        name: Must begin with ``task.``.
        benchmark: The adapter supplying it, e.g. ``tau-voice``. Recorded on
            every result so a parity disagreement has somewhere to point.

    Raises:
        TaskNamespaceError: If ``name`` is outside the reserved namespace.
    """
    if not name.startswith(TASK_PREFIX):
        raise TaskNamespaceError(
            f"task metric {name!r} must be named '{TASK_PREFIX}<something>'; "
            f"the prefix is what tells a reader the number came from the "
            f"benchmark rather than from harbor-voice"
        )
    if not benchmark:
        raise TaskNamespaceError(
            f"task metric {name!r} must name the adapter that supplies it"
        )
    parsed = tuple(Precondition.parse(r) for r in requires)
    target = REGISTRY if registry is None else registry

    def decorate(fn: MetricFn) -> MetricFn:
        target.add(
            MetricDef(
                name=name,
                family=Family.TASK,
                direction=direction,
                fn=fn,
                requires=tuple(str(p) for p in parsed),
                unit=unit,
                doc=(fn.__doc__ or "").strip(),
                benchmark=benchmark,
            )
        )
        return fn

    return decorate


# ------------------------------------------------------- adapter validation


@dataclass(frozen=True, slots=True)
class SelectionProblem:
    """One reason a configuration will not measure what it claims."""

    kind: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"[{self.kind}] {self.detail}"


class SelectionError(ValueError):
    """A metric selection is incoherent. A load error, per rule 1."""

    def __init__(self, problems: Sequence[SelectionProblem]) -> None:
        self.problems = tuple(problems)
        joined = "\n  ".join(str(p) for p in problems)
        super().__init__(
            f"{len(problems)} problem(s) with metric selection:\n  {joined}"
        )


def check_selection(
    selected: Sequence[str],
    *,
    registry: MetricRegistry | None = None,
    active_layers: Iterable[LayerKey | str] = (),
    strict_layers: bool = True,
) -> list[SelectionProblem]:
    """Validate a task's metric selection against what is registered.

    Two checks, both of which catch a mistake that would otherwise show up as
    a *missing column* — the failure mode nobody investigates, because an
    absent number raises no question.

    1. **Every selected metric exists.** A typo in ``voice.toml`` becomes a
       load error rather than a silently absent column.
    2. **Every active perturbation layer has a paired metric.** A
       layer switched on with nothing reading it is only making the number
       worse. Pairing is satisfied two ways, because there are two kinds of
       perturbation: a metric that **requires** the layer (selectivity —
       there is no backchannel score without a backchannel), or one declaring
       itself **sensitive** to it (WER is defined on a clean line; a codec is
       what makes it interesting). Requiring the first form everywhere would
       make WER inapplicable to the clean baseline that every degraded run is
       compared against.

    Args:
        strict_layers: When false, rule 2 is skipped. Used by ``doctor`` to
            report the full picture rather than stopping at the first fault.

    Returns:
        The problems found, empty when the selection is sound.
    """
    reg = REGISTRY if registry is None else registry
    problems: list[SelectionProblem] = []

    known = set(reg.names())
    for name in selected:
        if name not in known:
            hint = ""
            if name.startswith(TASK_PREFIX):
                hint = (
                    " — a 'task.' metric must be registered by the adapter "
                    "before the task that selects it runs"
                )
            problems.append(
                SelectionProblem("unknown_metric", f"{name!r} is not registered{hint}")
            )

    if strict_layers:
        keys = [
            layer if isinstance(layer, LayerKey) else LayerKey.parse(str(layer))
            for layer in active_layers
        ]
        for key in keys:
            # Preconditions are stored parsed, so the precondition match is
            # against "layer:L5.backchannel" rather than the bare key.
            needed = f"layer:{key}"
            paired = [
                name
                for name in selected
                if name in known
                and (
                    needed in reg.get(name).requires
                    or key.layer in reg.get(name).sensitive_to
                )
            ]
            if not paired:
                problems.append(
                    SelectionProblem(
                        "unpaired_layer",
                        f"{key} is active but no selected metric reads it"
                        f"; a perturbation with no paired metric is "
                        f"only making the number worse. Select a metric that "
                        f"requires it, or one declaring "
                        f"sensitive_to={key.layer.value}",
                    )
                )

    return problems


def require_valid_selection(
    selected: Sequence[str],
    *,
    registry: MetricRegistry | None = None,
    active_layers: Iterable[LayerKey | str] = (),
) -> None:
    """Raise unless the selection is sound.

    Raises:
        SelectionError: Listing **every** problem, not the first. Fixing one
            fault only to meet the next is a three-run debug cycle on a
            benchmark whose runs are not cheap.
    """
    problems = check_selection(selected, registry=registry, active_layers=active_layers)
    if problems:
        raise SelectionError(problems)
