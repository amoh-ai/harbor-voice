"""Preflight for **meaning**.

harbor preflights *running*: can the agent start, are the resources there.
This checks whether a configuration **measures what the task claims**. That
failure is expensive because it doesn't crash: it produces a results table
with a column quietly missing, or a number resting on a capability nobody
verified.

``doctor`` answers one question: *if I start this run, what will I get?* It
answers it before a single container starts, without credentials, and without
constructing a provider — which is the whole reason capabilities are
a property of the class.

Five verdicts, ordered by how much they should worry you:

``ERROR``
    The run will not measure what it claims. A task requires a capability the
    agent does not have, or a selected metric does not exist.
``BLOCKED``
    A profile or provider cannot be constructed here at all.
``WARN``
    It will run and something about the result is misleading.
``UNATTESTED``
    A number will appear that no conformance run has verified — usable for
    debugging, not for a published claim.
``N/A``
    A metric will abstain, and why.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Any
from typing import override

from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Requirement
from harbor_voice.capabilities import Verdict
from harbor_voice.capabilities import check_requirements
from harbor_voice.metrics.base import REGISTRY
from harbor_voice.metrics.base import MetricRegistry
from harbor_voice.perturbation.profiles import Profile
from harbor_voice.perturbation.profiles import UnknownProfile
from harbor_voice.perturbation.profiles import get_profile
from harbor_voice.perturbation.profiles import summarise
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.providers.registry import UnknownProvider


class Severity(StrEnum):
    ERROR = "ERROR"
    BLOCKED = "BLOCKED"
    WARN = "WARN"
    UNATTESTED = "UNATTESTED"
    NOT_APPLICABLE = "N/A"
    OK = "OK"

    @property
    def is_fatal(self) -> bool:
        """Whether this stops the run rather than qualifying it."""
        return self in (Severity.ERROR, Severity.BLOCKED)


_ORDER = {s: i for i, s in enumerate(Severity)}


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing doctor has to say."""

    severity: Severity
    subject: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.severity.value:<11}{self.subject}  {self.detail}".rstrip()


@dataclass(frozen=True, slots=True)
class Report:
    """What this configuration will produce."""

    findings: tuple[Finding, ...] = ()
    matrix: dict[str, Any] = field(default_factory=dict)
    will_report: tuple[str, ...] = ()
    will_abstain: tuple[str, ...] = ()
    unpublishable: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the run would measure what it claims."""
        return not any(f.severity.is_fatal for f in self.findings)

    def of(self, severity: Severity) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is severity)

    def render(self) -> str:
        """The doctor output. Sorted by severity, because the first line matters."""
        lines: list[str] = []
        if self.matrix:
            pairs = "  ".join(f"{k}={v}" for k, v in self.matrix.items())
            lines.append(f"{'MATRIX':<11}{pairs}")
        for finding in sorted(self.findings, key=lambda f: _ORDER[f.severity]):
            lines.append(str(finding))

        summary = (
            f"{len(self.will_report)} metrics will report, {len(self.will_abstain)} N/A"
        )
        if self.unpublishable:
            summary += f", {len(self.unpublishable)} unattested"
        blocked = len([f for f in self.findings if f.severity.is_fatal])
        if blocked:
            summary += f", {blocked} blocked"
        lines.append(
            f"{(Severity.OK if self.ok else Severity.ERROR).value:<11}{summary}"
        )
        return "\n".join(lines)


def diagnose(
    *,
    capabilities: Capabilities | None = None,
    agent: str | None = None,
    requirement: Requirement | None = None,
    profile: str | Profile | None = None,
    metrics: Sequence[str] | None = None,
    providers: dict[str, str] | None = None,
    provider_registry: ProviderRegistry | None = None,
    metric_registry: MetricRegistry | None = None,
    getenv: Any = None,
) -> Report:
    """Resolve task x agent x profile x metrics and say what will happen.

    Every argument is optional, because ``doctor`` has to be useful on a
    half-written task. A missing piece becomes a finding, not an exception —
    a preflight that crashes on incomplete input is one nobody runs early
    enough to benefit from.
    """
    findings: list[Finding] = []
    metric_reg = REGISTRY if metric_registry is None else metric_registry
    matrix: dict[str, Any] = {}

    resolved_profile = _resolve_profile(profile, findings)
    if capabilities is not None:
        matrix.update(
            duplex="|".join(sorted(d.value for d in capabilities.duplex)),
            pipeline=capabilities.pipeline.value,
            turn_control=capabilities.turn_control.value,
        )
    if resolved_profile is not None:
        matrix["profile"] = resolved_profile.name

    _check_requirements(capabilities, requirement, agent, findings)
    _check_profile(resolved_profile, findings)
    active = (
        frozenset(resolved_profile.layers)
        if resolved_profile is not None
        else frozenset()
    )
    selected = _check_metrics(metrics, metric_reg, active, findings)
    _check_providers(providers, provider_registry, findings, getenv)

    will_report, will_abstain, unpublishable = _project(
        selected, metric_reg, capabilities, active, findings
    )
    return Report(
        findings=tuple(findings),
        matrix=matrix,
        will_report=will_report,
        will_abstain=will_abstain,
        unpublishable=unpublishable,
    )


def _resolve_profile(
    profile: str | Profile | None, findings: list[Finding]
) -> Profile | None:
    if profile is None or isinstance(profile, Profile):
        return profile
    try:
        return get_profile(profile)
    except UnknownProfile as exc:
        findings.append(Finding(Severity.ERROR, "profile", str(exc)))
        return None


def _check_requirements(
    capabilities: Capabilities | None,
    requirement: Requirement | None,
    agent: str | None,
    findings: list[Finding],
) -> None:
    """Rule 1: a task demanding what no party can serve is a load error."""
    if capabilities is None:
        findings.append(
            Finding(
                Severity.WARN,
                "agent",
                "no capabilities given; metric applicability cannot be checked",
            )
        )
        return
    if requirement is None:
        return
    # Returns *every* violation rather than raising on the first: fixing one
    # only to meet the next is a three-run debug cycle, and doctor exists to
    # make that a single run.
    for violation in check_requirements(
        requirement, capabilities, party=agent or "agent"
    ):
        findings.append(Finding(Severity.ERROR, "requires", violation))


def _check_profile(profile: Profile | None, findings: list[Finding]) -> None:
    if profile is None:
        return
    if not profile.is_buildable:
        findings.append(
            Finding(
                Severity.BLOCKED,
                f"profile.{profile.name}",
                profile.note or "declared but not implemented yet",
            )
        )
        return
    for layer, readers in summarise(profile).paired_metrics.items():
        if not readers:
            findings.append(
                Finding(
                    Severity.WARN,
                    layer,
                    "active but no registered metric reads it; "
                    "the perturbation only makes the number worse",
                )
            )


def _check_metrics(
    metrics: Sequence[str] | None,
    registry: MetricRegistry,
    active: frozenset[Any],
    findings: list[Finding],
) -> list[str]:
    from harbor_voice.metrics.task import check_selection

    selected = list(metrics) if metrics is not None else registry.names()
    for problem in check_selection(
        selected, registry=registry, active_layers=active, strict_layers=False
    ):
        findings.append(Finding(Severity.ERROR, "metric", problem.detail))
    return [name for name in selected if name in set(registry.names())]


def _check_providers(
    providers: dict[str, str] | None,
    registry: ProviderRegistry | None,
    findings: list[Finding],
    getenv: Any,
) -> None:
    """Names, capabilities and credentials — without constructing anything."""
    if not providers:
        return
    if registry is None:
        findings.append(
            Finding(Severity.WARN, "providers", "no registry given; not checked")
        )
        return
    for slot, spec_string in providers.items():
        try:
            entry, model = registry.resolve(spec_string)
        except (UnknownProvider, ValueError) as exc:
            findings.append(Finding(Severity.ERROR, f"{slot}", str(exc)))
            continue

        if not entry.is_implemented:
            findings.append(
                Finding(
                    Severity.BLOCKED,
                    f"{slot}={entry.spec.name}",
                    "registered but not implemented",
                )
            )
        status = registry.credentials(
            spec_string, **({"getenv": getenv} if getenv is not None else {})
        )
        if not status.needs_nothing and not status.ready:
            findings.append(
                Finding(Severity.ERROR, f"{slot}.credentials", status.explain())
            )
        if entry.spec.capabilities.attested == frozenset():
            findings.append(
                Finding(
                    Severity.UNATTESTED,
                    f"{slot}={entry.spec.name}/{model}",
                    "no conformance run on record; its numbers cannot gate a "
                    "published claim",
                )
            )


def _project(
    selected: Sequence[str],
    registry: MetricRegistry,
    capabilities: Capabilities | None,
    active: frozenset[Any],
    findings: list[Finding],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Say which metrics will report, abstain, or report unpublishably.

    Uses the real applicability path rather than a parallel guess, so what
    ``doctor`` predicts and what the run produces cannot drift apart.
    """
    if capabilities is None:
        return tuple(selected), (), ()

    from harbor_voice.capabilities import Precondition
    from harbor_voice.capabilities import evaluate as evaluate_preconditions

    reporting: list[str] = []
    abstaining: list[str] = []
    unattested: list[str] = []

    for name in selected:
        defn = registry.get(name)
        if not defn.requires:
            reporting.append(name)
            continue
        verdict = evaluate_preconditions(
            tuple(Precondition.parse(r) for r in defn.requires),
            capabilities,
            active_layers=active,
        )
        match verdict.verdict:
            case Verdict.NOT_APPLICABLE:
                abstaining.append(name)
                findings.append(
                    Finding(Severity.NOT_APPLICABLE, name, "; ".join(verdict.reasons))
                )
            case Verdict.UNATTESTED:
                reporting.append(name)
                unattested.append(name)
                findings.append(
                    Finding(Severity.UNATTESTED, name, "; ".join(verdict.reasons))
                )
            case Verdict.SATISFIED:
                reporting.append(name)
    return tuple(reporting), tuple(abstaining), tuple(unattested)
