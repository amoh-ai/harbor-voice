"""Aggregating across trials.

**Report the median and the failure rate alongside the mean.** The mean
of a metric with a heavy failure tail is largely a *failure-rate statistic*
wearing a central-tendency costume: it moves when the number of catastrophes
moves and barely notices what the typical trial did. Reporting all three is
strictly more informative and costs nothing.

**Invalid trials leave the denominator.** Harbor's
``aggregate_reward_dicts`` maps a ``None`` reward to ``0``, which puts a
crashed trial in the same bucket as an agent that genuinely scored zero. We do
not inherit that: an invalid trial is reported as a rate and excluded from
every statistic, because averaging in a number that was never measured is how
a harness bug becomes a model result.

And before claiming any margin, bootstrap: a published difference smaller than
the spread of a single draw is not a finding.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from harbor_voice.capabilities import Verdict
from harbor_voice.metrics.base import Direction
from harbor_voice.metrics.base import Family
from harbor_voice.metrics.base import MetricResult
from harbor_voice.metrics.outputs import MEASUREMENT_FILENAME
from harbor_voice.metrics.outputs import VALIDITY_FILENAME

DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 1234


@dataclass(frozen=True, slots=True)
class Bootstrap:
    """What a single draw of this many trials is worth."""

    std: float
    low: float
    high: float
    samples: int
    confidence: float = 0.95

    @property
    def width(self) -> float:
        """The interval's span — the margin a claim has to beat."""
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class Rollup:
    """One metric across many trials."""

    name: str
    family: Family
    direction: Direction
    unit: str | None = None

    n_trials: int = 0
    """Every trial considered, valid or not."""

    n_reported: int = 0
    """Trials that produced a number."""

    n_not_applicable: int = 0
    n_invalid: int = 0
    """Excluded from every statistic below."""

    n_unpublishable: int = 0
    """Reported but resting on an unattested capability — rule 3."""

    mean: float | None = None
    median: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    failure_rate: float | None = None
    failure_threshold: float | None = None
    bootstrap: Bootstrap | None = None

    @property
    def applicable_rate(self) -> float | None:
        """Fraction of valid trials that could report at all."""
        denominator = self.n_trials - self.n_invalid
        return self.n_reported / denominator if denominator else None

    @property
    def publishable(self) -> bool:
        """Whether every contributing trial was fully attested."""
        return self.n_reported > 0 and self.n_unpublishable == 0

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "family": self.family.value,
            "direction": self.direction.value,
            "n_trials": self.n_trials,
            "n_reported": self.n_reported,
            "n_not_applicable": self.n_not_applicable,
            "n_invalid": self.n_invalid,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
        }
        if self.unit is not None:
            out["unit"] = self.unit
        if self.failure_rate is not None:
            out["failure_rate"] = self.failure_rate
            out["failure_threshold"] = self.failure_threshold
        if self.minimum is not None:
            out["min"] = self.minimum
            out["max"] = self.maximum
        if self.n_unpublishable:
            out["n_unpublishable"] = self.n_unpublishable
        if self.bootstrap is not None:
            out["bootstrap"] = {
                "std": self.bootstrap.std,
                "ci_low": self.bootstrap.low,
                "ci_high": self.bootstrap.high,
                "samples": self.bootstrap.samples,
                "confidence": self.bootstrap.confidence,
            }
        return out


@dataclass(frozen=True, slots=True)
class TrialResults:
    """One trial's metric results, plus whether the trial counted at all."""

    results: Sequence[MetricResult]
    measurement_valid: bool = True
    trial_id: str | None = None
    comparable_on: dict[str, Any] = field(default_factory=dict)
    """The pinned dimensions this trial may be compared along.

    Carried from the trajectory through ``measurement.json`` so the robustness
    family can pair a degraded run with its baseline from **data** rather than
    from a filename convention somebody has to remember.
    """

    def get(self, name: str) -> MetricResult | None:
        for result in self.results:
            if result.name == name:
                return result
        return None

    @property
    def profile(self) -> str | None:
        """Which named condition produced this trial."""
        value = self.comparable_on.get("profile")
        return None if value is None else str(value)

    @property
    def pairing_key(self) -> tuple[tuple[str, str], ...]:
        """Everything that must match for a Δ to mean anything — except profile.

        Profile is the one dimension a robustness comparison is *allowed* to
        differ on. Everything else — task, duplex, pipeline, seed, schema —
        has to be identical, or the delta is measuring two changes at once.
        """
        return tuple(
            sorted(
                (k, repr(v)) for k, v in self.comparable_on.items() if k != "profile"
            )
        )


def is_failure(value: float, threshold: float, direction: Direction) -> bool:
    """Whether a value counts as a catastrophic trial.

    Direction decides which side of the threshold is bad. ``NEUTRAL`` metrics
    have no failure notion at all — overlap of 0.4 is not a failure, it is a
    description — so they are never counted, rather than being counted with
    an arbitrary polarity.
    """
    match direction:
        case Direction.HIGHER_BETTER:
            return value < threshold
        case Direction.LOWER_BETTER:
            return value > threshold
        case Direction.NEUTRAL:
            return False


def bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> Bootstrap | None:
    """Resample with replacement to size what one draw of these trials is worth.

    Seeded: an unseeded bootstrap gives a slightly different
    confidence interval every time the same data is re-scored, and someone
    will eventually notice and distrust the whole table.

    Returns ``None`` below two values — a confidence interval over one
    observation is not a wide interval, it is no interval.
    """
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()

    centre = sum(means) / len(means)
    variance = sum((m - centre) ** 2 for m in means) / (len(means) - 1)
    tail = (1.0 - confidence) / 2.0
    low = means[max(0, int(tail * len(means)))]
    high = means[min(len(means) - 1, int((1.0 - tail) * len(means)))]
    return Bootstrap(
        std=math.sqrt(variance),
        low=low,
        high=high,
        samples=samples,
        confidence=confidence,
    )


def roll_up(
    name: str,
    trials: Sequence[TrialResults],
    *,
    failure_threshold: float | None = None,
    bootstrap: bool = True,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Rollup | None:
    """Aggregate one metric across trials.

    Returns ``None`` when no trial even knows the metric — different from a
    metric every trial found inapplicable, which is a real and reportable
    state.
    """
    known = [t for t in trials if t.get(name) is not None]
    if not known:
        return None

    exemplar = known[0].get(name)
    if exemplar is None:  # pragma: no cover - `known` filtered on this
        return None

    n_invalid = sum(1 for t in trials if not t.measurement_valid)
    values: list[float] = []
    n_not_applicable = 0
    n_unpublishable = 0

    for trial in trials:
        if not trial.measurement_valid:
            continue
        result = trial.get(name)
        if result is None or result.value is None:
            n_not_applicable += 1
            continue
        values.append(result.value)
        if result.verdict is not Verdict.SATISFIED:
            n_unpublishable += 1

    base = Rollup(
        name=name,
        family=exemplar.family,
        direction=exemplar.direction,
        unit=exemplar.unit,
        n_trials=len(trials),
        n_reported=len(values),
        n_not_applicable=n_not_applicable,
        n_invalid=n_invalid,
        n_unpublishable=n_unpublishable,
    )
    if not values:
        return base

    ordered = sorted(values)
    mean = sum(values) / len(values)
    variance = (
        sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    failure_rate = None
    if failure_threshold is not None:
        failures = sum(
            1 for v in values if is_failure(v, failure_threshold, exemplar.direction)
        )
        failure_rate = failures / len(values)

    return Rollup(
        name=base.name,
        family=base.family,
        direction=base.direction,
        unit=base.unit,
        n_trials=base.n_trials,
        n_reported=base.n_reported,
        n_not_applicable=base.n_not_applicable,
        n_invalid=base.n_invalid,
        n_unpublishable=base.n_unpublishable,
        mean=mean,
        median=_median(ordered),
        std=math.sqrt(variance),
        minimum=ordered[0],
        maximum=ordered[-1],
        failure_rate=failure_rate,
        failure_threshold=failure_threshold if failure_rate is not None else None,
        bootstrap=(
            bootstrap_mean(values, samples=bootstrap_samples, seed=seed)
            if bootstrap
            else None
        ),
    )


def roll_up_all(
    trials: Sequence[TrialResults],
    *,
    failure_thresholds: dict[str, float] | None = None,
    bootstrap: bool = True,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Rollup]:
    """Aggregate every metric any trial reported."""
    names = sorted({r.name for t in trials for r in t.results})
    thresholds = failure_thresholds or {}
    out: dict[str, Rollup] = {}
    for name in names:
        rollup = roll_up(
            name,
            trials,
            failure_threshold=thresholds.get(name),
            bootstrap=bootstrap,
            seed=seed,
        )
        if rollup is not None:
            out[name] = rollup
    return out


def paired_delta(
    name: str,
    degraded: Rollup | None,
    clean: Rollup | None,
) -> Rollup | None:
    """Δ against the matched clean baseline — the robustness family.

    Reported as *degraded minus clean* on the raw scale, with the direction
    carried over. Not expressed as a percentage: a WER going from 0.02 to 0.04
    is "+100%", which sounds alarming and means two words, while 0.40 to 0.44
    is "+10%" and means a broken call.

    Returns ``None`` unless both sides have a mean, because a delta against a
    missing baseline is not zero.
    """
    if degraded is None or clean is None:
        return None
    if degraded.mean is None or clean.mean is None:
        return None
    delta = degraded.mean - clean.mean
    return Rollup(
        name=f"robustness.delta.{name}",
        family=Family.ROBUSTNESS,
        direction=degraded.direction,
        unit=degraded.unit,
        n_trials=degraded.n_trials + clean.n_trials,
        n_reported=degraded.n_reported + clean.n_reported,
        n_invalid=degraded.n_invalid + clean.n_invalid,
        n_unpublishable=degraded.n_unpublishable + clean.n_unpublishable,
        mean=delta,
        median=(
            degraded.median - clean.median
            if degraded.median is not None and clean.median is not None
            else None
        ),
    )


# ----------------------------------------------------- the robustness family


DEFAULT_BASELINE_PROFILE = "clean"


@dataclass(frozen=True, slots=True)
class Unpaired:
    """A trial that produced no delta, and why.

    Named rather than dropped, for the same reason the rollup keeps invalid
    trials visible: a robustness table built from four pairs out of fifty
    trials looks exactly like one built from fifty, and the difference is the
    entire claim.
    """

    trial_id: str | None
    profile: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    """Δ against the baseline profile, per degraded profile."""

    baseline: str
    deltas: dict[str, dict[str, Rollup]] = field(default_factory=dict)
    """profile -> metric -> the delta rollup."""

    pairs: dict[str, int] = field(default_factory=dict)
    """profile -> how many trials were actually matched."""

    unpaired: tuple[Unpaired, ...] = ()

    @property
    def profiles(self) -> tuple[str, ...]:
        return tuple(sorted(self.deltas))

    @property
    def is_empty(self) -> bool:
        return not any(self.deltas.values())

    def flat(self) -> dict[str, float]:
        """One name per number, for a results table.

        ``robustness.telephony.delta.divergence.wer_said_heard`` — long, and
        every segment earns its place: which family, which condition, which
        metric. A shorter name would need a legend.
        """
        out: dict[str, float] = {}
        for profile, metrics in self.deltas.items():
            for name, rollup in metrics.items():
                if rollup.mean is not None:
                    out[f"robustness.{profile}.delta.{name}"] = rollup.mean
        return out


def robustness(
    trials: Sequence[TrialResults],
    *,
    baseline: str = DEFAULT_BASELINE_PROFILE,
    bootstrap: bool = True,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> RobustnessReport:
    """Pair each degraded trial with its matched baseline and diff them.

    The robustness row is "Δ metric vs the ``clean`` profile at matched
    seed". Matched means matched on *everything but the profile*: same task,
    same duplex, same pipeline, same seed, same schema. Pairing on seed alone
    would happily diff a half-duplex clean run against a full-duplex noisy
    one and call the difference robustness.

    Invalid trials are excluded before pairing, not after — a
    degraded run that was never valid has nothing to contribute, and pairing
    it would silently remove its baseline partner from the clean side too.
    """
    usable = [t for t in trials if t.measurement_valid]
    unpaired: list[Unpaired] = [
        Unpaired(t.trial_id, t.profile, "measurement_invalid")
        for t in trials
        if not t.measurement_valid
    ]

    baselines: dict[tuple[tuple[str, str], ...], TrialResults] = {}
    degraded: dict[str, list[tuple[TrialResults, tuple[tuple[str, str], ...]]]] = {}
    for trial in usable:
        if trial.profile is None:
            unpaired.append(Unpaired(trial.trial_id, None, "no profile recorded"))
            continue
        key = trial.pairing_key
        if trial.profile == baseline:
            if key in baselines:
                unpaired.append(
                    Unpaired(
                        trial.trial_id,
                        trial.profile,
                        "a second baseline shares this comparison key; the "
                        "first one was used",
                    )
                )
                continue
            baselines[key] = trial
        else:
            degraded.setdefault(trial.profile, []).append((trial, key))

    deltas: dict[str, dict[str, Rollup]] = {}
    pairs: dict[str, int] = {}
    for profile, entries in sorted(degraded.items()):
        matched: list[TrialResults] = []
        partners: list[TrialResults] = []
        for trial, key in entries:
            partner = baselines.get(key)
            if partner is None:
                unpaired.append(
                    Unpaired(
                        trial.trial_id,
                        profile,
                        f"no {baseline!r} trial shares its comparison key",
                    )
                )
                continue
            matched.append(trial)
            partners.append(partner)
        pairs[profile] = len(matched)
        if not matched:
            deltas[profile] = {}
            continue

        degraded_rolled = roll_up_all(matched, bootstrap=bootstrap, seed=seed)
        clean_rolled = roll_up_all(partners, bootstrap=bootstrap, seed=seed)
        per_metric: dict[str, Rollup] = {}
        for name, rolled in sorted(degraded_rolled.items()):
            delta = paired_delta(name, rolled, clean_rolled.get(name))
            if delta is not None:
                per_metric[name] = delta
        deltas[profile] = per_metric

    return RobustnessReport(
        baseline=baseline,
        deltas=deltas,
        pairs=pairs,
        unpaired=tuple(unpaired),
    )


def _median(ordered: Sequence[float]) -> float:
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


# ------------------------------------------- the job level, out of band


def load_trial_dir(directory: Path) -> TrialResults | None:
    """Read one trial's verifier output back into scoreable form.

    Reads ``measurement.json`` and ``validity.json`` — **not**
    ``reward.json``. The reward file is harbor's flat surface and has already
    lost the verdicts, the sample sizes and the reasons.

    Returns ``None`` when the directory holds no measurement, which is
    different from a measurement that came out invalid: the first is a trial
    that never scored, the second is one that scored and did not hold.
    """
    measurement_path = directory / MEASUREMENT_FILENAME
    if not measurement_path.exists():
        return None
    payload = json.loads(measurement_path.read_text(encoding="utf-8"))

    valid = bool(payload.get("measurement_valid", True))
    validity_path = directory / VALIDITY_FILENAME
    if validity_path.exists():
        validity = json.loads(validity_path.read_text(encoding="utf-8"))
        valid = bool(validity.get("measurement_valid", valid))

    results: list[MetricResult] = []
    for name, entry in (payload.get("metrics") or {}).items():
        results.append(
            MetricResult(
                name=name,
                family=Family(entry["family"]),
                direction=Direction(entry["direction"]),
                verdict=Verdict(entry["verdict"]),
                value=entry.get("value"),
                unit=entry.get("unit"),
                n=entry.get("n"),
                reasons=tuple(entry.get("reasons") or ()),
                detail=entry.get("detail") or {},
                benchmark=entry.get("benchmark"),
            )
        )
    return TrialResults(
        results=results,
        measurement_valid=valid,
        trial_id=payload.get("trial_id"),
        comparable_on=dict(payload.get("comparable_on") or {}),
    )


def load_job(root: Path) -> list[TrialResults]:
    """Collect every trial under ``root``, in a stable order.

    Sorted by path so a job's rollup is reproducible: filesystem order varies
    between machines, and a bootstrap seeded identically over differently
    ordered input is not the same computation.
    """
    trials: list[TrialResults] = []
    for measurement in sorted(root.rglob(MEASUREMENT_FILENAME)):
        trial = load_trial_dir(measurement.parent)
        if trial is not None:
            trials.append(trial)
    return trials


def harbor_metric_payload(
    trials: Sequence[TrialResults],
    *,
    primary: str | None = None,
) -> dict[str, float | int]:
    """What a conforming harbor ``metric.py`` should emit.

    Harbor's hub and leaderboard want a flat ``dict[str, float|int]``, and its
    own metric template reads a JSONL in which invalid trials have **already**
    been flattened to ``0`` — unrecoverably, since that file carries no trial
    ids and no validity column. So the invalidity travels out of band, through
    the trial directories, and arrives here.

    The payload reports the mean over **valid** trials and carries
    ``n_invalid`` beside it, so the number harbor displays and the number we
    publish cannot silently differ. Someone reading the leaderboard can see
    that trials were dropped and how many.
    """
    rolled = roll_up_all(trials, bootstrap=False)
    valid = sum(1 for t in trials if t.measurement_valid)
    payload: dict[str, float | int] = {
        "n_trials": len(trials),
        "n_valid": valid,
        "n_invalid": len(trials) - valid,
    }
    for name, rollup in rolled.items():
        if rollup.mean is not None:
            payload[name] = rollup.mean
        if rollup.median is not None:
            # Emitted only when it exists: a median of 0.0 is a real value, so
            # defaulting to it would publish a measurement nobody made.
            payload[f"{name}.median"] = rollup.median
    if primary is not None:
        chosen = rolled.get(primary)
        if chosen is not None and chosen.mean is not None:
            payload["mean"] = chosen.mean
    return payload
