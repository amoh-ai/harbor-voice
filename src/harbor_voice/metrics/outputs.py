"""Three outputs, layered so harbor's waist stays intact.

::

    tests/test.sh
        │  (a library call — authors select metrics, they do not parse JSON)
        ├──► measurement.json   rich: four columns, N/A, abstain, validity
        ├──► reward.json        dict[str, float]  <- BYTE-COMPATIBLE WITH HARBOR
        └──► validity.json      measurement_valid + named checks

The layering is the design. Harbor's ``reward.json`` is ``dict[str, float|int]``
and ``aggregate_reward_dicts`` unions keys across trials and aggregates each
independently — a genuinely multi-dimensional surface. We keep it unchanged,
so harbor's metrics, hub and leaderboard work with zero modification, and put
everything harbor has no vocabulary for into the two files beside it.

What harbor has no vocabulary for is the interesting part:

``not_applicable``
    Harbor would have to write ``0``. A metric that does not apply and a
    metric that scored zero are not the same claim.
``measurement_invalid``
    Harbor's aggregator maps a ``None`` reward to ``0``, putting a crashed
    trial in the same bucket as a genuine failure.
``unattested``
    A number resting on a capability no conformance run verified. Usable for
    debugging, not for a paper.

**An invalid trial writes no ``reward.json`` at all.** Writing one with
zeros would be indistinguishable from a bad agent the moment harbor aggregates
it, and the whole point of the three-file split is that the distinction
survives. An absent file is the only encoding harbor already understands as
"nothing to count here" — and ``validity.json`` says why.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from harbor_voice.capabilities import Verdict
from harbor_voice.metrics.base import MetricResult
from harbor_voice.trajectory.model import ValidityRecord

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.base import Usage
    from harbor_voice.providers.cache import Cassette

MEASUREMENT_SCHEMA = "voice-measurement.1"

REWARD_FILENAME = "reward.json"
MEASUREMENT_FILENAME = "measurement.json"
VALIDITY_FILENAME = "validity.json"


@dataclass(frozen=True, slots=True)
class Outputs:
    """The three documents a trial produces."""

    measurement: dict[str, Any]
    validity: dict[str, Any]
    reward: dict[str, float] | None
    """``None`` for an invalid trial — see the module docstring."""

    @property
    def measurement_valid(self) -> bool:
        return bool(self.validity["measurement_valid"])

    def write(self, directory: Path) -> list[Path]:
        """Write the documents, returning what was written.

        ``reward.json`` is removed when the trial is invalid rather than
        left behind from an earlier attempt — a stale reward file next to a
        fresh invalidity record is the worst of both.
        """
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for filename, payload in (
            (MEASUREMENT_FILENAME, self.measurement),
            (VALIDITY_FILENAME, self.validity),
        ):
            path = directory / filename
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            written.append(path)

        reward_path = directory / REWARD_FILENAME
        if self.reward is None:
            reward_path.unlink(missing_ok=True)
        else:
            reward_path.write_text(
                json.dumps(self.reward, indent=2, sort_keys=True), encoding="utf-8"
            )
            written.append(reward_path)
        return written


def build_reward(results: Sequence[MetricResult]) -> dict[str, float]:
    """The harbor-compatible surface: ``dict[str, float]``, nothing else.

    Only metrics that reported a number appear. An inapplicable metric is
    omitted rather than zeroed, which is the honest encoding available within
    harbor's format: ``aggregate_reward_dicts`` unions keys across trials and
    substitutes ``0`` for a key a given trial lacks.

    That substitution is a real limitation and it is worth being precise about
    where it bites. Across a run where a metric is *consistently*
    inapplicable, the key never appears and nothing is distorted. Where it is
    applicable for some trials and not others, harbor will average zeros in.
    ``measurement.json`` carries the ``n_not_applicable`` count that makes
    this visible, and that asymmetry is exactly why the richer file exists.
    """
    return {
        result.name: float(result.value)
        for result in results
        if result.value is not None
    }


def build_measurement(
    results: Sequence[MetricResult],
    *,
    trial_id: str | None = None,
    task: str | None = None,
    agent: str | None = None,
    config_matrix: dict[str, Any] | None = None,
    providers: Sequence[dict[str, Any]] = (),
    comparable_on: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The rich document: four columns per metric, plus what produced them."""
    metrics: dict[str, Any] = {r.name: r.as_dict() for r in results}
    counts = {
        "reported": sum(1 for r in results if r.reported),
        "not_applicable": sum(
            1 for r in results if r.verdict is Verdict.NOT_APPLICABLE
        ),
        "unattested": sum(1 for r in results if r.verdict is Verdict.UNATTESTED),
        "publishable": sum(1 for r in results if r.publishable),
    }
    return {
        "schema": MEASUREMENT_SCHEMA,
        "trial_id": trial_id,
        "task": task,
        "agent": agent,
        "config_matrix": config_matrix or {},
        "comparable_on": comparable_on or {},
        "providers": list(providers),
        "counts": counts,
        "metrics": metrics,
    }


def build_validity(
    record: ValidityRecord,
    *,
    extra_checks: Sequence[tuple[str, bool, str | None]] = (),
) -> dict[str, Any]:
    """The quarantine channel.

    ``measurement_valid`` is the conjunction of every check, recomputed here
    rather than trusted from the record: the record's flag and its checks can
    disagree if one was set by hand, and a validity file that says ``true``
    above a failed check is worse than no file.
    """
    checks: list[dict[str, Any]] = [
        {"name": c.name, "ok": c.ok, "detail": c.detail, "score": c.score}
        for c in record.checks
    ]
    checks.extend(
        {"name": name, "ok": ok, "detail": detail, "score": None}
        for name, ok, detail in extra_checks
    )
    valid = record.measurement_valid and all(c["ok"] for c in checks)
    return {
        "measurement_valid": valid,
        "checks": checks,
        "reasons": [c["detail"] or c["name"] for c in checks if not c["ok"]],
    }


def build_outputs(
    results: Sequence[MetricResult],
    validity: ValidityRecord,
    *,
    trial_id: str | None = None,
    task: str | None = None,
    agent: str | None = None,
    config_matrix: dict[str, Any] | None = None,
    comparable_on: dict[str, Any] | None = None,
    providers: Sequence[dict[str, Any]] = (),
    usages: Sequence[Usage] = (),
    cassette: Cassette | None = None,
) -> Outputs:
    """Build all three, refusing to publish a replayed run.

    The ``require_live`` call is the enforcement point: the agent-side
    cassette exists to make CI cheap, and the moment a replayed call can feed
    a reported number it has become a way to fabricate results. Checked here,
    at the one place a number becomes a published artifact, rather than at
    each of the call sites that could produce one.

    Raises:
        ReplayedRunError: If any provider call was served from a cassette.
    """
    from harbor_voice.providers.cache import require_live

    require_live(usages, cassette=cassette)

    validity_doc = build_validity(validity)
    measurement = build_measurement(
        results,
        trial_id=trial_id,
        task=task,
        agent=agent,
        config_matrix=config_matrix,
        providers=providers,
        comparable_on=comparable_on,
    )
    measurement["measurement_valid"] = validity_doc["measurement_valid"]

    return Outputs(
        measurement=measurement,
        validity=validity_doc,
        reward=(build_reward(results) if validity_doc["measurement_valid"] else None),
    )
