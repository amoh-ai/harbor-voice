"""What ``tests/test.sh`` calls.

::

    #!/bin/bash
    set -euo pipefail
    python3 -m harbor_voice.verify \\
        --trajectory /logs/trajectory.json \\
        --out /logs/verifier \\
        --metric divergence.wer_said_heard \\
        --metric timing.response_latency_s

Scoring is a library call: authors select metrics, they do not parse JSON.
This is that call, given a name and a CLI so a shell script can reach it.

**It reads one file and writes three.** That restraint is trajectory-only scoring turned into a
deployment property: everything scoring needs is in the trajectory, so
``regrade`` over stored trajectories is the *same computation* as the original
scoring rather than a second implementation that drifts. The moment this
needed a transport, a credential or a live session, that guarantee would be
gone and nobody would notice until two numbers disagreed.

It is also why this module imports neither the provider layer nor the recorder.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from harbor_voice.metrics.base import REGISTRY
from harbor_voice.metrics.base import MetricContext
from harbor_voice.metrics.base import MetricRegistry
from harbor_voice.metrics.outputs import Outputs
from harbor_voice.metrics.outputs import build_outputs
from harbor_voice.metrics.task import require_valid_selection
from harbor_voice.trajectory.model import VoiceTrajectory

DEFAULT_TRAJECTORY = Path("/logs/trajectory.json")
DEFAULT_OUT = Path("/logs/verifier")


def load_trajectory(path: Path) -> VoiceTrajectory:
    """Read an ATIF-voice document, from either of its two shapes.

    A file written by :meth:`VoiceTrajectory.to_atif` is a plain ATIF
    ``Trajectory`` carrying the voice payload under ``extra``; one written
    directly is the voice document itself. Both are legitimate on disk, so
    both are accepted rather than requiring the author to know which producer
    wrote their file.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "extra" in payload:
        from harbor.models.trajectories.trajectory import Trajectory

        return VoiceTrajectory.from_atif(Trajectory.model_validate(payload))
    return VoiceTrajectory.model_validate(payload)


def score(
    trajectory: VoiceTrajectory,
    *,
    metrics: Sequence[str] | None = None,
    registry: MetricRegistry | None = None,
    clean_baseline: VoiceTrajectory | None = None,
    validate_selection: bool = True,
) -> Outputs:
    """Score a trajectory and build the three documents.

    Args:
        metrics: Names to report. ``None`` means every registered metric,
            which is the right default for exploration and the wrong one for
            a published task — a task pins its selection in ``voice.toml`` so
            that adding a metric to harbor-voice cannot silently change what
            an existing benchmark reports.
        validate_selection: Check the selection against the registry and
            against the perturbation–metric pairing before scoring. A typo becomes a load error rather
            than a missing column.
    """
    reg = REGISTRY if registry is None else registry
    context = MetricContext(trajectory=trajectory, clean_baseline=clean_baseline)
    names = list(metrics) if metrics is not None else reg.names()

    if validate_selection:
        require_valid_selection(
            names, registry=reg, active_layers=context.active_layers
        )

    results = reg.evaluate(names, context)
    matrix = trajectory.config_matrix
    return build_outputs(
        results,
        trajectory.validity,
        trial_id=trajectory.trial_id,
        task=trajectory.task,
        agent=trajectory.agent,
        config_matrix=matrix.model_dump(mode="json") if matrix else {},
        comparable_on=trajectory.comparable_on,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Returns a process exit code.

    ``0`` for a scored trial, **valid or not**. An invalid measurement is not
    a verifier failure — it is a measurement that did not hold, recorded as
    such in ``validity.json`` and quarantined by the job. Exiting
    non-zero would make harbor treat it as a crashed test, which is precisely
    the conflation the three-document split exists to prevent.

    ``2`` is reserved for the verifier itself being wrong: an unreadable
    trajectory, an unknown metric, an unpaired layer.
    """
    parser = argparse.ArgumentParser(
        prog="harbor_voice.verify",
        description="Score an ATIF-voice trajectory into harbor's verifier outputs.",
    )
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric to report; repeatable. Omit to report every registered metric.",
    )
    parser.add_argument(
        "--clean-baseline",
        type=Path,
        default=None,
        help="Matched-seed clean trajectory, for the robustness family.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip selection validation. For debugging a broken task only.",
    )
    args = parser.parse_args(argv)

    try:
        trajectory = load_trajectory(args.trajectory)
        baseline = load_trajectory(args.clean_baseline) if args.clean_baseline else None
        outputs = score(
            trajectory,
            metrics=args.metrics,
            clean_baseline=baseline,
            validate_selection=not args.no_validate,
        )
    except Exception as exc:  # noqa: BLE001 - the CLI boundary; reported, not raised
        print(f"verify: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    outputs.write(args.out)
    counts = outputs.measurement["counts"]
    print(
        f"verify: {counts['reported']} reported, "
        f"{counts['not_applicable']} n/a, "
        f"{counts['unattested']} unattested, "
        f"measurement_valid={outputs.measurement_valid}"
    )
    if not outputs.measurement_valid:
        for reason in outputs.validity["reasons"]:
            print(f"  invalid: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
