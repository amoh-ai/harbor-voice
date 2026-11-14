"""EVA's run-level aggregation, over trials the sidecar graded one by one.

harbor-voice runs each trial in its own sidecar; ``GET /files`` returns each
trial's EVA run directory. This lays their ``metrics.json`` files out as one
EVA run (``records/<record>/trial_<k>/metrics.json``) and runs EVA's own
``--aggregate-only`` on it (``MetricsRunner.run_aggregate_only``): per-metric
means, the EVA composites, pass@k and the bootstrap CIs, seeded from the run
id as EVA seeds them (``run_seed(run_dir.name)``). Pooling across domains is
done by EVA's analysis scripts, outside ``src/``, and not here.

Run in the sidecar image; prints EVA's ``metrics_summary.json``::

    python /opt/eva-sidecar/aggregate.py --run-id RUN_ID --num-draws 3 \\
        TRIAL_DIR [TRIAL_DIR ...] > metrics_summary.json

Each ``TRIAL_DIR`` is an extracted ``/files`` tar (or any EVA run
directory). Archived attempts (``_failed_attempt_N``) are left out, as EVA
leaves them out.

EVA finds records with ``Path.iterdir()`` and does not sort them, and the
bootstrap CIs depend on the order of the values. So EVA's CIs change with how
the filesystem lists folders, which differs between machines and kernels.
Here the records go to EVA sorted by id, so the same trials always give the
same summary. Means, counts and pass@k do not depend on the order.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from eva.metrics.runner import MetricsRunner
from eva.utils.pass_at_k import ATTEMPT_SUFFIX_PATTERN


def collect(sources: list[Path]) -> dict[str, Path]:
    """Each trial's ``metrics.json``, by EVA output id (``<record>/trial_<k>``)."""
    found: dict[str, Path] = {}
    for source in sources:
        for record_id, record_dir in MetricsRunner._discover_record_dirs(source):
            if ATTEMPT_SUFFIX_PATTERN.search(record_id):
                continue
            metrics = record_dir / "metrics.json"
            if not metrics.exists():
                continue
            if record_id in found:
                raise SystemExit(
                    f"{record_id} is in more than one input ({found[record_id]}, {metrics})"
                )
            found[record_id] = metrics
    return found


def aggregate(sources: list[Path], run_id: str, num_draws: int) -> str:
    """EVA's ``metrics_summary.json`` text for the trials in ``sources``."""
    found = collect(sources)
    if not found:
        raise SystemExit("no metrics.json found in the inputs")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / run_id
        for record_id, metrics in found.items():
            target = run_dir / "records" / record_id / "metrics.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(metrics, target)
        discover = MetricsRunner._discover_record_dirs
        MetricsRunner._discover_record_dirs = staticmethod(  # type: ignore[method-assign]
            lambda run_dir, record_ids=None: sorted(discover(run_dir, record_ids))
        )
        try:
            asyncio.run(MetricsRunner.run_aggregate_only(run_dir, num_draws=num_draws))
        finally:
            MetricsRunner._discover_record_dirs = staticmethod(discover)  # type: ignore[method-assign]
        return (run_dir / "metrics_summary.json").read_text()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-id", required=True, help="EVA's run id; it seeds the bootstrap CIs"
    )
    parser.add_argument(
        "--num-draws", type=int, default=3, help="k for pass@k (EVA's num_trials)"
    )
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.write(aggregate(args.sources, args.run_id, args.num_draws))


if __name__ == "__main__":
    main()
