"""The parity sample: which tau2 tasks both sides of a comparison run.

::

    tau-voice sample --tau2-root ../tau2-bench       # writes parity_tasks.json
    tau-voice sample --tau2-root ../tau2-bench --seed 1 -o tasks.json

A fixed-seed random sample of tau2's ``test`` split (the split the τ-voice
leaderboard reports), proportional per domain (:data:`SAMPLE`), ids in split
order. The converter's ``--split parity`` reads the checked-in file, and
tau2's own run takes the same ids as ``--task-ids``
(:func:`tau_voice.original.tau2_command`).
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tau_voice.main import PARITY_TASKS
from tau_voice.main import TAU2_RUN_SEED

SAMPLE = {"airline": 8, "retail": 16, "telecom": 16}
"""40 tasks, proportional to the ``test`` split (20 / 40 / 40)."""


def sample(
    tau2_root: Path, *, split: str = "test", seed: int = TAU2_RUN_SEED
) -> dict[str, Any]:
    """A fixed-seed random sample of ``split``, per domain, in split order."""
    rng = random.Random(seed)
    chosen: dict[str, Any] = {"split": split, "seed": seed}
    for domain, count in SAMPLE.items():
        path = tau2_root / "data" / "tau2" / "domains" / domain / "split_tasks.json"
        ids = list(json.loads(path.read_text(encoding="utf-8"))[split])
        picked = set(rng.sample(ids, count))
        chosen[domain] = [task for task in ids if task in picked]
    return chosen


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """``sample``'s options."""
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=TAU2_RUN_SEED)
    parser.add_argument("-o", "--out", type=Path, default=PARITY_TASKS)


def write_sample(args: argparse.Namespace) -> Path:
    """Draw the sample the options name and write it (by default the
    checked-in ``parity_tasks.json``)."""
    tasks = sample(args.tau2_root, split=args.split, seed=args.seed)
    args.out.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    counts = {d: len(tasks[d]) for d in SAMPLE}
    print(f"wrote {args.out}: {counts}")
    return args.out


def main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="tau-voice sample",
        description="write the tasks both sides of a parity comparison run",
    )
    add_arguments(parser)
    write_sample(parser.parse_args(list(argv)))
