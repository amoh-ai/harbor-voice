"""The task's verifier: EVA's grade, from the ``eva`` sidecar.

Run by ``tests/test.sh`` in ``main``, as harbor runs every verifier. The
grade needs EVA, so the sidecar runs it (``POST /grade``) on the record it
holds. The call driver graded already; EVA's ``MetricsRunner`` computes only
the metrics missing from the record's ``metrics.json``, so this calls no
judge again and reads back the driver's numbers.

Writes ``result.json`` and, when EVA scored the record, ``reward.json``:
``reward`` is the chosen EVA composite (``--reward``), and each of EVA's
composites (``EVA-A_pass``, ``EVA-X_pass``, ``EVA-overall_pass``) is beside it
under its own name. A record that no attempt validated (``/grade`` answers
409), or whose chosen composite EVA leaves unset (a component metric
missing), has no reward: EVA leaves it out of its means, and harbor then
reports the trial as unscored.

Standard library only: it runs on ``main``'s system Python.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

COMPOSITES = ("EVA-A_pass", "EVA-X_pass", "EVA-overall_pass")
"""EVA's record composites (``eva_adapter.trial.REWARDS``)."""


def rewards(aggregates: dict[str, Any], reward_key: str) -> dict[str, float]:
    """``reward``, then each composite EVA set; empty when the chosen one is
    unset (``eva_adapter.trial.rewards``)."""
    if aggregates.get(reward_key) is None:
        return {}
    named = {
        key: float(aggregates[key])
        for key in COMPOSITES
        if aggregates.get(key) is not None
    }
    return {"reward": named[reward_key], **named}


def scores(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each EVA metric's headline numbers, without its details."""
    return {
        name: {
            key: score.get(key)
            for key in ("score", "normalized_score", "skipped", "error")
        }
        for name, score in metrics.items()
        if isinstance(score, dict)
    }


def grade(sidecar: str, reward_key: str, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{sidecar.rstrip('/')}/grade",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            graded = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            raise
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("detail", body)
        except ValueError:
            detail = body
        return {
            "graded": False,
            "reward_key": reward_key,
            "reward": None,
            "detail": detail,
        }
    value = graded["aggregates"].get(reward_key)
    return {
        "graded": True,
        "reward_key": reward_key,
        "reward": None if value is None else float(value),
        "rewards": rewards(graded["aggregates"], reward_key),
        "record_id": graded["record_id"],
        "aggregates": graded["aggregates"],
        "metrics": scores(graded["metrics"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reward", required=True, help="the EVA composite")
    parser.add_argument("--sidecar", default="http://eva:8000")
    parser.add_argument("--out", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    result = grade(args.sidecar, args.reward, args.timeout)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if result.get("rewards"):
        (args.out / "reward.json").write_text(
            json.dumps(result["rewards"]), encoding="utf-8"
        )
    print(json.dumps({"reward": result["reward"], "graded": result["graded"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
