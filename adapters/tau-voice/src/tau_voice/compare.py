"""Parity: the harbor side against tau2's own run, in harbor's format.

Reads harbor job folders (``harbor-voice jobs start``) and tau2 result folders
(``tau-voice original``), and reports, per domain and overall:

- **average reward** (Pass^1): the mean over graded tasks in each run, then
  mean ± sample SEM over runs, as harbor's ``parity_experiment.json`` states it;
- **Pass^k** for k = 1..runs, tau2's estimator: per task, C(c, k) / C(n, k)
  with c successes in n runs, averaged over tasks;
- diagnostics that explain a gap without being a parity claim: hallucination
  retries, termination reasons, call length, errored trials.

Errored trials (no grade) are left out of the averages and counted, as tau2
leaves infrastructure errors out::

    tau-voice compare --harbor jobs/parity --original parity/original \\
        --out parity/parity_experiment.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from tau_voice.main import TAU2_RUN_SEED
from tau_voice.main import trial_seeds


@dataclass(frozen=True, slots=True)
class Record:
    """One task in one run, on one side."""

    domain: str
    task_id: str
    run: int
    reward: float | None
    """None when the trial errored before it was graded."""
    retries: int = 0
    termination: str | None = None
    call_s: float | None = None


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def harbor_records(
    job_dirs: Iterable[Path], run_seed: int = TAU2_RUN_SEED
) -> list[Record]:
    """Trials of harbor jobs; the run is the index of the trial's tau2 seed."""
    seeds = trial_seeds(run_seed, 32)
    records = []
    for job in job_dirs:
        for result_path in sorted(job.glob("*/result.json")):
            trial = result_path.parent
            outcome_path = trial / "agent" / "call" / "outcome.json"
            config = _json(trial / "config.json")
            seed = (config.get("agent") or {}).get("kwargs", {}).get("seed")
            run = seeds.index(seed) if seed in seeds else 0
            result = _json(result_path)
            verifier = result.get("verifier_result") or {}
            reward = (verifier.get("rewards") or {}).get("reward")
            if not outcome_path.exists():
                task = str(result.get("task_name") or trial.name)
                domain, _, task_id = task.rpartition("/")[2].partition("-")
                records.append(Record(domain, task_id, run, None))
                continue
            summary = _json(outcome_path).get("summary") or {}
            attempts = summary.get("attempts") or [{}]
            records.append(
                Record(
                    domain=str(summary["domain"]),
                    task_id=str(summary["task_id"]),
                    run=run,
                    reward=None if reward is None else float(reward),
                    retries=len(attempts) - 1,
                    termination=attempts[-1].get("termination"),
                    call_s=attempts[-1].get("call_s"),
                )
            )
    return records


def _simulations(results_dir: Path) -> list[dict[str, Any]]:
    """tau2 saves either one ``results.json`` or one file per simulation."""
    results = _json(results_dir / "results.json")
    simulations = list(results.get("simulations") or [])
    if not simulations:
        simulations = [
            _json(p) for p in sorted((results_dir / "simulations").glob("*.json"))
        ]
    info = results.get("info") or {}
    domain = (info.get("environment_info") or {}).get("domain_name")
    tick_s = (info.get("audio_native_config") or {}).get("tick_duration_seconds")
    for simulation in simulations:
        simulation.setdefault("_domain", domain)
        simulation.setdefault("_tick_s", tick_s)
    return simulations


def original_records(result_dirs: Iterable[Path]) -> list[Record]:
    """tau2's own simulations; tau2 numbers its trials from 0."""
    records = []
    for results_dir in result_dirs:
        for sim in _simulations(results_dir):
            reward_info = sim.get("reward_info")
            reward = reward_info.get("reward") if reward_info else None
            # Call length in audio seconds (ticks x tick), like the harbor
            # side's; tau2's ``duration`` is wall time, overhead included.
            ticks = sim.get("ticks") or []
            tick_s = sim.get("_tick_s")
            call_s = len(ticks) * float(tick_s) if ticks and tick_s else None
            records.append(
                Record(
                    domain=str(
                        sim.get("_domain") or results_dir.name.rsplit("-", 1)[-1]
                    ),
                    task_id=str(sim["task_id"]),
                    run=int(sim.get("trial") or 0),
                    reward=None if reward is None else float(reward),
                    retries=int(sim.get("hallucination_retries_used") or 0),
                    termination=sim.get("termination_reason"),
                    call_s=None if call_s is None else round(call_s, 1),
                )
            )
    return records


def _found(paths: Iterable[Path], marker: str) -> list[Path]:
    """Every folder under ``paths`` holding ``marker`` (a job, a tau2 save)."""
    found: list[Path] = []
    for path in paths:
        if (path / marker).exists():
            found.append(path)
        found.extend(p.parent for p in sorted(path.glob(f"*/{marker}")))
    return sorted(set(found))


def run_means(records: list[Record]) -> list[float]:
    """Average reward of each run, over the tasks graded in it."""
    by_run: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record.reward is not None:
            by_run[record.run].append(record.reward)
    return [statistics.fmean(by_run[r]) for r in sorted(by_run)]


def pass_hat_k(records: list[Record], k: int) -> float | None:
    """tau2's Pass^k: per task C(c, k) / C(n, k), averaged over tasks."""
    by_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if record.reward is not None:
            by_task[(record.domain, record.task_id)].append(record.reward)
    scores = [
        math.comb(sum(1 for r in rewards if r >= 1.0), k) / math.comb(len(rewards), k)
        for rewards in by_task.values()
        if len(rewards) >= k
    ]
    return statistics.fmean(scores) if scores else None


def summarise(records: list[Record], runs: int) -> dict[str, Any]:
    means = run_means(records)
    graded = [r for r in records if r.reward is not None]
    calls = [r.call_s for r in graded if r.call_s is not None]
    return {
        "runs": [round(m * 100, 2) for m in means],
        "mean": statistics.fmean(means) * 100 if means else None,
        "sem": statistics.stdev(means) / math.sqrt(len(means)) * 100
        if len(means) > 1
        else None,
        "pass_hat_k": {k: pass_hat_k(records, k) for k in range(1, runs + 1)},
        "graded": len(graded),
        "errored": len(records) - len(graded),
        "retry_rate": statistics.fmean(1.0 if r.retries else 0.0 for r in graded)
        if graded
        else None,
        "terminations": dict(Counter(r.termination for r in graded)),
        "call_s_median": statistics.median(calls) if calls else None,
    }


def _pct(summary: dict[str, Any]) -> str:
    if summary["mean"] is None:
        return "n/a"
    sem = summary["sem"]
    return (
        f"{summary['mean']:.2f}% +/- {sem:.2f}%"
        if sem is not None
        else f"{summary['mean']:.2f}%"
    )


def compare(
    harbor: list[Record], original: list[Record], runs: int
) -> dict[str, dict[str, Any]]:
    """Per domain, plus ``overall``, each side summarised."""
    domains = sorted({r.domain for r in harbor} | {r.domain for r in original})
    report = {}
    for domain in [*domains, "overall"]:

        def pick(records: list[Record], d: str = domain) -> list[Record]:
            return records if d == "overall" else [r for r in records if r.domain == d]

        report[domain] = {
            "harbor": summarise(pick(harbor), runs),
            "original": summarise(pick(original), runs),
        }
    return report


def parity_experiment(
    report: dict[str, dict[str, Any]],
    *,
    model: str,
    agent: str,
    runs: int,
    tasks: int,
    adapted_size: int = 278,
    notes: str = "",
) -> list[dict[str, Any]]:
    """harbor's ``parity_experiment.json`` entry."""
    return [
        {
            "adapter_name": "tau-voice",
            "agent": agent,
            "model": model,
            "date": date.today().isoformat(),
            "adapted_benchmark_size": adapted_size,
            "parity_benchmark_size": tasks,
            "number_of_runs": runs,
            "notes": notes,
            "original_parity_repo": "https://github.com/sierra-research/tau2-bench",
            "metrics": [
                {
                    "benchmark_name": f"tau-voice/{domain}",
                    "metric": "Average Reward",
                    "original": _pct(sides["original"]),
                    "harbor": _pct(sides["harbor"]),
                    "original_runs": sides["original"]["runs"],
                    "harbor_runs": sides["harbor"]["runs"],
                }
                for domain, sides in report.items()
            ],
        }
    ]


def table(report: dict[str, dict[str, Any]], runs: int) -> str:
    lines = [
        "| Domain | Side | Average reward | Runs | "
        + " | ".join(f"Pass^{k}" for k in range(1, runs + 1))
        + " | Graded | Errored | Retried | Median call (s) |",
        "|---|---|---|---|" + "---|" * runs + "---|---|---|---|",
    ]
    for domain, sides in report.items():
        for side in ("original", "harbor"):
            s = sides[side]
            passes = " | ".join(
                "n/a" if s["pass_hat_k"][k] is None else f"{s['pass_hat_k'][k]:.3f}"
                for k in range(1, runs + 1)
            )
            retried = "n/a" if s["retry_rate"] is None else f"{s['retry_rate']:.0%}"
            call = "n/a" if s["call_s_median"] is None else f"{s['call_s_median']:.0f}"
            lines.append(
                f"| {domain} | {side} | {_pct(s)} | {s['runs']} | {passes} | "
                f"{s['graded']} | {s['errored']} | {retried} | {call} |"
            )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="tau-voice compare", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("--harbor", type=Path, nargs="+", required=True)
    parser.add_argument("--original", type=Path, nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="gpt-realtime-1.5")
    parser.add_argument("--agent", default="harbor-voice@0.1.0")
    parser.add_argument("--notes", default="")
    parser.add_argument("--out", type=Path, help="write parity_experiment.json here")
    args = parser.parse_args(argv)

    # A harbor job folder has job.log; its trial folders have trial.log.
    harbor = harbor_records(_found(args.harbor, "job.log"))
    original = original_records(_found(args.original, "results.json"))
    report = compare(harbor, original, args.runs)
    print(table(report, args.runs))
    if args.out:
        tasks = len(
            {(r.domain, r.task_id) for r in original}
            | {(r.domain, r.task_id) for r in harbor}
        )
        entry = parity_experiment(
            report,
            model=args.model,
            agent=args.agent,
            runs=args.runs,
            tasks=tasks,
            notes=args.notes,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        (args.out.with_suffix(".report.json")).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"wrote {args.out}")
