"""Parity: the harbor side against tau2's own run, in harbor's format.

Reads harbor job folders (``harbor-voice jobs start``) and tau2 result folders
(tau2's ``--save-to`` folders, :func:`tau_voice.original.tau2_command`), and
reports, per domain and overall:

- **average reward** (Pass^1): the mean over graded tasks in each run, then
  mean ± sample SEM over runs, as harbor's ``parity_experiment.json`` states it;
- **Pass^k** for k = 1..runs, tau2's estimator: per task, C(c, k) / C(n, k)
  with c successes in n runs, averaged over tasks;
- diagnostics that explain a gap without being a parity claim: hallucination
  retries, termination reasons, call length, errored trials;
- with ``--metrics``, tau2's interaction metrics for both sides (L_R, L_Y,
  R_R, R_Y, I_A, S_BC, S_VT, S_ND), computed by tau2's code
  (``sidecar/interaction_metrics.py``, run in the tau2 image);
- tokens each side consumed, every attempt included (set-aside hallucination
  reruns too): the agent's and the caller LLM's. Counts, not dollars.

One speech complexity at a time: ``--complexity`` keeps that preset's
records when the folders hold several. Errored trials (no grade) are left
out of the averages and counted, as tau2 leaves infrastructure errors out::

    tau-voice compare --harbor jobs/parity-regular --original tau2-saves \\
        --complexity regular --metrics metrics-regular.json \\
        --out parity_experiment-regular.json
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
    complexity: str | None = None


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
                    complexity=summary.get("speech_complexity"),
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
                    complexity=(sim.get("speech_environment") or {}).get("complexity"),
                )
            )
    return records


TOKENS = (
    "agent_input",
    "agent_cached",
    "agent_output",
    "agent_audio_in",
    "agent_audio_out",
    "caller_input",
    "caller_output",
)
"""Token counts compared per side: the agent under test, the caller's LLM."""


def harbor_tokens(job_dirs: Iterable[Path], complexity: str | None) -> dict[str, int]:
    """The harbor trials' tokens, every attempt (``usage`` of each trial)."""
    total = dict.fromkeys(TOKENS, 0)
    for job in job_dirs:
        for outcome in sorted(job.glob("*/agent/call/outcome.json")):
            summary = _json(outcome).get("summary") or {}
            if complexity and summary.get("speech_complexity") != complexity:
                continue
            for u in summary.get("usage") or []:
                # Every part of the agent: OpenAI Live's tokens are its
                # backend's (``agent.backend``).
                component = str(u.get("component") or "")
                if component == "agent" or component.startswith("agent."):
                    total["agent_input"] += u.get("input_tokens") or 0
                    total["agent_cached"] += u.get("cached_input_tokens") or 0
                    total["agent_output"] += u.get("output_tokens") or 0
                    total["agent_audio_in"] += u.get("input_audio_tokens") or 0
                    total["agent_audio_out"] += u.get("output_audio_tokens") or 0
                elif u.get("component") == "caller.llm":
                    total["caller_input"] += u.get("input_tokens") or 0
                    total["caller_output"] += u.get("output_tokens") or 0
    return total


def original_tokens(
    result_dirs: Iterable[Path], complexity: str | None
) -> dict[str, int]:
    """tau2's tokens, every attempt (its simulations and its set-aside reruns):
    the agent's from ``agent_usage``, the caller's from its messages."""
    total = dict.fromkeys(TOKENS, 0)
    for results_dir in result_dirs:
        sims = list(_simulations(results_dir))
        for path in sorted((results_dir / "hallucination_discarded").glob("*.json")):
            data = _json(path)
            sims.extend(data.get("simulations") or [data])
        for sim in sims:
            env = sim.get("speech_environment") or {}
            if complexity and env.get("complexity") != complexity:
                continue
            for r in (sim.get("agent_usage") or {}).get("records") or []:
                total["agent_input"] += r.get("input_tokens") or 0
                total["agent_cached"] += r.get("input_cached_tokens") or 0
                total["agent_output"] += r.get("output_tokens") or 0
                total["agent_audio_in"] += r.get("input_audio_tokens") or 0
                total["agent_audio_out"] += r.get("output_audio_tokens") or 0
            for tick in sim.get("ticks") or []:
                usage = (tick.get("user_chunk") or {}).get("usage") or {}
                total["caller_input"] += usage.get("prompt_tokens") or 0
                total["caller_output"] += usage.get("completion_tokens") or 0
    return total


def harbor_model(job_dirs: Iterable[Path]) -> str | None:
    """The agent under test, as the harbor jobs recorded it (``config.json``):
    ``openai.live/gpt-live-1 (backend gpt-6-luna)``.

    Read from the run rather than given on the command line, which once
    labelled a Live run with the flag's ``gpt-realtime-1.5`` default. ``None``
    when no job records one; an error when the jobs disagree, since one
    parity entry names one model.
    """
    models: set[str] = set()
    for job in job_dirs:
        config = job / "config.json"
        if not config.exists():
            continue
        for agent in _json(config).get("agents") or []:
            name = agent.get("model_name")
            if not name:
                continue
            spec = json.loads((agent.get("kwargs") or {}).get("agent_json") or "{}")
            backend = ((spec.get("config") or {}).get("s2s") or {}).get("backend_model")
            models.add(f"{name} (backend {backend})" if backend else name)
    if len(models) > 1:
        raise SystemExit(
            f"the harbor jobs ran different agents ({', '.join(sorted(models))}); "
            "compare one at a time, or name it with --model"
        )
    return next(iter(models), None)


def token_table(tokens: dict[str, dict[str, int]]) -> str:
    lines = [
        "| Side | " + " | ".join(TOKENS) + " |",
        "|---|" + "---|" * len(TOKENS),
    ]
    for side in ("original", "harbor"):
        lines.append(
            f"| {side} | " + " | ".join(f"{tokens[side][k]:,}" for k in TOKENS) + " |"
        )
    return "\n".join(lines)


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


INTERACTION = {
    "response_latency_mean": "L_R (s)",
    "yield_latency_mean": "L_Y (s)",
    "response_rate": "R_R",
    "yield_rate": "R_Y",
    "agent_interruption_rate": "I_A",
    "selectivity_backchannel": "S_BC",
    "selectivity_vocal_tic": "S_VT",
    "selectivity_non_directed": "S_ND",
}
"""tau2's panel fields and the paper's names for them."""


def interaction_table(metrics: dict[str, Any]) -> str:
    """tau2's interaction panel, both sides, per domain and overall."""
    scopes = sorted(
        set(metrics["original"]["domains"]) | set(metrics["harbor"]["domains"])
    )
    lines = [
        "| Domain | Side | " + " | ".join(INTERACTION.values()) + " |",
        "|---|---|" + "---|" * len(INTERACTION),
    ]
    for scope in [*scopes, "overall"]:
        for side in ("original", "harbor"):
            panel = metrics[side]
            values = (
                panel["overall"] if scope == "overall" else panel["domains"].get(scope)
            )
            cells = [
                "n/a" if not values or values.get(key) is None else f"{values[key]:.3f}"
                for key in INTERACTION
            ]
            lines.append(f"| {scope} | {side} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


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
    parser.add_argument(
        "--complexity", help="only this speech complexity's records (e.g. regular)"
    )
    parser.add_argument(
        "--metrics", type=Path, help="interaction_metrics.py output to report too"
    )
    parser.add_argument(
        "--model", help="the agent under test (default: what the harbor jobs ran)"
    )
    parser.add_argument("--agent", default="harbor-voice@0.1.0")
    parser.add_argument("--notes", default="")
    parser.add_argument("--out", type=Path, help="write parity_experiment.json here")
    args = parser.parse_args(argv)

    # A harbor job folder has job.log; its trial folders have trial.log.
    jobs = _found(args.harbor, "job.log")
    harbor = harbor_records(jobs)
    original = original_records(_found(args.original, "results.json"))
    if args.complexity:
        harbor = [r for r in harbor if r.complexity == args.complexity]
        original = [r for r in original if r.complexity == args.complexity]
    report = compare(harbor, original, args.runs)
    print(table(report, args.runs))
    tokens = {
        "harbor": harbor_tokens(jobs, args.complexity),
        "original": original_tokens(
            _found(args.original, "results.json"), args.complexity
        ),
    }
    print()
    print(token_table(tokens))
    interaction = _json(args.metrics) if args.metrics else None
    if interaction is not None:
        print()
        print(interaction_table(interaction))
    if args.out:
        model = args.model or harbor_model(jobs)
        if model is None:
            raise SystemExit("no harbor job records its agent; name it with --model")
        tasks = len(
            {(r.domain, r.task_id) for r in original}
            | {(r.domain, r.task_id) for r in harbor}
        )
        entry = parity_experiment(
            report,
            model=model,
            agent=args.agent,
            runs=args.runs,
            tasks=tasks,
            notes=args.notes,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        report["tokens"] = tokens
        if interaction is not None:
            report["interaction_metrics"] = interaction
        (args.out.with_suffix(".report.json")).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"wrote {args.out}")
