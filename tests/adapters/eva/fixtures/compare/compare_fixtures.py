"""Synthetic outputs of both sides, in their real layouts.

A harbor job folder as harbor writes it (``<trial>/agent/call/eva/``, the
driver's summary, metrics and EVA run directory) and EVA run folders as EVA
writes them (``records/<id>/trial_<k>/``, ``_failed_attempt_<n>`` archives,
``evaluation_summary.json``). Nothing is run.

:func:`sides` lays out two domains (airline ``1.2.2``, ``2.2.5``; itsm
``16``) at k = 3: harbor's job and EVA's run per domain, and a second EVA
run (the noise floor) on request. The plans below are what each trial came
to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eva_adapter.compare import COMPONENTS
from eva_adapter.conditions import PerturbationConfig
from eva_adapter.main import task_dir_name

ELEVEN = {
    "agent_id": "agent-f",
    "metadata": {
        "call_duration_secs": 60,
        "charging": {
            "llm_usage": {
                "irreversible_generation": {
                    "model_usage": {
                        "gpt-5.1": {
                            "input": {"tokens": 1000},
                            "output_total": {"tokens": 50},
                        }
                    }
                },
                "initiated_generation": {
                    "model_usage": {
                        "gpt-5.1": {
                            "input": {"tokens": 1200},
                            "output_total": {"tokens": 60},
                        }
                    }
                },
            }
        },
    },
}
EVENTS = [
    {"type": "user_speech", "data": {"text": "hello there", "is_final": True}},
    {"type": "assistant_speech", "data": {"text": "hi, how can I help"}},
    {"type": "user_speech", "data": {"text": "bye now please", "is_final": True}},
]
RECORDS = {"airline": ["1.2.2", "2.2.5"], "itsm": ["16"]}

HARBOR_AIRLINE = {
    ("1.2.2", 0): {"a": 1.0, "x": 1.0},
    ("1.2.2", 1): {"a": 1.0, "x": 0.0},
    ("1.2.2", 2): {"a": 0.0, "x": 0.0},
    ("2.2.5", 0): {"a": 0.0},
    ("2.2.5", 1): "errored",
    ("2.2.5", 2): {"validated": False, "archives": 3},
}
HARBOR_ITSM = {("16", 0): {"a": 1.0}, ("16", 1): {"a": 1.0}, ("16", 2): "missing"}
ORIGINAL_AIRLINE = {
    ("1.2.2", 0): {"a": 1.0},
    ("1.2.2", 1): {"a": 1.0, "archives": 1},
    ("1.2.2", 2): {"a": 1.0},
    ("2.2.5", 0): {"a": 0.0},
    ("2.2.5", 1): {"a": 0.0},
    ("2.2.5", 2): {
        "validated": False,
        "archives": 3,
        "failed_end": "inactivity_timeout",
    },
}
ORIGINAL_ITSM = {("16", 0): {"a": 1.0}, ("16", 1): {"a": 0.0}, ("16", 2): {"a": 1.0}}
PLANS = (
    ("airline", HARBOR_AIRLINE, ORIGINAL_AIRLINE),
    ("itsm", HARBOR_ITSM, ORIGINAL_ITSM),
)


def metrics(a: float, x: float, speeds: list[float]) -> dict[str, Any]:
    scores = {n: {"name": n, "score": 0.9, "normalized_score": 0.9} for n in COMPONENTS}
    scores["response_speed"] = {
        "name": "response_speed",
        "score": sum(speeds) / len(speeds),
        "details": {"per_turn_speeds": speeds},
    }
    return {
        "record_id": "r",
        "metrics": scores,
        "aggregate_metrics": {
            "EVA-A_pass": a,
            "EVA-X_pass": x,
            "EVA-overall_pass": float(a == 1.0 and x == 1.0),
        },
    }


def record(
    d: Path, *, end: str = "goodbye", call_s: float = 120.0, metrics: Any = None
) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(
        json.dumps({"conversation_ended_reason": end, "duration_seconds": call_s})
    )
    (d / "user_simulator_events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in EVENTS)
    )
    (d / "pipecat_metrics.jsonl").write_text(
        json.dumps(
            {
                "type": "LLMTokenUsageMetricsData",
                "value": {"prompt_tokens": 100, "completion_tokens": 10},
            }
        )
        + "\n"
        + json.dumps({"type": "LatencyMetric", "value": 0.5})
    )
    (d / "elevenlabs_conversation_details.json").write_text(json.dumps(ELEVEN))
    if metrics is not None:
        (d / "metrics.json").write_text(json.dumps(metrics))


def judge_usage(run_dir: Path) -> None:
    usage = run_dir / "judge_token_usage" / "judge_token_usage.json"
    usage.parent.mkdir(parents=True, exist_ok=True)
    usage.write_text(
        json.dumps(
            {
                "faithfulness": {
                    "total_input_tokens": 10,
                    "total_output_tokens": 2,
                    "num_calls": 1,
                }
            }
        )
    )


def eva_run(
    run_dir: Path,
    domain: str,
    plan: dict[tuple[str, int], dict[str, Any]],
    *,
    model: str = "gpt-realtime-1.5",
    perturbation: PerturbationConfig | None = None,
    metric_configs: dict[str, Any] | None = None,
) -> None:
    """An EVA run folder: graded trials, archived attempts, failures."""
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "domain": domain,
                "framework": "openai_realtime",
                "model": {"s2s": model},
                "num_trials": 3,
                # As EVA's RunConfig dumps it: the whole PerturbationConfig, or null.
                "perturbation": None
                if perturbation is None
                else {**perturbation.as_dict(), "accent": None, "behavior": None},
                "metric_configs": metric_configs or {},
            }
        )
    )
    passed, failed = [], []
    for (rid, k), spec in plan.items():
        for n in range(1, spec.get("archives", 0) + 1):
            record(
                run_dir / "records" / rid / f"trial_{k}_failed_attempt_{n}",
                end=spec.get("failed_end", "error"),
            )
        if spec.get("validated", True):
            m = metrics(spec["a"], spec.get("x", 1.0), [1.0, 1.3])
            record(run_dir / "records" / rid / f"trial_{k}", metrics=m)
            passed.append(f"{rid}/trial_{k}")
        else:
            failed.append(f"{rid}/trial_{k}")
    (run_dir / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "simulation": {
                    "successful_record_ids": passed,
                    "failed_record_ids": failed,
                }
            }
        )
    )
    judge_usage(run_dir)


def harbor_job(
    job: Path,
    domain: str,
    plan: dict[tuple[str, int], Any],
    perturbation: PerturbationConfig | None = None,
    metric_configs: dict[str, Any] | None = None,
) -> None:
    """A harbor job folder: driver summaries, metrics, EVA run dirs; an
    errored trial (no summary) and a trial never started (no folder)."""
    job.mkdir(parents=True)
    (job / "config.json").write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "model_name": "openai.realtime/gpt-realtime-1.5",
                        "kwargs": {"seed": 0},
                    }
                ]
            }
        )
    )
    (job / "job.log").write_text("")
    for i, ((rid, k), spec) in enumerate(plan.items()):
        if spec == "missing":
            continue
        task = task_dir_name(domain, rid, perturbation)
        trial = job / f"{task}__{i}"
        trial.mkdir()
        (trial / "trial.log").write_text("done")
        (trial / "config.json").write_text(
            json.dumps(
                {
                    "task": {"path": f"/work/datasets/x/{task}"},
                    "agent": {"kwargs": {"seed": k}},
                }
            )
        )
        if spec == "errored":
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": f"eva/{task}",
                        "verifier_result": None,
                        "exception_info": {
                            "exception_type": "VoiceInfrastructureError",
                            "exception_message": "the call did not run (1): \n│\n│ ProviderError: [throttled] quota exhausted │\n│",
                        },
                    }
                )
            )
            continue
        eva = trial / "agent" / "call" / "eva"
        validated = spec.get("validated", True)
        archives = spec.get("archives", 0)
        for n in range(1, archives + 1):
            record(
                eva / "run" / "records" / rid / f"trial_{k}_failed_attempt_{n}",
                end="inactivity_timeout",
            )
        m = metrics(spec.get("a", 0.0), spec.get("x", 1.0), [1.0, 1.2])
        if validated:
            record(eva / "run" / "records" / rid / f"trial_{k}", metrics=m)
            (eva / "metrics.json").write_text(json.dumps(m))
        (eva / "run" / "config.json").write_text(
            json.dumps({"domain": domain, "metric_configs": metric_configs or {}})
        )
        judge_usage(eva / "run")
        (eva / "summary.json").write_text(
            json.dumps(
                {
                    "domain": domain,
                    "record_id": rid,
                    "trial": k,
                    "output_id": f"{rid}/trial_{k}",
                    "validated": validated,
                    "attempts_used": archives + (1 if validated else 0),
                    "aggregates": m["aggregate_metrics"] if validated else None,
                    **(
                        {
                            "condition": perturbation.label,
                            "perturbation": perturbation.as_dict(),
                        }
                        if perturbation is not None
                        else {}
                    ),
                }
            )
        )
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "verifier_result": {
                        "rewards": {"reward": m["aggregate_metrics"]["EVA-A_pass"]}
                        if validated
                        else {}
                    },
                    "exception_info": None,
                }
            )
        )


def original_plan(side: str, domain: str) -> dict[tuple[str, int], Any]:
    """EVA's plan for a domain; ``original2`` differs from ``original`` once."""
    plan = dict(ORIGINAL_AIRLINE if domain == "airline" else ORIGINAL_ITSM)
    if side == "original2" and domain == "airline":
        plan[("1.2.2", 2)] = {"a": 0.0}  # the second run differs once
    return plan


def sides(
    root: Path,
    *,
    noise: bool = False,
    model: str = "gpt-realtime-1.5",
    perturbation: PerturbationConfig | None = None,
    metric_configs: dict[str, Any] | None = None,
) -> dict[str, list[Path]]:
    """Both sides' folders under ``root``, per side: a harbor job and an EVA
    run per domain (``<side>-<domain>``), the noise floor's too with ``noise``."""
    found: dict[str, list[Path]] = {"harbor": [], "original": []}
    for domain, harbor, _ in PLANS:
        job = root / f"harbor-{domain}"
        harbor_job(job, domain, harbor, perturbation, metric_configs)
        found["harbor"].append(job)
        for side in ("original", "original2") if noise else ("original",):
            run_dir = root / f"{side}-{domain}"
            eva_run(
                run_dir,
                domain,
                original_plan(side, domain),
                model=model,
                perturbation=perturbation,
                metric_configs=metric_configs,
            )
            found.setdefault(side, []).append(run_dir)
    return found
