"""tau-voice verifier — grade a call from the sidecar's state log.

Separate from the sidecar, as harbor separates agent run and verifier, so a
call can be re-graded without being re-run. Runs in the same tau2 image.

Writes ``reward.txt`` (the tau2 reward) and ``result.json`` (reward info,
plus the hallucination check harbor-voice uses to decide on a rerun, exactly
as tau2's ``run_unit`` does).
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.simulation import VoiceRunConfig
from tau2.data_model.simulation import AudioNativeConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.evaluator.evaluator import evaluate_simulation
from tau2.evaluator.reviewer import check_hallucination
from tau2.evaluator.reviewer import format_hallucination_feedback
from tau2.evaluator import auth_classifier
from tau2.evaluator import evaluator_nl_assertions
from tau2.evaluator import hallucination_reviewer
from tau2.orchestrator.modes import CommunicationMode
from tau2.runner.build import _build_env_kwargs
from tau2.runner.helpers import get_tasks

GRADER_LLMS = {
    hallucination_reviewer: "grader.hallucination_review",
    evaluator_nl_assertions: "grader.nl_assertions",
    auth_classifier: "grader.auth_classifier",
}
"""tau2's modules that call an LLM while grading, and what to call their usage."""


@contextlib.contextmanager
def metered() -> Iterator[list[dict[str, Any]]]:
    """Every LLM call the grader makes, as harbor-voice usage records.

    tau2 keeps a dollar cost at most; the counts are read off each call's
    message (``usage``, with cached tokens when the sidecar adds them).
    The list fills when the block exits.
    """
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    saved = {module: module.generate for module in GRADER_LLMS}

    def counted(module: Any, real: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            message = real(*args, **kwargs)
            model = kwargs.get("model", args[0] if args else None)
            calls.append(
                (GRADER_LLMS[module], model, getattr(message, "usage", None) or {})
            )
            return message

        return call

    for module, real in saved.items():
        module.generate = counted(module, real)
    records: list[dict[str, Any]] = []
    try:
        yield records
    finally:
        for module, real in saved.items():
            module.generate = real
        merged: dict[tuple[str, Any], dict[str, Any]] = {}
        for component, model, usage in calls:
            record = merged.setdefault(
                (component, model),
                {"component": component, "model": model, "requests": 0,
                 "input_tokens": 0, "output_tokens": 0},
            )  # fmt: skip
            record["requests"] += 1
            record["input_tokens"] += int(usage.get("prompt_tokens") or 0)
            record["output_tokens"] += int(usage.get("completion_tokens") or 0)
            if usage.get("cached_tokens") is not None:
                record["cached_input_tokens"] = record.get(
                    "cached_input_tokens", 0
                ) + int(usage["cached_tokens"])
        records.extend(merged.values())


def grade(
    state: dict[str, Any],
    *,
    review_model: str | None = None,
    hallucination_check: bool = True,
) -> dict[str, Any]:
    """Grade one call from the sidecar's state; the shared core of CLI and /grade."""
    opened = state["open"]
    simulation = SimulationRun.model_validate(state["simulation"])
    task = get_tasks(opened["domain"], task_ids=[opened["task_id"]])[0]
    config = VoiceRunConfig(
        domain=opened["domain"],
        audio_native_config=AudioNativeConfig(),
        llm_user=opened["user_llm"],
        speech_complexity=opened["speech_complexity"],
        seed=opened["seed"],
    )
    env_kwargs = _build_env_kwargs(config, task) or None

    with metered() as usage:
        # The same call tau2's run_simulation makes for a full-duplex run.
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=opened["domain"],
            mode=CommunicationMode.FULL_DUPLEX,
            env_kwargs=env_kwargs,
        )
        simulation.reward_info = reward_info

        hallucination = None
        if hallucination_check:
            check = check_hallucination(
                simulation, task, review_model=review_model or config.review_model
            )
            hallucination = {
                "found": check.hallucination_found,
                "errors": len(check.errors),
                "feedback": format_hallucination_feedback(check),
                "detail": check.model_dump(mode="json"),
            }
    return {
        "task_id": task.id,
        "domain": opened["domain"],
        "termination": simulation.termination_reason.value,
        "reward": reward_info.reward,
        "reward_info": reward_info.model_dump(mode="json"),
        "hallucination": hallucination,
        # The grader's own LLM calls, for the attempt's usage record.
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--review-model",
        default=None,
        help="hallucination reviewer; defaults to tau2's (VoiceRunConfig.review_model)",
    )
    parser.add_argument("--skip-hallucination-check", action="store_true")
    args = parser.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    result = grade(
        state,
        review_model=args.review_model,
        hallucination_check=not args.skip_hallucination_check,
    )
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(f"{result['reward']}\n", encoding="utf-8")
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
