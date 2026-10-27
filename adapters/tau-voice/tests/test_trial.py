"""run_trial: tau2's hallucination retry, without a sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest

from tau_voice.run import run_trial

pytestmark = pytest.mark.unit


def _fakes(found: list[bool]):
    runs: list[dict] = []
    verdicts = iter(found)

    async def attempt(**kw):
        runs.append(kw)
        return {"termination": "user_stop", "call_s": 10.0}

    async def grade(attempt_dir: Path):
        hit = next(verdicts)
        return {
            "reward": 0.0 if hit else 1.0,
            "reward_info": {},
            "hallucination": {
                "found": hit,
                "feedback": f"fix #{len(runs)}" if hit else None,
            },
        }

    return runs, attempt, grade


async def test_retries_with_feedback_and_shifted_seeds(tmp_path: Path) -> None:
    runs, attempt, grade = _fakes([True, True, False])
    trial = await run_trial(
        grade=grade,
        out_dir=tmp_path,
        seed=42,
        attempt=attempt,
        domain="airline",
        task_id="1",
    )
    assert [r["seed"] for r in runs] == [42, 1042, 2042]
    assert [r["hallucination_feedback"] for r in runs] == [None, "fix #1", "fix #2"]
    assert trial["reward"] == 1.0
    assert trial["final_attempt"] == 2 and trial["hallucination_retries_used"] == 2
    assert trial["seed"] == 42
    assert (tmp_path / "trial.json").exists()


async def test_the_last_attempt_counts_when_retries_run_out(tmp_path: Path) -> None:
    runs, attempt, grade = _fakes([True] * 4)
    trial = await run_trial(grade=grade, out_dir=tmp_path, attempt=attempt, task_id="1")
    assert len(runs) == 4  # one attempt plus tau2's three retries
    assert trial["reward"] == 0.0 and trial["final_attempt"] == 3


async def test_retries_can_be_switched_off(tmp_path: Path) -> None:
    runs, attempt, grade = _fakes([True])
    await run_trial(
        grade=grade,
        out_dir=tmp_path,
        attempt=attempt,
        hallucination_retries=0,
        task_id="1",
    )
    assert len(runs) == 1
