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
        return {
            "termination": "user_stop",
            "call_s": 10.0,
            "usage": [
                {
                    "component": "agent",
                    "model": "m",
                    "requests": 2,
                    "input_tokens": 100,
                },
                {"component": "caller.tts", "characters": 50},
            ],
        }

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


async def test_every_attempts_tokens_count(tmp_path: Path) -> None:
    """A set-aside attempt consumed tokens too."""
    _, attempt, grade = _fakes([True, False])
    trial = await run_trial(grade=grade, out_dir=tmp_path, attempt=attempt, task_id="1")
    assert trial["usage"] == [
        {"component": "agent", "model": "m", "requests": 4, "input_tokens": 200},
        {"component": "caller.tts", "characters": 100},
    ]


async def test_grading_usage_joins_the_attempts_record(tmp_path: Path) -> None:
    """The grader's own LLM calls (the hallucination review) count too: in the
    attempt's summary, its trajectory and the trial's total."""
    import json

    from harbor.models.trajectories.trajectory import Trajectory

    from harbor_voice.trajectory.model import UsageRecord
    from harbor_voice.trajectory.model import VoiceTrajectory

    review = {
        "component": "grader.hallucination_review",
        "model": "bedrock/claude",
        "requests": 1,
        "input_tokens": 5000,
        "cached_input_tokens": 0,
        "output_tokens": 300,
    }

    async def attempt(*, out_dir: Path, **kw):
        out_dir.mkdir(parents=True)
        summary = {"termination": "user_stop", "call_s": 10.0, "usage": [
            {"component": "agent", "model": "m", "requests": 1, "input_tokens": 10}
        ]}  # fmt: skip
        (out_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        voice = VoiceTrajectory(usage=(UsageRecord(component="agent", model="m"),))
        (out_dir / "trajectory.json").write_text(
            voice.to_atif().model_dump_json(), encoding="utf-8"
        )
        return summary

    async def grade(attempt_dir: Path):
        return {"reward": 1.0, "hallucination": {"found": False}, "usage": [review]}

    trial = await run_trial(grade=grade, out_dir=tmp_path, attempt=attempt, task_id="1")
    assert review in trial["usage"]
    saved = json.loads((tmp_path / "attempt-0" / "summary.json").read_text())
    assert saved["usage"][-1] == review
    atif = Trajectory.model_validate_json(
        (tmp_path / "attempt-0" / "trajectory.json").read_text(encoding="utf-8")
    )
    components = [u["component"] for u in atif.final_metrics.extra["usage"]]
    assert components == ["agent", "grader.hallucination_review"]
    assert VoiceTrajectory.from_atif(atif).usage[-1].model == "bedrock/claude"
