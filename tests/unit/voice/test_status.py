"""harbor-voice status: a running job's state, read from harbor's job folder."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harbor_voice.status import render

pytestmark = pytest.mark.unit


def _task(root: Path, name: str, domain: str) -> Path:
    task = root / "tasks" / name
    task.mkdir(parents=True)
    (task / "task.toml").write_text(f'[metadata]\ndomain = "{domain}"\n')
    return task


def _trial(job: Path, name: str, task: Path, *, log: str, result: dict | None) -> None:
    trial = job / name
    trial.mkdir()
    (trial / "config.json").write_text(json.dumps({"task": {"path": str(task)}}))
    (trial / "trial.log").write_text(log)
    if result is not None:
        (trial / "result.json").write_text(json.dumps(result))


def test_a_running_job(tmp_path: Path) -> None:
    job = tmp_path / "parity"
    job.mkdir()
    (job / "config.json").write_text(json.dumps({"n_concurrent_trials": 2}))
    (job / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 4,
                "started_at": "2026-10-27T10:00:00",
                "stats": {"n_retries": 1},
            }
        )
    )
    airline, retail = _task(tmp_path, "a", "airline"), _task(tmp_path, "r", "retail")
    times = {"started_at": "2026-10-27T10:00:00", "finished_at": "2026-10-27T10:06:00"}
    _trial(
        job, "a-0", airline, log="call: done in 5.0 min\n",
        result={"verifier_result": {"rewards": {"reward": 1.0}}, **times},
    )  # fmt: skip
    _trial(
        job, "r-0", retail, log="x\n",
        result={
            "exception_info": {
                "exception_type": "VoiceInfrastructureError",
                "exception_message": "the call did not run (1): provider down\ntrace",
            },
            **times,
        },
    )  # fmt: skip
    _trial(
        job, "r-1", retail,
        log="install: harbor-voice 0.1.0 into main (14 s)\nattempt 0: started, seed 7\n",
        result=None,
    )  # fmt: skip
    started = (job / "r-1" / "config.json").stat().st_mtime
    os.utime(job / "r-1" / "config.json", (started - 180, started - 180))

    text = render(job, now=started)
    assert text.splitlines()[0].startswith(
        "parity: done 2/4 · running 1 · errored 1 · retries 1"
    )
    assert "ETA ~6m" in text  # 2 left x 6 min / 2 at a time
    assert "airline    reward 1.000 (1 graded)" in text
    assert "r-1" in text and "3m" in text and "attempt 0: started, seed 7" in text
    assert "VoiceInfrastructureError: the call did not run (1): provider down" in text
