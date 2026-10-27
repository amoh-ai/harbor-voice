"""The job config: tau2's trials and averaging, expressed in harbor's job runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from harbor.models.job.config import JobConfig

from tau_voice.job import METRIC
from tau_voice.job import job_config


def test_one_agent_entry_per_tau2_seed(tmp_path: Path) -> None:
    raw = job_config(
        dataset=tmp_path, model="openai.realtime/gpt-realtime-1.5", trials=3
    )
    config = JobConfig.model_validate(raw)

    assert config.n_attempts == 1
    assert [agent.kwargs["seed"] for agent in config.agents] == [
        626729,
        373753,
        361454,
    ]
    assert {agent.model_name for agent in config.agents} == {
        "openai.realtime/gpt-realtime-1.5"
    }
    assert config.retry.max_retries == 3
    assert config.retry.include_exceptions == {"VoiceInfrastructureError"}


def test_the_backend_is_named_and_nothing_account_specific(tmp_path: Path) -> None:
    raw = job_config(dataset=tmp_path, model="m", environment="daytona")
    assert raw["environment"] == {"type": "daytona"}
    assert JobConfig.model_validate(raw).environment.type == "daytona"
    assert "environment" not in job_config(dataset=tmp_path, model="m")


def test_the_mean_leaves_errored_trials_out(tmp_path: Path) -> None:
    rewards = tmp_path / "rewards.jsonl"
    rewards.write_text('{"reward": 1.0}\nnull\n{"reward": 0.0}\n{"reward": 1.0}\n')
    out = tmp_path / "metric.json"
    subprocess.run(
        [sys.executable, str(METRIC), "-i", str(rewards), "-o", str(out)], check=True
    )
    assert json.loads(out.read_text()) == {
        "tau2_mean": 2 / 3,
        "graded": 3,
        "errored": 1,
    }
