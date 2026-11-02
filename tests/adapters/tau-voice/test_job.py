"""The job config: tau2's trials and averaging, expressed in harbor's job runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from harbor.models.job.config import JobConfig

from tau_voice.job import METRIC
from tau_voice.job import RETRY_ON
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
    assert config.retry.include_exceptions == set(RETRY_ON)
    # Environment failures before the call retry too (parity-40 lost a trial
    # to a dropped clone inside the image build).
    assert "RuntimeError" in RETRY_ON


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


def test_an_agents_settings_travel_as_its_agent_json(tmp_path: Path) -> None:
    raw = job_config(
        dataset=tmp_path,
        model="openai.live/gpt-live-1",
        agent_config={"backend_model": "gpt-6-luna"},
    )
    config = JobConfig.model_validate(raw)
    (agent,) = config.agents
    assert agent.model_name == "openai.live/gpt-live-1"
    spec = json.loads(agent.kwargs["agent_json"])
    assert spec["s2s"] == "openai.live/gpt-live-1"
    assert spec["config"] == {"s2s": {"backend_model": "gpt-6-luna"}}
    plain = job_config(dataset=tmp_path, model="m")
    assert "agent_json" not in plain["agents"][0]["kwargs"]


def test_a_hub_dataset_is_named_as_this_checkouts_registry_pins_it(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    from tau_voice import job

    registry = tmp_path / "registry.json"
    entry = {"name": "tau-voice-control", "description": "", "tasks": []}
    versions = [{**entry, "version": "0.1.0"}, {**entry, "version": "0.2.0"}]
    registry.write_text(json.dumps(versions), encoding="utf-8")
    monkeypatch.setattr(job, "REGISTRY", registry)

    raw = job_config(hub="tau-voice-control@0.1.0", model="m", task_names=["airline-*"])
    (dataset,) = JobConfig.model_validate(raw).datasets
    assert dataset.is_registry() and dataset.registry_path == registry
    assert (dataset.name, dataset.version) == ("tau-voice-control", "0.1.0")
    assert dataset.task_names == ["airline-*"]
    # No version: harbor runs the latest.
    assert "version" not in job.hub_dataset("tau-voice-control")
    with pytest.raises(ValueError, match="no version '9'"):
        job.hub_dataset("tau-voice-control@9")
    with pytest.raises(ValueError, match="not in registry.json"):
        job.hub_dataset("eva@0.1.0")
    with pytest.raises(ValueError, match="name one of"):
        job_config(dataset=tmp_path, hub="tau-voice-control", model="m")
