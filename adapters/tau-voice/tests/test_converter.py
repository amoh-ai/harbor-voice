"""The converter writes a harbor task that harbor-voice accepts as-is."""

from __future__ import annotations

from pathlib import Path

import yaml

from harbor_voice.trial import VoiceTaskConfig
from tau_voice.main import trial_seeds
from tau_voice.main import write_task


def test_a_task_is_harbor_shaped_and_agent_free(tmp_path: Path) -> None:
    task = write_task(tmp_path, "airline", "7", complexity="control", seed=626729)

    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    assert config.agent is None  # the run picks it
    assert config.call is not None and config.call.driver == "tau_voice.trial:drive"
    assert config.call.config["seed"] == 626729
    assert config.verifier is not None and config.verifier.service == "tau2"
    assert config.environment is not None
    # Templates only: no value is written into a task.
    assert config.environment.env["OPENAI_API_KEY"] == "${OPENAI_API_KEY}"

    compose = yaml.safe_load(
        (task / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
    )
    assert compose["services"]["main"]["depends_on"]["tau2"]["condition"] == (
        "service_healthy"
    )
    assert compose["services"]["tau2"]["build"] == "./sidecar"
    assert (task / "environment" / "Dockerfile").exists()
    for name in ("Dockerfile", "server.py", "evaluate.py", "test.sh"):
        assert (task / "environment" / "sidecar" / name).exists()
    assert "COPY test.sh /tests/test.sh" in (
        task / "environment" / "sidecar" / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert (task / "tests" / "test.sh").read_bytes() == (
        task / "environment" / "sidecar" / "test.sh"
    ).read_bytes()


def test_trial_seeds_match_tau2() -> None:
    assert trial_seeds(300, 3) == [626729, 373753, 361454]
