"""The converter writes a harbor task that harbor-voice accepts as-is."""

from __future__ import annotations

from pathlib import Path

import yaml

from harbor_voice.trial import VoiceTaskConfig
from tau_voice import images
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
    # Built from the task by default, as harbor tasks are; one shared tag.
    assert compose["services"]["tau2"]["build"] == "./sidecar"
    assert compose["services"]["tau2"]["image"] == images.tau2_image()
    assert config.environment.docker_image is None
    for name in images.SIDECAR_FILES:
        assert (task / "environment" / "sidecar" / name).exists()
    assert (task / "tests" / "test.sh").read_bytes() == (
        images.SIDECAR / "test.sh"
    ).read_bytes()


def test_a_prebuilt_task_names_its_images(tmp_path: Path) -> None:
    task = write_task(
        tmp_path, "airline", "7", complexity="control", seed=626729, prebuilt=True
    )

    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    compose = yaml.safe_load(
        (task / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
    )
    assert "build" not in compose["services"]["tau2"]
    assert compose["services"]["tau2"]["image"] == images.tau2_image()
    assert config.environment is not None
    assert config.environment.docker_image == images.main_image()
    assert not (task / "environment" / "sidecar").exists()


def test_image_tags_follow_their_contents(monkeypatch) -> None:
    before = images.main_image()
    monkeypatch.setattr(images, "MAIN_DOCKERFILE", images.MAIN_DOCKERFILE + "# x")
    assert images.main_image() != before
    assert images.tau2_image().startswith("tau-voice-runtime:2174a60-")


def test_trial_seeds_match_tau2() -> None:
    assert trial_seeds(300, 3) == [626729, 373753, 361454]
