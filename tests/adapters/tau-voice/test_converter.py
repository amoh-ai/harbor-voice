"""The converter writes a harbor task that harbor-voice accepts as-is."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harbor_voice.trial import VoiceTaskConfig
from tau_voice import images
from tau_voice.main import TRIAL_TIMEOUT_S
from tau_voice.main import trial_seeds
from tau_voice.main import write_task


def test_a_task_is_harbor_shaped_and_agent_free(tmp_path: Path) -> None:
    task = write_task(tmp_path, "airline", "7", complexity="control", seed=626729)

    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    assert config.agent is None  # the run picks it
    assert config.call is not None and config.call.driver == "tau_voice.trial:drive"
    assert config.call.config["seed"] == 626729
    assert config.verifier is not None and config.verifier.timeout_sec == 900
    assert config.environment is not None
    # Templates only: no value is written into a task.
    assert config.environment.env["OPENAI_API_KEY"] == "${OPENAI_API_KEY}"

    compose = yaml.safe_load(
        (task / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
    )
    assert compose["services"]["main"]["depends_on"]["tau2"]["condition"] == (
        "service_healthy"
    )
    # main names its own env: harbor's EC2 does not pass the task's to main.
    assert compose["services"]["main"]["environment"] == [
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "HARBOR_VOICE_TTS_CACHE",
    ]
    # Built from the task by default, as harbor tasks are; one shared tag.
    assert compose["services"]["tau2"]["build"] == "./sidecar"
    assert compose["services"]["tau2"]["image"] == images.tau2_image()
    assert config.environment.docker_image is None
    for name in images.SIDECAR_FILES:
        assert (task / "environment" / "sidecar" / name).exists()
    test_sh = (task / "tests" / "test.sh").read_bytes()
    assert b"http://tau2:8000/grade" in test_sh and b"\r\n" not in test_sh


def test_a_task_is_also_a_harbor_task(tmp_path: Path) -> None:
    from harbor.models.task.config import TaskConfig

    from harbor_voice.harbor_agent import call_spec

    task = write_task(tmp_path, "airline", "7", complexity="control", seed=626729)
    harbor = TaskConfig.model_validate_toml(
        (task / "task.toml").read_text(encoding="utf-8")
    )
    assert harbor.task is not None and harbor.task.name == f"tau-voice/{task.name}"
    # Room for every hallucination retry, each call run to tau2's limit.
    assert harbor.agent.timeout_sec == TRIAL_TIMEOUT_S >= 4 * 1200
    assert harbor.environment.env["ELEVENLABS_API_KEY"] == "${ELEVENLABS_API_KEY}"
    # Harbor hands the agent only the instruction; it carries the call spec.
    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    assert (
        call_spec(instruction)
        == (task / "voice.toml").read_text(encoding="utf-8").rstrip()
    )


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


def test_a_registry_names_where_trials_pull_from(tmp_path: Path) -> None:
    task = write_task(
        tmp_path,
        "airline",
        "7",
        complexity="regular",
        seed=626729,
        registry="ghcr.io/org/",
    )
    compose = yaml.safe_load(
        (task / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
    )
    tau2 = compose["services"]["tau2"]
    assert "build" not in tau2  # a registry means prebuilt
    assert tau2["image"] == images.tau2_image("ghcr.io/org")
    assert tau2["image"].startswith("ghcr.io/org/tau-voice-runtime:2174a60-")
    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    assert config.environment is not None
    assert config.environment.docker_image == "ghcr.io/org/" + images.main_image()


def test_the_callers_model_is_tau2s_unless_named(tmp_path: Path) -> None:
    default = write_task(tmp_path / "a", "airline", "7", complexity="control", seed=1)
    assert VoiceTaskConfig.from_toml(default / "voice.toml").call is not None
    assert "user_llm" not in (default / "voice.toml").read_text(encoding="utf-8")
    named = write_task(
        tmp_path / "b", "airline", "7", complexity="control", seed=1,
        user_llm="gpt-4.1-mini-2025-04-14",
    )  # fmt: skip
    spec = (named / "voice.toml").read_text(encoding="utf-8")
    assert 'user_llm = "gpt-4.1-mini-2025-04-14"' in spec
    assert VoiceTaskConfig.from_toml(named / "voice.toml").call is not None


def test_image_tags_follow_their_contents(monkeypatch) -> None:
    before = images.main_image()
    monkeypatch.setattr(images, "MAIN_DOCKERFILE", images.MAIN_DOCKERFILE + "# x")
    assert images.main_image() != before
    assert images.tau2_image().startswith("tau-voice-runtime:2174a60-")


def test_trial_seeds_match_tau2() -> None:
    assert trial_seeds(300, 3) == [626729, 373753, 361454]


def test_test_sh_is_lf_even_from_a_crlf_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tau_voice.main as converter

    template = tmp_path / "template"
    (template / "tests").mkdir(parents=True)
    source = (converter.TEMPLATE / "tests" / "test.sh").read_bytes()
    (template / "tests" / "test.sh").write_bytes(source.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(converter, "TEMPLATE", template)

    task = write_task(tmp_path / "out", "airline", "7", complexity="control", seed=1)
    written = (task / "tests" / "test.sh").read_bytes()
    assert b"\r\n" not in written and written.startswith(b"#!/bin/bash\n")


def test_every_compose_variable_reaches_a_remote_backend(tmp_path: Path) -> None:
    """Remote backends run compose with the task's env only, not this host's."""
    import re

    from harbor.models.task.config import TaskConfig

    task = write_task(tmp_path, "airline", "7", complexity="control", seed=1)
    compose = (task / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
    passed = set(re.findall(r"^\s+- ([A-Z0-9_]+)$", compose, re.MULTILINE))
    templated = set(re.findall(r"\$\{([A-Z0-9_]+)", compose))
    host_only = {"HARBOR_VOICE_CACHE_DIR", "HOME"}  # local cache mounts; optional
    declared = set(
        TaskConfig.model_validate_toml(
            (task / "task.toml").read_text(encoding="utf-8")
        ).environment.env
    )
    assert passed | (templated - host_only) <= declared
