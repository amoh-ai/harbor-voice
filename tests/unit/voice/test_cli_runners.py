"""harbor-voice jobs / trials: harbor's own commands, mounted, agent defaulted."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from harbor_voice import cli
from harbor_voice.cli import CALL_AGENT
from harbor_voice.cli import SKIP_MODEL_PARSING
from harbor_voice.cli import with_default_agent
from harbor_voice.cli import with_default_registry
from harbor_voice.harbor_agent import harbor_parses_model_names

pytestmark = pytest.mark.unit

# harbor 0.25+ resolves -m through litellm; our provider specs skip that.
SKIP = [SKIP_MODEL_PARSING] if harbor_parses_model_names() else []


@pytest.mark.parametrize("runner", ["jobs", "job", "trials", "trial"])
def test_start_without_an_agent_runs_the_call_agent(runner: str) -> None:
    argv = [runner, "start", "-p", "task", "-m", "openai.realtime/gpt-realtime-1.5"]
    assert with_default_agent(argv) == [
        runner,
        "start",
        "--agent",
        CALL_AGENT,
        *SKIP,
        "-p",
        "task",
        "-m",
        "openai.realtime/gpt-realtime-1.5",
    ]


@pytest.mark.parametrize(
    "chosen",
    [
        ["-a", "oracle"],
        ["--agent", "oracle"],
        ["--agent=oracle"],
        ["-c", "job.yaml"],
        ["--config=job.yaml"],
    ],
)
def test_an_explicit_agent_or_config_is_left_alone(chosen: list[str]) -> None:
    argv = ["jobs", "start", *chosen]
    assert with_default_agent(argv) == argv


def test_agent_options_are_not_mistaken_for_an_agent() -> None:
    argv = ["trials", "start", "--agent-kwarg", "seed=1", "--ak", "x=y"]
    assert with_default_agent(argv)[2:4] == ["--agent", CALL_AGENT]


def test_other_commands_pass_through() -> None:
    for argv in (["doctor", "task"], ["jobs", "resume", "-p", "x"], []):
        assert with_default_agent(argv) == argv


def test_harbors_commands_are_mounted() -> None:
    from harbor.cli.jobs import jobs_app
    from harbor.cli.trials import trials_app

    cli._mount_harbor()
    # Checked structurally: help text is rich-rendered, and in CI its colour
    # codes split option names.
    mounted = {
        group.name: group.typer_instance for group in cli.voice_app.registered_groups
    }
    assert mounted["jobs"] is jobs_app and mounted["trials"] is trials_app
    result = CliRunner().invoke(cli.voice_app, ["jobs", "start", "--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize("runner", ["jobs", "trials"])
def test_the_flags_we_add_are_harbors(runner: str) -> None:
    """The defaulted agent and the model-parsing flag go through harbor's own
    parser: an option harbor does not have fails before ``--help`` prints."""
    cli._mount_harbor()
    argv = with_default_agent([runner, "start", "-m", "openai.live/gpt-live-1"])
    result = CliRunner().invoke(cli.voice_app, [*argv, "--help"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "dataset", [["-d", "eva@0.1.0"], ["--dataset", "eva"], ["--dataset=eva"]]
)
def test_a_named_dataset_reads_harbor_voices_registry(
    dataset: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from harbor_voice import datasets

    argv = ["jobs", "start", *dataset, "-m", "openai.realtime/gpt-realtime-1.5"]
    monkeypatch.setattr(datasets, "checkout_registry", lambda: None)
    assert with_default_registry(argv) == [*argv, "--repo", datasets.REPO]
    monkeypatch.setattr(datasets, "checkout_registry", lambda: Path("registry.json"))
    assert with_default_registry(argv)[-2:] == ["--registry-path", "registry.json"]


@pytest.mark.parametrize(
    "argv",
    [
        ["jobs", "start", "-d", "eva", "--repo", "someone/registry"],
        ["jobs", "start", "-d", "eva", "--registry-path=registry.json"],
        ["jobs", "start", "-d", "eva", "--registry-url", "https://example.com/r.json"],
        ["jobs", "start", "-c", "job.yaml"],
        ["jobs", "start", "-p", "datasets/eva"],
        ["trials", "start", "-p", "datasets/eva/task"],
        ["jobs", "resume", "-p", "jobs/x"],
    ],
)
def test_a_chosen_registry_or_no_dataset_is_left_alone(argv: list[str]) -> None:
    assert with_default_registry(argv) == argv


def test_the_registry_flags_are_harbors(monkeypatch: pytest.MonkeyPatch) -> None:
    from harbor_voice import datasets

    cli._mount_harbor()
    monkeypatch.setattr(datasets, "checkout_registry", lambda: None)
    argv = with_default_registry(with_default_agent(["jobs", "start", "-d", "eva"]))
    result = CliRunner().invoke(cli.voice_app, [*argv, "--help"])
    assert result.exit_code == 0, result.output
