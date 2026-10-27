"""harbor-voice jobs / trials: harbor's own commands, mounted, agent defaulted."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from harbor_voice import cli
from harbor_voice.cli import CALL_AGENT
from harbor_voice.cli import with_default_agent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("runner", ["jobs", "job", "trials", "trial"])
def test_start_without_an_agent_runs_the_call_agent(runner: str) -> None:
    argv = [runner, "start", "-p", "task", "-m", "openai.realtime/gpt-realtime-1.5"]
    assert with_default_agent(argv) == [
        runner,
        "start",
        "--agent",
        CALL_AGENT,
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
