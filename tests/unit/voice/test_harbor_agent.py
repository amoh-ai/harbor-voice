"""VoiceCallAgent: a voice call as a harbor agent, for harbor's job runner."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from harbor_voice import harbor_agent
from harbor_voice import install
from harbor_voice.harbor_agent import VoiceCallAgent
from harbor_voice.harbor_agent import VoiceInfrastructureError
from harbor_voice.harbor_agent import call_spec
from harbor_voice.trial import AgentSpec
from harbor_voice.trial import VoiceTaskConfig

pytestmark = pytest.mark.unit

SPEC = """\
task = "fake/1"
seed = 7

[call]
driver = "fake_bench:drive"

[call.config]
domain = "x"
seed = 7
"""

INSTRUCTION = f"# fake task\n\nSome prose.\n\n```toml\n{SPEC}```\n"


class _Exec:
    def __init__(self, return_code: int = 0, stderr: str = "") -> None:
        self.return_code, self.stdout, self.stderr = return_code, "", stderr


class _Environment:
    def __init__(self, drive_code: int = 0) -> None:
        self.calls: list[tuple] = []
        self.uploaded: dict[str, str] = {}
        self._drive_code = drive_code

    async def exec(self, command: str, **kw: object) -> _Exec:
        self.calls.append(("exec", command))
        if command.startswith("harbor-voice drive"):
            return _Exec(self._drive_code, stderr="provider down")
        return _Exec()

    async def upload_file(self, source: Path, target: str) -> None:
        self.calls.append(("upload", target))
        if target.endswith(".toml"):
            self.uploaded[target] = Path(source).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _no_wheel_builds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def build(modules: list[str], out: Path) -> list[install.Wheel]:
        name = "harbor" if modules == ["harbor_voice"] else modules[0]
        return [install.Wheel(name, "0.0", tmp_path / f"{name}-0.0.whl")]

    monkeypatch.setattr(install, "build_wheels", build)
    monkeypatch.setattr(install, "distribution_of", lambda module: "harbor")
    monkeypatch.setattr(harbor_agent, "_wheels", harbor_agent._Wheels())


def _agent(tmp_path: Path, **kwargs: object) -> VoiceCallAgent:
    return VoiceCallAgent(logs_dir=tmp_path / "agent", **kwargs)  # type: ignore[arg-type]


def test_the_instruction_carries_the_call_spec() -> None:
    assert call_spec(INSTRUCTION) == SPEC.rstrip("\n")
    with pytest.raises(VoiceInfrastructureError, match="toml"):
        call_spec("just words")


def test_the_model_is_the_agent_under_test(tmp_path: Path) -> None:
    agent = _agent(tmp_path, model_name="openai.realtime/gpt-realtime-1.5")
    assert json.loads(agent.agent_spec_json())["s2s"] == (
        "openai.realtime/gpt-realtime-1.5"
    )

    cascade = AgentSpec.model_validate({"pipeline": "s2s", "s2s": "offline.echo-s2s"})
    agent = _agent(tmp_path, agent_json=cascade.model_dump_json())
    assert json.loads(agent.agent_spec_json())["s2s"] == "offline.echo-s2s"

    with pytest.raises(ValueError, match="no agent under test"):
        _agent(tmp_path).agent_spec_json()


async def test_install_puts_harbor_voice_into_main(tmp_path: Path) -> None:
    environment = _Environment()
    await _agent(tmp_path, model_name="offline.echo-s2s").install(environment)  # type: ignore[arg-type]
    (command,) = [
        c[1] for c in environment.calls if c[0] == "exec" and "uv pip" in c[1]
    ]
    assert "harbor-0.0.whl[voice]" in command


async def test_run_installs_the_driver_and_drives_the_call(tmp_path: Path) -> None:
    environment = _Environment()
    agent = _agent(tmp_path, model_name="offline.echo-s2s", seed=626729)
    await agent.run(INSTRUCTION, environment, context=None)  # type: ignore[arg-type]

    execs = [c[1] for c in environment.calls if c[0] == "exec"]
    assert any("fake_bench-0.0.whl" in c and "[voice]" not in c for c in execs)
    drive = next(c for c in execs if c.startswith("harbor-voice drive"))
    assert "--seed 626729" in drive and "offline.echo-s2s" in drive
    assert environment.uploaded["/task/voice.toml"] == SPEC.rstrip("\n")


async def test_a_call_that_did_not_run_is_retryable_infra(tmp_path: Path) -> None:
    agent = _agent(tmp_path, model_name="offline.echo-s2s")
    with pytest.raises(VoiceInfrastructureError, match="provider down"):
        await agent.run(INSTRUCTION, _Environment(drive_code=1), context=None)  # type: ignore[arg-type]


def test_the_outcome_lands_in_the_trial_context(tmp_path: Path) -> None:
    agent = _agent(tmp_path, model_name="offline.echo-s2s")
    call = tmp_path / "agent" / "call"
    call.mkdir(parents=True)
    (call / "outcome.json").write_text(json.dumps({"summary": {"reward": 1.0}}))
    context = types.SimpleNamespace(metadata=None)
    agent.populate_context_post_run(context)  # type: ignore[arg-type]
    assert context.metadata == {"call": {"reward": 1.0}}


def test_with_seed_moves_the_task_and_its_driver() -> None:
    import tomllib

    config = VoiceTaskConfig.model_validate(tomllib.loads(SPEC)).with_seed(9)
    assert config.seed == 9
    assert config.call is not None and config.call.config["seed"] == 9
