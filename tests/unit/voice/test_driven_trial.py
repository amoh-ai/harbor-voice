"""Driven trials: a task's [call] driver, here or in a harbor environment."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from harbor_voice.driver import DriverContext
from harbor_voice.driver import DriverError
from harbor_voice.driver import DriverOutcome
from harbor_voice.driver import load_driver
from harbor_voice.driver import VerifierResult
from harbor_voice.trial import AgentSpec
from harbor_voice.trial import DrivenOutcome
from harbor_voice.trial import TrialConfigError
from harbor_voice.trial import VoiceTaskConfig
from harbor_voice.trial import VoiceTrial

pytestmark = pytest.mark.unit

AGENT = {"pipeline": "s2s", "s2s": "offline.echo-s2s"}

seen: list[DriverContext] = []


async def _driver(ctx: DriverContext) -> DriverOutcome:
    seen.append(ctx)
    (ctx.out_dir / "call.txt").write_text("driven", encoding="utf-8")
    return DriverOutcome(
        verified=VerifierResult(reward=0.75, result={"note": "graded"}),
        summary={"steps": 3},
    )


@pytest.fixture(autouse=True)
def _module(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("fake_bench")
    module.drive = _driver  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_bench", module)
    seen.clear()


def test_caller_is_optional_only_when_a_driver_runs_the_call() -> None:
    VoiceTaskConfig.model_validate(
        {"agent": AGENT, "call": {"driver": "fake_bench:drive"}}
    )
    with pytest.raises(ValidationError, match="caller"):
        VoiceTaskConfig.model_validate({"agent": AGENT})


def test_a_verifier_needs_an_environment() -> None:
    with pytest.raises(ValidationError, match="declare one"):
        VoiceTaskConfig.model_validate(
            {
                "agent": AGENT,
                "call": {"driver": "fake_bench:drive"},
                "verifier": {"service": "tau2"},
            }
        )


def test_load_driver_explains_a_bad_path() -> None:
    with pytest.raises(DriverError, match="module:function"):
        load_driver("no_colon_here")
    with pytest.raises(DriverError, match="not importable"):
        load_driver("definitely_not_installed_xyz:drive")


async def test_a_driven_trial_writes_the_verified_reward(tmp_path: Path) -> None:
    config = VoiceTaskConfig.model_validate(
        {
            "agent": AGENT,
            "task": "fake/1",
            "call": {"driver": "fake_bench:drive", "config": {"domain": "x"}},
        }
    )
    outcome = await VoiceTrial(config, out_dir=tmp_path, task_dir=tmp_path).run()

    assert isinstance(outcome, DrivenOutcome)
    assert outcome.reward == {"reward": 0.75}
    assert json.loads((tmp_path / "reward.json").read_text()) == {"reward": 0.75}
    trial = json.loads((tmp_path / "trial.json").read_text())
    assert trial["driver"] == "fake_bench:drive"
    assert trial["summary"] == {"steps": 3}
    assert trial["verifier"] == {"note": "graded"}
    (ctx,) = seen
    assert ctx.driver_config == {"domain": "x"}
    assert ctx.services == {} and ctx.verify is None
    assert (tmp_path / "call.txt").read_text() == "driven"
    assert json.loads((tmp_path / "outcome.json").read_text())["verified"] == {
        "reward": 0.75,
        "result": {"note": "graded"},
    }


async def test_the_run_supplies_the_agent_not_the_task(tmp_path: Path) -> None:
    config = VoiceTaskConfig.model_validate({"call": {"driver": "fake_bench:drive"}})
    with pytest.raises(TrialConfigError, match="no agent under test"):
        await VoiceTrial(config, out_dir=tmp_path, task_dir=tmp_path).run()

    chosen = AgentSpec.model_validate(AGENT)
    trial = VoiceTrial(config, out_dir=tmp_path, task_dir=tmp_path, agent_spec=chosen)
    await trial.run()
    assert json.loads((tmp_path / "trial.json").read_text())["agent"]["s2s"] == (
        "offline.echo-s2s"
    )


async def test_a_driven_trial_needs_its_directories() -> None:
    config = VoiceTaskConfig.model_validate(
        {"agent": AGENT, "call": {"driver": "fake_bench:drive"}}
    )
    with pytest.raises(TrialConfigError, match="task_dir and out_dir"):
        await VoiceTrial(config).run()


class _Exec:
    def __init__(self, return_code: int = 0) -> None:
        self.return_code, self.stdout, self.stderr = return_code, "", ""


class _FakeEnvironment:
    """Records harbor's environment calls; plays main and a tau2 service."""

    def __init__(self, paths: object) -> None:
        self.paths = paths
        self.calls: list[tuple] = []
        self.capabilities = types.SimpleNamespace(mounted=True)

    async def start(self, force_build: bool) -> None:
        self.calls.append(("start",))

    async def exec(self, command: str, **kw: object) -> _Exec:
        self.calls.append(("exec", command))
        if command.startswith("harbor-voice drive"):
            call = Path(self.paths.agent_dir) / "call"  # type: ignore[attr-defined]
            call.mkdir(parents=True, exist_ok=True)
            (call / "outcome.json").write_text(
                json.dumps({"verified": {"reward": 1.0}, "summary": {"steps": 9}})
            )
        return _Exec()

    async def upload_file(self, source: Path, target: str) -> None:
        self.calls.append(("upload", target))

    async def download_dir(self, source: str, target: Path) -> None:
        self.calls.append(("download", "main", source))

    async def service_exec(self, command: str, *, service: str, **kw: object) -> _Exec:
        self.calls.append(("service_exec", service, command))
        return _Exec()

    async def service_download_dir(
        self, source: str, target: Path, *, service: str
    ) -> None:
        self.calls.append(("download", service, source))
        target.mkdir(parents=True, exist_ok=True)
        (target / "reward.txt").write_text("0.0")
        (target / "result.json").write_text(json.dumps({"by": "verifier"}))

    async def stop(self, delete: bool) -> None:
        self.calls.append(("stop", delete))


async def test_an_environment_trial_runs_harbors_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harbor.environments.factory import EnvironmentFactory

    from harbor_voice import install

    created: list[_FakeEnvironment] = []

    def create(*args: object, **kw: object) -> _FakeEnvironment:
        created.append(_FakeEnvironment(kw["trial_paths"]))
        return created[-1]

    wheel = install.Wheel("harbor", "0.0", tmp_path / "harbor-0.0-py3-none-any.whl")
    monkeypatch.setattr(EnvironmentFactory, "create_environment", create)
    monkeypatch.setattr(install, "build_wheels", lambda modules, out: [wheel])

    task = tmp_path / "task"
    task.mkdir()
    (task / "voice.toml").write_text("")
    config = VoiceTaskConfig.model_validate(
        {
            "task": "fake/1",
            "call": {"driver": "fake_bench:drive"},
            "environment": {},
            "verifier": {"service": "tau2"},
        }
    )
    out = tmp_path / "out"
    outcome = await VoiceTrial(
        config, out_dir=out, task_dir=task, agent_spec=AgentSpec.model_validate(AGENT)
    ).run()

    (env,) = created
    kinds = [c[0] for c in env.calls]
    assert kinds[0] == "start" and kinds[-1] == "stop"
    assert env.calls[-1] == ("stop", True)
    assert any(
        "uv pip install" in c[1] and "[voice]" in c[1]
        for c in env.calls
        if c[0] == "exec"
    )
    assert ("upload", "/task/voice.toml") in env.calls
    drive = next(
        c[1] for c in env.calls if c[0] == "exec" and "harbor-voice drive" in c[1]
    )
    assert "offline.echo-s2s" in drive
    assert ("service_exec", "tau2", "bash /tests/test.sh") in env.calls
    # The verifier's grade is the trial's reward, not the driver's own.
    assert outcome.reward == {"reward": 0.0}
    trial = json.loads((out / "trial.json").read_text())
    assert trial["environment"] == "docker"
    assert trial["installed"][0]["distribution"] == "harbor"
    assert trial["summary"] == {"steps": 9}
