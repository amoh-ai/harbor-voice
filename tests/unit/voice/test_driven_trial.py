"""Driven trials: a task's [call] driver, between core's services and verifier."""

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
from harbor_voice.services import VerifierResult
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


def test_a_verifier_must_run_in_a_declared_service() -> None:
    with pytest.raises(ValidationError, match="does not declare"):
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


async def test_a_driven_trial_needs_its_directories() -> None:
    config = VoiceTaskConfig.model_validate(
        {"agent": AGENT, "call": {"driver": "fake_bench:drive"}}
    )
    with pytest.raises(TrialConfigError, match="task_dir and out_dir"):
        await VoiceTrial(config).run()
