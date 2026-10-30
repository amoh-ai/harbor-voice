"""The EVA driver without Docker: the attempt loop, the parity guard, the trajectory.

The sidecar is a fake that answers as EVA's would (``/open`` numbering the
attempts, ``/validate`` scripted, ``/files`` laying out EVA's run
directory), and the call is a stub that writes record files. The record
used is the call golden's (``fixtures/call/g5``): what harbor's call loop wrote for the
scripted call EVA's own code was checked against. The end to end with the
real sidecar is ``test_eva_e2e.py``.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from harbor_voice.capabilities import Pipeline
from harbor_voice.driver import DriverContext
from harbor_voice.ids import PartyId
from harbor_voice.providers.openai.realtime import OpenAIRealtimeTransport
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.trajectory.model import EventType
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.trial import AgentSpec
from harbor_voice.trial import VoiceTaskConfig

from eva_adapter import trajectory as eva_trajectory
from eva_adapter.call import CallOutcome
from eva_adapter.client import AgentSetup
from eva_adapter.client import CallerSetup
from eva_adapter.client import CallInfo
from eva_adapter.client import Grade
from eva_adapter.client import Opened
from eva_adapter.client import SidecarError
from eva_adapter.client import ToolReply
from eva_adapter.client import Validation
from eva_adapter.conditions import PerturbationConfig
from eva_adapter.tools import SidecarToolExecutor
from eva_adapter.tools import tool_specs
from eva_adapter.trial import AttemptCall
from eva_adapter.trial import EvaConfigError
from eva_adapter.trial import EvaParityError
from eva_adapter.trial import TrialConfig
from eva_adapter.trial import _preflight
from eva_adapter.trial import _require_realtime
from eva_adapter.trial import check_session
from eva_adapter.trial import eva_realtime_settings
from eva_adapter.trial import judge_usage
from eva_adapter.trial import run_trial
from eva_adapter.vendored import perturbation as vendored

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures"
CALL_GOLDEN = FIXTURES / "call" / "g5"
SESSION_GOLDEN = json.loads(
    (FIXTURES / "realtime" / "eva_session.json").read_text(encoding="utf-8")
)
RECORD_FILES = ("audit_log.json", "user_simulator_events.jsonl")
CONFIG = TrialConfig(domain="airline", record_id="1.1.2")
JUDGES = {
    "faithfulness": {
        "model_id": "us.anthropic.claude-opus-4-6-v1",
        "total_input_tokens": 9000,
        "total_output_tokens": 300,
        "num_calls": 1,
    },
    "user_behavioral_fidelity": {
        "model_id": "gpt-5.2-medium",
        "total_input_tokens": 4000,
        "total_output_tokens": 100,
        "num_calls": 2,
    },
}


def _opened(attempt: int, *, session: dict[str, Any] | None = None) -> Opened:
    return Opened(
        conversation_id="1.1.2/trial_0",
        record_id="1.1.2",
        trial=0,
        attempt=attempt,
        max_attempts=3,
        time_limit_s=600.0,
        caller_timeout_s=620.0,
        agent=AgentSetup(
            system_prompt="You are an agent.",
            tools=[],
            greeting="Hello! How can I help you today?",
            opening_text="Say: 'Hello! How can I help you today?'",
            session=session,
        ),
        caller=CallerSetup(prompt="You are a caller.", gender="F", language="en"),
        raw={"agent": {"model": "gpt-realtime-1.5"}},
    )


def _validation(attempt: int, reason: str, *, rerun: bool) -> Validation:
    passed = reason in ("passed", "time_limit_accepted")
    raw = {
        "attempt": attempt,
        "passed": passed,
        "gate": reason != "not_finished",
        "reason": reason,
        "rerun": rerun,
        "rerun_history": [{"attempt": 1, "reason": "not_finished"}],
    }
    return Validation(
        attempt=attempt,
        passed=passed,
        gate=raw["gate"],
        reason=reason,
        scores={},
        failed_metrics=[],
        rerun=rerun,
        restored_attempt=None,
        raw=raw,
    )


class FakeSidecar:
    """EVA's sidecar as the driver sees it."""

    def __init__(
        self,
        verdicts: list[tuple[str, bool]],
        *,
        grade: Grade | Exception | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self.verdicts = verdicts
        self._grade = grade
        self.fetch_error = fetch_error
        self.attempt = 0
        self.calls: list[tuple[str, Any]] = []
        self.recorded: list[tuple[CallInfo, list[str]]] = []

    async def open(self, **kwargs: Any) -> Opened:
        self.attempt += 1
        self.calls.append(("open", kwargs))
        return _opened(self.attempt)

    async def record(self, record_dir: Path, call: CallInfo) -> dict[str, Any]:
        files = sorted(p.name for p in record_dir.iterdir())
        self.recorded.append((call, files))
        self.calls.append(("record", call.attempt))
        return {"record_dir": "x"}

    async def validate(self, attempt: int) -> Validation:
        self.calls.append(("validate", attempt))
        reason, rerun = self.verdicts[attempt - 1]
        return _validation(attempt, reason, rerun=rerun)

    async def grade(self) -> Grade:
        self.calls.append(("grade", None))
        if isinstance(self._grade, Exception):
            raise self._grade
        assert self._grade is not None
        return self._grade

    async def fetch(self, dest: Path) -> Path:
        """EVA's run directory: the record, its archives, the judges' usage."""
        self.calls.append(("fetch", None))
        if self.fetch_error is not None:
            raise self.fetch_error
        records = dest / "records" / "1.1.2"
        passed = self.verdicts[self.attempt - 1][0] == "passed"
        for n in range(1, self.attempt + 1):
            counted = passed and n == self.attempt
            target = records / ("trial_0" if counted else f"trial_0_failed_attempt_{n}")
            target.mkdir(parents=True)
            for name in RECORD_FILES:
                shutil.copyfile(CALL_GOLDEN / name, target / name)
            (target / "result.json").write_text("{}", encoding="utf-8")
            if counted:
                (target / "metrics.json").write_text(
                    json.dumps({"record_id": "1.1.2/trial_0", "eva": True}),
                    encoding="utf-8",
                )
        usage = dest / "judge_token_usage" / "judge_token_usage.json"
        usage.parent.mkdir(parents=True)
        usage.write_text(json.dumps(JUDGES), encoding="utf-8")
        return dest


GRADE = Grade(
    record_id="1.1.2/trial_0",
    metrics={
        "task_completion": {"score": 1.0, "normalized_score": 1.0, "details": {"x": 1}}
    },
    aggregates={"EVA-A_pass": 1.0, "EVA-X_pass": 0.0, "EVA-overall_pass": 0.0},
    context=None,
)


def _stub_call(*, error: str | None = None):
    async def call(opened: Opened, record_dir: Path) -> AttemptCall:
        record_dir.mkdir(parents=True, exist_ok=True)
        for name in RECORD_FILES:
            shutil.copyfile(CALL_GOLDEN / name, record_dir / name)
        start = datetime(2026, 10, 28, 12, 0, 0)
        return AttemptCall(
            outcome=CallOutcome(
                conversation_ended_reason="error" if error else "goodbye",
                started_at=start,
                ended_at=start + timedelta(seconds=30),
                out_dir=record_dir,
                error=error,
            ),
            usage=(
                UsageRecord(component="agent", model="gpt-realtime-1.5", requests=3),
                UsageRecord(component="caller", model="agent_f", seconds=30.0),
            ),
            agent={"provider": "openai.realtime", "model": "gpt-realtime-1.5"},
            caller={"end_reason": "goodbye"},
            tools=[{"name": "get_reservation"}],
            session_check="matched",
        )

    return call


async def test_a_rerun_then_a_pass_is_graded_and_written(tmp_path: Path) -> None:
    sidecar = FakeSidecar([("not_finished", True), ("passed", False)], grade=GRADE)
    config = TrialConfig(domain="airline", record_id="1.1.2", reward="EVA-overall_pass")
    seen: list[str] = []

    run = await run_trial(
        sidecar, config, out_dir=tmp_path, call=_stub_call(), progress=seen.append
    )

    assert [c[0] for c in sidecar.calls] == [
        "open",
        "record",
        "validate",
        "open",
        "record",
        "validate",
        "grade",
        "fetch",
    ]
    assert sidecar.calls[0][1] == {
        "domain": "airline",
        "record_id": "1.1.2",
        "language": "en",
        "trial": 0,
        "num_trials": 3,
    }
    # Every attempt's record went to EVA with its own number.
    assert [call.attempt for call, _ in sidecar.recorded] == [1, 2]
    assert sidecar.recorded[0][1] == sorted(RECORD_FILES)
    # The reward is the chosen composite; the result names it.
    assert run.reward == 0.0
    assert run.result["reward_key"] == "EVA-overall_pass"
    # harbor's rewards: the chosen one as ``reward``, every composite by name.
    assert run.result["rewards"] == {
        "reward": 0.0,
        "EVA-A_pass": 1.0,
        "EVA-X_pass": 0.0,
        "EVA-overall_pass": 0.0,
    }
    assert run.result["metrics"] == {
        "task_completion": {
            "score": 1.0,
            "normalized_score": 1.0,
            "skipped": None,
            "error": None,
        }
    }
    assert run.result["validation"] == {
        "passed": True,
        "reason": "passed",
        "attempts": 2,
        "counted_attempt": 2,
    }
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary == run.summary
    assert [a["validation"]["reason"] for a in summary["attempts"]] == [
        "not_finished",
        "passed",
    ]
    assert summary["attempts"][0]["timings"]["call_s"] == 30.0
    assert summary["rerun_history"] == [{"attempt": 1, "reason": "not_finished"}]
    # Usage: both attempts' agent and caller, and EVA's judges.
    usage = {(u["component"], u.get("model")): u for u in summary["usage"]}
    assert usage[("agent", "gpt-realtime-1.5")]["requests"] == 6
    assert usage[("caller", "agent_f")]["seconds"] == 60.0
    assert usage[("judge.user_behavioral_fidelity", "gpt-5.2-medium")]["requests"] == 2
    assert [u["component"] for u in summary["usage_by_component"]["judges"]] == [
        "judge.faithfulness",
        "judge.user_behavioral_fidelity",
    ]
    # EVA's metrics.json for the record that counts; no second copy of records.
    assert json.loads((tmp_path / "metrics.json").read_text())["eva"] is True
    assert (tmp_path / "attempt-1" / "attempt.json").exists()
    assert not (tmp_path / "attempt-1" / "record").exists()
    # The trajectory is the counted attempt's, with that attempt's usage.
    assert run.trajectory is not None
    assert run.trajectory.attempt == 2
    assert run.trajectory.task == "eva/airline/1.1.2"
    components = [u.component for u in run.trajectory.usage]
    assert components.count("agent") == 1 and "judge.faithfulness" in components
    assert seen[-1] == "graded: EVA-overall_pass 0.0"


async def test_no_validated_attempt_is_not_graded(tmp_path: Path) -> None:
    sidecar = FakeSidecar(
        [("not_finished", True), ("validation_failed", True), ("not_finished", False)]
    )

    run = await run_trial(sidecar, CONFIG, out_dir=tmp_path, call=_stub_call())

    assert "grade" not in [c[0] for c in sidecar.calls]
    assert ("fetch", None) in sidecar.calls
    assert run.reward is None
    assert run.result["graded"] is False
    assert run.summary["counted_attempt"] is None
    assert not (tmp_path / "metrics.json").exists()
    # The last attempt, as EVA archived it, is the trajectory.
    assert run.trajectory is not None and run.trajectory.attempt == 3
    assert run.summary["files"]["record"] == (
        "run/records/1.1.2/trial_0_failed_attempt_3"
    )


async def test_a_failed_grade_still_fetches_then_fails(tmp_path: Path) -> None:
    sidecar = FakeSidecar([("passed", False)], grade=SidecarError("judges down"))

    with pytest.raises(SidecarError, match="judges down"):
        await run_trial(sidecar, CONFIG, out_dir=tmp_path, call=_stub_call())
    assert sidecar.calls[-1] == ("fetch", None)


async def test_without_evas_files_the_local_record_stays(tmp_path: Path) -> None:
    sidecar = FakeSidecar(
        [("passed", False)], grade=GRADE, fetch_error=SidecarError("gone")
    )

    run = await run_trial(sidecar, CONFIG, out_dir=tmp_path, call=_stub_call())

    assert (tmp_path / "attempt-1" / "record" / "audit_log.json").exists()
    assert run.summary["files"]["fetch_error"] == "SidecarError: gone"
    assert run.summary["files"]["record"] == "attempt-1/record"
    assert run.trajectory is not None
    # EVA's metrics are not on disk here, so the grade's are written instead.
    assert json.loads((tmp_path / "metrics.json").read_text())["record_id"] == (
        "1.1.2/trial_0"
    )


async def test_a_failed_call_is_recorded_with_its_error(tmp_path: Path) -> None:
    sidecar = FakeSidecar([("not_finished", False)])

    await run_trial(sidecar, CONFIG, out_dir=tmp_path, call=_stub_call(error="boom"))

    call, _ = sidecar.recorded[0]
    assert call.error == "boom" and call.conversation_ended_reason == "error"


# ------------------------------------------------------------- the guard


def _eva_transport(
    session: dict[str, Any], **overrides: Any
) -> OpenAIRealtimeTransport:
    sidecar: Any = None
    executor = SidecarToolExecutor(sidecar, tool_specs(session["tools"]))
    settings = eva_realtime_settings(
        f"Say: '{SESSION_GOLDEN['session']['initial_message']}'"
    )
    if overrides:
        import dataclasses

        settings = dataclasses.replace(settings, **overrides)
    return OpenAIRealtimeTransport(
        party=PartyId("agent"),
        system_prompt=session["instructions"],
        tools=ToolRouter(executor),
        settings=settings,
    )


def test_the_driver_sets_the_agent_up_exactly_as_eva_does() -> None:
    session = SESSION_GOLDEN["wire"][0]["session"]
    opened = _opened(1, session=session)
    assert check_session(_eva_transport(session), opened) == "matched"


def test_any_difference_from_evas_session_fails_the_trial() -> None:
    session = SESSION_GOLDEN["wire"][0]["session"]
    with pytest.raises(EvaParityError) as caught:
        check_session(
            _eva_transport(session, voice="alloy"), _opened(1, session=session)
        )
    assert "session.audio.output.voice: 'alloy', EVA sends 'marin'" in str(caught.value)


def test_the_guard_says_when_it_does_not_apply() -> None:
    assert check_session(object(), _opened(1, session={"x": 1})).startswith(
        "not applicable: object"
    )
    assert check_session(object(), _opened(1)).startswith("not applicable")


async def test_tools_go_to_the_sidecar_and_failures_raise() -> None:
    class Tools:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        async def tool(self, name: str, arguments: dict[str, Any]) -> ToolReply:
            if self.fail:
                raise SidecarError("sidecar /tool returned 500")
            return ToolReply(result={"status": "success", "n": 1}, output="{}")

    ok: Any = Tools(False)
    executor = SidecarToolExecutor(ok, [])
    result = await executor.execute(ToolInvocation("c1", "get_x", {"a": 1}))
    assert result == {"status": "success", "n": 1}
    assert executor.log[0]["result"] == result and executor.log[0]["ended"] > 0

    bad: Any = Tools(True)
    failing = SidecarToolExecutor(bad, [])
    with pytest.raises(SidecarError):
        await failing.execute(ToolInvocation("c2", "get_x", {}))
    assert "500" in failing.log[0]["error"]


# ------------------------------------------------------------ the config


def _context(config: dict[str, Any], *, seed: int = 0, agent: AgentSpec | None = None):
    voice = VoiceTaskConfig.model_validate(
        {"seed": seed, "call": {"driver": "eva_adapter.trial:drive", "config": config}}
    )
    return DriverContext(
        config=voice,
        driver_config=config,
        task_dir=Path("."),
        out_dir=Path("."),
        services={},
        build_agent=lambda **_: None,  # ty: ignore[invalid-argument-type]
        agent=agent,
    )


def test_the_seed_is_evas_trial_index_and_the_reward_is_checked() -> None:
    config = TrialConfig.from_context(
        _context({"domain": "itsm", "record_id": "12"}, seed=2)
    )
    assert config.trial == 2 and config.output_id == "12/trial_2"
    assert config.reward == "EVA-A_pass" and config.sidecar_url == "http://eva:8000"
    with pytest.raises(EvaConfigError, match="num_trials 3"):
        TrialConfig(domain="itsm", record_id="12", trial=3)
    with pytest.raises(EvaConfigError, match="not one of EVA's"):
        TrialConfig(domain="itsm", record_id="12", reward="EVA-X_mean")


def test_only_evas_reference_provider_and_keys_are_checked_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_realtime(None)
    _require_realtime(AgentSpec(pipeline=Pipeline.S2S, s2s="openai.realtime/x"))
    with pytest.raises(EvaConfigError, match="openai.live"):
        _require_realtime(AgentSpec(pipeline=Pipeline.S2S, s2s="openai.live/x"))

    ctx = _context({"domain": "itsm", "record_id": "12"})
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(EvaConfigError, match="ELEVENLABS_API_KEY"):
        _preflight(ctx, CONFIG)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    _preflight(ctx, CONFIG)
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(EvaConfigError, match="OPENAI_API_KEY"):
        _preflight(ctx, CONFIG)


def test_judge_usage_reads_evas_running_totals(tmp_path: Path) -> None:
    assert judge_usage(tmp_path) == ()
    path = tmp_path / "judge_token_usage" / "judge_token_usage.json"
    path.parent.mkdir()
    path.write_text(json.dumps(JUDGES), encoding="utf-8")
    faithfulness, behaviour = judge_usage(tmp_path)
    assert faithfulness == UsageRecord(
        component="judge.faithfulness",
        model="us.anthropic.claude-opus-4-6-v1",
        requests=1,
        input_tokens=9000,
        output_tokens=300,
    )
    assert behaviour.requests == 2


# -------------------------------------------------------- the trajectory


def test_the_trajectory_is_read_from_evas_record() -> None:
    voice = eva_trajectory.build(CALL_GOLDEN, task="eva/airline/1.1.2", attempt=1)

    starts = [e for e in voice.timeline if e.type == EventType.SPEECH_START]
    assert {e.party for e in starts} == {"agent", "caller"}
    # The caller's four lines, each its own exchange; the greeting first.
    assert voice.turns[0].said is None
    assert voice.turns[0].reply == "Hello! How can I help you today?"
    said = [t.said for t in voice.turns[1:]]
    assert said == [
        "I need to change my flight.",
        "Yes, I want to move it to Friday.",
        "Wait, what about the evening?",
        "Yes, please. Thanks, goodbye.",
    ]
    # What the agent transcribed of the caller, and EVA's [interrupted] partial.
    assert voice.turns[1].heard == "I need to change my flight."
    cut = [t for t in voice.turns if t.stop_reason == "cut_off"]
    assert len(cut) == 1 and "[interrupted]" not in (cut[0].reply or "")
    assert any(e.type == EventType.BARGE_IN for e in voice.timeline)
    tools = [e for e in voice.timeline if e.type == EventType.TOOL_CALL]
    assert len(tools) == 1
    assert tools[0].meta["name"] == "get_reservation"
    assert tools[0].meta["arguments"] == {
        "confirmation_number": "ABC123",
        "last_name": "Smith",
    }
    assert tools[0].meta["result"]["status"] == "success"
    assert all(e.t >= 0 for e in voice.timeline)
    # The call golden's fixture keeps no WAVs: nothing to reference, nothing inlined.
    assert voice.artifacts.mixed_timeline is None


def test_words_without_audio_still_reach_the_trajectory(tmp_path: Path) -> None:
    (tmp_path / "user_simulator_events.jsonl").write_text(
        json.dumps(
            {
                "timestamp": 1000_000,
                "type": "user_speech",
                "data": {"text": "hello"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    voice = eva_trajectory.build(tmp_path)
    assert voice.turns[0].said == "hello"


# ------------------------------------------------------------ perturbations

STATIC = PerturbationConfig("bad_connection_static", 5.0)


def test_the_condition_is_read_from_the_task_and_named() -> None:
    config = TrialConfig.from_context(
        _context(
            {
                "domain": "itsm",
                "record_id": "12",
                "perturbation": {
                    "background_noise": "coffee_shop",
                    "snr_db": 10.0,
                    "connection_degradation": True,
                },
            }
        )
    )
    assert config.perturbation == PerturbationConfig("coffee_shop", 10.0, True)
    assert config.condition == "coffee_shop-10db-connection_degradation"
    assert config.task == "eva/itsm/12@coffee_shop-10db-connection_degradation"
    assert CONFIG.condition == "clean" and CONFIG.task == "eva/airline/1.1.2"
    for bad, message in (
        ({"background_noise": "jungle"}, "not one of EVA's"),
        ({"accent": "french"}, "not supported yet"),
        ("coffee_shop", "must be a table"),
    ):
        with pytest.raises(EvaConfigError, match=message):
            TrialConfig.from_context(
                _context({"domain": "itsm", "record_id": "12", "perturbation": bad})
            )


def test_a_missing_noise_file_fails_before_any_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(vendored, "_ASSETS_DIR", tmp_path)
    ctx = _context({"domain": "itsm", "record_id": "12"})
    noisy = TrialConfig(
        domain="itsm", record_id="12", perturbation=PerturbationConfig("coffee_shop")
    )
    with pytest.raises(EvaConfigError, match="coffee_shop-15db cannot run here"):
        _preflight(ctx, noisy)
    # Static and degradation need no file.
    _preflight(ctx, TrialConfig(domain="itsm", record_id="12", perturbation=STATIC))


async def test_the_summary_and_the_trajectory_name_the_condition(
    tmp_path: Path,
) -> None:
    sidecar = FakeSidecar([("passed", False)], grade=GRADE)
    config = TrialConfig(domain="airline", record_id="1.1.2", perturbation=STATIC)
    run = await run_trial(sidecar, config, out_dir=tmp_path, call=_stub_call())

    assert run.summary["condition"] == "bad_connection_static-5db"
    assert run.summary["perturbation"] == {
        "background_noise": "bad_connection_static",
        "snr_db": 5.0,
        "connection_degradation": False,
    }
    assert run.summary["numpy"]
    assert run.trajectory is not None
    assert run.trajectory.task == "eva/airline/1.1.2@bad_connection_static-5db"
    assert run.trajectory.config_matrix is not None
    assert run.trajectory.config_matrix.profile == "eva/bad_connection_static-5db"
    assert run.trajectory.is_degraded
    plan = run.trajectory.perturbation_plan
    assert plan["layers"] == {
        "eva.background_noise": {"type": "bad_connection_static", "snr_db": 5.0}
    }
    assert plan["seeded"] is False and plan["numpy"]

    clean = await run_trial(
        FakeSidecar([("passed", False)], grade=GRADE),
        CONFIG,
        out_dir=tmp_path / "clean",
        call=_stub_call(),
    )
    assert (
        clean.summary["condition"] == "clean" and clean.summary["perturbation"] is None
    )
    assert clean.trajectory is not None and not clean.trajectory.is_degraded
    assert clean.trajectory.config_matrix is not None
    assert clean.trajectory.config_matrix.profile == "clean"


async def test_each_attempt_gets_a_new_eva_perturbator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EVA builds one per conversation; the call loop gets it as EVA's bridge does."""
    import eva_adapter.trial as trial_module
    from eva_adapter.events import EventLog
    from eva_adapter.trial import realtime_attempts

    made: list[dict[str, Any]] = []

    class FakeCall:
        def __init__(self, **kwargs: Any) -> None:
            made.append(kwargs)
            self.events = EventLog(tmp_path / f"e{len(made)}.jsonl", provider="x")
            self.bridge = None

        async def run(self, caller: Any) -> CallOutcome:
            now = datetime(2026, 10, 29)
            return CallOutcome("goodbye", now, now, made[-1]["out_dir"])

    monkeypatch.setattr(trial_module, "RealtimeCall", FakeCall)
    monkeypatch.setenv("EVA_EN_USER_F", "agent-f")
    for config, expect in (
        (TrialConfig(domain="airline", record_id="1.1.2", perturbation=STATIC), True),
        (CONFIG, False),
    ):
        made.clear()
        config = TrialConfig(
            domain=config.domain,
            record_id=config.record_id,
            perturbation=config.perturbation,
            snapshot_caller=False,
        )
        attempt = realtime_attempts(
            _context({"domain": "airline", "record_id": "1.1.2"}),
            config,
            None,
            tmp_path,
        )
        for n in (1, 2):
            await attempt(_opened(n), tmp_path / f"attempt-{n}")
        first, second = (call["perturbator"] for call in made)
        if expect:
            assert isinstance(first, vendored.AudioPerturbator)
            assert isinstance(second, vendored.AudioPerturbator)
            assert first is not second and first.has_ambient_noise
        else:
            assert first is None and second is None
