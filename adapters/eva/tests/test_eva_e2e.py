"""One EVA trial end to end, offline: the driver against EVA's real sidecar.

Everything but EVA is faked, on real local sockets (``fixtures/e2e``): the
agent under test is a fake OpenAI Realtime server reached by harbor's own
``openai.realtime`` transport, and the caller is a fake ElevenLabs Agents
server (websocket and REST) reached by harbor's ``elevenlabs.agents``
transport and ``ElevenLabsCaller``. The sidecar is the real image, EVA's
code at the pinned commit, with only its judges faked
(``scripts/goldens/sidecar_fake_judges.py``). No API is called.

The trial runs the way ``harbor-voice drive`` runs it inside ``main``: a
task written by the converter, its ``voice.toml``, ``VoiceTrial.build_agent``
from a provider registry, and core's ``drive_here``, which loads
``eva_adapter.trial:drive`` by name. Two scenarios:

- one attempt that EVA validates, graded, with every output file;
- a first attempt whose record shows no ``end_call`` (the caller's end
  reason is ``session_ended``, EVA's gate rejects it, the attempt is
  archived) and a rerun that passes and is graded.

Each call is about 21 s of wall time plus EVA's teardown (the ``end_call``
check, the 4 s grace). Skips without Docker or EVA's base image.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from harbor.models.trajectories.trajectory import Trajectory
from harbor_voice.capabilities import Pipeline
from harbor_voice.driven import drive_here
from harbor_voice.driver import DriverOutcome
from harbor_voice.trajectory.model import EventType
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trial import AgentSpec
from harbor_voice.trial import VoiceTaskConfig
from harbor_voice.trial import VoiceTrial

from eva_adapter import images
from eva_adapter.main import TEMPLATE
from eva_adapter.main import write_task

HERE = Path(__file__).parent
GOLDENS = HERE.parent / "scripts" / "goldens"
sys.path.insert(0, str(HERE / "fixtures" / "e2e"))

from eva_e2e_fakes import REPLIES  # noqa: E402
from eva_e2e_fakes import TOOL_NAME  # noqa: E402
from eva_e2e_fakes import FakeElevenLabs  # noqa: E402
from eva_e2e_fakes import FakeRealtime  # noqa: E402

pytestmark = pytest.mark.integration

IMAGE = "eva-sidecar:e2e"
DOMAIN, RECORD = "airline", "1.1.2"
"""EVA's processor-golden record: persona 1, so the female caller agent."""
AGENT_F, AGENT_M = "agent_en_f_fake", "agent_en_m_fake"


def _docker(
    *args: str, check: bool = True, timeout: float = 600
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
        check=check,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def image() -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    try:
        found = _docker("image", "inspect", images.EVA_BASE, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"docker unavailable: {exc}")
    if found.returncode != 0:
        pytest.skip(f"{images.EVA_BASE} not built (eva-adapter build-image)")
    _docker("build", "-q", "-t", IMAGE, str(images.SIDECAR))
    return IMAGE


class Sidecar:
    """One sidecar container, its judges faked, on a free local port."""

    def __init__(self, image: str) -> None:
        self.id = _docker(
            "run",
            "-d",
            "--rm",
            "-p",
            "127.0.0.1::8000",
            "-v",
            f"{GOLDENS}:/goldens:ro",
            "--entrypoint",
            "/opt/venv/bin/python",
            image,
            "/goldens/sidecar_fake_judges.py",
        ).stdout.strip()
        port = _docker("port", self.id, "8000/tcp").stdout.strip().splitlines()[0]
        self.url = f"http://{port}"

    def ready(self) -> Sidecar:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        logs = _docker("logs", self.id, check=False).stderr[-3000:]
        raise RuntimeError(f"sidecar did not come up:\n{logs}")

    def judge_calls(self) -> int:
        return len(httpx.get(f"{self.url}/_fake_judges", timeout=10).json())

    def stop(self) -> None:
        _docker("rm", "-f", self.id, check=False)


@pytest.fixture
def sidecar(image: str) -> Iterator[Sidecar]:
    container = Sidecar(image)
    try:
        yield container.ready()
    finally:
        container.stop()


def _registry() -> Any:
    from harbor_voice.providers import openai
    from harbor_voice.providers.registry import default_registry

    registry = default_registry()
    openai.register_all(registry)
    return registry


async def _run(
    tmp_path: Path, sidecar: Sidecar, end_call: list[bool]
) -> tuple[DriverOutcome, Path, FakeRealtime, FakeElevenLabs]:
    """``harbor-voice drive`` for a converted task, in this process."""
    task = write_task(tmp_path / "tasks", DOMAIN, RECORD)
    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    assert config.call is not None
    out = tmp_path / "call"
    async with FakeRealtime() as agent, FakeElevenLabs(end_call) as caller:
        call = config.call.model_copy(
            update={
                "config": {
                    **config.call.config,
                    "sidecar_url": sidecar.url,
                    "caller_api_base": caller.api_base,
                }
            }
        )
        config = config.model_copy(update={"call": call})
        spec = AgentSpec(
            pipeline=Pipeline.S2S,
            s2s="openai.realtime/gpt-realtime-1.5",
            config={"s2s": {"url": agent.url, "api_key": "sk-fake"}},
        )
        trial = VoiceTrial(
            config, registry=_registry(), out_dir=out, task_dir=task, agent_spec=spec
        )
        outcome = await drive_here(
            config,
            task_dir=task,
            out_dir=out,
            build_agent=trial.build_agent,
            agent=trial.agent_spec(),
        )
    return outcome, out, agent, caller


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials only; nothing may reach a real service."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-fake")
    monkeypatch.setenv("EVA_EN_USER_F", AGENT_F)
    monkeypatch.setenv("EVA_EN_USER_M", AGENT_M)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_graded_trial(outcome: DriverOutcome, out: Path, attempts: int) -> None:
    eva = out / "eva"
    summary = _json(eva / "summary.json")
    assert outcome.summary == summary

    # The reward is EVA's composite, as the sidecar graded it.
    assert outcome.verified is not None
    reward = outcome.verified.reward
    assert reward in (0.0, 1.0)
    assert summary["reward_key"] == "EVA-A_pass"
    assert summary["aggregates"]["EVA-A_pass"] == reward
    assert outcome.verified.result["graded"] is True
    assert outcome.verified.result["validation"]["passed"] is True
    # Every attempt's call held both caller turns (a stalled call shows here).
    turns = [(a["attempt"], a["stats"].get("num_turns")) for a in summary["attempts"]]
    assert all(n == 2 for _, n in turns), turns
    assert summary["validated"] and summary["attempts_used"] == attempts
    assert summary["counted_attempt"] == attempts
    last = summary["attempts"][-1]
    assert last["end_reason"] == "goodbye"
    assert last["validation"]["reason"] == "passed"
    assert last["session_check"] == "matched"
    assert last["tool_calls"] == 1
    assert last["caller"]["agent_id"] == AGENT_F
    assert last["caller"]["end_call_found"] is True
    for name in ("call_s", "record_s", "validate_s"):
        assert last["timings"][name] > 0

    # Usage per component: the agent's meter, the caller's record, EVA's judges.
    by = summary["usage_by_component"]
    assert by["agent"][0]["requests"] >= len(REPLIES)
    assert {u["component"] for u in by["caller"]} >= {"caller", "caller.llm"}
    assert any(u["model"] == "gpt-5.1" for u in by["caller"])
    judges = {u["component"] for u in by["judges"]}
    assert {"judge.user_behavioral_fidelity", "judge.user_speech_fidelity"} <= judges
    assert "judge.faithfulness" in judges

    # EVA's own files: its run directory, the record that counts, its metrics.
    record = eva / "run" / "records" / RECORD / "trial_0"
    for name in (
        "result.json",
        "metrics.json",
        "audit_log.json",
        "user_simulator_events.jsonl",
        "audio_assistant.wav",
        "audio_user_clean.wav",
        "final_scenario_db.json",
        "elevenlabs_conversation_details.json",
    ):
        assert (record / name).exists(), name
    assert (eva / "run" / "config.json").exists()
    assert (eva / "run" / "evaluation_summary.json").exists()
    metrics = _json(eva / "metrics.json")
    assert metrics == _json(record / "metrics.json")
    assert metrics["aggregate_metrics"]["EVA-A_pass"] == reward
    for n in range(1, attempts + 1):
        assert (eva / f"attempt-{n}" / "attempt.json").exists()
        # EVA's run directory holds the record now; no second copy is kept.
        assert not (eva / f"attempt-{n}" / "record").exists()
    assert (eva / "elevenlabs" / "elevenlabs_agent_en_f.json").exists()

    # The ATIF-voice trajectory of the attempt that counts, audio by hash only.
    atif = (out / "trajectory.json").read_text(encoding="utf-8")
    voice = VoiceTrajectory.from_atif(Trajectory.model_validate_json(atif))
    assert voice.task == f"eva/{DOMAIN}/{RECORD}"
    assert voice.attempt == attempts
    kinds = {(e.type, e.party) for e in voice.timeline}
    assert (EventType.SPEECH_START, "agent") in kinds
    assert (EventType.SPEECH_START, "caller") in kinds
    tools = [e for e in voice.timeline if e.type == EventType.TOOL_CALL]
    assert [t.meta["name"] for t in tools] == [TOOL_NAME]
    assert voice.turns[0].said is None and "help" in (voice.turns[0].reply or "")
    assert any(t.said and "reservation" in t.said for t in voice.turns)
    assert voice.artifacts.mixed_timeline is not None
    assert set(voice.artifacts.per_party) == {"agent", "caller"}
    assert any(u.component.startswith("judge.") for u in voice.usage)
    assert len(atif) < 200_000  # no audio inlined


async def test_one_validated_attempt_is_graded(
    tmp_path: Path, sidecar: Sidecar, env: None
) -> None:
    outcome, out, agent, caller = await _run(tmp_path, sidecar, end_call=[True])

    _check_graded_trial(outcome, out, attempts=1)
    # The caller got EVA's prompt as its one dynamic variable; the agent got
    # EVA's session and the tool's output.
    sent = caller.conversations[0].initiation
    assert set(sent["dynamic_variables"]) == {"prompt"}
    assert len(sent["dynamic_variables"]["prompt"]) > 500
    assert caller.conversations[0].heard_bytes > 8000  # the agent's speech
    session = agent.sessions[0]
    assert session.session is not None and session.session["audio"]["input"][
        "format"
    ] == {"type": "audio/pcm", "rate": 24000}
    assert [json.loads(o["output"])["status"] for o in session.tool_outputs] in (
        ["success"],
        ["error"],
    )

    # The task's verifier (tests/grade.py, as test.sh runs it) reads back the
    # driver's grade: same reward, and no judge is called a second time.
    before = sidecar.judge_calls()
    verifier = tmp_path / "verifier"
    subprocess.run(
        [
            sys.executable,
            str(TEMPLATE / "tests" / "grade.py"),
            "--reward",
            "EVA-A_pass",
            "--sidecar",
            sidecar.url,
            "--out",
            str(verifier),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        timeout=600,
    )
    graded = _json(verifier / "result.json")
    assert outcome.verified is not None
    driver = dict(outcome.verified.result)
    driver.pop("validation")
    assert graded == driver
    assert _json(verifier / "reward.json")["reward"] == outcome.verified.reward
    assert sidecar.judge_calls() == before


async def test_a_failed_validation_is_rerun_and_the_rerun_counts(
    tmp_path: Path, sidecar: Sidecar, env: None
) -> None:
    outcome, out, _agent, caller = await _run(tmp_path, sidecar, end_call=[False, True])

    _check_graded_trial(outcome, out, attempts=2)
    first = outcome.summary["attempts"][0]
    assert first["end_reason"] == "session_ended"
    assert first["caller"]["end_call_found"] is False
    assert first["validation"]["reason"] == "not_finished"
    assert first["validation"]["rerun"] is True
    assert outcome.summary["rerun_history"][0]["reason"] == "not_finished"
    # EVA archived the failed attempt beside the record that counts.
    records = out / "eva" / "run" / "records" / RECORD
    assert (
        records / "trial_0_failed_attempt_1" / "user_simulator_events.jsonl"
    ).exists()
    assert len(caller.conversations) == 2
