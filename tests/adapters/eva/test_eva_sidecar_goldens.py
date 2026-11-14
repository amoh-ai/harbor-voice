"""The sidecar against EVA's own code, one golden per interface.

Every golden in ``fixtures/sidecar/`` was produced by EVA's code run natively
(``adapters/eva/scripts/goldens/sidecar_goldens.py``, in ``eva-base:039efa0a``). Here the
sidecar image is built on that base, run, and driven over HTTP through
``eva_adapter.client``, the way harbor-voice drives it; its answers must
equal EVA's.

Judges are fakes (``adapters/eva/scripts/goldens/sidecar_fake_judges.py``, the sidecar's
entrypoint in these tests): no API is called, and every judge request is
captured for the judge-prompt golden. The tests skip when Docker or the base image is missing.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from eva_adapter.client import CallInfo
from eva_adapter.client import Grade
from eva_adapter.client import Sidecar
from eva_adapter.client import SidecarError
from eva_adapter.client import ToolReply
from eva_adapter.client import Validation

ADAPTER = Path(__file__).resolve().parents[3] / "adapters" / "eva"
FIXTURES = Path(__file__).parent / "fixtures" / "sidecar"
GOLDENS = ADAPTER / "scripts" / "goldens"
SIDECAR = ADAPTER / "sidecar"
BASE_IMAGE = "eva-base:039efa0a"
IMAGE = "eva-sidecar:goldens"
RECORD = "1.1.2"
CASCADE_MODEL = {
    "llm": "gpt-5.2",
    "stt": "deepgram",
    "stt_params": {"model": "nova-3"},
    "tts": "cartesia",
    "tts_params": {"model": "sonic"},
}
"""EVA's processor golden was recorded on a cascade run; the processor golden grades it as one."""


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


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _wire(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


@pytest.fixture(scope="module")
def image() -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    try:
        found = _docker("image", "inspect", BASE_IMAGE, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"docker unavailable: {exc}")
    if found.returncode != 0:
        pytest.skip(f"{BASE_IMAGE} not built (EVA's Dockerfile at 039efa0a)")
    _docker("build", "-q", "-t", IMAGE, str(SIDECAR))
    return IMAGE


class _Container:
    """One sidecar, its judges faked, on a free local port."""

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

    def ready(self) -> str:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return self.url
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        logs = _docker("logs", self.id, check=False).stderr[-3000:]
        raise RuntimeError(f"sidecar did not come up:\n{logs}")

    def stop(self) -> None:
        _docker("rm", "-f", self.id, check=False)


@pytest.fixture(scope="module")
def sidecars(image: str) -> Iterator[dict[str, _Container]]:
    """All of this module's sidecars, started together (one session each)."""
    started: dict[str, _Container] = {}
    try:
        for name in ("open", "s2s", "cascade", "attempts"):
            started[name] = _Container(image)
        yield started
    finally:
        for container in started.values():
            container.stop()


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """EVA's recorded 1.1.2 run (``record/``, ``eva/``) with synthetic tracks."""
    dest = tmp_path_factory.mktemp("eva_record")
    with tarfile.open(FIXTURES / "eva_record_1.1.2.tar.gz", "r:gz") as tar:
        tar.extractall(dest, filter="data")
    return dest


def _eva_result(bundle: Path) -> dict[str, Any]:
    return json.loads((bundle / "eva" / "result.json").read_text(encoding="utf-8"))


def _call(bundle: Path, attempt: int, reason: str | None = None) -> CallInfo:
    result = _eva_result(bundle)
    return CallInfo(
        attempt=attempt,
        conversation_ended_reason=reason or result["conversation_ended_reason"],
        started_at=datetime.fromisoformat(result["started_at"]),
        ended_at=datetime.fromisoformat(result["ended_at"]),
    )


async def _replay_tools(sidecar: Sidecar) -> list[ToolReply]:
    return [
        await sidecar.tool(c["name"], c["arguments"])
        for c in _fixture("g4_tools.json")["calls"]
    ]


def _call_loop_record(bundle: Path, replies: list[ToolReply], dest: Path) -> Path:
    """The record as harbor's call loop hands it over: EVA's recording, with
    each tool result as ``/tool`` returned it during the call."""
    shutil.copytree(bundle / "record", dest)
    audit_log = json.loads((dest / "audit_log.json").read_text(encoding="utf-8"))
    results = [
        e for e in audit_log["transcript"] if e["message_type"] == "tool_response"
    ]
    for entry, reply in zip(results, replies, strict=True):
        entry["value"]["response"] = reply.result
    (dest / "audit_log.json").write_text(
        json.dumps(audit_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dest


async def _fake_judges(url: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=30) as client:
        return (await client.get(f"{url}/_fake_judges")).json()


async def _graded_run(
    url: str, bundle: Path, dest: Path, **open_kwargs: Any
) -> dict[str, Any]:
    """One trial, as harbor's driver runs it: open, the call's tools, record,
    validate, grade, fetch."""
    sidecar = Sidecar(url)
    try:
        opened = await sidecar.open(domain="airline", record_id=RECORD, **open_kwargs)
        tools = await _replay_tools(sidecar)
        record_dir = _call_loop_record(bundle, tools, dest / "upload")
        recorded = await sidecar.record(record_dir, _call(bundle, opened.attempt))
        try:
            await sidecar.tool("get_reservation", {})
            late_tool: str | None = None
        except SidecarError as exc:
            late_tool = str(exc)
        validation = await sidecar.validate(opened.attempt)
        grade = await sidecar.grade()
        run_dir = await sidecar.fetch(dest / "files")
        return {
            "opened": opened,
            "tools": tools,
            "recorded": recorded,
            "late_tool": late_tool,
            "validation": validation,
            "grade": grade,
            "judges": await _fake_judges(url),
            "run_dir": run_dir,
        }
    finally:
        await sidecar.aclose()


@pytest.fixture(scope="module")
def s2s_run(
    sidecars: dict[str, _Container],
    bundle: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    url = sidecars["s2s"].ready()
    return asyncio.run(_graded_run(url, bundle, tmp_path_factory.mktemp("s2s_files")))


@pytest.fixture(scope="module")
def cascade_run(
    sidecars: dict[str, _Container],
    bundle: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    url = sidecars["cascade"].ready()
    return asyncio.run(
        _graded_run(
            url,
            bundle,
            tmp_path_factory.mktemp("cascade_files"),
            framework="pipecat",
            model=CASCADE_MODEL,
        )
    )


# ── /open renders what EVA sends, for every record ─────────────────────


def test_g8_open_matches_eva_for_all_213_records(
    sidecars: dict[str, _Container],
) -> None:
    url = sidecars["open"].ready()
    golden = _fixture("g8_open.json")["records"]
    assert len(golden) == 213

    async def run() -> list[str]:
        sidecar = Sidecar(url)
        mismatches: list[str] = []
        try:
            for row in golden:
                opened = await sidecar.open(
                    domain=row["domain"], record_id=row["record_id"]
                )
                agent, caller = opened.agent, opened.caller
                assert agent.system_prompt is not None and agent.tools is not None
                got = {
                    "model": opened.raw["agent"]["model"],
                    "system_prompt": _sha(agent.system_prompt),
                    "tools": _sha(_wire(agent.tools)),
                    "tool_count": len(agent.tools),
                    "session": _sha(_wire(agent.session)),
                    "greeting": agent.greeting,
                    "opening_text": agent.opening_text,
                    "caller_prompt": _sha(caller.prompt),
                    "caller_prompt_chars": len(caller.prompt),
                    "gender": caller.gender,
                    "language": caller.language,
                }
                for key, value in got.items():
                    if row[key] != value:
                        mismatches.append(f"{row['domain']}/{row['record_id']} {key}")
        finally:
            await sidecar.aclose()
        return mismatches

    assert asyncio.run(run()) == []


# ── /tool is EVA's executor on EVA's scenario DB ───────────────────────


def test_g4_replayed_tools_give_evas_responses_and_db_hash(
    s2s_run: dict[str, Any],
) -> None:
    golden = _fixture("g4_tools.json")
    replies = s2s_run["tools"]
    assert [r.result for r in replies] == [c["response"] for c in golden["calls"]]
    assert [r.output for r in replies] == [
        json.dumps(c["response"], ensure_ascii=False) for c in golden["calls"]
    ]
    result = s2s_run["recorded"]["result"]
    assert result["initial_scenario_db_hash"] == golden["initial_scenario_db_hash"]
    assert result["final_scenario_db_hash"] == golden["final_scenario_db_hash"]


def test_audit_log_holds_tool_results_as_evas_does(s2s_run: dict[str, Any]) -> None:
    """EVA's audit log serialises tool results when the call ends: a result
    sharing objects with the DB shows the DB as the call left it. The call
    loop uploaded what the model was sent; the record holds EVA's form."""
    golden = _fixture("g4_tools.json")
    record = s2s_run["run_dir"] / "records" / RECORD / "trial_0"
    audit_log = json.loads((record / "audit_log.json").read_text(encoding="utf-8"))
    logged = [
        e["value"]["response"]
        for e in audit_log["transcript"]
        if e["message_type"] == "tool_response"
    ]
    assert logged == [c["logged"] for c in golden["calls"]]
    # The difference is real on this record: a later rebook took a seat.
    assert any(c["logged"] != c["response"] for c in golden["calls"])


def test_tools_are_refused_once_the_call_is_recorded(s2s_run: dict[str, Any]) -> None:
    assert s2s_run["late_tool"] is not None and "409" in s2s_run["late_tool"]


# ── /record: result.json as EVA's worker writes it ─────────────────────────


def test_record_completes_result_json_as_evas_worker(
    s2s_run: dict[str, Any], bundle: Path
) -> None:
    """The fields EVA's worker wrote for this call, and the latency stats its
    own helpers compute from these files (``g1_record.json``)."""
    eva = _eva_result(bundle)
    result = s2s_run["recorded"]["result"]
    for field in (
        "record_id",
        "completed",
        "error",
        "num_turns",
        "num_tool_calls",
        "tools_called",
        "conversation_ended_reason",
        "started_at",
        "ended_at",
        "duration_seconds",
    ):
        assert result[field] == eva[field], field
    for field, value in _fixture("g1_record.json")["latency"].items():
        assert result[field] == value, field
    assert result["audio_user_path"].endswith("/audio_user_clean.wav")
    assert result["time_limit_accepted"] is False


# ── The judges get EVA's prompts, byte for byte ────────────────────────


def test_g3_judge_requests_are_byte_identical_to_evas(s2s_run: dict[str, Any]) -> None:
    golden = _fixture("g3_judges.json")

    def key(request: dict[str, Any]) -> str:
        return json.dumps(request, sort_keys=True)

    assert sorted(map(key, s2s_run["judges"])) == sorted(map(key, golden["requests"]))
    grade: Grade = s2s_run["grade"]
    prompts = {
        name: _sha(score["details"]["judge_prompt"])
        for name, score in grade.metrics.items()
        if "judge_prompt" in (score.get("details") or {})
    }
    assert prompts == golden["judge_prompts"]
    assert sorted(grade.metrics) == golden["metrics"]


def test_validation_then_grade_follow_evas_runner(s2s_run: dict[str, Any]) -> None:
    validation: Validation = s2s_run["validation"]
    assert validation.passed and validation.gate and validation.reason == "passed"
    assert not validation.rerun and validation.restored_attempt is None
    assert set(validation.scores) == {
        "conversation_valid_end",
        "user_behavioral_fidelity",
        "user_speech_fidelity",
    }
    grade: Grade = s2s_run["grade"]
    assert grade.record_id == f"{RECORD}/trial_0"
    assert {
        "EVA-A_pass",
        "EVA-X_pass",
        "EVA-overall_pass",
        "EVA-A_mean",
        "EVA-X_mean",
    } <= set(grade.aggregates)
    # S2S: EVA's cascade-only metrics do not run.
    assert "stt_wer" not in grade.metrics and "speakability" not in grade.metrics


def test_files_hold_a_native_eva_run_directory(s2s_run: dict[str, Any]) -> None:
    run_dir: Path = s2s_run["run_dir"]
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["framework"] == "openai_realtime"
    assert config["model"]["s2s"] == "gpt-realtime-1.5"
    assert config["model"]["s2s_params"]["api_key"] == "***"
    assert config["num_trials"] == 3 and config["domain"] == "airline"
    record = run_dir / "records" / RECORD / "trial_0"
    for name in (
        "result.json",
        "metrics.json",
        "initial_scenario_db.json",
        "final_scenario_db.json",
        "logs.log",
    ):
        assert (record / name).is_file(), name
    summary = json.loads(
        (run_dir / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["simulation"]["successful_record_ids"] == [f"{RECORD}/trial_0"]
    assert (run_dir / "metrics_summary.json").is_file()


# ── EVA's processor golden, through the sidecar ────────────────────────


def test_g1_processor_context_matches_evas_golden(
    cascade_run: dict[str, Any], bundle: Path
) -> None:
    """The processor context the sidecar's ``/grade`` built (``metrics.json``)
    for the record ``/record`` completed, against EVA's processor on the same
    record (``g1_record.json``: EVA's own ``expected_processor_context.json``
    but for the tool result today's data answers differently)."""
    expected = _fixture("g1_record.json")["context"]
    context = cascade_run["grade"].context
    assert context is not None
    assert context["pipeline_type"] == expected["pipeline_type"] == "cascade"
    for field, value in expected.items():
        if field in ("pipeline_type", "tool_called"):
            continue  # tool_called is the processor's own; metrics.json keeps tool_params
        if field.endswith("interrupted_turns"):
            assert sorted(context[field]) == value, field
        else:
            assert context[field] == value, field


# ── Attempts: EVA's rerun loop and its time-limit rule ─────────────────────


def _unfinished(record_dir: Path) -> Path:
    """The record with the caller's goodbye cut off: EVA's gate fails it."""
    events = record_dir / "elevenlabs_events.jsonl"
    lines = events.read_text(encoding="utf-8").splitlines(keepends=True)
    events.write_text("".join(lines[:-1]), encoding="utf-8")
    return record_dir


def test_reruns_archive_attempts_and_accept_at_the_time_limit(
    sidecars: dict[str, _Container], bundle: Path, tmp_path: Path
) -> None:
    url = sidecars["attempts"].ready()

    async def run() -> dict[str, Any]:
        sidecar = Sidecar(url)
        try:
            first = await sidecar.open(domain="airline", record_id=RECORD)
            replies = await _replay_tools(sidecar)
            record_dir = _unfinished(
                _call_loop_record(bundle, replies, tmp_path / "attempt1")
            )
            await sidecar.record(record_dir, _call(bundle, 1))
            not_finished = await sidecar.validate(1)
            second = await sidecar.open(domain="airline", record_id=RECORD)
            replies = await _replay_tools(sidecar)
            record_dir = _unfinished(
                _call_loop_record(bundle, replies, tmp_path / "attempt2")
            )
            await sidecar.record(
                record_dir, _call(bundle, 2, reason="time_limit_exceeded")
            )
            accepted = await sidecar.validate(2)
            try:
                await sidecar.open(domain="airline", record_id=RECORD)
                reopened: str | None = None
            except SidecarError as exc:
                reopened = str(exc)
            grade = await sidecar.grade()
            files = await sidecar.fetch(tmp_path / "files")
            return {
                "attempts": (first.attempt, second.attempt),
                "not_finished": not_finished,
                "accepted": accepted,
                "reopened": reopened,
                "grade": grade,
                "files": files,
            }
        finally:
            await sidecar.aclose()

    out = asyncio.run(run())
    assert out["attempts"] == (1, 2)
    first: Validation = out["not_finished"]
    assert (first.passed, first.gate, first.reason, first.rerun) == (
        False,
        False,
        "not_finished",
        True,
    )
    second: Validation = out["accepted"]
    assert (second.passed, second.gate, second.reason, second.rerun) == (
        True,
        False,
        "time_limit_accepted",
        False,
    )
    assert second.raw["rerun_history"] == [
        {"attempt": 1, "reason": "not_finished"},
        {"attempt": 2, "reason": "time_limit_exceeded"},
    ]
    assert out["reopened"] is not None and "409" in out["reopened"]
    assert out["grade"].record_id == f"{RECORD}/trial_0"
    records = out["files"] / "records" / RECORD
    assert (records / "trial_0_failed_attempt_1" / "result.json").is_file()
    result = json.loads(
        (records / "trial_0" / "result.json").read_text(encoding="utf-8")
    )
    assert result["time_limit_accepted"] is True
    assert result["conversation_ended_reason"] == "time_limit_exceeded"
    summary = json.loads(
        (out["files"] / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["simulation"]["time_limit_accepted_record_ids"] == [
        f"{RECORD}/trial_0"
    ]


# ── EVA's aggregation over trials graded one by one ─────────────────────


def test_g6_aggregation_over_per_trial_runs_matches_evas(
    image: str, tmp_path: Path
) -> None:
    """Each trial's metrics in its own run directory, as ``/files`` returns
    them; ``aggregate.py`` must give EVA's ``--aggregate-only`` summary.

    EVA lists records unsorted, and its bootstrap CIs follow that order, so
    aggregate.py sorts them. The golden's CI bounds are EVA's own code run
    in sorted order (the same on x86 and ARM); every other number is EVA's
    original output."""
    golden = _fixture("g6_aggregation.json")
    sources = []
    for n, (oid, metrics) in enumerate(sorted(golden["inputs"].items())):
        path = tmp_path / f"trial-run-{n}" / "records" / oid / "metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(metrics), encoding="utf-8")
        sources.append(f"/in/trial-run-{n}")
    out = _docker(
        "run",
        "--rm",
        "-v",
        f"{tmp_path}:/in:ro",
        "--entrypoint",
        "/opt/venv/bin/python",
        image,
        "/opt/eva-sidecar/aggregate.py",
        "--run-id",
        golden["run_id"],
        "--num-draws",
        str(golden["num_draws"]),
        *sources,
    )
    assert json.loads(out.stdout) == golden["summary"]


def test_g6_composites_per_record_are_evas(s2s_run: dict[str, Any]) -> None:
    """``/grade``'s composites are EVA's rule on its own metrics: recomputed
    here from EVA_COMPOSITES' published thresholds, they must agree."""
    grade: Grade = s2s_run["grade"]

    def value(name: str) -> float | None:
        score = grade.metrics.get(name)
        if score is None or score.get("error"):
            return None
        return (
            score["normalized_score"]
            if score["normalized_score"] is not None
            else score["score"]
        )

    a = [
        value("task_completion"),
        value("faithfulness"),
        value("agent_speech_fidelity"),
    ]
    if None not in a:
        expected = float(a[0] == 1.0 and a[1] >= 0.5 and a[2] >= 0.95)
        assert grade.aggregates["EVA-A_pass"] == expected


def test_record_refuses_files_only_evas_code_writes(
    sidecars: dict[str, _Container], bundle: Path
) -> None:
    """A fresh session cannot be handed result.json: EVA's worker writes it."""
    url = sidecars["open"].ready()

    async def run() -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            sidecar = Sidecar(url, client=client)
            opened = await sidecar.open(domain="airline", record_id="1.2.1")
            body = io.BytesIO()
            with tarfile.open(fileobj=body, mode="w") as tar:
                for name, data in (
                    ("result.json", b"{}"),
                    (
                        "call.json",
                        json.dumps(_call(bundle, opened.attempt).to_json()).encode(),
                    ),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            response = await client.post(f"{url}/record", content=body.getvalue())
            return f"{response.status_code} {response.text}"

    assert asyncio.run(run()).startswith("422")
