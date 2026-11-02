"""The sidecar client: request shapes, the /record tar, replies, errors.

Against a fake sidecar (``httpx.MockTransport``); the real one is exercised
in ``test_eva_sidecar_goldens.py``.
"""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from eva_adapter.client import CALL_FILE
from eva_adapter.client import SIDECAR_API
from eva_adapter.client import CallInfo
from eva_adapter.client import Sidecar
from eva_adapter.client import SidecarError
from eva_adapter.client import SidecarMismatch
from eva_adapter.client import pack_record
from eva_adapter.client import unpack

OPENED = {
    "conversation_id": "1.1.2/trial_0",
    "output_id": "1.1.2/trial_0",
    "run_id": "run",
    "domain": "airline",
    "record_id": "1.1.2",
    "trial": 0,
    "attempt": 1,
    "max_attempts": 3,
    "time_limit_s": 600,
    "caller_timeout_s": 620,
    "agent": {
        "system_prompt": "You are an AI voice assistant",
        "tools": [
            {
                "type": "function",
                "name": "get_reservation",
                "description": "d",
                "parameters": {},
            }
        ],
        "greeting": "Hello! How can I help you today?",
        "opening_text": "Say: 'Hello! How can I help you today?'",
        "session": {"type": "realtime"},
        "model": "gpt-realtime-1.5",
    },
    "caller": {"prompt": "You are a passenger", "gender": "F", "language": "en"},
}

CALL = CallInfo(
    attempt=1,
    conversation_ended_reason="goodbye",
    started_at=datetime(2026, 3, 1, 17, 22, 3, 569036),
    ended_at=datetime(2026, 3, 1, 17, 27, 47, 827190),
)


class FakeSidecar:
    """Records each request; answers from a route table."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if (request.method, request.url.path) == ("GET", "/health"):
            # Not recorded: every session checks the API before its first open.
            answer = self.routes.get(
                "GET /health", {"status": "ok", "api": SIDECAR_API}
            )
            return httpx.Response(200, json=answer)
        self.requests.append(request)
        answer = self.routes.get(f"{request.method} {request.url.path}")
        if answer is None:
            return httpx.Response(404, json={"detail": "no route"})
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)


def _sidecar(fake: FakeSidecar) -> Sidecar:
    return Sidecar(
        "http://sidecar:8000/",
        client=httpx.AsyncClient(transport=httpx.MockTransport(fake)),
    )


async def test_open_sends_the_trial_and_parses_evas_setup() -> None:
    fake = FakeSidecar({"POST /open": OPENED})
    opened = await _sidecar(fake).open(domain="airline", record_id="1.1.2", trial=2)
    assert json.loads(fake.requests[0].content) == {
        "domain": "airline",
        "record_id": "1.1.2",
        "language": "en",
        "trial": 2,
    }
    assert opened.attempt == 1 and opened.max_attempts == 3
    assert opened.agent.opening_text == "Say: 'Hello! How can I help you today?'"
    assert opened.agent.tools == OPENED["agent"]["tools"]
    assert opened.caller.gender == "F" and opened.caller.prompt == "You are a passenger"
    assert opened.caller_timeout_s == 620.0


async def test_open_passes_an_eva_run_config_override() -> None:
    fake = FakeSidecar({"POST /open": OPENED})
    model = {"llm": "gpt-5.2"}
    await _sidecar(fake).open(
        domain="itsm", record_id="1", framework="pipecat", model=model, num_trials=1
    )
    body = json.loads(fake.requests[0].content)
    assert (
        body["framework"] == "pipecat"
        and body["model"] == model
        and body["num_trials"] == 1
    )


async def test_open_refuses_an_unknown_gender() -> None:
    bad = {**OPENED, "caller": {**OPENED["caller"], "gender": "X"}}
    with pytest.raises(SidecarError, match="gender"):
        await _sidecar(FakeSidecar({"POST /open": bad})).open(
            domain="airline", record_id="1.1.2"
        )


async def test_tool_returns_evas_result_and_the_model_output() -> None:
    result = {"status": "success", "seat": "12A"}
    fake = FakeSidecar({"POST /tool": {"result": result, "output": json.dumps(result)}})
    reply = await _sidecar(fake).tool("assign_seat", {"seat": "12A"})
    assert json.loads(fake.requests[0].content) == {
        "name": "assign_seat",
        "arguments": {"seat": "12A"},
    }
    assert reply.result == result and json.loads(reply.output) == result


def _members(data: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        out = {}
        for member in tar.getmembers():
            handle = tar.extractfile(member)
            if handle is not None:
                out[member.name] = handle.read()
        return out


def test_pack_record_carries_every_file_and_call_json(tmp_path: Path) -> None:
    (tmp_path / "audit_log.json").write_text("{}")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "x.wav").write_bytes(b"RIFF")
    members = _members(pack_record(tmp_path, CALL))
    assert set(members) == {"audit_log.json", "nested/x.wav", CALL_FILE}
    assert json.loads(members[CALL_FILE]) == {
        "attempt": 1,
        "conversation_ended_reason": "goodbye",
        "started_at": "2026-03-01T17:22:03.569036",
        "ended_at": "2026-03-01T17:27:47.827190",
        "error": None,
    }


def test_pack_record_refuses_a_call_json_on_disk(tmp_path: Path) -> None:
    (tmp_path / CALL_FILE).write_text("{}")
    with pytest.raises(ValueError, match=CALL_FILE):
        pack_record(tmp_path, CALL)


async def test_record_posts_the_tar(tmp_path: Path) -> None:
    (tmp_path / "audit_log.json").write_text("{}")
    fake = FakeSidecar(
        {"POST /record": {"record_dir": "/r", "attempt": 1, "result": {}}}
    )
    reply = await _sidecar(fake).record(tmp_path, CALL)
    request = fake.requests[0]
    assert request.headers["content-type"] == "application/x-tar"
    assert set(_members(request.content)) == {"audit_log.json", CALL_FILE}
    assert reply["record_dir"] == "/r"


async def test_validate_parses_the_rerun_decision() -> None:
    answer = {
        "attempt": 1,
        "passed": False,
        "gate": False,
        "reason": "not_finished",
        "scores": {},
        "failed_metrics": [],
        "rerun": True,
        "restored_attempt": None,
        "rerun_history": [{"attempt": 1, "reason": "not_finished"}],
    }
    fake = FakeSidecar({"POST /validate": answer})
    validation = await _sidecar(fake).validate(1)
    assert json.loads(fake.requests[0].content) == {"attempt": 1}
    assert (
        validation.passed,
        validation.gate,
        validation.reason,
        validation.rerun,
    ) == (
        False,
        False,
        "not_finished",
        True,
    )
    assert validation.raw["rerun_history"] == answer["rerun_history"]


async def test_grade_returns_metrics_and_composites() -> None:
    answer = {
        "record_id": "1.1.2/trial_0",
        "metrics": {"task_completion": {"score": 1.0}},
        "aggregates": {"EVA-A_pass": 1.0, "EVA-X_pass": None},
        "context": {"num_turns": 14},
    }
    grade = await _sidecar(FakeSidecar({"POST /grade": answer})).grade()
    assert grade.aggregates == {"EVA-A_pass": 1.0, "EVA-X_pass": None}
    assert grade.metrics["task_completion"]["score"] == 1.0
    assert grade.context == {"num_turns": 14}


async def test_refusals_and_outages_raise_sidecar_error() -> None:
    refused = FakeSidecar(
        {"POST /grade": httpx.Response(409, json={"detail": "nothing to grade"})}
    )
    with pytest.raises(SidecarError, match="409.*nothing to grade"):
        await _sidecar(refused).grade()

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    sidecar = Sidecar(
        "http://sidecar:8000",
        client=httpx.AsyncClient(transport=httpx.MockTransport(down)),
    )
    with pytest.raises(SidecarError, match="unreachable"):
        await sidecar.tool("x", {})
    assert await sidecar.health() is False


async def test_fetch_extracts_the_run_directory(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        data = b'{"framework": "openai_realtime"}'
        info = tarfile.TarInfo("config.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    fake = FakeSidecar({"GET /files": httpx.Response(200, content=buffer.getvalue())})
    dest = await _sidecar(fake).fetch(tmp_path / "run")
    assert (
        json.loads((dest / "config.json").read_text())["framework"] == "openai_realtime"
    )


def test_unpack_refuses_paths_that_leave_the_destination(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(tarfile.TarError):
        unpack(buffer.getvalue(), tmp_path / "dest")
    assert not (tmp_path / "escape.txt").exists()


async def test_a_sidecar_of_another_api_stops_the_trial_before_open() -> None:
    for health in ({"status": "ok", "api": SIDECAR_API + 1}, {"status": "ok"}):
        fake = FakeSidecar({"GET /health": health, "POST /open": OPENED})
        with pytest.raises(SidecarMismatch, match="reconvert") as raised:
            await _sidecar(fake).open(domain="airline", record_id="1.1.2")
        assert fake.requests == []  # /open never sent
    assert "predates the check" in str(raised.value)


async def test_the_api_is_checked_once_per_session() -> None:
    seen: list[str] = []
    fake = FakeSidecar({"POST /open": OPENED})

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return fake(request)

    sidecar = Sidecar(
        "http://sidecar:8000",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await sidecar.open(domain="airline", record_id="1.1.2")
    await sidecar.open(domain="airline", record_id="1.1.2")  # EVA's rerun
    assert seen == ["GET /health", "POST /open", "POST /open"]
