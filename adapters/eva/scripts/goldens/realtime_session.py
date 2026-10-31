"""The Realtime-session golden: EVA's OpenAI Realtime session, as EVA puts it on the wire.

EVA's own ``OpenAIRealtimeAssistantServer`` is built for one airline record
(1.1.2, English, ``gpt-realtime-1.5``, EVA's defaults) the way EVA's worker
builds it, and its ``_handle_session`` runs against a fake socket under the
OpenAI SDK EVA pins. Recorded:

- ``session``: ``_build_session_config()`` and its parts (prompt, tools,
  greeting), for building harbor's side from the same inputs;
- ``wire``: every client event EVA sent, as JSON on the socket — the
  ``session.update``, the greeting item and its ``response.create``, then
  what EVA sends when the server reports two tool calls in one response
  (``script``), with the tools really run on the record's scenario DB;
- ``response_done``: what ``_response_has_function_calls`` and
  ``_extract_response_text`` return for ``response.done`` payloads of
  several shapes, each parsed by the SDK exactly as EVA receives it.

``tests/adapters/eva/test_eva_realtime_session.py`` checks harbor's
``openai.realtime`` transport against it; harbor's tests never import EVA.
Regenerate inside EVA's image::

    docker run --rm -v "$PWD/adapters/eva:/adapter" \\
        -v "$PWD/tests/adapters/eva/fixtures:/fixtures" \\
        --entrypoint /opt/venv/bin/python eva-base:039efa0a \\
        /adapter/scripts/goldens/realtime_session.py \\
        /fixtures/realtime/eva_session.json
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import openai
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.resources.realtime.realtime import AsyncRealtimeConnectionManager

from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer
from eva.models.agents import AgentConfig
from eva.models.config import ModelConfig
from eva.models.record import EvaluationRecord
from eva.utils.culture import resolve_scenario_db

DOMAIN = "airline"
RECORD = "1.1.2"
LANGUAGE = "en"
MODEL = "gpt-realtime-1.5"

CALLS = [
    {
        "call_id": "call_golden_1",
        "item_id": "item_golden_1",
        "name": "get_reservation",
        "arguments": json.dumps(
            {"confirmation_number": "ZK3FFW", "last_name": "Rodriguez"}
        ),
    },
    {
        "call_id": "call_golden_2",
        "item_id": "item_golden_2",
        "name": "get_flight_status",
        "arguments": json.dumps(
            {"flight_number": "SK621", "flight_date": "2026-03-20"}
        ),
    },
]

SCRIPT: list[dict[str, Any]] = [
    {
        "type": "session.created",
        "event_id": "event_1",
        "session": {"id": "sess_golden"},
    },
    {
        "type": "session.updated",
        "event_id": "event_2",
        "session": {"id": "sess_golden"},
    },
    {
        "type": "response.created",
        "event_id": "event_3",
        "response": {
            "id": "resp_golden",
            "object": "realtime.response",
            "status": "in_progress",
        },
    },
    *(
        {
            "type": "response.function_call_arguments.done",
            "event_id": f"event_{4 + i}",
            "response_id": "resp_golden",
            "item_id": call["item_id"],
            "output_index": i,
            "call_id": call["call_id"],
            "name": call["name"],
            "arguments": call["arguments"],
        }
        for i, call in enumerate(CALLS)
    ),
    {
        "type": "response.done",
        "event_id": "event_6",
        "response": {
            "id": "resp_golden",
            "object": "realtime.response",
            "status": "completed",
            "output": [
                {
                    "id": call["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                }
                for call in CALLS
            ],
            "usage": {"total_tokens": 1300, "input_tokens": 1200, "output_tokens": 100},
        },
    },
]
"""What the fake server sends, in order, after the session opens."""


def _message(*parts: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "item_msg",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": list(parts),
    }


def _call_item(n: int) -> dict[str, Any]:
    return {
        "id": f"item_call_{n}",
        "type": "function_call",
        "status": "completed",
        "call_id": f"call_{n}",
        "name": "get_reservation",
        "arguments": "{}",
    }


RESPONSE_DONE: dict[str, dict[str, Any]] = {
    "ga_audio": {
        "id": "r1",
        "status": "completed",
        "output": [
            _message(
                {
                    "type": "output_audio",
                    "transcript": "Hello! How can I help you today?",
                }
            )
        ],
    },
    "ga_text": {
        "id": "r2",
        "status": "completed",
        "output": [_message({"type": "output_text", "text": "Hello there."})],
    },
    "beta_audio": {
        "id": "r3",
        "status": "completed",
        "output": [_message({"type": "audio", "transcript": "  Hello there.  "})],
    },
    "beta_text_and_audio": {
        "id": "r4",
        "status": "completed",
        "output": [
            _message(
                {"type": "text", "text": "One. "},
                {"type": "audio", "transcript": None, "text": "Two."},
            )
        ],
    },
    "function_calls": {
        "id": "r5",
        "status": "completed",
        "output": [_call_item(1), _call_item(2)],
    },
    "speech_then_call": {
        "id": "r6",
        "status": "completed",
        "output": [
            _message({"type": "audio", "transcript": "Let me check."}),
            _call_item(3),
        ],
    },
    "cancelled": {
        "id": "r7",
        "status": "cancelled",
        "status_details": {"type": "cancelled", "reason": "turn_detected"},
        "output": [_message({"type": "output_audio", "transcript": "I can help with"})],
    },
    "no_output": {"id": "r8", "status": "failed", "output": []},
}


class FakeSocket:
    """The websocket the SDK talks to: replays ``SCRIPT``, records sends."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.sent: list[dict[str, Any]] = []
        self.drained = asyncio.Event()

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def recv(self, decode: bool = False) -> bytes:
        if self.script:
            return json.dumps(self.script.pop(0)).encode()
        self.drained.set()  # everything before this was fully handled
        await asyncio.Event().wait()  # then idle until cancelled
        raise AssertionError("unreachable")

    async def close(self, code: int = 1000, reason: str = "") -> None:
        return None


class FakeTwilio:
    """The caller's side of EVA's server: stops once the script is handled."""

    def __init__(self, upstream: FakeSocket) -> None:
        self.upstream = upstream

    async def receive_text(self) -> str:
        await self.upstream.drained.wait()
        return json.dumps({"event": "stop"})

    async def send_text(self, text: str) -> None:
        return None


def _server(workdir: Path) -> OpenAIRealtimeAssistantServer:
    """As ``ConversationWorker._start_assistant`` builds it."""
    records = EvaluationRecord.load_dataset(f"data/{DOMAIN}_dataset.json")
    record = next(r for r in records if r.id == RECORD)
    db = json.loads(
        Path(f"data/{DOMAIN}_scenarios/{RECORD}.json").read_text(encoding="utf-8")
    )
    resolved = resolve_scenario_db(
        db,
        record.culture_overrides,
        LANGUAGE,
        record.romanized_culture_overrides,
        aliases_dir=Path(f"data/{DOMAIN}_aliases"),
    )
    db_path = workdir / "scenario_db.json"
    db_path.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return OpenAIRealtimeAssistantServer(
        current_date_time=record.current_date_time,
        pipeline_config=ModelConfig(
            s2s=MODEL,
            s2s_params={"model": MODEL, "api_key": "sk-golden-never-used"},
        ),
        agent=AgentConfig.from_yaml(f"configs/agents/{DOMAIN}_agent.yaml"),
        agent_config_path=f"configs/agents/{DOMAIN}_agent.yaml",
        scenario_db_path=str(db_path),
        output_dir=workdir,
        port=0,
        conversation_id=record.id,
        language=LANGUAGE,
    )


async def _wire(server: OpenAIRealtimeAssistantServer) -> list[dict[str, Any]]:
    socket = FakeSocket(SCRIPT)

    async def connect(self: Any, extra_query: Any, extra_headers: Any) -> FakeSocket:
        return socket

    AsyncRealtimeConnectionManager._connect_ws = connect  # type: ignore[method-assign]
    await asyncio.wait_for(server._handle_session(FakeTwilio(socket)), timeout=60)
    if not socket.drained.is_set():
        raise SystemExit("EVA's session ended before handling the whole script")
    return socket.sent


def _response_done() -> dict[str, Any]:
    parse = AsyncRealtimeConnection(FakeSocket([])).parse_event  # type: ignore[arg-type]
    cases: dict[str, Any] = {}
    for name, response in {**RESPONSE_DONE, "no_response": None}.items():
        raw = {"type": "response.done", "event_id": f"event_{name}"}
        if response is not None:
            raw["response"] = response
        event = parse(json.dumps(raw))
        cases[name] = {
            "event": raw,
            "status": getattr(getattr(event, "response", None), "status", None),
            "has_function_calls": OpenAIRealtimeAssistantServer._response_has_function_calls(
                event
            ),
            "text": OpenAIRealtimeAssistantServer._extract_response_text(event),
        }
    return cases


def main(out: Path) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="eva-g7-"))
    server = _server(workdir)
    session = {
        "model": MODEL,
        "record": RECORD,
        "domain": DOMAIN,
        "language": LANGUAGE,
        "instructions": server._system_prompt,
        "tools": server._realtime_tools,
        "initial_message": server.initial_message,
        "session_config": server._build_session_config(),
    }
    wire = asyncio.run(_wire(server))
    fixture = {
        "generated_by": "adapters/eva/scripts/goldens/realtime_session.py",
        "eva_commit": "039efa0a",
        "openai_sdk": openai.__version__,
        "session": session,
        "script": SCRIPT,
        "wire": wire,
        "response_done": _response_done(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    kinds = [e["type"] for e in wire]
    print(f"wrote {out}: {len(wire)} client events {kinds}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
