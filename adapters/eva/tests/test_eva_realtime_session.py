"""The Realtime-session golden: harbor's ``openai.realtime`` with EVA's settings against EVA's own wire.

The fixture is EVA's ``OpenAIRealtimeAssistantServer`` run in EVA's image
(``scripts/goldens/realtime_session.py``): the ``session.update`` it sends,
its greeting, what it sends when one response holds two tool calls, and what
it makes of ``response.done`` payloads. harbor's transport, given the same
prompt and tools and EVA's settings, must send the same events with the same
contents, and report responses as EVA reads them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from harbor_voice.ids import PartyId
from harbor_voice.providers import MappingExecutor
from harbor_voice.providers import ToolRouter
from harbor_voice.providers import ToolSpec
from harbor_voice.providers.openai.realtime import OpenAIRealtimeTransport
from harbor_voice.providers.openai.realtime import RealtimeSettings
from harbor_voice.providers.openai.realtime import response_has_function_calls
from harbor_voice.providers.openai.realtime import response_text
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.stream import ResponseDone
from harbor_voice.transport.stream import ToolCall

pytestmark = pytest.mark.unit

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "realtime" / "eva_session.json").read_text(
        encoding="utf-8"
    )
)
SESSION = GOLDEN["session"]

EVA = RealtimeSettings(
    voice="marin",
    vad_threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=200,
    transcription_model="whisper-1",
    language=None,
    noise_reduction=None,
    tool_choice=None,
    audio_format="pcm24k",
    opening_text=f"Say: '{SESSION['initial_message']}'",
    tool_flow="serial",
    tool_output_ascii=False,
    extended_events=True,
)
"""EVA's session as harbor settings: ``_build_session_config`` with
``s2s_params`` holding only the model (EVA's gpt-realtime-1.5 run), and
``_handle_session``'s greeting."""


def _results() -> dict[str, dict[str, Any]]:
    """What EVA's tools returned, by tool name, from the outputs it sent."""
    calls = {c["call_id"]: c["name"] for c in GOLDEN["script"] if "call_id" in c}
    return {
        calls[e["item"]["call_id"]]: json.loads(e["item"]["output"])
        for e in GOLDEN["wire"]
        if e.get("item", {}).get("type") == "function_call_output"
    }


def _router(seen: list[tuple[str, dict[str, Any]]]) -> ToolRouter:
    results = _results()

    def handler(name: str):
        def run(**arguments: Any) -> dict[str, Any]:
            seen.append((name, arguments))
            return results[name]

        return run

    specs = [
        ToolSpec(
            name=tool["name"],
            description=tool["description"],
            input_schema=tool["parameters"],
        )
        for tool in SESSION["tools"]
    ]
    return ToolRouter(
        MappingExecutor(specs=specs, handlers={s.name: handler(s.name) for s in specs}),
        sleep=lambda _s: asyncio.sleep(0),
    )


class ScriptedServer:
    """Replays EVA's script to harbor's transport; records what it sends."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.inbox: asyncio.Queue[str | None] = asyncio.Queue()
        for event in script:
            self.inbox.put_nowait(json.dumps(event))
        self.sent: list[dict[str, Any]] = []

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def recv(self) -> str:
        raw = await self.inbox.get()
        assert raw is not None
        return raw

    def __aiter__(self) -> ScriptedServer:
        return self

    async def __anext__(self) -> str:
        raw = await self.inbox.get()
        if raw is None:
            raise StopAsyncIteration
        return raw

    async def close(self) -> None:
        self.inbox.put_nowait(None)


def _transport(
    server: ScriptedServer | None = None,
    seen: list[tuple[str, dict[str, Any]]] | None = None,
    settings: RealtimeSettings = EVA,
) -> OpenAIRealtimeTransport:
    async def connect(url: str, *, additional_headers: dict[str, str]) -> Any:
        assert url.endswith(f"?model={GOLDEN['session']['model']}")
        return server

    return OpenAIRealtimeTransport(
        party=PartyId("agent"),
        model=SESSION["model"],
        system_prompt=SESSION["instructions"],
        tools=_router(seen if seen is not None else []),
        settings=settings,
        api_key="k",
        connect=connect,
    )


async def _run(settings: RealtimeSettings = EVA):
    server = ScriptedServer(GOLDEN["script"])
    seen: list[tuple[str, dict[str, Any]]] = []
    transport = _transport(server, seen, settings)
    await transport.open(SessionConfig(party=PartyId("agent")))
    for _ in range(50):
        await asyncio.sleep(0)
    await transport.close()
    events = [e async for e in transport.events()]
    return server, seen, events, transport


def test_the_session_update_is_evas_field_for_field() -> None:
    ours = _transport().session_update()
    assert ours == GOLDEN["wire"][0]
    # EVA's SDK put _build_session_config() on the wire untouched.
    assert ours == {"type": "session.update", "session": SESSION["session_config"]}


async def test_the_call_sends_what_eva_sends_in_evas_order() -> None:
    """Greeting, then one ``response.create`` per tool output — two calls in
    one response get two continuations, as EVA sends them."""
    server, seen, events, transport = await _run()
    assert server.sent == GOLDEN["wire"]
    assert seen == [
        (call["name"], json.loads(call["arguments"]))
        for call in GOLDEN["script"]
        if call["type"] == "response.function_call_arguments.done"
    ]
    calls = [e for e in events if isinstance(e, ToolCall)]
    (done,) = [e for e in events if isinstance(e, ResponseDone)]
    assert [c.call_id for c in calls] == ["call_golden_1", "call_golden_2"]
    assert events.index(calls[-1]) < events.index(done)  # tools ran before it
    assert (done.status, done.has_function_calls, done.text) == ("completed", True, "")
    assert (done.input_tokens, done.output_tokens) == (1200, 100)
    assert transport.errors == []


async def test_tau2s_tool_flow_would_not_be_evas() -> None:
    """Why ``tool_flow`` exists: batched, the two outputs share one
    continuation, and EVA's wire is not reproduced."""
    server, *_ = await _run(dataclasses.replace(EVA, tool_flow="batch"))
    kinds = [e["type"] for e in server.sent]
    assert kinds.count("response.create") == 2  # the greeting's and one more
    assert server.sent != GOLDEN["wire"]


@pytest.mark.parametrize("case", sorted(GOLDEN["response_done"]))
def test_responses_read_as_eva_reads_them(case: str) -> None:
    golden = GOLDEN["response_done"][case]
    response = golden["event"].get("response") or {}
    assert response_has_function_calls(response) is golden["has_function_calls"]
    assert response_text(response) == golden["text"]


@pytest.mark.parametrize("case", sorted(GOLDEN["response_done"]))
def test_the_response_done_event_carries_evas_reading(case: str) -> None:
    golden = GOLDEN["response_done"][case]
    transport = _transport()
    transport._dispatch(golden["event"])
    event = transport._inbox.get_nowait()
    assert isinstance(event, ResponseDone)
    assert (event.status, event.has_function_calls, event.text) == (
        golden["status"] or "",
        golden["has_function_calls"],
        golden["text"],
    )
