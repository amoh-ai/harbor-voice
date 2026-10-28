"""OpenAI Live's transport against a fake connection: tau2's session, events
in arrival order, tool batches continued only when whole, fatal errors."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from harbor_voice.audio import g711
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import OptionalOp
from harbor_voice.ids import PartyId
from harbor_voice.providers import MappingExecutor
from harbor_voice.providers import ToolRouter
from harbor_voice.providers import ToolSpec
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.openai.live import LIVE_AUDIO
from harbor_voice.providers.openai.live import LiveSettings
from harbor_voice.providers.openai.live import OpenAILiveTransport
from harbor_voice.providers.openai.live import build
from harbor_voice.providers.openai.realtime import TELEPHONY
from harbor_voice.providers.openai import register_all
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript

pytestmark = pytest.mark.unit

SPEC = ToolSpec(
    name="get_user",
    description="Look a user up",
    input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
)


class FakeConnection:
    def __init__(self, on_event, on_audio, on_error, start_events) -> None:
        self.on_event = on_event
        self.on_audio = on_audio
        self.on_error = on_error
        self.start_events = start_events
        self.sent: list[dict[str, Any]] = []
        self.audio: list[bytes] = []
        self.started_with: dict[str, Any] = {}
        self.closed = False

    async def start(self, session, *, api_key, url):
        self.started_with = {"session": session, "api_key": api_key, "url": url}
        for event in self.start_events:
            self.on_event(event)
        return {"id": "sess_1"}

    def send(self, event):
        self.sent.append(event)
        if event["type"] == "session.close":
            self.on_event({"type": "session.closed"})

    def append(self, pcm):
        self.audio.append(pcm)

    async def close(self):
        self.closed = True


def _transport(
    *,
    tools: ToolRouter | None = None,
    start_events=({"type": "session.started"},),
    **settings: Any,
) -> tuple[OpenAILiveTransport, list[FakeConnection]]:
    made: list[FakeConnection] = []

    def connect(on_event, on_audio, on_error):
        made.append(FakeConnection(on_event, on_audio, on_error, start_events))
        return made[-1]

    transport = OpenAILiveTransport(
        party=PartyId("agent"),
        settings=LiveSettings(backend_model="gpt-6-luna", **settings),
        system_prompt="You are an airline agent.",
        tools=tools,
        api_key="k",
        connect=connect,
    )
    return transport, made


def _router(handler=None) -> ToolRouter:
    def get_user(id: str = "u1") -> dict[str, str]:
        return {"id": id}

    return ToolRouter(
        MappingExecutor(specs=[SPEC], handlers={"get_user": handler or get_user}),
        sleep=lambda _s: asyncio.sleep(0),
    )


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def _call(delegation: str, call_id: str) -> dict[str, Any]:
    return {
        "type": "response.event",
        "delegation_id": delegation,
        "event": {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": call_id,
                "name": "get_user",
                "arguments": '{"id": "u7"}',
            },
        },
    }


def _created(delegation: str, response: str) -> dict[str, Any]:
    return {
        "type": "response.event",
        "delegation_id": delegation,
        "event": {"type": "response.created", "response": {"id": response}},
    }


def _completed(delegation: str, response: str, usage=None) -> dict[str, Any]:
    return {
        "type": "response.event",
        "delegation_id": delegation,
        "event": {
            "type": "response.completed",
            "response": {"id": response, "usage": usage or {}},
        },
    }


def test_the_session_is_tau2s() -> None:
    transport, _ = _transport(tools=_router())
    assert transport.session_start() == {
        "model": "gpt-live-1",
        "instructions": "You are an airline agent.",
        "audio": {"output": {"voice": "marin"}},
        "delegation": {
            "type": "responses",
            "responses": {
                "model": "gpt-6-luna",
                "instructions": "You are an airline agent.",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_user",
                        "description": "Look a user up",
                        "parameters": SPEC.input_schema,
                    }
                ],
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            },
        },
    }
    other, _ = _transport(backend_instructions="Backend.", reasoning_effort="low")
    responses = other.session_start()["delegation"]["responses"]
    assert responses["instructions"] == "Backend."
    assert responses["reasoning"] == {"effort": "low"}


def test_it_streams_events_but_never_truncates() -> None:
    transport, _ = _transport()
    ops = transport.capabilities.optional_ops
    assert OptionalOp.EVENT_STREAM in ops and OptionalOp.TRUNCATE not in ops
    assert transport.capabilities.audio_in == LIVE_AUDIO


async def test_open_waits_for_the_session_to_start() -> None:
    transport, made = _transport()
    await transport.open(SessionConfig(party=PartyId("agent")))
    assert transport.is_open and transport.session_id == "sess_1"
    assert made[0].started_with["api_key"] == "k"
    assert made[0].started_with["url"] == "https://api.openai.com/v1/live/sessions"


async def test_an_error_before_start_fails_open_at_once() -> None:
    transport, made = _transport(
        start_events=(
            {
                "type": "error",
                "error": {"code": "live_api_access_denied", "message": "no access"},
            },
        )
    )
    with pytest.raises(ProviderError, match="live_api_access_denied"):
        await asyncio.wait_for(transport.open(SessionConfig(party=PartyId("a"))), 1)
    assert made[0].closed and not transport.is_open


async def test_events_arrive_in_order_with_transcripts_on_the_audio_timeline() -> None:
    transport, made = _transport()
    await transport.open(SessionConfig(party=PartyId("agent")))
    conn = made[0]
    conn.on_audio(b"\x01\x00" * 100)
    conn.on_event({"type": "session.output_transcript.delta", "delta": "Hello"})
    conn.on_audio(b"\x01\x00" * 50)
    conn.on_event(
        {"type": "session.input_transcript.delta", "delta": "hi", "start_ms": 40}
    )
    conn.on_event({"type": "session.output_transcript.delta", "delta": " there"})
    conn.on_event({"type": "session.delegation.created"})
    await transport.close()

    events = [e async for e in transport.events()]
    assert [type(e) for e in events] == [
        OutputAudio,
        OutputTranscript,
        OutputAudio,
        InputSpeech,
        OutputTranscript,
    ]
    assert events[1].audio_position == 200 and events[1].utterance is None
    assert events[4].audio_position == 300
    assert events[3].started is True
    assert transport.delegations == 1


async def test_a_tool_batch_continues_once_whole_and_completed() -> None:
    """tau2's rule: ``response.create`` only after every call of a completed
    response is answered — once, whichever comes last."""
    release = asyncio.Event()

    async def slow(id: str = "u1") -> dict[str, str]:
        await release.wait()
        return {"id": id}

    transport, made = _transport(tools=_router(slow))
    await transport.open(SessionConfig(party=PartyId("agent")))
    conn = made[0]
    conn.on_event(_created("d1", "r1"))
    conn.on_event(_call("d1", "c1"))
    conn.on_event(_call("d1", "c2"))
    conn.on_event(_call("d1", "c2"))  # repeated: surfaced once
    conn.on_event(_completed("d1", "r1"))
    await _settle()
    assert conn.sent == []  # completed, but the answers are not back
    release.set()
    await _settle()
    kinds = [(e["type"], e.get("item", {}).get("call_id")) for e in conn.sent]
    assert kinds == [
        ("response.item.create", "c1"),
        ("response.item.create", "c2"),
        ("response.create", None),
    ]
    assert conn.sent[0]["item"]["output"] == '{"id": "u7"}'
    await transport.close()


async def test_answers_back_before_completion_wait_for_it() -> None:
    transport, made = _transport(tools=_router())
    await transport.open(SessionConfig(party=PartyId("agent")))
    conn = made[0]
    conn.on_event(_created("d1", "r1"))
    conn.on_event(_call("d1", "c1"))
    await _settle()
    assert [e["type"] for e in conn.sent] == ["response.item.create"]
    conn.on_event(_completed("d1", "r1"))
    assert [e["type"] for e in conn.sent] == ["response.item.create", "response.create"]
    await transport.close()


async def test_usage_is_the_backends_tokens() -> None:
    transport, made = _transport()
    await transport.open(SessionConfig(party=PartyId("agent")))
    conn = made[0]
    usage = {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 800},
        "output_tokens": 40,
        "output_tokens_details": {"reasoning_tokens": 10},
    }
    conn.on_event({"type": "session.delegation.created"})
    conn.on_event(_created("d1", "r1"))
    conn.on_event(_completed("d1", "r1", usage))
    conn.on_event(_created("d2", "r2"))
    conn.on_event(_completed("d2", "r2", usage))
    frontend, backend = transport.usage_records()
    assert (frontend.component, frontend.model, frontend.requests) == (
        "agent",
        "gpt-live-1",
        1,
    )
    assert frontend.input_tokens is None
    assert (backend.component, backend.model, backend.requests) == (
        "agent.backend",
        "gpt-6-luna",
        2,
    )
    assert (
        backend.input_tokens,
        backend.cached_input_tokens,
        backend.output_tokens,
    ) == (
        2000,
        1600,
        80,
    )
    await transport.close()


@pytest.mark.parametrize(
    "event",
    [
        {"type": "error", "error": {"code": "server_error", "message": "boom"}},
        {"type": "session.closed"},
        {
            "type": "response.event",
            "delegation_id": "d1",
            "event": {"type": "response.failed", "response": {"id": "r1"}},
        },
    ],
)
async def test_a_failed_session_ends_the_stream_and_the_next_send(event) -> None:
    """tau2 raises on all three; the trial's retry takes it from there."""
    transport, made = _transport()
    await transport.open(SessionConfig(party=PartyId("agent")))
    made[0].on_event(event)
    assert [e async for e in transport.events()] == []
    with pytest.raises(ProviderError) as raised:
        await transport.tx(
            AudioFrame(b"\x00\x00" * 480, 0, PartyId("c"), 0.0, LIVE_AUDIO)
        )
    assert raised.value.kind is ProviderFailure.STREAM_BROKEN
    assert transport.errors


async def test_a_lost_peer_connection_fails_the_session() -> None:
    transport, made = _transport()
    await transport.open(SessionConfig(party=PartyId("agent")))
    made[0].on_error("peer connection failed")
    assert transport.pump_error is not None
    assert transport.errors == [
        {"code": "webrtc_connection_lost", "message": "peer connection failed"}
    ]


async def test_close_ends_the_session_cleanly() -> None:
    transport, made = _transport()
    await transport.open(SessionConfig(party=PartyId("agent")))
    await transport.close()
    assert made[0].sent == [{"type": "session.close"}]
    assert made[0].closed and transport.errors == []  # its session.closed is no error


async def test_caller_audio_goes_out_at_24_khz() -> None:
    transport, made = _transport()
    await transport.open(SessionConfig(party=PartyId("agent")))
    pcm = b"\x10\x00" * 480
    await transport.tx(AudioFrame(pcm, 0, PartyId("c"), 0.0, LIVE_AUDIO))
    ulaw = g711.encode(b"\x00\x10" * 160)
    await transport.tx(AudioFrame(ulaw, 1, PartyId("c"), 0.0, TELEPHONY))
    assert made[0].audio[0] == pcm
    assert len(made[0].audio[1]) == 480 * 2  # 20 ms at 24 kHz
    assert transport.converted_frames == 1
    await transport.close()


def test_the_registry_builds_it_with_a_backend() -> None:
    registry = ProviderRegistry()
    register_all(registry)
    assert "openai.live" in registry
    with pytest.raises(ValueError, match="backend_model"):
        build()
    transport = build(backend_model="gpt-6-luna", voice="cedar")
    assert transport.session_start()["audio"]["output"]["voice"] == "cedar"
    assert transport.ref.model == "gpt-live-1"


async def test_the_input_track_clocks_20_ms_frames_at_48_khz() -> None:
    """tau2's track: queued audio goes out at 48 kHz, zero-filled when short."""
    from harbor_voice.providers.openai.live_webrtc import _InputTrack

    track = _InputTrack()
    track.append(b"\x10\x00" * 240)  # 10 ms at 24 kHz
    first, second = await track.recv(), await track.recv()
    assert (first.samples, first.sample_rate, first.pts) == (960, 48_000, 0)
    assert second.pts == 960
    assert first.to_ndarray().any()
    assert not second.to_ndarray().any()  # nothing queued: silence, not a gap


async def test_the_voice_sessions_billed_seconds_are_the_last_report() -> None:
    """``session.usage.updated`` carries cumulative snapshots (never summed);
    ``session.closed`` carries the final figure."""
    transport, made = _transport()
    await transport.open(SessionConfig(party=PartyId("agent")))
    conn = made[0]
    conn.on_event({"type": "session.usage.updated", "usage": {"seconds": 12}})
    conn.on_event({"type": "session.usage.updated", "usage": {"seconds": 30}})
    assert transport.usage_records()[0].seconds == 30.0

    def closing(event: dict) -> None:
        conn.sent.append(event)
        conn.on_event({"type": "session.closed", "usage": {"seconds": 31.4}})

    conn.send = closing  # type: ignore[method-assign]
    await transport.close()
    frontend, backend = transport.usage_records()
    assert (frontend.model, frontend.seconds) == ("gpt-live-1", 31.4)
    assert backend.seconds is None
