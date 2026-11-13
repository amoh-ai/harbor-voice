"""OpenAI Realtime's transport against a fake socket: tau2's session byte for
byte by default; the settings another benchmark's session needs (24 kHz PCM,
fields left out, an opening message, serial tool answers); and the opt-in
session events."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest

from harbor_voice.audio import g711
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.providers import MappingExecutor
from harbor_voice.providers import ToolRouter
from harbor_voice.providers import ToolSpec
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.openai.realtime import PCM24K
from harbor_voice.providers.openai.realtime import TELEPHONY
from harbor_voice.providers.openai.realtime import OpenAIRealtimeTransport
from harbor_voice.providers.openai.realtime import RealtimeSettings
from harbor_voice.providers.openai.realtime import response_has_function_calls
from harbor_voice.providers.openai.realtime import response_text
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import InputTranscript
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import OutputTranscriptDone
from harbor_voice.transport.stream import ResponseDone
from harbor_voice.transport.stream import ToolCall
from tests.unit.voice import fakes

pytestmark = pytest.mark.unit

SPEC = ToolSpec(
    name="get_user",
    description="Look a user up",
    input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
)

TAU2_WIRE = (
    '{"type": "session.update", "session": {"type": "realtime", "instructions": '
    '"You are an airline agent.", "output_modalities": ["audio"], "tools": '
    '[{"type": "function", "name": "get_user", "description": "Look a user up", '
    '"parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}], '
    '"tool_choice": "auto", "audio": {"input": {"format": {"type": "audio/pcmu"}, '
    '"transcription": {"model": "gpt-4o-transcribe", "language": "en"}, '
    '"noise_reduction": {"type": "near_field"}, "turn_detection": {"type": '
    '"server_vad", "threshold": 0.5, "prefix_padding_ms": 300, '
    '"silence_duration_ms": 500}}, "output": {"format": {"type": "audio/pcmu"}, '
    '"voice": "alloy"}}}}'
)
"""What the transport sent before these settings existed, captured from it."""

TAU2_DIGEST = "sha256:ceca2da964f3942b4773b40a8fdaf38990aa2691bd5b167b6f032035cc841cdf"


class FakeSocket(fakes.FakeSocket):
    """A Realtime server: hands out queued events, records what it is sent."""

    def __init__(self, *events: dict[str, Any]) -> None:
        super().__init__(
            {"type": "session.created", "session": {"id": "sess_1"}},
            {"type": "session.updated"},
            *events,
        )


def _router(handler=None) -> ToolRouter:
    def get_user(id: str = "u1") -> dict[str, str]:
        return {"id": id}

    return ToolRouter(
        MappingExecutor(specs=[SPEC], handlers={"get_user": handler or get_user}),
        sleep=lambda _s: asyncio.sleep(0),
    )


def _transport(
    socket: FakeSocket | None = None,
    *,
    tools: ToolRouter | None = None,
    **settings: Any,
) -> OpenAIRealtimeTransport:
    async def connect(url: str, *, additional_headers: dict[str, str]) -> FakeSocket:
        assert socket is not None
        return socket

    return OpenAIRealtimeTransport(
        party=PartyId("agent"),
        system_prompt="You are an airline agent.",
        tools=tools,
        settings=RealtimeSettings(**settings),
        api_key="k",
        connect=connect,
    )


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def _events(transport: OpenAIRealtimeTransport) -> list[Any]:
    await _settle()
    await transport.close()
    return [e async for e in transport.events()]


def _call(call_id: str, name: str = "get_user", arguments: str = '{"id": "u7"}'):
    return {
        "type": "response.function_call_arguments.done",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


# ----------------------------------------------------------------- tau2's session


def test_the_default_session_is_tau2s_byte_for_byte() -> None:
    transport = _transport(tools=_router())
    assert json.dumps(transport.session_update()) == TAU2_WIRE
    assert transport.ref.config_digest == TAU2_DIGEST
    assert transport.opening() == []
    caps = transport.capabilities
    assert caps.audio_in == TELEPHONY and caps.audio_out == TELEPHONY


async def test_by_default_only_the_stepped_events_surface() -> None:
    """A stepped consumer raises on a kind it does not know (tau-voice's
    ``AgentStepper.feed``), so the session events stay off unless asked for."""
    socket = FakeSocket(
        {"type": "response.output_audio.delta", "item_id": "i1", "delta": "AAA="},
        {
            "type": "response.output_audio_transcript.delta",
            "item_id": "i1",
            "delta": "Hi",
        },
        {
            "type": "response.output_audio_transcript.done",
            "item_id": "i1",
            "transcript": "Hi",
        },
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "u1",
            "transcript": "hello",
        },
        {"type": "input_audio_buffer.speech_started", "audio_start_ms": 40},
        {"type": "response.done", "response": {"id": "r1", "status": "completed"}},
    )
    transport = _transport(socket)
    await transport.open(SessionConfig(party=PartyId("agent")))
    events = await _events(transport)
    assert [type(e) for e in events] == [OutputAudio, OutputTranscript, InputSpeech]
    assert [m["type"] for m in socket.sent] == ["session.update"]


async def test_two_calls_in_one_response_get_one_continuation_by_default() -> None:
    socket = FakeSocket(_call("c1"), _call("c2"))
    transport = _transport(socket, tools=_router())
    await transport.open(SessionConfig(party=PartyId("agent")))
    await _settle()
    assert [(m["type"], m.get("item", {}).get("call_id")) for m in socket.sent[1:]] == [
        ("conversation.item.create", "c1"),
        ("conversation.item.create", "c2"),
        ("response.create", None),
    ]
    await transport.close()


# ------------------------------------------------------------- other sessions


def test_fields_set_to_none_are_left_out_not_sent_as_null() -> None:
    transport = _transport(
        tools=_router(),
        tool_choice=None,
        language=None,
        noise_reduction=None,
        parallel_tool_calls=False,
    )
    session = transport.session_update()["session"]
    assert "tool_choice" not in session
    assert session["parallel_tool_calls"] is False
    audio_in = session["audio"]["input"]
    assert "noise_reduction" not in audio_in
    assert audio_in["transcription"] == {"model": "gpt-4o-transcribe"}


def test_new_settings_join_the_digest_only_when_changed() -> None:
    assert "audio_format" not in RealtimeSettings().as_config()
    config = RealtimeSettings(audio_format="pcm24k", tool_choice=None).as_config()
    assert config["audio_format"] == "pcm24k" and config["tool_choice"] is None
    assert "opening_text" not in config


async def test_pcm24k_passes_24_khz_through_and_converts_the_rest() -> None:
    socket = FakeSocket(
        {"type": "response.output_audio.delta", "item_id": "i1", "delta": "AAAAAAAA"}
    )
    transport = _transport(socket, audio_format="pcm24k")
    session = transport.session_update()["session"]
    pcm = {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["input"]["format"] == pcm
    assert session["audio"]["output"]["format"] == pcm
    assert transport.capabilities.audio_in == PCM24K == transport.capabilities.audio_out

    await transport.open(SessionConfig(party=PartyId("agent")))
    frame = b"\x10\x00" * 480  # 20 ms at 24 kHz
    await transport.tx(AudioFrame(frame, 0, PartyId("c"), 0.0, PCM24K))
    ulaw = g711.encode(b"\x00\x10" * 160)  # 20 ms at 8 kHz
    await transport.tx(AudioFrame(ulaw, 1, PartyId("c"), 0.0, TELEPHONY))
    appended = [
        m["audio"] for m in socket.sent if m["type"] == "input_audio_buffer.append"
    ]
    assert base64.b64decode(appended[0]) == frame
    assert len(base64.b64decode(appended[1])) == 480 * 2
    assert transport.converted_frames == 1

    rx = transport.rx()
    first = await asyncio.wait_for(anext(rx), 1)
    assert first.fmt == PCM24K and first.pcm == b"\x00" * 6
    await transport.close()


async def test_pcmu_still_refuses_mulaw_at_another_rate() -> None:
    socket = FakeSocket()
    transport = _transport(socket)
    await transport.open(SessionConfig(party=PartyId("agent")))
    other = AudioFormat(sample_rate_hz=16_000, channels=1, encoding="mulaw")
    with pytest.raises(ProviderError, match="μ-law 8 kHz or mono PCM16"):
        await transport.tx(AudioFrame(b"\x00" * 320, 0, PartyId("c"), 0.0, other))
    await transport.close()


async def test_the_opening_text_follows_the_session_update_at_once() -> None:
    socket = FakeSocket()
    transport = _transport(socket, opening_text="Say: 'Hello!'")
    await transport.open(SessionConfig(party=PartyId("agent")))
    assert socket.sent[1:] == [
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say: 'Hello!'"}],
            },
        },
        {"type": "response.create"},
    ]
    await transport.close()


async def test_serial_tools_are_answered_in_the_receive_loop_one_by_one() -> None:
    """EVA's order: a call blocks the loop until answered, and every answer
    asks for a response — two calls, two ``response.create``."""
    release = asyncio.Event()

    async def slow(id: str = "u1") -> dict[str, str]:
        await release.wait()
        return {"id": id}

    socket = FakeSocket(
        _call("c1"),
        {"type": "input_audio_buffer.speech_started", "audio_start_ms": 10},
        _call("c2"),
    )
    transport = _transport(
        socket, tools=_router(slow), tool_flow="serial", extended_events=True
    )
    await transport.open(SessionConfig(party=PartyId("agent")))
    await _settle()
    assert transport._inbox.empty()  # the speech event waits behind the call
    assert [m["type"] for m in socket.sent] == ["session.update"]
    release.set()
    await _settle()
    assert [(m["type"], m.get("item", {}).get("call_id")) for m in socket.sent[1:]] == [
        ("conversation.item.create", "c1"),
        ("response.create", None),
        ("conversation.item.create", "c2"),
        ("response.create", None),
    ]
    events = await _events(transport)
    assert [type(e) for e in events] == [ToolCall, InputSpeech, ToolCall]
    first = events[0]
    assert (first.call_id, first.name, first.arguments) == (
        "c1",
        "get_user",
        {"id": "u7"},
    )
    assert first.result == {"id": "u7"} and not first.is_error
    assert first.started <= first.t


def _accented(id: str = "u1") -> dict[str, str]:
    return {"name": "José Müller"}


@pytest.mark.parametrize(
    ("ascii_only", "expected"),
    [
        (True, '{"name": "Jos\\u00e9 M\\u00fcller"}'),
        (False, '{"name": "José Müller"}'),
    ],
)
async def test_tool_output_escapes_non_ascii_only_when_asked(
    ascii_only: bool, expected: str
) -> None:
    socket = FakeSocket(_call("c1"))
    transport = _transport(
        socket, tools=_router(_accented), tool_output_ascii=ascii_only
    )
    await transport.open(SessionConfig(party=PartyId("agent")))
    await _settle()
    assert socket.sent[1]["item"]["output"] == expected
    await transport.close()


async def test_a_text_payload_goes_verbatim_either_way() -> None:
    socket = FakeSocket(_call("c1"))
    transport = _transport(
        socket,
        tools=_router(lambda id="u": {"text": "Café: ok"}),
        tool_output_ascii=False,
    )
    await transport.open(SessionConfig(party=PartyId("agent")))
    await _settle()
    assert socket.sent[1]["item"]["output"] == "Café: ok"
    await transport.close()


async def test_a_call_without_tools_is_still_answered_and_reported() -> None:
    socket = FakeSocket(_call("c1"))
    transport = _transport(socket, extended_events=True)
    await transport.open(SessionConfig(party=PartyId("agent")))
    events = await _events(transport)
    (call,) = events
    assert isinstance(call, ToolCall) and call.is_error
    assert socket.sent[1]["item"]["output"] == json.dumps(call.result)


# ------------------------------------------------------------- session events


async def test_session_events_arrive_in_order_when_asked_for() -> None:
    usage = {"total_tokens": 30, "input_tokens": 20, "output_tokens": 10}
    socket = FakeSocket(
        {"type": "input_audio_buffer.speech_started", "audio_start_ms": 0},
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "u1",
            "transcript": " I need to change my flight. ",
        },
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": "u2",
            "error": {"type": "invalid_request_error", "message": "no access"},
        },
        {
            "type": "response.output_audio_transcript.done",
            "item_id": "a1",
            "transcript": "Sure.",
        },
        {
            "type": "response.done",
            "response": {
                "id": "r1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_audio", "transcript": "Sure."}],
                    }
                ],
                "usage": usage,
            },
        },
        {"type": "response.done", "response": {"id": "r2", "status": "cancelled"}},
    )
    transport = _transport(socket, extended_events=True)
    await transport.open(SessionConfig(party=PartyId("agent")))
    events = await _events(transport)
    assert [type(e) for e in events] == [
        InputSpeech,
        InputTranscript,
        InputTranscript,
        OutputTranscriptDone,
        ResponseDone,
        ResponseDone,
    ]
    heard, failed, done, completed, cancelled = events[1:]
    assert (heard.item_id, heard.text, heard.error) == (
        "u1",
        " I need to change my flight. ",
        None,
    )
    assert (failed.item_id, failed.text, failed.error) == ("u2", "", "no access")
    assert (done.utterance, done.text) == ("a1", "Sure.")
    assert completed == ResponseDone(
        response_id="r1",
        status="completed",
        has_function_calls=False,
        text="",  # a GA part type; EVA's extraction reads only beta's
        t=completed.t,
        input_tokens=20,
        output_tokens=10,
    )
    assert (cancelled.status, cancelled.input_tokens) == ("cancelled", None)
    assert transport.usage_record().requests == 2


def test_response_helpers_read_output_items_as_eva_does() -> None:
    call = {"type": "function_call", "call_id": "c1"}
    beta = {"type": "message", "content": [{"type": "audio", "transcript": " Hi. "}]}
    assert response_has_function_calls({"output": [beta, call]})
    assert not response_has_function_calls({"output": [beta]})
    assert response_text({"output": [beta, call]}) == "Hi."
    assert response_text({}) == ""
