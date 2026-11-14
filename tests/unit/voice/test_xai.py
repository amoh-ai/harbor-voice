"""xAI providers offline: the realtime transport against a fake socket, the
REST TTS and STT against ``httpx.MockTransport``. No test calls the API."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from typing import Any

import httpx
import pytest

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.providers import MappingExecutor
from harbor_voice.providers import ToolRouter
from harbor_voice.providers import ToolSpec
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.credentials import access_for
from harbor_voice.providers.openai.realtime import TELEPHONY
from harbor_voice.providers.openai.realtime import RealtimeSettings
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.providers.xai import register_all
from harbor_voice.providers.xai import realtime as xr
from harbor_voice.providers.xai import speech
from harbor_voice.transport.base import SessionConfig
from tests.unit.voice import fakes

pytestmark = pytest.mark.unit

SPEC = ToolSpec(
    name="get_user",
    description="Look a user up",
    input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
)
SESSION = SessionConfig(party=PartyId("agent"))


# ------------------------------------------------------------------ realtime


class FakeSocket(fakes.FakeSocket):
    """A realtime server: hands out queued events, records what it is sent."""

    def __init__(self, *events: dict[str, Any]) -> None:
        super().__init__(
            {"type": "session.created", "session": {"id": "sess_1"}},
            {"type": "session.updated"},
            *events,
        )


class Connect:
    def __init__(self, socket: FakeSocket) -> None:
        self.socket = socket
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.options: list[dict[str, Any]] = []

    async def __call__(
        self, url: str, *, additional_headers: dict[str, str], **options: Any
    ) -> FakeSocket:
        self.calls.append((url, additional_headers))
        self.options.append(options)
        return self.socket


def _router() -> ToolRouter:
    return ToolRouter(
        MappingExecutor(
            specs=[SPEC], handlers={"get_user": lambda id="u1": {"id": id}}
        ),
        sleep=lambda _s: asyncio.sleep(0),
    )


def _transport(
    socket: FakeSocket, **kwargs: Any
) -> tuple[xr.XAIRealtimeTransport, Connect]:
    connect = Connect(socket)
    kwargs.setdefault("api_key", "k")
    transport = xr.build(
        system_prompt="You are an airline agent.", connect=connect, **kwargs
    )
    return transport, connect


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


def test_session_update_is_xais_documented_shape() -> None:
    transport, _ = _transport(FakeSocket(), tools=_router())
    assert transport.session_update() == {
        "type": "session.update",
        "session": {
            "instructions": "You are an airline agent.",
            "voice": "eve",
            "turn_detection": {"type": "server_vad"},
            "tools": [
                {
                    "type": "function",
                    "name": "get_user",
                    "description": "Look a user up",
                    "parameters": SPEC.input_schema,
                }
            ],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "transcription": {"language_hint": "en"},
                },
                "output": {"format": {"type": "audio/pcmu"}},
            },
        },
    }
    assert transport.capabilities.audio_in == TELEPHONY


def test_reasoning_and_pcm24k_are_sent_when_set() -> None:
    settings = RealtimeSettings(
        voice="ara",
        language=None,
        noise_reduction=None,
        tool_choice=None,
        reasoning_effort="none",
        audio_format="pcm24k",
    )
    transport, _ = _transport(FakeSocket(), settings=settings)
    session = transport.session_update()["session"]
    assert session["reasoning"] == {"effort": "none"}
    assert session["voice"] == "ara"
    assert session["audio"] == {
        "input": {"format": {"type": "audio/pcm", "rate": 24_000}},
        "output": {"format": {"type": "audio/pcm", "rate": 24_000}},
    }


@pytest.mark.parametrize(
    "change",
    [{"vad_threshold": 0.7}, {"tool_choice": "auto"}, {"noise_reduction": "far_field"}],
)
def test_a_setting_xai_does_not_document_is_refused(change: dict[str, Any]) -> None:
    settings = replace(xr.XAI_SETTINGS, **change)
    with pytest.raises(ValueError, match="has no setting for"):
        _transport(FakeSocket(), settings=settings)


async def test_it_connects_to_xai_with_the_xai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    transport, connect = _transport(FakeSocket(), api_key=None)
    await transport.open(SESSION)
    await transport.close()
    ((url, headers),) = connect.calls
    assert url == "wss://api.x.ai/v1/realtime?model=grok-voice-3.0"
    assert headers == {"Authorization": "Bearer xai-test"}


async def test_without_an_xai_key_the_openai_key_is_never_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    transport, connect = _transport(FakeSocket(), api_key=None)
    with pytest.raises(ProviderError) as info:
        await transport.open(SESSION)
    assert info.value.kind is ProviderFailure.AUTH
    assert info.value.provider == "xai.realtime"
    assert connect.calls == []


async def test_a_call_streams_audio_answers_tools_and_meters_the_session() -> None:
    socket = FakeSocket(
        {"type": "response.output_audio.delta", "item_id": "i1", "delta": "AAA="},
        {
            "type": "response.function_call_arguments.done",
            "call_id": "c1",
            "name": "get_user",
            "arguments": '{"id": "u7"}',
        },
        {
            "type": "response.done",
            "response": {
                "id": "r1",
                "status": "completed",
                "usage": {"input_tokens": 12, "output_tokens": 30, "total_tokens": 42},
            },
        },
    )
    transport, _ = _transport(socket, tools=_router())
    await transport.open(SESSION)
    await _settle()
    await transport.close()
    frames = [f async for f in transport.rx()]

    assert [f.pcm for f in frames] == [b"\x00\x00"]
    assert [m["type"] for m in socket.sent] == [
        "session.update",
        "conversation.item.create",
        "response.create",
    ]
    output = socket.sent[1]["item"]
    assert output["type"] == "function_call_output" and output["call_id"] == "c1"
    assert json.loads(output["output"]) == {"id": "u7"}

    record = transport.usage_record()
    assert (record.requests, record.input_tokens, record.output_tokens) == (1, 12, 30)
    assert record.seconds is not None and record.seconds > 0
    assert transport.usage.audio_seconds == record.seconds
    assert transport.ref.provider == "xai.realtime"
    assert transport.ref.model == "grok-voice-3.0"


def test_register_all_adds_three_providers_under_harbors_xai_slug() -> None:
    registry = ProviderRegistry()
    register_all(registry)
    specs = {r.spec.name: r.spec for r in registry}
    assert {n: (s.modality, s.default_model) for n, s in specs.items()} == {
        "xai.realtime": (Modality.S2S, "grok-voice-3.0"),
        "xai.tts": (Modality.TTS, "eve"),
        "xai.stt": (Modality.STT, "grok-voice-transcribe-2.0"),
    }
    assert {s.credential_provider for s in specs.values()} == {"xai"}
    access = access_for("xai")
    assert access is not None and access.api_key_envs == ("XAI_API_KEY",)
    tts = registry.create("xai.tts/rex", modality=Modality.TTS, api_key="k")
    assert tts.voice == "rex"


# ----------------------------------------------------------------- REST


def _client(handler) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(record)), seen


async def _speak(tts: speech.XAITTS, text: str) -> bytes:
    return b"".join([c async for c in tts.synthesize(text, party=PartyId("caller"))])


async def test_tts_asks_for_raw_pcm_at_the_callers_rate() -> None:
    pcm = bytes(48_000)  # one second at 24 kHz
    client, seen = _client(lambda _r: httpx.Response(200, content=pcm))
    tts = speech.XAITTS(client=client, api_key="test-key")

    assert await _speak(tts, "Hello there.") == pcm
    (request,) = seen
    assert str(request.url) == "https://api.x.ai/v1/tts"
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "text": "Hello there.",
        "voice_id": "eve",
        "language": "en",
        "output_format": {"codec": "pcm", "sample_rate": 24_000},
    }
    assert tts.characters_billed == len("Hello there.")


async def test_tts_says_nothing_for_no_text() -> None:
    client, seen = _client(lambda _r: httpx.Response(500))
    tts = speech.XAITTS(client=client, api_key="k")
    assert await _speak(tts, "") == b""
    assert seen == []


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, ProviderFailure.AUTH),
        (404, ProviderFailure.NOT_ENABLED),
        (429, ProviderFailure.THROTTLED),
        (503, ProviderFailure.STREAM_BROKEN),
        (400, ProviderFailure.MALFORMED),
    ],
)
async def test_tts_http_errors_are_classified(
    status: int, kind: ProviderFailure
) -> None:
    client, _ = _client(lambda _r: httpx.Response(status, text="nope"))
    tts = speech.XAITTS(client=client, api_key="k")
    with pytest.raises(ProviderError) as info:
        await _speak(tts, "hi")
    assert info.value.kind is kind
    assert info.value.provider == "xai.tts"
    assert tts.characters_billed == 0


async def test_a_missing_key_fails_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    client, seen = _client(lambda _r: httpx.Response(200, content=b"\0\0"))
    with pytest.raises(ProviderError) as info:
        await _speak(speech.XAITTS(client=client), "hi")
    assert info.value.kind is ProviderFailure.AUTH
    assert seen == []


def test_tts_refuses_a_format_the_api_cannot_produce() -> None:
    with pytest.raises(ValueError, match="mono PCM16"):
        speech.XAITTS(audio_format=AudioFormat(sample_rate_hz=32_000))


def _frames(pcm: bytes, fmt: AudioFormat) -> list[AudioFrame]:
    return [
        AudioFrame(pcm=pcm, seq=0, party=PartyId("caller"), t_produced=0.0, fmt=fmt)
    ]


async def test_stt_sends_raw_audio_last_and_meters_the_reported_duration() -> None:
    body = {"text": "I need to change a flight.", "language": "en", "duration": 1.5}
    client, seen = _client(lambda _r: httpx.Response(200, json=body))
    stt = speech.XAISTT(client=client, api_key="test-key", language="en")

    (transcript,) = [t async for t in stt.transcribe(_frames(bytes(8_000), TELEPHONY))]

    assert (transcript.text, transcript.is_final) == (
        "I need to change a flight.",
        True,
    )
    assert transcript.audio_duration_s == 1.5
    assert stt.usage.audio_seconds == 1.5
    (request,) = seen
    assert str(request.url) == "https://api.x.ai/v1/stt"
    assert request.headers["authorization"] == "Bearer test-key"
    content = request.content.decode("latin-1")
    names = re.findall(r'form-data; name="([^"]+)"', content)
    assert names == ["model", "audio_format", "sample_rate", "language", "file"]
    assert 'name="audio_format"\r\n\r\nmulaw' in content
    assert 'name="sample_rate"\r\n\r\n8000' in content
    assert "grok-voice-transcribe-2.0" in content


async def test_stt_sends_nothing_for_no_audio() -> None:
    client, seen = _client(lambda _r: httpx.Response(500))
    stt = speech.XAISTT(client=client, api_key="k")
    (transcript,) = [t async for t in stt.transcribe([])]
    assert transcript.text == "" and seen == []


async def test_stt_an_unreadable_body_is_malformed() -> None:
    client, _ = _client(lambda _r: httpx.Response(200, text="not json"))
    stt = speech.XAISTT(client=client, api_key="k")
    with pytest.raises(ProviderError) as info:
        _ = [t async for t in stt.transcribe(_frames(bytes(320), AudioFormat()))]
    assert info.value.kind is ProviderFailure.MALFORMED


async def test_a_dropped_connection_is_a_broken_stream() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client, _ = _client(fail)
    stt = speech.XAISTT(client=client, api_key="k")
    with pytest.raises(ProviderError) as info:
        _ = [t async for t in stt.transcribe(_frames(bytes(320), AudioFormat()))]
    assert info.value.kind is ProviderFailure.STREAM_BROKEN
