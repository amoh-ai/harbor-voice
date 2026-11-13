"""Gemini Live's transport against a fake socket: both sign-in modes, the
setup message, audio both ways, VAD, tool calls through the router, usage and
error mapping. No test reaches Google."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.providers import MappingExecutor
from harbor_voice.providers import ToolRouter
from harbor_voice.providers import ToolSpec
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.google import live
from harbor_voice.providers.google.live import PCM16K
from harbor_voice.providers.google.live import GeminiLiveTransport
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import ResponseDone

pytestmark = pytest.mark.unit

SPEC = ToolSpec(
    name="get_user",
    description="Look a user up",
    input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
)
CONFIG = SessionConfig(party=PartyId("agent"))
_ENVS = (
    *live.API_KEY_ENVS,
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)


class FakeSocket:
    """A Live server: answers setup, hands out queued messages, records sends."""

    def __init__(self, *messages: dict[str, Any], first: Any = None) -> None:
        self.inbox: asyncio.Queue[str | None] = asyncio.Queue()
        self.push(first if first is not None else {"setupComplete": {}})
        for message in messages:
            self.push(message)
        self.sent: list[dict[str, Any]] = []
        self.url: str | None = None
        self.headers: dict[str, str] = {}
        self.closed = False

    def push(self, message: dict[str, Any]) -> None:
        self.inbox.put_nowait(json.dumps(message))

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def recv(self) -> str:
        raw = await self.inbox.get()
        assert raw is not None
        return raw

    def __aiter__(self) -> FakeSocket:
        return self

    async def __anext__(self) -> str:
        raw = await self.inbox.get()
        if raw is None:
            raise StopAsyncIteration
        return raw

    async def close(self) -> None:
        self.closed = True
        self.inbox.put_nowait(None)


def _router() -> ToolRouter:
    def get_user(id: str = "u1") -> dict[str, str]:
        return {"id": id}

    return ToolRouter(
        MappingExecutor(specs=[SPEC], handlers={"get_user": get_user}),
        sleep=lambda _s: asyncio.sleep(0),
    )


def _transport(socket: FakeSocket, **kwargs: Any) -> GeminiLiveTransport:
    async def connect(url: str, *, additional_headers: dict[str, str]) -> FakeSocket:
        socket.url, socket.headers = url, additional_headers
        return socket

    return GeminiLiveTransport(party=PartyId("agent"), connect=connect, **kwargs)


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def _events(transport: GeminiLiveTransport) -> list[Any]:
    await _settle()
    await transport.close()
    return [e async for e in transport.events()]


def _audio(pcm: bytes) -> dict[str, Any]:
    data = base64.b64encode(pcm).decode()
    return {
        "serverContent": {
            "modelTurn": {
                "parts": [{"inlineData": {"mimeType": "audio/pcm", "data": data}}]
            }
        }
    }


# ------------------------------------------------------------------ sign-in


async def test_ai_studio_sends_the_key_as_a_header_not_in_the_url(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    socket = FakeSocket()
    transport = _transport(socket)
    await transport.open(CONFIG)
    assert socket.url == live.STUDIO_URL
    assert "test-key" not in socket.url
    assert socket.headers == {"x-goog-api-key": "test-key"}
    assert socket.sent[0]["setup"]["model"] == "models/gemini-3.8-live"
    await transport.close()


async def test_google_api_key_wins_when_both_are_set(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "second")
    monkeypatch.setenv("GOOGLE_API_KEY", "first")
    socket = FakeSocket()
    transport = _transport(socket)
    await transport.open(CONFIG)
    assert socket.headers == {"x-goog-api-key": "first"}
    await transport.close()


async def test_no_key_is_an_auth_failure_before_connecting() -> None:
    socket = FakeSocket()
    with pytest.raises(ProviderError) as info:
        await _transport(socket).open(CONFIG)
    assert info.value.kind is ProviderFailure.AUTH
    assert socket.url is None


async def test_vertex_uses_adc_the_regional_host_and_a_resource_path(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
    monkeypatch.setattr(live, "_adc_token", lambda: "adc-token")
    socket = FakeSocket()
    transport = _transport(socket)
    await transport.open(CONFIG)
    assert socket.url == (
        "wss://europe-west4-aiplatform.googleapis.com"
        "/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
    )
    assert socket.headers == {"Authorization": "Bearer adc-token"}
    assert socket.sent[0]["setup"]["model"] == (
        "projects/proj/locations/europe-west4/publishers/google/models/gemini-3.8-live"
    )
    assert transport.ref.region == "europe-west4"
    await transport.close()


async def test_vertex_by_argument_defaults_the_location(monkeypatch) -> None:
    monkeypatch.setattr(live, "_adc_token", lambda: "adc-token")
    socket = FakeSocket()
    transport = _transport(socket, vertex=True, project="p")
    await transport.open(CONFIG)
    assert socket.url is not None and socket.url.startswith(
        "wss://us-central1-aiplatform.googleapis.com/"
    )
    await transport.close()


async def test_vertex_without_a_project_is_an_auth_failure() -> None:
    with pytest.raises(ProviderError) as info:
        await _transport(FakeSocket(), vertex=True).open(CONFIG)
    assert info.value.kind is ProviderFailure.AUTH


# -------------------------------------------------------------------- setup


def test_setup_declares_audio_prompt_voice_transcription_and_tools() -> None:
    transport = _transport(
        FakeSocket(), system_prompt="Be brief.", voice="Kore", tools=_router()
    )
    assert transport.setup() == {
        "setup": {
            "model": "models/gemini-3.8-live",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
                },
            },
            "outputAudioTranscription": {},
            "systemInstruction": {"parts": [{"text": "Be brief."}]},
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "get_user",
                            "description": "Look a user up",
                            "parametersJsonSchema": SPEC.input_schema,
                        }
                    ]
                }
            ],
        }
    }


def test_unset_options_are_left_out_of_setup() -> None:
    setup = _transport(FakeSocket()).setup()["setup"]
    assert set(setup) == {"model", "generationConfig", "outputAudioTranscription"}
    assert setup["generationConfig"] == {"responseModalities": ["AUDIO"]}


async def test_a_rejected_setup_is_classified_and_the_socket_dropped(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    class Refused(FakeSocket):
        async def recv(self) -> str:
            raise ConnectionError(
                "received 1008 (policy violation) models/x is not found"
            )

    socket = Refused()
    transport = _transport(socket)
    with pytest.raises(ProviderError) as info:
        await transport.open(CONFIG)
    assert info.value.kind is ProviderFailure.NOT_ENABLED
    assert socket.closed and not transport.is_open


async def test_a_reply_other_than_setup_complete_is_malformed(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with pytest.raises(ProviderError) as info:
        await _transport(FakeSocket(first={"goAway": {}})).open(CONFIG)
    assert info.value.kind is ProviderFailure.MALFORMED


async def test_a_session_longer_than_a_connection_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    socket = FakeSocket()
    with pytest.raises(ProviderError):
        await _transport(socket).open(
            SessionConfig(party=PartyId("agent"), max_session_s=900)
        )
    assert socket.url is None


# -------------------------------------------------------------------- audio


async def test_input_is_sent_as_16k_pcm_and_resampled_from_canonical(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    socket = FakeSocket()
    transport = _transport(socket)
    await transport.open(CONFIG)
    native = b"\x01\x00" * 160
    await transport.tx(
        AudioFrame(pcm=native, seq=0, party=PartyId("c"), t_produced=0.0, fmt=PCM16K)
    )
    await transport.tx(
        AudioFrame(
            pcm=b"\x00\x00" * 480,
            seq=1,
            party=PartyId("c"),
            t_produced=0.0,
            fmt=CANONICAL,
        )
    )
    first, second = (m["realtimeInput"]["audio"] for m in socket.sent[1:])
    assert first == {
        "data": base64.b64encode(native).decode(),
        "mimeType": "audio/pcm;rate=16000",
    }
    assert len(base64.b64decode(second["data"])) == 320 * 2
    assert transport.converted_frames == 1
    await transport.close()


async def test_events_carry_audio_transcript_vad_and_cutoffs_in_order(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    socket = FakeSocket(
        {"voiceActivity": {"type": "ACTIVITY_START", "audioOffset": "1.250s"}},
        _audio(b"\x01\x02"),
        {"serverContent": {"outputTranscription": {"text": "Hi"}}},
        {"serverContent": {"turnComplete": True}},
        {"voiceActivity": {"type": "ACTIVITY_END", "audioOffset": "2s"}},
        _audio(b"\x03\x04"),
        {"serverContent": {"interrupted": True}},
        _audio(b"\x05\x06"),
    )
    transport = _transport(socket)
    await transport.open(CONFIG)
    events = await _events(transport)
    assert [type(e) for e in events] == [
        InputSpeech,
        OutputAudio,
        OutputTranscript,
        InputSpeech,
        OutputAudio,
        ResponseDone,
        OutputAudio,
    ]
    # Gemini stops itself on barge-in; the stream says which utterance.
    assert (events[5].response_id, events[5].status) == ("turn-1", "cancelled")
    assert (events[0].started, events[0].audio_ms) == (True, 1250)
    assert (events[3].started, events[3].audio_ms) == (False, 2000)
    assert [e.utterance for e in events if isinstance(e, OutputAudio)] == [
        "turn-0",
        "turn-1",
        "turn-2",
    ]
    assert events[2].utterance == "turn-0" and events[2].delta == "Hi"


async def test_rx_yields_24k_frames(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    transport = _transport(FakeSocket(_audio(b"\x01\x02\x03\x04")))
    await transport.open(CONFIG)
    await _settle()
    await transport.close()
    frames = [f async for f in transport.rx()]
    assert [f.pcm for f in frames] == [b"\x01\x02\x03\x04"]
    assert frames[0].fmt == CANONICAL


async def test_send_text_is_realtime_text(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    socket = FakeSocket()
    transport = _transport(socket)
    await transport.open(CONFIG)
    await transport.send_text("hello")
    assert socket.sent[-1] == {"realtimeInput": {"text": "hello"}}
    await transport.close()


# -------------------------------------------------------------------- tools


async def test_one_tool_call_message_gets_one_response_through_the_router(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    router = _router()
    socket = FakeSocket(
        {
            "toolCall": {
                "functionCalls": [
                    {"id": "c1", "name": "get_user", "args": {"id": "u7"}},
                    {"id": "c2", "name": "nope", "args": {}},
                ]
            }
        }
    )
    transport = _transport(socket, tools=router)
    await transport.open(CONFIG)
    await _settle()
    assert socket.sent[-1] == {
        "toolResponse": {
            "functionResponses": [
                {"id": "c1", "name": "get_user", "response": {"id": "u7"}},
                {
                    "id": "c2",
                    "name": "nope",
                    "response": {
                        "error": "unknown_tool",
                        "message": "no tool named 'nope'",
                    },
                },
            ]
        }
    }
    assert [r.tool_use_id for r in router.records] == ["c1", "c2"]
    await transport.close()


async def test_a_call_with_no_tools_configured_is_still_answered(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    socket = FakeSocket(
        {"toolCall": {"functionCalls": [{"id": "c1", "name": "x", "args": {}}]}}
    )
    transport = _transport(socket)
    await transport.open(CONFIG)
    await _settle()
    response = socket.sent[-1]["toolResponse"]["functionResponses"][0]
    assert response["id"] == "c1" and "error" in response["response"]
    await transport.close()


# -------------------------------------------------------------------- usage


async def test_usage_is_summed_by_modality(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    report = {
        "usageMetadata": {
            "promptTokenCount": 100,
            "cachedContentTokenCount": 10,
            "responseTokenCount": 40,
            "thoughtsTokenCount": 5,
            "promptTokensDetails": [
                {"modality": "AUDIO", "tokenCount": 80},
                {"modality": "TEXT", "tokenCount": 20},
            ],
            "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 40}],
        }
    }
    transport = _transport(FakeSocket(report, report))
    await transport.open(CONFIG)
    await _settle()
    record = transport.usage_record()
    assert (record.requests, record.input_tokens, record.output_tokens) == (2, 200, 90)
    assert (record.input_audio_tokens, record.output_audio_tokens) == (160, 80)
    assert record.cached_input_tokens == 20
    usage = transport.usage
    assert usage.speech_tokens == 240 and usage.text_tokens == 40
    await transport.close()


# ------------------------------------------------------------- declaration


def test_declaration_matches_implementation_and_registers() -> None:
    transport = _transport(FakeSocket())
    assert transport.check_contract() == []
    caps = transport.capabilities
    assert caps.audio_in == PCM16K and caps.audio_out == CANONICAL
    assert caps.max_session_s == live.MAX_SESSION_S

    from harbor_voice.providers import google

    registry = ProviderRegistry()
    google.register_all(registry)
    built = registry.create(
        "google.live", modality=Modality.S2S, party="agent", connect=None
    )
    assert isinstance(built, GeminiLiveTransport)
    assert live.SPEC.default_model == "gemini-3.8-live"


def test_classify_maps_handshake_and_close_reasons() -> None:
    cases = {
        "server rejected WebSocket connection: HTTP 403": ProviderFailure.AUTH,
        "received 1007 API key not valid. Please pass a valid API key.": (
            ProviderFailure.AUTH
        ),
        "HTTP 429": ProviderFailure.THROTTLED,
        "received 1011 internal error": ProviderFailure.STREAM_BROKEN,
    }
    for text, kind in cases.items():
        assert live._classify(RuntimeError(text), "m").kind is kind
    assert live._classify(TimeoutError(), "m").kind is ProviderFailure.TIMEOUT
