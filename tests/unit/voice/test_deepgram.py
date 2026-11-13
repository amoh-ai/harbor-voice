"""Deepgram STT, TTS and Voice Agent against fake sockets and a mock HTTP
transport. No test calls a real API."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlsplit

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
from harbor_voice.providers.deepgram import register_all
from harbor_voice.providers.deepgram.agent import AgentSettings
from harbor_voice.providers.deepgram.agent import DeepgramAgentTransport
from harbor_voice.providers.deepgram.stt import DeepgramSTT
from harbor_voice.providers.deepgram.tts import DeepgramTTS
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import InputTranscript
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import ToolCall

pytestmark = pytest.mark.unit

AGENT = PartyId("agent")
FRAME = bytes(960)  # 20 ms of canonical audio


class FakeSocket:
    """A Deepgram websocket: hands out queued messages, records what it is sent."""

    def __init__(self, *messages: dict[str, Any] | bytes | None) -> None:
        self.inbox: asyncio.Queue[str | bytes | None] = asyncio.Queue()
        for message in messages:
            self.push(message)
        self.sent: list[Any] = []
        self.closed = False

    def push(self, message: dict[str, Any] | bytes | None) -> None:
        self.inbox.put_nowait(
            message
            if message is None or isinstance(message, bytes)
            else json.dumps(message)
        )

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data if isinstance(data, bytes) else json.loads(data))

    async def recv(self) -> str | bytes:
        raw = await self.inbox.get()
        assert raw is not None
        return raw

    def __aiter__(self) -> FakeSocket:
        return self

    async def __anext__(self) -> str | bytes:
        raw = await self.inbox.get()
        if raw is None:
            raise StopAsyncIteration
        return raw

    async def close(self) -> None:
        self.closed = True
        self.inbox.put_nowait(None)


def _connect(socket: FakeSocket, seen: list[tuple[str, dict[str, str]]]) -> Any:
    async def connect(url: str, *, additional_headers: dict[str, str]) -> FakeSocket:
        seen.append((url, additional_headers))
        return socket

    return connect


def _refused(status: int) -> Any:
    class Response:
        status_code = status

    class InvalidStatus(Exception):
        response = Response()

    async def connect(url: str, *, additional_headers: dict[str, str]) -> FakeSocket:
        raise InvalidStatus(f"server rejected WebSocket connection: HTTP {status}")

    return connect


def _frames(n: int, fmt: AudioFormat | None = None) -> list[AudioFrame]:
    fmt = fmt or AudioFormat()
    size = fmt.frame_bytes()
    return [
        AudioFrame(
            pcm=bytes(size), seq=i, party=PartyId("caller"), t_produced=0.0, fmt=fmt
        )
        for i in range(n)
    ]


def _results(text: str, *, final: bool, confidence: float = 0.9) -> dict[str, Any]:
    return {
        "type": "Results",
        "is_final": final,
        "channel": {
            "alternatives": [
                {
                    "transcript": text,
                    "confidence": confidence,
                    "words": [
                        {"word": w, "start": 0.0, "end": 0.1} for w in text.split()
                    ],
                }
            ]
        },
    }


# ---------------------------------------------------------------------- STT


class TestSTT:
    async def test_streams_the_turn_then_closes_and_assembles_the_transcript(
        self,
    ) -> None:
        socket = FakeSocket(
            _results("I need", final=False),
            _results("I need to change", final=True, confidence=0.8),
            _results("my flight", final=False),
            _results("my flight", final=True, confidence=1.0),
            {"type": "Metadata", "duration": 0.06},
            None,
        )
        seen: list[tuple[str, dict[str, str]]] = []
        stt = DeepgramSTT(api_key="k", connect=_connect(socket, seen))

        out = [t async for t in stt.transcribe(_frames(3))]

        ((url, headers),) = seen
        assert headers == {"Authorization": "Token k"}
        parts = urlsplit(url)
        assert f"{parts.scheme}://{parts.netloc}{parts.path}" == (
            "wss://api.deepgram.com/v1/listen"
        )
        assert {k: v[0] for k, v in parse_qs(parts.query).items()} == {
            "model": "nova-3",
            "language": "en",
            "encoding": "linear16",
            "sample_rate": "24000",
            "channels": "1",
            "interim_results": "true",
            "smart_format": "true",
        }
        assert socket.sent == [FRAME, FRAME, FRAME, {"type": "CloseStream"}]
        assert [(t.text, t.is_final) for t in out] == [
            ("I need", False),
            ("I need to change my flight", False),
            ("I need to change my flight", True),
        ]
        assert out[-1].confidence == pytest.approx(0.9)
        assert out[-1].audio_duration_s == pytest.approx(0.06)
        assert socket.closed

    async def test_mulaw_at_8k_is_named_as_deepgram_names_it(self) -> None:
        socket = FakeSocket(None)
        seen: list[tuple[str, dict[str, str]]] = []
        stt = DeepgramSTT(api_key="k", connect=_connect(socket, seen))
        fmt = AudioFormat(sample_rate_hz=8_000, encoding="mulaw")

        (final,) = [t async for t in stt.transcribe(_frames(2, fmt))]

        query = parse_qs(urlsplit(seen[0][0]).query)
        assert query["encoding"] == ["mulaw"]
        assert query["sample_rate"] == ["8000"]
        assert final.text == ""
        assert final.confidence is None
        assert final.audio_duration_s == pytest.approx(0.04)

    async def test_no_audio_is_an_empty_final_without_connecting(self) -> None:
        stt = DeepgramSTT(api_key="k", connect=_refused(500))
        (final,) = [t async for t in stt.transcribe([])]
        assert final.is_final and final.text == ""

    async def test_an_error_message_fails_the_call(self) -> None:
        socket = FakeSocket({"type": "Error", "variant": "x", "description": "bad"})
        stt = DeepgramSTT(api_key="k", connect=_connect(socket, []))
        with pytest.raises(ProviderError) as caught:
            [t async for t in stt.transcribe(_frames(1))]
        assert caught.value.kind is ProviderFailure.MALFORMED
        assert socket.closed

    @pytest.mark.parametrize(
        ("status", "kind"),
        [
            (401, ProviderFailure.AUTH),
            (429, ProviderFailure.THROTTLED),
            (400, ProviderFailure.MALFORMED),
            (503, ProviderFailure.STREAM_BROKEN),
        ],
    )
    async def test_a_refused_handshake_maps_its_status(
        self, status: int, kind: ProviderFailure
    ) -> None:
        stt = DeepgramSTT(api_key="k", connect=_refused(status))
        with pytest.raises(ProviderError) as caught:
            [t async for t in stt.transcribe(_frames(1))]
        assert caught.value.kind is kind

    async def test_a_missing_key_is_an_auth_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        stt = DeepgramSTT(connect=_refused(500))
        with pytest.raises(ProviderError) as caught:
            [t async for t in stt.transcribe(_frames(1))]
        assert caught.value.kind is ProviderFailure.AUTH


# ---------------------------------------------------------------------- TTS


def _tts(handler: Any, **kw: Any) -> tuple[DeepgramTTS, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return DeepgramTTS(client=client, api_key="k", **kw), seen


class TestTTS:
    async def test_asks_for_raw_audio_in_its_format_and_yields_20ms_chunks(
        self,
    ) -> None:
        audio = bytes(960 * 2 + 101)  # two frames, then 50 samples and a stray byte
        tts, seen = _tts(
            lambda _: httpx.Response(200, content=audio, headers={"dg-char-count": "5"})
        )

        chunks = [c async for c in tts.synthesize("Hello", party=AGENT)]

        (request,) = seen
        assert request.url.path == "/v1/speak"
        assert dict(request.url.params) == {
            "model": "aura-2-thalia-en",
            "encoding": "linear16",
            "sample_rate": "24000",
            "container": "none",
        }
        assert request.headers["authorization"] == "Token k"
        assert json.loads(request.content) == {"text": "Hello"}
        assert [len(c) for c in chunks] == [960, 960, 100]
        assert tts.characters_billed == 5

    async def test_telephony_mulaw_is_requested_natively(self) -> None:
        fmt = AudioFormat(sample_rate_hz=8_000, encoding="mulaw")
        tts, seen = _tts(
            lambda _: httpx.Response(200, content=bytes(160)), audio_format=fmt
        )
        assert [len(c) async for c in tts.synthesize("Hi", party=AGENT)] == [160]
        assert seen[0].url.params["encoding"] == "mulaw"
        assert seen[0].url.params["sample_rate"] == "8000"
        assert tts.characters_billed == 2  # no header: the text's length

    def test_a_rate_deepgram_does_not_produce_is_refused_up_front(self) -> None:
        with pytest.raises(ValueError, match="mulaw"):
            DeepgramTTS(
                audio_format=AudioFormat(sample_rate_hz=24_000, encoding="mulaw")
            )

    async def test_empty_text_sends_nothing(self) -> None:
        tts, seen = _tts(lambda _: httpx.Response(500))
        assert [c async for c in tts.synthesize("", party=AGENT)] == []
        assert seen == []

    @pytest.mark.parametrize(
        ("status", "body", "kind"),
        [
            (401, b"no", ProviderFailure.AUTH),
            (429, b"slow down", ProviderFailure.THROTTLED),
            (400, b"bad model", ProviderFailure.MALFORMED),
            (502, b"", ProviderFailure.STREAM_BROKEN),
            (200, b"", ProviderFailure.MALFORMED),
        ],
    )
    async def test_failures_are_classified(
        self, status: int, body: bytes, kind: ProviderFailure
    ) -> None:
        tts, _ = _tts(lambda _: httpx.Response(status, content=body))
        with pytest.raises(ProviderError) as caught:
            [c async for c in tts.synthesize("Hello", party=AGENT)]
        assert caught.value.kind is kind


# -------------------------------------------------------------------- agent

SPEC = ToolSpec(
    name="get_user",
    description="Look a user up",
    input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
)


def _router(handler: Any = None) -> ToolRouter:
    def get_user(id: str = "u1") -> dict[str, str]:
        return {"id": id}

    return ToolRouter(
        MappingExecutor(specs=[SPEC], handlers={"get_user": handler or get_user}),
        sleep=lambda _s: asyncio.sleep(0),
    )


def _agent(
    socket: FakeSocket, *, tools: ToolRouter | None = None, **settings: Any
) -> DeepgramAgentTransport:
    return DeepgramAgentTransport(
        party=AGENT,
        system_prompt="You are an airline agent.",
        tools=tools,
        settings=AgentSettings(**settings),
        api_key="k",
        connect=_connect(socket, []),
    )


def _session(*messages: dict[str, Any] | bytes) -> FakeSocket:
    return FakeSocket(
        {"type": "Welcome", "request_id": "req-1"},
        {"type": "SettingsApplied"},
        *messages,
    )


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def _events(transport: DeepgramAgentTransport) -> list[Any]:
    await _settle()
    await transport.close()
    return [e async for e in transport.events()]


CONFIG = SessionConfig(party=AGENT)


class TestAgent:
    async def test_opens_with_the_documented_settings(self) -> None:
        socket = _session()
        transport = _agent(socket, tools=_router(), greeting="Hi there.")
        await transport.open(CONFIG)

        assert transport.request_id == "req-1"
        assert socket.sent == [
            {
                "type": "Settings",
                "audio": {
                    "input": {"encoding": "linear16", "sample_rate": 24000},
                    "output": {
                        "encoding": "linear16",
                        "sample_rate": 24000,
                        "container": "none",
                    },
                },
                "agent": {
                    "listen": {
                        "provider": {
                            "type": "deepgram",
                            "model": "flux-general-en",
                            "version": "v2",
                        }
                    },
                    "think": {
                        "provider": {"type": "open_ai", "model": "gpt-5.6-luna"},
                        "prompt": "You are an airline agent.",
                        "functions": [
                            {
                                "name": "get_user",
                                "description": "Look a user up",
                                "parameters": SPEC.input_schema,
                            }
                        ],
                    },
                    "speak": {
                        "provider": {
                            "type": "deepgram",
                            "model": "aura-2-thalia-en",
                            "version": "v1",
                        }
                    },
                    "greeting": "Hi there.",
                },
            }
        ]
        assert transport.check_contract() == []
        await transport.close()
        assert socket.closed

    async def test_rejected_settings_fail_the_open_and_drop_the_socket(self) -> None:
        socket = FakeSocket(
            {"type": "Welcome", "request_id": "r"},
            {"type": "Error", "code": "INVALID_SETTINGS", "description": "no"},
        )
        transport = _agent(socket)
        with pytest.raises(ProviderError) as caught:
            await transport.open(CONFIG)
        assert caught.value.kind is ProviderFailure.MALFORMED
        assert "INVALID_SETTINGS" in str(caught.value)
        assert socket.closed and not transport.is_open

    async def test_a_refused_key_is_an_auth_failure(self) -> None:
        transport = DeepgramAgentTransport(
            party=AGENT, api_key="k", connect=_refused(401)
        )
        with pytest.raises(ProviderError) as caught:
            await transport.open(CONFIG)
        assert caught.value.kind is ProviderFailure.AUTH

    async def test_audio_goes_up_raw_and_other_formats_are_refused(self) -> None:
        socket = _session()
        transport = _agent(socket)
        await transport.open(CONFIG)
        await transport.tx(_frames(1)[0])
        assert socket.sent[-1] == FRAME
        with pytest.raises(ProviderError):
            await transport.tx(_frames(1, AudioFormat(sample_rate_hz=16_000))[0])
        await transport.close()

    async def test_events_are_tagged_by_agent_turn_and_barge_in_is_reported(
        self,
    ) -> None:
        transport = _agent(
            _session(
                {"type": "ConversationText", "role": "assistant", "content": "Hello."},
                {"type": "AgentStartedSpeaking", "total_latency": 0.5},
                b"\x01\x00",
                {"type": "ConversationText", "role": "assistant", "content": "How?"},
                b"\x02\x00",
                {"type": "UserStartedSpeaking"},
                {"type": "ConversationText", "role": "user", "content": "Wait."},
                b"\x03\x00",
                {"type": "AgentAudioDone"},
                b"\x04\x00",
            )
        )
        await transport.open(CONFIG)
        events = await _events(transport)

        assert events[:5] == [
            OutputTranscript("agent-1", "Hello.", events[0].t),
            OutputAudio(b"\x01\x00", "agent-1", events[1].t),
            OutputTranscript("agent-1", " How?", events[2].t),
            OutputAudio(b"\x02\x00", "agent-1", events[3].t),
            InputSpeech(True, None, events[4].t),
        ]
        # The user's transcript is opt-in; the turns after it are new ones.
        assert [(type(e), getattr(e, "utterance", None)) for e in events[5:]] == [
            (OutputAudio, "agent-2"),
            (OutputAudio, "agent-3"),
        ]

    async def test_rx_yields_frames_in_the_session_format(self) -> None:
        transport = _agent(_session(FRAME, {"type": "AgentAudioDone"}, FRAME))
        await transport.open(CONFIG)
        await _settle()
        await transport.close()
        frames = [f async for f in transport.rx()]
        assert [f.seq for f in frames] == [0, 1]
        assert all(f.fmt == AudioFormat() and f.party == AGENT for f in frames)

    async def test_client_side_calls_are_answered_through_the_router(self) -> None:
        socket = _session(
            {
                "type": "FunctionCallRequest",
                "functions": [
                    {
                        "id": "fc_1",
                        "name": "get_user",
                        "arguments": '{"id": "u7"}',
                        "client_side": True,
                    },
                    {
                        "id": "fc_2",
                        "name": "lookup",
                        "arguments": "{}",
                        "client_side": False,
                    },
                ],
            },
            {"type": "ConversationText", "role": "user", "content": "Hi."},
        )
        transport = _agent(socket, tools=_router(), extended_events=True)
        await transport.open(CONFIG)
        events = await _events(transport)

        assert socket.sent[1:] == [
            {
                "type": "FunctionCallResponse",
                "id": "fc_1",
                "name": "get_user",
                "content": '{"id": "u7"}',
            }
        ]
        assert InputTranscript(None, "Hi.", events[0].t) in events
        (call,) = [e for e in events if isinstance(e, ToolCall)]
        assert (call.call_id, call.arguments, call.result) == (
            "fc_1",
            {"id": "u7"},
            {"id": "u7"},
        )

    async def test_a_call_without_tools_is_still_answered(self) -> None:
        socket = _session(
            {
                "type": "FunctionCallRequest",
                "functions": [
                    {"id": "fc_1", "name": "x", "arguments": "", "client_side": True}
                ],
            }
        )
        transport = _agent(socket)
        await transport.open(CONFIG)
        await _events(transport)
        assert socket.sent[1]["type"] == "FunctionCallResponse"
        assert "error" in json.loads(socket.sent[1]["content"])

    async def test_a_cancelled_call_gets_no_answer(self) -> None:
        release = asyncio.Event()

        async def slow(id: str = "u1") -> dict[str, str]:
            await release.wait()
            return {"id": id}

        socket = _session(
            {
                "type": "FunctionCallRequest",
                "functions": [
                    {
                        "id": "fc_1",
                        "name": "get_user",
                        "arguments": "{}",
                        "client_side": True,
                    }
                ],
            },
            {
                "type": "FunctionCallCancelled",
                "functions": [{"id": "fc_1", "name": "get_user"}],
            },
        )
        transport = _agent(socket, tools=_router(slow))
        await transport.open(CONFIG)
        await _settle()
        release.set()
        await _events(transport)
        assert [m for m in socket.sent if isinstance(m, dict)][1:] == []

    async def test_errors_after_open_are_kept_and_usage_is_session_seconds(
        self,
    ) -> None:
        transport = _agent(_session({"type": "Error", "code": "X", "description": "y"}))
        await transport.open(CONFIG)
        await _events(transport)
        assert transport.errors == [{"type": "Error", "code": "X", "description": "y"}]
        record = transport.usage_record()
        assert record.model == "gpt-5.6-luna"
        assert record.seconds is not None and record.seconds >= 0


def test_register_all_adds_the_three_providers() -> None:
    registry = ProviderRegistry()
    register_all(registry)
    assert {r.spec.name: r.spec.default_model for r in registry} == {
        "deepgram.stt": "nova-3",
        "deepgram.tts": "aura-2-thalia-en",
        "deepgram.agent": "gpt-5.6-luna",
    }


async def test_the_agent_sends_keepalives_while_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deepgram drops a session idle for about 10 s, e.g. during a tool call."""
    from harbor_voice.providers.deepgram import agent as dg_agent

    monkeypatch.setattr(dg_agent, "KEEPALIVE_S", 0.01)
    socket = _session()
    transport = _agent(socket)
    await transport.open(CONFIG)
    await asyncio.sleep(0.05)
    await transport.close()
    assert {"type": "KeepAlive"} in socket.sent
