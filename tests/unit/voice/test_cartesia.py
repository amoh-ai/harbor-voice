"""Cartesia TTS against ``httpx.MockTransport`` and STT against a fake socket.
No test calls the real API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.perturbation.persona import VoiceSelection
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.cartesia import register_all
from harbor_voice.providers.cartesia import stt as cstt
from harbor_voice.providers.cartesia import tts as ctts
from harbor_voice.providers.registry import default_registry

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
FMT = AudioFormat()  # 24 kHz pcm16
FRAME = FMT.frame_bytes()


def _tts(handler, **kw) -> tuple[ctts.CartesiaTTS, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    kw.setdefault("voice_id", "voice-abc")
    return ctts.CartesiaTTS(client=client, api_key="test-key", **kw), seen


async def _drip(data: bytes, step: int) -> AsyncIterator[bytes]:
    for start in range(0, len(data), step):
        yield data[start : start + step]


async def _speak(tts: ctts.CartesiaTTS, text: str = "Hello there.") -> list[bytes]:
    return [c async for c in tts.synthesize(text, party=CALLER)]


class TestTTS:
    async def test_the_request_follows_the_api_reference(self) -> None:
        tts, seen = _tts(lambda _: httpx.Response(200, content=bytes(FRAME)))
        await _speak(tts)

        (request,) = seen
        assert request.url == "https://api.cartesia.ai/tts/bytes"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["cartesia-version"] == "2026-08-14"
        assert json.loads(request.content) == {
            "model_id": "sonic-3.6",
            "transcript": "Hello there.",
            "voice": "voice-abc",
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": 24_000,
            },
        }

    async def test_a_telephony_format_is_requested_as_such(self) -> None:
        tts, seen = _tts(
            lambda _: httpx.Response(200, content=bytes(160)),
            audio_format=AudioFormat(sample_rate_hz=8_000, encoding="mulaw"),
        )
        await _speak(tts)
        assert json.loads(seen[0].content)["output_format"] == {
            "container": "raw",
            "encoding": "pcm_mulaw",
            "sample_rate": 8_000,
        }

    async def test_a_ragged_stream_is_reframed_on_sample_boundaries(self) -> None:
        audio = bytes(range(256)) * 15 + b"\x01\x02\x03"  # 3843 bytes, odd
        tts, _ = _tts(lambda _: httpx.Response(200, content=_drip(audio, 333)))
        chunks = await _speak(tts)

        assert all(len(c) == FRAME for c in chunks[:-1])
        assert len(chunks[-1]) % 2 == 0
        assert b"".join(chunks) == audio[:-1]  # the stray trailing byte dropped
        assert tts.usage.characters == len("Hello there.")

    async def test_empty_text_sends_nothing(self) -> None:
        tts, seen = _tts(lambda _: httpx.Response(200, content=bytes(FRAME)))
        assert await _speak(tts, "") == []
        assert seen == []

    async def test_an_unsupported_rate_is_refused_up_front(self) -> None:
        with pytest.raises(ValueError, match="cannot produce"):
            ctts.CartesiaTTS(audio_format=AudioFormat(sample_rate_hz=32_000))

    @pytest.mark.parametrize(
        ("status", "kind"),
        [
            (401, ProviderFailure.AUTH),
            (429, ProviderFailure.THROTTLED),
            (500, ProviderFailure.STREAM_BROKEN),
            (400, ProviderFailure.MALFORMED),
        ],
    )
    async def test_http_errors_map_to_failures(
        self, status: int, kind: ProviderFailure
    ) -> None:
        tts, _ = _tts(lambda _: httpx.Response(status, text='{"title": "no"}'))
        with pytest.raises(ProviderError) as info:
            await _speak(tts)
        assert info.value.kind is kind

    async def test_an_empty_200_is_malformed(self) -> None:
        tts, _ = _tts(lambda _: httpx.Response(200, content=b""))
        with pytest.raises(ProviderError) as info:
            await _speak(tts)
        assert info.value.kind is ProviderFailure.MALFORMED

    async def test_a_timeout_is_a_timeout(self) -> None:
        def hang(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        tts, _ = _tts(hang)
        with pytest.raises(ProviderError) as info:
            await _speak(tts)
        assert info.value.kind is ProviderFailure.TIMEOUT

    async def test_a_missing_key_is_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
        tts = ctts.CartesiaTTS(voice_id="v")
        with pytest.raises(ProviderError) as info:
            await _speak(tts)
        assert info.value.kind is ProviderFailure.AUTH

    async def test_no_voice_is_an_error_not_a_default(self) -> None:
        tts, seen = _tts(lambda _: httpx.Response(200), voice_id=None)
        with pytest.raises(ValueError, match="no voice selected"):
            await _speak(tts)
        assert seen == []

    def test_only_an_explicit_voice_id_is_honoured(self) -> None:
        tts, _ = _tts(lambda _: httpx.Response(200), voice_id=None)
        assert tts.select_voice(VoiceSelection()) is False
        assert tts.select_voice(VoiceSelection(voice_id="v2")) is True
        assert tts.voice_id == "v2"


# ------------------------------------------------------------------- STT


class FakeSTTSocket:
    """A Cartesia STT server: plays ``during`` once audio arrives, and
    ``after`` once the client sends ``close``."""

    def __init__(self, during: list[dict[str, Any]], after: list[dict[str, Any]]):
        self.during = during
        self.after = after
        self.inbox: asyncio.Queue[str | None] = asyncio.Queue()
        self.sent: list[bytes | str] = []
        self.closed = False

    async def send(self, data: bytes | str) -> None:
        if not self.sent:
            for event in self.during:
                self.inbox.put_nowait(json.dumps(event))
        self.sent.append(data)
        if data == "close":
            for event in self.after:
                self.inbox.put_nowait(json.dumps(event))
            self.inbox.put_nowait(None)

    def __aiter__(self) -> FakeSTTSocket:
        return self

    async def __anext__(self) -> str:
        raw = await self.inbox.get()
        if raw is None:
            raise StopAsyncIteration
        return raw

    async def close(self) -> None:
        self.closed = True


def _stt(socket: FakeSTTSocket) -> tuple[cstt.CartesiaSTT, list[tuple[str, dict]]]:
    calls: list[tuple[str, dict]] = []

    async def connect(url: str, *, additional_headers: dict[str, str]) -> Any:
        calls.append((url, additional_headers))
        return socket

    return cstt.CartesiaSTT(api_key="test-key", connect=connect), calls


def _frames(seconds: float, fmt: AudioFormat = AudioFormat(sample_rate_hz=16_000)):
    n = int(seconds * 1000 / 20)
    return [
        AudioFrame(
            pcm=bytes(fmt.frame_bytes()),
            seq=i,
            party=CALLER,
            t_produced=i * 0.02,
            fmt=fmt,
        )
        for i in range(n)
    ]


def _transcript(text: str, is_final: bool) -> dict[str, Any]:
    return {"type": "transcript", "is_final": is_final, "request_id": "r", "text": text}


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def instant(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(cstt.asyncio, "sleep", instant)
    return waits


class TestSTT:
    async def test_the_session_follows_the_api_reference(
        self, slept: list[float]
    ) -> None:
        socket = FakeSTTSocket(
            during=[_transcript("How are", False)],
            after=[
                _transcript("How are you", True),
                _transcript(" doing?", True),
                {"type": "flush_done", "request_id": "r"},
                {"type": "done", "request_id": "r"},
            ],
        )
        stt, calls = _stt(socket)
        out = [t async for t in stt.transcribe(_frames(0.5))]

        ((url, headers),) = calls
        assert url == (
            "wss://api.cartesia.ai/stt/websocket?model=ink-2&encoding=pcm_s16le"
            "&sample_rate=16000&cartesia_version=2026-08-14"
        )
        assert headers == {"X-API-Key": "test-key"}
        audio = [m for m in socket.sent if isinstance(m, bytes)]
        assert [len(m) for m in audio] == [3_200] * 5  # 100 ms at 16 kHz
        assert slept == [0.1] * 5  # paced to real time
        assert socket.sent[-2:] == ["finalize", "close"]
        assert socket.closed

        assert [(t.text, t.is_final) for t in out] == [
            ("How are", False),
            ("How are you", False),
            ("How are you doing?", False),
            ("How are you doing?", True),
        ]
        assert out[-1].audio_duration_s == pytest.approx(0.5)
        assert stt.usage.audio_seconds == pytest.approx(0.5)

    async def test_no_audio_sends_nothing(self) -> None:
        stt, calls = _stt(FakeSTTSocket([], []))
        out = [t async for t in stt.transcribe([])]
        assert [(t.text, t.is_final) for t in out] == [("", True)]
        assert calls == []

    async def test_an_error_event_maps_to_a_failure(self, slept: list[float]) -> None:
        error = {
            "type": "error",
            "status_code": 429,
            "title": "Too many concurrent requests",
            "message": "concurrency limit",
        }
        stt, _ = _stt(FakeSTTSocket([], [error]))
        with pytest.raises(ProviderError) as info:
            [t async for t in stt.transcribe(_frames(0.1))]
        assert info.value.kind is ProviderFailure.THROTTLED

    async def test_a_socket_that_ends_before_done_is_broken(
        self, slept: list[float]
    ) -> None:
        stt, _ = _stt(FakeSTTSocket([], [_transcript("hi", True)]))
        with pytest.raises(ProviderError) as info:
            [t async for t in stt.transcribe(_frames(0.1))]
        assert info.value.kind is ProviderFailure.STREAM_BROKEN

    async def test_a_rejected_handshake_maps_by_status(self) -> None:
        class Rejected(Exception):
            response = httpx.Response(401)

        async def connect(url: str, *, additional_headers: dict[str, str]) -> Any:
            raise Rejected("server rejected WebSocket connection: HTTP 401")

        stt = cstt.CartesiaSTT(api_key="k", connect=connect)
        with pytest.raises(ProviderError) as info:
            [t async for t in stt.transcribe(_frames(0.1))]
        assert info.value.kind is ProviderFailure.AUTH


def test_both_register_with_honest_capabilities() -> None:
    registry = default_registry()
    register_all(registry)
    tts = registry.lookup("cartesia.tts").spec
    stt = registry.lookup("cartesia.stt").spec
    assert (tts.default_model, tts.capabilities.streams) == ("sonic-3.6", True)
    assert (stt.default_model, stt.capabilities.emits_partials) == ("ink-2", True)
    assert {tts.credential_provider, stt.credential_provider} == {"cartesia"}
