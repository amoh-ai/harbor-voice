"""Meta Muse Voice Transcribe, offline: a fake realtime socket."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.providers import meta
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.meta import stt as meta_stt
from harbor_voice.providers.meta.stt import MetaSTT
from harbor_voice.providers.registry import ProviderRegistry

pytestmark = pytest.mark.unit

PARTY = PartyId("agent")
ACK = {"sessionId": "550e8400"}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr(meta_stt.asyncio, "sleep", instant)


def _frames(seconds: float, rate: int = 24_000) -> list[AudioFrame]:
    fmt = AudioFormat(sample_rate_hz=rate)
    pcm = bytes(int(rate * seconds) * 2)
    return [AudioFrame(pcm=pcm, seq=0, party=PARTY, t_produced=0.0, fmt=fmt)]


class Closed(Exception):
    """Stands in for ``websockets.ConnectionClosedError``."""

    def __init__(self, code: int) -> None:
        super().__init__(f"closed {code}")
        self.rcvd = type("Close", (), {"code": code})()


class FakeSocket:
    """A Meta server: acknowledges the handshake, answers after ``endStream``."""

    def __init__(
        self,
        *events: dict[str, Any],
        ack: dict[str, Any] = ACK,
        close: Exception | None = None,
    ) -> None:
        self.ack = ack
        self.events = events
        self.close_with = close
        self.inbox: asyncio.Queue[str | None] = asyncio.Queue()
        self.text: list[dict[str, Any]] = []
        self.audio = b""
        self.closed = False

    async def send(self, data: str | bytes) -> None:
        if isinstance(data, bytes):
            self.audio += data
            return
        message = json.loads(data)
        self.text.append(message)
        if message.get("type") == "endStream":
            for event in self.events:
                self.inbox.put_nowait(json.dumps(event))
            self.inbox.put_nowait(None)

    async def recv(self) -> str:
        return json.dumps(self.ack)

    def __aiter__(self) -> FakeSocket:
        return self

    async def __anext__(self) -> str:
        raw = await self.inbox.get()
        if raw is None:
            if self.close_with is not None:
                raise self.close_with
            raise StopAsyncIteration
        return raw

    async def close(self) -> None:
        self.closed = True


def _stt(socket: FakeSocket, **kw: Any) -> tuple[MetaSTT, list[str]]:
    urls: list[str] = []

    async def connect(url: str) -> FakeSocket:
        urls.append(url)
        return socket

    return MetaSTT(api_key="test-key", connect=connect, **kw), urls


async def _collect(stt: MetaSTT, frames: list[AudioFrame]) -> list[Any]:
    return [t async for t in stt.transcribe(frames)]


TURN = (
    {
        "type": "transcript",
        "transcript": "how is the",
        "final": False,
        "audioProcessedMs": 800,
    },
    {
        "type": "transcript",
        "transcript": "how is the weather",
        "final": False,
        "audioProcessedMs": 1600,
    },
    {
        "type": "transcript",
        "transcript": "How is the weather?",
        "final": True,
        "audioProcessedMs": 2500,
    },
)


class TestSession:
    async def test_the_key_travels_in_the_handshake(self) -> None:
        socket = FakeSocket(*TURN)
        stt, urls = _stt(socket, language_bias=["English", "Hindi"])
        await _collect(stt, _frames(2.5))

        assert urls == ["wss://api.meta.ai/v1/asr/realtime"]
        handshake, end = socket.text
        assert handshake == {
            "authorization": {"accessToken": "Bearer test-key"},
            "audioEncoding": "PCM_24KHZ",
            "model": "muse-voice-transcribe-1.0",
            "mode": "PUSH_TO_TALK",
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
            "languageBias": ["English", "Hindi"],
        }
        assert end == {"type": "endStream"}
        assert socket.audio == bytes(120_000)  # 2.5 s at 24 kHz, unchanged
        assert socket.closed

    async def test_partials_replace_then_one_final_and_whole_billed_seconds(
        self,
    ) -> None:
        stt, _ = _stt(FakeSocket(*TURN))
        out = await _collect(stt, _frames(2.5))
        assert [(t.text, t.is_final) for t in out] == [
            ("how is the", False),
            ("how is the weather", False),
            ("How is the weather?", True),
        ]
        assert out[-1].audio_duration_s == pytest.approx(2.5)
        assert stt.usage.audio_seconds == 2.0

    async def test_16k_is_sent_as_16k_and_8k_is_resampled_to_24k(self) -> None:
        socket = FakeSocket(*TURN)
        stt, _ = _stt(socket)
        await _collect(stt, _frames(1.0, rate=16_000))
        assert socket.text[0]["audioEncoding"] == "PCM_16KHZ"
        assert len(socket.audio) == 32_000

        socket = FakeSocket(*TURN)
        stt, _ = _stt(socket)
        await _collect(stt, _frames(1.0, rate=8_000))
        assert socket.text[0]["audioEncoding"] == "PCM_24KHZ"
        assert len(socket.audio) == 48_000

    async def test_the_key_falls_back_to_model_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("META_API_KEY", raising=False)
        monkeypatch.setenv("MODEL_API_KEY", "from-env")
        socket = FakeSocket(*TURN)

        async def connect(url: str) -> FakeSocket:
            return socket

        await _collect(MetaSTT(connect=connect), _frames(0.2))
        assert socket.text[0]["authorization"]["accessToken"] == "Bearer from-env"


class TestFailures:
    async def test_no_key_is_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("META_API_KEY", raising=False)
        monkeypatch.delenv("MODEL_API_KEY", raising=False)
        with pytest.raises(ProviderError) as info:
            await _collect(MetaSTT(connect=None), _frames(0.2))
        assert info.value.kind is ProviderFailure.AUTH

    async def test_a_rejected_handshake_is_classified_by_error_type(self) -> None:
        ack = {
            "type": "error",
            "message": "bad key",
            "sessionId": "",
            "errorType": "authentication_error",
        }
        stt, _ = _stt(FakeSocket(ack=ack))
        with pytest.raises(ProviderError) as info:
            await _collect(stt, _frames(0.2))
        assert info.value.kind is ProviderFailure.AUTH

    async def test_an_unexpected_ack_is_malformed(self) -> None:
        stt, _ = _stt(FakeSocket(ack={"hello": 1}))
        with pytest.raises(ProviderError) as info:
            await _collect(stt, _frames(0.2))
        assert info.value.kind is ProviderFailure.MALFORMED

    async def test_rate_limit_close_is_throttled(self) -> None:
        stt, _ = _stt(FakeSocket(close=Closed(1013)))
        with pytest.raises(ProviderError) as info:
            await _collect(stt, _frames(0.2))
        assert info.value.kind is ProviderFailure.THROTTLED

    async def test_a_clean_close_without_a_final_is_a_broken_stream(self) -> None:
        stt, _ = _stt(FakeSocket(TURN[0]))
        with pytest.raises(ProviderError) as info:
            await _collect(stt, _frames(0.2))
        assert info.value.kind is ProviderFailure.STREAM_BROKEN


def test_register_all_adds_the_stt() -> None:
    registry = ProviderRegistry()
    meta.register_all(registry)
    entry, model = registry.resolve("meta.stt")
    assert entry.spec.modality is Modality.STT
    assert model == "muse-voice-transcribe-1.0"
    assert entry.spec.credential_provider == "meta"
