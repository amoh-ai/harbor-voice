"""Sarvam STT and TTS, offline: a fake socket and ``httpx.MockTransport``."""

from __future__ import annotations

import base64
import json
import urllib.parse
from typing import Any

import httpx
import pytest

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.providers import sarvam
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.providers.sarvam import stt as sarvam_stt
from harbor_voice.providers.sarvam.stt import SarvamSTT
from harbor_voice.providers.sarvam.tts import SarvamTTS
from tests.unit.voice import fakes
from tests.unit.voice.fakes import Closed
from tests.unit.voice.fakes import refusing

pytestmark = pytest.mark.unit

PARTY = PartyId("agent")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr(sarvam_stt.asyncio, "sleep", instant)


def _frames(seconds: float, rate: int = 24_000) -> list[AudioFrame]:
    fmt = AudioFormat(sample_rate_hz=rate)
    return [
        AudioFrame(
            pcm=bytes(int(rate * seconds) * 2),
            seq=0,
            party=PARTY,
            t_produced=0.0,
            fmt=fmt,
        )
    ]


class FakeSocket(fakes.FakeSocket):
    """A Sarvam server: answers once the client sends ``end``."""

    def __init__(self, *events: dict[str, Any], close: Exception | None = None) -> None:
        super().__init__(close_with=close)
        self.events = events

    async def send(self, data: str | bytes) -> None:
        await super().send(data)
        if self.sent[-1]["event"] == "end":
            for event in self.events:
                self.push(event)
            self.push(None)


def _stt(socket: FakeSocket, **kw: Any) -> tuple[SarvamSTT, dict[str, Any]]:
    seen: dict[str, Any] = {}

    async def connect(url: str, *, additional_headers: dict[str, str]) -> FakeSocket:
        seen["url"] = url
        seen["headers"] = additional_headers
        return socket

    return SarvamSTT(api_key="test-key", connect=connect, **kw), seen


async def _collect(stt: SarvamSTT, frames: list[AudioFrame]) -> list[Any]:
    return [t async for t in stt.transcribe(frames)]


FINAL = (
    {"event": "session.begin", "request_id": "r1"},
    {"event": "transcript.partial", "utterance_idx": 0, "text": "change my"},
    {"event": "transcript.partial", "utterance_idx": 0, "text": "change my flight"},
    {"event": "transcript.final", "utterance_idx": 0, "text": "Change my flight."},
    {"event": "session.end", "request_id": "r1", "audio_duration_s": "1.02"},
)


class TestSTT:
    async def test_the_session_is_one_manual_utterance(self) -> None:
        socket = FakeSocket(*FINAL)
        stt, seen = _stt(socket)
        await _collect(stt, _frames(1.0))

        url = urllib.parse.urlsplit(seen["url"])
        assert url.netloc == "api.sarvam.ai"
        assert url.path == "/speech-to-text-realtime/ws"
        assert dict(urllib.parse.parse_qsl(url.query)) == {
            "model": "saaras:v4",
            "language_code": "en-IN",
            "endpointing": "manual",
            "stream_type": "fast",
            "encoding": "linear16",
            "sample_rate": "16000",
        }
        assert seen["headers"] == {"Api-Subscription-Key": "test-key"}
        kinds = [m["event"] for m in socket.sent]
        assert kinds[0] == "speech_start"
        assert kinds[-2:] == ["speech_end", "end"]
        assert set(kinds[1:-2]) == {"audio_input"}
        audio = b"".join(base64.b64decode(m["audio"]) for m in socket.sent[1:-2])
        assert len(audio) == 32_000  # 1 s of 24 kHz resampled to 16 kHz
        assert socket.closed

    async def test_partials_then_one_final_and_the_billed_seconds(self) -> None:
        stt, _ = _stt(FakeSocket(*FINAL))
        out = await _collect(stt, _frames(1.0))

        assert [(t.text, t.is_final) for t in out] == [
            ("change my", False),
            ("change my flight", False),
            ("Change my flight.", True),
        ]
        assert out[-1].audio_duration_s == pytest.approx(1.0)
        assert stt.usage.audio_seconds == pytest.approx(1.02)

    async def test_16k_audio_goes_out_unchanged(self) -> None:
        socket = FakeSocket(*FINAL)
        stt, _ = _stt(socket)
        await _collect(stt, _frames(0.5, rate=16_000))
        audio = b"".join(base64.b64decode(m["audio"]) for m in socket.sent[1:-2])
        assert audio == bytes(16_000)

    async def test_no_audio_is_an_empty_final_without_a_connection(self) -> None:
        stt = SarvamSTT(api_key="k", connect=None)
        (only,) = await _collect(stt, [])
        assert only.is_final and only.text == ""

    async def test_a_fatal_error_event_is_classified_by_its_status(self) -> None:
        stt, _ = _stt(
            FakeSocket(
                {
                    "event": "error",
                    "code": "invalid_api_key_error",
                    "is_fatal": True,
                    "message": "bad key",
                    "status_code": 403,
                }
            )
        )
        with pytest.raises(ProviderError) as info:
            await _collect(stt, _frames(0.2))
        assert info.value.kind is ProviderFailure.AUTH

    async def test_a_documented_close_code_is_classified(self) -> None:
        stt, _ = _stt(FakeSocket(close=Closed(1003)))
        with pytest.raises(ProviderError) as info:
            await _collect(stt, _frames(0.2))
        assert info.value.kind is ProviderFailure.THROTTLED

    async def test_a_clean_close_without_a_final_is_a_broken_stream(self) -> None:
        stt, _ = _stt(FakeSocket({"event": "session.begin", "request_id": "r"}))
        with pytest.raises(ProviderError) as info:
            await _collect(stt, _frames(0.2))
        assert info.value.kind is ProviderFailure.STREAM_BROKEN

    @pytest.mark.parametrize(
        ("close", "kind"),
        [
            (ConnectionResetError("reset by peer"), ProviderFailure.STREAM_BROKEN),
            (TimeoutError(), ProviderFailure.TIMEOUT),
        ],
    )
    async def test_a_dropped_socket_is_classified_by_type(
        self, close: Exception, kind: ProviderFailure
    ) -> None:
        stt, _ = _stt(FakeSocket(close=close))
        with pytest.raises(ProviderError) as info:
            await _collect(stt, _frames(0.2))
        assert info.value.kind is kind

    async def test_a_refused_connection_is_not_a_timeout(self) -> None:
        stt = SarvamSTT(api_key="k", connect=refusing(ConnectionRefusedError()))
        with pytest.raises(ProviderError) as info:
            await _collect(stt, _frames(0.2))
        assert info.value.kind is ProviderFailure.STREAM_BROKEN

    async def test_a_missing_key_fails_before_connecting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        stt = SarvamSTT(connect=None)
        with pytest.raises(ProviderError) as info:
            await _collect(stt, _frames(0.2))
        assert info.value.kind is ProviderFailure.AUTH


# ------------------------------------------------------------------- TTS


def _tts(handler, **kw: Any) -> tuple[SarvamTTS, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return SarvamTTS(client=client, api_key="test-key", **kw), seen


async def _speak(tts: SarvamTTS, text: str = "Namaste") -> list[bytes]:
    return [c async for c in tts.synthesize(text, party=PARTY)]


class TestTTS:
    async def test_the_request_streams_linear16_at_the_output_rate(self) -> None:
        tts, seen = _tts(lambda _: httpx.Response(200, content=bytes(4_801)))
        chunks = await _speak(tts, "Hello there")

        (request,) = seen
        assert str(request.url) == "https://api.sarvam.ai/text-to-speech/stream"
        assert request.headers["api-subscription-key"] == "test-key"
        assert json.loads(request.content) == {
            "text": "Hello there",
            "model": "bulbul:v4-flash",
            "language_code": "en-IN",
            "speech_sample_rate": 24_000,
            "output_audio_codec": "linear16",
        }
        assert [len(c) for c in chunks] == [960] * 5  # the odd byte is dropped
        assert tts.usage.characters == len("Hello there")

    async def test_speaker_and_language_are_sent_when_set(self) -> None:
        tts, seen = _tts(
            lambda _: httpx.Response(200, content=bytes(960)),
            speaker="simran_en_customer",
            language_code="hi-IN",
            audio_format=AudioFormat(sample_rate_hz=16_000),
        )
        await _speak(tts)
        body = json.loads(seen[0].content)
        assert body["speaker"] == "simran_en_customer"
        assert body["language_code"] == "hi-IN"
        assert body["speech_sample_rate"] == 16_000

    @pytest.mark.parametrize(
        ("status", "kind"),
        [
            (403, ProviderFailure.AUTH),
            (429, ProviderFailure.THROTTLED),
            (503, ProviderFailure.STREAM_BROKEN),
            (422, ProviderFailure.MALFORMED),
        ],
    )
    async def test_http_errors_are_classified(
        self, status: int, kind: ProviderFailure
    ) -> None:
        body = {"error": {"message": "no", "code": "x"}}
        tts, _ = _tts(lambda _: httpx.Response(status, json=body))
        with pytest.raises(ProviderError) as info:
            await _speak(tts)
        assert info.value.kind is kind

    async def test_an_empty_200_is_malformed(self) -> None:
        tts, _ = _tts(lambda _: httpx.Response(200, content=b""))
        with pytest.raises(ProviderError) as info:
            await _speak(tts)
        assert info.value.kind is ProviderFailure.MALFORMED

    def test_a_rate_streaming_cannot_produce_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot stream"):
            SarvamTTS(audio_format=AudioFormat(sample_rate_hz=48_000))


def test_register_all_adds_both_slots() -> None:
    registry = ProviderRegistry()
    sarvam.register_all(registry)
    assert registry.lookup("sarvam.stt").spec.modality is Modality.STT
    assert registry.lookup("sarvam.tts").spec.modality is Modality.TTS
    assert registry.resolve("sarvam.tts")[1] == "bulbul:v4-flash"
    assert registry.lookup("sarvam.stt").spec.capabilities.emits_partials
