"""Fish Audio TTS and STT against ``httpx.MockTransport``. No test calls the
real API."""

from __future__ import annotations

import io
import json
import wave
from collections.abc import AsyncIterator

import httpx
import pytest

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.perturbation.persona import VoiceSelection
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.fishaudio import register_all
from harbor_voice.providers.fishaudio import stt as fstt
from harbor_voice.providers.fishaudio import tts as ftts
from harbor_voice.providers.registry import default_registry

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
FRAME = AudioFormat().frame_bytes()


def _client(handler) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(record)), seen


def _tts(handler, **kw) -> tuple[ftts.FishAudioTTS, list[httpx.Request]]:
    client, seen = _client(handler)
    kw.setdefault("reference_id", "voice-abc")
    return ftts.FishAudioTTS(client=client, api_key="test-key", **kw), seen


async def _drip(data: bytes, step: int) -> AsyncIterator[bytes]:
    for start in range(0, len(data), step):
        yield data[start : start + step]


async def _speak(tts: ftts.FishAudioTTS, text: str = "Hello there.") -> list[bytes]:
    return [c async for c in tts.synthesize(text, party=CALLER)]


class TestTTS:
    async def test_the_request_follows_the_api_reference(self) -> None:
        tts, seen = _tts(lambda _: httpx.Response(200, content=bytes(FRAME)))
        await _speak(tts)

        (request,) = seen
        assert request.url == "https://api.fish.audio/v1/tts"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["model"] == "s2.1-pro"
        assert json.loads(request.content) == {
            "text": "Hello there.",
            "reference_id": "voice-abc",
            "format": "pcm",
            "sample_rate": 24_000,
            "latency": "balanced",
        }

    async def test_a_ragged_stream_is_reframed_on_sample_boundaries(self) -> None:
        audio = bytes(range(256)) * 15 + b"\x01\x02\x03"
        tts, _ = _tts(lambda _: httpx.Response(200, content=_drip(audio, 333)))
        chunks = await _speak(tts)

        assert all(len(c) == FRAME for c in chunks[:-1])
        assert b"".join(chunks) == audio[:-1]

    async def test_usage_counts_utf8_bytes_the_billed_unit(self) -> None:
        tts, _ = _tts(lambda _: httpx.Response(200, content=bytes(FRAME)))
        await _speak(tts, "café")
        assert tts.usage.characters == 5

    @pytest.mark.parametrize(
        "fmt",
        [
            AudioFormat(sample_rate_hz=22_050),
            AudioFormat(sample_rate_hz=8_000, encoding="mulaw"),
        ],
    )
    def test_an_unsupported_format_is_refused_up_front(self, fmt: AudioFormat) -> None:
        with pytest.raises(ValueError, match="cannot produce"):
            ftts.FishAudioTTS(audio_format=fmt)

    @pytest.mark.parametrize(
        ("status", "kind"),
        [
            (401, ProviderFailure.AUTH),
            (402, ProviderFailure.AUTH),
            (429, ProviderFailure.THROTTLED),
            (503, ProviderFailure.STREAM_BROKEN),
            (400, ProviderFailure.MALFORMED),
        ],
    )
    async def test_http_errors_map_to_failures(
        self, status: int, kind: ProviderFailure
    ) -> None:
        tts, _ = _tts(lambda _: httpx.Response(status, json={"status": status}))
        with pytest.raises(ProviderError) as info:
            await _speak(tts)
        assert info.value.kind is kind

    async def test_a_missing_key_is_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FISH_API_KEY", raising=False)
        with pytest.raises(ProviderError) as info:
            await _speak(ftts.FishAudioTTS(reference_id="v"))
        assert info.value.kind is ProviderFailure.AUTH

    async def test_no_voice_is_an_error_not_a_default(self) -> None:
        tts, seen = _tts(lambda _: httpx.Response(200), reference_id=None)
        with pytest.raises(ValueError, match="no voice selected"):
            await _speak(tts)
        assert seen == []
        assert tts.select_voice(VoiceSelection(voice_id="v2")) is True
        assert tts.reference_id == "v2"


# ------------------------------------------------------------------- STT


def _frames(seconds: float) -> list[AudioFrame]:
    fmt = AudioFormat(sample_rate_hz=16_000)
    return [
        AudioFrame(
            pcm=bytes(fmt.frame_bytes()),
            seq=i,
            party=CALLER,
            t_produced=i * 0.02,
            fmt=fmt,
        )
        for i in range(int(seconds * 50))
    ]


def _stt(handler) -> tuple[fstt.FishAudioSTT, list[httpx.Request]]:
    client, seen = _client(handler)
    return fstt.FishAudioSTT(client=client, api_key="test-key"), seen


def _form(request: httpx.Request) -> dict[str, bytes]:
    """The multipart parts by name; enough of a parser for this request."""
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    parts: dict[str, bytes] = {}
    for chunk in request.content.split(b"--" + boundary)[1:-1]:
        head, _, body = chunk.partition(b"\r\n\r\n")
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        parts[name] = body[:-2]
    return parts


class TestSTT:
    async def test_the_request_follows_the_api_reference(self) -> None:
        stt, seen = _stt(
            lambda _: httpx.Response(200, json={"text": " Hi there. ", "duration": 0.5})
        )
        out = [t async for t in stt.transcribe(_frames(0.5))]

        (request,) = seen
        assert request.url == "https://api.fish.audio/v1/asr"
        assert request.headers["model"] == "transcribe-1-pro"
        assert request.headers["authorization"] == "Bearer test-key"
        form = _form(request)
        assert form["ignore_timestamps"] == b"true"
        assert form["diarize"] == b"false"
        assert form["tag_audio_events"] == b"false"
        with wave.open(io.BytesIO(form["audio"])) as wav:
            assert (wav.getframerate(), wav.getnframes()) == (16_000, 8_000)

        assert [(t.text, t.is_final) for t in out] == [("Hi there.", True)]
        assert out[0].audio_duration_s == pytest.approx(0.5)
        assert stt.usage.audio_seconds == pytest.approx(0.5)

    async def test_no_audio_sends_nothing(self) -> None:
        stt, seen = _stt(lambda _: httpx.Response(500))
        out = [t async for t in stt.transcribe([])]
        assert [(t.text, t.is_final) for t in out] == [("", True)]
        assert seen == []

    async def test_a_response_without_text_is_malformed(self) -> None:
        stt, _ = _stt(lambda _: httpx.Response(200, json={"duration": 1.0}))
        with pytest.raises(ProviderError) as info:
            [t async for t in stt.transcribe(_frames(0.1))]
        assert info.value.kind is ProviderFailure.MALFORMED

    async def test_an_http_error_maps_to_a_failure(self) -> None:
        stt, _ = _stt(lambda _: httpx.Response(429, json={"status": 429}))
        with pytest.raises(ProviderError) as info:
            [t async for t in stt.transcribe(_frames(0.1))]
        assert info.value.kind is ProviderFailure.THROTTLED


def test_both_register_with_honest_capabilities() -> None:
    registry = default_registry()
    register_all(registry)
    tts = registry.lookup("fishaudio.tts").spec
    stt = registry.lookup("fishaudio.stt").spec
    assert (tts.default_model, tts.capabilities.streams) == ("s2.1-pro", True)
    assert (stt.default_model, stt.capabilities.streams) == ("transcribe-1-pro", False)
    assert {tts.credential_provider, stt.credential_provider} == {"fishaudio"}
