"""selfhosted.stt / selfhosted.tts: the OpenAI audio routes vLLM and SGLang
serve, against ``httpx.MockTransport`` — offline."""

from __future__ import annotations

import io
import json
import wave

import httpx
import pytest

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.providers.selfhosted import speech as sh

pytestmark = pytest.mark.unit

FMT_16K = AudioFormat(sample_rate_hz=16_000)
CALLER = PartyId("caller")


def _client(handler) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(record)), seen


def _frames(seconds: float = 0.5) -> list[AudioFrame]:
    pcm = bytes(FMT_16K.frame_bytes() * int(seconds * 50))
    return [AudioFrame(pcm=pcm, seq=0, party=CALLER, t_produced=0.0, fmt=FMT_16K)]


def _wav(seconds: float, rate: int, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(2 * channels * int(seconds * rate)))
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr(sh.asyncio, "sleep", instant)
    monkeypatch.delenv(sh.API_KEY_ENV, raising=False)
    monkeypatch.delenv(sh.STT_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(sh.TTS_BASE_URL_ENV, raising=False)


class TestSTT:
    async def test_posts_a_wav_and_yields_one_final(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(sh.STT_BASE_URL_ENV, "http://asr:8000/")
        monkeypatch.setenv(sh.API_KEY_ENV, "token-abc123")
        client, seen = _client(
            lambda _: httpx.Response(
                200,
                json={"text": " Hello. ", "usage": {"type": "duration", "seconds": 1}},
            )
        )
        stt = sh.SelfHostedSTT(
            model="openai/whisper-large-v3-turbo", language="en", client=client
        )
        results = [t async for t in stt.transcribe(_frames(0.5))]

        assert [(t.text, t.is_final) for t in results] == [("Hello.", True)]
        assert results[0].audio_duration_s == pytest.approx(0.5)
        assert stt.usage.audio_seconds == pytest.approx(0.5)
        (request,) = seen
        assert str(request.url) == "http://asr:8000/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer token-abc123"
        body = request.content
        assert b'name="model"\r\n\r\nopenai/whisper-large-v3-turbo' in body
        assert b'name="language"\r\n\r\nen' in body
        assert b'name="response_format"\r\n\r\njson' in body
        assert b"RIFF" in body and b'filename="audio.wav"' in body

    async def test_no_key_means_no_auth_header(self) -> None:
        client, seen = _client(lambda _: httpx.Response(200, json={"text": "hi"}))
        stt = sh.SelfHostedSTT(model="m", base_url="http://asr:8000", client=client)
        _ = [t async for t in stt.transcribe(_frames())]
        assert "authorization" not in seen[0].headers

    async def test_silence_sends_nothing(self) -> None:
        client, seen = _client(lambda _: httpx.Response(500))
        stt = sh.SelfHostedSTT(model="m", base_url="http://asr:8000", client=client)
        results = [t async for t in stt.transcribe([])]
        assert [t.text for t in results] == [""] and not seen

    async def test_a_reply_without_text_is_malformed(self) -> None:
        client, _ = _client(lambda _: httpx.Response(200, json={"error": "?"}))
        stt = sh.SelfHostedSTT(model="m", base_url="http://asr:8000", client=client)
        with pytest.raises(ProviderError) as info:
            _ = [t async for t in stt.transcribe(_frames())]
        assert info.value.kind is ProviderFailure.MALFORMED

    async def test_no_base_url_is_a_configuration_failure(self) -> None:
        stt = sh.SelfHostedSTT(model="m")
        with pytest.raises(ProviderError) as info:
            _ = [t async for t in stt.transcribe(_frames())]
        assert info.value.kind is ProviderFailure.AUTH
        assert sh.STT_BASE_URL_ENV in str(info.value)


class TestTTS:
    async def test_requests_wav_and_resamples_to_the_slot_format(self) -> None:
        client, seen = _client(lambda _: httpx.Response(200, content=_wav(1.0, 24_000)))
        tts = sh.SelfHostedTTS(
            model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            voice="vivian",
            base_url="http://tts:8091",
            audio_format=FMT_16K,
            client=client,
        )
        chunks = [c async for c in tts.synthesize("Hello there.", party=CALLER)]

        (request,) = seen
        assert str(request.url) == "http://tts:8091/v1/audio/speech"
        assert json.loads(request.content) == {
            "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "input": "Hello there.",
            "response_format": "wav",
            "voice": "vivian",
        }
        assert len(b"".join(chunks)) == 32_000  # one second at 16 kHz
        assert all(len(c) <= FMT_16K.frame_bytes() for c in chunks)

    async def test_same_rate_passes_through_and_voice_is_optional(self) -> None:
        client, seen = _client(lambda _: httpx.Response(200, content=_wav(0.5, 16_000)))
        tts = sh.SelfHostedTTS(
            model="m", base_url="http://tts:8091", audio_format=FMT_16K, client=client
        )
        assert len(await tts.synthesize_pcm("hi")) == 16_000
        assert "voice" not in json.loads(seen[0].content)

    async def test_empty_text_sends_nothing(self) -> None:
        client, seen = _client(lambda _: httpx.Response(500))
        tts = sh.SelfHostedTTS(model="m", base_url="http://tts:8091", client=client)
        assert [c async for c in tts.synthesize("", party=CALLER)] == [] and not seen

    @pytest.mark.parametrize(
        "content", [b"not audio", _wav(0.1, 24_000, channels=2)], ids=["raw", "stereo"]
    )
    async def test_unusable_audio_is_malformed(self, content: bytes) -> None:
        client, _ = _client(lambda _: httpx.Response(200, content=content))
        tts = sh.SelfHostedTTS(model="m", base_url="http://tts:8091", client=client)
        with pytest.raises(ProviderError) as info:
            await tts.synthesize_pcm("hi")
        assert info.value.kind is ProviderFailure.MALFORMED


class TestFailures:
    @pytest.mark.parametrize(
        ("status", "kind", "calls"),
        [
            (401, ProviderFailure.AUTH, 1),
            (404, ProviderFailure.NOT_ENABLED, 1),
            (400, ProviderFailure.MALFORMED, 1),
            (429, ProviderFailure.THROTTLED, sh.RETRY_ATTEMPTS),
            (503, ProviderFailure.STREAM_BROKEN, sh.RETRY_ATTEMPTS),
        ],
    )
    async def test_status_codes_map_and_only_transient_ones_retry(
        self, status: int, kind: ProviderFailure, calls: int
    ) -> None:
        client, seen = _client(lambda _: httpx.Response(status, text="nope"))
        tts = sh.SelfHostedTTS(model="m", base_url="http://tts:8091", client=client)
        with pytest.raises(ProviderError) as info:
            await tts.synthesize_pcm("hi")
        assert info.value.kind is kind
        assert len(seen) == calls

    async def test_a_server_that_is_down_times_out_after_retries(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        client, seen = _client(refuse)
        stt = sh.SelfHostedSTT(model="m", base_url="http://asr:8000", client=client)
        with pytest.raises(ProviderError) as info:
            _ = [t async for t in stt.transcribe(_frames())]
        assert info.value.kind is ProviderFailure.TIMEOUT
        assert len(seen) == sh.RETRY_ATTEMPTS

    async def test_a_transient_failure_then_success(self) -> None:
        replies = iter([httpx.Response(503), httpx.Response(200, json={"text": "ok"})])
        client, seen = _client(lambda _: next(replies))
        stt = sh.SelfHostedSTT(model="m", base_url="http://asr:8000", client=client)
        assert [t.text async for t in stt.transcribe(_frames())] == ["ok"]
        assert len(seen) == 2


class TestRegistration:
    def test_both_slots_register_with_the_model_from_the_spec(self) -> None:
        from harbor_voice.providers import selfhosted

        registry = ProviderRegistry()
        selfhosted.register_all(registry)
        stt = registry.create(
            "selfhosted.stt/Qwen/Qwen3-ASR-1.7B", modality=Modality.STT
        )
        tts = registry.create(
            "selfhosted.tts/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", modality=Modality.TTS
        )
        assert isinstance(stt, sh.SelfHostedSTT) and stt.model == "Qwen/Qwen3-ASR-1.7B"
        assert isinstance(tts, sh.SelfHostedTTS)
        for name in ("selfhosted.stt/m", "selfhosted.tts/m"):
            caps = registry.capabilities(name)
            assert not caps.streams and not caps.emits_partials
