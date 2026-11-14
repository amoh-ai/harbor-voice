"""ElevenLabs TTS: tau2's request, the cache, the retry policy — offline."""

from __future__ import annotations

import json

import httpx
import pytest

from harbor_voice.audio.format import AudioFormat
from harbor_voice.ids import PartyId
from harbor_voice.perturbation.persona import Accent
from harbor_voice.perturbation.persona import VoiceSelection
from harbor_voice.providers import base as provider_base
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.cache import BlobCache
from harbor_voice.providers.credentials import access_for
from harbor_voice.providers.elevenlabs import tts as el
from harbor_voice.providers.registry import default_registry

pytestmark = pytest.mark.unit

ONE_SECOND_16K = bytes(32_000)


def _tts(handler, **kw) -> tuple[el.ElevenLabsTTS, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    kw.setdefault("voice_id", "voice-abc")
    return el.ElevenLabsTTS(client=client, api_key="test-key", **kw), seen


def _ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=ONE_SECOND_16K)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr(provider_base.asyncio, "sleep", instant)


class TestRequestMatchesTau2:
    async def test_the_request_is_tau2s_field_for_field(self) -> None:
        tts, seen = _tts(_ok)
        await tts.synthesize_pcm("Hi, I need to change a flight.")

        (request,) = seen
        assert request.url.path == "/v1/text-to-speech/voice-abc"
        assert request.url.params["output_format"] == "pcm_16000"
        assert request.headers["xi-api-key"] == "test-key"
        assert json.loads(request.content) == {
            "text": "Hi, I need to change a flight.",
            "model_id": "eleven_v3",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": False,
            },
        }

    async def test_a_seed_is_sent_only_when_set(self) -> None:
        tts, seen = _tts(_ok, seed=7)
        await tts.synthesize_pcm("x")
        assert json.loads(seen[0].content)["seed"] == 7

    async def test_16k_is_converted_to_the_requested_format(self) -> None:
        canonical, _ = _tts(_ok)
        assert len(await canonical.synthesize_pcm("x")) == 48_000  # 1 s at 24 kHz

        native, _ = _tts(_ok, audio_format=AudioFormat(sample_rate_hz=16_000))
        assert await native.synthesize_pcm("x") == ONE_SECOND_16K

    async def test_synthesize_yields_frames(self) -> None:
        tts, _ = _tts(_ok)
        chunks = [c async for c in tts.synthesize("x", party=PartyId("caller"))]
        assert all(len(c) == tts.audio_format.frame_bytes() for c in chunks)
        assert sum(map(len, chunks)) == 48_000


class TestCache:
    async def test_a_repeat_line_costs_nothing(self) -> None:
        tts, seen = _tts(_ok, cache=BlobCache())
        first = await tts.synthesize_pcm("Same line.")
        assert tts.last_cache_hit is False
        second = await tts.synthesize_pcm("Same line.")

        assert first == second
        assert len(seen) == 1
        assert tts.last_cache_hit is True
        assert tts.characters_billed == len("Same line.")

    async def test_anything_that_changes_the_bytes_changes_the_key(self) -> None:
        base, _ = _tts(_ok)
        keys = {
            base.key("x").digest,
            _tts(_ok, voice_id="other")[0].key("x").digest,
            _tts(_ok, seed=1)[0].key("x").digest,
            _tts(_ok, model="eleven_flash_v2_5")[0].key("x").digest,
            _tts(_ok, voice_settings=el.VoiceSettings(stability=0.9))[0]
            .key("x")
            .digest,
        }
        assert len(keys) == 5


class TestFailures:
    async def test_429_then_success_is_retried(self) -> None:
        replies = iter(
            [httpx.Response(429), httpx.Response(200, content=ONE_SECOND_16K)]
        )
        tts, seen = _tts(lambda _: next(replies))
        await tts.synthesize_pcm("x")
        assert len(seen) == 2

    async def test_retries_stop_after_three_attempts(self) -> None:
        tts, seen = _tts(lambda _: httpx.Response(503))
        with pytest.raises(ProviderError) as info:
            await tts.synthesize_pcm("x")
        assert info.value.kind is ProviderFailure.STREAM_BROKEN
        assert len(seen) == el.RETRY_ATTEMPTS

    async def test_a_rejected_key_is_not_retried(self) -> None:
        tts, seen = _tts(lambda _: httpx.Response(401, text="bad key"))
        with pytest.raises(ProviderError) as info:
            await tts.synthesize_pcm("x")
        assert info.value.kind is ProviderFailure.AUTH
        assert len(seen) == 1

    @pytest.mark.parametrize(
        ("status", "kind", "attempts"),
        [
            (402, ProviderFailure.AUTH, 1),
            (404, ProviderFailure.MALFORMED, 1),
            (504, ProviderFailure.TIMEOUT, el.RETRY_ATTEMPTS),
        ],
    )
    async def test_statuses_map_like_every_other_provider(
        self, status: int, kind: ProviderFailure, attempts: int
    ) -> None:
        tts, seen = _tts(lambda _: httpx.Response(status, text="no"))
        with pytest.raises(ProviderError) as info:
            await tts.synthesize_pcm("x")
        assert info.value.kind is kind
        assert len(seen) == attempts

    async def test_an_empty_200_is_malformed_and_not_retried(self) -> None:
        tts, seen = _tts(lambda _: httpx.Response(200, content=b""))
        with pytest.raises(ProviderError) as info:
            await tts.synthesize_pcm("x")
        assert info.value.kind is ProviderFailure.MALFORMED
        assert len(seen) == 1

    async def test_the_last_frame_is_short_but_whole_samples(self) -> None:
        tts, _ = _tts(
            lambda _: httpx.Response(200, content=bytes(1_001)),
            audio_format=AudioFormat(sample_rate_hz=16_000),
        )
        chunks = [c async for c in tts.synthesize("x", party=PartyId("caller"))]
        assert [len(c) for c in chunks] == [640, 360]

    async def test_no_voice_is_an_error_not_a_default(self) -> None:
        tts, seen = _tts(_ok, voice_id=None)
        with pytest.raises(ValueError, match="no voice selected"):
            await tts.synthesize_pcm("x")
        assert seen == []


class TestVoiceSelection:
    def test_only_an_explicit_voice_id_is_honoured(self) -> None:
        tts, _ = _tts(_ok, voice_id=None)
        assert tts.select_voice(VoiceSelection(accent=Accent.INDIAN)) is False
        assert tts.voice_id is None
        assert tts.select_voice(VoiceSelection(voice_id="persona-1")) is True
        assert tts.voice_id == "persona-1"


def test_registered_with_its_credential() -> None:
    registry = default_registry()
    from harbor_voice.providers.elevenlabs import register_all

    register_all(registry)
    registration = registry.lookup("elevenlabs.tts")
    assert registration.spec.capabilities.streams is False
    access = access_for("elevenlabs")
    assert access is not None
    assert access.api_key_envs == ("ELEVENLABS_API_KEY",)
