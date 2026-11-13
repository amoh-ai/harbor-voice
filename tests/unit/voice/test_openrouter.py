"""openrouter.stt / openrouter.tts: the self-hosted speech classes on
OpenRouter's audio routes, against ``httpx.MockTransport`` — offline."""

from __future__ import annotations

import httpx
import pytest

from harbor_voice.audio.format import CANONICAL
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.openrouter import speech as orsp
from harbor_voice.providers.selfhosted import speech as sh
from tests.unit.voice.test_selfhosted import _client
from tests.unit.voice.test_selfhosted import _frames

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr(sh.asyncio, "sleep", instant)
    monkeypatch.setenv(orsp.KEY_ENV, "or-key")
    monkeypatch.delenv(orsp.BASE_URL_ENV, raising=False)
    monkeypatch.delenv(sh.API_KEY_ENV, raising=False)


async def test_stt_posts_to_openrouter_with_its_key() -> None:
    client, seen = _client(lambda _: httpx.Response(200, json={"text": "Hello."}))
    stt = orsp.build_stt(model="openai/whisper-large-v3-turbo", client=client)
    results = [t async for t in stt.transcribe(_frames(0.5))]
    assert [t.text for t in results] == ["Hello."]
    (request,) = seen
    assert str(request.url) == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert request.headers["Authorization"] == "Bearer or-key"


async def test_tts_asks_for_pcm_and_resamples_from_the_content_type() -> None:
    one_second_44k = bytes(2 * 44_100)
    client, seen = _client(
        lambda _: httpx.Response(
            200,
            content=one_second_44k,
            headers={"content-type": "audio/pcm;rate=44100;channels=1"},
        )
    )
    tts = orsp.build_tts(model="fish-audio/s2-pro", voice="sarah", client=client)
    pcm = await tts.synthesize_pcm("Hello.")
    assert abs(len(pcm) - 2 * CANONICAL.sample_rate_hz) <= 4
    (request,) = seen
    assert str(request.url) == "https://openrouter.ai/api/v1/audio/speech"
    body = request.read().decode()
    assert '"response_format":"pcm"' in body.replace(" ", "")
    assert '"voice":"sarah"' in body.replace(" ", "")


@pytest.mark.parametrize(
    "content_type", ["audio/pcm", "audio/pcm;rate=24000;channels=2"]
)
async def test_tts_refuses_pcm_it_cannot_place(content_type: str) -> None:
    client, _ = _client(
        lambda _: httpx.Response(
            200, content=b"\0\0", headers={"content-type": content_type}
        )
    )
    tts = orsp.build_tts(
        model="deepgram/aura-2", voice="aura-2-thalia-en", client=client
    )
    with pytest.raises(ProviderError) as info:
        await tts.synthesize_pcm("Hello.")
    assert info.value.kind is ProviderFailure.MALFORMED


async def test_the_base_url_can_be_moved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(orsp.BASE_URL_ENV, "https://proxy.example/api/")
    client, seen = _client(lambda _: httpx.Response(200, json={"text": "hi"}))
    stt = orsp.build_stt(model="deepgram/nova-3", client=client)
    [_ async for _ in stt.transcribe(_frames(0.5))]
    assert str(seen[0].url) == "https://proxy.example/api/v1/audio/transcriptions"
