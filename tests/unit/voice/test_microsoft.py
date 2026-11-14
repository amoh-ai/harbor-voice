"""MAI-Transcribe through Azure Speech, against ``httpx.MockTransport``. No
test calls the real API."""

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
from harbor_voice.providers.credentials import resolve
from harbor_voice.providers.microsoft import register_all
from harbor_voice.providers.microsoft import stt as mstt
from harbor_voice.providers.registry import default_registry

pytestmark = pytest.mark.unit

ENDPOINT = "https://res.cognitiveservices.azure.com/"


def _stt(handler, **config) -> tuple[mstt.MicrosoftSTT, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    stt = mstt.MicrosoftSTT(
        client=client, api_key="test-key", endpoint=ENDPOINT, **config
    )
    return stt, seen


def _frames(seconds: float) -> list[AudioFrame]:
    fmt = AudioFormat(sample_rate_hz=16_000)
    return [
        AudioFrame(
            pcm=bytes(fmt.frame_bytes()),
            seq=i,
            party=PartyId("caller"),
            t_produced=i * 0.02,
            fmt=fmt,
        )
        for i in range(int(seconds * 50))
    ]


def _form(request: httpx.Request) -> dict[str, bytes]:
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    parts: dict[str, bytes] = {}
    for chunk in request.content.split(b"--" + boundary)[1:-1]:
        head, _, body = chunk.partition(b"\r\n\r\n")
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        parts[name] = body[:-2]
    return parts


REPLY = {
    "durationMilliseconds": 480,
    "combinedPhrases": [{"text": " Hi there. "}],
    "phrases": [],
}


class TestSTT:
    async def test_the_request_follows_the_api_reference(self) -> None:
        stt, seen = _stt(lambda _: httpx.Response(200, json=REPLY))
        out = [t async for t in stt.transcribe(_frames(0.5))]

        (request,) = seen
        assert str(request.url) == (
            "https://res.cognitiveservices.azure.com/speechtotext/"
            "transcriptions:transcribe?api-version=2025-10-15"
        )
        assert request.headers["ocp-apim-subscription-key"] == "test-key"
        form = _form(request)
        assert json.loads(form["definition"]) == {
            "enhancedMode": {"enabled": True, "model": "MAI-Transcribe-2"}
        }
        with wave.open(io.BytesIO(form["audio"])) as wav:
            assert (wav.getframerate(), wav.getnframes()) == (16_000, 8_000)

        assert [(t.text, t.is_final) for t in out] == [("Hi there.", True)]
        assert out[0].audio_duration_s == pytest.approx(0.5)
        assert stt.usage.audio_seconds == pytest.approx(0.48)

    async def test_a_locale_is_sent_when_given(self) -> None:
        stt, seen = _stt(lambda _: httpx.Response(200, json=REPLY), locale="en")
        [t async for t in stt.transcribe(_frames(0.1))]
        assert json.loads(_form(seen[0])["definition"])["locales"] == ["en"]

    async def test_no_audio_sends_nothing(self) -> None:
        stt, seen = _stt(lambda _: httpx.Response(500))
        out = [t async for t in stt.transcribe([])]
        assert [(t.text, t.is_final) for t in out] == [("", True)]
        assert seen == []

    async def test_a_missing_endpoint_is_an_auth_failure(self, monkeypatch) -> None:
        monkeypatch.delenv("AZURE_SPEECH_ENDPOINT", raising=False)
        stt = mstt.MicrosoftSTT(api_key="test-key")
        with pytest.raises(ProviderError) as info:
            [t async for t in stt.transcribe(_frames(0.1))]
        assert info.value.kind is ProviderFailure.AUTH

    async def test_a_response_without_text_is_malformed(self) -> None:
        stt, _ = _stt(lambda _: httpx.Response(200, json={"durationMilliseconds": 1}))
        with pytest.raises(ProviderError) as info:
            [t async for t in stt.transcribe(_frames(0.1))]
        assert info.value.kind is ProviderFailure.MALFORMED

    @pytest.mark.parametrize(
        ("status", "kind"),
        [
            (401, ProviderFailure.AUTH),
            (404, ProviderFailure.REGION),
            (429, ProviderFailure.THROTTLED),
            (503, ProviderFailure.STREAM_BROKEN),
        ],
    )
    async def test_an_http_error_maps_to_a_failure(self, status, kind) -> None:
        stt, _ = _stt(lambda _: httpx.Response(status, json={}))
        with pytest.raises(ProviderError) as info:
            [t async for t in stt.transcribe(_frames(0.1))]
        assert info.value.kind is kind


def test_registered_with_its_credentials() -> None:
    registry = default_registry()
    register_all(registry)
    assert registry.resolve("microsoft.stt")[0].spec is mstt.MICROSOFT_STT_SPEC
    status = resolve("azure-speech", getenv=lambda _: None)
    assert status.missing == ("AZURE_SPEECH_KEY", "AZURE_SPEECH_ENDPOINT")
