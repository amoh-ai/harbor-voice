"""Sarvam text to speech: Bulbul, streamed over ``POST /text-to-speech/stream``.

Sarvam's HTTP streaming guide picks this endpoint when the whole text is known
before synthesis starts, as it is here; the websocket only pays off when text
arrives in pieces (docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/
streaming-api/http-stream). The response is the binary audio itself, not
JSON or base64.

- ``api-subscription-key`` header, from ``SARVAM_API_KEY``
- ``output_audio_codec=linear16`` at :attr:`audio_format`'s rate, so nothing
  is resampled. Streaming accepts 8, 16, 22.05 and 24 kHz only
- ``bulbul:v4-flash``, the model Sarvam recommends for new integrations;
  ``speaker`` left unset uses that model's default voice
- billed per character, counted in :attr:`usage`

Plain HTTP via ``httpx``; the ``sarvamai`` SDK adds nothing to one request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.format import FrameCutter
from harbor_voice.ids import PartyId
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import HTTPSession
from harbor_voice.providers.base import PooledHTTP
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import http_failure
from harbor_voice.providers.base import require_key
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities

NAME = "sarvam.tts"
DEFAULT_MODEL = "bulbul:v4-flash"
API_BASE = "https://api.sarvam.ai"
API_KEY_ENV = "SARVAM_API_KEY"
DEFAULT_LANGUAGE = "en-IN"

SAMPLE_RATES = frozenset({8_000, 16_000, 22_050, 24_000})
"""``speech_sample_rate`` values the streaming endpoints accept."""


class SarvamTTS(PooledHTTP):
    """Streaming Bulbul synthesis.

    Args:
        model: Sarvam TTS model id.
        speaker: A voice valid for ``model`` (lowercase). ``None`` sends no
            ``speaker``, so Sarvam uses the model's default.
        language_code: BCP-47 code of the text, e.g. ``hi-IN``.
        audio_format: What :meth:`synthesize` yields: 16-bit mono at one of
            :data:`SAMPLE_RATES`.
        api_key: Defaults to ``SARVAM_API_KEY``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        speaker: str | None = None,
        language_code: str = DEFAULT_LANGUAGE,
        audio_format: AudioFormat = CANONICAL,
        api_key: str | None = None,
        api_base: str = API_BASE,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if (
            audio_format.sample_rate_hz not in SAMPLE_RATES
            or audio_format.channels != 1
            or audio_format.encoding != "pcm16"
        ):
            raise ValueError(
                f"{NAME}: Sarvam cannot stream {audio_format}; pcm16 mono at "
                f"one of {sorted(SAMPLE_RATES)} Hz"
            )
        self.name = NAME
        self.model = model
        self.speaker = speaker
        self.language_code = language_code
        self._format = audio_format
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._timeout_s = timeout_s
        self._http = HTTPSession(client)
        self.usage = Usage(characters=0)
        """The last utterance's characters, the unit Sarvam bills."""

    @property
    def audio_format(self) -> AudioFormat:
        return self._format

    async def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        """Yield frame-sized PCM chunks as the response body arrives."""
        del party
        if not text:
            return
        body: dict[str, Any] = {
            "text": text,
            "model": self.model,
            "language_code": self.language_code,
            "speech_sample_rate": self._format.sample_rate_hz,
            "output_audio_codec": "linear16",
        }
        if self.speaker is not None:
            body["speaker"] = self.speaker
        key = require_key(self._api_key, API_KEY_ENV, provider=NAME, model=self.model)
        headers = {"api-subscription-key": key}
        cutter = FrameCutter(self._format)
        received = 0
        client = self._http.client(self._timeout_s)
        try:
            async with client.stream(
                "POST",
                f"{self._api_base}/text-to-speech/stream",
                json=body,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise error(response.status_code, response.text, NAME, self.model)
                async for chunk in response.aiter_bytes():
                    received += len(chunk)
                    for frame in cutter.feed(chunk):
                        yield frame
        except httpx.TimeoutException as exc:
            raise ProviderError(
                ProviderFailure.TIMEOUT, str(exc), provider=NAME, model=self.model
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                ProviderFailure.STREAM_BROKEN, str(exc), provider=NAME, model=self.model
            ) from exc
        if not received:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                "Sarvam returned 200 with no audio",
                provider=NAME,
                model=self.model,
            )
        self.usage = Usage(characters=len(text))
        if tail := cutter.tail():
            yield tail


def error(status: int, detail: str, provider: str, model: str) -> ProviderError:
    """An HTTP status (or a websocket error event's) as a ProviderError."""
    return ProviderError(
        http_failure(status),
        f"Sarvam returned {status}: {detail[:300]}",
        provider=provider,
        model=model,
    )


SARVAM_TTS_SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.TTS,
    capabilities=ProviderCapabilities(modality=Modality.TTS, streams=True),
    credential_provider="sarvam",
    default_model=DEFAULT_MODEL,
)
"""``streams=True``: chunks are yielded as the response body arrives."""


def build_tts(*, model: str = DEFAULT_MODEL, **config: Any) -> SarvamTTS:
    return SarvamTTS(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SARVAM_TTS_SPEC, build_tts),)
