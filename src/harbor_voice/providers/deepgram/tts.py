"""Deepgram Aura text to speech, streamed.

``POST https://api.deepgram.com/v1/speak?model=…&encoding=…&sample_rate=…&container=none``
with ``{"text": …}`` (https://developers.deepgram.com/reference/text-to-speech/speak-request).
The response body is the audio, sent with chunked transfer encoding, so
playback can start at the first byte
(https://developers.deepgram.com/docs/text-to-speech). ``container=none``
makes it raw samples with no WAV header
(https://developers.deepgram.com/docs/tts-media-output-settings).

Deepgram is asked for exactly :attr:`DeepgramTTS.audio_format`, so nothing
is resampled here. Linear16 comes at 8, 16, 24, 32 or 48 kHz; μ-law and
A-law at 8 or 16 kHz.

Aura is billed per character; the response's ``dg-char-count`` header is
the count, kept in :attr:`DeepgramTTS.characters_billed`.

The voice is part of the model ID (``aura-2-thalia-en``), so choosing a
voice means choosing a model.
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
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import http_failure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.deepgram._common import auth_header
from harbor_voice.providers.deepgram._common import encoding

NAME = "deepgram.tts"
DEFAULT_MODEL = "aura-2-thalia-en"
URL = "https://api.deepgram.com/v1/speak"

SAMPLE_RATES = {
    "linear16": frozenset({8_000, 16_000, 24_000, 32_000, 48_000}),
    "mulaw": frozenset({8_000, 16_000}),
    "alaw": frozenset({8_000, 16_000}),
}
"""What ``/v1/speak`` produces per encoding."""


class DeepgramTTS:
    """Aura synthesis, yielded in 20 ms chunks as the response streams in.

    Args:
        model: An Aura model, which names the voice too.
        audio_format: What :meth:`synthesize` yields, requested as is.
        api_key: Defaults to ``DEEPGRAM_API_KEY``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        audio_format: AudioFormat = CANONICAL,
        api_key: str | None = None,
        url: str = URL,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        wire = encoding(audio_format)
        if audio_format.sample_rate_hz not in SAMPLE_RATES[wire]:
            raise ValueError(
                f"{NAME}: {wire} comes at {sorted(SAMPLE_RATES[wire])} Hz, "
                f"not {audio_format.sample_rate_hz}"
            )
        self.name = NAME
        self.model = model
        self._format = audio_format
        self._params = {
            "model": model,
            "encoding": wire,
            "sample_rate": audio_format.sample_rate_hz,
            "container": "none",
        }
        self._api_key = api_key
        self._url = url
        self._timeout_s = timeout_s
        self._client = client
        self.characters_billed = 0
        """Characters Deepgram counted (``dg-char-count``), summed."""

    @property
    def audio_format(self) -> AudioFormat:
        return self._format

    async def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        del party
        if not text:
            return
        headers = auth_header(self._api_key, provider=NAME, model=self.model)
        cutter = FrameCutter(self._format)
        client = self._client or httpx.AsyncClient(timeout=self._timeout_s)
        try:
            async with client.stream(
                "POST",
                self._url,
                params=self._params,
                json={"text": text},
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise ProviderError(
                        http_failure(response.status_code),
                        f"HTTP {response.status_code}: {response.text[:300]}",
                        provider=NAME,
                        model=self.model,
                    )
                self.characters_billed += int(
                    response.headers.get("dg-char-count") or len(text)
                )
                received = 0
                async for chunk in response.aiter_bytes():
                    received += len(chunk)
                    for frame in cutter.feed(chunk):
                        yield frame
                if tail := cutter.tail():
                    yield tail
                if not received:
                    raise ProviderError(
                        ProviderFailure.MALFORMED,
                        "Deepgram returned 200 with no audio",
                        provider=NAME,
                        model=self.model,
                    )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                ProviderFailure.TIMEOUT, str(exc), provider=NAME, model=self.model
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                ProviderFailure.STREAM_BROKEN,
                f"{type(exc).__name__}: {exc}",
                provider=NAME,
                model=self.model,
            ) from exc
        finally:
            if self._client is None:
                await client.aclose()


SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.TTS,
    capabilities=ProviderCapabilities(modality=Modality.TTS, streams=True),
    credential_provider="deepgram",
    requires_extra="voice",
    default_model=DEFAULT_MODEL,
)
"""``streams=True``: the first chunk is yielded as soon as 20 ms of audio has
arrived, not after the whole utterance."""


def build(*, model: str = DEFAULT_MODEL, **config: Any) -> DeepgramTTS:
    return DeepgramTTS(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
