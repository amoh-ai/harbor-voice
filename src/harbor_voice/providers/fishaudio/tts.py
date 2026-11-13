"""Fish Audio text to speech, streamed over ``POST /v1/tts``.

The request, per the API reference (docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech):

- ``Authorization: Bearer <key>``; the model goes in a ``model`` header,
  ``s2.1-pro`` by default, the model Fish recommends for production
- a JSON body with ``text``, ``reference_id`` (the voice), ``format: pcm``
  (raw 16-bit mono, no container) at the requested ``sample_rate``, so
  nothing is resampled
- ``latency: balanced``, sent explicitly: the guide calls it the default
  (about 300 ms to first audio) and the reference says ``normal``
- the response is chunked and read as it arrives

Fish bills TTS per UTF-8 byte of input, so :attr:`usage` counts bytes in
``characters``; for ASCII text the two are the same number.

Plain HTTP via ``httpx``, as ``elevenlabs.tts`` does.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.format import FrameCutter
from harbor_voice.ids import PartyId
from harbor_voice.perturbation.persona import VoiceSelection
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

NAME = "fishaudio.tts"
DEFAULT_MODEL = "s2.1-pro"
API_BASE = "https://api.fish.audio"
API_KEY_ENV = "FISH_API_KEY"
LATENCY = "balanced"

SAMPLE_RATES = frozenset({8_000, 16_000, 24_000, 32_000, 44_100})
"""The WAV / PCM rates the API reference lists."""


class FishAudioTTS(PooledHTTP):
    """Streaming Fish Audio synthesis.

    Args:
        reference_id: A Fish Audio voice model id. Required before
            :meth:`synthesize`; can also arrive via :meth:`select_voice`.
        model: Fish Audio TTS model, sent as the ``model`` header.
        audio_format: What :meth:`synthesize` yields, and what is requested:
            16-bit mono PCM at one of :data:`SAMPLE_RATES`.
        api_key: Defaults to ``FISH_API_KEY``.
    """

    def __init__(
        self,
        *,
        reference_id: str | None = None,
        model: str = DEFAULT_MODEL,
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
                f"{NAME}: Fish Audio cannot produce {audio_format}; "
                f"pcm16 mono at one of {sorted(SAMPLE_RATES)} Hz"
            )
        self.name = NAME
        self.reference_id = reference_id
        self.model = model
        self._format = audio_format
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._timeout_s = timeout_s
        self._http = HTTPSession(client)
        self.usage = Usage(characters=0)
        """The last utterance's UTF-8 bytes, the unit Fish Audio bills."""

    @property
    def audio_format(self) -> AudioFormat:
        return self._format

    def select_voice(self, selection: VoiceSelection) -> bool:
        """Adopt an explicit voice id as ``reference_id``; nothing else maps."""
        if selection.voice_id is None:
            return False
        self.reference_id = selection.voice_id
        return True

    async def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        """Yield frame-sized PCM chunks as the response body arrives."""
        del party
        if not text:
            return
        if self.reference_id is None:
            raise ValueError(
                f"{self.name}: no voice selected; pass reference_id or call "
                f"select_voice() with one"
            )
        body = {
            "text": text,
            "reference_id": self.reference_id,
            "format": "pcm",
            "sample_rate": self._format.sample_rate_hz,
            "latency": LATENCY,
        }
        headers = {
            "Authorization": f"Bearer {api_key(self._api_key, NAME, self.model)}",
            "model": self.model,
        }
        cutter = FrameCutter(self._format)
        received = 0
        client = self._http.client(self._timeout_s)
        try:
            async with client.stream(
                "POST", f"{self._api_base}/v1/tts", json=body, headers=headers
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
                "Fish Audio returned 200 with no audio",
                provider=NAME,
                model=self.model,
            )
        self.usage = Usage(characters=len(text.encode()))
        if tail := cutter.tail():
            yield tail


def api_key(explicit: str | None, provider: str, model: str) -> str:
    return require_key(explicit, API_KEY_ENV, provider=provider, model=model)


def error(status: int, detail: str, provider: str, model: str) -> ProviderError:
    """An HTTP status (or a websocket error event's) as a ProviderError."""
    return ProviderError(
        http_failure(status),
        f"Fish Audio returned {status}: {detail[:300]}",
        provider=provider,
        model=model,
    )


FISHAUDIO_TTS_SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.TTS,
    capabilities=ProviderCapabilities(modality=Modality.TTS, streams=True),
    credential_provider="fishaudio",
    requires_extra="voice",
    default_model=DEFAULT_MODEL,
)
"""``streams=True``: chunks are yielded as the chunked response arrives."""


def build_tts(*, model: str = DEFAULT_MODEL, **config: Any) -> FishAudioTTS:
    return FishAudioTTS(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((FISHAUDIO_TTS_SPEC, build_tts),)
