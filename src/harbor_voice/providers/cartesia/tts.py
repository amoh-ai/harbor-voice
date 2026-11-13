"""Cartesia text to speech: Sonic, streamed over ``POST /tts/bytes``.

Cartesia's endpoint guide picks bytes when the whole transcript is known
before synthesis starts, as it is here: the response body streams while the
audio is generated, with no JSON or base64 around it. The websocket only pays
off when text arrives in pieces (docs.cartesia.ai/use-the-api/compare-tts-endpoints).

- ``Authorization: Bearer <key>`` and ``Cartesia-Version: 2026-08-14``, the
  only version the current API reference accepts
- ``output_format``: ``raw`` at the requested rate and encoding, so the bytes
  are already in :attr:`audio_format` and nothing is resampled
- billed at about one credit per character, counted in :attr:`usage`

Plain HTTP via ``httpx``, as ``elevenlabs.tts`` does; the SDK adds nothing to
one streamed request.
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
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import http_failure
from harbor_voice.providers.base import require_key
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities

NAME = "cartesia.tts"
DEFAULT_MODEL = "sonic-3.6"
API_BASE = "https://api.cartesia.ai"
API_VERSION = "2026-08-14"
API_KEY_ENV = "CARTESIA_API_KEY"

SAMPLE_RATES = frozenset({8_000, 16_000, 22_050, 24_000, 44_100, 48_000})
"""``RawOutputFormat.sample_rate`` in the API reference."""

ENCODINGS = {"pcm16": "pcm_s16le", "mulaw": "pcm_mulaw", "alaw": "pcm_alaw"}
"""Our encodings, by Cartesia's name. Both TTS and STT accept all three."""


class CartesiaTTS:
    """Streaming Sonic synthesis.

    Args:
        voice_id: A Cartesia voice id. Required before :meth:`synthesize`;
            can also arrive via :meth:`select_voice`.
        model: Cartesia model id.
        audio_format: What :meth:`synthesize` yields, and what is requested:
            mono, at one of :data:`SAMPLE_RATES`.
        api_key: Defaults to ``CARTESIA_API_KEY``.
    """

    def __init__(
        self,
        *,
        voice_id: str | None = None,
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
        ):
            raise ValueError(
                f"{NAME}: Cartesia cannot produce {audio_format}; "
                f"mono at one of {sorted(SAMPLE_RATES)} Hz"
            )
        self.name = NAME
        self.voice_id = voice_id
        self.model = model
        self._format = audio_format
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._timeout_s = timeout_s
        self._client = client
        self.usage = Usage(characters=0)
        """The last utterance's characters, the unit Cartesia bills."""

    @property
    def audio_format(self) -> AudioFormat:
        return self._format

    def select_voice(self, selection: VoiceSelection) -> bool:
        """Adopt an explicit ``voice_id``; nothing else maps onto a voice."""
        if selection.voice_id is None:
            return False
        self.voice_id = selection.voice_id
        return True

    async def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        """Yield frame-sized PCM chunks as the response body arrives."""
        del party
        if not text:
            return
        if self.voice_id is None:
            raise ValueError(
                f"{self.name}: no voice selected; pass voice_id or call "
                f"select_voice() with one"
            )
        body: dict[str, Any] = {
            "model_id": self.model,
            "transcript": text,
            "voice": self.voice_id,
            "output_format": {
                "container": "raw",
                "encoding": ENCODINGS[self._format.encoding],
                "sample_rate": self._format.sample_rate_hz,
            },
        }
        headers = {
            "Authorization": f"Bearer {api_key(self._api_key, NAME, self.model)}",
            "Cartesia-Version": API_VERSION,
        }
        cutter = FrameCutter(self._format)
        received = 0
        client = self._client or httpx.AsyncClient(timeout=self._timeout_s)
        try:
            async with client.stream(
                "POST", f"{self._api_base}/tts/bytes", json=body, headers=headers
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
        finally:
            if self._client is None:
                await client.aclose()
        if not received:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                "Cartesia returned 200 with no audio",
                provider=NAME,
                model=self.model,
            )
        self.usage = Usage(characters=len(text))
        if tail := cutter.tail():
            yield tail


def api_key(explicit: str | None, provider: str, model: str) -> str:
    return require_key(explicit, API_KEY_ENV, provider=provider, model=model)


def error(status: int, detail: str, provider: str, model: str) -> ProviderError:
    """An HTTP status (or a websocket error event's) as a ProviderError."""
    return ProviderError(
        http_failure(status),
        f"Cartesia returned {status}: {detail[:300]}",
        provider=provider,
        model=model,
    )


CARTESIA_TTS_SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.TTS,
    capabilities=ProviderCapabilities(modality=Modality.TTS, streams=True),
    credential_provider="cartesia",
    requires_extra="voice",
    default_model=DEFAULT_MODEL,
)
"""``streams=True``: chunks are yielded as the response body arrives."""


def build_tts(*, model: str = DEFAULT_MODEL, **config: Any) -> CartesiaTTS:
    return CartesiaTTS(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((CARTESIA_TTS_SPEC, build_tts),)
