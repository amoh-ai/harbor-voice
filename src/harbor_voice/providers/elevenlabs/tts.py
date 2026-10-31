"""ElevenLabs text to speech.

The caller voice tau-voice uses, so the request is tau2's request, field
for field (``tau2/voice/utils/elevenlabs_utils.py::tts_elevenlabs``):

- ``POST /v1/text-to-speech/{voice_id}?output_format=pcm_16000``
- ``model_id=eleven_v3``, tau2's voice settings, tau2's seed (``None``)
- whole-utterance synthesis, not streaming — tau2 synthesises a line, then
  plays it, and a caller that started speaking at first byte would be faster
  than the one it is being compared with
- three attempts with exponential backoff on timeouts, 409, 429 and 5xx

That is the parity contract. Anything that changes the bytes — model, voice,
settings, seed — is part of the cache key, so a rerun spends no credits and a
changed setting can never be served stale audio.

Plain HTTP via ``httpx`` rather than the ``elevenlabs`` SDK: the SDK's
``text_to_speech.convert`` is this one request, and harbor already depends on
``httpx``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from typing import override

import httpx

from harbor_voice._deps import numpy as _numpy
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.ids import PartyId
from harbor_voice.perturbation import dsp
from harbor_voice.perturbation.persona import VoiceSelection
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.cache import BlobCache
from harbor_voice.providers.cache import SynthesisKey
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities

NAME = "elevenlabs.tts"
DEFAULT_MODEL = "eleven_v3"
API_BASE = "https://api.elevenlabs.io"

REQUEST_FORMAT = AudioFormat(sample_rate_hz=16_000)
"""``pcm_16000``: tau2's ``DEFAULT_ELEVENLABS_OUTPUT_FORMAT``."""

RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT_S = 1.0
RETRY_MAX_WAIT_S = 10.0
"""tau2's ``tts_retry``: ``DEFAULT_RETRY_*`` in ``tau2/config.py``."""


@dataclass(frozen=True, slots=True)
class VoiceSettings:
    """ElevenLabs voice settings. Defaults are tau2's."""

    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = False

    def as_json(self) -> dict[str, Any]:
        return {
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
        }

    @override
    def __str__(self) -> str:
        return (
            f"stab={self.stability}/sim={self.similarity_boost}/"
            f"style={self.style}/boost={int(self.use_speaker_boost)}"
        )


TAU2_VOICE_SETTINGS = VoiceSettings()


class ElevenLabsTTS:
    """Whole-utterance ElevenLabs synthesis, cached, at tau2's settings.

    Args:
        voice_id: The ElevenLabs voice. Usually pinned per task (tau-voice
            names one per persona); can also arrive via :meth:`select_voice`.
        model: ElevenLabs model id.
        voice_settings: Synthesis settings; tau2's by default.
        seed: Passed through; ``None`` matches tau2.
        audio_format: What :meth:`synthesize` yields. The API is asked for
            16 kHz (tau2's rate) and the result is converted with an
            anti-aliased resampler, the adapter-edge converter.
        cache: Synthesis cache. ``None`` disables caching.
        api_key: Defaults to ``ELEVENLABS_API_KEY``.
    """

    def __init__(
        self,
        *,
        voice_id: str | None = None,
        model: str = DEFAULT_MODEL,
        voice_settings: VoiceSettings = TAU2_VOICE_SETTINGS,
        seed: int | None = None,
        audio_format: AudioFormat = CANONICAL,
        cache: BlobCache | None = None,
        api_key: str | None = None,
        api_base: str = API_BASE,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = NAME
        self.voice_id = voice_id
        self.model = model
        self.voice_settings = voice_settings
        self.seed = seed
        self._format = audio_format
        self.cache = cache
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._timeout_s = timeout_s
        self._client = client
        self.last_cache_hit: bool | None = None
        """Whether the last utterance came from the cache; for the trajectory."""
        self.characters_billed = 0
        """Characters actually sent to the API — cache hits cost nothing."""

    @property
    def audio_format(self) -> AudioFormat:
        return self._format

    # ---------------------------------------------------------- L1 voice

    def select_voice(self, selection: VoiceSelection) -> bool:
        """Adopt an L1 voice selection, or report that it cannot be served.

        Implements :class:`~harbor_voice.perturbation.persona.VoiceSelectable`.
        Only an explicit ``voice_id`` is honoured. ElevenLabs voices are
        account-specific — tau-voice creates its personas per account with
        Voice Design — so there is no fixed catalogue to map an accent or a
        gender onto, and pretending otherwise would put a default voice under
        an accent's name.
        """
        if selection.voice_id is None:
            return False
        self.voice_id = selection.voice_id
        return True

    # -------------------------------------------------------------- speak

    def key(self, text: str) -> SynthesisKey:
        """The cache key: everything that changes the bytes."""
        return SynthesisKey(
            provider=self.name,
            model=f"{self.model}:{self.voice_id}:{self.voice_settings}:seed={self.seed}",
            text=text,
            audio_format=self._format,
        )

    async def synthesize_pcm(self, text: str) -> bytes:
        """The whole utterance, in :attr:`audio_format`."""
        if self.voice_id is None:
            # A setup mistake, not a provider failure: nothing was sent.
            raise ValueError(
                f"{self.name}: no voice selected; pass voice_id or call "
                f"select_voice() with one"
            )
        key = self.key(text)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                self.last_cache_hit = True
                return cached
        raw = await self._request(text)
        pcm = self._convert(raw)
        self.last_cache_hit = False
        if self.cache is not None:
            self.cache.put(key, pcm)
        return pcm

    async def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        """The :class:`~harbor_voice.providers.protocols.TTSProvider` shape.

        Yields frame-sized chunks of an utterance synthesised whole, so the
        first chunk does not mark a streaming first byte — the spec declares
        ``streams=False`` for exactly that reason.
        """
        del party
        if not text:
            return
        pcm = await self.synthesize_pcm(text)
        size = self._format.frame_bytes()
        for start in range(0, len(pcm), size):
            yield pcm[start : start + size]

    # ----------------------------------------------------------- internals

    def _convert(self, raw: bytes) -> bytes:
        if len(raw) % 2:
            raw = raw[:-1]
        if self._format.sample_rate_hz == REQUEST_FORMAT.sample_rate_hz:
            return raw
        converted = dsp.resample_clean(
            dsp.to_array(raw),
            REQUEST_FORMAT.sample_rate_hz,
            self._format.sample_rate_hz,
        )
        return dsp.to_pcm(_numpy().asarray(converted))

    def _key_or_raise(self) -> str:
        key = self._api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            raise ProviderError(
                ProviderFailure.AUTH,
                "ELEVENLABS_API_KEY is not set",
                provider=self.name,
                model=self.model,
            )
        return key

    async def _request(self, text: str) -> bytes:
        body: dict[str, Any] = {
            "text": text,
            "model_id": self.model,
            "voice_settings": self.voice_settings.as_json(),
        }
        if self.seed is not None:
            body["seed"] = self.seed
        url = f"{self._api_base}/v1/text-to-speech/{self.voice_id}"
        headers = {"xi-api-key": self._key_or_raise(), "accept": "audio/pcm"}
        params = {"output_format": "pcm_16000"}

        client = self._client or httpx.AsyncClient(timeout=self._timeout_s)
        try:
            for attempt in range(1, RETRY_ATTEMPTS + 1):
                try:
                    response = await client.post(
                        url, json=body, headers=headers, params=params
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == RETRY_ATTEMPTS:
                        raise ProviderError(
                            ProviderFailure.TIMEOUT,
                            f"ElevenLabs request failed after {attempt} attempts: {exc}",
                            provider=self.name,
                            model=self.model,
                        ) from exc
                    await asyncio.sleep(_backoff(attempt))
                    continue
                if response.status_code == 200 and response.content:
                    self.characters_billed += len(text)
                    return response.content
                if _retryable(response.status_code) and attempt < RETRY_ATTEMPTS:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise _error(response, self.name, self.model)
        finally:
            if self._client is None:
                await client.aclose()
        # Every iteration returns, raises or continues; reaching here means
        # RETRY_ATTEMPTS was set below one.
        raise ValueError(  # pragma: no cover
            f"RETRY_ATTEMPTS must be at least 1, got {RETRY_ATTEMPTS}"
        )


def _retryable(status: int) -> bool:
    return status in (409, 429) or status >= 500


def _backoff(attempt: int) -> float:
    """tenacity's ``wait_exponential(multiplier=1, min=1, max=10)``."""
    return min(RETRY_MAX_WAIT_S, max(RETRY_MIN_WAIT_S, 2.0 ** (attempt - 1)))


def _error(response: httpx.Response, provider: str, model: str) -> ProviderError:
    status = response.status_code
    detail = response.text[:300]
    if status == 200:
        kind = ProviderFailure.MALFORMED
        message = "ElevenLabs returned 200 with no audio"
    elif status in (401, 403):
        kind = ProviderFailure.AUTH
        message = f"ElevenLabs rejected the key ({status}): {detail}"
    elif status == 429:
        kind = ProviderFailure.THROTTLED
        message = f"ElevenLabs rate limit or quota ({status}): {detail}"
    elif status >= 500:
        kind = ProviderFailure.STREAM_BROKEN
        message = f"ElevenLabs server error ({status}): {detail}"
    else:
        kind = ProviderFailure.MALFORMED
        message = f"ElevenLabs rejected the request ({status}): {detail}"
    return ProviderError(kind, message, provider=provider, model=model)


ELEVENLABS_TTS_SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.TTS,
    capabilities=ProviderCapabilities(modality=Modality.TTS, streams=False),
    credential_provider="elevenlabs",
    requires_extra="voice",
    default_model=DEFAULT_MODEL,
)
"""``streams=False``: whole-utterance synthesis, as tau2 does it."""


def build_tts(*, model: str = DEFAULT_MODEL, **config: Any) -> ElevenLabsTTS:
    return ElevenLabsTTS(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = (
    (ELEVENLABS_TTS_SPEC, build_tts),
)
