"""xAI text to speech and speech to text, over REST.

Both are one request each
(https://docs.x.ai/developers/rest-api-reference/inference/voice):

- ``POST /v1/tts``: JSON ``text``, ``voice_id``, ``language`` and
  ``output_format`` in, raw audio bytes out. Raw ``pcm`` is 16-bit
  little-endian, asked for at the rate the caller wants, so nothing is
  resampled here. Billed per input character.
- ``POST /v1/stt``: multipart ``model``, optional ``language``, the raw
  audio's ``audio_format`` and ``sample_rate``, then ``file`` (last, as the
  API requires) in; JSON ``text`` and ``duration`` out. Billed per hour of
  audio.

Whole-utterance on both sides, so neither declares ``streams``. xAI also has
websocket versions of both (``wss://api.x.ai/v1/tts``, ``wss://api.x.ai/v1/stt``)
that this module does not use. There is no TTS model to choose, so
``xai.tts/<voice>`` names the voice, as ``elevenlabs.agents/<id>`` names an
agent. Plain ``httpx``, which harbor already depends on.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import Any

import httpx

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import HTTPSession
from harbor_voice.providers.base import PooledHTTP
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import http_failure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.protocols import Transcript

API_BASE = "https://api.x.ai"
KEY_ENV = "XAI_API_KEY"
TTS_NAME = "xai.tts"
STT_NAME = "xai.stt"
DEFAULT_VOICE = "eve"
DEFAULT_STT_MODEL = "grok-voice-transcribe-2.0"
TTS_RATES = frozenset({8_000, 16_000, 22_050, 24_000, 44_100, 48_000})
STT_ENCODINGS = {"pcm16": "pcm", "mulaw": "mulaw", "alaw": "alaw"}
"""Our encodings as ``/v1/stt``'s raw ``audio_format`` names them."""


class _XAIRest(PooledHTTP):
    """What both providers share: the key, the client, one POST."""

    name: str

    def __init__(
        self,
        *,
        api_key: str | None,
        api_base: str,
        timeout_s: float,
        client: httpx.AsyncClient | None,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._timeout_s = timeout_s
        self._http = HTTPSession(client)

    async def _post(self, path: str, model: str, **request: Any) -> httpx.Response:
        """One authenticated POST; any failure as a :class:`ProviderError`."""
        key = self._api_key or os.environ.get(KEY_ENV)
        if not key:
            raise ProviderError(
                ProviderFailure.AUTH,
                f"{KEY_ENV} is not set",
                provider=self.name,
                model=model,
            )
        client = self._http.client(self._timeout_s)
        try:
            response = await client.post(
                f"{self._api_base}{path}",
                headers={"Authorization": f"Bearer {key}"},
                **request,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                ProviderFailure.TIMEOUT, str(exc), provider=self.name, model=model
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                ProviderFailure.STREAM_BROKEN,
                f"{type(exc).__name__}: {exc}",
                provider=self.name,
                model=model,
            ) from exc
        if response.status_code != 200:
            raise _error(response, self.name, model)
        return response


class XAITTS(_XAIRest):
    """Whole-utterance xAI synthesis.

    Args:
        voice: A built-in voice (``GET /v1/tts/voices``) or a custom voice ID.
        language: BCP-47 code or ``auto``. Required by the API.
        audio_format: What :meth:`synthesize` yields: mono PCM16 at one of the
            API's rates.
        api_key: Defaults to ``XAI_API_KEY``.
    """

    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        language: str = "en",
        audio_format: AudioFormat = CANONICAL,
        api_key: str | None = None,
        api_base: str = API_BASE,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if (
            audio_format.encoding != "pcm16"
            or audio_format.channels != 1
            or audio_format.sample_rate_hz not in TTS_RATES
        ):
            raise ValueError(
                f"{TTS_NAME} yields mono PCM16 at {sorted(TTS_RATES)} Hz, "
                f"not {audio_format}"
            )
        super().__init__(
            api_key=api_key, api_base=api_base, timeout_s=timeout_s, client=client
        )
        self.name = TTS_NAME
        self.voice = voice
        self.language = language
        self._format = audio_format
        self.characters_billed = 0

    @property
    def audio_format(self) -> AudioFormat:
        return self._format

    async def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        """Yields frame-sized chunks of one utterance synthesised whole."""
        del party
        if not text:
            return
        body = {
            "text": text,
            "voice_id": self.voice,
            "language": self.language,
            "output_format": {
                "codec": "pcm",
                "sample_rate": self._format.sample_rate_hz,
            },
        }
        response = await self._post("/v1/tts", self.voice, json=body)
        if not response.content:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                "returned 200 with no audio",
                provider=self.name,
                model=self.voice,
            )
        self.characters_billed += len(text)
        pcm = response.content[: len(response.content) // 2 * 2]
        size = self._format.frame_bytes()
        for start in range(0, len(pcm), size):
            yield pcm[start : start + size]


class XAISTT(_XAIRest):
    """One-shot xAI transcription: one final transcript, never a partial.

    Args:
        model: The transcription model.
        language: Biases recognition toward one language; ``None`` lets the
            model detect it.
        api_key: Defaults to ``XAI_API_KEY``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_STT_MODEL,
        language: str | None = None,
        api_key: str | None = None,
        api_base: str = API_BASE,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key, api_base=api_base, timeout_s=timeout_s, client=client
        )
        self.name = STT_NAME
        self.model = model
        self.language = language
        self.usage = Usage(audio_seconds=0.0)
        """Audio seconds billed so far, as the API reports them."""

    async def transcribe(
        self, frames: Sequence[AudioFrame]
    ) -> AsyncIterator[Transcript]:
        """Yield exactly one final transcript."""
        if not frames or not any(f.pcm for f in frames):
            yield Transcript(text="", is_final=True)
            return
        fmt = frames[0].fmt
        if fmt.encoding not in STT_ENCODINGS or fmt.channels != 1:
            raise ValueError(
                f"{self.name} takes mono {'/'.join(STT_ENCODINGS)}, got {fmt}"
            )
        data = b"".join(f.pcm for f in frames)
        form: dict[str, str] = {
            "model": self.model,
            "audio_format": STT_ENCODINGS[fmt.encoding],
            "sample_rate": str(fmt.sample_rate_hz),
        }
        if self.language is not None:
            form["language"] = self.language
        response = await self._post(
            "/v1/stt",
            self.model,
            data=form,
            files={"file": ("audio", data, "application/octet-stream")},
        )
        try:
            body = response.json()
            text = str(body["text"])
            duration = float(body.get("duration") or fmt.duration_s(len(data)))
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"unreadable transcription: {response.text[:300]}",
                provider=self.name,
                model=self.model,
            ) from exc
        self.usage = Usage(audio_seconds=(self.usage.audio_seconds or 0.0) + duration)
        yield Transcript(text=text, is_final=True, audio_duration_s=duration)


def _error(response: httpx.Response, provider: str, model: str) -> ProviderError:
    """A 404 is an unknown voice or model."""
    status = response.status_code
    return ProviderError(
        http_failure(status, not_found=ProviderFailure.NOT_ENABLED),
        f"HTTP {status}: {response.text[:300]}",
        provider=provider,
        model=model,
    )


TTS_SPEC = ProviderSpec(
    name=TTS_NAME,
    modality=Modality.TTS,
    capabilities=ProviderCapabilities(modality=Modality.TTS, streams=False),
    credential_provider="xai",
    requires_extra="voice",
    default_model=DEFAULT_VOICE,
)
"""The spec string names the voice: ``xai.tts/<voice_id>``."""

STT_SPEC = ProviderSpec(
    name=STT_NAME,
    modality=Modality.STT,
    capabilities=ProviderCapabilities(modality=Modality.STT, streams=False),
    credential_provider="xai",
    requires_extra="voice",
    default_model=DEFAULT_STT_MODEL,
)
"""``word_timings`` and ``diarizes`` stay off: the API offers both, but a
:class:`~harbor_voice.providers.protocols.Transcript` carries neither."""


def build_tts(*, model: str = DEFAULT_VOICE, **config: Any) -> XAITTS:
    return XAITTS(voice=model, **config)


def build_stt(*, model: str = DEFAULT_STT_MODEL, **config: Any) -> XAISTT:
    return XAISTT(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = (
    (TTS_SPEC, build_tts),
    (STT_SPEC, build_stt),
)
