"""Speech models on your own server, over the OpenAI audio API.

vLLM and SGLang both serve speech models behind OpenAI's routes:

``selfhosted.stt``
    ``POST {base_url}/v1/audio/transcriptions``: multipart, ``file`` and
    ``model``, JSON ``{"text": ...}`` back. vLLM (Whisper, Qwen3-ASR, Voxtral,
    Canary...) and SGLang-Omni (Qwen3-ASR, Whisper...).
``selfhosted.tts``
    ``POST {base_url}/v1/audio/speech``: JSON ``input``, ``model``,
    ``voice``, a WAV file back. vLLM-Omni and SGLang-Omni (Qwen3-TTS,
    Voxtral TTS, CosyVoice3...).

Both are whole-utterance: one request, one result. The servers can stream
each, but their streaming formats differ, so neither is used and both
declare ``streams=False``.

The model half of the spec is whatever the server was launched with, e.g.
``selfhosted.stt/openai/whisper-large-v3-turbo``. One server runs one model,
and speech recognition and synthesis usually run on different servers, so
each slot has its own base URL: ``SELFHOSTED_STT_BASE_URL`` and
``SELFHOSTED_TTS_BASE_URL`` (without ``/v1``), or ``base_url=`` in config.
``SELFHOSTED_API_KEY``, if set, is sent as a bearer token (``vllm serve
--api-key``).
"""

from __future__ import annotations

import io
import os
import wave
from collections.abc import AsyncIterator
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import httpx

from harbor_voice._deps import numpy as _numpy
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.format import FrameCutter
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.perturbation import dsp
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import http_failure
from harbor_voice.providers.base import HTTPSession
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.base import post_with_retries
from harbor_voice.providers.base import require_key
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.protocols import Transcript

STT_NAME = "selfhosted.stt"
TTS_NAME = "selfhosted.tts"
STT_BASE_URL_ENV = "SELFHOSTED_STT_BASE_URL"
TTS_BASE_URL_ENV = "SELFHOSTED_TTS_BASE_URL"
API_KEY_ENV = "SELFHOSTED_API_KEY"

RETRY_ATTEMPTS = 3


@dataclass(slots=True)
class SelfHostedSTT:
    """One-shot transcription: the utterance as a WAV file, one final back.

    Args:
        model: The model the server was launched with.
        language: ISO code, e.g. ``en``. Optional for Whisper, required by
            some models (``nvidia/canary-1b-v2``).
        base_url: Defaults to ``SELFHOSTED_STT_BASE_URL``.
    """

    model: str
    language: str | None = None
    base_url: str | None = None
    timeout_s: float = 60.0
    fmt: AudioFormat = CANONICAL
    name: str = STT_NAME
    api_key_env: str = API_KEY_ENV
    """The variable holding the bearer token, sent when it is set."""
    key_required: bool = False
    """Fail with AUTH when ``api_key_env`` is unset, as a hosted service would."""
    usage: Usage = field(default_factory=Usage)
    client: httpx.AsyncClient | None = None
    _http: HTTPSession = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._http = HTTPSession(self.client)

    async def aclose(self) -> None:
        """Close this provider's connections. Safe to call more than once."""
        await self._http.aclose()

    async def transcribe(
        self, frames: Sequence[AudioFrame]
    ) -> AsyncIterator[Transcript]:
        """Yield exactly one final transcript. Never a partial."""
        pcm = b"".join(f.pcm for f in frames)
        fmt = frames[0].fmt if frames else self.fmt
        duration = fmt.duration_s(len(pcm))
        if not pcm:
            yield Transcript(text="", is_final=True, audio_duration_s=0.0)
            return

        from harbor_voice.trajectory.render import to_wav

        data = {"model": self.model, "response_format": "json"}
        if self.language:
            data["language"] = self.language
        response = await _post(
            self,
            _base_url(self.base_url, STT_BASE_URL_ENV, self.name, self.model)
            + "/v1/audio/transcriptions",
            data=data,
            files={"file": ("audio.wav", to_wav(pcm, fmt), "audio/wav")},
        )
        try:
            text = str(response.json()["text"]).strip()
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"no text in the reply: {response.text[:300]}",
                provider=self.name,
                model=self.model,
            ) from exc
        self.usage = Usage(audio_seconds=duration)
        yield Transcript(text=text, is_final=True, audio_duration_s=duration)


@dataclass(slots=True)
class SelfHostedTTS:
    """Whole-utterance synthesis, converted to :attr:`audio_format`.

    Asks for WAV, which both servers default to, and reads the sample rate
    from its header: the native rate differs by model (24 kHz for Qwen3-TTS,
    44.1 kHz for Fish Speech), and a header is the same on every server.
    With ``response_format="pcm"`` (OpenRouter, which serves no WAV) the
    rate comes from the reply's ``Content-Type``,
    ``audio/pcm;rate=44100;channels=1``.

    Args:
        model: The model the server was launched with.
        voice: Speaker name; ``None`` leaves the server's default
            (``vivian`` on vLLM-Omni, ``default`` on SGLang-Omni).
        base_url: Defaults to ``SELFHOSTED_TTS_BASE_URL``.
    """

    model: str
    voice: str | None = None
    base_url: str | None = None
    timeout_s: float = 60.0
    audio_format: AudioFormat = CANONICAL
    name: str = TTS_NAME
    api_key_env: str = API_KEY_ENV
    """The variable holding the bearer token, sent when it is set."""
    key_required: bool = False
    """Fail with AUTH when ``api_key_env`` is unset, as a hosted service would."""
    response_format: str = "wav"
    """``wav``, or ``pcm`` with the rate in the reply's ``Content-Type``."""
    client: httpx.AsyncClient | None = None
    _http: HTTPSession = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._http = HTTPSession(self.client)

    async def aclose(self) -> None:
        """Close this provider's connections. Safe to call more than once."""
        await self._http.aclose()

    async def synthesize_pcm(self, text: str) -> bytes:
        """The whole utterance, in :attr:`audio_format`."""
        body: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "response_format": self.response_format,
        }
        if self.voice:
            body["voice"] = self.voice
        response = await _post(
            self,
            _base_url(self.base_url, TTS_BASE_URL_ENV, self.name, self.model)
            + "/v1/audio/speech",
            json=body,
        )
        if self.response_format == "pcm":
            return self._resample(response.content, _pcm_rate(response, self))
        return self._convert(response.content)

    async def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        """Frame-sized chunks of an utterance synthesised whole.

        The first chunk is not a streaming first byte, hence ``streams=False``.
        """
        del party
        if not text:
            return
        cutter = FrameCutter(self.audio_format)
        for frame in cutter.feed(await self.synthesize_pcm(text)):
            yield frame
        if tail := cutter.tail():
            yield tail

    def _convert(self, wav: bytes) -> bytes:
        try:
            with wave.open(io.BytesIO(wav)) as handle:
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                rate = handle.getframerate()
                pcm = handle.readframes(handle.getnframes())
        except (wave.Error, EOFError) as exc:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"the reply is not a WAV file: {exc}",
                provider=self.name,
                model=self.model,
            ) from exc
        if channels != 1 or width != 2:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"expected mono 16-bit audio, got {channels} channel(s) of "
                f"{8 * width}-bit",
                provider=self.name,
                model=self.model,
            )
        return self._resample(pcm, rate)

    def _resample(self, pcm: bytes, rate: int) -> bytes:
        if rate == self.audio_format.sample_rate_hz:
            return pcm
        converted = dsp.resample_clean(
            dsp.to_array(pcm), rate, self.audio_format.sample_rate_hz
        )
        return dsp.to_pcm(_numpy().asarray(converted))


def _pcm_rate(response: httpx.Response, owner: SelfHostedTTS) -> int:
    """The sample rate from ``audio/pcm;rate=44100;channels=1``; mono only."""
    params = dict(
        part.strip().partition("=")[::2]
        for part in response.headers.get("content-type", "").split(";")[1:]
    )
    try:
        rate = int(params["rate"])
    except (KeyError, ValueError) as exc:
        raise ProviderError(
            ProviderFailure.MALFORMED,
            f"PCM reply without a rate: {response.headers.get('content-type')!r}",
            provider=owner.name,
            model=owner.model,
        ) from exc
    if params.get("channels", "1") != "1":
        raise ProviderError(
            ProviderFailure.MALFORMED,
            f"expected mono audio, got {params['channels']} channels",
            provider=owner.name,
            model=owner.model,
        )
    return rate


# ------------------------------------------------------------------ http


def _base_url(configured: str | None, env: str, provider: str, model: str) -> str:
    url = configured or os.environ.get(env)
    if not url:
        raise ProviderError(
            ProviderFailure.AUTH,
            f"{env} is not set and no base_url was configured",
            provider=provider,
            model=model,
        )
    return url.rstrip("/").removesuffix("/v1")


async def _post(
    owner: SelfHostedSTT | SelfHostedTTS, url: str, **request: Any
) -> httpx.Response:
    """POST with retries on timeouts, 409, 429 and 5xx; 200 or a ProviderError."""
    if owner.key_required:
        key = require_key(
            None, owner.api_key_env, provider=owner.name, model=owner.model
        )
    else:
        key = os.environ.get(owner.api_key_env)
    response = await post_with_retries(
        owner._http.client(owner.timeout_s),
        url,
        attempts=RETRY_ATTEMPTS,
        provider=owner.name,
        model=owner.model,
        headers={"Authorization": f"Bearer {key}"} if key else {},
        **request,
    )
    if response.status_code == 200 and response.content:
        return response
    raise _error(response, owner.name, owner.model)


def _error(response: httpx.Response, provider: str, model: str) -> ProviderError:
    """A 404 is a model the server does not serve."""
    status = response.status_code
    kind = (
        ProviderFailure.MALFORMED  # a 200 with an empty body
        if status == 200
        else http_failure(status, not_found=ProviderFailure.NOT_ENABLED)
    )
    return ProviderError(
        kind,
        f"server returned {status}: {response.text[:300]}",
        provider=provider,
        model=model,
    )


# -------------------------------------------------------------- registry

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})


def sandbox_env(env: dict[str, str]) -> dict[str, str]:
    """Refuse a server address that a sandbox cannot reach.

    ``localhost`` inside Docker, Modal or Daytona is the sandbox itself, so
    every request would be refused and retried until it timed out.
    """
    for name, value in env.items():
        if name.endswith("_BASE_URL") and httpx.URL(value).host in LOOPBACK_HOSTS:
            raise ValueError(
                f"{name}={value} points at this machine, which a sandbox cannot "
                f"reach; use an address it can (host.docker.internal for local "
                f"Docker, or the server's public URL)"
            )
    return env


SELFHOSTED_STT_SPEC = ProviderSpec(
    name=STT_NAME,
    modality=Modality.STT,
    capabilities=ProviderCapabilities(modality=Modality.STT, streams=False),
    credential_provider="selfhosted-stt",
    sandbox_env=sandbox_env,
)

SELFHOSTED_TTS_SPEC = ProviderSpec(
    name=TTS_NAME,
    modality=Modality.TTS,
    capabilities=ProviderCapabilities(modality=Modality.TTS, streams=False),
    credential_provider="selfhosted-tts",
    sandbox_env=sandbox_env,
)
"""No default models: the server decides what it serves."""


def build_stt(*, model: str, **config: Any) -> SelfHostedSTT:
    return SelfHostedSTT(model=model, **config)


def build_tts(*, model: str, **config: Any) -> SelfHostedTTS:
    return SelfHostedTTS(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = (
    (SELFHOSTED_STT_SPEC, build_stt),
    (SELFHOSTED_TTS_SPEC, build_tts),
)
