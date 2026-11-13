"""Microsoft MAI-Transcribe speech to text, through Azure Speech.

MAI-Transcribe has no endpoint of its own: it is Azure Speech's fast
transcription operation, ``POST {endpoint}/speechtotext/transcriptions:transcribe``,
with ``enhancedMode`` naming the model in the ``definition`` form field
(learn.microsoft.com/azure/ai-services/speech-service/mai-transcribe).

One synchronous request per turn: the turn as a WAV upload, one JSON result
back. No partials and no first-byte mark, which the spec declares.
``MAI-Transcribe-1`` was retired on 2026-08-20, so the default is
``MAI-Transcribe-2``. The style stays ``verbatim`` (the service default),
because said-vs-heard compares words and ``clean`` would drop disfluencies.

The endpoint is the Speech resource's own
(``https://<resource>.cognitiveservices.azure.com``), in a region that offers
MAI-Transcribe. Billed per hour of audio, which :attr:`usage` records from the
response's ``durationMilliseconds``.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import Any

import httpx

from harbor_voice.audio.frame import AudioFrame
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
from harbor_voice.trajectory.render import to_wav

NAME = "microsoft.stt"
DEFAULT_MODEL = "MAI-Transcribe-2"
API_VERSION = "2025-10-15"
KEY_ENV = "AZURE_SPEECH_KEY"
ENDPOINT_ENV = "AZURE_SPEECH_ENDPOINT"


class MicrosoftSTT(PooledHTTP):
    """Whole-turn transcription with MAI-Transcribe.

    Args:
        model: Sent as ``enhancedMode.model``.
        locale: One language code such as ``en``; ``None`` lets the service
            detect it.
        api_key: Defaults to ``AZURE_SPEECH_KEY``.
        endpoint: Defaults to ``AZURE_SPEECH_ENDPOINT``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        locale: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = NAME
        self.model = model
        self.locale = locale
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout_s = timeout_s
        self._http = HTTPSession(client)
        self.usage = Usage(audio_seconds=0.0)
        """The last turn's audio seconds, as Azure reports them."""

    async def transcribe(
        self, frames: Sequence[AudioFrame]
    ) -> AsyncIterator[Transcript]:
        """Yield exactly one final transcript. Never a partial."""
        pcm = b"".join(f.pcm for f in frames)
        if not pcm:
            yield Transcript(text="", is_final=True, audio_duration_s=0.0)
            return
        fmt = frames[0].fmt
        duration = fmt.duration_s(len(pcm))
        definition: dict[str, Any] = {
            "enhancedMode": {"enabled": True, "model": self.model}
        }
        if self.locale:
            definition["locales"] = [self.locale]
        url = (
            f"{self._env(self._endpoint, ENDPOINT_ENV).rstrip('/')}"
            f"/speechtotext/transcriptions:transcribe?api-version={API_VERSION}"
        )
        headers = {"Ocp-Apim-Subscription-Key": self._env(self._api_key, KEY_ENV)}
        files = {"audio": ("turn.wav", to_wav(pcm, fmt), "audio/wav")}
        client = self._http.client(self._timeout_s)
        try:
            response = await client.post(
                url,
                headers=headers,
                data={"definition": json.dumps(definition)},
                files=files,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                ProviderFailure.TIMEOUT, str(exc), provider=NAME, model=self.model
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                ProviderFailure.STREAM_BROKEN, str(exc), provider=NAME, model=self.model
            ) from exc
        if response.status_code != 200:
            raise _error(response.status_code, response.text, self.model)
        try:
            result = response.json()
            text = " ".join(str(p["text"]) for p in result["combinedPhrases"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"no transcript in the response: {response.text[:300]}",
                provider=NAME,
                model=self.model,
            ) from exc
        billed = result.get("durationMilliseconds")
        self.usage = Usage(
            audio_seconds=billed / 1000 if isinstance(billed, int) else duration
        )
        yield Transcript(text=text.strip(), is_final=True, audio_duration_s=duration)

    def _env(self, explicit: str | None, name: str) -> str:
        """The argument, else the variable. Never logged."""
        value = explicit or os.environ.get(name)
        if not value:
            raise ProviderError(
                ProviderFailure.AUTH,
                f"{name} is not set",
                provider=NAME,
                model=self.model,
            )
        return value


def _error(status: int, detail: str, model: str) -> ProviderError:
    """A 404 is a region without MAI-Transcribe."""
    return ProviderError(
        http_failure(status, not_found=ProviderFailure.REGION),
        f"Azure Speech returned {status}: {detail[:300]}",
        provider=NAME,
        model=model,
    )


MICROSOFT_STT_SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.STT,
    capabilities=ProviderCapabilities(
        modality=Modality.STT,
        streams=False,
        emits_partials=False,
        word_timings=False,
        diarizes=False,
    ),
    credential_provider="azure-speech",
    default_model=DEFAULT_MODEL,
)


def build_stt(*, model: str = DEFAULT_MODEL, **config: Any) -> MicrosoftSTT:
    return MicrosoftSTT(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((MICROSOFT_STT_SPEC, build_stt),)
