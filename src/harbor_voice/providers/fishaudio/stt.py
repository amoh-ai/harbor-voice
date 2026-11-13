"""Fish Audio speech to text: ``transcribe-1-pro`` over ``POST /v1/asr``.

One request per turn, synchronous: the turn as a WAV upload (raw PCM is
rejected), the model in a ``model`` header, and one JSON transcript back
(docs.fish.audio/api-reference/endpoint/openapi-v1/speech-to-text). So no
partials and no first-byte mark, which the spec declares.

Two request fields are fixed, because said-vs-heard compares words:
``diarize=false`` keeps ``<|speaker:N|>`` markers out of the text, and
``tag_audio_events=false`` keeps cues such as ``[laughter]`` out of it.

The header must be exactly ``transcribe-1-pro``; anything else is served
and billed as ``transcribe-1`` without an error. Billed per second of audio,
which :attr:`usage` records from the response's ``duration``.

The realtime websocket (``transcribe-stream-1``) is in Preview and not used.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import Any

import httpx

from harbor_voice.audio.frame import AudioFrame
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.fishaudio.tts import API_BASE
from harbor_voice.providers.fishaudio.tts import api_key
from harbor_voice.providers.fishaudio.tts import error
from harbor_voice.providers.protocols import Transcript
from harbor_voice.trajectory.render import to_wav

NAME = "fishaudio.stt"
DEFAULT_MODEL = "transcribe-1-pro"
FIELDS = {"ignore_timestamps": "true", "diarize": "false", "tag_audio_events": "false"}


class FishAudioSTT:
    """Whole-turn transcription.

    Args:
        model: Fish Audio STT model, sent as the ``model`` header.
        api_key: Defaults to ``FISH_API_KEY``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        api_base: str = API_BASE,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = NAME
        self.model = model
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._timeout_s = timeout_s
        self._client = client
        self.usage = Usage(audio_seconds=0.0)
        """The last turn's audio seconds, as Fish Audio reports them."""

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
        headers = {
            "Authorization": f"Bearer {api_key(self._api_key, NAME, self.model)}",
            "model": self.model,
        }
        files = {"audio": ("turn.wav", to_wav(pcm, fmt), "audio/wav")}
        client = self._client or httpx.AsyncClient(timeout=self._timeout_s)
        try:
            response = await client.post(
                f"{self._api_base}/v1/asr", headers=headers, data=FIELDS, files=files
            )
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
        if response.status_code != 200:
            raise error(response.status_code, response.text, NAME, self.model)
        try:
            result = response.json()
            text = str(result["text"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"no transcript in the response: {response.text[:300]}",
                provider=NAME,
                model=self.model,
            ) from exc
        self.usage = Usage(audio_seconds=float(result.get("duration", duration)))
        yield Transcript(text=text.strip(), is_final=True, audio_duration_s=duration)


FISHAUDIO_STT_SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.STT,
    capabilities=ProviderCapabilities(
        modality=Modality.STT,
        streams=False,
        emits_partials=False,
        word_timings=False,
        diarizes=False,
    ),
    credential_provider="fishaudio",
    requires_extra="voice",
    default_model=DEFAULT_MODEL,
)


def build_stt(*, model: str = DEFAULT_MODEL, **config: Any) -> FishAudioSTT:
    return FishAudioSTT(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((FISHAUDIO_STT_SPEC, build_stt),)
