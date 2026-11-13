"""Speech to text and text to speech through OpenRouter.

OpenRouter serves both on OpenAI's audio routes, so these are the
self-hosted classes pointed at ``https://openrouter.ai/api`` with
``OPENROUTER_API_KEY``. Two things differ from a vLLM server:

- speech comes back as ``pcm`` (or ``mp3``), never WAV, with the rate in the
  ``Content-Type``;
- most voices must be named. Each model's ``supported_voices`` are in
  ``GET /api/v1/models?output_modalities=speech``.

The model half of the spec is OpenRouter's id, e.g.
``openrouter.stt/openai/whisper-large-v3-turbo`` or
``openrouter.tts/elevenlabs/eleven-flash-v2.5`` with ``voice="sarah"``.
"""

from __future__ import annotations

import os
from typing import Any

from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.selfhosted.speech import SelfHostedSTT
from harbor_voice.providers.selfhosted.speech import SelfHostedTTS

STT_NAME = "openrouter.stt"
TTS_NAME = "openrouter.tts"
KEY_ENV = "OPENROUTER_API_KEY"
BASE_URL_ENV = "OPENROUTER_BASE_URL"
BASE_URL = "https://openrouter.ai/api"
CREDENTIALS = "openrouter"


def _base_url(config: dict[str, Any]) -> str:
    return config.pop("base_url", None) or os.environ.get(BASE_URL_ENV) or BASE_URL


def build_stt(*, model: str, **config: Any) -> SelfHostedSTT:
    return SelfHostedSTT(
        model=model,
        base_url=_base_url(config),
        name=STT_NAME,
        api_key_env=KEY_ENV,
        **config,
    )


def build_tts(*, model: str, **config: Any) -> SelfHostedTTS:
    return SelfHostedTTS(
        model=model,
        base_url=_base_url(config),
        name=TTS_NAME,
        api_key_env=KEY_ENV,
        response_format="pcm",
        **config,
    )


OPENROUTER_STT_SPEC = ProviderSpec(
    name=STT_NAME,
    modality=Modality.STT,
    capabilities=ProviderCapabilities(modality=Modality.STT, streams=False),
    credential_provider=CREDENTIALS,
)

OPENROUTER_TTS_SPEC = ProviderSpec(
    name=TTS_NAME,
    modality=Modality.TTS,
    capabilities=ProviderCapabilities(modality=Modality.TTS, streams=False),
    credential_provider=CREDENTIALS,
)
"""No default models: name one of OpenRouter's."""


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = (
    (OPENROUTER_STT_SPEC, build_stt),
    (OPENROUTER_TTS_SPEC, build_tts),
)
