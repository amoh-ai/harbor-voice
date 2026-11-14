"""OpenRouter's omni models (audio in, text out): ``openrouter.omni``.

The self-hosted class pointed at ``https://openrouter.ai/api`` with
``OPENROUTER_API_KEY``, which is required. The model half of the spec is
OpenRouter's id, e.g. ``openrouter.omni/qwen/qwen3.8-omni-flash``.
"""

from __future__ import annotations

import os
from typing import Any

from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import require_key
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.openrouter.speech import BASE_URL
from harbor_voice.providers.openrouter.speech import BASE_URL_ENV
from harbor_voice.providers.openrouter.speech import CREDENTIALS
from harbor_voice.providers.openrouter.speech import KEY_ENV
from harbor_voice.providers.selfhosted.omni import OmniLLM

NAME = "openrouter.omni"


def build(*, model: str, **config: Any) -> OmniLLM:
    key = require_key(config.pop("api_key", None), KEY_ENV, provider=NAME, model=model)
    base = config.pop("base_url", None) or os.environ.get(BASE_URL_ENV) or BASE_URL
    return OmniLLM(model=model, base_url=base, api_key=key, name=NAME, **config)


SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.OMNI,
    capabilities=ProviderCapabilities(modality=Modality.OMNI, streams=False),
    credential_provider=CREDENTIALS,
)
"""No default model: name one of OpenRouter's audio-input models."""

REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
