"""An LLM on your own server, over OpenAI's chat completions route.

vLLM, SGLang and Ollama all serve ``/v1/chat/completions``, so this is
:class:`~harbor_voice.providers.openai.llm.OpenAILLM` pointed at
``SELFHOSTED_LLM_BASE_URL`` (with or without ``/v1``), with
``SELFHOSTED_API_KEY`` as the bearer token when the server was started with
``--api-key``. The model half of the spec is the served model, e.g.
``selfhosted.llm/Qwen/Qwen3-8B``.

``litellm.llm/hosted_vllm/<model>`` reaches the same servers through
litellm; this goes direct, with no litellm in between.
"""

from __future__ import annotations

import os
from typing import Any

from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.openai.llm import OpenAILLM
from harbor_voice.providers.selfhosted.speech import API_KEY_ENV
from harbor_voice.providers.selfhosted.speech import sandbox_env

NAME = "selfhosted.llm"
BASE_URL_ENV = "SELFHOSTED_LLM_BASE_URL"


def build(*, model: str, **config: Any) -> OpenAILLM:
    base = config.pop("api_base", None) or os.environ.get(BASE_URL_ENV)
    if not base:
        raise ProviderError(
            ProviderFailure.AUTH,
            f"{BASE_URL_ENV} is not set and no api_base was configured",
            provider=NAME,
            model=model,
        )
    # A server started without --api-key ignores the token, so any value does.
    key = config.pop("api_key", None) or os.environ.get(API_KEY_ENV) or "none"
    llm = OpenAILLM(
        model=model,
        api_base=base.rstrip("/").removesuffix("/v1") + "/v1",
        api_key=key,
        **config,
    )
    llm.name = NAME
    return llm


SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.LLM,
    capabilities=ProviderCapabilities(modality=Modality.LLM, streams=False),
    credential_provider="selfhosted-llm",
    sandbox_env=sandbox_env,
)
"""No default model: the server decides what it serves."""

REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
