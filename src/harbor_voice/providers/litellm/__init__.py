"""litellm providers. Imported lazily, like the other vendor packages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.litellm import llm
from harbor_voice.providers.litellm.llm import LITELLM_LLM_SPEC
from harbor_voice.providers.litellm.llm import LiteLLMChat
from harbor_voice.providers.litellm.llm import credential_slug

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = [
    "LITELLM_LLM_SPEC",
    "LiteLLMChat",
    "credential_slug",
    "register_all",
]


def register_all(registry: ProviderRegistry) -> None:
    """Add every litellm provider to a registry. See ``aws.register_all``."""
    for spec, build in llm.REGISTRATIONS:
        registry.register(spec, build)
