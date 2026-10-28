"""OpenAI providers. Imported lazily, like the AWS and ElevenLabs packages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.openai.llm import OPENAI_LLM_SPEC
from harbor_voice.providers.openai.llm import ChatReply
from harbor_voice.providers.openai.llm import ChatToolCall
from harbor_voice.providers.openai.llm import OpenAILLM

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = ["OPENAI_LLM_SPEC", "ChatReply", "ChatToolCall", "OpenAILLM"]


def register_all(registry: ProviderRegistry) -> None:
    """Add every OpenAI provider to a registry. See ``aws.register_all``."""
    from harbor_voice.providers.openai import live
    from harbor_voice.providers.openai import llm
    from harbor_voice.providers.openai import realtime

    for spec, build in (
        *llm.REGISTRATIONS,
        *realtime.REGISTRATIONS,
        *live.REGISTRATIONS,
    ):
        registry.register(spec, build)
