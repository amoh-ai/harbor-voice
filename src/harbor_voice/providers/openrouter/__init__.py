"""OpenRouter's speech models: ``openrouter.stt``, ``openrouter.tts`` and
``openrouter.omni`` (audio in, text out).

OpenRouter's LLMs go through ``litellm.llm/openrouter/<model>``. Imported
lazily, like the other vendor packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.openrouter import omni
from harbor_voice.providers.openrouter import speech
from harbor_voice.providers.openrouter.speech import OPENROUTER_STT_SPEC
from harbor_voice.providers.openrouter.speech import OPENROUTER_TTS_SPEC

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = [
    "OPENROUTER_STT_SPEC",
    "OPENROUTER_TTS_SPEC",
    "register_all",
]


def register_all(registry: ProviderRegistry) -> None:
    """Add OpenRouter's speech providers to a registry. See ``aws.register_all``."""
    for spec, build in (*speech.REGISTRATIONS, *omni.REGISTRATIONS):
        registry.register(spec, build)
