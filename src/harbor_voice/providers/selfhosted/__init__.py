"""Models on your own server (vLLM, vLLM-Omni, SGLang, Ollama): speech to
text, an LLM, an omni model (audio in, text out), text to speech and
speech to speech, each over its OpenAI
route. Imported lazily, like the other vendor packages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.selfhosted import llm
from harbor_voice.providers.selfhosted import omni
from harbor_voice.providers.selfhosted import realtime
from harbor_voice.providers.selfhosted import speech
from harbor_voice.providers.selfhosted.realtime import SelfHostedRealtimeTransport
from harbor_voice.providers.selfhosted.speech import SELFHOSTED_STT_SPEC
from harbor_voice.providers.selfhosted.speech import SELFHOSTED_TTS_SPEC
from harbor_voice.providers.selfhosted.speech import SelfHostedSTT
from harbor_voice.providers.selfhosted.speech import SelfHostedTTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = [
    "SELFHOSTED_STT_SPEC",
    "SELFHOSTED_TTS_SPEC",
    "SelfHostedRealtimeTransport",
    "SelfHostedSTT",
    "SelfHostedTTS",
    "register_all",
]


def register_all(registry: ProviderRegistry) -> None:
    """Add every self-hosted provider to a registry. See ``aws.register_all``."""
    for spec, build in (
        *speech.REGISTRATIONS,
        *llm.REGISTRATIONS,
        *omni.REGISTRATIONS,
        *realtime.REGISTRATIONS,
    ):
        registry.register(spec, build)
