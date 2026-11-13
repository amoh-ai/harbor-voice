"""Self-hosted speech models (vLLM, vLLM-Omni, SGLang): speech to text, text
to speech and speech to speech. Imported lazily, like the other vendor
packages. Self-hosted LLMs go through ``litellm.llm``."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
]


def register_all(registry: ProviderRegistry) -> None:
    """Add every self-hosted provider to a registry. See ``aws.register_all``."""
    from harbor_voice.providers.selfhosted import realtime
    from harbor_voice.providers.selfhosted import speech

    for spec, build in (*speech.REGISTRATIONS, *realtime.REGISTRATIONS):
        registry.register(spec, build)
