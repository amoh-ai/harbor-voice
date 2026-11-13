"""xAI providers: the Grok Voice Agent API (``xai.realtime``) and the REST
text to speech (``xai.tts``) and speech to text (``xai.stt``).

Imported lazily, like the AWS package. Grok text models need nothing here:
harbor's ``xai`` entry already reaches them through litellm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.xai.realtime import XAIRealtimeTransport
from harbor_voice.providers.xai.speech import XAISTT
from harbor_voice.providers.xai.speech import XAITTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = ["XAISTT", "XAITTS", "XAIRealtimeTransport"]


def register_all(registry: ProviderRegistry) -> None:
    """Add every xAI provider to a registry. See ``aws.register_all``."""
    from harbor_voice.providers.xai import realtime
    from harbor_voice.providers.xai import speech

    for spec, build in (*realtime.REGISTRATIONS, *speech.REGISTRATIONS):
        registry.register(spec, build)
