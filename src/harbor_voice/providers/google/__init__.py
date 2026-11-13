"""Google providers. Imported lazily, like the other vendor packages.

``google.live``: Gemini Live as a speech-to-speech transport, signed in with
an AI Studio API key or with Vertex AI's Application Default Credentials.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.google.live import DEFAULT_MODEL
from harbor_voice.providers.google.live import GeminiLiveTransport

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = ["DEFAULT_MODEL", "GeminiLiveTransport"]


def register_all(registry: ProviderRegistry) -> None:
    """Add every Google provider to a registry. See ``aws.register_all``."""
    from harbor_voice.providers.google import live

    for spec, build in live.REGISTRATIONS:
        registry.register(spec, build)
