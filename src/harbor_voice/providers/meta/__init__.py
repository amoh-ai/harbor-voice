"""Meta providers: Muse Voice Transcribe speech to text (``meta.stt``).

Meta Model API has no text to speech and no speech-to-speech API, so this is
the only Meta voice provider. Muse models with text output go through litellm.

Imported lazily, like the AWS package: nothing here is reachable from
``harbor_voice.providers`` at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.meta.stt import META_STT_SPEC
from harbor_voice.providers.meta.stt import MetaSTT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = ["META_STT_SPEC", "MetaSTT"]


def register_all(registry: ProviderRegistry) -> None:
    """Add every Meta provider to a registry. See ``aws.register_all``."""
    from harbor_voice.providers.meta import stt

    for spec, build in stt.REGISTRATIONS:
        registry.register(spec, build)
