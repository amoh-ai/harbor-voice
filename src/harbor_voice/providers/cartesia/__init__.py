"""Cartesia providers: Sonic text to speech (``cartesia.tts``) and Ink speech
to text (``cartesia.stt``).

Imported lazily, like the AWS package: nothing here is reachable from
``harbor_voice.providers`` at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.cartesia.stt import CARTESIA_STT_SPEC
from harbor_voice.providers.cartesia.stt import CartesiaSTT
from harbor_voice.providers.cartesia.tts import CARTESIA_TTS_SPEC
from harbor_voice.providers.cartesia.tts import CartesiaTTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = ["CARTESIA_STT_SPEC", "CARTESIA_TTS_SPEC", "CartesiaSTT", "CartesiaTTS"]


def register_all(registry: ProviderRegistry) -> None:
    """Add every Cartesia provider to a registry. See ``aws.register_all``."""
    from harbor_voice.providers.cartesia import stt
    from harbor_voice.providers.cartesia import tts

    for spec, build in (*tts.REGISTRATIONS, *stt.REGISTRATIONS):
        registry.register(spec, build)
