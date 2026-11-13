"""Fish Audio providers: text to speech (``fishaudio.tts``) and speech to
text (``fishaudio.stt``).

Imported lazily, like the AWS package: nothing here is reachable from
``harbor_voice.providers`` at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.fishaudio.stt import FISHAUDIO_STT_SPEC
from harbor_voice.providers.fishaudio.stt import FishAudioSTT
from harbor_voice.providers.fishaudio.tts import FISHAUDIO_TTS_SPEC
from harbor_voice.providers.fishaudio.tts import FishAudioTTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = ["FISHAUDIO_STT_SPEC", "FISHAUDIO_TTS_SPEC", "FishAudioSTT", "FishAudioTTS"]


def register_all(registry: ProviderRegistry) -> None:
    """Add every Fish Audio provider to a registry. See ``aws.register_all``."""
    from harbor_voice.providers.fishaudio import stt
    from harbor_voice.providers.fishaudio import tts

    for spec, build in (*tts.REGISTRATIONS, *stt.REGISTRATIONS):
        registry.register(spec, build)
