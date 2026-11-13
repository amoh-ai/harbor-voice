"""Sarvam providers: Bulbul text to speech (``sarvam.tts``) and Saaras speech
to text (``sarvam.stt``).

Sarvam's chat models are OpenAI-compatible and go through litellm, so there is
no LLM here.

Imported lazily, like the AWS package: nothing here is reachable from
``harbor_voice.providers`` at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.sarvam.stt import SARVAM_STT_SPEC
from harbor_voice.providers.sarvam.stt import SarvamSTT
from harbor_voice.providers.sarvam.tts import SARVAM_TTS_SPEC
from harbor_voice.providers.sarvam.tts import SarvamTTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = ["SARVAM_STT_SPEC", "SARVAM_TTS_SPEC", "SarvamSTT", "SarvamTTS"]


def register_all(registry: ProviderRegistry) -> None:
    """Add every Sarvam provider to a registry. See ``aws.register_all``."""
    from harbor_voice.providers.sarvam import stt
    from harbor_voice.providers.sarvam import tts

    for spec, build in (*tts.REGISTRATIONS, *stt.REGISTRATIONS):
        registry.register(spec, build)
