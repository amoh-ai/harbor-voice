"""Deepgram providers: streaming STT (``deepgram.stt``), Aura TTS
(``deepgram.tts``) and the Voice Agent API as a speech-to-speech transport
(``deepgram.agent``). All three read ``DEEPGRAM_API_KEY``.

Imported lazily, like the other vendor packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry


def register_all(registry: ProviderRegistry) -> None:
    """Add every Deepgram provider to a registry. See ``aws.register_all``."""
    from harbor_voice.providers.deepgram import agent
    from harbor_voice.providers.deepgram import stt
    from harbor_voice.providers.deepgram import tts

    for spec, build in (*stt.REGISTRATIONS, *tts.REGISTRATIONS, *agent.REGISTRATIONS):
        registry.register(spec, build)
