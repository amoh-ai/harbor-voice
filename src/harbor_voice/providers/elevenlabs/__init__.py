"""ElevenLabs providers — the caller voice tau-voice uses, and ElevenLabs
Agents (``elevenlabs.agents``), the caller EVA uses.

Imported lazily, like the AWS package: nothing here is reachable from
``harbor_voice.providers`` at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.elevenlabs import agents
from harbor_voice.providers.elevenlabs import tts
from harbor_voice.providers.elevenlabs.agents import ElevenLabsAgentsTransport
from harbor_voice.providers.elevenlabs.tts import ELEVENLABS_TTS_SPEC
from harbor_voice.providers.elevenlabs.tts import ElevenLabsTTS
from harbor_voice.providers.elevenlabs.tts import TAU2_VOICE_SETTINGS
from harbor_voice.providers.elevenlabs.tts import VoiceSettings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = [
    "ELEVENLABS_TTS_SPEC",
    "TAU2_VOICE_SETTINGS",
    "ElevenLabsAgentsTransport",
    "ElevenLabsTTS",
    "VoiceSettings",
    "register_all",
]


def register_all(registry: ProviderRegistry) -> None:
    """Add every ElevenLabs provider to a registry. See ``aws.register_all``."""
    for spec, build in (*tts.REGISTRATIONS, *agents.REGISTRATIONS):
        registry.register(spec, build)
