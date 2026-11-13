"""Microsoft providers: MAI-Transcribe speech to text (``microsoft.stt``).

Imported lazily, like the AWS package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.microsoft.stt import MICROSOFT_STT_SPEC
from harbor_voice.providers.microsoft.stt import MicrosoftSTT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = ["MICROSOFT_STT_SPEC", "MicrosoftSTT"]


def register_all(registry: ProviderRegistry) -> None:
    """Add every Microsoft provider to a registry. See ``aws.register_all``."""
    from harbor_voice.providers.microsoft import stt

    for spec, build in stt.REGISTRATIONS:
        registry.register(spec, build)
