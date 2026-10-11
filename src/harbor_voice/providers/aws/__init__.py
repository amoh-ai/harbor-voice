"""AWS providers.

Imported lazily. The offline reference set and the whole scoring path must
keep working with no cloud SDK installed, so nothing here is reachable from
``harbor_voice.providers`` at import time.

Two stacks live here, and which one you get depends on IAM:

``nova_sonic``
    Speech-to-speech as a transport. Needs only Bedrock.
``bedrock``
    STT, LLM and TTS slots **without Polly or Transcribe** — the fallback for
    a key that has Bedrock and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harbor_voice.providers.aws.bedrock import BedrockLLM
from harbor_voice.providers.aws.bedrock import NovaSonicTTS
from harbor_voice.providers.aws.bedrock import SpokenAudio
from harbor_voice.providers.aws.bedrock import VoxtralSTT
from harbor_voice.providers.aws.nova_sonic import MODEL_ID
from harbor_voice.providers.aws.nova_sonic import NovaSonicTransport

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.registry import ProviderRegistry

__all__ = [
    "MODEL_ID",
    "BedrockLLM",
    "NovaSonicTTS",
    "NovaSonicTransport",
    "SpokenAudio",
    "VoxtralSTT",
]


def register_all(registry: ProviderRegistry) -> None:
    """Add every AWS provider to a registry.

    Separate from import so that merely importing the package does not mutate
    the default registry — an import with a side effect on global state makes
    the provider you get depend on import order.
    """
    from harbor_voice.providers.aws import bedrock, nova_sonic

    for spec, build in (*nova_sonic.REGISTRATIONS, *bedrock.REGISTRATIONS):
        registry.register(spec, build)
