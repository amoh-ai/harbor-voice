"""Model providers.

A provider is a **metered, fallible, capability-declaring resource**, not a
function. Its limits decide whether a metric is meaningful, its failures decide
whether a *trial* is valid, and its meter is the cost column.

The package is top-level rather than a detail of ``cascade/`` because the
cascade is one of four consumers; see ``protocols.py`` for the table.
"""

from __future__ import annotations

from harbor_voice.providers.base import Disposition
from harbor_voice.providers.base import ProviderCall
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderRef
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.capabilities import ProviderCapability
from harbor_voice.providers.capabilities import ProviderCapabilityError
from harbor_voice.providers.credentials import CredentialStatus
from harbor_voice.providers.offline import EchoLLM
from harbor_voice.providers.offline import ScriptedLLM
from harbor_voice.providers.offline import ScriptedSTT
from harbor_voice.providers.offline import ToneTTS
from harbor_voice.providers.protocols import LLMProvider
from harbor_voice.providers.protocols import STTProvider
from harbor_voice.providers.protocols import Transcript
from harbor_voice.providers.protocols import TTSProvider
from harbor_voice.providers.registry import ModalityMismatch
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.providers.registry import Registration
from harbor_voice.providers.registry import UnknownProvider
from harbor_voice.providers.registry import default_registry
from harbor_voice.providers.registry import parse_spec

__all__ = [
    "CredentialStatus",
    "default_registry",
    "Disposition",
    "EchoLLM",
    "LLMProvider",
    "Modality",
    "ModalityMismatch",
    "parse_spec",
    "ProviderCall",
    "ProviderCapabilities",
    "ProviderCapability",
    "ProviderCapabilityError",
    "ProviderError",
    "ProviderFailure",
    "ProviderRef",
    "ProviderRegistry",
    "ProviderSpec",
    "Registration",
    "ScriptedLLM",
    "ScriptedSTT",
    "STTProvider",
    "ToneTTS",
    "Transcript",
    "TTSProvider",
    "UnknownProvider",
    "Usage",
]
