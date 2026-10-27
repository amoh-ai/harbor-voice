"""Transport: a continuous, bidirectional frame stream."""

from __future__ import annotations

from harbor_voice.transport.base import DEFAULT_GAP_BUDGET_S
from harbor_voice.transport.base import ContractError
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import TransportClosedError
from harbor_voice.transport.base import TransportError
from harbor_voice.transport.base import TransportNotConnectedError
from harbor_voice.transport.base import UnsupportedOperationError
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.events import ControlEvent
from harbor_voice.transport.events import ControlEventKind
from harbor_voice.transport.events import GapBudgetExceeded
from harbor_voice.transport.events import GapTracker
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import StreamEvent

__all__ = [
    "DEFAULT_GAP_BUDGET_S",
    "ContractError",
    "ControlEvent",
    "ControlEventKind",
    "GapBudgetExceeded",
    "GapTracker",
    "InputSpeech",
    "OutputAudio",
    "OutputTranscript",
    "SessionConfig",
    "StreamEvent",
    "TransportClosedError",
    "TransportError",
    "TransportNotConnectedError",
    "UnsupportedOperationError",
    "VoiceTransport",
]
