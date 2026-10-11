"""ATIF-voice: the trajectory format and its recorder.

The **model** and the **renderer** import eagerly; the **recorder** does not.
Reading a trajectory and writing one have different dependency
cones: scoring needs this data model and nothing else, while recording needs
transports, simulators and the provider layer. Importing a submodule runs its
package ``__init__``, so an eager ``Recorder`` here put the credential table
for ninety-five providers into every trial container that only wanted to
score a JSON file.
"""

from __future__ import annotations

from harbor_voice.trajectory.model import VOICE_EXTRA_KEY
from harbor_voice.trajectory.model import VOICE_SCHEMA_VERSION
from harbor_voice.trajectory.model import Artifacts
from harbor_voice.trajectory.model import AudioRef
from harbor_voice.trajectory.model import BargeInOutcome
from harbor_voice.trajectory.model import Clocks
from harbor_voice.trajectory.model import ConfigMatrix
from harbor_voice.trajectory.model import EventType
from harbor_voice.trajectory.model import PartyRecord
from harbor_voice.trajectory.model import PartyRole
from harbor_voice.trajectory.model import StageRecord
from harbor_voice.trajectory.model import TimelineEvent
from harbor_voice.trajectory.model import TokenSpan
from harbor_voice.trajectory.model import ValidityCheck
from harbor_voice.trajectory.model import ValidityRecord
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.model import VoiceTurn
from harbor_voice.trajectory.render import PlacedAudio
from harbor_voice.trajectory.render import RenderResult
from harbor_voice.trajectory.render import downsample_to_fit
from harbor_voice.trajectory.render import placed_from_frames
from harbor_voice.trajectory.render import render_timeline
from harbor_voice.trajectory.render import to_wav

__all__ = [
    "VOICE_EXTRA_KEY",
    "VOICE_SCHEMA_VERSION",
    "Artifacts",
    "AudioRef",
    "BargeInOutcome",
    "Clocks",
    "ConfigMatrix",
    "EventType",
    "PartyRecord",
    "PartyRole",
    "PlacedAudio",
    "Recorder",
    "RecorderConfig",
    "RenderResult",
    "StageRecord",
    "TimelineEvent",
    "TokenSpan",
    "ValidityCheck",
    "ValidityRecord",
    "VoiceTrajectory",
    "VoiceTurn",
    "downsample_to_fit",
    "matrix_from",
    "placed_from_frames",
    "render_timeline",
    "to_wav",
]


_LAZY = {"Recorder", "RecorderConfig", "matrix_from"}
"""Recorder names, resolved on first touch — see the module docstring."""


def __getattr__(name: str) -> object:
    """PEP 562 lazy export, so ``harbor_voice.metrics`` stays light."""
    if name in _LAZY:
        from harbor_voice.trajectory import recorder

        return getattr(recorder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | _LAZY)
