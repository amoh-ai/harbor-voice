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
    "Segment",
    "StageRecord",
    "TimelineEvent",
    "TokenSpan",
    "ToolEvent",
    "ValidityCheck",
    "ValidityRecord",
    "VoiceTrajectory",
    "VoiceTurn",
    "downsample_to_fit",
    "from_timeline",
    "matrix_from",
    "placed_from_frames",
    "render_timeline",
    "to_wav",
]


_LAZY = {
    "Recorder": "recorder",
    "RecorderConfig": "recorder",
    "matrix_from": "recorder",
    "Segment": "timed",
    "ToolEvent": "timed",
    "from_timeline": "timed",
}
"""Builder names, resolved on first touch — see the module docstring. The
real-time builder (``timed``) is light, but it is a writer like the
recorder, so it is reached the same way. ``ToolEvent`` here is the real-time
one; the lockstep builder's lives in ``trajectory.stepped``."""


def __getattr__(name: str) -> object:
    """PEP 562 lazy export, so ``harbor_voice.metrics`` stays light."""
    module = _LAZY.get(name)
    if module is not None:
        from importlib import import_module

        return getattr(import_module(f"{__name__}.{module}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY))
