"""Audio primitives: the canonical format, four clock domains, frames, stages.

Frozen first and changed last — everything else depends on this.
"""

from __future__ import annotations

from harbor_voice.audio.clock import Clock
from harbor_voice.audio.clock import ClockError
from harbor_voice.audio.clock import Interval
from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import CANONICAL_CHANNELS
from harbor_voice.audio.format import CANONICAL_FRAME_MS
from harbor_voice.audio.format import CANONICAL_SAMPLE_RATE_HZ
from harbor_voice.audio.format import FRAME_BYTES
from harbor_voice.audio.format import FRAME_SAMPLES
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.format import AudioFormatError
from harbor_voice.audio.format import Encoding
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import CASCADE_ONLY_STAGES
from harbor_voice.audio.stages import CASCADE_STAGES
from harbor_voice.audio.stages import S2S_OPTIONAL_STAGES
from harbor_voice.audio.stages import S2S_REQUIRED_STAGES
from harbor_voice.audio.stages import S2S_STAGES
from harbor_voice.audio.stages import Stage
from harbor_voice.audio.stages import StageMark

__all__ = [
    "CANONICAL",
    "CANONICAL_CHANNELS",
    "CANONICAL_FRAME_MS",
    "CANONICAL_SAMPLE_RATE_HZ",
    "CASCADE_ONLY_STAGES",
    "CASCADE_STAGES",
    "FRAME_BYTES",
    "FRAME_SAMPLES",
    "S2S_OPTIONAL_STAGES",
    "S2S_REQUIRED_STAGES",
    "S2S_STAGES",
    "AudioFormat",
    "AudioFormatError",
    "AudioFrame",
    "Clock",
    "ClockError",
    "Encoding",
    "Interval",
    "SessionClock",
    "Stage",
    "StageMark",
]
