"""The canonical audio format, and the rules for leaving it.

One internal format; conversion happens at the adapter edge
only, and every conversion is recorded as a stage — a resample to 8 kHz for
telephony is a declared channel event, not an invisible accident.

::

    CANONICAL  =  PCM16-LE · mono · 24 kHz · 20 ms frames
                  480 samples · 960 bytes per frame

24 kHz because OpenAI Realtime and Gemini Live both emit there; 20 ms because
that is the telephony and WebRTC frame. Both are revisitable, so nothing below
hard-codes the numbers twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import override

CANONICAL_SAMPLE_RATE_HZ = 24_000
CANONICAL_CHANNELS = 1
CANONICAL_FRAME_MS = 20

Encoding = Literal["pcm16", "mulaw", "alaw"]

_BYTES_PER_SAMPLE: dict[Encoding, int] = {"pcm16": 2, "mulaw": 1, "alaw": 1}


class AudioFormatError(ValueError):
    """An audio format is malformed, or a buffer does not match its format."""


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """A sample rate, channel count and encoding.

    Defaults to the canonical format, so ``AudioFormat()`` is the thing almost
    everything uses and a non-default is visible at the call site.
    """

    sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ
    channels: int = CANONICAL_CHANNELS
    encoding: Encoding = "pcm16"

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise AudioFormatError(
                f"sample_rate_hz must be positive, got {self.sample_rate_hz}"
            )
        if self.channels <= 0:
            raise AudioFormatError(f"channels must be positive, got {self.channels}")
        if self.encoding not in _BYTES_PER_SAMPLE:
            raise AudioFormatError(
                f"unknown encoding {self.encoding!r}; "
                f"expected one of {sorted(_BYTES_PER_SAMPLE)}"
            )

    @property
    def bytes_per_sample(self) -> int:
        """Bytes for one sample in one channel."""
        return _BYTES_PER_SAMPLE[self.encoding]

    @property
    def bytes_per_frame_slot(self) -> int:
        """Bytes for one sample across all channels."""
        return self.bytes_per_sample * self.channels

    @property
    def is_canonical(self) -> bool:
        """Whether this is the format the rest of harbor-voice assumes."""
        return self == CANONICAL

    def frame_bytes(self, duration_ms: int = CANONICAL_FRAME_MS) -> int:
        """Bytes in a frame of ``duration_ms``.

        Raises:
            AudioFormatError: if the duration does not land on a whole sample.
        """
        samples, remainder = divmod(self.sample_rate_hz * duration_ms, 1000)
        if remainder:
            raise AudioFormatError(
                f"{duration_ms} ms is not a whole number of samples at "
                f"{self.sample_rate_hz} Hz"
            )
        return samples * self.bytes_per_frame_slot

    def duration_s(self, n_bytes: int) -> float:
        """Duration of ``n_bytes`` in this format."""
        self.validate_buffer(n_bytes)
        samples = n_bytes // self.bytes_per_frame_slot
        return samples / self.sample_rate_hz

    def validate_buffer(self, n_bytes: int) -> None:
        """Raise unless ``n_bytes`` is a whole number of multi-channel samples.

        This is the odd-length rejection invariant. A PCM16 buffer with an odd
        byte count is a truncated or misaligned read, and silently accepting it
        shifts every subsequent sample by one byte — which sounds like noise and
        reads like a transport bug.
        """
        if n_bytes < 0:
            raise AudioFormatError(f"buffer length cannot be negative: {n_bytes}")
        slot = self.bytes_per_frame_slot
        if n_bytes % slot:
            raise AudioFormatError(
                f"buffer of {n_bytes} bytes is not a whole number of samples "
                f"for {self} ({slot} bytes per sample slot)"
            )

    @override
    def __str__(self) -> str:
        return f"{self.encoding}/{self.sample_rate_hz}Hz/{self.channels}ch"


CANONICAL = AudioFormat()
"""The one internal format. See the module docstring."""

FRAME_SAMPLES = CANONICAL_SAMPLE_RATE_HZ * CANONICAL_FRAME_MS // 1000
"""Samples in one canonical frame: 480."""

FRAME_BYTES = CANONICAL.frame_bytes()
"""Bytes in one canonical frame: 960."""
