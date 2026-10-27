"""tau-voice's phone line: the caller's audio as the agent receives it.

A port of tau2's ``StreamingTelephonyConverter`` (``voice/synthesis/
audio_effects/effects.py``), kept bit-exact — verified against tau2's own
output in the tests. Per frame, state carried across frames:

1. ``audioop.ratecv`` from the synthesis rate (16 kHz) to 8 kHz;
2. a 4th-order Butterworth band-pass, 300–3400 Hz (``scipy.signal.lfilter``);
3. ×0.9 headroom, clip to int16;
4. G.711 μ-law.

It is a :class:`~harbor_voice.perturbation.encoder.WireEncoder`: the last
stage of the caller's signal chain, after any layers, where the format
changes. The resample, filter and headroom are tau2's recipe, so the encoder
lives with the benchmark; the codec itself is core's
:mod:`harbor_voice.audio.g711`. tau2 converts 0.2 s chunks; this is fed 20 ms
frames, and the tests prove the bytes are the same.
"""

from __future__ import annotations

import audioop
from typing import Any
from typing import override

from harbor_voice._deps import numpy as _numpy
from harbor_voice.audio import g711
from harbor_voice.audio.format import AudioFormat
from harbor_voice.perturbation.base import SeedSet
from harbor_voice.perturbation.encoder import TELEPHONY
from harbor_voice.perturbation.encoder import WireEncoder

TELEPHONY_RATE_HZ = TELEPHONY.sample_rate_hz
LOW_HZ = 300
HIGH_HZ = 3_400
HEADROOM = 0.9
"""tau2's ``TELEPHONY_LOW_FREQ`` / ``TELEPHONY_HIGH_FREQ`` / ``TELEPHONY_HEADROOM``."""


class Tau2Telephony(WireEncoder):
    """Streaming PCM16 → 8 kHz G.711 μ-law, as tau2 does it."""

    def __init__(self) -> None:
        from scipy import signal

        super().__init__(name="tau2-telephony")
        nyquist = TELEPHONY_RATE_HZ / 2
        self._b, self._a = signal.butter(
            4, [LOW_HZ / nyquist, HIGH_HZ / nyquist], btype="band"
        )
        self._lfilter = signal.lfilter
        self._zi: Any = None
        self._ratecv_state: Any = None
        self.reset(SeedSet.derive(0))

    @property
    @override
    def output_format(self) -> AudioFormat:
        return TELEPHONY

    @override
    def _encode(self, pcm: bytes, fmt: AudioFormat) -> bytes:
        if not pcm:
            return b""
        np = _numpy()
        resampled, self._ratecv_state = audioop.ratecv(
            pcm, 2, 1, fmt.sample_rate_hz, TELEPHONY_RATE_HZ, self._ratecv_state
        )
        samples = np.frombuffer(resampled, dtype=np.int16).astype(np.float64)
        if len(samples) > 0:
            samples, self._zi = self._lfilter(self._b, self._a, samples, zi=self._zi)
        clipped = np.clip(samples * HEADROOM, -32768, 32767).astype(np.int16)
        return g711.encode(clipped.tobytes())

    @override
    def reset(self, seeds: SeedSet) -> None:
        super().reset(seeds)
        np = _numpy()
        self._zi = np.zeros(max(len(self._a), len(self._b)) - 1)
        self._ratecv_state = None

    @override
    def params(self) -> dict[str, Any]:
        return {
            "recipe": "tau2 StreamingTelephonyConverter",
            "band_hz": [LOW_HZ, HIGH_HZ],
            "headroom": HEADROOM,
            "encoding": "mulaw",
            "sample_rate_hz": TELEPHONY_RATE_HZ,
        }


def ulaw_to_pcm16(ulaw: bytes) -> bytes:
    """G.711 μ-law → PCM16-LE at the same rate. For recording and listening."""
    return g711.decode(ulaw)
