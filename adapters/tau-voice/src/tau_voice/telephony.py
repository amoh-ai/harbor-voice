"""tau-voice's phone line: the caller's audio as the agent receives it.

A port of tau2's ``StreamingTelephonyConverter`` (``voice/synthesis/
audio_effects/effects.py``), kept bit-exact — verified against tau2's own
output in the tests. Per chunk, state carried across chunks:

1. ``audioop.ratecv`` from the synthesis rate (16 kHz) to 8 kHz;
2. a 4th-order Butterworth band-pass, 300–3400 Hz (``scipy.signal.lfilter``);
3. ×0.9 headroom, clip to int16;
4. G.711 μ-law (``audioop.lin2ulaw``).

The resample, filter and headroom are tau2's recipe, so they live with the
benchmark; the codec itself is core's :mod:`harbor_voice.audio.g711`.
"""

from __future__ import annotations

import audioop
from typing import Any

from harbor_voice._deps import numpy as _numpy
from harbor_voice.audio import g711

TELEPHONY_RATE_HZ = 8_000
LOW_HZ = 300
HIGH_HZ = 3_400
HEADROOM = 0.9
"""tau2's ``TELEPHONY_LOW_FREQ`` / ``TELEPHONY_HIGH_FREQ`` / ``TELEPHONY_HEADROOM``."""


class TelephonyConverter:
    """Streaming PCM16 → 8 kHz G.711 μ-law, as tau2 does it."""

    def __init__(self, input_rate_hz: int = 16_000) -> None:
        from scipy import signal

        np = _numpy()
        self.input_rate_hz = input_rate_hz
        nyquist = TELEPHONY_RATE_HZ / 2
        self._b, self._a = signal.butter(
            4, [LOW_HZ / nyquist, HIGH_HZ / nyquist], btype="band"
        )
        self._zi = np.zeros(max(len(self._a), len(self._b)) - 1)
        self._ratecv_state: Any = None
        self._lfilter = signal.lfilter

    def convert(self, pcm16: bytes) -> bytes:
        """One chunk of PCM16-LE at ``input_rate_hz`` → μ-law bytes at 8 kHz."""
        if not pcm16:
            return b""
        np = _numpy()
        resampled, self._ratecv_state = audioop.ratecv(
            pcm16, 2, 1, self.input_rate_hz, TELEPHONY_RATE_HZ, self._ratecv_state
        )
        samples = np.frombuffer(resampled, dtype=np.int16).astype(np.float64)
        if len(samples) > 0:
            samples, self._zi = self._lfilter(self._b, self._a, samples, zi=self._zi)
        clipped = np.clip(samples * HEADROOM, -32768, 32767).astype(np.int16)
        return g711.encode(clipped.tobytes())


def ulaw_to_pcm16(ulaw: bytes) -> bytes:
    """G.711 μ-law → PCM16-LE at the same rate. For recording and listening."""
    return g711.decode(ulaw)
