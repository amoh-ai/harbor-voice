"""Format conversion at an adapter edge: PCM16, μ-law and A-law, any rate.

A transport declares the format it takes and the format it speaks
(``capabilities.audio_in`` / ``audio_out``). Whoever drives it converts on the
way in and back to the recording format on the way out, with an anti-aliased
resample (:func:`~harbor_voice.perturbation.dsp.resample_clean`): the agent
must hear the caller as spoken, not a corrupted copy.
"""

from __future__ import annotations

import audioop

from harbor_voice.audio import g711
from harbor_voice.audio.format import AudioFormat
from harbor_voice.perturbation import dsp

_SILENCE_BYTE = {"pcm16": b"\x00", "mulaw": g711.SILENCE, "alaw": b"\xd5"}


def _to_pcm16(data: bytes, encoding: str) -> bytes:
    if encoding == "mulaw":
        return g711.decode(data)
    if encoding == "alaw":
        return audioop.alaw2lin(data, 2)
    return data


def _from_pcm16(pcm: bytes, encoding: str) -> bytes:
    if encoding == "mulaw":
        return g711.encode(pcm)
    if encoding == "alaw":
        return audioop.lin2alaw(pcm, 2)
    return pcm


def convert(data: bytes, src: AudioFormat, dst: AudioFormat) -> bytes:
    """``data`` in ``src`` as ``dst``. Mono only."""
    if src == dst or not data:
        return data
    if src.channels != 1 or dst.channels != 1:
        raise ValueError(f"cannot convert {src} to {dst}: mono only")
    pcm = _to_pcm16(data, src.encoding)
    if src.sample_rate_hz != dst.sample_rate_hz:
        samples = dsp.resample_clean(
            dsp.to_array(pcm), src.sample_rate_hz, dst.sample_rate_hz
        )
        pcm = dsp.to_pcm(samples)
    return _from_pcm16(pcm, dst.encoding)


def silence(fmt: AudioFormat, seconds: float) -> bytes:
    """``seconds`` of digital silence in ``fmt``."""
    samples = int(round(seconds * fmt.sample_rate_hz)) * fmt.channels
    return _SILENCE_BYTE[fmt.encoding] * (samples * fmt.bytes_per_sample)


QUIET_PEAK = 100
"""Peak below which PCM16 audio is quiet (about -50 dBFS): the silence an
agent that streams continuously (WebRTC) sends between replies."""


def is_quiet(pcm16: bytes) -> bool:
    """Whether PCM16 audio carries no speech: nothing above :data:`QUIET_PEAK`."""
    if not pcm16:
        return True
    import numpy as np

    samples = np.frombuffer(pcm16[: len(pcm16) // 2 * 2], dtype="<i2")
    return int(np.abs(samples.astype(np.int32)).max()) < QUIET_PEAK
