"""Audio arithmetic for the perturbation layers.

Everything here is pure: PCM in, PCM out, no state, no clock. The stateful
parts — a bed that advances with the call, a channel whose loss is bursty —
live in the perturbations themselves.

Loudness is handled in **RMS**, not peak, because the run mixes to a
*target SNR* rather than a raw gain. ``mixed = signal + noise * volume`` with
a fixed volume and no normalisation on either side lets the effective SNR
drift with TTS loudness sentence to sentence — the number you set is not the
number you get, and nothing records the number you got.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from harbor_voice._deps import numpy as _numpy

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

INT16_MAX = 32767
INT16_MIN = -32768
_EPSILON = 1e-12


def to_array(pcm: bytes) -> np.ndarray:
    """Decode PCM16-LE into float64 samples in [-1, 1)."""
    np = _numpy()
    if len(pcm) % 2:
        raise ValueError(f"PCM16 buffer has odd length {len(pcm)}")
    return np.frombuffer(pcm, dtype="<i2").astype("float64") / -INT16_MIN


def to_pcm(samples: np.ndarray) -> bytes:
    """Encode float samples back to PCM16-LE, clipping at full scale.

    Returns:
        The encoded bytes. Clipping is silent here by design; the caller that
        cares reports it, via :func:`clipped_fraction`.
    """
    np = _numpy()
    scaled = np.clip(samples * -INT16_MIN, INT16_MIN, INT16_MAX)
    return scaled.astype("<i2").tobytes()


def clipped_fraction(samples: np.ndarray) -> float:
    """Fraction of samples that would clip when encoded.

    Reported into the plan's achieved values, because silent clipping is how a
    mix that looks fine on paper becomes distortion nobody accounted for.
    """
    np = _numpy()
    if samples.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(samples) >= 1.0) / samples.size)


def rms(samples: np.ndarray) -> float:
    """Root-mean-square level. Zero for an empty or silent buffer."""
    np = _numpy()
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def db(ratio: float) -> float:
    """Convert a power ratio to decibels, floored rather than infinite."""
    return 20.0 * math.log10(max(ratio, _EPSILON))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio of two aligned buffers, in dB.

    Returns ``+inf`` when there is no noise and ``-inf`` when there is no
    signal. Both are **infinities on purpose**: the ratio is undefined, and
    callers filter infinities out of their statistics. Letting the epsilon
    floor turn "no signal" into a finite -240 dB would put a believable-looking
    number into a published mean.
    """
    signal_rms, noise_rms = rms(signal), rms(noise)
    if noise_rms <= _EPSILON:
        return math.inf
    if signal_rms <= _EPSILON:
        return -math.inf
    return db(signal_rms / noise_rms)


def scale_to_snr(
    signal: np.ndarray, noise: np.ndarray, target_db: float
) -> tuple[np.ndarray, float]:
    """Scale ``noise`` so that mixing it with ``signal`` hits ``target_db``.

    Returns:
        The scaled noise and the gain applied. When the signal is silent there
        is no ratio to target, so the noise is passed through unscaled and the
        caller records an undefined achieved SNR rather than a fabricated one.
    """
    signal_rms, noise_rms = rms(signal), rms(noise)
    if signal_rms <= _EPSILON or noise_rms <= _EPSILON:
        return noise, 1.0
    wanted_noise_rms = signal_rms / (10.0 ** (target_db / 20.0))
    gain = wanted_noise_rms / noise_rms
    return noise * gain, gain


def mix_at_snr(
    signal: np.ndarray, noise: np.ndarray, target_db: float
) -> tuple[np.ndarray, float]:
    """Mix noise into signal at a target SNR.

    Returns:
        The mixture and the **achieved** SNR, measured after scaling rather
        than assumed from the request.
    """
    scaled, _ = scale_to_snr(signal, noise, target_db)
    return signal + scaled, snr_db(signal, scaled)


def tile_to(source: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    """Take ``length`` samples from ``source``, wrapping, starting at ``offset``.

    The offset is what makes a bed *continuous*: the caller advances
    it across the whole call, so the same three-second asset does not restart
    at sample zero every utterance.
    """
    np = _numpy()
    if source.size == 0:
        return np.zeros(length, dtype="float64")
    if length <= 0:
        return np.zeros(0, dtype="float64")
    indices = (np.arange(length) + offset) % source.size
    return source[indices]


def linear_to_mulaw(samples: np.ndarray) -> np.ndarray:
    """Simplified mu-law, for simulating the codec's sound: sign bit plus a
    7-bit compressed magnitude.

    **Not G.711 wire format** — real G.711 inverts the bits, so silence is
    ``0xFF``, while this encodes silence as ``0x00``. For audio that goes on
    a wire, use :mod:`harbor_voice.audio.g711`.

    Sign and magnitude are kept separate rather than affine-mapping [-1, 1]
    onto [0, 255]. The affine form puts zero at code 127.5, which is not an
    integer, so digital silence decodes to a constant DC offset — a bug that
    quietly corrupts every RMS and SNR measurement taken afterwards.
    """
    np = _numpy()
    mu = 255.0
    clipped = np.clip(samples, -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(clipped)) / np.log1p(mu)
    quantised = np.round(magnitude * 127.0).astype("uint8")
    sign_bit = (clipped < 0).astype("uint8") << 7
    return sign_bit | quantised


def mulaw_to_linear(codes: np.ndarray) -> np.ndarray:
    """G.711 mu-law decode. Exactly inverts :func:`linear_to_mulaw` at zero."""
    np = _numpy()
    mu = 255.0
    negative = (codes & 0x80) != 0
    quantised = (codes & 0x7F).astype("float64") / 127.0
    magnitude = np.expm1(quantised * np.log1p(mu)) / mu
    return np.where(negative, -magnitude, magnitude)


def resample(samples: np.ndarray, from_hz: int, to_hz: int) -> np.ndarray:
    """Linear-interpolation resample.

    Deliberately simple and deliberately documented as such: there is no
    anti-alias filter, so downsampling folds energy above the new Nyquist back
    into the band. For telephony simulation that is *part of the effect* —
    real 8 kHz links sound like this — but it is not a general-purpose
    resampler and must not be used as one.
    """
    np = _numpy()
    if from_hz == to_hz or samples.size == 0:
        return samples
    if from_hz <= 0 or to_hz <= 0:
        raise ValueError(f"sample rates must be positive: {from_hz} -> {to_hz}")
    n_out = max(1, int(round(samples.size * to_hz / from_hz)))
    source_positions = np.linspace(0, samples.size - 1, num=n_out)
    return np.interp(source_positions, np.arange(samples.size), samples)


def resample_clean(samples: np.ndarray, from_hz: int, to_hz: int) -> np.ndarray:
    """Anti-aliased resample, for **format adaptation** rather than effect.

    :func:`resample` deliberately aliases: a real 8 kHz telephone link folds
    energy back into the band and that folding *is* part of the channel being
    simulated. Using it at an adapter edge would be a different thing entirely
    — corrupting the audio a model hears and then attributing the damage to
    the model.

    Conversion happens at the adapter edge. This is the converter: a
    windowed-sinc low-pass at the lower of the two Nyquist limits, then linear
    interpolation. Downsampling filters first; upsampling interpolates first
    and filters after, since the images appear above the original Nyquist.
    """
    if from_hz == to_hz or samples.size == 0:
        return samples
    if from_hz <= 0 or to_hz <= 0:
        raise ValueError(f"sample rates must be positive: {from_hz} -> {to_hz}")

    if to_hz < from_hz:
        filtered = low_pass(samples, cutoff_hz=to_hz / 2.0, rate_hz=from_hz)
        return resample(filtered, from_hz, to_hz)
    stretched = resample(samples, from_hz, to_hz)
    return low_pass(stretched, cutoff_hz=from_hz / 2.0, rate_hz=to_hz)


def low_pass(
    samples: np.ndarray, *, cutoff_hz: float, rate_hz: int, taps: int = 63
) -> np.ndarray:
    """Windowed-sinc FIR low-pass, zero-phase by centring the kernel.

    Kept short and explicit rather than pulling in scipy: harbor ships no
    scipy, and 63 taps with a Hamming window is ample for a 24k->16k step
    while staying readable enough to audit.
    """
    np = _numpy()
    normalised = min(cutoff_hz / rate_hz, 0.5) * 0.9
    half = taps // 2
    n = np.arange(taps) - half
    kernel = np.sinc(2.0 * normalised * n) * np.hamming(taps)
    kernel = kernel / np.sum(kernel)
    padded = np.pad(samples, (half, half), mode="edge")
    return np.convolve(padded, kernel, mode="valid")
