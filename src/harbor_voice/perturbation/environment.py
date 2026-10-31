"""L3 — the acoustic environment.

Two things here that no implementation we surveyed gets right together:

The bed is continuous. EVA and tau-voice converged on
this independently — the room does not fall silent when the speaker stops.
Tiling a three-second asset from sample zero *per utterance* puts the same
clink at the same offset every turn, and silence that bypasses the effect
chain drops the room entirely. Here the bed's position advances with
:attr:`~harbor_voice.perturbation.base.ApplyContext.elapsed_s`, which counts
the whole call.

SNR is targeted and then measured. The requested dB is normalised in RMS
against the actual signal, and the *achieved* ratio is recorded per frame, so
a run reports what it got rather than what it asked for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import override

from harbor_voice._deps import numpy as _numpy
from harbor_voice.capabilities import Layer
from harbor_voice.perturbation import dsp
from harbor_voice.perturbation.base import ApplyContext
from harbor_voice.perturbation.base import Perturbation
from harbor_voice.perturbation.base import RunningMean
from harbor_voice.perturbation.base import SeedSet

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


class BackgroundBed(Perturbation):
    """Continuous ambient noise mixed at a target SNR, with optional drift.

    Args:
        source: The bed, as float samples in [-1, 1]. Looped indefinitely.
        target_snr_db: Desired signal-to-noise ratio.
        snr_drift_db: Peak deviation of a slow random walk around the target.
            A constant-SNR bed is a stationary signal that an agent's noise
            suppression adapts to once and then ignores; drift is what makes
            it hard, and tau-voice is the only prior art that has it.
        variation_speed: How fast the walk moves, in dB per second.
        continuous: Whether the bed plays through silence. Default true, and
            you should have a reason to turn it off.
        sample_rate_hz: Rate ``source`` was captured at, for position maths.
    """

    def __init__(
        self,
        *,
        source: np.ndarray,
        target_snr_db: float = 15.0,
        snr_drift_db: float = 0.0,
        variation_speed: float = 0.1,
        continuous: bool = True,
        sample_rate_hz: int = 24_000,
        name: str = "background_bed",
    ) -> None:
        super().__init__(name=name, layer=Layer.ENVIRONMENT)
        if snr_drift_db < 0:
            raise ValueError(f"snr_drift_db cannot be negative: {snr_drift_db}")
        if variation_speed < 0:
            raise ValueError(f"variation_speed cannot be negative: {variation_speed}")
        self._source = source
        self._target_snr_db = target_snr_db
        self._snr_drift_db = snr_drift_db
        self._variation_speed = variation_speed
        self._continuous = continuous
        self._sample_rate_hz = sample_rate_hz

        self._rng: Any = None
        self._drift = 0.0
        self._achieved_snr = RunningMean()
        self._clipping = RunningMean()

    @property
    def target_snr_db(self) -> float:
        return self._target_snr_db

    @override
    def reset(self, seeds: SeedSet) -> None:
        self._rng = seeds.rng("channel")
        self._drift = 0.0
        self._achieved_snr = RunningMean()
        self._clipping = RunningMean()
        self._achieved.clear()

    @override
    def apply(self, pcm: bytes, ctx: ApplyContext) -> bytes:
        np = _numpy()
        signal = dsp.to_array(pcm)
        if signal.size == 0:
            return pcm
        if not self._continuous and not ctx.is_speech:
            return pcm

        offset = int(ctx.elapsed_s * self._sample_rate_hz) % max(self._source.size, 1)
        bed = dsp.tile_to(self._source, signal.size, offset=offset)

        target = self._target_snr_db + self._step_drift(ctx)
        mixed, achieved = dsp.mix_at_snr(signal, bed, target)

        if achieved not in (float("inf"), float("-inf")):
            self._achieved_snr.add(achieved)
        self._clipping.add(dsp.clipped_fraction(mixed))
        self._publish()
        return dsp.to_pcm(np.asarray(mixed))

    def _step_drift(self, ctx: ApplyContext) -> float:
        """A bounded random walk, so drift wanders rather than jumping."""
        if self._snr_drift_db <= 0 or self._rng is None:
            return 0.0
        frame_s = 0.02
        step = self._rng.normal(0.0, self._variation_speed * frame_s)
        self._drift = float(
            min(max(self._drift + step, -self._snr_drift_db), self._snr_drift_db)
        )
        return self._drift

    def _publish(self) -> None:
        self._achieved.clear()
        self._record(**self._achieved_snr.as_record("achieved_snr_db"))
        self._record(**self._clipping.as_record("clipped_fraction"))

    @override
    def params(self) -> dict[str, Any]:
        return {
            "target_snr_db": self._target_snr_db,
            "snr_drift_db": self._snr_drift_db,
            "variation_speed": self._variation_speed,
            "continuous": self._continuous,
            "source_samples": int(self._source.size),
            "sample_rate_hz": self._sample_rate_hz,
        }


def synthetic_bed(
    *,
    seconds: float = 3.0,
    sample_rate_hz: int = 24_000,
    seed: int = 0,
    kind: str = "pink",
) -> np.ndarray:
    """A deterministic stand-in bed, for tests and for the ``clean`` baseline.

    Real beds are assets with licences and digests. This is
    procedurally generated so the DSP is testable without shipping audio, and
    so a reproducible run needs no file.
    """
    np = _numpy()
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate_hz)
    white = rng.normal(0.0, 0.1, size=n)
    if kind == "white":
        return white
    if kind != "pink":
        raise ValueError(f"unknown bed kind {kind!r}; expected 'pink' or 'white'")
    # One-pole lowpass: a cheap 1/f-ish tilt, enough to be spectrally distinct
    # from white without pretending to be a real room.
    out = np.empty_like(white)
    state = 0.0
    for i, sample in enumerate(white):
        state = 0.98 * state + 0.02 * sample
        out[i] = state
    peak = float(np.max(np.abs(out))) or 1.0
    return out / peak * 0.1
