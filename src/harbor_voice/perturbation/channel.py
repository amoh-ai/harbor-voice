"""L4 — the channel.

Applied last, to the finished mix, because that is the order the physics runs
in: noise enters the room, then the room goes down the wire. Applying the
codec before the room noise produces audio no telephone ever carried.

Two things here that matter more than they look:

**Loss is bursty, not Bernoulli.** EVA drops frames independently at 3%;
tau-voice models burst duration with Gilbert-Elliott. These measure different
things — 3% i.i.d. loss is an inaudible hiss, 3% in 200 ms bursts is a dropped
word. Any loss axis that is not burst
parameterised measures the wrong thing.

**The codec is a real codec.** mu-law at 8 kHz, with the band limiting that
implies, rather than a label on an unchanged buffer.
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
from harbor_voice.perturbation.base import SeedSet

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

TELEPHONY_HZ = 8_000


class MuLawCodec(Perturbation):
    """G.711 mu-law at a reduced rate, then back. The telephony channel.

    The round trip is the effect: downsample (no anti-alias filter, so the
    band folds exactly as a real link does), quantise to 8-bit mu-law, decode,
    and resample back to the canonical rate so everything downstream still
    sees one format.
    """

    def __init__(
        self,
        *,
        rate_hz: int = TELEPHONY_HZ,
        source_rate_hz: int = 24_000,
        name: str = "codec",
    ) -> None:
        super().__init__(name=name, layer=Layer.CHANNEL)
        if rate_hz <= 0 or source_rate_hz <= 0:
            raise ValueError("sample rates must be positive")
        self._rate_hz = rate_hz
        self._source_rate_hz = source_rate_hz

    @override
    def reset(self, seeds: SeedSet) -> None:
        self._achieved.clear()
        self._record(rate_hz=float(self._rate_hz))

    @override
    def apply(self, pcm: bytes, ctx: ApplyContext) -> bytes:
        np = _numpy()
        samples = dsp.to_array(pcm)
        if samples.size == 0:
            return pcm
        narrow = dsp.resample(samples, self._source_rate_hz, self._rate_hz)
        decoded = dsp.mulaw_to_linear(dsp.linear_to_mulaw(narrow))
        restored = dsp.resample(decoded, self._rate_hz, self._source_rate_hz)
        # Resampling can land a sample short or long; the frame contract is
        # fixed-size, so pad or trim rather than emit a ragged frame.
        if restored.size != samples.size:
            restored = np.resize(restored, samples.size)
        return dsp.to_pcm(np.asarray(restored))

    @override
    def params(self) -> dict[str, Any]:
        return {
            "codec": "mulaw",
            "rate_hz": self._rate_hz,
            "source_rate_hz": self._source_rate_hz,
        }


class GilbertElliottLoss(Perturbation):
    """Bursty frame loss — a two-state Markov chain over GOOD and BAD.

    Parameterised the way an operator thinks about it: an overall loss rate
    and a mean burst length. Those map to the chain's transition
    probabilities, so ``rate=0.02, burst_duration_ms=200`` is "2% of audio is
    gone, in chunks about a fifth of a second long" rather than two opaque
    probabilities.

    A lost frame becomes silence of the same length. Dropping it outright
    would shorten the timeline and corrupt every duration downstream.
    """

    def __init__(
        self,
        *,
        rate: float = 0.02,
        burst_duration_ms: float = 200.0,
        frame_ms: float = 20.0,
        name: str = "frame_loss",
    ) -> None:
        super().__init__(name=name, layer=Layer.CHANNEL)
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"rate must be in [0, 1), got {rate}")
        if burst_duration_ms <= 0:
            raise ValueError(
                f"burst_duration_ms must be positive, got {burst_duration_ms}"
            )
        self._rate = rate
        self._burst_duration_ms = burst_duration_ms
        self._frame_ms = frame_ms

        self._rng: Any = None
        self._in_burst = False
        self._frames = 0
        self._lost = 0
        self._bursts = 0

    @property
    def p_recover(self) -> float:
        """P(leave BAD each frame) — the reciprocal of the burst length."""
        frames_per_burst = max(1.0, self._burst_duration_ms / self._frame_ms)
        return 1.0 / frames_per_burst

    @property
    def p_fail(self) -> float:
        """P(enter BAD each frame), solved so the stationary loss equals ``rate``."""
        if self._rate <= 0:
            return 0.0
        return (self._rate * self.p_recover) / (1.0 - self._rate)

    @override
    def reset(self, seeds: SeedSet) -> None:
        self._rng = seeds.rng("channel")
        self._in_burst = False
        self._frames = self._lost = self._bursts = 0
        self._achieved.clear()

    @override
    def apply(self, pcm: bytes, ctx: ApplyContext) -> bytes:
        if self._rng is None or self._rate <= 0:
            return pcm
        self._frames += 1
        draw = float(self._rng.random())

        if self._in_burst:
            if draw < self.p_recover:
                self._in_burst = False
        elif draw < self.p_fail:
            self._in_burst = True
            self._bursts += 1

        self._publish()
        if not self._in_burst:
            return pcm
        self._lost += 1
        self._publish()
        return bytes(len(pcm))

    def _publish(self) -> None:
        if not self._frames:
            return
        self._record(
            achieved_loss_rate=self._lost / self._frames,
            frames_total=float(self._frames),
            frames_lost=float(self._lost),
            bursts=float(self._bursts),
            mean_burst_frames=(self._lost / self._bursts) if self._bursts else 0.0,
        )

    @override
    def params(self) -> dict[str, Any]:
        return {
            "model": "gilbert_elliott",
            "rate": self._rate,
            "burst_duration_ms": self._burst_duration_ms,
            "frame_ms": self._frame_ms,
            "p_fail": self.p_fail,
            "p_recover": self.p_recover,
        }
