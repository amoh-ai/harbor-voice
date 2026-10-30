"""L4 jitter — the perturbation that frames unlock.

A transport that sends whole utterances has no inter-packet interval to
perturb; frames do.

That is the whole argument, and it is worth being precise about why. Jitter is
**packet delay variation**: frames leave evenly spaced and arrive unevenly.
A harness whose unit on the wire is an utterance has exactly one packet per
turn, and the variation of a single arrival is not a quantity. There is
nothing to perturb. With 20 ms frames there are fifty packets a second and the
spacing between them is a real, measurable property of the line.

### Not a sample transform

Every other L4 layer rewrites a frame's audio. Jitter does not touch a sample;
it changes **when each frame arrives and in what order**. So it cannot be a
:class:`~harbor_voice.perturbation.base.Perturbation`, whose contract is
``apply(pcm, ctx) -> bytes`` on one frame in isolation — jitter is not even
*definable* on one frame, because variation needs a sequence.

It is therefore a :class:`FramePerturbation`, applied by
:meth:`~harbor_voice.perturbation.plan.PerturbationPipeline.process_all` after
the per-frame chain. The ordering is physical: a codec runs in the handset,
loss and delay happen on the network, so the packets that jitter reorders are
already encoded ones.

### What a jitter buffer does, and why we model its failures

A receiver holds incoming frames briefly so it can play them in order despite
uneven arrival. That buffer is the thing jitter actually attacks, and it fails
in two distinct ways:

``late``
    A frame arrives after its playout deadline. The buffer has already moved
    on, so the frame is **discarded** — it is loss, caused by delay rather
    than by a dropped packet, and it sounds identical to the listener.
``reordered``
    A frame arrives after a later one but still inside the window. The buffer
    puts it back in sequence and nobody hears anything wrong. Counted anyway,
    because a line that reorders is a line that is about to start dropping.

Both are recorded. A run that asked for 30 ms of jitter and discarded a tenth
of the caller's audio should say so, rather than reporting a WER that looks
like a bad recogniser.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import Any
from typing import override

from harbor_voice.audio.clock import Clock
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import LayerKey
from harbor_voice.perturbation.base import RunningMean
from harbor_voice.perturbation.base import SeedSet

_CHANNEL_STREAM = "channel"


class JitterError(ValueError):
    """A jitter layer was configured in a way that cannot produce a measurement."""


class FramePerturbation(ABC):
    """An L4 layer that transforms the frame **stream** rather than a frame.

    Applied by ``process_all`` after every per-frame perturbation, because
    the packets a network delays are already-encoded ones. ``process`` on a
    single frame deliberately cannot apply these: delay *variation* is not
    defined on one arrival, and silently doing nothing would be worse than
    not offering it.
    """

    def __init__(self, *, name: str) -> None:
        self._name = name
        self._rng: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def layer(self) -> Layer:
        return Layer.CHANNEL

    @property
    def key(self) -> LayerKey:
        return LayerKey(layer=Layer.CHANNEL, name=self._name)

    @abstractmethod
    def apply(self, frames: Sequence[AudioFrame]) -> tuple[AudioFrame, ...]:
        """Transform a whole stream."""

    def reset(self, seeds: SeedSet) -> None:
        self._rng = seeds.rng(_CHANNEL_STREAM)

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """The requested settings, as written into the plan."""

    @abstractmethod
    def achieved(self) -> dict[str, float]:
        """What was actually delivered."""

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.key})"


class Distribution(StrEnum):
    """How delay varies around the mean.

    Both are real network shapes and they break a buffer differently.
    """

    GAUSSIAN = "gaussian"
    """Symmetric wobble. A buffer sized at 2 sigma copes."""

    PARETO = "pareto"
    """A heavy tail: mostly fine, occasionally terrible.

    What congested links actually do, and what a buffer sized on the mean is
    least ready for. tau and EVA both model loss; neither models this.
    """


class Jitter(FramePerturbation):
    """Vary each frame's arrival, then play out through a finite buffer.

    Args:
        mean_ms: Average added delay. A constant delay is latency, not
            jitter, and it is the *variation* a buffer has to absorb.
        std_ms: Spread of the variation.
        distribution: ``gaussian`` or ``pareto``.
        buffer_ms: How much the receiver holds before playing out. A frame
            later than this misses its slot and is discarded. Set it to zero
            and every out-of-order frame is lost, which is the honest model
            of a receiver that does not buffer at all.
        keep_late: Whether a late frame is kept anyway. ``False`` is the real
            behaviour and the default; ``True`` exists so an experiment can
            separate *the audio was mangled* from *the audio was late*.
    """

    def __init__(
        self,
        *,
        mean_ms: float = 30.0,
        std_ms: float = 12.0,
        distribution: Distribution = Distribution.GAUSSIAN,
        buffer_ms: float = 60.0,
        keep_late: bool = False,
    ) -> None:
        super().__init__(name="jitter")
        if mean_ms < 0:
            raise JitterError(f"mean_ms cannot be negative: {mean_ms}")
        if std_ms < 0:
            raise JitterError(f"std_ms cannot be negative: {std_ms}")
        if buffer_ms < 0:
            raise JitterError(f"buffer_ms cannot be negative: {buffer_ms}")
        self._mean_ms = mean_ms
        self._std_ms = std_ms
        self._distribution = distribution
        self._buffer_ms = buffer_ms
        self._keep_late = keep_late

        self._delay = RunningMean()
        self._seen = 0
        self._late = 0
        self._reordered = 0

    # -------------------------------------------------------------- running

    @override
    def apply(self, frames: Sequence[AudioFrame]) -> tuple[AudioFrame, ...]:
        if not frames:
            return ()

        scheduled: list[tuple[float, int, AudioFrame]] = []
        for index, frame in enumerate(frames):
            nominal = index * frame.duration_s
            delay_s = self._draw() / 1000.0
            self._delay.add(delay_s * 1000.0)
            scheduled.append((nominal + delay_s, index, frame))
        self._seen += len(frames)

        # Sort by arrival: this *is* the reordering. Ties break on the
        # original index so two frames that arrive together stay in the order
        # they were sent, which is what a sequence number would do.
        scheduled.sort(key=lambda item: (item[0], item[1]))
        self._reordered += sum(
            1 for position, (_, index, _f) in enumerate(scheduled) if position != index
        )

        deadline_s = self._buffer_ms / 1000.0
        out: list[AudioFrame] = []
        for arrival, index, frame in scheduled:
            nominal = index * frame.duration_s
            if arrival - nominal > deadline_s:
                self._late += 1
                if not self._keep_late:
                    # Delay loss. The buffer played silence in this slot and
                    # the listener heard a hole, exactly as for a dropped
                    # packet -- so it is dropped, not silently kept.
                    continue
            out.append(frame.stamp(Clock.VOICED, arrival))
        return tuple(out)

    def _draw(self) -> float:
        """One delay sample, in milliseconds, never negative.

        A negative delay would be a frame arriving before it was sent. The
        gaussian is clamped at zero rather than resampled, because resampling
        would quietly raise the mean above the one that was requested.
        """
        if self._std_ms == 0.0:
            return self._mean_ms
        if self._distribution is Distribution.PARETO:
            # shape 2 has a finite mean; scaled so the mean lands on mean_ms.
            draw = float(self._rng.pareto(2.0)) * self._mean_ms
            return draw
        return max(0.0, float(self._rng.normal(self._mean_ms, self._std_ms)))

    @override
    def reset(self, seeds: SeedSet) -> None:
        super().reset(seeds)
        self._delay = RunningMean()
        self._seen = 0
        self._late = 0
        self._reordered = 0

    # --------------------------------------------------------------- record

    @override
    def params(self) -> dict[str, Any]:
        return {
            "mean_ms": self._mean_ms,
            "std_ms": self._std_ms,
            "distribution": self._distribution.value,
            "buffer_ms": self._buffer_ms,
            "keep_late": self._keep_late,
        }

    @override
    def achieved(self) -> dict[str, float]:
        if not self._seen:
            return {}
        return {
            "frames": float(self._seen),
            "late_fraction": self._late / self._seen,
            "reordered_fraction": self._reordered / self._seen,
            **self._delay.as_record("delay_ms"),
        }
