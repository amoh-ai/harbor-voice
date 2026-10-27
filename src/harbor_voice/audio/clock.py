"""Four clock domains, reported side by side and never reconciled silently.

Four domains — produced / on_wire / rendered / voiced — reported side by
side rather than picking one.

The rule this module exists to enforce:

    **A latency computed from two clock domains must name both.**

:class:`Interval` makes it hard to report a turn's *full response latency*
under a TTFB name: you cannot construct one without saying which clock each
end came from, and :attr:`Interval.label` derives the metric suffix from those
two names.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from enum import StrEnum


class Clock(StrEnum):
    """Where in a frame's life a timestamp was taken."""

    PRODUCED = "produced"
    """Content existed — TTS emitted it, or the model committed to it."""

    TX = "tx"
    """Frame handed to the transport."""

    RX = "rx"
    """Frame received from the transport."""

    VOICED = "voiced"
    """Frame became audible, after any jitter buffer or playback delay."""


class ClockError(ValueError):
    """A timing value is impossible, or two domains were mixed unnamed."""


class SessionClock:
    """Monotonic, session-relative time in seconds.

    ``perf_counter`` rather than wall time, because wall time can step
    backwards and a call is minutes long. The wall-clock start is kept
    separately so a trajectory can be placed in real time without using a
    steppable clock for any measurement.
    """

    __slots__ = ("_t0_monotonic", "_started_at")

    def __init__(self) -> None:
        self._t0_monotonic = time.perf_counter()
        self._started_at = datetime.now(UTC)

    def now(self) -> float:
        """Seconds since the session started."""
        return time.perf_counter() - self._t0_monotonic

    @property
    def started_at(self) -> datetime:
        """Wall-clock instant the session started, UTC. Never used for deltas."""
        return self._started_at

    def interval(
        self, start: float, end: float, start_clock: Clock, end_clock: Clock
    ) -> Interval:
        """Build an :class:`Interval` in this session's time base."""
        return Interval(
            start=start, end=end, start_clock=start_clock, end_clock=end_clock
        )


@dataclass(frozen=True, slots=True)
class Interval:
    """A duration whose endpoints each name their clock domain.

    Example:
        >>> iv = Interval(1.0, 1.25, Clock.PRODUCED, Clock.VOICED)
        >>> iv.label
        'produced_to_voiced'
        >>> round(iv.seconds, 3)
        0.25
        >>> iv.is_same_domain
        False
    """

    start: float
    end: float
    start_clock: Clock
    end_clock: Clock

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ClockError(
                f"interval ends before it starts: {self.start} -> {self.end} "
                f"({self.label})"
            )

    @property
    def seconds(self) -> float:
        """Elapsed seconds."""
        return self.end - self.start

    @property
    def label(self) -> str:
        """Metric-name suffix naming both domains, e.g. ``tx_to_rx``.

        Metric names are built from this rather than written by hand, so a
        metric cannot claim one clock pair and measure another.
        """
        return f"{self.start_clock.value}_to_{self.end_clock.value}"

    @property
    def is_same_domain(self) -> bool:
        """Whether both endpoints came from the same clock."""
        return self.start_clock is self.end_clock

    def named(self, base: str) -> str:
        """Full metric name: ``{base}_{label}``.

        >>> Interval(0, 1, Clock.TX, Clock.RX).named("transit")
        'transit_tx_to_rx'
        """
        return f"{base}_{self.label}"
