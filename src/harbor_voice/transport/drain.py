"""Draining an inbound stream to tail silence.

A turn does not end when the clock runs out; it ends when the other party
stops talking. Waiting out a fixed timeout is both slower and wrong — it
reports a turn length that is really the harness's patience.

Why it stopped matters as much as what it collected. A turn that ended on
:attr:`DrainStop.IDLE` is a complete reply; one that ended on
:attr:`DrainStop.TIMEOUT` is a reply we gave up on, and any duration derived
from it is a floor, not a measurement.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum

from harbor_voice.audio.clock import Clock
from harbor_voice.audio.frame import AudioFrame

DEFAULT_IDLE_S = 0.2
"""Quiet period after which a reply is considered finished."""

_POLL_S = 0.01


class DrainStop(StrEnum):
    """Why a drain stopped."""

    IDLE = "idle"
    """Tail silence — the reply finished. The only clean ending."""

    TIMEOUT = "timeout"
    """We gave up. Durations from this drain are lower bounds."""

    STREAM_END = "stream_end"
    """The transport closed the stream."""

    EMPTY = "empty"
    """Nothing arrived at all."""


@dataclass(frozen=True, slots=True)
class DrainResult:
    """Frames collected from one drain, and how it ended."""

    frames: tuple[AudioFrame, ...]
    stop: DrainStop

    @property
    def is_complete(self) -> bool:
        """Whether the reply finished on its own terms."""
        return self.stop in (DrainStop.IDLE, DrainStop.STREAM_END)

    @property
    def first_rx(self) -> float | None:
        """Receive time of the first frame — a real flow point, not an estimate."""
        for frame in self.frames:
            t = frame.at(Clock.RX)
            if t is not None:
                return t
        return None

    @property
    def last_rx(self) -> float | None:
        """Receive time of the last frame."""
        for frame in reversed(self.frames):
            t = frame.at(Clock.RX)
            if t is not None:
                return t
        return None

    @property
    def audio_duration_s(self) -> float:
        """Total audio carried, summed from the frames themselves."""
        return sum(f.duration_s for f in self.frames)


async def drain_until_idle(
    stream: AsyncIterator[AudioFrame],
    *,
    idle_s: float = DEFAULT_IDLE_S,
    timeout_s: float = 30.0,
    cancel_stream: bool = True,
) -> DrainResult:
    """Collect frames until tail silence, stream end, or timeout.

    The consumer runs as a task and is cancelled once, at the end, so no
    ``anext`` is interrupted mid-await. Only the cancellation we requested is
    suppressed — a genuine error out of the stream propagates, because a
    transport that breaks mid-turn is a finding, not a quiet short reply.

    Args:
        stream: Inbound frames, typically ``transport.rx()``.
        idle_s: Quiet period that ends the turn.
        timeout_s: Hard ceiling.
        cancel_stream: Cancel the consumer when done. Set false when the caller
            intends to keep draining the same stream across turns.

    Returns:
        The frames and why collection stopped.
    """
    if idle_s <= 0:
        raise ValueError(f"idle_s must be positive, got {idle_s}")
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")

    received: list[AudioFrame] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    effective_idle = min(idle_s, timeout_s)

    async def _consume() -> None:
        async for frame in stream:
            received.append(frame)

    task = asyncio.create_task(_consume())
    stop = DrainStop.TIMEOUT
    try:
        seen = 0
        quiet_since = loop.time()
        while True:
            if task.done():
                stop = DrainStop.STREAM_END
                break
            now = loop.time()
            if now >= deadline:
                stop = DrainStop.TIMEOUT
                break
            if len(received) != seen:
                seen, quiet_since = len(received), now
            elif received and now - quiet_since >= effective_idle:
                stop = DrainStop.IDLE
                break
            await asyncio.sleep(_POLL_S)
    finally:
        if cancel_stream:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        elif task.done():
            await task

    if not received:
        stop = DrainStop.EMPTY
    return DrainResult(frames=tuple(received), stop=stop)
