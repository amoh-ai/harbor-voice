"""One long-lived consumer of an inbound frame stream.

The transport is *always* draining, which has a consequence that
bites immediately: ``rx()`` is one async generator for the whole session, and
only one consumer may hold it. Draining per turn leaves a suspended ``anext``
on a shared generator, and the next turn fails with "asynchronous generator is
already running" — which is exactly what it did before this existed.

So the pump owns ``rx()`` for the session and everything else reads the inbox.
Two consumers need that — :class:`~harbor_voice.adapter.VoiceAgentAdapter` for
turn-shaped calls and :class:`~harbor_voice.duplex.DuplexDriver` for a genuinely
concurrent one — and a second copy of this logic is a second place for the
generator rule to be got wrong.

A consequence worth having: frames arriving between turns simply wait in the
inbox, which is what makes :attr:`~harbor_voice.adapter.Turn.bled` real rather
than aspirational.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from harbor_voice.audio.frame import AudioFrame
from harbor_voice.transport.base import VoiceTransport


class PumpError(RuntimeError):
    """The inbound stream failed, or was used before it was started."""


class InboundPump:
    """Drains a transport's ``rx()`` into a queue for the whole session."""

    def __init__(self, transport: VoiceTransport) -> None:
        self._transport = transport
        self._inbox: asyncio.Queue[AudioFrame] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None

    @property
    def error(self) -> BaseException | None:
        """The exception that ended the stream, if one did."""
        return self._error

    @property
    def pending(self) -> int:
        """Frames waiting in the inbox right now."""
        return self._inbox.qsize()

    def start(self) -> None:
        """Begin draining. Clears any inbox left from a previous session."""
        self.stop_sync()
        self._inbox = asyncio.Queue()
        self._error = None
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            async for frame in self._transport.rx():
                await self._inbox.put(frame)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - surfaced via `check()`
            self._error = exc

    async def stop(self) -> None:
        """Cancel the consumer and await it. Idempotent."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def stop_sync(self) -> None:
        """Cancel without awaiting, for use from a non-async setup path."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def check(self) -> None:
        """Raise if the pump is not running or the stream has failed.

        Raises:
            PumpError: Named after the original, with it as the cause, because
                "stream failed" without the vendor's exception is a bug report
                nobody can act on.
        """
        if self._task is None:
            raise PumpError("the inbound pump is not running; call start() first")
        if self._error is not None:
            raise PumpError(
                f"inbound stream failed: {type(self._error).__name__}: {self._error}"
            ) from self._error

    def stream(self) -> AsyncIterator[AudioFrame]:
        """A fresh view of the inbox.

        Cancelling a pending ``Queue.get()`` does not consume an item, so
        ending a turn mid-wait cannot lose a frame.
        """

        async def _iterate() -> AsyncIterator[AudioFrame]:
            while True:
                yield await self._inbox.get()

        return _iterate()

    def take_pending(self) -> tuple[AudioFrame, ...]:
        """Everything already queued: audio that crossed a turn boundary."""
        pending: list[AudioFrame] = []
        while True:
            try:
                pending.append(self._inbox.get_nowait())
            except asyncio.QueueEmpty:
                return tuple(pending)
