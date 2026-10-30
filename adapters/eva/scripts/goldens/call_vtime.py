"""A virtual clock for the call golden: both sides of the golden on the same, noise-free time.

On the wall clock EVA does not reproduce itself. Its bridge sends silence to
the agent only on 250 ms ticks of its input stream, so when the agent's VAD
can close a turn depends on where a tick falls; and its pacer never catches
up, so its speed depends on how late each sleep wakes. Two runs of EVA's own
code on the same script differ by up to 250 ms in absolute time, and so do
EVA and harbor. Under this clock nothing takes time but sleeping, so two runs
of the same logic give the same timestamps, and any difference between EVA
and harbor is a difference in what the code does.

How it works:

- ``time.time``, ``time.monotonic`` and ``time.perf_counter`` return their
  values at start plus the virtual seconds elapsed; the event loop's
  ``time()`` is virtual too;
- when the loop would sleep, it first polls real I/O without blocking (EVA's
  bridge and server talk over a real local WebSocket, whose bytes arrive as
  they are sent) and, if none is ready, jumps straight to the next timer;
- ``run_in_executor`` (EVA runs the SDK's ``start_session`` and the WAV
  writes there) runs the call inline and resolves after the delay declared
  for it, as a thread that took that long would.

Used as ``with VirtualClock() as clock: clock.run(main())``.
"""

from __future__ import annotations

import asyncio
import selectors
import time
from collections.abc import Callable
from collections.abc import Coroutine
from typing import Any

EXECUTOR_DELAYS: dict[str, float] = {}
"""Virtual seconds a function run in an executor takes, by ``__name__``."""


class VirtualClock:
    def __init__(self, epoch0: float | None = None) -> None:
        """``epoch0``: what ``time.time()`` reads at the start; now by default.

        Log stamps are epoch milliseconds truncated, so two events a fraction
        of a millisecond apart share a stamp or not depending on the epoch's
        sub-millisecond phase; EVA's processor breaks such ties by log. Start
        at the epoch of the run being compared with and the ties are its ties.
        """
        self.elapsed = 0.0
        self._epoch0 = time.time() if epoch0 is None else epoch0
        self._mono0 = time.monotonic()
        self._perf0 = time.perf_counter()
        self._saved: dict[str, Callable[[], float]] = {}

    def __enter__(self) -> VirtualClock:
        self._saved = {
            "time": time.time,
            "monotonic": time.monotonic,
            "perf_counter": time.perf_counter,
        }
        time.time = lambda: self._epoch0 + self.elapsed
        time.monotonic = lambda: self._mono0 + self.elapsed
        time.perf_counter = lambda: self._perf0 + self.elapsed
        return self

    def __exit__(self, *exc: object) -> None:
        for name, fn in self._saved.items():
            setattr(time, name, fn)

    def loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
        loop.time = lambda: self._mono0 + self.elapsed  # type: ignore[method-assign]
        selector = loop._selector  # type: ignore[attr-defined]
        real_select = selector.select

        def select(timeout: float | None = None) -> list[Any]:
            events = real_select(0)
            if events or timeout == 0:
                return events
            if timeout is None:
                return real_select(None)  # nothing scheduled: wait for real I/O
            self.elapsed += timeout
            return []

        selector.select = select

        def run_in_executor(
            executor: Any, func: Callable[..., Any], *args: Any
        ) -> asyncio.Future[Any]:
            future: asyncio.Future[Any] = loop.create_future()
            try:
                result = func(*args)
            except BaseException as exc:  # noqa: BLE001 - delivered through the future
                loop.call_soon(future.set_exception, exc)
                return future
            delay = EXECUTOR_DELAYS.get(getattr(func, "__name__", ""), 0.0)
            loop.call_later(delay, lambda: future.done() or future.set_result(result))
            return future

        loop.run_in_executor = run_in_executor  # type: ignore[method-assign]
        return loop

    def run(self, main: Coroutine[Any, Any, Any]) -> Any:
        with asyncio.Runner(loop_factory=self.loop) as runner:
            return runner.run(main)
