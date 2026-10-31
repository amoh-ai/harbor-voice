"""One EVA call on the wall clock: the caller, EVA's bridge, EVA's server, the agent.

:class:`RealtimeCall` runs what EVA's worker runs for one conversation
(``eva/orchestrator/worker.py:134-312`` @039efa0a), with harbor's transports
where EVA has its own clients:

::

    caller transport ──16 kHz PCM──► Bridge ──8 kHz μ-law, 20 ms──► Recorder ──24 kHz PCM──► agent
    (elevenlabs.agents) ◄─μ-law, 250 ms─ (bridge.py) ◄──μ-law, paced 20 ms── (recorder.py) ◄─24 kHz─ (openai.realtime)

Real time, one clock. Nothing is stepped: audio moves when it arrives,
frames are paced on the event loop's clock, silence is streamed while the
caller is quiet, and every log line is stamped with ``time.time()`` when it
happens, because EVA merges its three logs by epoch milliseconds. On Windows
the call raises the system timer resolution to 1 ms for its duration
(:func:`fine_timer`); without it asyncio sleeps in 15.6 ms steps and 20 ms
pacing runs at two-thirds speed.

Who does what. The caller object (``eva_adapter.caller.ElevenLabsCaller``)
owns EVA's simulator session: ``connected`` / ``session_started``, the
keep-alive, its own time limit, the post-hoc ``end_call`` check,
``session_ended``, the latencies file, the clean caller track and the 4 s
grace. This class owns the audio and the order of things:

1. the bridge's loops start (EVA's ``start_async``), and the agent is opened
   beside them, as EVA's assistant server connects to OpenAI when the
   bridge connects (``openai_realtime_server.py:239-282``);
2. :meth:`Caller.start` (connected, signed URL, ``session_started``, socket,
   then ``bridge.start(transport.send_audio)``, the SDK's
   ``audio_interface.start``: from then on the caller hears the assistant);
3. caller events are pumped: audio to :meth:`Bridge.output`, everything else
   to :meth:`Caller.on_event`, until :attr:`Caller.should_end`;
4. :meth:`Caller.finish`, then the bridge's ``stop_async`` (the assistant
   session ends on its ``stop``), then the event log is saved
   (``elevenlabs.py:203-265``, ``:111-113``);
5. the agent is closed and the recorder writes the audit log and WAVs
   (``worker.py:415-424``, ``base_server.py:138-177``).

The whole of 1–4 runs under EVA's worker guard (620 s), which ends the call
as ``time_limit_exceeded`` (``worker.py:166-176``). The bridge reports its
end-of-session and disconnect to :meth:`Caller.end`, as EVA's bridge calls the
simulator's ``_on_conversation_end``, and its clean caller audio to
:meth:`Caller.record_audio`.

The agent transport must be harbor's ``openai.realtime`` (or anything with
the same events) opened with EVA's settings: ``audio_format="pcm24k"``,
``opening_text="Say: '<greeting>'"``, ``tool_flow="serial"``,
``tool_output_ascii=False`` and ``extended_events=True`` (the recorder
writes EVA's tool entries from its ``ToolCall`` events).
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import logging
import sys
from collections.abc import AsyncIterator
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Protocol

from harbor_voice.audio.format import CANONICAL
from harbor_voice.ids import PartyId
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.stream import OutputAudio

from eva_adapter.bridge import Bridge
from eva_adapter.client import CallInfo
from eva_adapter.bridge import Perturbator
from eva_adapter.events import EVENTS_FILE
from eva_adapter.events import EventLog
from eva_adapter.recorder import Recorder

logger = logging.getLogger(__name__)

GUARD_TIMEOUT_S = 620.0
"""EVA's worker guard: ``conversation_time_limit_seconds`` (600) plus
``USER_SIMULATOR_SHUTDOWN_GRACE_SECONDS`` (20) (``worker.py:23``, ``:398-400``)."""

TIME_LIMIT_EXCEEDED = "time_limit_exceeded"
"""The end reason when the guard fires (``worker.py:175``)."""

ERROR = "error"
CANCELLED = "cancelled"

AGENT_SHUTDOWN_S = 5.0
"""How long the agent session gets to end after the bridge's ``stop``
(EVA waits as long for its server to exit, ``openai_realtime_server.py:159-171``)."""

SIMULATOR_PROVIDER = "elevenlabs"
"""The ``provider`` on every simulator event (``elevenlabs.py:78``)."""


class CallerTransport(Protocol):
    """The caller's side, as ``elevenlabs.agents`` provides it."""

    def events(self) -> AsyncIterator[Any]: ...

    async def close(self) -> None: ...


class Caller(Protocol):
    """EVA's simulator session, as ``eva_adapter.caller.ElevenLabsCaller`` provides it.

    It must log into the same :class:`EventLog` as the call
    (:attr:`RealtimeCall.events`) and hold the call's bridge
    (:attr:`RealtimeCall.bridge`) for its end-of-utterance cue, keep-alive
    and latencies.
    """

    @property
    def should_end(self) -> bool: ...

    @property
    def end_reason(self) -> str: ...

    async def start(self) -> None: ...

    def on_event(self, event: Any) -> None: ...

    def end(self, reason: str) -> None: ...

    def record_audio(self, source: str, audio: bytes) -> None: ...

    async def wait(self) -> None: ...

    async def finish(self) -> str: ...


@dataclass(frozen=True, slots=True)
class CallOutcome:
    """How the call ended, for ``result.json`` (``models/results.py``)."""

    conversation_ended_reason: str
    started_at: datetime
    ended_at: datetime
    out_dir: Path
    error: str | None = None
    agent_error: str | None = None
    """Why the agent session failed, if it did (EVA only logs this)."""
    stats: dict[str, Any] = field(default_factory=dict)
    """The audit log's ``get_stats()``: ``num_turns``, ``num_tool_calls``,
    ``tools_called``, as the worker collects them (``worker.py:180-183``)."""

    @property
    def duration_seconds(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()

    def call_info(self, attempt: int) -> CallInfo:
        """``call.json`` for the sidecar's ``/record``."""
        return CallInfo(
            attempt=attempt,
            conversation_ended_reason=self.conversation_ended_reason,
            started_at=self.started_at,
            ended_at=self.ended_at,
            error=self.error,
        )


@contextlib.contextmanager
def fine_timer() -> Iterator[None]:
    """1 ms system timer resolution on Windows, for real-time pacing; elsewhere nothing.

    Windows wakes timed waits on its 15.6 ms system tick unless a process
    asks for finer, so ``asyncio.sleep(0.02)`` takes ~31 ms there. Measured:
    with ``timeBeginPeriod(1)``, 20.6 ms on average.
    """
    if sys.platform != "win32":
        yield
        return
    win_dll = getattr(ctypes, "WinDLL")
    _keep_timer_resolution(win_dll("kernel32"))
    winmm = win_dll("winmm")
    winmm.timeBeginPeriod(1)
    try:
        yield
    finally:
        winmm.timeEndPeriod(1)


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [  # noqa: RUF012 - ctypes' declaration form
        ("Version", ctypes.c_ulong),
        ("ControlMask", ctypes.c_ulong),
        ("StateMask", ctypes.c_ulong),
    ]


def _keep_timer_resolution(kernel32: Any) -> None:
    """Opt out of Windows 11 ignoring ``timeBeginPeriod`` for background processes.

    Without it, a process Windows deems not visible is quietly put back on
    the 15.6 ms tick mid-call (measured: agent audio paced at 32 ms a chunk).
    ``SetProcessInformation(ProcessPowerThrottling)`` with
    ``PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION`` in the control mask
    and not in the state mask keeps the requested resolution.
    """
    process_power_throttling = 4
    ignore_timer_resolution = 0x4
    state = _PowerThrottlingState(1, ignore_timer_resolution, 0)
    with contextlib.suppress(Exception):
        kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(),
            process_power_throttling,
            ctypes.byref(state),
            ctypes.sizeof(state),
        )


class RealtimeCall:
    """One EVA call between a caller and an agent under test.

    Build it, build the caller with :attr:`events` and :attr:`bridge`, then
    :meth:`run`. Every record file but ``result.json`` and the scenario
    databases lands in ``out_dir``.

    Args:
        agent: The agent under test, opened here unless already open.
        caller_transport: The caller's transport; :meth:`Caller.start` opens it.
        out_dir: The record directory.
        conversation_id: EVA's ``conversation_id`` (the record id).
        model: The agent's model id, for ``pipecat_metrics.jsonl``.
        recorder: A recorder to use instead of a new one.
        guard_timeout_s: EVA's worker guard: the sidecar's ``caller_timeout_s``
            (``/open``), the time limit plus 20 s. This class owns the guard,
            so that the record is still written when it fires; don't wrap
            :meth:`run` in the same timeout.
        perturbator: EVA's ``AudioPerturbator`` for this conversation
            (:func:`eva_adapter.conditions.perturbator`), or none: the clean
            call. The bridge applies it as EVA's does.
        agent_session: How to open the agent, if it is not open.
    """

    def __init__(
        self,
        *,
        agent: VoiceTransport,
        caller_transport: CallerTransport,
        out_dir: Path,
        conversation_id: str,
        model: str = "gpt-realtime-1.5",
        recorder: Recorder | None = None,
        guard_timeout_s: float = GUARD_TIMEOUT_S,
        perturbator: Perturbator | None = None,
        agent_session: SessionConfig | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self._agent = agent
        self._caller_transport = caller_transport
        self._guard_timeout_s = guard_timeout_s
        self._agent_session = agent_session or SessionConfig(
            party=PartyId("agent"), audio_in=CANONICAL, audio_out=CANONICAL
        )
        self.events = EventLog(self.out_dir / EVENTS_FILE, provider=SIMULATOR_PROVIDER)
        self.recorder = recorder or Recorder(self.out_dir, model=model)
        self.bridge = Bridge(
            self.recorder,
            conversation_id,
            event_logger=self.events,
            perturbator=perturbator,
        )
        self._agent_task: asyncio.Task[None] | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self.agent_error: BaseException | None = None

    # ---------------------------------------------------------------- run

    async def run(self, caller: Caller) -> CallOutcome:
        """The worker's ``run`` (``worker.py:134-312``), up to ``result.json``."""
        started_at = datetime.now()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bridge.record_callback = caller.record_audio
        self.bridge.conversation_done_callback = caller.end
        error: str | None = None
        reason = ERROR
        stats: dict[str, Any] = {}
        with fine_timer():
            try:
                reason = await asyncio.wait_for(
                    self._simulate(caller), timeout=self._guard_timeout_s
                )
            except TimeoutError:
                reason = TIME_LIMIT_EXCEEDED
                logger.warning("Conversation exceeded the time limit")
            except asyncio.CancelledError:
                reason = CANCELLED
                raise
            except Exception as exc:  # noqa: BLE001 - the worker's error path
                error = str(exc)
                reason = ERROR
                logger.error("Conversation error: %s", exc, exc_info=True)
            finally:
                stats = self.recorder.get_conversation_stats()
                await self._cleanup()
        return CallOutcome(
            conversation_ended_reason=reason,
            started_at=started_at,
            ended_at=datetime.now(),
            out_dir=self.out_dir,
            error=error,
            agent_error=None if self.agent_error is None else str(self.agent_error),
            stats=stats,
        )

    async def _simulate(self, caller: Caller) -> str:
        """The simulator's ``run_conversation`` (``elevenlabs.py:87-265``) around the caller.

        The caller's ``finish`` runs on every path, as EVA's teardown does.
        If the guard fires during the call rather than during ``finish``,
        EVA skips the session part (``end_session``, the ``end_call`` check,
        ``session_ended``) and runs only the teardown; the caller has no
        entry point for that alone, so ``finish`` runs whole there. In EVA
        that path needs ElevenLabs to outlive its own 600 s maximum.
        """
        self.recorder.start()
        await self.bridge.start_async()
        self._agent_task = asyncio.create_task(self._serve_agent())
        started = finishing = False
        try:
            # Raises only before EVA would have logged anything (no API key);
            # EVA's teardown is not reached then either.
            await caller.start()  # ends with bridge.start(transport.send_audio)
            started = True
            self._pump_task = asyncio.create_task(self._pump(caller))
            await caller.wait()
            finishing = True
            return await caller.finish()
        finally:
            try:
                if started and not finishing:
                    await caller.finish()
            finally:
                await _cancel(self._pump_task)
                await self.bridge.stop_async()
                self.events.save()

    async def _serve_agent(self) -> None:
        """EVA's ``_handle_session``: connect, then the three session tasks."""
        try:
            if not self._agent.is_open:
                await self._agent.open(self._agent_session)
            await self.recorder.serve(self._agent, self.bridge.receive_media)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - EVA logs it; the bridge sees the socket close
            self.agent_error = exc
            logger.error("Agent session error: %s", exc, exc_info=True)
        finally:
            self.recorder.end_session()
            # EVA's bridge sees its socket close here: if the call is still
            # running, that ends it as a disconnect.
            self.bridge.wire_closed()
            with contextlib.suppress(Exception):
                await self._agent.close()

    async def _pump(self, caller: Caller) -> None:
        """The SDK's receive thread: audio to the bridge, the rest to the caller."""
        try:
            async for event in self._caller_transport.events():
                if isinstance(event, OutputAudio):
                    self.bridge.output(event.pcm)
                else:
                    caller.on_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the SDK ends the session on any error
            logger.error("Caller event stream failed: %s", exc, exc_info=True)
        # Whatever ended the stream, the SDK's end_session stops the bridge.
        self.bridge.stop()

    async def _cleanup(self) -> None:
        """The worker's ``_cleanup``: the server stops and writes its outputs."""
        await _cancel(self._pump_task)
        task = self._agent_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=AGENT_SHUTDOWN_S)
            except TimeoutError:
                await _cancel(task)
        with contextlib.suppress(Exception):
            await self._caller_transport.close()
        try:
            self.recorder.save_outputs()
        except Exception as exc:  # noqa: BLE001 - the worker logs and goes on
            logger.warning("Error saving the assistant's outputs: %s", exc)


async def _cancel(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
