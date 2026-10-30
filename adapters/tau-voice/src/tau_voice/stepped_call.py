"""A full-duplex call run in lockstep steps, the way tau-voice runs one.

tau2's ``FullDuplexOrchestrator.step`` gives each side the *other side's
previous chunk*: in step ``k`` the caller hears what the agent produced in
step ``k-1``, and the agent receives the caller audio of step ``k-1``. The
agent side is real time — that step's caller audio goes out in 20 ms packets
over 0.2 s while the agent's output is collected — and the caller side is not:
when the caller generates a line, the loop simply waits, so no call time
passes and the agent never hears the caller thinking.

Agent tool calls follow tau2's timing exactly. A call belongs to the step
whose receive window it arrived in — or, if it arrived between windows (while
the caller was generating), to the next one, which is the window tau2 would
have read it in. It is executed once that window has closed, so after the
caller's own tool calls of the same step, as tau2's orchestrator orders them,
and its result reaches the model at the start of the following step. An
executor brackets its work with :meth:`SteppedCall.tool_window` and
:meth:`SteppedCall.tool_done`, then waits on :class:`StepGate` for delivery.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import OptionalOp
from harbor_voice.ids import PartyId
from tau_voice.agent_steps import AgentStep
from tau_voice.agent_steps import AgentStepper
from tau_voice.caller import CallerChunk
from tau_voice.caller import ReactiveCaller
from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.reactive import Heard
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.stream import StreamEvent

GREETING = "Hi! How can I help you today?"
"""tau2's ``create_initial_message`` default: text only, never spoken."""


class Termination(StrEnum):
    """tau2's ``TerminationReason`` values a stepped call can end with."""

    AGENT_STOP = "agent_stop"
    USER_STOP = "user_stop"
    MAX_STEPS = "max_steps"
    TOO_MANY_ERRORS = "too_many_errors"


class StepGate:
    """Lets a tool executor wait for the next step boundary."""

    def __init__(self) -> None:
        self.step = 0
        self._changed = asyncio.Condition()

    async def advance(self, step: int) -> None:
        async with self._changed:
            self.step = step
            self._changed.notify_all()

    async def after(self, step: int) -> None:
        """Return once a step later than ``step`` has begun."""
        async with self._changed:
            await self._changed.wait_for(lambda: self.step > step)


@dataclass(frozen=True, slots=True)
class StepLog:
    """Both sides of one step, as the grader sees them."""

    step: int
    t_s: float
    user: CallerChunk
    agent_text: str | None
    agent_speech: bool
    agent: AgentStep | None = None
    user_tool_results: tuple[CallerToolResult, ...] = ()


@dataclass(slots=True)
class CallResult:
    steps: list[StepLog] = field(default_factory=list)
    termination: Termination = Termination.MAX_STEPS
    wall_s: float = 0.0
    truncations: list[tuple[int, str, int]] = field(default_factory=list)


UserTools = Callable[
    [Sequence[CallerToolCall], int], Awaitable[tuple[CallerToolResult, ...]]
]


class SteppedCall:
    """Run one call between a :class:`ReactiveCaller` and an agent transport.

    The transport must declare ``EVENT_STREAM`` and be open, and
    ``TRUNCATE`` too when the stepper cuts the agent off on a barge-in.

    Args:
        run_user_tools: Executes the caller's own tool calls made at a step.
        step_s: Step length; tau2's ``tick_duration_seconds``.
        packet_ms: Pacing of caller audio within a step; tau2's VoIP interval.
        max_steps: tau2's ``max_steps_ticks`` (1200 s / 0.2 s).
        audio_format: Of the wire, both ways; the transport's ``audio_in``.
        stepper: The provider's tick (:class:`AgentStepper`, tau2's Realtime
            adapter, by default; :class:`LiveAgentStepper` for OpenAI Live).
            A stepper with ``convert_input`` turns each step of caller audio
            into the provider's format, sent whole (``send_format``), as
            tau2's Live adapter queues a tick at once for WebRTC to clock.
        realtime_pacing: tau2's ``realtime_pacing``: steps start on a fixed
            schedule rather than when the last one ended, so after a pause
            (the caller thinking) they run back to back until caught up.
    """

    def __init__(
        self,
        *,
        caller: ReactiveCaller,
        transport: VoiceTransport,
        run_user_tools: UserTools,
        audio_format: AudioFormat,
        gate: StepGate | None = None,
        step_s: float = 0.2,
        packet_ms: int = 20,
        max_steps: int = 6000,
        greeting: str | None = GREETING,
        party: PartyId = PartyId("caller"),
        stepper: AgentStepper | None = None,
        realtime_pacing: bool = False,
        send_format: AudioFormat | None = None,
    ) -> None:
        caps = transport.capabilities
        step_bytes = (
            int(audio_format.sample_rate_hz * step_s) * audio_format.bytes_per_sample
        )
        self.stepper = stepper or AgentStepper(
            bytes_per_step=step_bytes,
            bytes_per_second=audio_format.sample_rate_hz
            * audio_format.bytes_per_sample,
        )
        needed = [OptionalOp.EVENT_STREAM]
        if self.stepper.truncates:
            needed.append(OptionalOp.TRUNCATE)
        for op in needed:
            if not caps.supports(op):
                raise ValueError(
                    f"{type(transport).__name__} does not declare {op.value}; a "
                    f"stepped call needs the agent's output in arrival order and "
                    f"a way to truncate it"
                )
        self._convert = getattr(self.stepper, "convert_input", None)
        if self._convert is not None and send_format is None:
            raise ValueError("a stepper that converts caller audio needs send_format")
        self.send_format = send_format
        self.realtime_pacing = realtime_pacing
        self._next_start: float | None = None
        self.caller = caller
        self.transport = transport
        self.run_user_tools = run_user_tools
        self.fmt = audio_format
        self.gate = gate or StepGate()
        self.step_s = step_s
        self.packet_bytes = audio_format.frame_bytes(packet_ms)
        self.step_bytes = step_bytes
        self.max_steps = max_steps
        self.greeting = greeting
        self.party = party
        self._stop: Termination | None = None
        self._incoming: list[StreamEvent] = []
        self._seq = 0
        self._window: int | None = None
        self._closed_through = 0
        self._tool_steps: dict[int, int] = {}
        self._tools = asyncio.Condition()

    def request_stop(self, reason: Termination) -> None:
        """Called from an executor: end the call after the current step."""
        if self._stop is None:
            self._stop = reason

    async def tool_window(self) -> int:
        """Register an agent tool call; return its step once it may execute.

        The step is the open receive window, or the next one if none is open.
        Returns after that window has closed. Pair with :meth:`tool_done`.
        """
        step = self._window if self._window is not None else self.gate.step + 1
        async with self._tools:
            self._tool_steps[step] = self._tool_steps.get(step, 0) + 1
            await self._tools.wait_for(lambda: self._closed_through >= step)
        return step

    async def tool_done(self, step: int) -> None:
        async with self._tools:
            self._tool_steps[step] -= 1
            self._tools.notify_all()

    async def run(self) -> CallResult:
        result = CallResult()
        started = time.perf_counter()
        reader = asyncio.create_task(self._read())
        try:
            await self._loop(result)
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            result.wall_s = time.perf_counter() - started
        return result

    async def _loop(self, result: CallResult) -> None:
        # Step 0, as FullDuplexOrchestrator._initialize_full_duplex: the
        # caller answers a silent dummy, the agent "says" the text greeting.
        user_prev = await self.caller.next_chunk(Heard(None, False))
        heard = Heard(self.greeting, False)
        result.steps.append(StepLog(0, 0.0, user_prev, self.greeting, False))
        if user_prev.hang_up is not None:
            result.termination = Termination.USER_STOP
            return
        pending_user: tuple[CallerToolResult, ...] | None = None
        if user_prev.tool_calls:
            pending_user = await self.run_user_tools(user_prev.tool_calls, 0)

        for k in range(1, self.max_steps + 1):
            delivered = pending_user
            user_k = await self.caller.next_chunk(heard, delivered)
            pending_user = None
            if user_k.tool_calls:
                pending_user = await self.run_user_tools(user_k.tool_calls, k)
            if user_k.hang_up is not None:
                self.request_stop(Termination.USER_STOP)

            agent_k = await self._agent_step(k, user_prev.audio, result)
            heard = Heard(agent_k.transcript or None, agent_k.contains_speech)
            result.steps.append(
                StepLog(
                    k,
                    k * self.step_s,
                    user_k,
                    agent_k.transcript or None,
                    agent_k.contains_speech,
                    agent=agent_k,
                    user_tool_results=delivered or (),
                )
            )
            user_prev = user_k
            if self._stop is not None:
                result.termination = self._stop
                return
        result.termination = Termination.MAX_STEPS

    async def _agent_step(self, k: int, audio: bytes, result: CallResult) -> AgentStep:
        """tau2's ``_async_run_tick`` for one step, real time."""
        loop = asyncio.get_running_loop()
        tick_start = loop.time()
        if self.realtime_pacing:
            if self._next_start is not None:
                tick_start = self._next_start
            self._next_start = tick_start + self.step_s
        # Tool results from the last step are answered first, as tau2 flushes
        # them before sending this step's audio.
        await self.gate.advance(k)
        self._window = k
        self.stepper.begin()
        if self._convert is not None:
            await self._send_whole(self._convert(audio))
        else:
            await self._send(audio)
        remaining = self.step_s - (loop.time() - tick_start)
        if self.realtime_pacing:
            # tau2 still listens 10 ms when a step starts late (catching up).
            await asyncio.sleep(max(0.01, remaining))
        elif remaining > 0:
            await asyncio.sleep(remaining)
        events, self._incoming = self._incoming, []
        for event in events:
            truncation = self.stepper.feed(event)
            if truncation is not None:
                result.truncations.append(
                    (k, truncation.utterance, truncation.audio_end_ms)
                )
                await self.transport.truncate(
                    truncation.utterance, truncation.audio_end_ms
                )
        step = self.stepper.end(user_bytes=len(audio))
        # Close the window, then let this step's agent tool calls execute —
        # after the caller's, as tau2 orders them — before the loop decides
        # whether a stop tool ended the call.
        self._window = None
        async with self._tools:
            self._closed_through = k
            self._tools.notify_all()
            await self._tools.wait_for(lambda: self._tool_steps.get(k, 0) == 0)
        return step

    async def _send(self, audio: bytes) -> None:
        for offset in range(0, len(audio), self.packet_bytes):
            packet = audio[offset : offset + self.packet_bytes]
            if len(packet) < self.packet_bytes:
                break
            await self.transport.tx(
                AudioFrame(
                    pcm=packet,
                    seq=self._seq,
                    party=self.party,
                    t_produced=0.0,
                    fmt=self.fmt,
                )
            )
            self._seq += 1
            await asyncio.sleep(
                self.packet_bytes
                / (self.fmt.sample_rate_hz * self.fmt.bytes_per_sample)
            )

    async def _send_whole(self, audio: bytes) -> None:
        """One step at once, in the provider's format (``send_audio_instant``)."""
        if not audio:
            return
        assert self.send_format is not None
        await self.transport.tx(
            AudioFrame(
                pcm=audio,
                seq=self._seq,
                party=self.party,
                t_produced=0.0,
                fmt=self.send_format,
            )
        )
        self._seq += 1

    async def _read(self) -> None:
        async for event in self.transport.events():
            self._incoming.append(event)
