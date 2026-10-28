"""SteppedCall mechanics with a fake agent: lag, pacing, tool gating, stops."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import override

import pytest

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import OptionalOp
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.providers.openai.realtime import TELEPHONY
from harbor_voice.simulator import PreparedLine
from tau_voice.caller import PatientPolicy
from tau_voice.caller import ReactiveCaller
from harbor_voice.simulator.content import SourceKind
from tau_voice.reactive import CallerReply
from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.reactive import HangUp
from tau_voice.reactive import Pending
from tau_voice.reactive import Say
from tau_voice.agent_steps import LiveAgentStepper
from tau_voice.reactive import StepRecord
from tau_voice.stepped_call import CallResult
from tau_voice.stepped_call import StepGate
from tau_voice.stepped_call import SteppedCall
from tau_voice.stepped_call import Termination
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import StreamEvent

pytestmark = pytest.mark.unit

SPEECH = b"\x10"
"""A μ-law byte that is not silence; the fake agent listens for it."""


class FakeAgent(VoiceTransport):
    """Replies with 0.4 s of audio + a transcript once the caller has spoken."""

    def __init__(self) -> None:
        self._open = False
        self._inbox: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self.packets: list[bytes] = []
        self.frames: list[AudioFrame] = []
        self.heard_speech = 0
        self.replied = False
        self.truncated: list[tuple[str, int]] = []

    @property
    @override
    def capabilities(self) -> Capabilities:
        return Capabilities(
            duplex=frozenset({Duplex.FULL}),
            pipeline=Pipeline.S2S,
            turn_control=TurnControl.NATIVE_VAD,
            optional_ops=frozenset({OptionalOp.EVENT_STREAM, OptionalOp.TRUNCATE}),
            audio_in=TELEPHONY,
            audio_out=TELEPHONY,
        )

    @property
    @override
    def is_open(self) -> bool:
        return self._open

    @override
    async def open(self, config: SessionConfig) -> None:
        self._open = True

    @override
    async def tx(self, frame: AudioFrame) -> None:
        self.packets.append(frame.pcm)
        self.frames.append(frame)
        if SPEECH in frame.pcm:
            self.heard_speech += 1
        elif self.heard_speech and not self.replied:
            self.replied = True
            self._inbox.put_nowait(OutputTranscript("r1", "Sure, one moment.", 0.0))
            self._inbox.put_nowait(OutputAudio(b"\x20" * 3200, "r1", 0.0))

    @override
    async def events(self) -> AsyncIterator[StreamEvent]:
        while (event := await self._inbox.get()) is not None:
            yield event

    @override
    async def rx(self) -> AsyncIterator[AudioFrame]:
        if False:  # pragma: no cover - events() is the interface used here
            yield

    @override
    async def truncate(self, utterance: str, audio_end_ms: int) -> None:
        self.truncated.append((utterance, audio_end_ms))

    @override
    async def close(self) -> None:
        self._open = False
        self._inbox.put_nowait(None)


class Script:
    def __init__(self, replies: list[CallerReply]) -> None:
        self.replies = iter(replies)
        self.seen: list[Pending] = []

    @property
    def kind(self) -> SourceKind:
        return SourceKind.SCRIPT

    @property
    def is_deterministic(self) -> bool:
        return True

    async def respond(
        self, steps: Sequence[StepRecord], pending: Pending
    ) -> CallerReply:
        self.seen.append(pending)
        return next(self.replies)

    async def run_tools(
        self, calls: Sequence[CallerToolCall], *, step: int
    ) -> tuple[CallerToolResult, ...]:
        return tuple(CallerToolResult(c.id, "ok") for c in calls)


class LoudVoice:
    """One 20 ms step of 'speech' per 5 characters; steps map PCM to μ-law."""

    step_bytes = 640  # 20 ms at 16 kHz
    last_effects = None

    def __init__(self) -> None:
        self.finished: list[tuple[str, bool]] = []

    async def prepare(self, text: str) -> PreparedLine:
        pcm = b"\x01\x01" * (320 * max(1, len(text) // 5))
        return PreparedLine(said=text, pcm=pcm, clean_digest=None)

    def emit_step(
        self, pcm: bytes, *, is_speech: bool, line: PreparedLine | None = None
    ) -> bytes:
        return (SPEECH if any(pcm) else b"\xff") * (len(pcm) // 4)

    def finish(self, line: PreparedLine, *, truncated: bool) -> None:
        self.finished.append((line.said, truncated))


def _call(script: Script, agent: FakeAgent, **kw) -> SteppedCall:
    caller = ReactiveCaller(
        content=script,
        voice=LoudVoice(),
        # Long enough after speaking for the agent's lagged reply to arrive.
        policy=PatientPolicy(wait_other=2, wait_self=40),
    )

    async def user_tools(calls, step):
        return await script.run_tools(calls, step=step)

    return SteppedCall(
        caller=caller,
        transport=agent,
        run_user_tools=user_tools,
        **{"audio_format": TELEPHONY, "step_s": 0.02, "packet_ms": 20, **kw},
    )


async def test_a_short_call_runs_in_lockstep_and_ends_on_hang_up() -> None:
    agent = FakeAgent()
    await agent.open(SessionConfig(party="agent"))
    script = Script(
        [Say("I need to change a flight."), HangUp("Thanks ###STOP###", "###STOP###")]
    )
    result = await _call(script, agent, max_steps=200).run()
    await agent.close()

    assert result.termination is Termination.USER_STOP
    # The greeting is text, never speech, and the caller's first look at the
    # agent is that text with nothing pending to read.
    assert result.steps[0].agent_text == "Hi! How can I help you today?"
    assert result.steps[0].agent_speech is False
    # Caller audio went out in 20 ms packets of 160 μ-law bytes.
    assert agent.packets and {len(p) for p in agent.packets} == {160}
    assert agent.heard_speech >= 2
    # The agent's reply reached the caller's record as text, after the
    # caller had spoken (the lag), and before it hung up.
    said = [s.step for s in result.steps if s.user.speech and s.user.text]
    heard = [s.step for s in result.steps if s.agent_text and s.agent_speech]
    assert said and heard and min(heard) > min(said)
    # The transcript is handed out in proportion to audio played, so across
    # the steps that played it, it reads back whole.
    assert "".join(s.agent_text or "" for s in result.steps).endswith(
        "Sure, one moment."
    )


async def test_max_steps_ends_a_call_nobody_ends() -> None:
    agent = FakeAgent()
    await agent.open(SessionConfig(party="agent"))
    script = Script([Say("Hello there."), Say("Still there?")] + [Say("x")] * 20)
    result = await _call(script, agent, max_steps=12).run()
    await agent.close()
    assert result.termination is Termination.MAX_STEPS
    assert len(result.steps) == 13  # step 0 plus twelve


async def test_a_tool_waits_for_the_next_step() -> None:
    gate = StepGate()
    await gate.advance(3)
    released = asyncio.Event()

    async def tool() -> None:
        await gate.after(3)
        released.set()

    task = asyncio.create_task(tool())
    await asyncio.sleep(0.01)
    assert not released.is_set()
    await gate.advance(4)
    await asyncio.wait_for(task, 1.0)
    assert released.is_set()


def test_a_transport_without_the_stream_is_refused() -> None:
    class Plain(FakeAgent):
        @property
        @override
        def capabilities(self) -> Capabilities:
            caps = super().capabilities
            return caps.model_copy(update={"optional_ops": frozenset()})

    with pytest.raises(ValueError, match="event_stream"):
        _call(Script([]), Plain())


async def test_a_tool_call_belongs_to_the_window_it_arrived_in() -> None:
    """Inside a window: that step. Between windows: the next one (tau2 reads it then)."""
    agent = FakeAgent()
    call = _call(Script([]), agent)

    call._window = 5  # an agent step's receive window is open
    inside = asyncio.create_task(call.tool_window())
    await asyncio.sleep(0.01)
    assert not inside.done(), "must not execute while its window is open"
    async with call._tools:
        call._closed_through = 5
        call._tools.notify_all()
    assert await asyncio.wait_for(inside, 1.0) == 5
    await call.tool_done(5)

    call._window = None  # between windows: the caller is generating
    await call.gate.advance(5)
    between = asyncio.create_task(call.tool_window())
    await asyncio.sleep(0.01)
    assert not between.done()
    async with call._tools:
        call._closed_through = 6
        call._tools.notify_all()
    assert await asyncio.wait_for(between, 1.0) == 6
    await call.tool_done(6)


class NoTruncate(FakeAgent):
    """OpenAI Live's shape: the event stream, but nothing to truncate."""

    @property
    @override
    def capabilities(self) -> Capabilities:
        caps = super().capabilities
        return caps.model_copy(
            update={"optional_ops": frozenset({OptionalOp.EVENT_STREAM})}
        )


def _live() -> LiveAgentStepper:
    return LiveAgentStepper(bytes_per_step=160, bytes_per_second=8000)


def test_only_a_stepper_that_truncates_needs_truncate() -> None:
    with pytest.raises(ValueError, match="does not declare truncate"):
        _call(Script([]), NoTruncate())
    _call(Script([]), NoTruncate(), stepper=_live(), send_format=AudioFormat())


def test_a_stepper_that_converts_needs_the_send_format() -> None:
    with pytest.raises(ValueError, match="send_format"):
        _call(Script([]), NoTruncate(), stepper=_live())


async def test_a_live_step_goes_out_whole_in_the_providers_format() -> None:
    """tau2's Live adapter converts a tick and sends it in one call."""
    agent = NoTruncate()
    call = _call(Script([]), agent, stepper=_live(), send_format=AudioFormat())
    audio = SPEECH * 160
    step = await call._agent_step(1, audio, CallResult())

    assert len(agent.frames) == 1
    assert agent.frames[0].fmt == AudioFormat()
    assert agent.frames[0].pcm == _live().convert_input(audio)
    assert step.was_truncated is False


async def test_realtime_pacing_makes_up_the_time_the_caller_spent() -> None:
    """tau2's ``realtime_pacing``: steps keep a fixed schedule, so after the
    caller pauses to generate, the next steps run back to back (each still
    listening 10 ms) until the call is back on time."""

    async def elapsed(paced: bool) -> float:
        call = _call(Script([]), FakeAgent(), step_s=0.1, realtime_pacing=paced)
        loop = asyncio.get_running_loop()
        start = loop.time()
        await call._agent_step(1, b"", CallResult())
        await asyncio.sleep(0.3)  # the caller generating
        for k in range(2, 7):
            await call._agent_step(k, b"", CallResult())
        return loop.time() - start

    assert await elapsed(True) == pytest.approx(0.6, abs=0.08)
    assert await elapsed(False) == pytest.approx(0.9, abs=0.08)
