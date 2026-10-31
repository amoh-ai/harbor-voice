"""A caller that speaks in steps: the *when* of a full-duplex caller.

:class:`ReactiveCaller` is a port of tau2's streaming user — the per-tick
loop of ``StreamingMixin.get_next_chunk``, the actions of
``VoiceStreamingUserSimulator._perform_turn_taking_action`` and
``_process_tool_result``, and the decisions of ``basic_turn_taking_policy`` —
with the *what* (a :class:`~tau_voice.reactive.
ReactiveContentSource`) and the *how it sounds* (a :class:`CallerVoice`)
injected.

Each :meth:`ReactiveCaller.next_chunk` is one step. It is handed what the
agent produced in the *previous* step — the lockstep lag tau2's orchestrator
imposes — and returns what the caller emits in this one. A step that needs
the caller to generate a line awaits the content source and the voice before
returning; nothing advances the step count meanwhile, which is how tau2's
default (non-real-time) generation behaves: the agent never hears the
caller's thinking time.

Every counter, threshold and edge case here is tau2's, including the ones
that look odd (a tool call does not advance the caller's silence counter; a
caller line is split across steps character-evenly; a backchannel's flag
outlives it). They decide when the caller talks, which decides what the
benchmark measures.

:class:`PatientPolicy` is tau2's policy for a caller that waits its turn
(``control``). :class:`ReactivePolicy` adds the listener branch: while the
agent speaks, a :class:`Listener` is asked every few steps whether to
interrupt or backchannel — tau2's decision model, through the sidecar.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Any
from typing import Protocol
from typing import override

from harbor_voice._deps import numpy as _numpy
from harbor_voice.simulator import PreparedLine
from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.reactive import HangUp
from tau_voice.reactive import Heard
from tau_voice.reactive import Pending
from tau_voice.reactive import ReactiveContentSource
from tau_voice.reactive import Say
from tau_voice.reactive import StepRecord
from tau_voice.reactive import UseTools
from tau_voice.tau2_audio.voice_config import BACKCHANNEL_PHRASES


class Action(StrEnum):
    """tau2's ``BasicActionType``."""

    KEEP_TALKING = "keep_talking"
    STOP_TALKING = "stop_talking"
    GENERATE = "generate_message"
    WAIT = "wait"
    BACKCHANNEL = "backchannel"


class CallerVoice(Protocol):
    """Prepares lines and emits steps — a streamed ``UniversalSimulator``.

    ``prepare`` synthesises a line (clean, nothing sent). ``emit_step`` sends
    one step of audio — speech or silence — through the signal chain and
    returns what goes on the wire; the chain is stateful (a filter rings on
    into the next step), so it is called for every step, in order.
    ``finish`` records a line as spoken, or as cut off.
    See :class:`tau_voice.voice.SimulatorVoice`.
    """

    @property
    def step_bytes(self) -> int:
        """Synthesis-format bytes in one step."""
        ...

    @property
    def last_effects(self) -> Mapping[str, Any] | None:
        """tau2's effect results for the step last emitted, or ``None``."""
        ...

    async def prepare(self, text: str) -> PreparedLine: ...

    def emit_step(
        self, pcm: bytes, *, is_speech: bool, line: PreparedLine | None = None
    ) -> bytes: ...

    def finish(self, line: PreparedLine, *, truncated: bool) -> None: ...


@dataclass(frozen=True, slots=True)
class CallerChunk:
    """What the caller emits in one step."""

    audio: bytes
    """Wire-format audio for this step."""
    speech: bool
    text: str | None = None
    """This step's share of the line being spoken; for a tool call or a
    hang-up, whatever the model wrote."""
    tool_calls: tuple[CallerToolCall, ...] = ()
    hang_up: HangUp | None = None
    action: Action | None = None
    """tau2's turn-taking action for the step; ``None`` where tau2 sets none
    (waiting on tools, and the reply to a tool result)."""
    utterance_id: str | None = None
    """One id per line, on every step of it (tau2's ``utterance_ids``)."""
    effects: Mapping[str, Any] | None = None
    """tau2's effect results for the step (see ``CallerVoice.last_effects``)."""

    @property
    def is_meaningful(self) -> bool:
        """tau2's ``_has_meaningful_content``."""
        return bool(self.tool_calls) or bool(self.text) or self.speech


@dataclass(slots=True)
class _Queued:
    pcm: bytes
    text: str
    line: PreparedLine
    last: bool
    utterance: str


@dataclass(slots=True)
class CallerState:
    """tau2's ``StreamingState``, the fields a non-real-time caller uses."""

    heard_buffer: list[Heard] = field(default_factory=list)
    queue: list[_Queued] = field(default_factory=list)
    since_self: int = 0
    since_other: int = 0
    tick_count: int = 0
    consecutive_self: int = 0
    consecutive_other: int = 0
    ticks_since_backchannel: int = 0
    is_backchanneling: bool = False
    waiting_for_tool_results: bool = False
    delivering_tool_result_speech: bool = False
    records: list[StepRecord] = field(default_factory=list)

    @property
    def is_talking(self) -> bool:
        return bool(self.queue)

    def ongoing_speech(self) -> int:
        """Trailing run of speech steps in the heard buffer."""
        run = 0
        for heard in reversed(self.heard_buffer):
            if not heard.speech:
                break
            run += 1
        return run

    def interrupted(self) -> bool:
        return self.is_talking and self.ongoing_speech() > 0

    @property
    def overlap_initiator(self) -> str | None:
        if self.consecutive_self == 0 or self.consecutive_other == 0:
            return None
        return "other" if self.consecutive_self >= self.consecutive_other else "self"


@dataclass(frozen=True, slots=True)
class PatientPolicy:
    """tau2's ``basic_turn_taking_policy`` with interruption and backchannel off.

    The control caller. Thresholds are in steps; defaults are tau2's at
    0.2 s steps (1.0 s / 5.0 s / 1.0 s / 5.0 s).
    """

    wait_other: int = 5
    wait_self: int = 25
    yield_when_interrupted: int | None = 5
    yield_when_interrupting: int | None = 25

    async def decide(self, state: CallerState) -> Action:
        if state.is_talking:
            return self._while_talking(state)
        return self._when_silent(state)

    def _while_talking(self, state: CallerState) -> Action:
        if state.is_backchanneling:
            return Action.KEEP_TALKING
        if state.interrupted():
            if state.delivering_tool_result_speech:
                threshold = self.yield_when_interrupting
            elif state.overlap_initiator == "self":
                threshold = self.yield_when_interrupting
            else:
                threshold = self.yield_when_interrupted
            if (
                self.yield_when_interrupted is not None
                and threshold is not None
                and state.ongoing_speech() >= threshold
            ):
                return Action.STOP_TALKING
        return Action.KEEP_TALKING

    def _when_silent(self, state: CallerState) -> Action:
        if state.since_other > self.wait_other and state.since_self > self.wait_self:
            return Action.GENERATE
        return Action.WAIT


@dataclass(frozen=True, slots=True)
class Reaction:
    """tau2's ``ListenerReactionResult``: interrupting wins over backchanneling."""

    interrupt: bool = False
    backchannel: bool = False


class Listener(Protocol):
    """Decides how the caller reacts to the agent's ongoing speech.

    tau2's ``evaluate_listener_reactions`` with its caller's callbacks; see
    :class:`tau_voice.content.Tau2Listener`. Asked mid-step, before the
    step's record is written: ``state.records`` is the past, the last of
    ``state.heard_buffer`` is what the agent is saying now.
    """

    async def react(self, state: CallerState) -> Reaction: ...


@dataclass(frozen=True, slots=True)
class ReactivePolicy(PatientPolicy):
    """tau2's ``basic_turn_taking_policy`` with its listener branch.

    While the caller is silent and the agent is speaking, every
    ``check_interval`` steps (by the step count, as tau2 counts; ``None``:
    every step) the listener is asked: interrupt means generate a line now,
    over the agent; backchannel means a short acknowledgement. Otherwise, or
    without a listener, the caller waits its turn as :class:`PatientPolicy`.
    The question blocks the step, as the caller's own generation does.
    """

    listener: Listener | None = None
    check_interval: int | None = 10

    @override
    async def decide(self, state: CallerState) -> Action:
        if state.is_talking:
            return self._while_talking(state)
        due = self.check_interval is None or state.tick_count % self.check_interval == 0
        if self.listener is not None and state.ongoing_speech() > 0 and due:
            reaction = await self.listener.react(state)
            if reaction.interrupt:
                return Action.GENERATE
            if reaction.backchannel:
                return Action.BACKCHANNEL
        return self._when_silent(state)


class TurnPolicy(Protocol):
    async def decide(self, state: CallerState) -> Action: ...


def tau2_backchannels(voice_seed: int) -> Iterator[str]:
    """tau2's backchannel phrases, in its order: ``Random(voice_seed + 100)``."""
    rng = random.Random(voice_seed + 100)
    while True:
        yield rng.choice(BACKCHANNEL_PHRASES)


FADE_OUT_SAMPLES = 100
"""tau2's ``DEFAULT_FADE_OUT_SAMPLES`` (~6 ms at 16 kHz)."""


def _pad(pcm: bytes, size: int) -> bytes:
    """tau2's ``pad_audio_with_zeros``: fade a short last chunk out, then pad.

    The fade is linear over the final 100 samples and truncates back to
    int16, exactly as tau2's ``apply_fade_out`` does.
    """
    if len(pcm) >= size:
        return pcm
    if pcm:
        np = _numpy()
        samples = np.frombuffer(pcm, dtype="<i2").copy()
        fade = min(FADE_OUT_SAMPLES, len(samples))
        window = np.linspace(1.0, 0.0, fade)
        samples[-fade:] = (samples[-fade:].astype(np.float64) * window).astype(np.int16)
        pcm = samples.tobytes()
    return pcm + bytes(size - len(pcm))


def split_text(text: str, pieces: int) -> list[str]:
    """tau2's ``_create_content_chunks``: characters spread evenly over pieces."""
    buckets: list[list[str]] = [[] for _ in range(pieces)]
    if text:
        interval = pieces / len(text)
        for index, char in enumerate(text):
            buckets[int(index * interval)].append(char)
    return ["".join(bucket) for bucket in buckets]


class ReactiveCaller:
    """One caller, stepped. See the module docstring."""

    def __init__(
        self,
        *,
        content: ReactiveContentSource,
        voice: CallerVoice,
        policy: TurnPolicy,
        backchannels: Iterator[str] | None = None,
    ) -> None:
        self.content = content
        self.voice = voice
        self.policy = policy
        self.backchannels = backchannels
        """Phrases for backchannels, in order (:func:`tau2_backchannels`)."""
        self.state = CallerState()

    async def next_chunk(
        self,
        heard: Heard,
        tool_results: tuple[CallerToolResult, ...] | None = None,
    ) -> CallerChunk:
        """One step: what the caller emits, given the agent's previous step."""
        st = self.state
        st.tick_count += 1
        st.heard_buffer.append(heard)
        just_started = heard.speech and st.since_other > 0
        if heard.speech:
            st.since_other = 0
            st.consecutive_other += 1
            st.ticks_since_backchannel += 1
        else:
            st.since_other += 1
            st.consecutive_other = 0
        if just_started:
            st.ticks_since_backchannel = 1

        if tool_results is not None:
            st.waiting_for_tool_results = False
            chunk = await self._after_tools(tool_results)
            if chunk.tool_calls:
                st.waiting_for_tool_results = True
        elif st.waiting_for_tool_results:
            chunk = self._waiting()
        else:
            chunk = await self._act(await self.policy.decide(st))
            if chunk.tool_calls:
                st.waiting_for_tool_results = True

        st.consecutive_self = st.consecutive_self + 1 if chunk.is_meaningful else 0
        heard_meaningful = heard.speech or bool(heard.text)
        st.records.append(
            StepRecord(
                step=len(st.records),
                t_s=0.0,
                said_text=chunk.text if chunk.is_meaningful else None,
                said_speech=chunk.speech if chunk.is_meaningful else False,
                said_tools=chunk.tool_calls,
                heard=heard if heard_meaningful else None,
                tool_results=tool_results or (),
            )
        )
        return chunk

    # -------------------------------------------------------------- actions

    def _silence(self, *, action: Action | None, text: str | None = "") -> CallerChunk:
        return CallerChunk(
            audio=self.voice.emit_step(bytes(self.voice.step_bytes), is_speech=False),
            speech=False,
            text=text,
            action=action,
            effects=self.voice.last_effects,
        )

    def _waiting(self) -> CallerChunk:
        """tau2's ``_emit_waiting_chunk``."""
        self.state.since_self += 1
        return self._silence(action=None)

    def _speak_next(self, action: Action | None) -> CallerChunk:
        st = self.state
        piece = st.queue.pop(0)
        # Only the keep-talking path clears these in tau2; the first piece of
        # a freshly generated line does not.
        if action is Action.KEEP_TALKING and not st.queue:
            st.is_backchanneling = False
            st.delivering_tool_result_speech = False
        st.since_self = 0
        audio = self.voice.emit_step(piece.pcm, is_speech=True, line=piece.line)
        effects = self.voice.last_effects
        if piece.last:
            self.voice.finish(piece.line, truncated=False)
        return CallerChunk(
            audio=audio,
            speech=True,
            text=piece.text,
            action=action,
            utterance_id=piece.utterance,
            effects=effects,
        )

    async def _queue_line(self, text: str) -> None:
        line = await self.voice.prepare(text)
        pcm = line.pcm
        size = self.voice.step_bytes
        steps = [pcm[i : i + size] for i in range(0, len(pcm), size)] or [b""]
        steps = [_pad(s, size) for s in steps]
        # tau2 spreads the text as synthesised (with any vocal tic) over steps.
        texts = split_text(line.said, len(steps))
        utterance = str(uuid.uuid4())
        self.state.queue.extend(
            _Queued(p, t, line, index == len(steps) - 1, utterance)
            for index, (p, t) in enumerate(zip(steps, texts, strict=True))
        )

    async def _act(self, action: Action) -> CallerChunk:
        st = self.state
        if action is Action.KEEP_TALKING:
            return self._speak_next(action)
        if action is Action.STOP_TALKING:
            if st.queue:
                self.voice.finish(st.queue[0].line, truncated=True)
            st.queue.clear()
            st.is_backchanneling = False
            st.since_self += 1
            return self._silence(action=action)
        if action is Action.GENERATE:
            pending = st.heard_buffer[-1] if st.heard_buffer else None
            if pending is not None and not pending.text:
                pending = None
            reply = await self.content.respond(st.records, pending)
            st.heard_buffer.clear()
            return await self._handle(reply, action)
        if action is Action.WAIT:
            st.since_self += 1
            return self._silence(action=action)
        if action is Action.BACKCHANNEL:
            if self.backchannels is None:
                raise ValueError("a backchannel was decided, but no phrases were given")
            # The heard buffer is kept: a backchannel is not a reply.
            await self._queue_line(next(self.backchannels))
            chunk = self._speak_next(action)
            # Set after the first piece, as tau2 does: only a keep-talking step
            # that empties the queue clears it, so a one-step backchannel
            # leaves it set into the caller's next line.
            st.is_backchanneling = True
            st.ticks_since_backchannel = 0
            return chunk
        raise ValueError(f"unknown action {action.value}")

    async def _handle(
        self, reply: Say | UseTools | HangUp, action: Action | None
    ) -> CallerChunk:
        if isinstance(reply, UseTools):
            # tau2 returns before touching the silence counter here.
            audio = self.voice.emit_step(bytes(self.voice.step_bytes), is_speech=False)
            effects = self.voice.last_effects
            if effects is not None:
                # tau2 copies only the source effects onto a tool call.
                effects = {**effects, "channel_effects": None}
            return CallerChunk(
                audio=audio,
                speech=False,
                text=reply.text,
                tool_calls=reply.calls,
                action=action,
                effects=effects,
            )
        if isinstance(reply, HangUp):
            # tau2 returns the stop message unsynthesised: no audio, and its
            # speech flag left at the message default (True). The call ends.
            return CallerChunk(
                audio=b"",
                speech=True,
                text=reply.text,
                hang_up=reply,
                action=action,
            )
        if not reply.text:
            return self._waiting()
        await self._queue_line(reply.text)
        return self._speak_next(action)

    async def _after_tools(self, results: tuple[CallerToolResult, ...]) -> CallerChunk:
        """tau2's ``_process_tool_result``: respond now, speak from next step."""
        st = self.state
        pending: Pending = results
        reply = await self.content.respond(st.records, pending)
        if isinstance(reply, (UseTools, HangUp)):
            # tau2 sets no turn-taking action on a reply to a tool result.
            return await self._handle(reply, None)
        if reply.text:
            await self._queue_line(reply.text)
            st.delivering_tool_result_speech = True
        return self._waiting()

    @property
    def records(self) -> Sequence[StepRecord]:
        return self.state.records
