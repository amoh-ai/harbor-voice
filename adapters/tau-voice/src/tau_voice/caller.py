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
caller line is split across steps character-evenly). They decide when the
caller talks, which decides what the benchmark measures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Protocol

from harbor_voice._deps import numpy as _numpy
from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.reactive import HangUp
from tau_voice.reactive import Heard
from tau_voice.reactive import Pending
from tau_voice.reactive import ReactiveContentSource
from tau_voice.reactive import Say
from tau_voice.reactive import StepRecord
from tau_voice.reactive import UseTools


class Action(StrEnum):
    """tau2's ``BasicActionType``."""

    KEEP_TALKING = "keep_talking"
    STOP_TALKING = "stop_talking"
    GENERATE = "generate_message"
    WAIT = "wait"
    BACKCHANNEL = "backchannel"


class CallerVoice(Protocol):
    """Turns a line into per-step audio, and any step's audio into the wire format.

    ``synthesize`` returns speech at the voice's synthesis rate; ``channel``
    takes one step of that (speech or silence) and returns what goes on the
    wire. The channel is stateful — a filter rings on into the next step — so
    it is called for **every** step, silent ones included, in order.
    """

    @property
    def step_bytes(self) -> int:
        """Synthesis-format bytes in one step."""
        ...

    async def synthesize(self, text: str) -> bytes: ...

    def channel(self, pcm: bytes) -> bytes: ...


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

    @property
    def is_meaningful(self) -> bool:
        """tau2's ``_has_meaningful_content``."""
        return bool(self.tool_calls) or bool(self.text) or self.speech


@dataclass(slots=True)
class _Queued:
    pcm: bytes
    text: str


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

    The **control** caller. Thresholds are in steps; defaults are tau2's at
    0.2 s steps (1.0 s / 5.0 s / 1.0 s / 5.0 s).
    """

    wait_other: int = 5
    wait_self: int = 25
    yield_when_interrupted: int | None = 5
    yield_when_interrupting: int | None = 25

    def decide(self, state: CallerState) -> Action:
        if state.is_talking:
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
        if state.since_other > self.wait_other and state.since_self > self.wait_self:
            return Action.GENERATE
        return Action.WAIT


class TurnPolicy(Protocol):
    def decide(self, state: CallerState) -> Action: ...


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
    ) -> None:
        self.content = content
        self.voice = voice
        self.policy = policy
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
            chunk = await self._act(self.policy.decide(st))
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
            audio=self.voice.channel(bytes(self.voice.step_bytes)),
            speech=False,
            text=text,
            action=action,
        )

    def _waiting(self) -> CallerChunk:
        """tau2's ``_emit_waiting_chunk``."""
        self.state.since_self += 1
        return self._silence(action=None)

    def _speak_next(self, action: Action) -> CallerChunk:
        st = self.state
        piece = st.queue.pop(0)
        # Only the keep-talking path clears these in tau2; the first piece of
        # a freshly generated line does not.
        if action is Action.KEEP_TALKING and not st.queue:
            st.is_backchanneling = False
            st.delivering_tool_result_speech = False
        st.since_self = 0
        return CallerChunk(
            audio=self.voice.channel(piece.pcm),
            speech=True,
            text=piece.text,
            action=action,
        )

    async def _queue_line(self, text: str) -> None:
        pcm = await self.voice.synthesize(text)
        size = self.voice.step_bytes
        frames = [pcm[i : i + size] for i in range(0, len(pcm), size)] or [b""]
        frames = [_pad(f, size) for f in frames]
        texts = split_text(text, len(frames))
        self.state.queue.extend(
            _Queued(p, t) for p, t in zip(frames, texts, strict=True)
        )

    async def _act(self, action: Action) -> CallerChunk:
        st = self.state
        if action is Action.KEEP_TALKING:
            return self._speak_next(action)
        if action is Action.STOP_TALKING:
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
        raise ValueError(f"{action.value} is not supported by this caller yet")

    async def _handle(
        self, reply: Say | UseTools | HangUp, action: Action
    ) -> CallerChunk:
        if isinstance(reply, UseTools):
            # tau2 returns before touching the silence counter here.
            return CallerChunk(
                audio=self.voice.channel(bytes(self.voice.step_bytes)),
                speech=False,
                text=reply.text,
                tool_calls=reply.calls,
                action=action,
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
            return await self._handle(reply, Action.GENERATE)
        if reply.text:
            await self._queue_line(reply.text)
            st.delivering_tool_result_speech = True
        return self._waiting()

    @property
    def records(self) -> Sequence[StepRecord]:
        return self.state.records
