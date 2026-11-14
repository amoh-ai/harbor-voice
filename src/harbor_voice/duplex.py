"""The genuinely full-duplex driver.

:class:`~harbor_voice.adapter.VoiceAgentAdapter` says in its own docstring that
"a genuinely full-duplex driver uses the transport and the floor directly and
does not go through here". This is that driver, and the reason it cannot go
through ``call()`` is simple: ``call()`` sends, then drains, then returns. An
interruption has to happen *during* the drain.

What it does, per turn:

1. Sends the caller's utterance under the floor, claiming it.
2. Drains the agent's reply while watching how much audio has arrived.
3. When the reply reaches a scheduled probe's offset, speaks again —
   with :class:`~harbor_voice.floor.Bid` saying whether this is a claim on the
   floor or a backchannel.
4. Observes what the agent did about it, and records whether the probe landed
   somewhere a conclusion could be drawn from at all.

### The floor runs on call time, not wall time

Everything here — probe offsets, floor intervals, overlap, yield latency — is
positioned on a call timeline built from audio durations plus the
*observed* response latency. A harness is free to push an utterance to the
transport in two milliseconds, but that utterance is still eight seconds of
speech, and the caller still held the floor for eight seconds.

Measuring the floor in wall time instead is not a rounding error, it is a
zero: a run that pushes bytes as fast as the queue accepts them reports
``turns.overlap_s = 0.000`` for a call in which the caller demonstrably talked
over the agent for half a second. That number is worse than missing, because
it looks like a finding.

So a probe at ``at_s=1.0`` fires one second into the agent's speech, and
``yield_latency_s`` is how much more of the agent's speech the caller heard
after interrupting. In a real-time call every one of these numbers equals its
wall-clock twin; off real time, these are the ones that stay meaningful. The
session clock is still used, but only for the one thing it is the right
instrument for: measuring how long the agent took to start answering.

### One mechanism, two families

Nothing here branches on "is this a barge-in or a selectivity probe". It
transmits the probe with its bid and records what happened; yielding is
success for one bid and failure for the other, and that polarity lives in
:attr:`~harbor_voice.perturbation.conversation.ProbeResult.correct`. Having
one code path is why a backchannel that the agent correctly talked through
cannot accidentally be counted as a failed interruption.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import override

from harbor_voice.audio.clock import Clock
from harbor_voice.audio import convert
from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.adapter import Turn
from harbor_voice.audio.stages import StageMark
from harbor_voice.capabilities import Duplex
from harbor_voice.floor import BargeInFloor
from harbor_voice.floor import Bid
from harbor_voice.floor import Decision
from harbor_voice.ids import PartyId
from harbor_voice.perturbation.conversation import ConversationPlan
from harbor_voice.perturbation.conversation import Landing
from harbor_voice.perturbation.conversation import Probe
from harbor_voice.perturbation.conversation import ProbeResult
from harbor_voice.simulator.simulator import UniversalSimulator
from harbor_voice.simulator.simulator import Utterance
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.drain import DEFAULT_IDLE_S
from harbor_voice.transport.drain import DrainStop
from harbor_voice.transport.pump import InboundPump

DEFAULT_TURN_TIMEOUT_S = 30.0
LINE_FRAME_S = 0.02
"""The line's pace: one frame every 20 ms for the whole call, caller speech
when there is some and silence otherwise, as a phone line carries. Agents
find the end of a turn by hearing the pause after it."""


@dataclass(slots=True)
class _Speech:
    """One caller utterance queued on the line, and what became of it."""

    caller: PartyId
    pending: list[tuple[AudioFrame, AudioFrame]]
    sent: list[AudioFrame] = field(default_factory=list)
    last_tx: float = 0.0
    done: asyncio.Event = field(default_factory=asyncio.Event)
    on_start: Callable[[], None] | None = None


_POLL_S = 0.005


class DuplexError(RuntimeError):
    """The driver was used in a way its contract does not allow."""


@dataclass(frozen=True, slots=True)
class _Injection:
    """One probe as it went out: where in the reply, and how much had arrived.

    ``frames_seen`` is the whole basis for telling a yield from a reply that
    had already finished — if no further frame ever arrives, nothing the agent
    did can be distinguished from it having stopped on its own.
    """

    probe: Probe
    at_reply_s: float
    frames_seen: int
    t_tx: float


@dataclass(frozen=True, slots=True)
class DuplexTurn:
    """One exchange, plus every probe that was aimed at it."""

    index: int
    utterance: Utterance
    received: tuple[AudioFrame, ...] = ()
    sent: tuple[AudioFrame, ...] = ()
    probes: tuple[ProbeResult, ...] = ()
    stages: tuple[StageMark, ...] = ()
    stop: DrainStop = DrainStop.EMPTY
    bled: tuple[AudioFrame, ...] = field(default=())
    party: PartyId = PartyId("agent")
    at_call_s: float = 0.0
    """Where this exchange starts on the call timeline."""

    def as_turn(self) -> Turn:
        """The same exchange in the shape the recorder and metrics expect.

        Without this a duplex run would produce probe events and no turns, so
        ``divergence.wer_said_heard``, ``timing.response_latency_s`` and
        ``speech.rate_wpm`` would all abstain on a call that has exactly the
        data they need. The duplex driver is a different way of running a
        turn, not a different kind of record.
        """
        return Turn(
            index=self.index,
            party=self.party,
            speaker=self.utterance.party,
            sent=self.sent,
            received=self.received,
            stages=self.stages,
            stop=self.stop,
            bled=self.bled,
        )

    @property
    def reply_audio_s(self) -> float:
        """Agent audio carried, summed from the frames themselves."""
        return sum(f.duration_s for f in self.received)

    @property
    def is_complete(self) -> bool:
        """Whether the reply ended on its own terms rather than our patience."""
        return self.stop in (DrainStop.IDLE, DrainStop.STREAM_END)

    @property
    def fired(self) -> tuple[ProbeResult, ...]:
        return tuple(r for r in self.probes if r.fired)

    @property
    def measurable(self) -> tuple[ProbeResult, ...]:
        """Probes that landed somewhere a conclusion can be drawn from."""
        return tuple(r for r in self.probes if r.is_measurable)

    @property
    def first_rx(self) -> float | None:
        for frame in self.received:
            t = frame.at(Clock.RX)
            if t is not None:
                return t
        return None

    @property
    def last_rx(self) -> float | None:
        for frame in reversed(self.received):
            t = frame.at(Clock.RX)
            if t is not None:
                return t
        return None


class DuplexDriver:
    """Runs a call where both parties may speak, and measures who yielded.

    Args:
        transport: The agent's frame stream.
        simulator: The caller. Supplies both content turns and probe audio,
            so a probe goes through the same synthesis and the same
            degradation as the turn it interrupts.
        party: The agent's party id, for floor bookkeeping.
        floor: The policy. Defaults to a :class:`BargeInFloor`, because a
            driver that injects interruptions and then cannot say whether they
            were answered is only making the audio harder.
        conversation: L5 layers. Defaults to the simulator's, which is where a
            profile puts them.
        clock: Session clock. Defaults to the simulator's, which is
            almost always what you want: two ``SessionClock`` instances start
            at different instants, so a driver with its own reads a smaller
            ``now()`` than the simulator did when it stamped ``t_produced``,
            and the frame's monotonic guard then rejects a transmit that
            looks like it preceded its own synthesis. That failure is
            timing-dependent -- it only appears when synthesis is quicker
            than the gap between the two constructors -- so it surfaces under
            load and nowhere else. Share it with the transport too, if the
            transport stamps its own frames.
        idle_s: Quiet period that ends the agent's turn.
        turn_timeout_s: Hard ceiling. A turn that hits it is a reply we gave
            up on, so any yield aimed at it stays unresolved rather than being
            recorded as an agent that eventually stopped.
    """

    def __init__(
        self,
        transport: VoiceTransport,
        *,
        simulator: UniversalSimulator,
        party: PartyId,
        floor: BargeInFloor | None = None,
        conversation: ConversationPlan | None = None,
        clock: SessionClock | None = None,
        idle_s: float = DEFAULT_IDLE_S,
        turn_timeout_s: float = DEFAULT_TURN_TIMEOUT_S,
    ) -> None:
        self._transport = transport
        self._simulator = simulator
        self._party = party
        self._floor = floor or BargeInFloor()
        self._conversation = conversation or simulator.pipeline.conversation
        self._clock = clock or simulator.clock
        self._idle_s = idle_s
        self._turn_timeout_s = turn_timeout_s
        self._pump = InboundPump(transport)
        self._queue: asyncio.Queue[_Speech] = asyncio.Queue()
        self._line_task: asyncio.Task[None] | None = None
        self._turn_index = 0
        self._call_s = 0.0

    # ------------------------------------------------------------- identity

    @property
    def floor(self) -> BargeInFloor:
        return self._floor

    @property
    def clock(self) -> SessionClock:
        return self._clock

    @property
    def conversation(self) -> ConversationPlan:
        return self._conversation

    @property
    def party(self) -> PartyId:
        return self._party

    @property
    def caller(self) -> PartyId:
        return self._simulator.party

    @property
    def call_s(self) -> float:
        """Current position on the call timeline, in seconds of conversation.

        Advances by the duration of the audio each party actually produced
        plus the agent's observed thinking time — never by how fast the
        harness managed to move bytes.
        """
        return self._call_s

    # ------------------------------------------------------------ lifecycle

    async def open(self, config: SessionConfig) -> None:
        """Open the transport and start draining it."""
        if not self._transport.capabilities.supports_duplex(Duplex.FULL):
            raise DuplexError(
                f"{type(self._transport).__name__} does not declare full duplex; "
                f"driving it from here would measure the harness, not the agent"
            )
        await self._transport.open(config)
        self._pump.start()
        self._line_task = asyncio.create_task(self._line())

    async def close(self) -> None:
        """Close the transport and stop the pump. Idempotent."""
        if self._line_task is not None:
            self._line_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._line_task
            self._line_task = None
        await self._transport.close()
        await self._pump.stop()

    # ----------------------------------------------------------------- turn

    async def exchange(self, utterance: Utterance) -> DuplexTurn:
        """Send one caller turn, drain the reply, inject this turn's probes."""
        self._pump.check()
        if not utterance.frames:
            raise DuplexError("exchange() needs an utterance with audio")

        index = self._turn_index
        self._turn_index += 1
        bled = tuple(self._from_wire(f) for f in self._pump.take_pending())
        started_at = self._call_s

        sent, last_tx = await self._speak(utterance.frames, bid=Bid.FLOOR)
        probes = self._conversation.schedule(index)
        received, stop, results = await self._drain(probes, last_tx=last_tx)

        self._conversation.observe(results)
        return DuplexTurn(
            index=index,
            utterance=utterance,
            received=received,
            sent=sent,
            probes=results,
            stages=self._mark_stages(received),
            stop=stop,
            bled=bled,
            party=self._party,
            at_call_s=started_at,
        )

    # -------------------------------------------------------------- speaking

    async def _speak(
        self,
        frames: Sequence[AudioFrame],
        *,
        bid: Bid,
        at_call_s: float | None = None,
        on_start: Callable[[], None] | None = None,
    ) -> tuple[tuple[AudioFrame, ...], float]:
        """Transmit the caller's audio and hold the floor for as long as it lasts.

        The frames carry real ``tx`` stamps, because those are a genuine flow
        point and the response-latency measurement needs them. The *floor*
        interval is the audio's own length: pushing eight seconds of speech
        into a queue in two milliseconds does not mean the caller spoke for
        two milliseconds.

        Returns:
            The stamped frames and the session time the last one left, which
            is one half of the agent's response latency.
        """
        caller = frames[0].party
        start = self._call_s if at_call_s is None else at_call_s
        self._floor.speech_start(caller, start, bid=bid)
        speech = _Speech(
            caller=caller,
            pending=list(zip(frames, self._to_wire(frames), strict=True)),
            last_tx=self._clock.now(),
            on_start=on_start,
        )
        await self._queue.put(speech)
        await self._until(speech.done)
        sent, last_tx = speech.sent, speech.last_tx
        spoken_s = sum(f.duration_s for f in frames)
        self._floor.speech_end(caller, start + spoken_s)
        if at_call_s is None:
            self._call_s = start + spoken_s
        return tuple(sent), last_tx

    # ------------------------------------------------------------ formats

    @property
    def _wire(self) -> Any:
        """The format the agent takes: its declared ``audio_in``."""
        return getattr(self._transport.capabilities, "audio_in", None) or CANONICAL

    def _to_wire(self, frames: Sequence[AudioFrame]) -> list[AudioFrame]:
        """The caller's frames in the agent's input format, one for one.

        The utterance is converted whole, so the resampler sees one signal
        rather than a click at every frame edge, then cut back at the
        original frames' boundaries.
        """
        wire = self._wire
        if all(f.fmt == wire for f in frames):
            return list(frames)
        fmt = frames[0].fmt
        if any(f.fmt != fmt for f in frames):
            return [
                dataclasses.replace(
                    f, pcm=convert.convert(f.pcm, f.fmt, wire), fmt=wire
                )
                for f in frames
            ]
        data = convert.convert(b"".join(f.pcm for f in frames), fmt, wire)
        step = wire.bytes_per_sample * wire.channels
        out: list[AudioFrame] = []
        elapsed, cut = 0.0, 0
        for f in frames:
            elapsed += f.duration_s
            end = min(len(data), int(round(elapsed * wire.sample_rate_hz)) * step)
            out.append(dataclasses.replace(f, pcm=data[cut:end], fmt=wire))
            cut = end
        return out

    def _from_wire(self, frame: AudioFrame) -> AudioFrame:
        """An agent frame in the recording format."""
        if frame.fmt == CANONICAL:
            return frame
        return dataclasses.replace(
            frame, pcm=convert.convert(frame.pcm, frame.fmt, CANONICAL), fmt=CANONICAL
        )

    # --------------------------------------------------------------- line

    async def _line(self) -> None:
        """Keep audio flowing to the agent in real time until the call closes.

        Queued caller speech goes out frame by frame at its own pace; between
        utterances the line carries silence in the agent's format. A frame
        the floor refuses (the caller yielding) is replaced by silence, so
        the line never stops.
        """
        loop = asyncio.get_running_loop()
        wire = self._wire
        quiet = convert.silence(wire, LINE_FRAME_S)
        due = loop.time()
        seq = 0
        speech: _Speech | None = None
        while True:
            if speech is None and not self._queue.empty():
                speech = self._queue.get_nowait()
            length = LINE_FRAME_S
            now = self._clock.now()
            if speech is not None and speech.pending:
                if speech.on_start is not None:
                    speech.on_start()
                    speech.on_start = None
                frame, on_wire = speech.pending.pop(0)
                length = frame.duration_s or LINE_FRAME_S
                if self._floor.may_transmit(speech.caller, now) is Decision.ALLOW:
                    await self._transport.tx(on_wire.stamp(Clock.TX, now))
                    speech.sent.append(frame.stamp(Clock.TX, now))
                    speech.last_tx = now
                else:
                    await self._transport.tx(self._quiet(quiet, wire, seq, now))
                if not speech.pending:
                    speech.done.set()
                    speech = None
            else:
                await self._transport.tx(self._quiet(quiet, wire, seq, now))
            seq += 1
            due += length
            await asyncio.sleep(max(0.0, due - loop.time()))

    def _quiet(self, pcm: bytes, wire: Any, seq: int, now: float) -> AudioFrame:
        return AudioFrame(
            pcm=pcm,
            seq=seq,
            party=self._simulator.party,
            t_produced=now,
            fmt=wire,
            filler=True,
        ).stamp(Clock.TX, now)

    async def _until(self, event: asyncio.Event) -> None:
        """Wait for ``event``; raise if the line died first."""
        line = self._line_task
        if line is None:
            raise DuplexError("the driver is not open")
        waiter = asyncio.create_task(event.wait())
        await asyncio.wait({waiter, line}, return_when=asyncio.FIRST_COMPLETED)
        if not event.is_set():
            waiter.cancel()
            error = line.exception() if line.done() else None
            if error is not None:
                raise error
            raise DuplexError("the line closed before the caller finished speaking")

    # -------------------------------------------------------------- draining

    async def _drain(
        self, probes: Sequence[Probe], *, last_tx: float
    ) -> tuple[tuple[AudioFrame, ...], DrainStop, tuple[ProbeResult, ...]]:
        """Collect the reply, firing probes as its audio passes their offsets.

        The floor is told about the agent's speech live — opened when the
        first frame arrives, closed when the reply ends — rather than
        retrospectively as :meth:`~harbor_voice.adapter.VoiceAgentAdapter.call`
        does. That is the whole difference: a yield record cannot be opened
        against a holder the floor does not yet know is holding.
        """
        received: list[AudioFrame] = []
        stream = self._pump.stream()

        voiced = 0

        async def _consume() -> None:
            # An agent that streams continuously (WebRTC) sends silence
            # between replies. Silence before the reply starts is not the
            # reply; the reply ends when the agent goes quiet, not when frames
            # stop arriving.
            nonlocal voiced
            async for frame in stream:
                frame = self._from_wire(frame)
                quiet = convert.is_quiet(frame.pcm)
                if quiet and not voiced:
                    continue
                received.append(frame)
                if not quiet:
                    voiced += 1

        task = asyncio.create_task(_consume())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._turn_timeout_s
        idle = min(self._idle_s, self._turn_timeout_s)

        pending = list(probes)
        fired: list[_Injection] = []

        holding = False
        agent_t0 = self._call_s
        seen = 0
        quiet_since = loop.time()
        stop = DrainStop.TIMEOUT
        try:
            while True:
                if task.done():
                    stop = DrainStop.STREAM_END
                    break
                now = loop.time()
                if now >= deadline:
                    stop = DrainStop.TIMEOUT
                    break
                if voiced != seen:
                    seen, quiet_since = voiced, now
                    if not holding:
                        holding = True
                        # The one thing the session clock is the right
                        # instrument for: how long the agent took to answer.
                        agent_t0 = self._call_s + max(0.0, self._clock.now() - last_tx)
                        self._floor.speech_start(self._party, agent_t0)
                elif received and now - quiet_since >= idle:
                    stop = DrainStop.IDLE
                    break

                reply_s = sum(f.duration_s for f in received)
                while pending and pending[0].at_s <= reply_s:
                    probe = pending.pop(0)
                    landed: list[tuple[float, int]] = []

                    def _landed() -> None:
                        # Where the reply had reached when the probe's first
                        # frame went out on the line, not when it was queued.
                        landed.append(
                            (sum(f.duration_s for f in received), len(received))
                        )

                    await self._inject(
                        probe, at_call_s=agent_t0 + reply_s, on_start=_landed
                    )
                    at_reply_s, seen = landed[0] if landed else (reply_s, len(received))
                    fired.append(
                        _Injection(
                            probe=probe,
                            at_reply_s=at_reply_s,
                            frames_seen=seen,
                            t_tx=agent_t0 + at_reply_s,
                        )
                    )
                    # The probe plays in real time; the agent's quiet period
                    # counts from when the caller stopped, not from before.
                    quiet_since = loop.time()
                    # Re-read: injecting awaits, so more frames may have landed
                    # and the next probe's offset may already have passed.
                    reply_s = sum(f.duration_s for f in received)

                await asyncio.sleep(_POLL_S)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if not received:
            stop = DrainStop.EMPTY

        reply_s = sum(f.duration_s for f in received)
        if holding:
            self._floor.speech_end(self._party, agent_t0 + reply_s)
        self._call_s = agent_t0 + reply_s

        return tuple(received), stop, self._results(probes, fired, received, stop)

    async def _inject(
        self,
        probe: Probe,
        *,
        at_call_s: float,
        on_start: Callable[[], None] | None = None,
    ) -> None:
        """Synthesise and transmit one probe, under its own bid.

        ``at_call_s`` pins the probe to the agent's speech rather than to the
        wall clock, so the floor records an overlap of the probe's real length
        at the position a listener would have heard it.
        """
        utterance = await self._simulator.probe_utterance(probe)
        if utterance.frames:
            await self._speak(
                utterance.frames, bid=probe.bid, at_call_s=at_call_s, on_start=on_start
            )

    def _results(
        self,
        probes: Sequence[Probe],
        fired: Sequence[_Injection],
        received: Sequence[AudioFrame],
        stop: DrainStop,
    ) -> tuple[ProbeResult, ...]:
        """Classify every probe, fired or not.

        ``reply_total_s`` stays ``None`` when the drain timed out: we stopped
        waiting before the agent did, so we do not know that it ever stopped,
        and ``yield_latency_s`` reads ``None`` for *never* rather than
        borrowing the timeout as a stand-in.
        """
        total_s = sum(f.duration_s for f in received)
        complete = stop in (DrainStop.IDLE, DrainStop.STREAM_END)
        injected = {id(i.probe): i for i in fired}
        deadline = self._floor.yield_within_s

        results: list[ProbeResult] = []
        for probe in probes:
            landed = injected.get(id(probe))
            if landed is None:
                landing = (
                    Landing.NO_REPLY if not received else Landing.REPLY_ALREADY_DONE
                )
                results.append(
                    ProbeResult(probe=probe, landing=landing, deadline_s=deadline)
                )
                continue

            landing = (
                Landing.DURING_SPEECH
                if len(received) > landed.frames_seen
                else Landing.INDISTINGUISHABLE
            )
            results.append(
                ProbeResult(
                    probe=probe,
                    landing=landing,
                    at_reply_s=landed.at_reply_s,
                    reply_total_s=total_s if complete else None,
                    t_injected=landed.t_tx,
                    deadline_s=deadline,
                )
            )
        return tuple(results)

    # -------------------------------------------------------------- stages

    def _mark_stages(self, received: Sequence[AudioFrame]) -> tuple[StageMark, ...]:
        """The transport's own seams if it has any, else the two from frames.

        A cascade knows when ASR finished and when the first token arrived;
        inferring two marks from frame arrivals would throw five of
        them away, and running a cascade full-duplex would silently lose the
        observability that made it the control condition in the first place.
        ``take_stages`` is the same seam
        :class:`~harbor_voice.cascade.adapter.CascadeAdapter` reads.
        """
        take = getattr(self._transport, "take_stages", None)
        if take is not None:
            own = tuple(take())
            if own:
                return own

        marks: list[StageMark] = []
        first = next((f.at(Clock.RX) for f in received if f.at(Clock.RX)), None)
        last = next(
            (f.at(Clock.RX) for f in reversed(received) if f.at(Clock.RX)), None
        )
        if first is not None:
            marks.append(
                StageMark(
                    stage=Stage.RESPONSE_START,
                    t=first,
                    party=self._party,
                    clock=Clock.RX,
                )
            )
        if last is not None:
            marks.append(
                StageMark(
                    stage=Stage.RESPONSE_DONE,
                    t=last,
                    party=self._party,
                    clock=Clock.RX,
                )
            )
        return tuple(marks)

    @override
    def __repr__(self) -> str:
        return (
            f"DuplexDriver(party={self._party!r}, floor={self._floor.name}, "
            f"layers={sorted(str(k) for k in self._conversation.active_layers)})"
        )
