"""The agent adapter.

Division of labour: **the base owns everything hard, adapters stay thin.**

The base owns drain-to-tail-silence, floor integration, stage marking, timing
**from real flow points**, and reconciliation of audio that bled across a turn
boundary. An adapter owns four transport methods and a capability declaration.

On timing: every duration here is built from an :class:`Interval`, which cannot
be constructed without naming both clock domains. Nothing is
back-computed from byte length, and a first turn's *full response latency*
cannot be reported under a time-to-first-byte name: that class of mistake is
unavailable here.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import override

from harbor_voice.audio.clock import Clock
from harbor_voice.audio.clock import Interval
from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.audio.stages import StageMark
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Pipeline
from harbor_voice.floor import BaseFloor
from harbor_voice.floor import Decision
from harbor_voice.floor import FullDuplexFloor
from harbor_voice.ids import PartyId
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.drain import DEFAULT_IDLE_S
from harbor_voice.transport.drain import DrainStop
from harbor_voice.transport.drain import drain_until_idle
from harbor_voice.transport.pump import InboundPump
from harbor_voice.transport.pump import PumpError

DEFAULT_TURN_TIMEOUT_S = 30.0


@dataclass(frozen=True, slots=True)
class Turn:
    """One exchange: what went out, what came back, and when.

    Every timing property returns an :class:`Interval` so the clock domains
    travel with the number.
    """

    index: int
    party: PartyId
    """The party this adapter speaks for — the one that *replied*."""

    speaker: PartyId
    """The party whose audio was sent in."""

    sent: tuple[AudioFrame, ...] = ()
    received: tuple[AudioFrame, ...] = ()
    stages: tuple[StageMark, ...] = ()
    stop: DrainStop = DrainStop.EMPTY
    bled: tuple[AudioFrame, ...] = field(default=())
    """Frames that arrived before this turn's input finished sending.

    Audio from the previous reply that crossed the turn boundary. Attributing
    it to this turn would inflate this turn's latency and shorten the last
    one's; keeping it separate is the only way either number is right.
    """

    dropped: int = 0
    queued: int = 0

    @property
    def is_complete(self) -> bool:
        """Whether the reply ended on its own terms rather than our patience."""
        return self.stop in (DrainStop.IDLE, DrainStop.STREAM_END)

    @property
    def last_tx(self) -> float | None:
        for frame in reversed(self.sent):
            t = frame.at(Clock.TX)
            if t is not None:
                return t
        return None

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

    @property
    def response_latency(self) -> Interval | None:
        """Last outbound frame to first inbound frame. ``tx_to_rx``.

        ``None`` when either end is missing, rather than a zero that would be
        averaged into a published number.
        """
        start, end = self.last_tx, self.first_rx
        if start is None or end is None:
            return None
        return Interval(start, end, Clock.TX, Clock.RX)

    @property
    def reply_span(self) -> Interval | None:
        """First to last inbound frame. ``rx_to_rx``."""
        start, end = self.first_rx, self.last_rx
        if start is None or end is None:
            return None
        return Interval(start, end, Clock.RX, Clock.RX)

    @property
    def reply_audio_s(self) -> float:
        """Audio actually carried, summed from the frames."""
        return sum(f.duration_s for f in self.received)

    def stage_at(self, stage: Stage) -> float | None:
        """Timestamp of the first mark for ``stage``, if emitted."""
        for mark in self.stages:
            if mark.stage is stage:
                return mark.t
        return None


class AdapterError(RuntimeError):
    """The adapter was used in a way its contract does not allow."""


class VoiceAgentAdapter(ABC):
    """Turn-level semantics over a frame transport.

    ``call()`` is the turn-shaped convenience: send a party's audio, drain the
    reply, return a :class:`Turn`. A genuinely full-duplex driver uses the
    transport and the floor directly and does not go through here — which is
    the point of keeping duplex in the floor rather than the interface.
    """

    def __init__(
        self,
        transport: VoiceTransport,
        *,
        party: PartyId,
        floor: BaseFloor | None = None,
        clock: SessionClock | None = None,
        idle_s: float = DEFAULT_IDLE_S,
        turn_timeout_s: float = DEFAULT_TURN_TIMEOUT_S,
    ) -> None:
        self._transport = transport
        self._party = party
        self._floor = floor or FullDuplexFloor()
        self._clock = clock or SessionClock()
        self._idle_s = idle_s
        self._turn_timeout_s = turn_timeout_s
        self._turn_index = 0
        self._pump = InboundPump(transport)

    # ------------------------------------------------------------- identity

    @property
    @abstractmethod
    def pipeline(self) -> Pipeline:
        """Cascade or speech-to-speech. Orthogonal to duplex."""

    @property
    def capabilities(self) -> Capabilities:
        """Defaults to the transport's declaration."""
        return self._transport.capabilities

    @property
    def party(self) -> PartyId:
        return self._party

    @property
    def floor(self) -> BaseFloor:
        return self._floor

    @property
    def clock(self) -> SessionClock:
        return self._clock

    @property
    def transport(self) -> VoiceTransport:
        return self._transport

    # ------------------------------------------------------------ lifecycle

    async def open(self, config: SessionConfig) -> None:
        """Open the transport and start the inbound stream."""
        if self.pipeline is not self.capabilities.pipeline:
            raise AdapterError(
                f"{type(self).__name__} declares pipeline={self.pipeline.value} "
                f"but its capabilities say {self.capabilities.pipeline.value}"
            )
        await self._transport.open(config)
        self._pump.start()

    async def close(self) -> None:
        """Close the transport and stop the pump. Idempotent."""
        await self._transport.close()
        await self._pump.stop()

    # ----------------------------------------------------------------- turn

    async def call(self, frames: Sequence[AudioFrame]) -> Turn:
        """Send one party's audio and drain this party's reply.

        Raises:
            AdapterError: if called before :meth:`open`.
        """
        if not self._pump.is_running:
            raise AdapterError("call() before open(); no inbound stream")
        try:
            self._pump.check()
        except PumpError as exc:
            raise AdapterError(str(exc)) from exc
        if not frames:
            raise AdapterError("call() needs at least one frame to send")

        speaker = frames[0].party
        if any(f.party != speaker for f in frames):
            raise AdapterError("all frames in one call() must be from one party")

        index = self._turn_index
        self._turn_index += 1

        bled = self._pump.take_pending()

        sent, dropped, queued = await self._send(speaker, frames)
        result = await drain_until_idle(
            self._pump.stream(),
            idle_s=self._idle_s,
            timeout_s=self._turn_timeout_s,
        )

        stages = self._mark_stages(result.first_rx, result.last_rx)
        self._record_reply_on_floor(result.first_rx, result.last_rx)

        turn = Turn(
            index=index,
            party=self._party,
            speaker=speaker,
            sent=sent,
            received=result.frames,
            stages=stages,
            stop=result.stop,
            bled=bled,
            dropped=dropped,
            queued=queued,
        )
        await self._on_turn_end(turn)
        return turn

    # -------------------------------------------------------------- helpers

    async def _send(
        self, speaker: PartyId, frames: Sequence[AudioFrame]
    ) -> tuple[tuple[AudioFrame, ...], int, int]:
        """Transmit under the floor policy, stamping a real tx time per frame."""
        self._floor.speech_start(speaker, self._clock.now())
        sent: list[AudioFrame] = []
        held: list[AudioFrame] = []
        dropped = 0

        for frame in frames:
            now = self._clock.now()
            decision = self._floor.may_transmit(speaker, now)
            if decision is Decision.ALLOW:
                stamped = frame.stamp(Clock.TX, now)
                await self._transport.tx(stamped)
                sent.append(stamped)
            elif decision is Decision.QUEUE:
                held.append(frame)
            else:
                dropped += 1

        queued = len(held)
        for frame in held:
            now = self._clock.now()
            if self._floor.may_transmit(speaker, now) is Decision.ALLOW:
                stamped = frame.stamp(Clock.TX, now)
                await self._transport.tx(stamped)
                sent.append(stamped)
            else:
                dropped += 1

        self._floor.speech_end(speaker, self._clock.now())
        return tuple(sent), dropped, queued

    def _mark_stages(
        self, first_rx: float | None, last_rx: float | None
    ) -> tuple[StageMark, ...]:
        """Emit the two stages every pipeline has.

        A cascade adapter overrides this to add the five it can also see.
        """
        marks: list[StageMark] = []
        if first_rx is not None:
            marks.append(
                StageMark(
                    stage=Stage.RESPONSE_START,
                    t=first_rx,
                    party=self._party,
                    clock=Clock.RX,
                )
            )
        if last_rx is not None:
            marks.append(
                StageMark(
                    stage=Stage.RESPONSE_DONE,
                    t=last_rx,
                    party=self._party,
                    clock=Clock.RX,
                )
            )
        return tuple(marks)

    def _record_reply_on_floor(
        self, first_rx: float | None, last_rx: float | None
    ) -> None:
        """Tell the floor about the reply, using real receive times."""
        if first_rx is None or last_rx is None:
            return
        self._floor.speech_start(self._party, first_rx)
        self._floor.speech_end(self._party, last_rx)

    async def _on_turn_end(self, turn: Turn) -> None:
        """Hook for subclasses and the recorder."""

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(party={self._party!r}, "
            f"pipeline={self.pipeline.value}, turns={self._turn_index})"
        )
