"""Duplex as an enforced, measured policy.

The transport is always bidirectional. Half-duplex is not a second
interface; it is a policy that says who may transmit, applied over a
transport that could carry both directions at once.

Three things that buys, none of which an interface split does:

1. Floor violations become a metric. "This agent talked over the caller 11
   times" is a finding, not a crash.
2. You can run a half-duplex-designed agent under a full-duplex floor and
   measure what breaks. That is an experiment, not a configuration error.
3. One code path. A turn-shaped interface forces each full-duplex
   provider to fake the other mode with its own workaround: background loops
   with terminal sentinels, a continuous 20 ms mic pump, AAD disabled plus
   explicit activity markers.

A floor counts; it does not decide what a count means. The trial turns
:attr:`BaseFloor.violations` and :attr:`BaseFloor.overlaps` into metrics, and
:class:`BargeInFloor` into yield latencies.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import override

from harbor_voice.capabilities import Duplex
from harbor_voice.ids import PartyId


class Decision(StrEnum):
    """What a party may do with a frame right now."""

    ALLOW = "allow"
    """Transmit it."""

    QUEUE = "queue"
    """Hold it until the floor frees. The caller owns the buffer."""

    DROP = "drop"
    """Discard it. Models a party that is physically muted."""


class Contention(StrEnum):
    """What a half-duplex floor does when a party speaks out of turn."""

    QUEUE = "queue"
    DROP = "drop"


class Bid(StrEnum):
    """Whether speaking over the holder is a claim on the floor.

    Conversation analysis has had this distinction for fifty years and voice
    benchmarks keep collapsing it. "Mm-hm" and "wait, stop" are both audio
    arriving while the agent talks, but they ask for opposite things: the
    first says *keep going, I'm listening*, the second says *stop*. An agent
    that yields to both is as wrong as one that yields to neither.

    One switch, two metric families. A :attr:`FLOOR` start opens a yield
    record and feeds ``turns.barge_in_success_rate``; a :attr:`BACKCHANNEL`
    start does not, and feeds the selectivity family, where *not* yielding is
    the success. Overlap is counted for both, because the overlap is real.
    """

    FLOOR = "floor"
    """A claim on the floor. The holder is expected to stop."""

    BACKCHANNEL = "backchannel"
    """Not a claim. The holder is expected to keep going."""


@dataclass(frozen=True, slots=True)
class OverlapRecord:
    """A period during which more than one party had audio on the wire."""

    parties: tuple[PartyId, ...]
    t_start: float
    t_end: float | None = None

    @property
    def duration_s(self) -> float | None:
        """Length, or ``None`` while still open."""
        return None if self.t_end is None else self.t_end - self.t_start

    @property
    def is_open(self) -> bool:
        return self.t_end is None


@dataclass(frozen=True, slots=True)
class ViolationRecord:
    """A party tried to transmit while another held the floor.

    One record per contiguous blocked period, not per frame — at 50 frames a
    second, per-frame records would drown the signal they exist to carry.
    """

    party: PartyId
    holder: PartyId
    t_start: float
    t_end: float | None = None
    decision: Decision = Decision.QUEUE

    @property
    def duration_s(self) -> float | None:
        return None if self.t_end is None else self.t_end - self.t_start


@dataclass(frozen=True, slots=True)
class YieldRecord:
    """A party kept speaking after being barged in on, and how long for."""

    holder: PartyId
    interrupter: PartyId
    t_interrupted: float
    t_yielded: float | None = None
    deadline_s: float = 0.0

    @property
    def latency_s(self) -> float | None:
        """How long the holder took to stop, or ``None`` if it never did."""
        return None if self.t_yielded is None else self.t_yielded - self.t_interrupted

    @property
    def yielded_in_time(self) -> bool | None:
        """``None`` while unresolved; otherwise whether it met the deadline."""
        latency = self.latency_s
        return None if latency is None else latency <= self.deadline_s


class FloorError(RuntimeError):
    """A floor was driven into an impossible state."""


class BaseFloor(ABC):
    """Bookkeeping shared by every policy. Subclasses decide; this counts."""

    def __init__(self) -> None:
        self._speaking: dict[PartyId, float] = {}
        self._bids: dict[PartyId, Bid] = {}
        self._overlaps: list[OverlapRecord] = []
        self._violations: list[ViolationRecord] = []
        self._blocked: dict[PartyId, int] = {}

    # ------------------------------------------------------------- identity

    @property
    @abstractmethod
    def duplex(self) -> Duplex:
        """Which duplex mode this policy implements."""

    @property
    def name(self) -> str:
        return type(self).__name__

    # ---------------------------------------------------------------- state

    @property
    def speaking(self) -> frozenset[PartyId]:
        """Parties with audio on the wire right now."""
        return frozenset(self._speaking)

    @property
    def holder(self) -> PartyId | None:
        """The party that has held the floor longest, if any."""
        if not self._speaking:
            return None
        return min(self._speaking, key=lambda p: self._speaking[p])

    @property
    def overlaps(self) -> tuple[OverlapRecord, ...]:
        return tuple(self._overlaps)

    @property
    def violations(self) -> tuple[ViolationRecord, ...]:
        return tuple(self._violations)

    # --------------------------------------------------------------- events

    def speech_start(self, party: PartyId, t: float, *, bid: Bid = Bid.FLOOR) -> None:
        """Record that ``party`` began transmitting.

        Args:
            party: Who started.
            t: Session time.
            bid: Whether this is a claim on the floor. Defaults to
                :attr:`Bid.FLOOR`, because an unannotated start is an ordinary
                turn and treating it as a backchannel would silently drop
                interruptions out of the barge-in denominator.
        """
        if party in self._speaking:
            raise FloorError(
                f"{party!r} is already speaking since {self._speaking[party]}"
            )
        was_busy = bool(self._speaking)
        self._speaking[party] = t
        self._bids[party] = bid
        if was_busy:
            self._open_overlap(t)
        self._on_speech_start(party, t, bid)

    def speech_end(self, party: PartyId, t: float) -> None:
        """Record that ``party`` stopped transmitting."""
        if party not in self._speaking:
            raise FloorError(f"{party!r} is not speaking")
        start = self._speaking.pop(party)
        self._bids.pop(party, None)
        if t < start:
            raise FloorError(f"{party!r} ended at {t} before starting at {start}")
        if len(self._speaking) < 2:
            self._close_overlap(t)
        self._close_violation(party, t)
        self._on_speech_end(party, t)

    def _on_speech_start(self, party: PartyId, t: float, bid: Bid) -> None:
        """Hook for subclasses."""

    def _on_speech_end(self, party: PartyId, t: float) -> None:
        """Hook for subclasses."""

    # ------------------------------------------------------------- decision

    @abstractmethod
    def may_transmit(self, party: PartyId, t: float) -> Decision:
        """Whether ``party`` may put a frame on the wire at ``t``."""

    # -------------------------------------------------------------- helpers

    def _open_overlap(self, t: float) -> None:
        if self._overlaps and self._overlaps[-1].is_open:
            return
        self._overlaps.append(
            OverlapRecord(parties=tuple(sorted(self._speaking)), t_start=t)
        )

    def _close_overlap(self, t: float) -> None:
        if not self._overlaps or not self._overlaps[-1].is_open:
            return
        last = self._overlaps[-1]
        self._overlaps[-1] = OverlapRecord(
            parties=last.parties, t_start=last.t_start, t_end=t
        )

    def _record_violation(
        self, party: PartyId, holder: PartyId, t: float, decision: Decision
    ) -> None:
        """Open a violation unless one is already open for this party."""
        if party in self._blocked:
            return
        self._blocked[party] = len(self._violations)
        self._violations.append(
            ViolationRecord(party=party, holder=holder, t_start=t, decision=decision)
        )

    def _close_violation(self, party: PartyId, t: float) -> None:
        index = self._blocked.pop(party, None)
        if index is None:
            return
        open_record = self._violations[index]
        self._violations[index] = ViolationRecord(
            party=open_record.party,
            holder=open_record.holder,
            t_start=open_record.t_start,
            t_end=t,
            decision=open_record.decision,
        )

    def _clear_block(self, party: PartyId, t: float) -> None:
        """Close a violation because the floor freed, not because speech ended."""
        self._close_violation(party, t)

    @override
    def __repr__(self) -> str:
        return (
            f"{self.name}(speaking={sorted(self._speaking)}, "
            f"overlaps={len(self._overlaps)}, violations={len(self._violations)})"
        )


class FullDuplexFloor(BaseFloor):
    """Both parties may transmit at any time. Overlaps are recorded, not refused."""

    @property
    @override
    def duplex(self) -> Duplex:
        return Duplex.FULL

    @override
    def may_transmit(self, party: PartyId, t: float) -> Decision:
        return Decision.ALLOW


class HalfDuplexFloor(BaseFloor):
    """One party at a time. An attempt to speak over the holder is counted.

    The attempt is a *measurement*, not an error: a half-duplex-designed agent
    that tries to interrupt is telling you something about itself.
    """

    def __init__(self, *, on_contention: Contention = Contention.QUEUE) -> None:
        super().__init__()
        self._on_contention = on_contention

    @property
    @override
    def duplex(self) -> Duplex:
        return Duplex.HALF

    @property
    def on_contention(self) -> Contention:
        return self._on_contention

    @override
    def may_transmit(self, party: PartyId, t: float) -> Decision:
        holder = self.holder
        if holder is None or holder == party:
            self._clear_block(party, t)
            return Decision.ALLOW
        decision = (
            Decision.QUEUE if self._on_contention is Contention.QUEUE else Decision.DROP
        )
        self._record_violation(party, holder, t, decision)
        return decision


class BargeInFloor(BaseFloor):
    """Full duplex, with an explicit rule for who yields and how fast.

    When a party starts while another holds the floor, the holder is expected
    to stop within ``yield_within_s``. Whether it did, and how long it took, is
    the measurement — it is the difference between an agent that handles
    interruption and one that merely survives it.
    """

    def __init__(self, *, yield_within_s: float = 0.4) -> None:
        super().__init__()
        if yield_within_s < 0:
            raise ValueError(f"yield_within_s cannot be negative: {yield_within_s}")
        self._yield_within_s = yield_within_s
        self._yields: list[YieldRecord] = []
        self._pending: dict[PartyId, int] = {}

    @property
    @override
    def duplex(self) -> Duplex:
        return Duplex.FULL

    @property
    def yield_within_s(self) -> float:
        return self._yield_within_s

    @property
    def yields(self) -> tuple[YieldRecord, ...]:
        """Every barge-in, resolved or not."""
        return tuple(self._yields)

    @property
    def unresolved_yields(self) -> tuple[YieldRecord, ...]:
        """Barge-ins where the holder never stopped."""
        return tuple(y for y in self._yields if y.t_yielded is None)

    @property
    def late_yields(self) -> tuple[YieldRecord, ...]:
        """Barge-ins the holder answered, but too slowly."""
        return tuple(y for y in self._yields if y.yielded_in_time is False)

    @override
    def may_transmit(self, party: PartyId, t: float) -> Decision:
        return Decision.ALLOW

    @override
    def _on_speech_start(self, party: PartyId, t: float, bid: Bid) -> None:
        if bid is not Bid.FLOOR:
            # A backchannel is not a request to stop, so there is nothing to
            # be late for. Opening a yield record here would count every
            # correctly-ignored "mm-hm" as a failed barge-in — the exact
            # inversion the selectivity family exists to measure.
            return
        for holder in self._speaking:
            if holder == party or holder in self._pending:
                continue
            self._pending[holder] = len(self._yields)
            self._yields.append(
                YieldRecord(
                    holder=holder,
                    interrupter=party,
                    t_interrupted=t,
                    deadline_s=self._yield_within_s,
                )
            )

    @override
    def _on_speech_end(self, party: PartyId, t: float) -> None:
        index = self._pending.pop(party, None)
        if index is None:
            return
        pending = self._yields[index]
        self._yields[index] = YieldRecord(
            holder=pending.holder,
            interrupter=pending.interrupter,
            t_interrupted=pending.t_interrupted,
            t_yielded=t,
            deadline_s=pending.deadline_s,
        )


def floor_for(duplex: Duplex) -> BaseFloor:
    """Default policy for a duplex mode, for callers that need no tuning."""
    return HalfDuplexFloor() if duplex is Duplex.HALF else FullDuplexFloor()
