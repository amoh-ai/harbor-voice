"""Control-plane events a transport surfaces alongside the audio stream.

Audio flows through :meth:`~harbor_voice.transport.base.VoiceTransport.rx`.
Everything that is *not* audio — the provider's own VAD, a barge-in it detected,
a socket that dropped — flows here, so neither stream has to carry a sentinel
or a union type.

The gap budget implements the second half of reconnection: a reconnecting
transport does not fail a run, but a run that spent too long disconnected is
not evidence about the agent, so it becomes ``measurement_invalid`` rather
than a failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from typing import Mapping
from typing import override

from harbor_voice.ids import PartyId

_EMPTY_META: Mapping[str, Any] = MappingProxyType({})


class ControlEventKind(StrEnum):
    """What a transport is telling us about, other than audio."""

    SPEECH_START = "speech_start"
    """Provider-side VAD opened. Only from a party with native turn control."""

    SPEECH_END = "speech_end"
    """Provider-side VAD closed."""

    INTERRUPTED = "interrupted"
    """The provider reports it was barged in on and stopped speaking."""

    TRANSPORT_GAP = "transport_gap"
    """The connection dropped. ``duration_s`` is set once it comes back."""

    PROVIDER_ERROR = "provider_error"
    """The provider reported an error that did not end the session."""

    PEER_CLOSED = "peer_closed"
    """The far end closed the session."""


@dataclass(frozen=True, slots=True)
class ControlEvent:
    """One non-audio event, timestamped in the session clock."""

    kind: ControlEventKind
    t: float
    party: PartyId
    duration_s: float | None = None
    meta: Mapping[str, Any] = field(default=_EMPTY_META)

    def __post_init__(self) -> None:
        if self.t < 0:
            raise ValueError(f"{self.kind.value} has negative time {self.t}")
        if self.duration_s is not None and self.duration_s < 0:
            raise ValueError(
                f"{self.kind.value} has negative duration {self.duration_s}"
            )
        if self.kind is ControlEventKind.TRANSPORT_GAP and self.duration_s is None:
            raise ValueError(
                "a transport_gap must carry duration_s; an unmeasured gap "
                "cannot be budgeted"
            )

    @override
    def __repr__(self) -> str:
        tail = f" {self.duration_s:.3f}s" if self.duration_s is not None else ""
        return f"ControlEvent({self.kind.value} @ {self.t:.3f}s {self.party!r}{tail})"


class GapBudgetExceeded(RuntimeError):
    """Cumulative transport downtime passed the budget.

    The caller turns this into ``measurement_invalid``, not a failed trial: the
    agent was not given a fair run, so the result is not evidence about it.
    """


@dataclass(slots=True)
class GapTracker:
    """Accumulates transport downtime against a budget.

    Deliberately mutable and deliberately dumb: it counts, it does not decide.
    The trial decides what to do when :meth:`exceeded` turns true.
    """

    budget_s: float
    total_s: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if self.budget_s < 0:
            raise ValueError(f"budget_s cannot be negative: {self.budget_s}")

    def record(self, event: ControlEvent) -> None:
        """Add a gap event. Ignores every other kind."""
        if event.kind is not ControlEventKind.TRANSPORT_GAP:
            return
        if event.duration_s is None:  # pragma: no cover - guarded in ControlEvent
            raise ValueError("transport_gap without duration_s")
        self.total_s += event.duration_s
        self.count += 1

    @property
    def exceeded(self) -> bool:
        """Whether cumulative downtime has passed the budget."""
        return self.total_s > self.budget_s

    def raise_if_exceeded(self) -> None:
        """Raise :class:`GapBudgetExceeded` once the budget is gone."""
        if self.exceeded:
            raise GapBudgetExceeded(
                f"transport was down {self.total_s:.3f}s across {self.count} "
                f"gap(s), budget is {self.budget_s:.3f}s"
            )
