"""A caller that reacts every step, not every turn.

:class:`~harbor_voice.simulator.content.ContentSource` answers one question —
*what do I say next, given the sentences I heard?* — and that is enough for a
turn-based call. It is not enough for a full-duplex one. A caller in tau-voice
decides what to say from a record that knows *when* things were said: which
of its own lines were cut off, what it heard only half of, how long both sides
were silent. Collapsing that to ``list[str]`` throws away exactly what the
benchmark is about.

So a reactive source is handed the **step record** instead — what the caller
emitted, what it heard, and which tool results it received, per step — plus
whatever the agent is saying right now. What it does with that record is its
own business; tau-voice's source has tau2 render it, so the caller's LLM sees
what tau2's caller would.

The *when* stays out of here. A source is asked only when a turn policy
decides the caller should respond; it never decides timing itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from types import MappingProxyType
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from harbor_voice.simulator.content import SourceKind

_EMPTY: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class CallerToolCall:
    """A tool the caller itself calls — telecom's caller toggles its own phone."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default=_EMPTY)


@dataclass(frozen=True, slots=True)
class CallerToolResult:
    id: str
    content: str
    error: bool = False


@dataclass(frozen=True, slots=True)
class Heard:
    """Agent speech as the caller perceived it in one step."""

    text: str | None
    speech: bool


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One step of a reactive call, from the caller's side.

    Attributes:
        said_text: The slice of the caller's line that went out this step.
        said_speech: Whether caller audio carried speech this step.
        said_tools: Tool calls the caller made this step.
        heard: The agent's speech this step, if any.
        tool_results: Results of earlier caller tool calls, delivered this step.
    """

    step: int
    t_s: float
    said_text: str | None = None
    said_speech: bool = False
    said_tools: tuple[CallerToolCall, ...] = ()
    heard: Heard | None = None
    tool_results: tuple[CallerToolResult, ...] = ()


@dataclass(frozen=True, slots=True)
class Say:
    """Speak this line. Empty text means the source had nothing to say."""

    text: str


@dataclass(frozen=True, slots=True)
class UseTools:
    """Call these tools now; results arrive on a later step.

    ``text`` is whatever the model wrote alongside the calls. It is not
    spoken, but it is part of the record, as tau2 keeps it.
    """

    calls: tuple[CallerToolCall, ...]
    text: str | None = None


@dataclass(frozen=True, slots=True)
class HangUp:
    """The caller ends the call. ``text`` is what it produced, not spoken."""

    text: str
    reason: str


CallerReply = Say | UseTools | HangUp

Pending = Heard | tuple[CallerToolResult, ...] | None
"""What arrived this step and is not yet in the record: agent speech, or the
results of the caller's own tool calls."""


@runtime_checkable
class ReactiveContentSource(Protocol):
    """Decides what the caller says, from the step record."""

    @property
    def kind(self) -> SourceKind: ...

    @property
    def is_deterministic(self) -> bool: ...

    async def respond(
        self, steps: Sequence[StepRecord], pending: Pending
    ) -> CallerReply:
        """The caller's next move, given everything so far.

        Args:
            steps: The record so far, oldest first.
            pending: What arrived this step and no step has recorded yet.
        """
        ...

    async def run_tools(
        self, calls: Sequence[CallerToolCall], *, step: int
    ) -> tuple[CallerToolResult, ...]:
        """Execute the caller's tool calls made at ``step``."""
        ...
