"""The transport contract.

**The transport is always full-duplex.** Both directions stream continuously,
always. Half-duplex is a *floor policy* layered on top, not a second
interface.

An interface shaped as ``send_audio(whole_utterance)`` -> ``recv_audio()``
would force every full-duplex provider to fake turns with its own workaround;
streaming frames needs none.

Four abstract methods — :meth:`VoiceTransport.open`, :meth:`~VoiceTransport.tx`,
:meth:`~VoiceTransport.rx`, :meth:`~VoiceTransport.close`. Everything else is
**declared** in the capability matrix and checked by
:meth:`VoiceTransport.check_contract`, rather than discovered with ``getattr``
at the call site.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from typing import Any
from typing import NoReturn
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import OptionalOp
from harbor_voice.ids import PartyId
from harbor_voice.transport.events import ControlEvent
from harbor_voice.transport.stream import StreamEvent

DEFAULT_GAP_BUDGET_S = 1.0
"""Cumulative transport downtime before a run stops being evidence."""


class TransportError(RuntimeError):
    """Base for every transport failure."""


class TransportNotConnectedError(TransportError):
    """An operation needed an open session and there wasn't one."""


class TransportClosedError(TransportError):
    """The session ended and cannot be used again."""


class UnsupportedOperationError(TransportError):
    """An optional operation was called on a transport that does not declare it."""


class ContractError(TypeError):
    """A transport's declaration and its implementation disagree."""


class SessionConfig(BaseModel):
    """Everything needed to open one session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    party: PartyId
    audio_in: AudioFormat = CANONICAL
    audio_out: AudioFormat = CANONICAL
    max_session_s: Annotated[float, Field(gt=0)] | None = None
    gap_budget_s: Annotated[float, Field(ge=0)] = DEFAULT_GAP_BUDGET_S
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific settings, passed through untouched.",
    )

    @property
    def converts_input(self) -> bool:
        """Whether frames must be resampled on the way in."""
        return not self.audio_in.is_canonical

    @property
    def converts_output(self) -> bool:
        """Whether frames must be resampled on the way out."""
        return not self.audio_out.is_canonical


class VoiceTransport(ABC):
    """A continuous, bidirectional frame stream to one party.

    Subclasses implement four methods and declare their optional surface in
    :attr:`capabilities`. Calling an undeclared optional raises
    :class:`UnsupportedOperationError` from the base implementation, so a
    caller never silently gets a no-op.
    """

    # ----------------------------------------------------------- declaration

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        """What this transport can do. The single source of truth."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether a session is currently usable."""

    # -------------------------------------------------------------- required

    @abstractmethod
    async def open(self, config: SessionConfig) -> None:
        """Establish the session.

        Must not return until frames can be sent and received, so a caller
        never has to poll. A transport that cannot connect raises rather than
        succeeding and deferring the failure to first audio, as a stub that
        assigns a sentinel ``object()`` in ``connect()`` would.
        """

    @abstractmethod
    async def tx(self, frame: AudioFrame) -> None:
        """Send one frame. Fire and forget; must not block on a reply."""

    @abstractmethod
    def rx(self) -> AsyncIterator[AudioFrame]:
        """Yield inbound frames until the session closes.

        Always draining — the caller is expected to consume this concurrently
        with :meth:`tx`, not alternately with it.
        """

    @abstractmethod
    async def close(self) -> None:
        """End the session. Must be idempotent."""

    # -------------------------------------------- optional, declared not probed

    async def send_text(self, text: str) -> None:
        """Inject text into the session, where the provider accepts it."""
        self._unsupported(OptionalOp.SEND_TEXT)

    async def send_dtmf(self, digits: str) -> None:
        """Send DTMF tones. Telephony transports only."""
        self._unsupported(OptionalOp.SEND_DTMF)

    def control(self) -> AsyncIterator[ControlEvent]:
        """Yield non-audio events: provider VAD, barge-in, gaps, errors."""
        self._unsupported(OptionalOp.CONTROL_STREAM)

    def events(self) -> AsyncIterator[StreamEvent]:
        """Yield agent audio, transcript and provider VAD in arrival order.

        See :mod:`harbor_voice.transport.stream`. Shares an inbox with
        :meth:`rx`; consume one or the other.
        """
        self._unsupported(OptionalOp.EVENT_STREAM)

    async def truncate(self, utterance: str, audio_end_ms: int) -> None:
        """Tell the provider only ``audio_end_ms`` of ``utterance`` was heard."""
        self._unsupported(OptionalOp.TRUNCATE)

    # ----------------------------------------------------------------- guards

    def _unsupported(self, op: OptionalOp) -> NoReturn:
        """Raise, naming the operation and what this transport does declare."""
        raise UnsupportedOperationError(
            f"{type(self).__name__} does not implement {op.value}. "
            f"Declared optional ops: "
            f"{sorted(o.value for o in self.capabilities.optional_ops) or 'none'}"
        )

    def require_open(self) -> None:
        """Raise unless a session is usable. Call at the top of tx/rx paths."""
        if not self.is_open:
            raise TransportNotConnectedError(
                f"{type(self).__name__} has no open session; call open() first"
            )

    def check_contract(self) -> list[str]:
        """Return every disagreement between declaration and implementation.

        This is the mechanism that replaces ``getattr`` probing. A transport
        that declares :attr:`OptionalOp.SEND_TEXT` but never overrides
        :meth:`send_text` is a bug the conformance suite catches here,
        rather than a ``NotImplementedError`` thrown at minute six of a call.

        Returns:
            Human-readable problems; empty means declaration and code agree.
        """
        problems: list[str] = []
        cls = type(self)

        for op, method in _OPTIONAL_METHODS.items():
            declared = self.capabilities.supports(op)
            overridden = getattr(cls, method) is not getattr(VoiceTransport, method)
            if declared and not overridden:
                problems.append(
                    f"{cls.__name__} declares {op.value} but does not override "
                    f"{method}()"
                )
            elif overridden and not declared:
                problems.append(
                    f"{cls.__name__} overrides {method}() but does not declare "
                    f"{op.value}; callers will not know it is available"
                )

        if self.capabilities.supports(
            OptionalOp.RECONNECT
        ) and not self.capabilities.supports(OptionalOp.CONTROL_STREAM):
            problems.append(
                f"{cls.__name__} declares reconnect but not control_stream; "
                f"a gap that nothing reports cannot be budgeted"
            )
        return problems

    def raise_on_contract_violation(self) -> None:
        """Raise :class:`ContractError` if :meth:`check_contract` found anything."""
        problems = self.check_contract()
        if problems:
            raise ContractError("; ".join(problems))

    # ------------------------------------------------------- context manager

    @asynccontextmanager
    async def session(self, config: SessionConfig) -> AsyncIterator[Self]:
        """Open for the duration of the block, closing even on error.

        ``close()`` is required to be idempotent, so an exception during
        teardown inside the body cannot leave a socket dangling.
        """
        await self.open(config)
        try:
            yield self
        finally:
            await self.close()


_OPTIONAL_METHODS: dict[OptionalOp, str] = {
    OptionalOp.SEND_TEXT: "send_text",
    OptionalOp.SEND_DTMF: "send_dtmf",
    OptionalOp.CONTROL_STREAM: "control",
    OptionalOp.EVENT_STREAM: "events",
    OptionalOp.TRUNCATE: "truncate",
}
"""Optional ops that correspond to a method. ``RECONNECT`` is behavioural."""
