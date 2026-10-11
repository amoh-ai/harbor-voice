"""Unit tests for the transport contract."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import override

import pytest

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import OptionalOp
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.ids import PartyId
from harbor_voice.transport import ContractError
from harbor_voice.transport import ControlEvent
from harbor_voice.transport import ControlEventKind
from harbor_voice.transport import GapBudgetExceeded
from harbor_voice.transport import GapTracker
from harbor_voice.transport import SessionConfig
from harbor_voice.transport import TransportNotConnectedError
from harbor_voice.transport import UnsupportedOperationError
from harbor_voice.transport import VoiceTransport

pytestmark = pytest.mark.unit

AGENT = PartyId("agent")


def _caps(**over: object) -> Capabilities:
    base: dict[str, object] = {
        "duplex": frozenset({Duplex.FULL}),
        "pipeline": Pipeline.S2S,
        "turn_control": TurnControl.NATIVE_VAD,
        "exposes_stages": frozenset({Stage.RESPONSE_START, Stage.RESPONSE_DONE}),
    }
    return Capabilities(**(base | over))  # type: ignore[arg-type]


class FakeTransport(VoiceTransport):
    """Minimal conforming transport: the four required methods and nothing else."""

    def __init__(self, capabilities: Capabilities | None = None) -> None:
        self._caps = capabilities or _caps()
        self._open = False
        self.sent: list[AudioFrame] = []
        self.inbound: list[AudioFrame] = []
        self.close_calls = 0

    @property
    @override
    def capabilities(self) -> Capabilities:
        return self._caps

    @property
    @override
    def is_open(self) -> bool:
        return self._open

    @override
    async def open(self, config: SessionConfig) -> None:
        self._open = True

    @override
    async def tx(self, frame: AudioFrame) -> None:
        self.require_open()
        self.sent.append(frame)

    @override
    def rx(self) -> AsyncIterator[AudioFrame]:
        async def _gen() -> AsyncIterator[AudioFrame]:
            for f in self.inbound:
                yield f

        return _gen()

    @override
    async def close(self) -> None:
        self._open = False
        self.close_calls += 1


def _frame(seq: int = 0) -> AudioFrame:
    return AudioFrame(pcm=b"\x00" * 960, seq=seq, party=AGENT, t_produced=0.0)


# ------------------------------------------------------------ session config


class TestSessionConfig:
    def test_defaults_are_canonical(self) -> None:
        cfg = SessionConfig(party=AGENT)
        assert cfg.audio_in.is_canonical
        assert not cfg.converts_input
        assert not cfg.converts_output

    def test_non_canonical_format_is_flagged_as_a_conversion(self) -> None:
        """A resample is a declared channel event, not an accident."""
        cfg = SessionConfig(
            party=AGENT, audio_in=AudioFormat(sample_rate_hz=8_000, encoding="mulaw")
        )
        assert cfg.converts_input
        assert not cfg.converts_output

    def test_is_frozen_and_rejects_typos(self) -> None:
        cfg = SessionConfig(party=AGENT)
        with pytest.raises(Exception):
            cfg.party = PartyId("other")  # type: ignore[misc]
        with pytest.raises(Exception):
            SessionConfig(party=AGENT, gap_budget=2.0)  # type: ignore[call-arg]

    def test_provider_options_pass_through(self) -> None:
        cfg = SessionConfig(party=AGENT, options={"voice": "alloy"})
        assert cfg.options["voice"] == "alloy"


# ----------------------------------------------------------------- required


class TestRequiredSurface:
    async def test_open_tx_close(self) -> None:
        t = FakeTransport()
        await t.open(SessionConfig(party=AGENT))
        assert t.is_open
        await t.tx(_frame())
        assert len(t.sent) == 1
        await t.close()
        assert not t.is_open

    async def test_tx_before_open_is_refused(self) -> None:
        t = FakeTransport()
        with pytest.raises(TransportNotConnectedError, match="no open session"):
            await t.tx(_frame())

    async def test_rx_is_an_async_iterator(self) -> None:
        t = FakeTransport()
        t.inbound = [_frame(0), _frame(1)]
        got = [f async for f in t.rx()]
        assert [f.seq for f in got] == [0, 1]

    async def test_session_closes_on_the_way_out(self) -> None:
        t = FakeTransport()
        async with t.session(SessionConfig(party=AGENT)) as s:
            assert s.is_open
        assert not t.is_open
        assert t.close_calls == 1

    async def test_session_closes_even_when_the_body_raises(self) -> None:
        t = FakeTransport()
        with pytest.raises(ZeroDivisionError):
            async with t.session(SessionConfig(party=AGENT)):
                raise ZeroDivisionError
        assert not t.is_open
        assert t.close_calls == 1


# ------------------------------------------- optional ops: declared, not probed


class TestOptionalOps:
    async def test_undeclared_optional_raises_and_names_what_is_available(self) -> None:
        t = FakeTransport()
        with pytest.raises(UnsupportedOperationError) as e:
            await t.send_text("hello")
        assert "does not implement send_text" in str(e.value)
        assert "none" in str(e.value)

    async def test_error_lists_declared_ops(self) -> None:
        class WithText(FakeTransport):
            @override
            async def send_text(self, text: str) -> None:
                return None

        t = WithText(_caps(optional_ops=frozenset({OptionalOp.SEND_TEXT})))
        with pytest.raises(UnsupportedOperationError) as e:
            await t.send_dtmf("1")
        assert "send_text" in str(e.value)

    def test_control_stream_is_optional_too(self) -> None:
        with pytest.raises(UnsupportedOperationError, match="control_stream"):
            FakeTransport().control()

    def test_capabilities_supports_is_the_question_to_ask(self) -> None:
        t = FakeTransport(_caps(optional_ops=frozenset({OptionalOp.SEND_TEXT})))
        assert t.capabilities.supports(OptionalOp.SEND_TEXT)
        assert not t.capabilities.supports(OptionalOp.SEND_DTMF)


# ------------------------------------------------------------ contract check


class TestContractCheck:
    def test_minimal_transport_is_consistent(self) -> None:
        assert FakeTransport().check_contract() == []

    def test_declared_but_not_implemented_is_caught(self) -> None:
        """The failure mode `getattr` probing hides until minute six of a call."""
        t = FakeTransport(_caps(optional_ops=frozenset({OptionalOp.SEND_TEXT})))
        problems = t.check_contract()
        assert len(problems) == 1
        assert "declares send_text but does not override send_text()" in problems[0]

    def test_implemented_but_not_declared_is_caught(self) -> None:
        class Quiet(FakeTransport):
            @override
            async def send_dtmf(self, digits: str) -> None:
                return None

        problems = Quiet().check_contract()
        assert len(problems) == 1
        assert "does not declare send_dtmf" in problems[0]

    def test_declared_and_implemented_is_clean(self) -> None:
        class Good(FakeTransport):
            @override
            async def send_text(self, text: str) -> None:
                return None

        t = Good(_caps(optional_ops=frozenset({OptionalOp.SEND_TEXT})))
        assert t.check_contract() == []

    def test_reconnect_without_control_stream_is_caught(self) -> None:
        """A gap nothing reports cannot be budgeted."""
        t = FakeTransport(_caps(optional_ops=frozenset({OptionalOp.RECONNECT})))
        problems = t.check_contract()
        assert any("declares reconnect but not control_stream" in p for p in problems)

    def test_reconnect_with_control_stream_is_clean(self) -> None:
        class Reconnecting(FakeTransport):
            @override
            def control(self) -> AsyncIterator[ControlEvent]:
                async def _gen() -> AsyncIterator[ControlEvent]:
                    return
                    yield  # pragma: no cover

                return _gen()

        t = Reconnecting(
            _caps(
                optional_ops=frozenset(
                    {OptionalOp.RECONNECT, OptionalOp.CONTROL_STREAM}
                )
            )
        )
        assert t.check_contract() == []

    def test_raise_on_violation(self) -> None:
        t = FakeTransport(_caps(optional_ops=frozenset({OptionalOp.SEND_TEXT})))
        with pytest.raises(ContractError, match="does not override"):
            t.raise_on_contract_violation()

    def test_raise_is_silent_when_clean(self) -> None:
        FakeTransport().raise_on_contract_violation()


# ----------------------------------------------------------- control events


class TestControlEvents:
    def test_minimal_event(self) -> None:
        e = ControlEvent(kind=ControlEventKind.SPEECH_START, t=1.0, party=AGENT)
        assert e.duration_s is None
        assert e.meta == {}

    def test_gap_without_duration_is_rejected(self) -> None:
        """An unmeasured gap cannot be budgeted."""
        with pytest.raises(ValueError, match="must carry duration_s"):
            ControlEvent(kind=ControlEventKind.TRANSPORT_GAP, t=1.0, party=AGENT)

    def test_negative_time_and_duration_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative time"):
            ControlEvent(kind=ControlEventKind.SPEECH_END, t=-1.0, party=AGENT)
        with pytest.raises(ValueError, match="negative duration"):
            ControlEvent(
                kind=ControlEventKind.PROVIDER_ERROR,
                t=1.0,
                party=AGENT,
                duration_s=-0.5,
            )

    def test_meta_is_read_only(self) -> None:
        e = ControlEvent(kind=ControlEventKind.PEER_CLOSED, t=1.0, party=AGENT)
        with pytest.raises(TypeError):
            e.meta["x"] = 1  # type: ignore[index]


class TestGapTracker:
    def _gap(self, t: float, dur: float) -> ControlEvent:
        return ControlEvent(
            kind=ControlEventKind.TRANSPORT_GAP, t=t, party=AGENT, duration_s=dur
        )

    def test_starts_empty(self) -> None:
        g = GapTracker(budget_s=1.0)
        assert g.total_s == 0.0 and g.count == 0 and not g.exceeded

    def test_accumulates_gaps(self) -> None:
        g = GapTracker(budget_s=1.0)
        g.record(self._gap(1.0, 0.3))
        g.record(self._gap(5.0, 0.4))
        assert g.count == 2
        assert g.total_s == pytest.approx(0.7)
        assert not g.exceeded

    def test_ignores_other_event_kinds(self) -> None:
        g = GapTracker(budget_s=1.0)
        g.record(ControlEvent(kind=ControlEventKind.SPEECH_START, t=1.0, party=AGENT))
        assert g.count == 0

    def test_budget_is_exceeded_strictly_above(self) -> None:
        g = GapTracker(budget_s=1.0)
        g.record(self._gap(1.0, 1.0))
        assert not g.exceeded, "exactly at budget is still within it"
        g.record(self._gap(2.0, 0.01))
        assert g.exceeded

    def test_raise_names_the_numbers(self) -> None:
        g = GapTracker(budget_s=0.5)
        g.record(self._gap(1.0, 0.9))
        with pytest.raises(GapBudgetExceeded) as e:
            g.raise_if_exceeded()
        assert "0.900s across 1 gap" in str(e.value)

    def test_negative_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            GapTracker(budget_s=-1.0)
