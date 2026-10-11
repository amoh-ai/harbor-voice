"""Unit tests for the drain helper and the adapter base."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import override

import pytest

from harbor_voice.adapter import AdapterError
from harbor_voice.adapter import Turn
from harbor_voice.adapter import VoiceAgentAdapter
from harbor_voice.audio.clock import Clock
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.floor import Contention
from harbor_voice.floor import HalfDuplexFloor
from harbor_voice.ids import PartyId
from harbor_voice.transport import SessionConfig
from harbor_voice.transport import VoiceTransport
from harbor_voice.transport.drain import DrainStop
from harbor_voice.transport.drain import drain_until_idle

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
AGENT = PartyId("agent")


def _caps(**over: object) -> Capabilities:
    base: dict[str, object] = {
        "duplex": frozenset({Duplex.FULL, Duplex.HALF}),
        "pipeline": Pipeline.S2S,
        "turn_control": TurnControl.NATIVE_VAD,
        "exposes_stages": frozenset({Stage.RESPONSE_START, Stage.RESPONSE_DONE}),
    }
    return Capabilities(**(base | over))  # type: ignore[arg-type]


def _frame(seq: int, party: PartyId = CALLER, t: float = 0.0) -> AudioFrame:
    return AudioFrame(pcm=b"\x00" * 960, seq=seq, party=party, t_produced=t)


async def _stream_of(
    *frames: AudioFrame, gap_s: float = 0.0
) -> AsyncIterator[AudioFrame]:
    for f in frames:
        if gap_s:
            await asyncio.sleep(gap_s)
        yield f


# ------------------------------------------------------------------ drain


class TestDrainUntilIdle:
    async def test_collects_everything_then_stops_on_idle(self) -> None:
        frames = [_frame(i, AGENT).stamp(Clock.RX, i * 0.02) for i in range(3)]
        result = await drain_until_idle(_stream_of(*frames), idle_s=0.05, timeout_s=2.0)
        assert [f.seq for f in result.frames] == [0, 1, 2]
        assert result.stop in (DrainStop.IDLE, DrainStop.STREAM_END)
        assert result.is_complete

    async def test_empty_stream_reports_empty(self) -> None:
        result = await drain_until_idle(_stream_of(), idle_s=0.05, timeout_s=0.3)
        assert result.frames == ()
        assert result.stop is DrainStop.EMPTY

    async def test_timeout_is_distinguishable_from_a_clean_end(self) -> None:
        """A turn we gave up on is not a turn that finished."""

        async def _never_ending() -> AsyncIterator[AudioFrame]:
            yield _frame(0, AGENT).stamp(Clock.RX, 0.0)
            while True:
                await asyncio.sleep(0.01)

        result = await drain_until_idle(_never_ending(), idle_s=5.0, timeout_s=0.2)
        assert result.stop is DrainStop.TIMEOUT
        assert not result.is_complete

    async def test_flow_points_come_from_the_frames(self) -> None:
        """Nothing back-computed from byte length."""
        frames = [_frame(i, AGENT).stamp(Clock.RX, 1.0 + i * 0.02) for i in range(3)]
        result = await drain_until_idle(_stream_of(*frames), idle_s=0.05, timeout_s=2.0)
        assert result.first_rx == pytest.approx(1.0)
        assert result.last_rx == pytest.approx(1.04)
        assert result.audio_duration_s == pytest.approx(0.06)

    async def test_unstamped_frames_yield_no_flow_points(self) -> None:
        result = await drain_until_idle(
            _stream_of(_frame(0, AGENT)), idle_s=0.05, timeout_s=1.0
        )
        assert result.first_rx is None and result.last_rx is None

    async def test_a_stream_error_propagates(self) -> None:
        """A transport that breaks mid-turn is a finding, not a short reply."""

        async def _broken() -> AsyncIterator[AudioFrame]:
            yield _frame(0, AGENT)
            raise ConnectionResetError("socket died")

        with pytest.raises(ConnectionResetError):
            await drain_until_idle(_broken(), idle_s=0.05, timeout_s=1.0)

    @pytest.mark.parametrize(("idle", "timeout"), [(0.0, 1.0), (-1.0, 1.0), (0.1, 0.0)])
    async def test_rejects_nonsense_bounds(self, idle: float, timeout: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            await drain_until_idle(_stream_of(), idle_s=idle, timeout_s=timeout)


# -------------------------------------------------------------- transport


class ScriptedTransport(VoiceTransport):
    """Replies with a fixed set of frames after the input arrives."""

    def __init__(
        self,
        reply: list[AudioFrame] | None = None,
        capabilities: Capabilities | None = None,
    ) -> None:
        self._caps = capabilities or _caps()
        self._open = False
        self._reply = reply if reply is not None else []
        self._q: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self.received: list[AudioFrame] = []
        self._replied = False

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
        self.received.append(frame)
        if not self._replied:
            self._replied = True
            for i, f in enumerate(self._reply):
                await self._q.put(f.stamp(Clock.RX, 10.0 + i * 0.02))

    @override
    def rx(self) -> AsyncIterator[AudioFrame]:
        async def _gen() -> AsyncIterator[AudioFrame]:
            while True:
                frame = await self._q.get()
                if frame is None:
                    return
                yield frame

        return _gen()

    @override
    async def close(self) -> None:
        self._open = False
        await self._q.put(None)


class S2SAdapter(VoiceAgentAdapter):
    @property
    @override
    def pipeline(self) -> Pipeline:
        return Pipeline.S2S


def _reply(n: int = 3) -> list[AudioFrame]:
    return [_frame(i, AGENT, t=10.0 + i * 0.02) for i in range(n)]


async def _opened(
    adapter: VoiceAgentAdapter,
) -> VoiceAgentAdapter:
    await adapter.open(SessionConfig(party=AGENT))
    return adapter


# ---------------------------------------------------------------- adapter


class TestAdapterLifecycle:
    async def test_call_before_open_is_refused(self) -> None:
        a = S2SAdapter(ScriptedTransport(), party=AGENT)
        with pytest.raises(AdapterError, match="before open"):
            await a.call([_frame(0)])

    async def test_pipeline_must_match_the_capability_declaration(self) -> None:
        """A cascade adapter over an s2s capability block is a wiring bug."""

        class Mismatched(VoiceAgentAdapter):
            @property
            @override
            def pipeline(self) -> Pipeline:
                return Pipeline.CASCADE

        a = Mismatched(ScriptedTransport(), party=AGENT)
        with pytest.raises(AdapterError, match="declares pipeline=cascade"):
            await a.open(SessionConfig(party=AGENT))

    async def test_empty_input_is_refused(self) -> None:
        a = await _opened(S2SAdapter(ScriptedTransport(), party=AGENT))
        with pytest.raises(AdapterError, match="at least one frame"):
            await a.call([])

    async def test_mixed_speakers_in_one_call_are_refused(self) -> None:
        a = await _opened(S2SAdapter(ScriptedTransport(), party=AGENT))
        with pytest.raises(AdapterError, match="one party"):
            await a.call([_frame(0, CALLER), _frame(1, AGENT)])

    async def test_close_is_delegated(self) -> None:
        t = ScriptedTransport()
        a = await _opened(S2SAdapter(t, party=AGENT))
        await a.close()
        assert not t.is_open


class TestTurn:
    async def test_a_turn_round_trips(self) -> None:
        t = ScriptedTransport(reply=_reply(3))
        a = await _opened(S2SAdapter(t, party=AGENT, idle_s=0.05, turn_timeout_s=0.5))
        turn = await a.call([_frame(0, CALLER), _frame(1, CALLER)])

        assert turn.index == 0
        assert turn.party == AGENT and turn.speaker == CALLER
        assert len(turn.sent) == 2
        assert len(turn.received) == 3
        assert turn.is_complete

    async def test_turn_index_advances(self) -> None:
        t = ScriptedTransport(reply=_reply(1))
        a = await _opened(S2SAdapter(t, party=AGENT, idle_s=0.05, turn_timeout_s=0.5))
        first = await a.call([_frame(0, CALLER)])
        second = await a.call([_frame(1, CALLER)])
        assert (first.index, second.index) == (0, 1)

    async def test_sent_frames_carry_a_real_tx_stamp(self) -> None:
        t = ScriptedTransport(reply=_reply(1))
        a = await _opened(S2SAdapter(t, party=AGENT, idle_s=0.05, turn_timeout_s=0.5))
        turn = await a.call([_frame(0, CALLER)])
        assert turn.sent[0].at(Clock.TX) is not None
        assert turn.last_tx is not None

    async def test_response_latency_names_both_clocks(self) -> None:
        """The clock domains travel with the number."""
        t = ScriptedTransport(reply=_reply(2))
        a = await _opened(S2SAdapter(t, party=AGENT, idle_s=0.05, turn_timeout_s=0.5))
        turn = await a.call([_frame(0, CALLER)])
        latency = turn.response_latency
        assert latency is not None
        assert latency.label == "tx_to_rx"
        assert latency.named("response") == "response_tx_to_rx"

    async def test_latency_is_none_rather_than_zero_when_unmeasurable(self) -> None:
        """A missing number must not be averaged into a published one."""
        t = ScriptedTransport(reply=[])
        a = await _opened(S2SAdapter(t, party=AGENT, idle_s=0.05, turn_timeout_s=0.3))
        turn = await a.call([_frame(0, CALLER)])
        assert turn.received == ()
        assert turn.response_latency is None
        assert turn.stop is DrainStop.EMPTY

    async def test_reply_span_and_audio_duration_are_different_things(self) -> None:
        t = ScriptedTransport(reply=_reply(3))
        a = await _opened(S2SAdapter(t, party=AGENT, idle_s=0.05, turn_timeout_s=0.5))
        turn = await a.call([_frame(0, CALLER)])
        span = turn.reply_span
        assert span is not None and span.label == "rx_to_rx"
        assert span.seconds == pytest.approx(0.04)
        assert turn.reply_audio_s == pytest.approx(0.06)

    async def test_stage_marks_bracket_the_reply(self) -> None:
        t = ScriptedTransport(reply=_reply(3))
        a = await _opened(S2SAdapter(t, party=AGENT, idle_s=0.05, turn_timeout_s=0.5))
        turn = await a.call([_frame(0, CALLER)])
        assert turn.stage_at(Stage.RESPONSE_START) == pytest.approx(10.0)
        assert turn.stage_at(Stage.RESPONSE_DONE) == pytest.approx(10.04)
        assert turn.stage_at(Stage.ASR_FINAL) is None, "s2s has no such seam"

    async def test_no_stages_when_nothing_came_back(self) -> None:
        t = ScriptedTransport(reply=[])
        a = await _opened(S2SAdapter(t, party=AGENT, idle_s=0.05, turn_timeout_s=0.3))
        turn = await a.call([_frame(0, CALLER)])
        assert turn.stages == ()


class TestFloorIntegration:
    async def test_the_floor_sees_both_parties(self) -> None:
        t = ScriptedTransport(reply=_reply(2))
        floor = HalfDuplexFloor()
        a = await _opened(
            S2SAdapter(t, party=AGENT, floor=floor, idle_s=0.05, turn_timeout_s=0.5)
        )
        await a.call([_frame(0, CALLER)])
        assert floor.speaking == frozenset(), "both turns closed"
        assert floor.violations == ()

    async def test_dropped_frames_are_counted_not_silently_lost(self) -> None:
        """A muted party's frames vanish, but the count survives."""
        t = ScriptedTransport(reply=_reply(1))
        floor = HalfDuplexFloor(on_contention=Contention.DROP)
        floor.speech_start(PartyId("third"), 0.0)
        a = await _opened(
            S2SAdapter(t, party=AGENT, floor=floor, idle_s=0.05, turn_timeout_s=0.5)
        )
        turn = await a.call([_frame(0, CALLER), _frame(1, CALLER)])
        assert turn.dropped == 2
        assert turn.sent == ()
        assert len(floor.violations) == 1


class TestBleedReconciliation:
    async def test_bleed_is_attributed_to_its_own_turn(self) -> None:
        """Audio crossing a turn boundary must not inflate the next latency."""
        turn = Turn(
            index=1,
            party=AGENT,
            speaker=CALLER,
            bled=(_frame(0, AGENT).stamp(Clock.RX, 9.0),),
            received=(_frame(1, AGENT).stamp(Clock.RX, 10.0),),
            sent=(_frame(0, CALLER).stamp(Clock.TX, 9.5),),
        )
        assert turn.first_rx == pytest.approx(10.0), "bleed is excluded"
        latency = turn.response_latency
        assert latency is not None and latency.seconds == pytest.approx(0.5)

    async def test_turns_start_with_no_bleed(self) -> None:
        t = ScriptedTransport(reply=_reply(1))
        a = await _opened(S2SAdapter(t, party=AGENT, idle_s=0.05, turn_timeout_s=0.5))
        turn = await a.call([_frame(0, CALLER)])
        assert turn.bled == ()


class TestRepr:
    def test_repr_names_the_pipeline(self) -> None:
        a = S2SAdapter(ScriptedTransport(), party=AGENT)
        assert "S2SAdapter" in repr(a)
        assert "pipeline=s2s" in repr(a)
