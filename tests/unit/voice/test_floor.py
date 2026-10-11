"""Unit tests for floor policies."""

from __future__ import annotations

import pytest

from harbor_voice.capabilities import Duplex
from harbor_voice.floor import BargeInFloor
from harbor_voice.floor import Contention
from harbor_voice.floor import Decision
from harbor_voice.floor import FloorError
from harbor_voice.floor import FullDuplexFloor
from harbor_voice.floor import HalfDuplexFloor
from harbor_voice.floor import OverlapRecord
from harbor_voice.floor import YieldRecord
from harbor_voice.floor import floor_for
from harbor_voice.ids import PartyId

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
AGENT = PartyId("agent")
THIRD = PartyId("interpreter")


# ------------------------------------------------------------- bookkeeping


class TestBookkeeping:
    def test_starts_empty(self) -> None:
        f = FullDuplexFloor()
        assert f.speaking == frozenset()
        assert f.holder is None
        assert f.overlaps == () and f.violations == ()

    def test_tracks_who_is_speaking(self) -> None:
        f = FullDuplexFloor()
        f.speech_start(CALLER, 1.0)
        assert f.speaking == {CALLER}
        assert f.holder == CALLER
        f.speech_end(CALLER, 2.0)
        assert f.speaking == frozenset()
        assert f.holder is None

    def test_holder_is_whoever_started_first(self) -> None:
        f = FullDuplexFloor()
        f.speech_start(CALLER, 1.0)
        f.speech_start(AGENT, 1.5)
        assert f.holder == CALLER
        f.speech_end(CALLER, 2.0)
        assert f.holder == AGENT

    def test_double_start_is_an_error(self) -> None:
        f = FullDuplexFloor()
        f.speech_start(CALLER, 1.0)
        with pytest.raises(FloorError, match="already speaking"):
            f.speech_start(CALLER, 1.2)

    def test_end_without_start_is_an_error(self) -> None:
        with pytest.raises(FloorError, match="not speaking"):
            FullDuplexFloor().speech_end(CALLER, 1.0)

    def test_end_before_start_is_an_error(self) -> None:
        f = FullDuplexFloor()
        f.speech_start(CALLER, 2.0)
        with pytest.raises(FloorError, match="before starting"):
            f.speech_end(CALLER, 1.0)


# ---------------------------------------------------------------- overlaps


class TestOverlaps:
    def test_no_overlap_when_turns_do_not_touch(self) -> None:
        f = FullDuplexFloor()
        f.speech_start(CALLER, 1.0)
        f.speech_end(CALLER, 2.0)
        f.speech_start(AGENT, 2.5)
        f.speech_end(AGENT, 3.0)
        assert f.overlaps == ()

    def test_overlap_is_recorded_with_its_duration(self) -> None:
        f = FullDuplexFloor()
        f.speech_start(CALLER, 1.0)
        f.speech_start(AGENT, 1.5)
        f.speech_end(CALLER, 2.0)
        f.speech_end(AGENT, 3.0)
        assert len(f.overlaps) == 1
        ov = f.overlaps[0]
        assert ov.t_start == 1.5
        assert ov.duration_s == pytest.approx(0.5)
        assert set(ov.parties) == {CALLER, AGENT}
        assert not ov.is_open

    def test_an_open_overlap_has_no_duration(self) -> None:
        f = FullDuplexFloor()
        f.speech_start(CALLER, 1.0)
        f.speech_start(AGENT, 1.5)
        assert f.overlaps[0].is_open
        assert f.overlaps[0].duration_s is None

    def test_three_parties_overlapping_is_one_record(self) -> None:
        f = FullDuplexFloor()
        f.speech_start(CALLER, 1.0)
        f.speech_start(AGENT, 1.2)
        f.speech_start(THIRD, 1.4)
        f.speech_end(THIRD, 1.6)
        assert len(f.overlaps) == 1, "still contended, so still one overlap"
        f.speech_end(AGENT, 1.8)
        assert f.overlaps[0].duration_s == pytest.approx(0.6)

    def test_successive_overlaps_are_separate_records(self) -> None:
        f = FullDuplexFloor()
        for start, end in ((1.0, 2.0), (3.0, 4.0)):
            f.speech_start(CALLER, start)
            f.speech_start(AGENT, start + 0.5)
            f.speech_end(AGENT, end - 0.2)
            f.speech_end(CALLER, end)
        assert len(f.overlaps) == 2


# ------------------------------------------------------------- full duplex


class TestFullDuplexFloor:
    def test_always_allows(self) -> None:
        f = FullDuplexFloor()
        f.speech_start(CALLER, 1.0)
        assert f.may_transmit(AGENT, 1.1) is Decision.ALLOW

    def test_records_no_violations(self) -> None:
        f = FullDuplexFloor()
        f.speech_start(CALLER, 1.0)
        f.may_transmit(AGENT, 1.1)
        assert f.violations == ()

    def test_declares_full(self) -> None:
        assert FullDuplexFloor().duplex is Duplex.FULL


# ------------------------------------------------------------- half duplex


class TestHalfDuplexFloor:
    def test_allows_when_the_floor_is_free(self) -> None:
        assert HalfDuplexFloor().may_transmit(AGENT, 1.0) is Decision.ALLOW

    def test_allows_the_holder_to_continue(self) -> None:
        f = HalfDuplexFloor()
        f.speech_start(CALLER, 1.0)
        assert f.may_transmit(CALLER, 1.1) is Decision.ALLOW

    def test_queues_a_contender_by_default(self) -> None:
        f = HalfDuplexFloor()
        f.speech_start(CALLER, 1.0)
        assert f.may_transmit(AGENT, 1.1) is Decision.QUEUE

    def test_drops_when_configured_to(self) -> None:
        f = HalfDuplexFloor(on_contention=Contention.DROP)
        f.speech_start(CALLER, 1.0)
        assert f.may_transmit(AGENT, 1.1) is Decision.DROP
        assert f.violations[0].decision is Decision.DROP

    def test_a_blocked_attempt_is_a_measurement(self) -> None:
        """'talked over the caller 11 times' is a finding, not a crash."""
        f = HalfDuplexFloor()
        f.speech_start(CALLER, 1.0)
        f.may_transmit(AGENT, 1.1)
        assert len(f.violations) == 1
        v = f.violations[0]
        assert v.party == AGENT and v.holder == CALLER and v.t_start == 1.1

    def test_one_violation_per_blocked_period_not_per_frame(self) -> None:
        """At 50 fps, per-frame records would drown the signal."""
        f = HalfDuplexFloor()
        f.speech_start(CALLER, 1.0)
        for i in range(50):
            f.may_transmit(AGENT, 1.1 + i * 0.02)
        assert len(f.violations) == 1

    def test_violation_closes_when_the_floor_frees(self) -> None:
        f = HalfDuplexFloor()
        f.speech_start(CALLER, 1.0)
        f.may_transmit(AGENT, 1.1)
        f.speech_end(CALLER, 2.0)
        assert f.may_transmit(AGENT, 2.1) is Decision.ALLOW
        assert f.violations[0].duration_s == pytest.approx(1.0)

    def test_a_second_contention_is_a_second_violation(self) -> None:
        f = HalfDuplexFloor()
        f.speech_start(CALLER, 1.0)
        f.may_transmit(AGENT, 1.1)
        f.speech_end(CALLER, 2.0)
        f.may_transmit(AGENT, 2.1)
        f.speech_start(CALLER, 3.0)
        f.may_transmit(AGENT, 3.1)
        assert len(f.violations) == 2

    def test_declares_half(self) -> None:
        assert HalfDuplexFloor().duplex is Duplex.HALF

    def test_a_half_duplex_agent_under_a_full_floor_is_an_experiment(self) -> None:
        """The same traffic under a different floor is measurable."""
        traffic = [(CALLER, 1.0, 2.0), (AGENT, 1.5, 2.5)]
        half, full = HalfDuplexFloor(), FullDuplexFloor()
        for floor in (half, full):
            for party, start, end in traffic:
                floor.may_transmit(party, start)
                floor.speech_start(party, start)
            for party, _, end in traffic:
                floor.speech_end(party, end)
        assert len(half.violations) == 1
        assert len(full.violations) == 0
        assert len(half.overlaps) == len(full.overlaps) == 1


# ----------------------------------------------------------------- barge-in


class TestBargeInFloor:
    def test_is_full_duplex_and_always_allows(self) -> None:
        f = BargeInFloor()
        f.speech_start(CALLER, 1.0)
        assert f.duplex is Duplex.FULL
        assert f.may_transmit(AGENT, 1.1) is Decision.ALLOW

    def test_no_barge_in_when_turns_do_not_touch(self) -> None:
        f = BargeInFloor()
        f.speech_start(AGENT, 1.0)
        f.speech_end(AGENT, 2.0)
        f.speech_start(CALLER, 2.5)
        assert f.yields == ()

    def test_records_a_prompt_yield(self) -> None:
        f = BargeInFloor(yield_within_s=0.4)
        f.speech_start(AGENT, 1.0)
        f.speech_start(CALLER, 2.0)
        f.speech_end(AGENT, 2.2)
        assert len(f.yields) == 1
        y = f.yields[0]
        assert y.holder == AGENT and y.interrupter == CALLER
        assert y.latency_s == pytest.approx(0.2)
        assert y.yielded_in_time is True
        assert f.late_yields == ()

    def test_records_a_late_yield(self) -> None:
        f = BargeInFloor(yield_within_s=0.4)
        f.speech_start(AGENT, 1.0)
        f.speech_start(CALLER, 2.0)
        f.speech_end(AGENT, 3.0)
        assert f.yields[0].yielded_in_time is False
        assert len(f.late_yields) == 1

    def test_an_unresolved_yield_has_no_latency(self) -> None:
        """The agent never stopped. Worse than late, and distinct from it."""
        f = BargeInFloor()
        f.speech_start(AGENT, 1.0)
        f.speech_start(CALLER, 2.0)
        assert f.yields[0].latency_s is None
        assert f.yields[0].yielded_in_time is None
        assert len(f.unresolved_yields) == 1
        assert f.late_yields == ()

    def test_exactly_at_the_deadline_counts_as_in_time(self) -> None:
        f = BargeInFloor(yield_within_s=0.4)
        f.speech_start(AGENT, 1.0)
        f.speech_start(CALLER, 2.0)
        f.speech_end(AGENT, 2.4)
        assert f.yields[0].yielded_in_time is True

    def test_one_record_per_holder_per_barge_in(self) -> None:
        f = BargeInFloor()
        f.speech_start(AGENT, 1.0)
        f.speech_start(THIRD, 1.1)
        f.speech_start(CALLER, 2.0)
        assert len(f.yields) == 2, "both prior speakers are expected to yield"
        assert {y.holder for y in f.yields} == {AGENT, THIRD}

    def test_a_holder_already_pending_is_not_recorded_twice(self) -> None:
        f = BargeInFloor()
        f.speech_start(AGENT, 1.0)
        f.speech_start(CALLER, 2.0)
        f.speech_start(THIRD, 2.1)
        holders = [y.holder for y in f.yields]
        assert holders.count(AGENT) == 1

    def test_overlaps_are_still_recorded(self) -> None:
        f = BargeInFloor()
        f.speech_start(AGENT, 1.0)
        f.speech_start(CALLER, 2.0)
        f.speech_end(AGENT, 2.3)
        assert len(f.overlaps) == 1
        assert f.overlaps[0].duration_s == pytest.approx(0.3)

    def test_negative_deadline_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            BargeInFloor(yield_within_s=-0.1)


# ------------------------------------------------------------------ shapes


class TestRecordShapes:
    def test_overlap_record_is_frozen(self) -> None:
        ov = OverlapRecord(parties=(CALLER,), t_start=1.0)
        with pytest.raises(AttributeError):
            ov.t_start = 2.0  # type: ignore[misc]

    def test_yield_record_defaults(self) -> None:
        y = YieldRecord(holder=AGENT, interrupter=CALLER, t_interrupted=1.0)
        assert y.latency_s is None and y.yielded_in_time is None

    def test_floor_for_picks_a_sensible_default(self) -> None:
        assert isinstance(floor_for(Duplex.HALF), HalfDuplexFloor)
        assert isinstance(floor_for(Duplex.FULL), FullDuplexFloor)

    def test_repr_is_useful(self) -> None:
        f = HalfDuplexFloor()
        f.speech_start(CALLER, 1.0)
        f.may_transmit(AGENT, 1.1)
        r = repr(f)
        assert "HalfDuplexFloor" in r and "violations=1" in r
