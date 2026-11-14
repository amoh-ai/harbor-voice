"""L5 conversation and the full-duplex driver.

The question every test here asks in some form: can this tell the difference
between an agent that handled an interruption and one that never got it?
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import override

import pytest

from harbor_voice.audio.clock import Clock
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.duplex import DuplexDriver
from harbor_voice.duplex import DuplexError
from harbor_voice.floor import BargeInFloor
from harbor_voice.floor import Bid
from harbor_voice.floor import FullDuplexFloor
from harbor_voice.ids import PartyId
from harbor_voice.metrics import REGISTRY
from harbor_voice.metrics import MetricContext
from harbor_voice.perturbation import ConversationPlan
from harbor_voice.perturbation import Landing
from harbor_voice.perturbation import PerturbationPipeline
from harbor_voice.perturbation import Probe
from harbor_voice.perturbation import ProbeKind
from harbor_voice.perturbation import ProbeResult
from harbor_voice.perturbation import ProbeSpec
from harbor_voice.perturbation import SeedSet
from harbor_voice.perturbation import get_profile
from harbor_voice.perturbation.conversation import ConversationError
from harbor_voice.perturbation.conversation import ProbeInjector
from harbor_voice.providers import ToneTTS
from harbor_voice.simulator import ScriptContent
from harbor_voice.simulator import UniversalSimulator
from harbor_voice.trajectory import Recorder
from harbor_voice.trajectory.model import BargeInOutcome
from harbor_voice.trajectory.model import EventType
from harbor_voice.trajectory.model import PartyRecord
from harbor_voice.trajectory.model import PartyRole
from harbor_voice.transport import SessionConfig
from harbor_voice.transport import VoiceTransport

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
AGENT = PartyId("agent")


def _caps(duplex: frozenset[Duplex] | None = None) -> Capabilities:
    return Capabilities(
        duplex=duplex or frozenset({Duplex.FULL, Duplex.HALF}),
        pipeline=Pipeline.S2S,
        turn_control=TurnControl.NATIVE_VAD,
        can_be_interrupted=True,
        exposes_stages=frozenset({Stage.RESPONSE_START, Stage.RESPONSE_DONE}),
    )


# ------------------------------------------------------------------- the bit


class TalkativeAgent(VoiceTransport):
    """An agent that streams a long reply and may or may not stop for you.

    ``yield_after`` is the knob the whole suite turns: how many more frames it
    emits once the caller speaks over it. ``None`` means it never stops, which
    is the agent the barge-in metric exists to catch.

    ``emit_gap_s`` is deliberately slower than ``DuplexDriver``'s 5 ms poll
    interval. The driver fires a probe when the reply's accumulated audio
    passes its offset, so an emitter that outruns the poll loop delivers the
    whole reply between two polls, and the probe lands on the final frame with
    nothing after it — a correct ``INDISTINGUISHABLE`` for a run that was
    supposed to be measurable. A ratio holds under contention where a constant
    does not: load stretches both sides equally.

    It waits for the caller's audio to go quiet before it starts replying, the
    way a real VAD endpoint does. That is not decoration: without it, whether
    the tail of the opening turn counts as an interruption depends on whether
    the runner was fast enough to finish transmitting before the first reply
    frame appeared. On a loaded CI runner it was not, the reply was truncated
    to a few frames, and every probe scheduled past that point silently never
    fired.
    """

    def __init__(
        self,
        *,
        reply_frames: int = 120,
        emit_gap_s: float = 0.01,
        input_idle_s: float = 0.05,
        yield_after: int | None = None,
        capabilities: Capabilities | None = None,
    ) -> None:
        self._caps = capabilities or _caps()
        self._open = False
        self._reply_frames = reply_frames
        self._gap = emit_gap_s
        self._input_idle_s = input_idle_s
        self._yield_after = yield_after
        self._q: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._emitter: asyncio.Task[None] | None = None
        self._limit = reply_frames
        self._emitted = 0
        self._last_tx = 0.0
        self._speaking = False
        self._interrupting = False
        self.received: list[AudioFrame] = []
        self.interruptions = 0

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
        # The driver keeps the line open with silence between utterances, as
        # a phone line does; like a real agent's speech detector, only sound
        # counts as the caller talking.
        if frame.filler:
            return
        self._last_tx = asyncio.get_running_loop().time()
        if self._emitter is None:
            self._emitter = asyncio.create_task(self._emit())
            return
        # Audio arriving while the agent holds the floor is an interruption.
        # Counted once per utterance, not once per frame: a probe is ~19
        # frames of "mm-hm" and counting each one would say the caller
        # interrupted nineteen times.
        if not self._speaking or self._interrupting:
            return
        self._interrupting = True
        self.interruptions += 1
        if self._yield_after is not None:
            self._limit = min(self._limit, self._emitted + self._yield_after)

    async def _emit(self) -> None:
        loop = asyncio.get_running_loop()
        while loop.time() - self._last_tx < self._input_idle_s:
            await asyncio.sleep(self._gap)
        self._speaking = True
        while self._emitted < self._limit:
            await asyncio.sleep(self._gap)
            await self._q.put(
                AudioFrame(
                    pcm=b"\x20" * 960,
                    seq=self._emitted,
                    party=AGENT,
                    t_produced=self._emitted * 0.02,
                ).stamp(Clock.RX, self._emitted * 0.02)
            )
            self._emitted += 1
            if loop.time() - self._last_tx >= self._input_idle_s:
                self._interrupting = False
        self._speaking = False

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
        if self._emitter is not None:
            self._emitter.cancel()
        await self._q.put(None)


def _plan(*kinds: ProbeKind, at_s: float = 0.2, seed: int = 7) -> ConversationPlan:
    """One probe per kind, all at the same offset, so tests are about outcome."""
    return ConversationPlan(
        injectors=tuple(
            ProbeInjector(ProbeSpec(kind=k, phrases=(f"<{k.value}>",), first_at_s=at_s))
            for k in kinds
        ),
        seeds=SeedSet.derive(seed),
    )


def _simulator(plan: ConversationPlan | None = None) -> UniversalSimulator:
    pipeline = PerturbationPipeline(
        seeds=SeedSet.derive(7), conversation=plan or ConversationPlan()
    )
    return UniversalSimulator(
        party=CALLER,
        content=ScriptContent(turns=["I need to reschedule"]),
        tts=ToneTTS(),
        pipeline=pipeline,
        seeds=SeedSet.derive(7),
    )


async def _run(
    transport: TalkativeAgent,
    *,
    plan: ConversationPlan | None = None,
    floor: BargeInFloor | None = None,
    turn_timeout_s: float = 5.0,
    idle_s: float = 0.4,
):  # noqa: ANN202 - returns the driver's own turn type
    """Run one exchange.

    The timing constants are deliberately slack. ``TalkativeAgent`` emits a
    frame every 2 ms, so a 400 ms idle window gives a 200x margin before a
    stalled runner can make the driver think the reply finished — and a reply
    that ends early is not a visible failure, it is a probe that silently
    never fires. The two tests that *want* a timeout ask for a short one.
    """
    simulator = _simulator(plan)
    driver = DuplexDriver(
        transport,
        simulator=simulator,
        party=AGENT,
        floor=floor or BargeInFloor(yield_within_s=0.4),
        idle_s=idle_s,
        turn_timeout_s=turn_timeout_s,
    )
    await driver.open(SessionConfig(party=AGENT))
    try:
        utterance = await simulator.next_utterance()
        assert utterance is not None
        return driver, await driver.exchange(utterance), simulator
    finally:
        await driver.close()


# --------------------------------------------------------------------- floor


class TestTheBidIsTheWholeDistinction:
    """Saying "mm-hm" and saying "stop" are both audio; they ask opposite things."""

    def test_a_floor_bid_opens_a_yield_record(self) -> None:
        floor = BargeInFloor(yield_within_s=0.4)
        floor.speech_start(AGENT, 0.0)
        floor.speech_start(CALLER, 1.0, bid=Bid.FLOOR)
        assert len(floor.yields) == 1

    def test_a_backchannel_opens_none(self) -> None:
        floor = BargeInFloor(yield_within_s=0.4)
        floor.speech_start(AGENT, 0.0)
        floor.speech_start(CALLER, 1.0, bid=Bid.BACKCHANNEL)
        assert floor.yields == ()

    def test_but_the_overlap_is_still_real(self) -> None:
        """Counting the overlap is bid-agnostic; only the expectation differs."""
        floor = BargeInFloor()
        floor.speech_start(AGENT, 0.0)
        floor.speech_start(CALLER, 1.0, bid=Bid.BACKCHANNEL)
        floor.speech_end(CALLER, 1.3)
        assert len(floor.overlaps) == 1
        assert floor.overlaps[0].duration_s == pytest.approx(0.3)

    def test_an_unannotated_start_claims_the_floor(self) -> None:
        """Defaulting to backchannel would drop interruptions silently."""
        floor = BargeInFloor()
        floor.speech_start(AGENT, 0.0)
        floor.speech_start(CALLER, 1.0)
        assert len(floor.yields) == 1

    def test_kinds_agree_with_their_bids(self) -> None:
        assert ProbeKind.INTERRUPTION.bid is Bid.FLOOR
        assert ProbeKind.INTERRUPTION.expects_yield
        for kind in (
            ProbeKind.BACKCHANNEL,
            ProbeKind.VOCAL_TIC,
            ProbeKind.NON_DIRECTED,
        ):
            assert kind.bid is Bid.BACKCHANNEL
            assert not kind.expects_yield


# ---------------------------------------------------------------- scheduling


class TestSchedulingIsSeededAndDeclaredUpFront:
    def test_the_same_seed_schedules_the_same_probes(self) -> None:
        """An unreproducible perturbation is not a measurement."""
        first = ConversationPlan.of(ProbeKind.BACKCHANNEL, seeds=SeedSet.derive(3))
        second = ConversationPlan.of(ProbeKind.BACKCHANNEL, seeds=SeedSet.derive(3))
        assert [(p.text, p.at_s) for p in first.schedule(0)] == [
            (p.text, p.at_s) for p in second.schedule(0)
        ]

    def test_a_different_seed_schedules_differently(self) -> None:
        a = ConversationPlan.of(ProbeKind.VOCAL_TIC, seeds=SeedSet.derive(1))
        b = ConversationPlan.of(ProbeKind.VOCAL_TIC, seeds=SeedSet.derive(99))
        drawn = {
            tuple((p.text, p.at_s) for p in plan.schedule(turn))
            for plan in (a, b)
            for turn in range(6)
        }
        assert len(drawn) > 1

    def test_probes_come_back_in_firing_order(self) -> None:
        plan = ConversationPlan(
            injectors=(
                ProbeInjector(
                    ProbeSpec(
                        kind=ProbeKind.NON_DIRECTED, phrases=("late",), first_at_s=2.0
                    )
                ),
                ProbeInjector(
                    ProbeSpec(
                        kind=ProbeKind.INTERRUPTION, phrases=("early",), first_at_s=0.5
                    )
                ),
            )
        )
        assert [p.text for p in plan.schedule(0)] == ["early", "late"]

    def test_one_layer_one_injector(self) -> None:
        """Two sets of parameters for one layer make the plan record a lie."""
        with pytest.raises(ConversationError, match="one layer, one injector"):
            ConversationPlan(
                injectors=(
                    ProbeInjector(
                        ProbeSpec(kind=ProbeKind.BACKCHANNEL, phrases=("a",))
                    ),
                    ProbeInjector(
                        ProbeSpec(kind=ProbeKind.BACKCHANNEL, phrases=("b",))
                    ),
                )
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("phrases", ()),
            ("first_at_s", -1.0),
            ("every_s", 0.0),
            ("max_per_turn", 0),
            ("probability", 1.5),
            ("jitter_s", -0.1),
        ],
    )
    def test_nonsense_is_refused_at_construction(
        self, field: str, value: object
    ) -> None:
        kwargs: dict[str, object] = {
            "kind": ProbeKind.BACKCHANNEL,
            "phrases": ("mm-hm",),
            field: value,
        }
        with pytest.raises(ConversationError):
            ProbeSpec(**kwargs)  # type: ignore[arg-type]


class TestL5SitsBesideTheSignalChain:
    def test_it_degrades_nothing(self) -> None:
        pipeline = get_profile("overlap").pipeline(SeedSet.derive(1))
        assert pipeline.is_clean, "no layer here touches a sample"
        assert not pipeline.is_unperturbed, "but the profile is not `clean`"

    def test_it_still_reaches_metric_applicability(self) -> None:
        pipeline = get_profile("overlap").pipeline(SeedSet.derive(1))
        assert {str(k) for k in pipeline.active_layers} == {
            "L5.interruption",
            "L5.backchannel",
        }

    def test_the_plan_is_written(self) -> None:
        """The plan is constructed and recorded, not just declared."""
        pipeline = get_profile("selectivity_probes").pipeline(SeedSet.derive(1))
        layers = pipeline.as_record().as_dict()["layers"]
        assert set(layers) == {"L5.backchannel", "L5.vocal_tic", "L5.non_directed"}
        assert layers["L5.backchannel"]["params"]["bid"] == "backchannel"
        assert layers["L5.backchannel"]["achieved"]["scheduled"] == 0.0

    def test_a_simulator_declares_the_layer(self) -> None:
        """A task asking for L5 must find a party that serves it."""
        simulator = _simulator(_plan(ProbeKind.BACKCHANNEL))
        assert Layer.CONVERSATION in simulator.capabilities.layers


# ------------------------------------------------------------------- driving


class TestTheDriverMeasuresWhoYielded:
    async def test_an_agent_that_stops_quickly_yields_in_time(self) -> None:
        transport = TalkativeAgent(yield_after=3)
        _, turn, sim = await _run(transport, plan=_plan(ProbeKind.INTERRUPTION))
        (result,) = turn.probes
        assert transport.interruptions >= 1
        assert result.landing is Landing.DURING_SPEECH
        assert result.yielded_in_time is True
        assert result.yield_latency_s == pytest.approx(0.06, abs=0.03)
        assert result.correct is True

    async def test_an_agent_that_runs_on_yields_late(self) -> None:
        transport = TalkativeAgent(yield_after=40)
        _, turn, sim = await _run(transport, plan=_plan(ProbeKind.INTERRUPTION))
        (result,) = turn.probes
        assert result.yielded_in_time is False
        assert result.yield_latency_s is not None
        assert result.yield_latency_s > 0.4
        assert result.correct is False

    async def test_an_agent_that_never_stops_is_not_merely_late(self) -> None:
        """``None`` means never, not unknown — the third outcome barge-in needs."""
        transport = TalkativeAgent(
            reply_frames=100_000, emit_gap_s=0.0005, yield_after=None
        )
        _, turn, sim = await _run(
            transport, plan=_plan(ProbeKind.INTERRUPTION), turn_timeout_s=1.5
        )
        (result,) = turn.probes
        assert result.fired, "the probe has to land for this to be about yielding"
        assert not turn.is_complete, "we gave up before the agent did"
        assert result.yield_latency_s is None
        assert result.yielded_in_time is None
        assert result.correct is False

    async def test_a_probe_the_reply_outran_never_fired(self) -> None:
        """A probe that never happened is not one the agent ignored."""
        transport = TalkativeAgent(reply_frames=3)
        _, turn, sim = await _run(
            transport, plan=_plan(ProbeKind.INTERRUPTION, at_s=9.0)
        )
        (result,) = turn.probes
        assert not result.fired
        assert result.landing is Landing.REPLY_ALREADY_DONE
        assert result.correct is None, "not evidence about the agent"

    async def test_a_silent_agent_had_nothing_to_interrupt(self) -> None:
        transport = TalkativeAgent(reply_frames=0)
        _, turn, sim = await _run(
            transport, plan=_plan(ProbeKind.INTERRUPTION), turn_timeout_s=1.0
        )
        (result,) = turn.probes
        assert result.landing is Landing.NO_REPLY
        assert not result.is_measurable


class TestSelectivityIsTheSameObservationInverted:
    async def test_talking_through_a_backchannel_is_correct(self) -> None:
        transport = TalkativeAgent(reply_frames=40, yield_after=None)
        _, turn, sim = await _run(transport, plan=_plan(ProbeKind.BACKCHANNEL))
        (result,) = turn.probes
        assert result.ignored is True
        assert result.correct is True

    async def test_stopping_for_a_backchannel_is_not(self) -> None:
        """The commonest deployed failure: an "mm-hm" trips the VAD."""
        transport = TalkativeAgent(yield_after=2)
        _, turn, sim = await _run(transport, plan=_plan(ProbeKind.BACKCHANNEL))
        (result,) = turn.probes
        assert result.ignored is False
        assert result.correct is False

    async def test_a_backchannel_never_enters_the_barge_in_denominator(self) -> None:
        driver, turn, sim = await _run(
            TalkativeAgent(reply_frames=40, yield_after=None),
            plan=_plan(ProbeKind.BACKCHANNEL),
        )
        assert driver.floor.yields == ()

    async def test_every_ignore_probe_behaves_the_same_way(self) -> None:
        for kind in (
            ProbeKind.BACKCHANNEL,
            ProbeKind.VOCAL_TIC,
            ProbeKind.NON_DIRECTED,
        ):
            _, turn, sim = await _run(
                TalkativeAgent(reply_frames=40, yield_after=None), plan=_plan(kind)
            )
            (result,) = turn.probes
            assert result.correct is True, kind


class TestWhatTheDriverRefuses:
    async def test_a_half_duplex_transport_is_refused(self) -> None:
        """Driving one from here would measure the harness, not the agent."""
        transport = TalkativeAgent(capabilities=_caps(frozenset({Duplex.HALF})))
        driver = DuplexDriver(transport, simulator=_simulator(), party=AGENT)
        with pytest.raises(DuplexError, match="full duplex"):
            await driver.open(SessionConfig(party=AGENT))

    async def test_exchange_before_open_is_refused(self) -> None:
        simulator = _simulator()
        driver = DuplexDriver(TalkativeAgent(), simulator=simulator, party=AGENT)
        utterance = await simulator.say("hello")
        with pytest.raises(Exception, match="not running"):
            await driver.exchange(utterance)


class TestTheAchievedPlan:
    async def test_it_counts_what_landed_not_what_was_asked_for(self) -> None:
        """Achieved, not requested, for a layer that has no SNR."""
        plan = _plan(ProbeKind.INTERRUPTION)
        _, _turn, _sim = await _run(TalkativeAgent(yield_after=3), plan=plan)
        (injector,) = plan.injectors
        achieved = injector.achieved()
        assert achieved["scheduled"] == 1.0
        assert achieved["fired"] == 1.0
        assert achieved["measurable"] == 1.0
        assert achieved["correct"] == 1.0

    def test_it_records_how_far_the_probe_slipped(self) -> None:
        """A probe lands on a frame boundary, so the offset is approximate.

        Accounted as a pure function, not through a live driver: the slip is
        whatever the agent's framing imposes, and asserting a bound on that
        would be asserting a bound on the runner's scheduler.
        """
        injector = _plan(ProbeKind.INTERRUPTION, at_s=0.2).injectors[0]
        (probe,) = injector.schedule(0)
        injector.observe(
            ProbeResult(
                probe=probe,
                landing=Landing.DURING_SPEECH,
                at_reply_s=0.5,
                reply_total_s=1.0,
                t_injected=1.0,
                deadline_s=0.4,
            )
        )
        assert injector.achieved()["offset_drift_s"] == pytest.approx(0.3)

    async def test_a_probe_that_never_fired_has_no_drift_to_report(self) -> None:
        """Averaging drift over zero injections would be a division by zero."""
        plan = _plan(ProbeKind.INTERRUPTION, at_s=9.0)
        _, _turn, _sim = await _run(TalkativeAgent(reply_frames=3), plan=plan)
        assert "offset_drift_s" not in plan.injectors[0].achieved()

    async def test_a_probe_that_never_landed_is_visible_in_the_record(self) -> None:
        plan = _plan(ProbeKind.INTERRUPTION, at_s=9.0)
        _, _turn, _sim = await _run(TalkativeAgent(reply_frames=3), plan=plan)
        achieved = plan.injectors[0].achieved()
        assert achieved["scheduled"] == 1.0
        assert achieved["fired"] == 0.0
        assert achieved["measurable"] == 0.0


# ------------------------------------------------------------ the metrics


def _trajectory(turn, simulator, *, floor: BargeInFloor):  # noqa: ANN202
    """Record the turn the way a real trial would, so metrics read real data.

    ``include_yields=False`` because the driver already reported every probe;
    letting the floor write a second ``BARGE_IN`` for the same interruption
    would double the denominator.
    """
    recorder = Recorder(trial_id="t", task="probe", agent="agent")
    recorder.add_party(
        PartyRecord(
            id=str(AGENT), role=PartyRole.SYSTEM_UNDER_TEST, capabilities=_caps()
        )
    )
    recorder.session_open(0.0)
    recorder.absorb_floor(floor, include_yields=False)
    recorder.record_probes(turn.probes, by=CALLER)
    recorder.session_close(max(5.0, turn.reply_audio_s))
    return recorder.finish(pipeline=simulator.pipeline, render=False)


def _value(trajectory, name: str):  # noqa: ANN202
    from harbor_voice.metrics import evaluate_metric

    return evaluate_metric(REGISTRY.get(name), MetricContext(trajectory=trajectory))


class TestTheMetricsFinallyReport:
    """The whole point: these three stop saying N/A."""

    async def test_barge_in_success_rate_reads_the_probe(self) -> None:
        driver, turn, sim = await _run(
            TalkativeAgent(yield_after=3), plan=_plan(ProbeKind.INTERRUPTION)
        )
        result = _value(
            _trajectory(turn, sim, floor=driver.floor), "turns.barge_in_success_rate"
        )
        assert result.value is not None
        assert result.value == pytest.approx(1.0)
        assert result.detail["yielded_in_time"] == 1

    async def test_a_late_yield_scores_zero_and_says_so(self) -> None:
        driver, turn, sim = await _run(
            TalkativeAgent(yield_after=40), plan=_plan(ProbeKind.INTERRUPTION)
        )
        # Assert the preconditions as themselves. A drain that timed out
        # reports reply_total_s=None, which reads as *never yielded* -- so
        # without this the next assertion fails as a wrong verdict about the
        # agent rather than as the harness running out of patience.
        assert turn.is_complete, f"the reply did not finish: {turn.stop}"
        (fired,) = turn.probes
        assert fired.is_measurable, fired.landing

        result = _value(
            _trajectory(turn, sim, floor=driver.floor), "turns.barge_in_success_rate"
        )
        assert result.value is not None
        assert result.value == pytest.approx(0.0)
        assert result.detail == {
            "yielded_in_time": 0,
            "yielded_late": 1,
            "never": 0,
        }

    async def test_an_interruption_that_never_fired_leaves_the_denominator(
        self,
    ) -> None:
        driver, turn, sim = await _run(
            TalkativeAgent(reply_frames=3), plan=_plan(ProbeKind.INTERRUPTION, at_s=9.0)
        )
        trajectory = _trajectory(turn, sim, floor=driver.floor)
        events = [e for e in trajectory.timeline if e.type is EventType.BARGE_IN]
        assert [e.meta["outcome"] for e in events] == [
            BargeInOutcome.PENDING_DONE.value
        ]
        result = _value(trajectory, "turns.barge_in_success_rate")
        assert result.value is None
        assert any("no barge-in fired" in r for r in result.reasons)

    async def test_yield_latency_reports_in_reply_audio_seconds(self) -> None:
        driver, turn, sim = await _run(
            TalkativeAgent(yield_after=3), plan=_plan(ProbeKind.INTERRUPTION)
        )
        trajectory = _trajectory(turn, sim, floor=driver.floor)
        (event,) = [e for e in trajectory.timeline if e.type is EventType.BARGE_IN]
        assert event.meta["latency_basis"] == "reply_audio"
        result = _value(trajectory, "timing.yield_latency_s")
        assert result.value is not None
        assert result.value == pytest.approx(0.06, abs=0.03)

    async def test_selectivity_reports_when_the_probe_was_ignored(self) -> None:
        driver, turn, sim = await _run(
            TalkativeAgent(reply_frames=40, yield_after=None),
            plan=_plan(ProbeKind.BACKCHANNEL),
        )
        result = _value(
            _trajectory(turn, sim, floor=driver.floor), "selectivity.backchannel"
        )
        assert result.value is not None
        assert result.value == pytest.approx(1.0)

    async def test_selectivity_catches_a_derailed_agent(self) -> None:
        driver, turn, sim = await _run(
            TalkativeAgent(yield_after=2), plan=_plan(ProbeKind.BACKCHANNEL)
        )
        result = _value(
            _trajectory(turn, sim, floor=driver.floor), "selectivity.backchannel"
        )
        assert result.value is not None
        assert result.value == pytest.approx(0.0)

    def test_an_unmeasurable_probe_does_not_count_as_derailment(self) -> None:
        """A probe nobody could react to is a fact about the call, not a score."""
        probe = Probe(kind=ProbeKind.BACKCHANNEL, text="mm-hm", at_s=0.2)
        unmeasurable = ProbeResult(
            probe=probe,
            landing=Landing.INDISTINGUISHABLE,
            at_reply_s=0.2,
            t_injected=1.0,
        )
        recorder = Recorder(trial_id="t")
        recorder.add_party(
            PartyRecord(
                id=str(AGENT), role=PartyRole.SYSTEM_UNDER_TEST, capabilities=_caps()
            )
        )
        recorder.session_open(0.0)
        recorder.record_probe(unmeasurable, by=CALLER)
        recorder.session_close(2.0)
        pipeline = PerturbationPipeline(
            seeds=SeedSet.derive(1),
            conversation=ConversationPlan.of(ProbeKind.BACKCHANNEL),
        )
        result = _value(
            recorder.finish(pipeline=pipeline, render=False), "selectivity.backchannel"
        )
        assert result.value is None
        assert any("could be observed" in r for r in result.reasons)

    def test_the_layer_being_off_abstains_for_a_different_reason(self) -> None:
        """Two abstentions, two sentences: the setup, then the call.

        ``NotApplicable`` is reached by looking and finding nothing; the
        precondition is reached without looking at all. Collapsing
        them would hide a mis-configured task behind a quiet run.
        """
        recorder = Recorder(trial_id="t")
        recorder.add_party(
            PartyRecord(
                id=str(AGENT), role=PartyRole.SYSTEM_UNDER_TEST, capabilities=_caps()
            )
        )
        recorder.session_open(0.0)
        recorder.session_close(2.0)
        result = _value(recorder.finish(render=False), "selectivity.backchannel")
        assert result.value is None
        assert any("L5.backchannel" in r for r in result.reasons)


class TestEveryProbeHasAReader:
    """A standing test: a perturbation with no metric is noise."""

    def test_each_probe_kind_is_read_by_some_metric(self) -> None:
        for kind in ProbeKind:
            needed = f"layer:{kind.key}"
            readers = [
                name
                for name in REGISTRY.names()
                if needed in REGISTRY.get(name).requires
                or Layer.CONVERSATION in REGISTRY.get(name).sensitive_to
            ]
            assert readers, f"{kind.key} moves no registered number"

    def test_the_selectivity_metrics_name_the_layers_that_exist(self) -> None:
        """A required layer no injector can produce is an unreachable metric."""
        produced = {f"layer:{kind.key}" for kind in ProbeKind}
        for name in REGISTRY.names():
            for requirement in REGISTRY.get(name).requires:
                if requirement.startswith("layer:L5."):
                    assert requirement in produced, f"{name} requires {requirement}"


class TestTheFloorStaysASeparateConcern:
    async def test_overlap_is_recorded_whichever_bid_it_was(self) -> None:
        driver, turn, sim = await _run(
            TalkativeAgent(reply_frames=40, yield_after=None),
            plan=_plan(ProbeKind.BACKCHANNEL),
            floor=BargeInFloor(yield_within_s=0.4),
        )
        trajectory = _trajectory(turn, sim, floor=driver.floor)
        overlaps = [e for e in trajectory.timeline if e.type is EventType.OVERLAP]
        assert overlaps, "the caller really did speak over the agent"

    async def test_the_driver_is_the_only_source_of_barge_in_events(self) -> None:
        """Two writers would double every interruption in the denominator."""
        driver, turn, sim = await _run(
            TalkativeAgent(yield_after=3), plan=_plan(ProbeKind.INTERRUPTION)
        )
        trajectory = _trajectory(turn, sim, floor=driver.floor)
        events = [e for e in trajectory.timeline if e.type is EventType.BARGE_IN]
        assert len(events) == 1

    def test_a_plain_full_duplex_floor_has_no_yields_to_absorb(self) -> None:
        recorder = Recorder(trial_id="t")
        floor = FullDuplexFloor()
        floor.speech_start(AGENT, 0.0)
        floor.speech_start(CALLER, 0.5)
        floor.speech_end(CALLER, 0.8)
        floor.speech_end(AGENT, 1.0)
        recorder.session_open(0.0)
        recorder.absorb_floor(floor)
        recorder.session_close(1.0)
        trajectory = recorder.finish(render=False)
        assert not [e for e in trajectory.timeline if e.type is EventType.BARGE_IN]
        assert [e for e in trajectory.timeline if e.type is EventType.OVERLAP]


class PausingAgent(TalkativeAgent):
    """Speaks ten frames, pauses half a second, then finishes its reply."""

    @override
    async def _emit(self) -> None:
        loop = asyncio.get_running_loop()
        while loop.time() - self._last_tx < self._input_idle_s:
            await asyncio.sleep(self._gap)
        for seq in range(self._reply_frames):
            if seq == 10:
                await asyncio.sleep(0.5)
            await asyncio.sleep(self._gap)
            await self._q.put(
                AudioFrame(
                    pcm=b"\x20" * 960, seq=seq, party=AGENT, t_produced=seq * 0.02
                ).stamp(Clock.RX, seq * 0.02)
            )


class TestAPauseMidReplyIsNotTheEndOfTheTurn:
    """An agent streaming at speaking pace pauses between sentences."""

    async def test_the_default_quiet_period_keeps_the_whole_reply(self) -> None:
        simulator = _simulator()
        driver = DuplexDriver(
            PausingAgent(reply_frames=20),
            simulator=simulator,
            party=AGENT,
            turn_timeout_s=5.0,
        )
        await driver.open(SessionConfig(party=AGENT))
        try:
            utterance = await simulator.next_utterance()
            assert utterance is not None
            turn = await driver.exchange(utterance)
        finally:
            await driver.close()
        assert len(turn.received) == 20
        assert turn.stop.value == "idle"


class QuietTailAgent(TalkativeAgent):
    """Replies, then keeps streaming silence, as a WebRTC agent does."""

    async def _emit(self) -> None:
        loop = asyncio.get_running_loop()
        while loop.time() - self._last_tx < self._input_idle_s:
            await asyncio.sleep(self._gap)
        for seq in range(self._reply_frames + 30):
            await asyncio.sleep(self._gap)
            voiced = seq < self._reply_frames
            await self._q.put(
                AudioFrame(
                    pcm=(b"\x20" if voiced else b"\x00") * 960,
                    seq=seq,
                    party=AGENT,
                    t_produced=seq * 0.02,
                ).stamp(Clock.RX, seq * 0.02)
            )


class TestTheQuietAfterAReplyIsNotPartOfIt:
    async def test_trailing_silence_is_trimmed(self) -> None:
        _, turn, _ = await _run(QuietTailAgent(reply_frames=20), idle_s=0.2)
        assert len(turn.received) == 20
        assert all(f.pcm[0] == 0x20 for f in turn.received)
