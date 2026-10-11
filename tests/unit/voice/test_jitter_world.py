"""L4 jitter and L6 world.

The two need different things:
jitter needs a frame-level wire and L6 needs a tool boundary the
harness owns.
"""

from __future__ import annotations

import pytest

from harbor_voice.audio.clock import Clock
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import Layer
from harbor_voice.ids import PartyId
from harbor_voice.perturbation import MuLawCodec
from harbor_voice.perturbation import PerturbationPipeline
from harbor_voice.perturbation import SeedSet
from harbor_voice.perturbation.jitter import Distribution
from harbor_voice.perturbation.jitter import Jitter
from harbor_voice.perturbation.jitter import JitterError
from harbor_voice.perturbation.world import LatencyProfile
from harbor_voice.perturbation.world import Outage
from harbor_voice.perturbation.world import ToolFailure
from harbor_voice.perturbation.world import WorldError
from harbor_voice.perturbation.world import WorldPlan

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")


def _frames(n: int = 50) -> list[AudioFrame]:
    return [
        AudioFrame(
            pcm=b"\x10\x00" * (CANONICAL.frame_bytes() // 2),
            seq=i,
            party=CALLER,
            t_produced=i * 0.02,
        )
        for i in range(n)
    ]


# ------------------------------------------------------------------- jitter


class TestJitterNeedsASequence:
    def test_a_stream_layer_is_kept_out_of_the_per_frame_chain(self) -> None:
        """Delay *variation* is not defined on one arrival."""
        pipeline = PerturbationPipeline(
            [MuLawCodec(), Jitter()], seeds=SeedSet.derive(1)
        )
        assert [p.name for p in pipeline.perturbations] == ["codec"]
        assert [p.name for p in pipeline.stream] == ["jitter"]

    def test_process_one_frame_does_not_silently_apply_it(self) -> None:
        """Doing nothing quietly would be worse than not offering it."""
        pipeline = PerturbationPipeline(
            [Jitter(mean_ms=50.0, std_ms=30.0, buffer_ms=0.0)], seeds=SeedSet.derive(1)
        )
        frame = _frames(1)[0]
        assert pipeline.process(frame).at(Clock.VOICED) is None

    def test_the_codec_runs_before_the_network_delays_it(self) -> None:
        """Physical order: the handset encodes, then the wire reorders."""
        pipeline = PerturbationPipeline(
            [Jitter(), MuLawCodec()], seeds=SeedSet.derive(1)
        )
        assert pipeline.perturbations[0].name == "codec"
        assert pipeline.stream[0].name == "jitter"

    def test_an_empty_stream_is_fine(self) -> None:
        assert Jitter().apply(()) == ()


class TestWhatAJitterBufferDoes:
    def test_a_generous_buffer_keeps_everything(self) -> None:
        jitter = Jitter(mean_ms=20.0, std_ms=5.0, buffer_ms=500.0)
        jitter.reset(SeedSet.derive(2))
        assert len(jitter.apply(_frames(40))) == 40
        assert jitter.achieved()["late_fraction"] == 0.0

    def test_a_frame_past_its_deadline_is_dropped(self) -> None:
        """Loss caused by delay. It sounds identical to a dropped packet."""
        jitter = Jitter(mean_ms=120.0, std_ms=5.0, buffer_ms=20.0)
        jitter.reset(SeedSet.derive(2))
        kept = jitter.apply(_frames(40))
        assert len(kept) < 40
        assert jitter.achieved()["late_fraction"] > 0.5

    def test_keep_late_separates_mangled_from_merely_late(self) -> None:
        """An experiment may want the audio without the delay loss."""
        jitter = Jitter(mean_ms=120.0, std_ms=5.0, buffer_ms=20.0, keep_late=True)
        jitter.reset(SeedSet.derive(2))
        assert len(jitter.apply(_frames(40))) == 40
        assert jitter.achieved()["late_fraction"] > 0.5

    def test_a_reordered_frame_is_put_back_and_counted(self) -> None:
        """Nobody hears it, but a line that reorders is about to drop."""
        jitter = Jitter(mean_ms=30.0, std_ms=40.0, buffer_ms=1000.0)
        jitter.reset(SeedSet.derive(4))
        kept = jitter.apply(_frames(50))
        assert jitter.achieved()["reordered_fraction"] > 0.0
        assert len(kept) == 50

    def test_no_variation_means_latency_not_jitter(self) -> None:
        """A constant delay shifts everything equally and reorders nothing."""
        jitter = Jitter(mean_ms=80.0, std_ms=0.0, buffer_ms=100.0)
        jitter.reset(SeedSet.derive(1))
        kept = jitter.apply(_frames(30))
        assert len(kept) == 30
        assert [f.seq for f in kept] == list(range(30))
        assert jitter.achieved()["reordered_fraction"] == 0.0

    def test_arrival_is_stamped_on_the_voiced_clock(self) -> None:
        """The domain for 'became audible, after any jitter buffer'."""
        jitter = Jitter(mean_ms=30.0, std_ms=5.0, buffer_ms=500.0)
        jitter.reset(SeedSet.derive(1))
        first = jitter.apply(_frames(5))[0]
        voiced = first.at(Clock.VOICED)
        assert voiced is not None and voiced > 0.0

    def test_delay_is_never_negative(self) -> None:
        """A frame arriving before it was sent is not a network condition."""
        jitter = Jitter(mean_ms=1.0, std_ms=200.0, buffer_ms=10_000.0)
        jitter.reset(SeedSet.derive(7))
        jitter.apply(_frames(200))
        assert jitter.achieved()["delay_ms_min"] >= 0.0

    def test_a_heavy_tail_reaches_further_than_a_symmetric_one(self) -> None:
        """Congested links are mostly fine and occasionally terrible."""
        gauss = Jitter(mean_ms=30.0, std_ms=10.0, distribution=Distribution.GAUSSIAN)
        pareto = Jitter(mean_ms=30.0, std_ms=10.0, distribution=Distribution.PARETO)
        for layer in (gauss, pareto):
            layer.reset(SeedSet.derive(9))
            layer.apply(_frames(300))
        assert pareto.achieved()["delay_ms_max"] > gauss.achieved()["delay_ms_max"]


class TestJitterIsReproducibleAndRecorded:
    def test_the_same_seed_delays_the_same_way(self) -> None:
        """An unreproducible perturbation is not a measurement."""
        out = []
        for _ in range(2):
            layer = Jitter(mean_ms=30.0, std_ms=20.0, buffer_ms=40.0)
            layer.reset(SeedSet.derive(11))
            out.append([f.seq for f in layer.apply(_frames(60))])
        assert out[0] == out[1]

    def test_the_plan_records_what_the_line_actually_did(self) -> None:
        """30 ms requested is not 30 ms delivered."""
        pipeline = PerturbationPipeline(
            [Jitter(mean_ms=30.0, std_ms=20.0, buffer_ms=40.0)],
            seeds=SeedSet.derive(3),
        )
        pipeline.process_all(_frames(50))
        entry = pipeline.as_record().as_dict()["layers"]["L4.jitter"]
        assert entry["params"]["mean_ms"] == 30.0
        assert entry["achieved"]["delay_ms_mean"] == pytest.approx(30.0, abs=6.0)
        assert entry["achieved"]["late_fraction"] > 0.0

    def test_an_unused_layer_claims_nothing(self) -> None:
        assert Jitter().achieved() == {}

    @pytest.mark.parametrize(
        ("field", "value"), [("mean_ms", -1.0), ("std_ms", -1.0), ("buffer_ms", -1.0)]
    )
    def test_nonsense_is_refused(self, field: str, value: float) -> None:
        with pytest.raises(JitterError):
            Jitter(**{field: value})


# -------------------------------------------------------------------- world


class TestL6LandsOnTheToolBoundary:
    def test_an_empty_plan_touches_nothing(self) -> None:
        plan = WorldPlan()
        assert plan.is_empty
        assert plan.before_call("lookup_patient").is_clean

    def test_each_axis_has_its_own_key(self) -> None:
        plan = WorldPlan(
            latency={"lookup": LatencyProfile(p50_ms=300.0, p99_ms=4000.0)},
            error_rate=0.1,
            outages=(Outage(start_s=5.0, duration_s=10.0),),
        )
        assert {str(k) for k in plan.active_layers} == {
            "L6.tool_latency",
            "L6.tool_error",
            "L6.backend_outage",
        }

    def test_it_reaches_the_pipeline_without_touching_audio(self) -> None:
        pipeline = PerturbationPipeline(
            seeds=SeedSet.derive(1),
            world=WorldPlan(latency={"*": LatencyProfile(p50_ms=100.0, p99_ms=900.0)}),
        )
        assert pipeline.is_clean, "L6 degrades no signal"
        assert not pipeline.is_unperturbed
        assert {str(k) for k in pipeline.active_layers} == {"L6.tool_latency"}


class TestLatency:
    def test_a_wildcard_covers_tools_with_no_entry(self) -> None:
        """A task should not have to enumerate fifty-nine tools."""
        plan = WorldPlan(latency={"*": LatencyProfile(p50_ms=200.0, p99_ms=200.0)})
        assert plan.before_call("anything").delay_ms == pytest.approx(200.0)

    def test_a_named_tool_wins_over_the_wildcard(self) -> None:
        plan = WorldPlan(
            latency={
                "*": LatencyProfile(p50_ms=10.0, p99_ms=10.0),
                "slow": LatencyProfile(p50_ms=900.0, p99_ms=900.0),
            }
        )
        assert plan.before_call("slow").delay_ms == pytest.approx(900.0)

    def test_the_tail_is_right_skewed_not_uniform(self) -> None:
        """Sampling flat between p50 and p99 makes the tail as likely as
        the typical case, which is not what a service does."""
        plan = WorldPlan(
            latency={"*": LatencyProfile(p50_ms=300.0, p99_ms=4000.0)},
            seeds=SeedSet.derive(5),
        )
        draws = sorted(plan.before_call("t").delay_ms for _ in range(400))
        median = draws[200]
        assert 150.0 < median < 600.0
        assert draws[-1] > median * 3

    def test_a_percentile_pair_that_is_not_a_distribution_is_refused(self) -> None:
        with pytest.raises(WorldError, match="not a distribution"):
            LatencyProfile(p50_ms=900.0, p99_ms=100.0)


class TestErrorsAndOutages:
    def test_failure_classes_say_whether_retrying_is_right(self) -> None:
        """An agent retrying a timeout is behaving; retrying a not_found
        is looping. One 'error' outcome cannot tell them apart."""
        assert ToolFailure.TIMEOUT.is_retryable
        assert ToolFailure.UNAVAILABLE.is_retryable
        assert not ToolFailure.NOT_FOUND.is_retryable
        assert not ToolFailure.INVALID.is_retryable

    def test_an_outage_takes_every_tool(self) -> None:
        plan = WorldPlan(outages=(Outage(start_s=1.0, duration_s=5.0),))
        inside = plan.before_call("anything", at_s=3.0)
        assert inside.failed
        assert "outage" in inside.reason

    def test_outside_the_window_nothing_happens(self) -> None:
        plan = WorldPlan(outages=(Outage(start_s=1.0, duration_s=5.0),))
        assert plan.before_call("anything", at_s=8.0).is_clean

    def test_an_outage_beats_latency(self) -> None:
        """A backend cannot be simultaneously down and responsive."""
        plan = WorldPlan(
            latency={"*": LatencyProfile(p50_ms=500.0, p99_ms=500.0)},
            outages=(Outage(start_s=0.0, duration_s=10.0),),
        )
        outcome = plan.before_call("t", at_s=1.0)
        assert outcome.failed
        assert outcome.delay_ms == 0.0

    def test_an_error_rate_with_nothing_to_fail_with_is_refused(self) -> None:
        with pytest.raises(WorldError, match="nothing to fail with"):
            WorldPlan(error_rate=0.5, failures=())

    def test_an_out_of_range_rate_is_refused(self) -> None:
        with pytest.raises(WorldError, match="out of range"):
            WorldPlan(error_rate=1.5)


class TestWhatTheAgentActuallyMet:
    def test_the_observed_rate_exposes_a_run_that_stayed_clean(self) -> None:
        """A 5% rate over eleven calls may inject nothing, and the run would
        be a clean run wearing a degraded profile's name."""
        plan = WorldPlan(
            error_rate=0.05, failures=(ToolFailure.TIMEOUT,), seeds=SeedSet.derive(1)
        )
        for _ in range(11):
            plan.before_call("t")
        achieved = plan.achieved()
        assert achieved["calls"] == 11.0
        assert "observed_error_rate" in achieved

    def test_nothing_is_claimed_before_a_call_is_made(self) -> None:
        assert WorldPlan(error_rate=0.5).achieved() == {}

    def test_the_same_seed_fails_the_same_calls(self) -> None:
        """Seeding on the tool boundary."""
        runs = []
        for _ in range(2):
            plan = WorldPlan(
                error_rate=0.4,
                failures=(ToolFailure.TIMEOUT, ToolFailure.RATE_LIMITED),
                seeds=SeedSet.derive(13),
            )
            runs.append(
                [
                    (o.failure.value if o.failure else None)
                    for o in (plan.before_call("t") for _ in range(30))
                ]
            )
        assert runs[0] == runs[1]

    def test_the_written_plan_carries_both_sides(self) -> None:
        pipeline = PerturbationPipeline(
            seeds=SeedSet.derive(1),
            world=WorldPlan(
                latency={"lookup": LatencyProfile(p50_ms=300.0, p99_ms=4000.0)}
            ),
        )
        pipeline.world.before_call("lookup")
        entry = pipeline.as_record().as_dict()["layers"]["L6.tool_latency"]
        assert entry["params"]["latency"]["lookup"]["p99_ms"] == 4000.0
        assert entry["achieved"]["delayed"] == 1.0


class TestEveryLayerNowExists:
    def test_all_six_are_reachable(self) -> None:
        """The audit that started this: three of six were missing."""
        pipeline = PerturbationPipeline(
            [MuLawCodec(), Jitter()],
            seeds=SeedSet.derive(1),
            world=WorldPlan(error_rate=0.1),
        )
        assert Layer.CHANNEL in {k.layer for k in pipeline.active_layers}
        assert Layer.WORLD in {k.layer for k in pipeline.active_layers}

    def test_l6_is_paired_with_a_metric(self) -> None:
        """A perturbation with no reader only makes the number worse."""
        from harbor_voice.metrics import REGISTRY

        readers = [
            name
            for name in REGISTRY.names()
            if Layer.WORLD in REGISTRY.get(name).sensitive_to
        ]
        assert readers, "no registered metric moves with L6"

    def test_l4_jitter_is_paired_with_a_metric(self) -> None:
        from harbor_voice.metrics import REGISTRY

        readers = [
            name
            for name in REGISTRY.names()
            if Layer.CHANNEL in REGISTRY.get(name).sensitive_to
        ]
        assert readers
