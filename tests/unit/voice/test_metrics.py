"""Unit tests for metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.capabilities import Verdict
from harbor_voice.metrics import REGISTRY
from harbor_voice.metrics import Direction
from harbor_voice.metrics import Family
from harbor_voice.metrics import MetricContext
from harbor_voice.metrics import MetricRegistry
from harbor_voice.metrics import TrialResults
from harbor_voice.metrics import Value
from harbor_voice.metrics import bootstrap_mean
from harbor_voice.metrics import build_outputs
from harbor_voice.metrics import build_reward
from harbor_voice.metrics import edit_counts
from harbor_voice.metrics import evaluate_metric
from harbor_voice.metrics import metric
from harbor_voice.metrics import paired_delta
from harbor_voice.metrics import roll_up
from harbor_voice.metrics import word_error_rate
from harbor_voice.providers.cache import Cassette
from harbor_voice.providers.cache import CassetteMode
from harbor_voice.providers.cache import ReplayedRunError
from harbor_voice.providers.base import Usage
from harbor_voice.trajectory.model import Clocks
from harbor_voice.trajectory.model import ConfigMatrix
from harbor_voice.trajectory.model import EventType
from harbor_voice.trajectory.model import PartyRecord
from harbor_voice.trajectory.model import PartyRole
from harbor_voice.trajectory.model import StageRecord
from harbor_voice.trajectory.model import TimelineEvent
from harbor_voice.trajectory.model import ValidityCheck
from harbor_voice.trajectory.model import ValidityRecord
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.model import VoiceTurn

pytestmark = pytest.mark.unit


CASCADE_CAPS = Capabilities(
    duplex=frozenset({Duplex.HALF, Duplex.FULL}),
    pipeline=Pipeline.CASCADE,
    turn_control=TurnControl.NATIVE_VAD,
    exposes_asr=True,
    exposes_partials=True,
    exposes_stages=frozenset(
        {Stage.ASR_PARTIAL, Stage.ASR_FINAL, Stage.TTS_FIRST_BYTE}
    ),
    can_be_interrupted=True,
    attested=frozenset({"exposes_asr", "can_be_interrupted"}),
)


def trajectory(
    *,
    turns: tuple[VoiceTurn, ...] = (),
    timeline: tuple[TimelineEvent, ...] = (),
    caps: Capabilities | None = CASCADE_CAPS,
    floor: str = "full",
    layers: dict[str, object] | None = None,
    validity: ValidityRecord | None = None,
    fidelity: dict[str, object] | None = None,
) -> VoiceTrajectory:
    return VoiceTrajectory(
        parties=(
            PartyRecord(
                id="agent", role=PartyRole.SYSTEM_UNDER_TEST, capabilities=caps
            ),
            PartyRecord(id="caller", role=PartyRole.SIMULATED_USER),
        ),
        config_matrix=ConfigMatrix(
            duplex=Duplex.FULL,
            pipeline=Pipeline.CASCADE,
            floor_policy=floor,
            profile="telephony",
            seed_root=7,
        ),
        perturbation_plan={"layers": layers or {}},
        turns=turns,
        timeline=timeline,
        validity=validity or ValidityRecord(),
        simulator_fidelity=fidelity or {},
    )


def ctx(**kwargs: object) -> MetricContext:
    return MetricContext(trajectory=trajectory(**kwargs))


# --------------------------------------------------------------------- text


class TestWordErrorRate:
    def test_a_perfect_transcript_scores_zero(self) -> None:
        assert word_error_rate("the patient is stable", "The patient is stable.") == 0.0

    def test_case_and_edge_punctuation_do_not_count_as_errors(self) -> None:
        assert word_error_rate("Hello, world!", "hello world") == 0.0

    def test_an_empty_reference_is_undefined_not_perfect(self) -> None:
        """0.0 would report a flawless transcription of nothing."""
        assert word_error_rate("", "hello") is None

    def test_the_three_error_types_are_kept_apart(self) -> None:
        counts = edit_counts("take two pills daily", "take to pills")
        assert counts.substitutions == 1
        assert counts.deletions == 1
        assert counts.insertions == 0
        assert counts.reference_words == 4
        assert counts.rate == pytest.approx(0.5)

    def test_insertions_are_counted(self) -> None:
        counts = edit_counts("take two pills", "take two more pills")
        assert counts.insertions == 1
        assert counts.substitutions == 0

    def test_a_rate_above_one_is_not_clamped(self) -> None:
        """A hypothesis longer than the reference is a real outcome."""
        rate = word_error_rate("yes", "yes i think that is right")
        assert rate is not None and rate > 1.0

    def test_numbers_are_not_normalised_away(self) -> None:
        """Aggressive normalisation hides the errors a bad line produces."""
        assert word_error_rate("take 2 pills", "take two pills") == pytest.approx(1 / 3)


# ------------------------------------------------- the abstention rule


class TestAMetricOutsideItsPreconditionsIsNeverCalled:
    """Stronger than returning a sentinel: the body must not run."""

    def test_the_function_body_does_not_execute(self) -> None:
        registry = MetricRegistry()
        calls: list[int] = []

        @metric(
            name="needs_full_duplex",
            family=Family.TURNS,
            direction=Direction.HIGHER_BETTER,
            requires=["duplex:full"],
            registry=registry,
        )
        def _m(context: MetricContext) -> Value:
            calls.append(1)
            return Value(value=1.0)

        half = Capabilities(
            duplex=frozenset({Duplex.HALF}),
            pipeline=Pipeline.S2S,
            turn_control=TurnControl.HARNESS_VAD,
        )
        result = evaluate_metric(
            registry.get("needs_full_duplex"),
            MetricContext(trajectory=trajectory(caps=half, floor="half")),
        )
        assert result.verdict is Verdict.NOT_APPLICABLE
        assert result.value is None
        assert calls == []

    def test_a_typo_in_requires_is_an_error_at_decoration(self) -> None:
        """A metric that is quietly always N/A is worse than one that crashes."""
        registry = MetricRegistry()
        with pytest.raises(ValueError, match="precondition"):

            @metric(
                name="typo",
                family=Family.TURNS,
                direction=Direction.HIGHER_BETTER,
                requires=["duplexfull"],
                registry=registry,
            )
            def _m(context: MetricContext) -> Value:
                return Value(value=1.0)

    def test_an_undeclared_agent_is_not_a_permissive_one(self) -> None:
        result = evaluate_metric(
            REGISTRY.get("divergence.wer_said_heard"),
            MetricContext(trajectory=trajectory(caps=None)),
        )
        assert result.verdict is Verdict.NOT_APPLICABLE
        assert "no agent party declared capabilities" in result.reasons[0]

    def test_an_unattested_capability_reports_but_cannot_publish(self) -> None:
        caps = CASCADE_CAPS.model_copy(update={"attested": frozenset()})
        turns = (
            VoiceTurn(idx=0, party="caller", said="hello there", heard="hello there"),
        )
        result = evaluate_metric(
            REGISTRY.get("divergence.wer_said_heard"),
            MetricContext(trajectory=trajectory(turns=turns, caps=caps)),
        )
        assert result.verdict is Verdict.UNATTESTED
        assert result.reported
        assert not result.publishable

    def test_abstaining_for_data_is_different_from_abstaining_for_setup(self) -> None:
        """Reached, looked, found nothing — not 'this could never work'."""
        result = evaluate_metric(
            REGISTRY.get("divergence.wer_said_heard"), ctx(turns=())
        )
        assert result.verdict is Verdict.NOT_APPLICABLE
        assert result.reasons == ("no turn carried both said and heard",)


# ----------------------------------------------------------------- families


class TestDivergence:
    def test_wer_is_pooled_not_averaged_per_turn(self) -> None:
        """A per-turn mean lets one misheard 'yes' dominate a whole call."""
        turns = (
            VoiceTurn(idx=0, party="caller", said="yes", heard="no"),
            VoiceTurn(
                idx=1,
                party="caller",
                said="i would like to reschedule my appointment for next tuesday",
                heard="i would like to reschedule my appointment for next tuesday",
            ),
        )
        result = evaluate_metric(
            REGISTRY.get("divergence.wer_said_heard"), ctx(turns=turns)
        )
        # 1 error over 11 reference words, not the 0.5 a per-turn mean gives.
        assert result.value == pytest.approx(1 / 11)
        assert result.n == 2
        assert result.detail["substitutions"] == 1

    def test_a_turn_missing_one_side_is_skipped(self) -> None:
        turns = (
            VoiceTurn(idx=0, party="caller", said="hello", heard=None),
            VoiceTurn(idx=1, party="caller", said="goodbye", heard="goodbye"),
        )
        result = evaluate_metric(
            REGISTRY.get("divergence.wer_said_heard"), ctx(turns=turns)
        )
        assert result.n == 1
        assert result.value == 0.0


class TestTiming:
    def test_response_latency_is_the_median(self) -> None:
        """One stall should not drag the typical experience."""
        turns = tuple(
            VoiceTurn(idx=i, party="agent", clocks=Clocks(tx=0.0, rx=rx))
            for i, rx in enumerate((0.2, 0.3, 9.0))
        )
        result = evaluate_metric(
            REGISTRY.get("timing.response_latency_s"), ctx(turns=turns)
        )
        assert result.value == pytest.approx(0.3)
        assert result.detail["mean"] > result.value
        assert result.detail["clock_pair"] == "tx_to_rx"

    def test_a_turn_without_both_clocks_is_not_guessed(self) -> None:
        turns = (VoiceTurn(idx=0, party="agent", clocks=Clocks(tx=0.0)),)
        result = evaluate_metric(
            REGISTRY.get("timing.response_latency_s"), ctx(turns=turns)
        )
        assert result.verdict is Verdict.NOT_APPLICABLE

    def test_first_byte_needs_the_stage_to_be_emitted(self) -> None:
        """An S2S party has no such seam; asking gives its total response time."""
        s2s = Capabilities(
            duplex=frozenset({Duplex.FULL}),
            pipeline=Pipeline.S2S,
            turn_control=TurnControl.NATIVE_VAD,
        )
        result = evaluate_metric(REGISTRY.get("timing.tts_first_byte_s"), ctx(caps=s2s))
        assert result.verdict is Verdict.NOT_APPLICABLE
        assert "tts_first_byte" in result.reasons[0]

    def test_first_byte_is_measured_from_the_turn_start(self) -> None:
        turns = (
            VoiceTurn(
                idx=0,
                party="agent",
                stages=(
                    StageRecord(stage="asr_final", t=10.0, clock="rx"),
                    StageRecord(stage="tts_first_byte", t=10.4, clock="produced"),
                ),
            ),
        )
        result = evaluate_metric(
            REGISTRY.get("timing.tts_first_byte_s"), ctx(turns=turns)
        )
        assert result.value == pytest.approx(0.4)


class TestTurnTaking:
    def test_overlap_is_neutral_because_polarity_is_task_dependent(self) -> None:
        """Rudeness in an intake call, attentiveness on a crisis line."""
        assert REGISTRY.get("turns.overlap_s").direction is Direction.NEUTRAL
        assert REGISTRY.get("turns.overlap_ratio").direction is Direction.NEUTRAL

    def test_no_overlap_is_zero_not_abstention(self) -> None:
        """Zero overlap is a measurement; the call simply had none."""
        result = evaluate_metric(REGISTRY.get("turns.overlap_s"), ctx())
        assert result.value == 0.0
        assert result.verdict is not Verdict.NOT_APPLICABLE

    def test_barge_in_separates_late_from_never(self) -> None:
        timeline = (
            TimelineEvent(
                t=1.0, type=EventType.BARGE_IN, meta={"yielded_in_time": True}
            ),
            TimelineEvent(
                t=2.0, type=EventType.BARGE_IN, meta={"yielded_in_time": False}
            ),
            TimelineEvent(t=3.0, type=EventType.BARGE_IN, meta={}),
        )
        result = evaluate_metric(
            REGISTRY.get("turns.barge_in_success_rate"), ctx(timeline=timeline)
        )
        assert result.value == pytest.approx(1 / 3)
        assert result.detail == {
            "yielded_in_time": 1,
            "yielded_late": 1,
            "never": 1,
        }

    def test_an_interruption_that_never_fired_leaves_the_denominator(self) -> None:
        """Not the agent's failure if the probe never reached it."""
        timeline = (
            TimelineEvent(
                t=1.0, type=EventType.BARGE_IN, meta={"yielded_in_time": True}
            ),
            TimelineEvent(
                t=2.0, type=EventType.BARGE_IN, meta={"outcome": "no_adapter"}
            ),
        )
        result = evaluate_metric(
            REGISTRY.get("turns.barge_in_success_rate"), ctx(timeline=timeline)
        )
        assert result.value == 1.0
        assert result.n == 1

    def test_yield_latency_excludes_the_ones_never_yielded_to(self) -> None:
        """Every stand-in duration for 'never' is wrong, so none is invented."""
        timeline = (
            TimelineEvent(
                t=1.0, type=EventType.BARGE_IN, meta={"yield_latency_s": 0.3}
            ),
            TimelineEvent(t=2.0, type=EventType.BARGE_IN, meta={}),
        )
        result = evaluate_metric(
            REGISTRY.get("timing.yield_latency_s"), ctx(timeline=timeline)
        )
        assert result.value == pytest.approx(0.3)
        assert result.n == 1


class TestSelectivityIsInertWithoutItsPerturbation:
    """The perturbation is the probe that makes the metric possible."""

    def test_it_abstains_when_the_layer_is_off(self) -> None:
        result = evaluate_metric(REGISTRY.get("selectivity.backchannel"), ctx())
        assert result.verdict is Verdict.NOT_APPLICABLE
        assert "L5.backchannel" in result.reasons[0]

    def test_it_measures_when_the_layer_is_on(self) -> None:
        timeline = (
            TimelineEvent(
                t=1.0,
                type=EventType.SPEECH_START,
                meta={"probe": "backchannel", "ignored": True},
            ),
            TimelineEvent(
                t=2.0,
                type=EventType.SPEECH_START,
                meta={"probe": "backchannel", "ignored": False},
            ),
        )
        result = evaluate_metric(
            REGISTRY.get("selectivity.backchannel"),
            ctx(timeline=timeline, layers={"L5.backchannel": {}}),
        )
        assert result.value == pytest.approx(0.5)
        assert result.n == 2


class TestValidityFamily:
    def test_unmeasured_fidelity_abstains_rather_than_scoring_perfect(self) -> None:
        """A default of 1.0 is how an invalid run passes its own gate."""
        result = evaluate_metric(
            REGISTRY.get("validity.simulator_in_persona"), ctx(fidelity={})
        )
        assert result.verdict is Verdict.NOT_APPLICABLE
        assert result.value is None

    def test_a_measured_fidelity_reports(self) -> None:
        result = evaluate_metric(
            REGISTRY.get("validity.simulator_in_persona"),
            ctx(fidelity={"in_persona": 0.91}),
        )
        assert result.value == pytest.approx(0.91)

    def test_checks_passed_names_the_failures(self) -> None:
        record = ValidityRecord(
            measurement_valid=False,
            checks=(
                ValidityCheck(name="clean_tx_parity", ok=True),
                ValidityCheck(
                    name="transport_gap_budget", ok=False, detail="1.4s > 1.0s"
                ),
            ),
        )
        result = evaluate_metric(
            REGISTRY.get("validity.checks_passed"), ctx(validity=record)
        )
        assert result.value == pytest.approx(0.5)
        assert result.detail["failed"] == ["transport_gap_budget"]


# ------------------------------------------------------------- rollup


def trial(value: float | None, *, valid: bool = True, verdict: Verdict | None = None):
    from harbor_voice.metrics.base import MetricResult

    return TrialResults(
        results=[
            MetricResult(
                name="m",
                family=Family.TASK,
                direction=Direction.HIGHER_BETTER,
                verdict=verdict
                or (Verdict.SATISFIED if value is not None else Verdict.NOT_APPLICABLE),
                value=value,
            )
        ],
        measurement_valid=valid,
    )


class TestRollup:
    def test_median_and_failure_rate_sit_beside_the_mean(self) -> None:
        """The mean is largely a failure-rate statistic; say so."""
        values = [0.9, 0.92, 0.88, 0.91, 0.02, 0.01]
        rollup = roll_up(
            "m", [trial(v) for v in values], failure_threshold=0.1, bootstrap=False
        )
        assert rollup is not None
        assert rollup.median == pytest.approx(0.89)
        assert rollup.mean < rollup.median
        assert rollup.failure_rate == pytest.approx(2 / 6)

    def test_an_invalid_trial_leaves_the_denominator(self) -> None:
        """Harbor would average it in as a zero."""
        trials = [trial(0.9), trial(0.9), trial(None, valid=False)]
        rollup = roll_up("m", trials, bootstrap=False)
        assert rollup is not None
        assert rollup.n_trials == 3
        assert rollup.n_invalid == 1
        assert rollup.n_reported == 2
        assert rollup.mean == pytest.approx(0.9)

    def test_a_not_applicable_trial_is_counted_not_zeroed(self) -> None:
        rollup = roll_up("m", [trial(0.8), trial(None)], bootstrap=False)
        assert rollup is not None
        assert rollup.n_not_applicable == 1
        assert rollup.mean == pytest.approx(0.8)
        assert rollup.applicable_rate == pytest.approx(0.5)

    def test_one_unattested_trial_makes_the_rollup_unpublishable(self) -> None:
        trials = [trial(0.8), trial(0.9, verdict=Verdict.UNATTESTED)]
        rollup = roll_up("m", trials, bootstrap=False)
        assert rollup is not None
        assert rollup.n_unpublishable == 1
        assert not rollup.publishable

    def test_neutral_metrics_have_no_failure_notion(self) -> None:
        """Overlap of 0.4 is a description, not a catastrophe."""
        from harbor_voice.metrics.rollup import is_failure

        assert not is_failure(0.4, 0.9, Direction.NEUTRAL)
        assert is_failure(0.4, 0.9, Direction.HIGHER_BETTER)
        assert is_failure(0.95, 0.9, Direction.LOWER_BETTER)

    def test_an_unknown_metric_rolls_up_to_nothing(self) -> None:
        assert roll_up("absent", [trial(0.5)]) is None


class TestBootstrap:
    def test_it_is_seeded_so_rescoring_reproduces(self) -> None:
        """An unseeded CI changes on every regrade and nobody trusts the table."""
        values = [0.9, 0.2, 0.85, 0.4, 0.95]
        first = bootstrap_mean(values, samples=200)
        second = bootstrap_mean(values, samples=200)
        assert first == second

    def test_one_observation_gives_no_interval(self) -> None:
        """A CI over one point is not a wide interval; it is no interval."""
        assert bootstrap_mean([0.9]) is None

    def test_the_interval_brackets_the_mean(self) -> None:
        values = [0.9, 0.2, 0.85, 0.4, 0.95, 0.1, 0.75]
        result = bootstrap_mean(values, samples=500)
        assert result is not None
        mean = sum(values) / len(values)
        assert result.low <= mean <= result.high
        assert result.width > 0

    def test_a_tight_sample_gives_a_tight_interval(self) -> None:
        tight = bootstrap_mean([0.90, 0.91, 0.89, 0.90], samples=500)
        loose = bootstrap_mean([0.1, 0.9, 0.2, 0.95], samples=500)
        assert tight is not None and loose is not None
        assert tight.width < loose.width


class TestPairedDelta:
    def test_delta_is_on_the_raw_scale_not_a_percentage(self) -> None:
        """0.02 -> 0.04 is '+100%' and means two words."""
        degraded = roll_up("m", [trial(0.40), trial(0.44)], bootstrap=False)
        clean = roll_up("m", [trial(0.02), trial(0.04)], bootstrap=False)
        delta = paired_delta("m", degraded, clean)
        assert delta is not None
        assert delta.mean == pytest.approx(0.39)
        assert delta.family is Family.ROBUSTNESS
        assert delta.name == "robustness.delta.m"

    def test_a_missing_baseline_is_not_zero(self) -> None:
        degraded = roll_up("m", [trial(0.4)], bootstrap=False)
        assert paired_delta("m", degraded, None) is None


# ----------------------------------------------------- three outputs


def results_for(traj: VoiceTrajectory, *names: str):
    return REGISTRY.evaluate(list(names), MetricContext(trajectory=traj))


def a_scored_trajectory() -> VoiceTrajectory:
    return trajectory(
        turns=(
            VoiceTurn(
                idx=0,
                party="caller",
                said="i need to reschedule",
                heard="i need to reschedule",
                clocks=Clocks(tx=0.0, rx=0.4),
            ),
        ),
        timeline=(
            TimelineEvent(t=0.0, type=EventType.SESSION_OPEN),
            TimelineEvent(t=4.0, type=EventType.SESSION_CLOSE),
        ),
        validity=ValidityRecord(
            checks=(ValidityCheck(name="clean_tx_parity", ok=True),)
        ),
    )


class TestRewardStaysHarborShaped:
    def test_it_is_a_flat_dict_of_floats(self) -> None:
        """harbor's aggregate_reward_dicts must work with zero modification."""
        results = results_for(
            a_scored_trajectory(),
            "divergence.wer_said_heard",
            "timing.response_latency_s",
        )
        reward = build_reward(results)
        assert set(reward) == {
            "divergence.wer_said_heard",
            "timing.response_latency_s",
        }
        assert all(isinstance(v, float) for v in reward.values())

    def test_harbors_aggregator_accepts_it_unchanged(self) -> None:
        from harbor.metrics.base import aggregate_reward_dicts

        reward = build_reward(
            results_for(
                a_scored_trajectory(),
                "divergence.wer_said_heard",
                "timing.response_latency_s",
            )
        )
        aggregated = aggregate_reward_dicts(
            [reward, reward], metric_name="mean", aggregate=lambda v: sum(v) / len(v)
        )
        assert aggregated["timing.response_latency_s"] == pytest.approx(0.4)

    def test_an_inapplicable_metric_is_omitted_not_zeroed(self) -> None:
        results = results_for(a_scored_trajectory(), "selectivity.backchannel")
        assert build_reward(results) == {}


class TestMeasurementCarriesWhatRewardCannot:
    def test_it_records_the_four_columns(self) -> None:
        from harbor_voice.metrics import build_measurement

        results = results_for(
            a_scored_trajectory(),
            "divergence.wer_said_heard",
            "selectivity.backchannel",
        )
        doc = build_measurement(results, trial_id="t1", agent="cascade")
        assert doc["counts"] == {
            "reported": 1,
            "not_applicable": 1,
            "unattested": 0,
            "publishable": 1,
        }
        blocked = doc["metrics"]["selectivity.backchannel"]
        assert blocked["verdict"] == "not_applicable"
        assert blocked["value"] is None
        assert blocked["reasons"]


class TestInvalidTrialsWriteNoReward:
    """Zeros would be indistinguishable from a bad agent."""

    def test_an_invalid_trial_has_no_reward(self) -> None:
        traj = a_scored_trajectory()
        invalid = ValidityRecord(
            checks=(
                ValidityCheck(
                    name="transport_gap_budget", ok=False, detail="1.4s > 1.0s"
                ),
            )
        )
        outputs = build_outputs(results_for(traj, "divergence.wer_said_heard"), invalid)
        assert not outputs.measurement_valid
        assert outputs.reward is None
        assert outputs.validity["reasons"] == ["1.4s > 1.0s"]

    def test_a_valid_trial_has_one(self) -> None:
        traj = a_scored_trajectory()
        outputs = build_outputs(
            results_for(traj, "divergence.wer_said_heard"), traj.validity
        )
        assert outputs.measurement_valid
        assert outputs.reward == {"divergence.wer_said_heard": 0.0}

    def test_a_flag_that_disagrees_with_its_checks_loses(self) -> None:
        """A validity file saying true above a failed check is worse than none."""
        lying = ValidityRecord(
            measurement_valid=True,
            checks=(ValidityCheck(name="clean_tx_parity", ok=False),),
        )
        outputs = build_outputs([], lying)
        assert not outputs.measurement_valid

    def test_writing_removes_a_stale_reward_file(self, tmp_path: Path) -> None:
        (tmp_path / "reward.json").write_text('{"old": 1.0}', encoding="utf-8")
        invalid = ValidityRecord(
            checks=(ValidityCheck(name="x", ok=False, detail="nope"),)
        )
        build_outputs([], invalid).write(tmp_path)
        assert not (tmp_path / "reward.json").exists()
        assert (
            json.loads((tmp_path / "validity.json").read_text())["measurement_valid"]
            is False
        )

    def test_a_valid_trial_writes_all_three(self, tmp_path: Path) -> None:
        traj = a_scored_trajectory()
        written = build_outputs(
            results_for(traj, "divergence.wer_said_heard"), traj.validity
        ).write(tmp_path)
        assert {p.name for p in written} == {
            "measurement.json",
            "validity.json",
            "reward.json",
        }


class TestAReplayedRunCannotBecomeAMeasurement:
    """Live-only scoring, enforced where a number becomes a published artifact."""

    def test_a_replayed_usage_blocks_the_outputs(self) -> None:
        traj = a_scored_trajectory()
        with pytest.raises(ReplayedRunError):
            build_outputs(
                results_for(traj, "divergence.wer_said_heard"),
                traj.validity,
                usages=[Usage(replayed=True)],
            )

    def test_a_used_cassette_blocks_the_outputs(self) -> None:
        traj = a_scored_trajectory()
        cassette = Cassette(mode=CassetteMode.REPLAY)
        cassette.record("k", [b"a"])
        cassette.replay("k")
        with pytest.raises(ReplayedRunError):
            build_outputs([], traj.validity, cassette=cassette)

    def test_a_simulator_cache_hit_does_not(self) -> None:
        traj = a_scored_trajectory()
        outputs = build_outputs([], traj.validity, usages=[Usage(cache_hit=True)])
        assert outputs.measurement_valid
