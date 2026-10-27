"""The whole chain, as one test.

simulator -> synthesis -> perturbation -> transport -> cascade -> turn

Unit tests prove each layer in isolation. This proves they compose, which is
the thing that is actually in doubt. Done is not every layer passing its own
tests; it is the top of the stack running a real voice call end to end, green.

No network. Deterministic. Runs in under a second.
"""

from __future__ import annotations

import pytest

from harbor_voice.audio.stages import Stage
from harbor_voice.cascade import CascadeAdapter
from harbor_voice.cascade import CascadeTransport
from harbor_voice.providers import EchoLLM
from harbor_voice.providers import ScriptedSTT
from harbor_voice.providers import ToneTTS
from harbor_voice.ids import PartyId
from harbor_voice.ids import PlanId
from harbor_voice.perturbation import BackgroundBed
from harbor_voice.perturbation import GilbertElliottLoss
from harbor_voice.perturbation import MuLawCodec
from harbor_voice.perturbation import PerturbationPipeline
from harbor_voice.perturbation import SeedSet
from harbor_voice.perturbation import synthetic_bed
from harbor_voice.simulator import ScriptContent
from harbor_voice.simulator import UniversalSimulator
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import Verdict
from harbor_voice.metrics import REGISTRY
from harbor_voice.metrics import MetricContext
from harbor_voice.metrics import build_outputs
from harbor_voice.trajectory import Recorder
from harbor_voice.trajectory.model import ConfigMatrix
from harbor_voice.trajectory.model import PartyRecord
from harbor_voice.trajectory.model import PartyRole
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.transport import SessionConfig

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
AGENT = PartyId("agent")
TURNS = ["I need to reschedule my appointment", "next Tuesday if possible"]


def _telephony_pipeline(seed: int) -> PerturbationPipeline:
    """The `telephony` profile: a noisy room down an 8 kHz line."""
    return PerturbationPipeline(
        [
            BackgroundBed(
                source=synthetic_bed(seconds=1.0, seed=1),
                target_snr_db=15.0,
                snr_drift_db=2.0,
                variation_speed=20.0,
            ),
            MuLawCodec(),
            GilbertElliottLoss(rate=0.1, burst_duration_ms=200.0),
        ],
        seeds=SeedSet.derive(seed),
        plan_id=PlanId("telephony"),
    )


async def _run_call(
    *, seed: int, clean: bool = False
) -> tuple[UniversalSimulator, CascadeTransport, list]:
    pipeline = (
        PerturbationPipeline(seeds=SeedSet.derive(seed))
        if clean
        else _telephony_pipeline(seed)
    )
    simulator = UniversalSimulator(
        party=CALLER,
        content=ScriptContent(turns=TURNS),
        tts=ToneTTS(),
        pipeline=pipeline,
        seeds=SeedSet.derive(seed),
    )
    # One clock for the whole call. The simulator stamps t_produced and the
    # adapter stamps t_tx; with two SessionClocks the later-started one reads
    # lower, and on the clean profile (no processing between the two stamps)
    # a send can appear to precede its own synthesis by microseconds.
    transport = CascadeTransport(
        stt=ScriptedSTT(turns=TURNS),
        llm=EchoLLM(prefix="understood: "),
        tts=ToneTTS(),
        party=AGENT,
        input_idle_s=0.05,
        clock=simulator.clock,
    )
    adapter = CascadeAdapter(
        transport,
        party=AGENT,
        idle_s=0.1,
        turn_timeout_s=3.0,
        clock=simulator.clock,
    )
    await adapter.open(SessionConfig(party=AGENT))

    turns = []
    heard: list[str] = []
    while (utterance := await simulator.next_utterance(heard)) is not None:
        turn = await adapter.call(utterance.frames)
        for frame in utterance.frames:
            pipeline.verify(frame)
        heard.append(transport.transcripts[-1].text)
        turns.append(turn)
    await adapter.close()
    return simulator, transport, turns


class TestFullChain:
    async def test_a_two_turn_call_completes(self) -> None:
        simulator, transport, turns = await _run_call(seed=2026)
        assert len(turns) == 2
        assert all(t.is_complete for t in turns)
        assert all(t.received for t in turns)
        assert len(simulator.utterances) == 2

    async def test_every_frame_keeps_its_clean_twin(self) -> None:
        """The clean reference, asserted across a real call rather than a single frame."""
        simulator, _, _ = await _run_call(seed=2026)
        store = simulator.pipeline.clean_store
        for utterance in simulator.utterances:
            assert utterance.clean_digest is not None
            assert store.get(utterance.clean_digest)
            for frame in utterance.frames:
                assert frame.clean_ref is not None
                assert len(store.get(frame.clean_ref)) == len(frame.pcm)

    async def test_both_sides_of_said_versus_heard_exist(self) -> None:
        """The reference and the hypothesis, from one call."""
        simulator, transport, _ = await _run_call(seed=2026)
        said = [u.said for u in simulator.utterances]
        heard = [t.text for t in transport.transcripts]
        assert len(said) == len(heard) == 2
        assert said == TURNS

    async def test_the_cascade_exposes_its_seams_through_the_whole_chain(self) -> None:
        _, _, turns = await _run_call(seed=2026)
        for turn in turns:
            stages = {m.stage for m in turn.stages}
            assert Stage.ASR_FINAL in stages
            assert Stage.LLM_FIRST_TOKEN in stages
            assert Stage.TTS_FIRST_BYTE in stages

    async def test_latency_is_measured_with_named_clocks(self) -> None:
        """No number without its clock pair, end to end."""
        _, _, turns = await _run_call(seed=2026)
        for turn in turns:
            latency = turn.response_latency
            assert latency is not None
            assert latency.named("response") == "response_tx_to_rx"
            assert latency.seconds > 0

    async def test_the_plan_records_what_actually_happened(self) -> None:
        """A run can state its own conditions."""
        simulator, _, _ = await _run_call(seed=2026)
        record = simulator.pipeline.as_record().as_dict()

        assert record["plan_id"] == "telephony"
        assert record["seeds"]["root"] == 2026
        assert record["frames_processed"] > 0

        bed = record["layers"]["L3.background_bed"]
        assert bed["params"]["target_snr_db"] == 15.0
        # The drift band is +/-2 dB and a short call does not average it out,
        # so the mean is asserted against the band rather than the target. A
        # tighter bound would be asserting that the drift we asked for did not
        # happen.
        achieved = bed["achieved"]
        drift = bed["params"]["snr_drift_db"]
        assert abs(achieved["achieved_snr_db_mean"] - 15.0) <= drift
        assert achieved["achieved_snr_db_min"] >= 15.0 - drift - 0.5
        assert achieved["achieved_snr_db_max"] <= 15.0 + drift + 0.5

        loss = record["layers"]["L4.frame_loss"]
        assert loss["params"]["rate"] == 0.1
        assert "achieved_loss_rate" in loss["achieved"]


class TestConditionsAreComparable:
    async def test_the_same_seed_reproduces_the_whole_call(self) -> None:
        first, _, _ = await _run_call(seed=7)
        second, _, _ = await _run_call(seed=7)
        assert [f.pcm for u in first.utterances for f in u.frames] == [
            f.pcm for u in second.utterances for f in u.frames
        ]

    async def test_clean_and_degraded_differ_on_identical_content(self) -> None:
        """The comparison the benchmark exists to make: same words, worse line."""
        clean, _, clean_turns = await _run_call(seed=7, clean=True)
        noisy, _, noisy_turns = await _run_call(seed=7)

        assert [u.said for u in clean.utterances] == [u.said for u in noisy.utterances]
        assert [f.pcm for u in clean.utterances for f in u.frames] != [
            f.pcm for u in noisy.utterances for f in u.frames
        ]
        assert len(clean_turns) == len(noisy_turns)

    async def test_the_clean_profile_records_no_layers(self) -> None:
        clean, _, _ = await _run_call(seed=7, clean=True)
        record = clean.pipeline.as_record().as_dict()
        assert record["layers"] == {}
        assert all(not u.is_degraded for u in clean.utterances)


class TestTheCallProducesPublishableNumbers:
    """A real call, scored end to end.

    Everything above proves the chain runs. This proves the chain *measures* —
    which is the only reason the rest of it exists.
    """

    async def _scored(self) -> tuple[VoiceTrajectory, list]:
        simulator, transport, turns = await _run_call(seed=2026)
        recorder = Recorder(trial_id="e2e", task="reschedule", agent="cascade")
        recorder.add_party(
            PartyRecord(
                id=str(AGENT),
                role=PartyRole.SYSTEM_UNDER_TEST,
                capabilities=transport.capabilities,
            )
        )
        recorder.add_party(PartyRecord(id=str(CALLER), role=PartyRole.SIMULATED_USER))
        recorder.session_open(0.0)
        for turn, utterance, transcript in zip(
            turns, simulator.utterances, transport.transcripts, strict=True
        ):
            recorder.record_turn(turn, utterance=utterance, heard=transcript.text)
        recorder.session_close(10.0)

        trajectory = recorder.finish(
            config_matrix=ConfigMatrix(
                duplex=Duplex.HALF,
                pipeline=Pipeline.CASCADE,
                floor_policy="half",
                profile="telephony",
                seed_root=2026,
            ),
            pipeline=simulator.pipeline,
        )
        results = REGISTRY.evaluate_all(MetricContext(trajectory=trajectory))
        return trajectory, results

    async def test_the_cascade_seams_become_timing_numbers(self) -> None:
        _, results = await self._scored()
        by_name = {r.name: r for r in results}
        for name in ("timing.asr_final_s", "timing.tts_first_byte_s"):
            assert by_name[name].reported, by_name[name].reasons
            assert by_name[name].value is not None

    async def test_said_versus_heard_scores_on_a_degraded_line(self) -> None:
        _, results = await self._scored()
        wer = next(r for r in results if r.name == "divergence.wer_said_heard")
        assert wer.reported
        assert wer.n == 2

    async def test_metrics_whose_conditions_were_absent_abstain(self) -> None:
        """No L5 layer ran, so selectivity reports nothing rather than zero."""
        _, results = await self._scored()
        for name in (
            "selectivity.backchannel",
            "selectivity.vocal_tic",
            "selectivity.non_directed",
        ):
            result = next(r for r in results if r.name == name)
            assert result.verdict is Verdict.NOT_APPLICABLE
            assert result.value is None

    async def test_the_three_documents_are_produced(self) -> None:
        trajectory, results = await self._scored()
        outputs = build_outputs(
            results,
            trajectory.validity,
            trial_id=trajectory.trial_id,
            comparable_on=trajectory.comparable_on,
        )
        assert outputs.measurement_valid
        assert outputs.reward is not None
        assert all(isinstance(v, float) for v in outputs.reward.values())
        assert "selectivity.backchannel" not in outputs.reward
        assert (
            outputs.measurement["metrics"]["selectivity.backchannel"]["verdict"]
            == "not_applicable"
        )

    async def test_harbor_can_aggregate_the_reward_unchanged(self) -> None:
        from harbor.metrics.base import aggregate_reward_dicts

        trajectory, results = await self._scored()
        reward = build_outputs(results, trajectory.validity).reward
        assert reward is not None
        aggregated = aggregate_reward_dicts(
            [reward, reward], metric_name="mean", aggregate=lambda v: sum(v) / len(v)
        )
        assert set(aggregated) == set(reward)
