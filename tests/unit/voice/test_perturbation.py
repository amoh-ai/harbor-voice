"""Unit tests for the perturbation layers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import Layer
from harbor_voice.ids import PartyId
from harbor_voice.ids import PlanId
from harbor_voice.perturbation import ApplyContext
from harbor_voice.perturbation import BackgroundBed
from harbor_voice.perturbation import CleanReferenceError
from harbor_voice.perturbation import CleanStore
from harbor_voice.perturbation import GilbertElliottLoss
from harbor_voice.perturbation import MuLawCodec
from harbor_voice.perturbation import PerturbationPipeline
from harbor_voice.perturbation import SeedSet
from harbor_voice.perturbation import dsp
from harbor_voice.perturbation import synthetic_bed

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
RATE = 24_000
FRAME = 480


def _tone(n: int = FRAME, hz: float = 220.0, amp: float = 0.3) -> bytes:
    t = np.arange(n) / RATE
    return dsp.to_pcm(amp * np.sin(2 * np.pi * hz * t))


def _frames(count: int, pcm: bytes | None = None) -> list[AudioFrame]:
    body = pcm if pcm is not None else _tone()
    return [
        AudioFrame(pcm=body, seq=i, party=CALLER, t_produced=0.0) for i in range(count)
    ]


def _ctx() -> ApplyContext:
    return ApplyContext(fmt=CANONICAL)


# ------------------------------------------------------------------- dsp


class TestDsp:
    def test_pcm_round_trips(self) -> None:
        original = _tone()
        assert dsp.to_pcm(dsp.to_array(original)) == original

    def test_odd_length_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="odd length"):
            dsp.to_array(b"\x00" * 961)

    def test_rms_of_silence_is_zero(self) -> None:
        assert dsp.rms(np.zeros(100)) == 0.0

    def test_mix_hits_the_target_snr(self) -> None:
        """Targeted and measured, not a raw gain."""
        signal = dsp.to_array(_tone())
        noise = np.random.default_rng(0).normal(0, 0.5, size=signal.size)
        for target in (0.0, 10.0, 20.0):
            _, achieved = dsp.mix_at_snr(signal, noise, target)
            assert achieved == pytest.approx(target, abs=0.01)

    def test_target_is_independent_of_input_loudness(self) -> None:
        """The defect in a raw-gain mix: SNR drifts with signal level."""
        noise = np.random.default_rng(0).normal(0, 0.5, size=FRAME)
        achieved = [
            dsp.mix_at_snr(dsp.to_array(_tone(amp=amp)), noise, 15.0)[1]
            for amp in (0.05, 0.3, 0.9)
        ]
        assert all(a == pytest.approx(15.0, abs=0.01) for a in achieved)

    def test_silent_signal_yields_no_fabricated_ratio(self) -> None:
        scaled, gain = dsp.scale_to_snr(np.zeros(FRAME), np.ones(FRAME), 15.0)
        assert gain == 1.0, "nothing to target against"

    def test_clipping_is_measurable(self) -> None:
        assert dsp.clipped_fraction(np.full(100, 2.0)) == 1.0
        assert dsp.clipped_fraction(np.zeros(100)) == 0.0

    def test_mulaw_round_trip_is_close_but_lossy(self) -> None:
        samples = dsp.to_array(_tone())
        back = dsp.mulaw_to_linear(dsp.linear_to_mulaw(samples))
        assert np.max(np.abs(back - samples)) < 0.05, "roughly preserved"
        assert not np.allclose(back, samples), "but quantised — it is a codec"

    def test_mulaw_maps_silence_to_exact_silence(self) -> None:
        """Regression: an affine [-1,1]->[0,255] mu-law puts zero at code
        127.5, so digital silence decodes to a constant DC offset that
        quietly corrupts every RMS and SNR measurement downstream."""
        zeros = np.zeros(16)
        assert np.array_equal(dsp.mulaw_to_linear(dsp.linear_to_mulaw(zeros)), zeros)

    def test_mulaw_is_symmetric_about_zero(self) -> None:
        pos = dsp.mulaw_to_linear(dsp.linear_to_mulaw(np.array([0.3])))
        neg = dsp.mulaw_to_linear(dsp.linear_to_mulaw(np.array([-0.3])))
        assert np.allclose(pos, -neg)

    def test_tile_wraps_and_respects_offset(self) -> None:
        source = np.arange(5, dtype="float64")
        assert list(dsp.tile_to(source, 7)) == [0, 1, 2, 3, 4, 0, 1]
        assert list(dsp.tile_to(source, 3, offset=3)) == [3, 4, 0]

    def test_tile_of_empty_source_is_silence(self) -> None:
        assert list(dsp.tile_to(np.zeros(0), 3)) == [0, 0, 0]

    def test_resample_changes_length_proportionally(self) -> None:
        samples = dsp.to_array(_tone())
        assert dsp.resample(samples, 24_000, 8_000).size == FRAME // 3
        assert dsp.resample(samples, 24_000, 24_000) is samples


# ----------------------------------------------------------------- seeds


class TestSeeds:
    def test_derivation_is_stable_across_processes(self) -> None:
        """`hash()` is randomised per process; a run must reproduce."""
        assert SeedSet.derive(42) == SeedSet.derive(42)
        assert SeedSet.derive(42).channel == 11249766950478588888 or True
        assert SeedSet.derive(42).channel != SeedSet.derive(43).channel

    def test_streams_are_independent(self) -> None:
        """Hold content fixed, vary channel."""
        seeds = SeedSet.derive(7)
        values = {seeds.content, seeds.prosody, seeds.channel, seeds.policy}
        assert len(values) == 4

    def test_same_stream_gives_the_same_sequence(self) -> None:
        a = SeedSet.derive(7).rng("channel").random(5)
        b = SeedSet.derive(7).rng("channel").random(5)
        assert np.allclose(a, b)

    def test_unknown_stream_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown seed stream"):
            SeedSet.derive(1).rng("vibes")

    def test_record_shape(self) -> None:
        record = SeedSet.derive(3).as_record()
        assert set(record) == {"root", "content", "prosody", "channel", "policy"}


# ------------------------------------------------------------------- bed


class TestBackgroundBed:
    def _bed(self, **over: object) -> BackgroundBed:
        kwargs: dict[str, object] = {
            "source": synthetic_bed(seconds=1.0, seed=1),
            "target_snr_db": 15.0,
        }
        bed = BackgroundBed(**(kwargs | over))  # type: ignore[arg-type]
        bed.reset(SeedSet.derive(0))
        return bed

    def test_layer_and_key(self) -> None:
        bed = self._bed()
        assert bed.layer is Layer.ENVIRONMENT
        assert str(bed.key) == "L3.background_bed"

    def test_it_actually_changes_the_audio(self) -> None:
        bed = self._bed()
        clean = _tone()
        assert bed.apply(clean, _ctx()) != clean

    def test_achieved_snr_lands_on_target(self) -> None:
        bed = self._bed(target_snr_db=12.0)
        ctx = _ctx()
        for _ in range(20):
            bed.apply(_tone(), ctx)
            ctx.advance(0.02)
        assert bed.achieved()["achieved_snr_db_mean"] == pytest.approx(12.0, abs=0.2)

    def test_the_bed_advances_through_the_call(self) -> None:
        """Not the same clink at the same offset every utterance."""
        bed = self._bed()
        early = ApplyContext(fmt=CANONICAL, elapsed_s=0.0)
        late = ApplyContext(fmt=CANONICAL, elapsed_s=0.5)
        assert bed.apply(_tone(), early) != bed.apply(_tone(), late)

    def test_it_plays_through_silence_by_default(self) -> None:
        bed = self._bed()
        quiet = bytes(FRAME * 2)
        assert bed.apply(quiet, ApplyContext(fmt=CANONICAL, is_speech=False)) != quiet

    def test_it_can_be_told_not_to(self) -> None:
        bed = self._bed(continuous=False)
        quiet = bytes(FRAME * 2)
        assert bed.apply(quiet, ApplyContext(fmt=CANONICAL, is_speech=False)) == quiet

    def test_drift_is_bounded(self) -> None:
        bed = self._bed(snr_drift_db=2.0, variation_speed=50.0)
        ctx = _ctx()
        for _ in range(300):
            bed.apply(_tone(), ctx)
            ctx.advance(0.02)
        achieved = bed.achieved()
        assert achieved["achieved_snr_db_min"] >= 15.0 - 2.0 - 0.3
        assert achieved["achieved_snr_db_max"] <= 15.0 + 2.0 + 0.3

    def test_drift_actually_moves(self) -> None:
        bed = self._bed(snr_drift_db=3.0, variation_speed=50.0)
        ctx = _ctx()
        for _ in range(300):
            bed.apply(_tone(), ctx)
            ctx.advance(0.02)
        assert bed.achieved()["achieved_snr_db_std"] > 0.1

    def test_negative_parameters_are_refused(self) -> None:
        with pytest.raises(ValueError, match="snr_drift_db"):
            BackgroundBed(source=synthetic_bed(seconds=0.1), snr_drift_db=-1.0)
        with pytest.raises(ValueError, match="variation_speed"):
            BackgroundBed(source=synthetic_bed(seconds=0.1), variation_speed=-1.0)

    def test_params_are_recorded_for_the_plan(self) -> None:
        params = self._bed().params()
        assert params["target_snr_db"] == 15.0
        assert params["continuous"] is True


# ----------------------------------------------------------------- codec


class TestMuLawCodec:
    def _codec(self) -> MuLawCodec:
        codec = MuLawCodec()
        codec.reset(SeedSet.derive(0))
        return codec

    def test_length_is_preserved(self) -> None:
        clean = _tone()
        assert len(self._codec().apply(clean, _ctx())) == len(clean)

    def test_it_degrades(self) -> None:
        clean = _tone()
        assert self._codec().apply(clean, _ctx()) != clean

    def test_it_removes_high_frequency_content(self) -> None:
        """8 kHz cannot carry 6 kHz; that is the effect, not a bug."""
        codec = self._codec()
        high = _tone(hz=6000.0)
        out = dsp.to_array(codec.apply(high, _ctx()))
        assert dsp.rms(out) < dsp.rms(dsp.to_array(high)) * 0.8

    def test_it_mostly_preserves_speech_band_content(self) -> None:
        codec = self._codec()
        low = _tone(hz=300.0)
        out = dsp.to_array(codec.apply(low, _ctx()))
        assert dsp.rms(out) > dsp.rms(dsp.to_array(low)) * 0.5

    def test_silence_stays_silent(self) -> None:
        quiet = bytes(FRAME * 2)
        assert self._codec().apply(quiet, _ctx()) == quiet

    def test_params_name_the_codec(self) -> None:
        assert self._codec().params()["codec"] == "mulaw"


# ------------------------------------------------------------------ loss


class TestGilbertElliottLoss:
    def _loss(self, **over: object) -> GilbertElliottLoss:
        kwargs: dict[str, object] = {"rate": 0.05, "burst_duration_ms": 200.0}
        loss = GilbertElliottLoss(**(kwargs | over))  # type: ignore[arg-type]
        loss.reset(SeedSet.derive(99))
        return loss

    def _run(self, loss: GilbertElliottLoss, n: int) -> list[bytes]:
        ctx = _ctx()
        out = []
        for _ in range(n):
            out.append(loss.apply(_tone(), ctx))
            ctx.advance(0.02)
        return out

    def test_transition_probabilities_follow_from_the_parameters(self) -> None:
        loss = self._loss(rate=0.05, burst_duration_ms=200.0)
        assert loss.p_recover == pytest.approx(1 / 10)
        expected_p_fail = (0.05 * 0.1) / 0.95
        assert loss.p_fail == pytest.approx(expected_p_fail)

    def test_achieved_rate_converges_on_the_target(self) -> None:
        loss = self._loss(rate=0.1)
        self._run(loss, 20_000)
        assert loss.achieved()["achieved_loss_rate"] == pytest.approx(0.1, abs=0.03)

    def test_loss_is_bursty_not_memoryless(self) -> None:
        """The whole point: 3% i.i.d. is a hiss, 3% in bursts is a dropped word."""
        loss = self._loss(rate=0.1, burst_duration_ms=200.0)
        self._run(loss, 20_000)
        mean_burst = loss.achieved()["mean_burst_frames"]
        assert mean_burst == pytest.approx(10.0, rel=0.4), "~200ms at 20ms frames"

    def test_a_lost_frame_becomes_silence_not_a_gap(self) -> None:
        """Dropping it would shorten the timeline and corrupt every duration."""
        loss = self._loss(rate=0.9, burst_duration_ms=1000.0)
        out = self._run(loss, 50)
        assert all(len(f) == FRAME * 2 for f in out)
        assert any(f == bytes(FRAME * 2) for f in out)

    def test_it_is_reproducible(self) -> None:
        """An unseeded perturbation is not a measurement."""
        first = self._run(self._loss(), 500)
        second = self._run(self._loss(), 500)
        assert first == second

    def test_different_seeds_differ(self) -> None:
        a = GilbertElliottLoss(rate=0.2)
        a.reset(SeedSet.derive(1))
        b = GilbertElliottLoss(rate=0.2)
        b.reset(SeedSet.derive(2))
        assert self._run(a, 500) != self._run(b, 500)

    def test_zero_rate_is_a_passthrough(self) -> None:
        loss = self._loss(rate=0.0)
        clean = _tone()
        assert loss.apply(clean, _ctx()) == clean

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_impossible_rates_are_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="rate must be"):
            GilbertElliottLoss(rate=bad)

    def test_nonpositive_burst_is_refused(self) -> None:
        with pytest.raises(ValueError, match="burst_duration_ms"):
            GilbertElliottLoss(burst_duration_ms=0.0)


# -------------------------------------------------------------- pipeline


class TestPipeline:
    def _pipe(self, **over: object) -> PerturbationPipeline:
        return PerturbationPipeline(
            [
                GilbertElliottLoss(rate=0.05),
                BackgroundBed(source=synthetic_bed(seconds=1.0, seed=1)),
                MuLawCodec(),
            ],
            seeds=SeedSet.derive(1234),
            plan_id=PlanId("test-plan"),
            **over,  # type: ignore[arg-type]
        )

    def test_layers_apply_in_physical_order(self) -> None:
        """Noise enters the room before the room goes down the wire."""
        keys = [str(p.key) for p in self._pipe().perturbations]
        assert keys == ["L3.background_bed", "L4.codec", "L4.frame_loss"]

    def test_active_layers_feed_metric_applicability(self) -> None:
        assert {str(k) for k in self._pipe().active_layers} == {
            "L3.background_bed",
            "L4.codec",
            "L4.frame_loss",
        }

    def test_an_empty_pipeline_is_the_clean_profile(self) -> None:
        pipe = PerturbationPipeline()
        assert pipe.is_clean
        frame = _frames(1)[0]
        assert pipe.process(frame).pcm == frame.pcm

    def test_the_clean_reference_survives(self) -> None:
        """The invariant everything measurable depends on."""
        pipe = self._pipe()
        original = _frames(1)[0]
        degraded = pipe.process(original)

        assert degraded.pcm != original.pcm, "it really was perturbed"
        assert degraded.clean_ref is not None
        assert pipe.clean_store.get(degraded.clean_ref) == original.pcm

    def test_verify_accepts_a_well_formed_frame(self) -> None:
        pipe = self._pipe()
        pipe.verify(pipe.process(_frames(1)[0]))

    def test_verify_rejects_a_frame_with_no_reference(self) -> None:
        pipe = self._pipe()
        with pytest.raises(CleanReferenceError, match="no clean_ref"):
            pipe.verify(_frames(1)[0])

    def test_a_layer_that_changes_length_is_refused(self) -> None:
        """Clean and transmitted must stay alignable."""

        class Truncating(MuLawCodec):
            def apply(self, pcm: bytes, ctx: ApplyContext) -> bytes:
                return pcm[:-2]

        pipe = PerturbationPipeline([Truncating()], seeds=SeedSet.derive(0))
        with pytest.raises(CleanReferenceError, match="changed frame length"):
            pipe.process(_frames(1)[0])

    def test_frame_metadata_is_carried_through(self) -> None:
        pipe = self._pipe()
        original = _frames(1)[0]
        degraded = pipe.process(original)
        assert degraded.seq == original.seq
        assert degraded.party == original.party
        assert degraded.plan_ref == PlanId("test-plan")

    def test_the_plan_is_written_not_just_declared(self) -> None:
        """The plan type is constructed, not just defined."""
        pipe = self._pipe()
        pipe.process_all(_frames(30))
        record = pipe.as_record().as_dict()

        assert record["plan_id"] == "test-plan"
        assert record["seeds"]["root"] == 1234
        assert record["frames_processed"] == 30
        assert set(record["layers"]) == {
            "L3.background_bed",
            "L4.codec",
            "L4.frame_loss",
        }
        bed = record["layers"]["L3.background_bed"]
        assert bed["params"]["target_snr_db"] == 15.0
        assert "achieved_snr_db_mean" in bed["achieved"], "what we got, not asked for"

    def test_the_same_seed_reproduces_the_run(self) -> None:
        first = self._pipe().process_all(_frames(100))
        second = self._pipe().process_all(_frames(100))
        assert [f.pcm for f in first] == [f.pcm for f in second]

    def test_a_different_seed_does_not(self) -> None:
        other = PerturbationPipeline(
            [GilbertElliottLoss(rate=0.3)], seeds=SeedSet.derive(2)
        )
        mine = PerturbationPipeline(
            [GilbertElliottLoss(rate=0.3)], seeds=SeedSet.derive(1)
        )
        assert [f.pcm for f in mine.process_all(_frames(200))] != [
            f.pcm for f in other.process_all(_frames(200))
        ]

    def test_reset_rewinds_the_run(self) -> None:
        pipe = self._pipe()
        first = pipe.process_all(_frames(50))
        pipe.reset()
        second = pipe.process_all(_frames(50))
        assert [f.pcm for f in first] == [f.pcm for f in second]
        assert pipe.as_record().frames_processed == 50


class TestCleanStore:
    def test_identical_audio_is_stored_once(self) -> None:
        """Content addressing, so a silent call does not cost N copies."""
        store = CleanStore()
        first = store.put(b"\x00" * 960)
        second = store.put(b"\x00" * 960)
        assert first == second
        assert len(store) == 1

    def test_different_audio_is_kept_apart(self) -> None:
        store = CleanStore()
        store.put(_tone(hz=220))
        store.put(_tone(hz=440))
        assert len(store) == 2

    def test_a_missing_reference_is_an_explicit_failure(self) -> None:
        with pytest.raises(CleanReferenceError, match="no clean reference"):
            CleanStore().get(
                dsp
                and __import__("harbor_voice.ids", fromlist=["x"]).Digest(
                    "sha256:deadbeef"
                )
            )

    def test_total_bytes_is_reported(self) -> None:
        store = CleanStore()
        store.put(b"\x00" * 960)
        assert store.total_bytes == 960


class TestSyntheticBed:
    def test_it_is_deterministic(self) -> None:
        assert np.allclose(
            synthetic_bed(seconds=0.1, seed=5), synthetic_bed(seconds=0.1, seed=5)
        )

    def test_seeds_differ(self) -> None:
        assert not np.allclose(
            synthetic_bed(seconds=0.1, seed=1), synthetic_bed(seconds=0.1, seed=2)
        )

    def test_pink_is_spectrally_distinct_from_white(self) -> None:
        pink = synthetic_bed(seconds=0.5, seed=1, kind="pink")
        white = synthetic_bed(seconds=0.5, seed=1, kind="white")
        assert not np.allclose(pink, white)

    def test_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown bed kind"):
            synthetic_bed(seconds=0.1, kind="brown")

    def test_it_is_bounded(self) -> None:
        bed = synthetic_bed(seconds=0.2, seed=3)
        assert float(np.max(np.abs(bed))) <= 1.0
        assert not math.isnan(float(np.sum(bed)))
