"""Unit tests for the audio atom."""

from __future__ import annotations

import pytest

from harbor_voice.audio import CANONICAL
from harbor_voice.audio import CANONICAL_FRAME_MS
from harbor_voice.audio import CANONICAL_SAMPLE_RATE_HZ
from harbor_voice.audio import FRAME_BYTES
from harbor_voice.audio import FRAME_SAMPLES
from harbor_voice.audio import AudioFormat
from harbor_voice.audio import AudioFormatError
from harbor_voice.audio import AudioFrame
from harbor_voice.audio import Clock
from harbor_voice.audio import ClockError
from harbor_voice.audio import Interval
from harbor_voice.audio import SessionClock
from harbor_voice.audio import Stage
from harbor_voice.audio import StageMark
from harbor_voice.audio.stages import CASCADE_ONLY_STAGES
from harbor_voice.audio.stages import CASCADE_STAGES
from harbor_voice.audio.stages import S2S_REQUIRED_STAGES
from harbor_voice.audio.stages import S2S_STAGES
from harbor_voice.ids import PartyId
from harbor_voice.ids import PlanId
from harbor_voice.ids import sha256_digest

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")


def _pcm(n_bytes: int = FRAME_BYTES) -> bytes:
    return b"\x00" * n_bytes


# --------------------------------------------------------------- format


class TestAudioFormat:
    def test_canonical_constants_agree(self) -> None:
        assert CANONICAL_SAMPLE_RATE_HZ == 24_000
        assert CANONICAL_FRAME_MS == 20
        assert FRAME_SAMPLES == 480
        assert FRAME_BYTES == 960
        assert CANONICAL.is_canonical

    def test_frame_bytes_matches_samples_times_width(self) -> None:
        assert CANONICAL.frame_bytes() == FRAME_SAMPLES * 2

    def test_mulaw_is_one_byte_per_sample(self) -> None:
        fmt = AudioFormat(sample_rate_hz=8_000, encoding="mulaw")
        assert fmt.bytes_per_sample == 1
        assert fmt.frame_bytes(20) == 160
        assert not fmt.is_canonical

    def test_duration_round_trips(self) -> None:
        assert CANONICAL.duration_s(FRAME_BYTES) == pytest.approx(0.020)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_nonpositive_rate(self, bad: int) -> None:
        with pytest.raises(AudioFormatError, match="sample_rate_hz"):
            AudioFormat(sample_rate_hz=bad)

    def test_rejects_unknown_encoding(self) -> None:
        with pytest.raises(AudioFormatError, match="unknown encoding"):
            AudioFormat(encoding="opus")  # type: ignore[arg-type]

    def test_odd_length_buffer_is_rejected(self) -> None:
        """The invariant: an odd PCM16 buffer misaligns every later sample."""
        with pytest.raises(AudioFormatError, match="whole number of samples"):
            CANONICAL.validate_buffer(961)

    def test_fractional_frame_duration_is_rejected(self) -> None:
        fmt = AudioFormat(sample_rate_hz=44_100)
        with pytest.raises(AudioFormatError, match="whole number of samples"):
            fmt.frame_bytes(1)

    def test_is_hashable_and_frozen(self) -> None:
        assert len({AudioFormat(), AudioFormat(), CANONICAL}) == 1
        with pytest.raises(AttributeError):
            CANONICAL.sample_rate_hz = 8_000  # type: ignore[misc]


# ---------------------------------------------------------------- clock


class TestClock:
    def test_session_clock_is_monotonic_from_zero(self) -> None:
        clock = SessionClock()
        first = clock.now()
        second = clock.now()
        assert 0 <= first <= second

    def test_started_at_is_utc_aware(self) -> None:
        assert SessionClock().started_at.tzinfo is not None

    def test_interval_names_both_domains(self) -> None:
        iv = Interval(1.0, 1.25, Clock.PRODUCED, Clock.VOICED)
        assert iv.label == "produced_to_voiced"
        assert iv.seconds == pytest.approx(0.25)
        assert not iv.is_same_domain

    def test_named_derives_metric_name_from_the_clocks(self) -> None:
        """A metric cannot claim one clock pair and measure another."""
        assert Interval(0, 1, Clock.TX, Clock.RX).named("transit") == "transit_tx_to_rx"

    def test_same_domain_is_detected(self) -> None:
        assert Interval(0, 1, Clock.RX, Clock.RX).is_same_domain

    def test_backwards_interval_is_rejected(self) -> None:
        with pytest.raises(ClockError, match="ends before it starts"):
            Interval(2.0, 1.0, Clock.TX, Clock.RX)


# ---------------------------------------------------------------- frame


class TestAudioFrame:
    def test_minimal_frame(self) -> None:
        f = AudioFrame(pcm=_pcm(), seq=0, party=CALLER, t_produced=0.0)
        assert f.duration_s == pytest.approx(0.020)
        assert f.is_clean
        assert f.at(Clock.PRODUCED) == 0.0
        assert f.at(Clock.TX) is None

    def test_odd_length_pcm_is_rejected(self) -> None:
        with pytest.raises(AudioFormatError):
            AudioFrame(pcm=b"\x00" * 961, seq=0, party=CALLER, t_produced=0.0)

    def test_negative_seq_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="seq must be non-negative"):
            AudioFrame(pcm=_pcm(), seq=-1, party=CALLER, t_produced=0.0)

    def test_short_final_frame_is_allowed(self) -> None:
        """An utterance need not end on a frame boundary."""
        f = AudioFrame(pcm=_pcm(320), seq=9, party=CALLER, t_produced=0.0)
        assert f.duration_s == pytest.approx(320 / 2 / 24_000)

    def test_stamp_returns_a_copy(self) -> None:
        f = AudioFrame(pcm=_pcm(), seq=0, party=CALLER, t_produced=1.0)
        g = f.stamp(Clock.TX, 1.1)
        assert f.t_tx is None, "original must be unchanged"
        assert g.t_tx == 1.1
        assert g.seq == f.seq

    def test_double_stamp_is_refused_by_default(self) -> None:
        f = AudioFrame(pcm=_pcm(), seq=0, party=CALLER, t_produced=1.0)
        g = f.stamp(Clock.TX, 1.1)
        with pytest.raises(ClockError, match="already has t_tx"):
            g.stamp(Clock.TX, 1.2)
        assert g.stamp(Clock.TX, 1.2, overwrite=True).t_tx == 1.2

    def test_produced_cannot_be_restamped(self) -> None:
        f = AudioFrame(pcm=_pcm(), seq=0, party=CALLER, t_produced=1.0)
        with pytest.raises(ClockError, match="set at construction"):
            f.stamp(Clock.PRODUCED, 2.0)

    def test_non_monotonic_clocks_are_rejected(self) -> None:
        with pytest.raises(ClockError, match="precedes"):
            AudioFrame(pcm=_pcm(), seq=0, party=CALLER, t_produced=1.0, t_tx=0.5)

    def test_non_monotonic_stamp_is_rejected(self) -> None:
        f = AudioFrame(pcm=_pcm(), seq=0, party=CALLER, t_produced=1.0, t_tx=1.2)
        with pytest.raises(ClockError, match="precedes"):
            f.stamp(Clock.RX, 1.1)

    def test_gaps_in_the_clock_chain_are_allowed(self) -> None:
        """Not every frame passes every domain; only the ones present order."""
        f = AudioFrame(pcm=_pcm(), seq=0, party=CALLER, t_produced=1.0, t_voiced=1.3)
        assert f.at(Clock.RX) is None
        assert f.at(Clock.VOICED) == 1.3

    def test_clean_ref_marks_a_perturbed_twin(self) -> None:
        """The clean signal stays addressable after degradation."""
        clean = _pcm()
        digest = sha256_digest(clean)
        f = AudioFrame(pcm=_pcm(), seq=0, party=CALLER, t_produced=0.0)
        assert f.is_clean
        g = f.with_clean_ref(digest, PlanId("plan-1"))
        assert not g.is_clean
        assert g.clean_ref == digest
        assert g.plan_ref == PlanId("plan-1")

    def test_digest_is_stable_and_prefixed(self) -> None:
        d = sha256_digest(b"abc")
        assert d.startswith("sha256:")
        assert d == sha256_digest(b"abc")
        assert d != sha256_digest(b"abd")

    def test_repr_lists_stamped_clocks(self) -> None:
        f = AudioFrame(pcm=_pcm(), seq=3, party=CALLER, t_produced=0.0, t_tx=0.1)
        r = repr(f)
        assert "seq=3" in r
        assert "produced,tx" in r
        assert "20ms" in r


# --------------------------------------------------------------- stages


class TestStages:
    def test_cascade_exposes_every_stage(self) -> None:
        assert CASCADE_STAGES == frozenset(Stage)

    def test_s2s_requires_only_response_boundaries(self) -> None:
        assert S2S_REQUIRED_STAGES == {Stage.RESPONSE_START, Stage.RESPONSE_DONE}

    def test_the_asymmetry_is_the_composed_seams(self) -> None:
        """Corrected: the gap is what harbor-voice *composed*.

        Nova Sonic emits the transcript of its own input, so ``asr_final``
        left this set. What remains is the five seams that exist only because
        we built the pipeline — and they are exactly the ones the reference-cascade
        ablation needs.
        """
        assert len(CASCADE_ONLY_STAGES) == 5
        assert Stage.ASR_FINAL not in CASCADE_ONLY_STAGES
        assert Stage.LLM_FIRST_TOKEN in CASCADE_ONLY_STAGES
        assert Stage.TTS_FIRST_BYTE in CASCADE_ONLY_STAGES
        assert Stage.RESPONSE_START not in CASCADE_ONLY_STAGES
        assert S2S_STAGES < CASCADE_STAGES

    def test_stage_mark_defaults_to_rx_domain(self) -> None:
        m = StageMark(stage=Stage.ASR_FINAL, t=1.5, party=CALLER)
        assert m.clock is Clock.RX
        assert m.meta == {}

    def test_stage_mark_meta_is_read_only(self) -> None:
        m = StageMark(stage=Stage.ASR_FINAL, t=1.5, party=CALLER)
        with pytest.raises(TypeError):
            m.meta["x"] = 1  # type: ignore[index]

    def test_negative_stage_time_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative time"):
            StageMark(stage=Stage.ASR_FINAL, t=-0.1, party=CALLER)

    def test_stage_values_are_stable_strings(self) -> None:
        """Stage names land in the trajectory; renaming one is a format break."""
        assert Stage.LLM_FIRST_TOKEN.value == "llm_first_token"
        assert str(Stage.ASR_PARTIAL) == "asr_partial"
