"""Unit tests for ATIF-voice."""

from __future__ import annotations

import numpy as np
import pytest

from harbor.models.trajectories.trajectory import Trajectory
from harbor_voice.adapter import Turn
from harbor_voice.audio.clock import Clock
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.audio.stages import StageMark
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.floor import BargeInFloor
from harbor_voice.floor import HalfDuplexFloor
from harbor_voice.ids import PartyId
from harbor_voice.ids import PlanId
from harbor_voice.perturbation import GilbertElliottLoss
from harbor_voice.perturbation import PerturbationPipeline
from harbor_voice.perturbation import SeedSet
from harbor_voice.perturbation import dsp
from harbor_voice.simulator import SimulatorFidelity
from harbor_voice.simulator import SourceKind
from harbor_voice.trajectory import VOICE_EXTRA_KEY
from harbor_voice.trajectory import VOICE_SCHEMA_VERSION
from harbor_voice.trajectory import BargeInOutcome
from harbor_voice.trajectory import ConfigMatrix
from harbor_voice.trajectory import EventType
from harbor_voice.trajectory import PartyRecord
from harbor_voice.trajectory import PartyRole
from harbor_voice.trajectory import PlacedAudio
from harbor_voice.trajectory import Recorder
from harbor_voice.trajectory import RecorderConfig
from harbor_voice.trajectory import TokenSpan
from harbor_voice.trajectory import VoiceTrajectory
from harbor_voice.trajectory import downsample_to_fit
from harbor_voice.trajectory import placed_from_frames
from harbor_voice.trajectory import render_timeline
from harbor_voice.trajectory import to_wav
from harbor_voice.transport.drain import DrainStop

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
AGENT = PartyId("agent")
RATE = 24_000


def _tone(secs: float, hz: float = 220.0) -> bytes:
    t = np.arange(int(RATE * secs)) / RATE
    return dsp.to_pcm(0.3 * np.sin(2 * np.pi * hz * t))


def _frame(seq: int, party: PartyId, rx: float | None = None) -> AudioFrame:
    f = AudioFrame(pcm=bytes(960), seq=seq, party=party, t_produced=0.0)
    return f.stamp(Clock.RX, rx) if rx is not None else f


# ------------------------------------------------------------- the format


class TestVoiceTrajectory:
    def test_it_emits_a_valid_atif_document(self) -> None:
        """Extends ATIF rather than replacing it."""
        voice = VoiceTrajectory(trial_id="t1", agent="cascade", model="echo")
        atif = voice.to_atif()
        assert isinstance(atif, Trajectory)
        assert atif.session_id == "t1"
        assert VOICE_EXTRA_KEY in (atif.extra or {})

    def test_a_plain_atif_reader_accepts_it(self) -> None:
        """Every ATIF model is `extra="forbid"`, so this is the real test."""
        payload = VoiceTrajectory(trial_id="t1").to_atif().model_dump(mode="json")
        reparsed = Trajectory.model_validate(payload)
        assert reparsed.session_id == "t1"

    def test_it_round_trips(self) -> None:
        original = VoiceTrajectory(
            trial_id="t1",
            task="intake-call",
            task_sha="abc123",
            config_matrix=ConfigMatrix(
                duplex=Duplex.FULL,
                pipeline=Pipeline.CASCADE,
                floor_policy="BargeInFloor",
                profile="telephony",
                seed_root=7,
            ),
        )
        recovered = VoiceTrajectory.from_atif(original.to_atif())
        assert recovered.task_sha == "abc123"
        assert recovered.config_matrix is not None
        assert recovered.config_matrix.profile == "telephony"

    def test_trajectory_json_reads_in_any_encoding(self, tmp_path, monkeypatch) -> None:
        """Harbor reads it with the platform's encoding: cp1252 on Windows."""
        from pathlib import Path

        from harbor.utils import trajectory_utils

        read_text = Path.read_text
        monkeypatch.setattr(  # Windows' default, wherever this runs
            Path,
            "read_text",
            lambda self, encoding=None, **kw: read_text(
                self, encoding=encoding or "cp1252", **kw
            ),
        )
        voice = VoiceTrajectory(trial_id="t1", task="You said “P”, café")
        path = tmp_path / "trajectory.json"

        # Unescaped, the UTF-8 holds 0x9d (in ”), which cp1252 can't decode.
        path.write_text(voice.to_atif().model_dump_json(), encoding="utf-8")
        with pytest.raises(UnicodeDecodeError):
            trajectory_utils.load_model_usage_from_trajectory(path)

        text = voice.to_atif_json()
        assert text.isascii() and "\\u201c" in text
        assert Trajectory.model_validate_json(text) == voice.to_atif()
        path.write_text(text, encoding="utf-8")
        trajectory_utils.load_model_usage_from_trajectory(path)  # reads now
        assert (
            VoiceTrajectory.from_atif(
                Trajectory.model_validate_json(path.read_text())
            ).task
            == voice.task
        )

    def test_a_non_voice_atif_document_is_refused(self) -> None:
        plain = VoiceTrajectory().to_atif()
        plain.extra = {}
        with pytest.raises(ValueError, match="no 'voice' payload"):
            VoiceTrajectory.from_atif(plain)

    def test_schema_version_is_declared(self) -> None:
        assert VoiceTrajectory().schema_version == VOICE_SCHEMA_VERSION
        assert "voice" in VOICE_SCHEMA_VERSION

    def test_comparable_on_pins_the_dimensions(self) -> None:
        """A noisy run must not be diffable against a clean one."""
        matrix = ConfigMatrix(
            duplex=Duplex.HALF,
            pipeline=Pipeline.S2S,
            floor_policy="HalfDuplexFloor",
            profile="telephony",
            seed_root=3,
        )
        pinned = VoiceTrajectory(task_sha="sha", config_matrix=matrix).comparable_on
        assert pinned["profile"] == "telephony"
        assert pinned["duplex"] == "half"
        assert pinned["seed_root"] == 3

    def test_is_degraded_follows_the_plan(self) -> None:
        assert not VoiceTrajectory().is_degraded
        assert VoiceTrajectory(
            perturbation_plan={"layers": {"L4.codec": {}}}
        ).is_degraded


class TestTokenSpan:
    def test_it_carries_the_rl_fields(self) -> None:
        """The only RL affordance expensive to retrofit."""
        span = TokenSpan(
            start_idx=0,
            end_idx=3,
            completion_token_ids=(10, 11, 12),
            logprobs=(-0.1, -0.2, -0.3),
        )
        assert span.length == 3
        assert len(span.completion_token_ids) == 3

    def test_an_empty_span_has_no_length(self) -> None:
        assert TokenSpan(start_idx=5, end_idx=5).length == 0


# ----------------------------------------------------------- the renderer


class TestRenderTimeline:
    def test_overlapping_sources_are_mixed_not_queued(self) -> None:
        """Concatenation makes an interruption sound orderly."""
        a = PlacedAudio(party=CALLER, start_s=0.0, pcm=_tone(1.0, 220))
        b = PlacedAudio(party=AGENT, start_s=0.6, pcm=_tone(1.0, 440))
        result = render_timeline([a, b])

        assert result.duration_s == pytest.approx(1.6, abs=0.01)
        concatenated = (len(a.pcm) + len(b.pcm)) / 2 / RATE
        assert concatenated == pytest.approx(2.0, abs=0.01)
        assert result.duration_s < concatenated, "the lie concatenation tells"

    def test_overlap_is_measured(self) -> None:
        a = PlacedAudio(party=CALLER, start_s=0.0, pcm=_tone(1.0))
        b = PlacedAudio(party=AGENT, start_s=0.6, pcm=_tone(1.0))
        assert render_timeline([a, b]).overlap_s == pytest.approx(0.4, abs=0.01)

    def test_no_overlap_when_turns_do_not_touch(self) -> None:
        a = PlacedAudio(party=CALLER, start_s=0.0, pcm=_tone(0.5))
        b = PlacedAudio(party=AGENT, start_s=1.0, pcm=_tone(0.5))
        result = render_timeline([a, b])
        assert result.overlap_s == 0.0
        assert not result.has_overlap

    def test_inter_turn_silence_is_preserved(self) -> None:
        """Its length is a measurement, not padding to be removed."""
        a = PlacedAudio(party=CALLER, start_s=0.0, pcm=_tone(0.5))
        b = PlacedAudio(party=AGENT, start_s=2.0, pcm=_tone(0.5))
        assert render_timeline([a, b]).duration_s == pytest.approx(2.5, abs=0.01)

    def test_clipping_is_reported(self) -> None:
        loud = dsp.to_pcm(np.full(RATE, 0.9))
        a = PlacedAudio(party=CALLER, start_s=0.0, pcm=loud)
        b = PlacedAudio(party=AGENT, start_s=0.0, pcm=loud)
        assert render_timeline([a, b]).clipped_fraction > 0.9

    def test_empty_input_renders_nothing(self) -> None:
        result = render_timeline([])
        assert result.pcm == b"" and result.duration_s == 0.0

    def test_a_format_mismatch_is_refused_not_silently_resampled(self) -> None:
        """A resample is a declared event, not an accident."""
        a = PlacedAudio(party=CALLER, start_s=0.0, pcm=_tone(0.1))
        b = PlacedAudio(
            party=AGENT,
            start_s=0.0,
            pcm=bytes(160),
            fmt=AudioFormat(sample_rate_hz=8_000, encoding="mulaw"),
        )
        with pytest.raises(ValueError, match="without a declared resample"):
            render_timeline([a, b])


class TestPlacedFromFrames:
    def test_contiguous_frames_of_one_party_become_one_run(self) -> None:
        frames = [_frame(i, CALLER, rx=i * 0.02) for i in range(3)]
        placed = placed_from_frames(frames)
        assert len(placed) == 1
        assert placed[0].start_s == 0.0
        assert len(placed[0].pcm) == 3 * 960

    def test_a_party_change_splits_the_run(self) -> None:
        frames = [_frame(0, CALLER, 0.0), _frame(1, AGENT, 0.02)]
        placed = placed_from_frames(frames)
        assert len(placed) == 2
        assert [p.party for p in placed] == [CALLER, AGENT]

    def test_position_comes_from_the_frame_clock(self) -> None:
        placed = placed_from_frames([_frame(0, CALLER, rx=5.0)])
        assert placed[0].start_s == 5.0


class TestWavAndDownsample:
    def test_wav_is_playable_sized(self) -> None:
        pcm = _tone(1.0)
        wav = to_wav(pcm)
        assert wav[:4] == b"RIFF"
        assert len(wav) > len(pcm)

    def test_wav_refuses_a_non_pcm_format(self) -> None:
        with pytest.raises(ValueError, match="needs pcm16"):
            to_wav(b"", AudioFormat(sample_rate_hz=8000, encoding="mulaw"))

    def test_downsampling_preserves_duration(self) -> None:
        """So every duration-based assertion still holds against the artifact."""
        pcm = _tone(2.0)
        before = CANONICAL.duration_s(len(pcm))
        shrunk, fmt = downsample_to_fit(pcm, max_bytes=len(pcm) // 3)
        assert len(shrunk) < len(pcm)
        assert fmt.duration_s(len(shrunk)) == pytest.approx(before, abs=0.01)

    def test_it_stops_at_the_floor_rather_than_destroying_the_artifact(self) -> None:
        pcm = _tone(1.0)
        shrunk, fmt = downsample_to_fit(pcm, max_bytes=10, floor_hz=3_000)
        assert fmt.sample_rate_hz == 3_000
        assert len(shrunk) > 10, "it did not truncate to fit"


# ------------------------------------------------------------- recorder


def _turn(index: int = 0, *, received: int = 3, stages: bool = True) -> Turn:
    sent = tuple(
        AudioFrame(pcm=bytes(960), seq=i, party=CALLER, t_produced=0.0).stamp(
            Clock.TX, 1.0 + i * 0.02
        )
        for i in range(2)
    )
    got = tuple(_frame(i, AGENT, rx=2.0 + i * 0.02) for i in range(received))
    marks = (
        (
            StageMark(stage=Stage.ASR_FINAL, t=1.5, party=AGENT),
            StageMark(stage=Stage.RESPONSE_START, t=2.0, party=AGENT),
        )
        if stages
        else ()
    )
    return Turn(
        index=index,
        party=AGENT,
        speaker=CALLER,
        sent=sent,
        received=got,
        stages=marks,
        stop=DrainStop.IDLE,
    )


class TestRecorder:
    def _recorder(self, **over: object) -> Recorder:
        return Recorder(
            trial_id="trial-1",
            task="intake-call",
            task_sha="sha123",
            agent="cascade",
            config=RecorderConfig(**over),  # type: ignore[arg-type]
        )

    def test_it_builds_a_trajectory(self) -> None:
        rec = self._recorder()
        rec.add_party(PartyRecord(id="agent", role=PartyRole.SYSTEM_UNDER_TEST))
        rec.session_open(0.0)
        rec.record_turn(_turn(), heard="hello")
        rec.session_close(5.0)
        traj = rec.finish()

        assert traj.trial_id == "trial-1"
        assert len(traj.turns) == 1
        assert len(traj.parties) == 1
        assert traj.events_of(EventType.SESSION_OPEN)

    def test_the_timeline_is_ordered(self) -> None:
        rec = self._recorder()
        rec.session_close(9.0)
        rec.session_open(0.0)
        rec.record_turn(_turn())
        times = [e.t for e in rec.finish().timeline]
        assert times == sorted(times)

    def test_both_sides_of_said_versus_heard_are_recorded(self) -> None:
        """Said and heard are both recorded, on every turn."""
        from harbor_voice.simulator import Utterance

        utterance = Utterance(
            index=0, party=CALLER, said="I need to reschedule", frames=()
        )
        rec = self._recorder()
        turn = rec.record_turn(_turn(), utterance=utterance, heard="I need to resched")
        assert turn.said == "I need to reschedule"
        assert turn.heard == "I need to resched"
        assert turn.has_both_sides

    def test_a_turn_without_a_transcript_says_so(self) -> None:
        turn = self._recorder().record_turn(_turn())
        assert not turn.has_both_sides

    def test_stages_are_carried_onto_the_turn(self) -> None:
        turn = self._recorder().record_turn(_turn())
        assert {s.stage for s in turn.stages} == {"asr_final", "response_start"}

    def test_token_spans_advance_across_turns(self) -> None:
        """Credit assignment needs to know which tokens went where."""
        rec = self._recorder()
        first = rec.record_turn(_turn(0), token_ids=[1, 2, 3])
        second = rec.record_turn(_turn(1), token_ids=[4, 5])
        assert first.token_span is not None and second.token_span is not None
        assert (first.token_span.start_idx, first.token_span.end_idx) == (0, 3)
        assert (second.token_span.start_idx, second.token_span.end_idx) == (3, 5)

    def test_no_token_span_without_tokens(self) -> None:
        assert self._recorder().record_turn(_turn()).token_span is None

    def test_barge_in_outcome_is_recorded(self) -> None:
        """Whether the perturbation actually landed."""
        rec = self._recorder()
        rec.barge_in(3.0, by=CALLER, outcome=BargeInOutcome.FIRED_AFTER_SPEECH)
        events = rec.finish().events_of(EventType.BARGE_IN)
        assert len(events) == 1
        assert events[0].meta["outcome"] == "fired_after_speech"

    def test_floor_records_become_timeline_events(self) -> None:
        floor = HalfDuplexFloor()
        floor.speech_start(CALLER, 1.0)
        floor.may_transmit(AGENT, 1.1)
        floor.speech_end(CALLER, 2.0)

        rec = self._recorder()
        rec.absorb_floor(floor)
        traj = rec.finish()
        violations = traj.events_of(EventType.FLOOR_VIOLATION)
        assert len(violations) == 1
        assert violations[0].meta["holder"] == "caller"

    def test_barge_in_floor_yields_become_events(self) -> None:
        floor = BargeInFloor(yield_within_s=0.4)
        floor.speech_start(AGENT, 1.0)
        floor.speech_start(CALLER, 2.0)
        floor.speech_end(AGENT, 2.2)

        rec = self._recorder()
        rec.absorb_floor(floor)
        assert rec.finish().events_of(EventType.BARGE_IN)

    def test_tool_latency_injection_is_recorded(self) -> None:
        """L6 world perturbation, visible in the record."""
        rec = self._recorder()
        rec.tool_call(4.0, "lookup_patient", injected_latency_ms=4000)
        event = rec.finish().events_of(EventType.TOOL_CALL)[0]
        assert event.meta["injected_latency_ms"] == 4000


class TestValidity:
    def _rec(self, **over: object) -> Recorder:
        return Recorder(config=RecorderConfig(**over))  # type: ignore[arg-type]

    def test_a_clean_run_is_valid(self) -> None:
        rec = self._rec()
        rec.record_turn(_turn())
        traj = rec.finish(pipeline=PerturbationPipeline(seeds=SeedSet.derive(0)))
        assert traj.validity.measurement_valid
        assert not traj.validity.failed

    def test_a_blown_gap_budget_invalidates_the_run(self) -> None:
        """Invalid is not the same as failed."""
        rec = self._rec(max_transport_gap_s=0.5)
        rec.transport_gap(2.0, 1.2)
        traj = rec.finish()
        assert not traj.validity.measurement_valid
        assert any("transport_gap" in c.name for c in traj.validity.failed)

    def test_a_gap_within_budget_is_fine(self) -> None:
        rec = self._rec(max_transport_gap_s=2.0)
        rec.transport_gap(2.0, 1.0)
        assert rec.finish().validity.measurement_valid

    def test_a_perturbed_frame_without_a_reference_is_caught(self) -> None:
        """The failure the invariant exists to prevent."""
        rec = self._rec()
        orphan = AudioFrame(
            pcm=bytes(960), seq=0, party=AGENT, t_produced=0.0
        ).with_clean_ref(None, PlanId("p"))  # type: ignore[arg-type]
        turn = Turn(index=0, party=AGENT, speaker=CALLER, received=(orphan,))
        rec.record_turn(turn)
        pipeline = PerturbationPipeline(
            [GilbertElliottLoss(rate=0.1)], seeds=SeedSet.derive(0)
        )
        traj = rec.finish(pipeline=pipeline)
        assert not traj.validity.measurement_valid
        assert any("clean_tx_parity" == c.name for c in traj.validity.failed)

    def test_an_unmeasured_persona_does_not_invalidate_but_is_noted(self) -> None:
        """Unmeasured is not the same as failed."""
        fidelity = SimulatorFidelity(
            utterances=2,
            deterministic_source=True,
            source_kind=SourceKind.SCRIPT,
            prerecorded_fraction=0.0,
            in_persona=None,
        )
        traj = self._rec().finish(fidelity=fidelity)
        assert traj.validity.measurement_valid
        check = next(
            c for c in traj.validity.checks if c.name == "simulator_in_persona"
        )
        assert "unmeasured" in (check.detail or "")

    def test_a_drifted_caller_invalidates_the_run(self) -> None:
        fidelity = SimulatorFidelity(
            utterances=2,
            deterministic_source=True,
            source_kind=SourceKind.SCRIPT,
            prerecorded_fraction=0.0,
            in_persona=0.4,
        )
        traj = self._rec(require_in_persona=0.85).finish(fidelity=fidelity)
        assert not traj.validity.measurement_valid

    def test_validity_reasons_are_readable(self) -> None:
        rec = self._rec(max_transport_gap_s=0.1)
        rec.transport_gap(1.0, 5.0)
        assert rec.finish().validity.reasons


class TestArtifacts:
    def test_rendering_produces_a_mixed_artifact(self) -> None:
        rec = Recorder()
        rec.record_turn(_turn(received=5))
        traj = rec.finish()
        assert traj.artifacts.mixed_timeline is not None
        assert traj.artifacts.mixed_timeline.digest.startswith("sha256:")
        assert traj.artifacts.tier == "T1"

    def test_segments_are_listed(self) -> None:
        rec = Recorder()
        rec.record_turn(_turn())
        segments = rec.finish().artifacts.segments
        assert segments
        assert {"party", "t0", "t1", "digest"} <= set(segments[0])

    def test_rendering_can_be_skipped_for_tier_zero(self) -> None:
        rec = Recorder()
        rec.record_turn(_turn())
        traj = rec.finish(render=False)
        assert traj.artifacts.mixed_timeline is None
        assert traj.artifacts.tier == "T0"
