"""The scoring boundary.

Who may define a metric, what `test.sh` calls, what comes along for the ride,
and how invalidity reaches the job level.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.capabilities import Verdict
from harbor_voice.metrics import Direction
from harbor_voice.metrics import Family
from harbor_voice.metrics import MetricContext
from harbor_voice.metrics import MetricRegistry
from harbor_voice.metrics import SelectionError
from harbor_voice.metrics import TaskNamespaceError
from harbor_voice.metrics import Value
from harbor_voice.metrics import check_selection
from harbor_voice.metrics import harbor_metric_payload
from harbor_voice.metrics import load_job
from harbor_voice.metrics import load_trial_dir
from harbor_voice.metrics import metric
from harbor_voice.metrics import task_metric
from harbor_voice.trajectory.model import ConfigMatrix
from harbor_voice.trajectory.model import PartyRecord
from harbor_voice.trajectory.model import PartyRole
from harbor_voice.trajectory.model import ValidityCheck
from harbor_voice.trajectory.model import ValidityRecord
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.model import VoiceTurn
from harbor_voice.verify import load_trajectory
from harbor_voice.verify import main
from harbor_voice.verify import score

pytestmark = pytest.mark.unit


# ------------------------------------------------- the container surface


class TestScoringDoesNotDragTheWorldIn:
    """A subprocess, because in-process this passes or fails on test order."""

    def _modules_after(self, statement: str) -> set[str]:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"{statement}; import sys, json; "
                "print(json.dumps(sorted(sys.modules)))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return set(json.loads(result.stdout))

    def test_metrics_does_not_import_the_provider_layer(self) -> None:
        """A trial container scoring a JSON file needs no credential table."""
        loaded = self._modules_after("import harbor_voice.metrics")
        assert not [m for m in loaded if m.startswith("harbor_voice.providers")]

    def test_metrics_does_not_import_the_recorder_or_simulator(self) -> None:
        loaded = self._modules_after("import harbor_voice.metrics")
        assert "harbor_voice.trajectory.recorder" not in loaded
        assert not [m for m in loaded if m.startswith("harbor_voice.simulator")]
        assert not [m for m in loaded if m.startswith("harbor_voice.cascade")]

    def test_verify_is_equally_light(self) -> None:
        loaded = self._modules_after("import harbor_voice.verify")
        assert not [m for m in loaded if m.startswith("harbor_voice.providers")]

    def test_atif_is_still_imported_because_the_format_needs_it(self) -> None:
        """ATIF-voice *is* ATIF; this dependency is honest, not accidental."""
        loaded = self._modules_after("import harbor_voice.metrics")
        assert "harbor.models.trajectories.trajectory" in loaded

    def test_the_recorder_still_resolves_lazily(self) -> None:
        from harbor_voice.trajectory import Recorder
        from harbor_voice.trajectory import RecorderConfig

        assert Recorder.__name__ == "Recorder"
        assert RecorderConfig.__name__ == "RecorderConfig"

    def test_an_unknown_attribute_still_raises(self) -> None:
        import harbor_voice.trajectory as pkg

        with pytest.raises(AttributeError, match="no attribute"):
            _ = pkg.NotAThing


# ------------------------------------------------------ the adapter seam


class TestOnlyAnAdapterMayDefineTaskSuccess:
    def test_a_task_metric_must_use_the_reserved_prefix(self) -> None:
        with pytest.raises(TaskNamespaceError, match="must be named"):

            @task_metric(
                name="booking_completed",
                direction=Direction.HIGHER_BETTER,
                benchmark="tau-voice",
                registry=MetricRegistry(),
            )
            def _m(context: MetricContext) -> Value:
                return Value(value=1.0)

    def test_a_task_metric_must_name_its_benchmark(self) -> None:
        """A parity disagreement needs somewhere to point."""
        with pytest.raises(TaskNamespaceError, match="must name the adapter"):

            @task_metric(
                name="task.booking_completed",
                direction=Direction.HIGHER_BETTER,
                benchmark="",
                registry=MetricRegistry(),
            )
            def _m(context: MetricContext) -> Value:
                return Value(value=1.0)

    def test_the_harness_is_refused_the_task_namespace(self) -> None:
        """harbor-voice must not quietly start defining task success."""
        with pytest.raises(ValueError, match="only an adapter may fill"):

            @metric(
                name="task.sneaky",
                family=Family.TASK,
                direction=Direction.HIGHER_BETTER,
                registry=MetricRegistry(),
            )
            def _m(context: MetricContext) -> Value:
                return Value(value=1.0)

    def test_a_registered_task_metric_carries_its_benchmark_to_the_record(
        self,
    ) -> None:
        registry = MetricRegistry()

        @task_metric(
            name="task.booking_completed",
            direction=Direction.HIGHER_BETTER,
            benchmark="tau-voice",
            registry=registry,
        )
        def _m(context: MetricContext) -> Value:
            return Value(value=1.0)

        defn = registry.get("task.booking_completed")
        assert defn.family is Family.TASK
        assert defn.benchmark == "tau-voice"

        result = registry.evaluate(
            ["task.booking_completed"],
            MetricContext(trajectory=VoiceTrajectory()),
        )[0]
        assert result.as_dict()["benchmark"] == "tau-voice"


class TestSelectionValidation:
    """A typo must be a load error, not a missing column nobody asks about."""

    def test_an_unknown_metric_is_caught(self) -> None:
        problems = check_selection(["divergence.wer_said_heardd"])
        assert len(problems) == 1
        assert problems[0].kind == "unknown_metric"

    def test_an_unregistered_task_metric_says_so_specifically(self) -> None:
        problems = check_selection(["task.booking_completed"])
        assert "registered by the adapter" in problems[0].detail

    def test_an_active_layer_with_no_reader_is_caught(self) -> None:
        """A perturbation with no metric only makes the number worse."""
        problems = check_selection(
            ["timing.response_latency_s"], active_layers=["L5.backchannel"]
        )
        assert [p.kind for p in problems] == ["unpaired_layer"]
        assert "no paired metric" in problems[0].detail

    def test_a_paired_layer_passes(self) -> None:
        assert (
            check_selection(
                ["selectivity.backchannel"], active_layers=["L5.backchannel"]
            )
            == []
        )

    def test_every_problem_is_reported_not_just_the_first(self) -> None:
        """Fixing one fault to meet the next is a three-run debug cycle."""
        problems = check_selection(
            ["nope", "also_nope"], active_layers=["L5.vocal_tics"]
        )
        assert len(problems) == 3

    def test_require_raises_with_all_of_them(self) -> None:
        from harbor_voice.metrics import require_valid_selection

        with pytest.raises(SelectionError) as exc:
            require_valid_selection(["nope"], active_layers=["L5.vocal_tics"])
        assert len(exc.value.problems) == 2
        assert "2 problem(s)" in str(exc.value)


# --------------------------------------------- what test.sh calls


def a_trajectory(*, valid: bool = True) -> VoiceTrajectory:
    checks = (
        (ValidityCheck(name="clean_tx_parity", ok=True),)
        if valid
        else (ValidityCheck(name="transport_gap_budget", ok=False, detail="1.4s"),)
    )
    return VoiceTrajectory(
        trial_id="t1",
        task="reschedule",
        agent="cascade",
        parties=(
            PartyRecord(
                id="agent",
                role=PartyRole.SYSTEM_UNDER_TEST,
                capabilities=Capabilities(
                    duplex=frozenset({Duplex.HALF}),
                    pipeline=Pipeline.CASCADE,
                    turn_control=TurnControl.HARNESS_VAD,
                    exposes_asr=True,
                    exposes_stages=frozenset({Stage.ASR_FINAL}),
                ),
            ),
        ),
        config_matrix=ConfigMatrix(
            duplex=Duplex.HALF,
            pipeline=Pipeline.CASCADE,
            floor_policy="half",
            profile="telephony",
            seed_root=7,
        ),
        turns=(
            VoiceTurn(idx=0, party="caller", said="hello there", heard="hello there"),
        ),
        validity=ValidityRecord(checks=checks),
    )


class TestVerifyEntryPoint:
    def test_it_reads_one_file_and_writes_three(self, tmp_path: Path) -> None:
        path = tmp_path / "trajectory.json"
        path.write_text(a_trajectory().model_dump_json(), encoding="utf-8")
        out = tmp_path / "verifier"

        code = main(
            [
                "--trajectory",
                str(path),
                "--out",
                str(out),
                "--metric",
                "timing.response_latency_s",
            ]
        )
        assert code == 0
        assert {p.name for p in out.iterdir()} == {
            "measurement.json",
            "validity.json",
            "reward.json",
        }

    def test_an_invalid_measurement_still_exits_zero(self, tmp_path: Path) -> None:
        """Not a verifier failure — a measurement that did not hold."""
        path = tmp_path / "trajectory.json"
        path.write_text(a_trajectory(valid=False).model_dump_json(), encoding="utf-8")
        out = tmp_path / "verifier"

        assert main(["--trajectory", str(path), "--out", str(out)]) == 0
        assert not (out / "reward.json").exists()
        assert (
            json.loads((out / "validity.json").read_text())["measurement_valid"]
            is False
        )

    def test_a_broken_trajectory_exits_two(self, tmp_path: Path) -> None:
        """Reserved for the verifier itself being wrong."""
        bad = tmp_path / "nope.json"
        assert main(["--trajectory", str(bad), "--out", str(tmp_path)]) == 2

    def test_an_unknown_metric_exits_two(self, tmp_path: Path) -> None:
        path = tmp_path / "trajectory.json"
        path.write_text(a_trajectory().model_dump_json(), encoding="utf-8")
        code = main(
            ["--trajectory", str(path), "--out", str(tmp_path), "--metric", "nope"]
        )
        assert code == 2

    def test_it_accepts_a_trajectory_written_as_plain_atif(
        self, tmp_path: Path
    ) -> None:
        """Both on-disk shapes are legitimate; the author should not have to know."""
        path = tmp_path / "atif.json"
        path.write_text(a_trajectory().to_atif().model_dump_json(), encoding="utf-8")
        loaded = load_trajectory(path)
        assert loaded.trial_id == "t1"
        assert len(loaded.turns) == 1

    def test_score_defaults_to_every_registered_metric(self) -> None:
        outputs = score(a_trajectory())
        assert outputs.measurement["counts"]["reported"] > 0

    def test_a_pinned_selection_is_what_a_task_should_use(self) -> None:
        """Adding a metric to harbor-voice must not change an existing task."""
        outputs = score(a_trajectory(), metrics=["divergence.wer_said_heard"])
        assert set(outputs.measurement["metrics"]) == {"divergence.wer_said_heard"}


# ------------------------------------ invalidity at the job level


def write_trial(root: Path, name: str, *, valid: bool) -> Path:
    directory = root / name
    main(
        [
            "--trajectory",
            str(_dump(root, name, valid=valid)),
            "--out",
            str(directory),
            "--metric",
            "divergence.wer_said_heard",
        ]
    )
    return directory


def _dump(root: Path, name: str, *, valid: bool) -> Path:
    path = root / f"{name}.json"
    path.write_text(a_trajectory(valid=valid).model_dump_json(), encoding="utf-8")
    return path


class TestInvalidityTravelsOutOfBand:
    """Harbor's flat JSONL has already lost it; the trial dirs have not."""

    def test_a_trial_dir_round_trips_the_verdicts(self, tmp_path: Path) -> None:
        directory = write_trial(tmp_path, "t1", valid=True)
        trial = load_trial_dir(directory)
        assert trial is not None
        assert trial.trial_id == "t1"
        result = trial.get("divergence.wer_said_heard")
        assert result is not None
        assert result.value == 0.0
        assert result.verdict is Verdict.UNATTESTED

    def test_an_invalid_trial_is_recognised_as_invalid_not_as_zero(
        self, tmp_path: Path
    ) -> None:
        directory = write_trial(tmp_path, "bad", valid=False)
        trial = load_trial_dir(directory)
        assert trial is not None
        assert not trial.measurement_valid
        assert not (directory / "reward.json").exists()

    def test_a_directory_without_a_measurement_is_none(self, tmp_path: Path) -> None:
        """Never scored is different from scored and did not hold."""
        assert load_trial_dir(tmp_path) is None

    def test_a_job_collects_in_a_stable_order(self, tmp_path: Path) -> None:
        for name in ("c", "a", "b"):
            write_trial(tmp_path, name, valid=True)
        first = [t.trial_id for t in load_job(tmp_path)]
        second = [t.trial_id for t in load_job(tmp_path)]
        assert first == second
        assert len(first) == 3

    def test_the_harbor_payload_carries_the_invalid_count(self, tmp_path: Path) -> None:
        """So the leaderboard number and ours cannot silently differ."""
        write_trial(tmp_path, "good1", valid=True)
        write_trial(tmp_path, "good2", valid=True)
        write_trial(tmp_path, "bad", valid=False)

        payload = harbor_metric_payload(load_job(tmp_path))
        assert payload["n_trials"] == 3
        assert payload["n_valid"] == 2
        assert payload["n_invalid"] == 1
        assert all(isinstance(v, (int, float)) for v in payload.values())

    def test_the_invalid_trial_is_excluded_from_the_mean(self, tmp_path: Path) -> None:
        write_trial(tmp_path, "good", valid=True)
        write_trial(tmp_path, "bad", valid=False)
        payload = harbor_metric_payload(load_job(tmp_path))
        assert payload["divergence.wer_said_heard"] == 0.0
        assert payload["n_invalid"] == 1

    def test_a_primary_metric_becomes_harbors_mean_key(self, tmp_path: Path) -> None:
        write_trial(tmp_path, "good", valid=True)
        payload = harbor_metric_payload(
            load_job(tmp_path), primary="divergence.wer_said_heard"
        )
        assert payload["mean"] == 0.0


class TestItWorksTheWayTestShWouldCallIt:
    """Through the process boundary, not through an in-process import.

    The in-process tests above prove the logic. This proves the thing a task
    author actually writes — one line in a shell script — reaches it.
    """

    def test_the_module_runs_as_a_subprocess(self, tmp_path: Path) -> None:
        path = tmp_path / "trajectory.json"
        path.write_text(a_trajectory().model_dump_json(), encoding="utf-8")
        out = tmp_path / "verifier"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "harbor_voice.verify",
                "--trajectory",
                str(path),
                "--out",
                str(out),
                "--metric",
                "divergence.wer_said_heard",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "1 reported" in result.stdout

        reward = json.loads((out / "reward.json").read_text())
        assert reward == {"divergence.wer_said_heard": 0.0}

    def test_harbors_verifier_would_accept_that_reward_file(
        self, tmp_path: Path
    ) -> None:
        """Harbor rejects non-numeric and non-finite values; ours are neither."""
        import math

        path = tmp_path / "trajectory.json"
        path.write_text(a_trajectory().model_dump_json(), encoding="utf-8")
        out = tmp_path / "verifier"
        main(["--trajectory", str(path), "--out", str(out)])

        reward = json.loads((out / "reward.json").read_text())
        assert isinstance(reward, dict)
        for key, value in reward.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))
            assert math.isfinite(value)


class TestTheTwoFormsOfLayerPairing:
    """Perturbation–metric pairing is satisfied two ways, because there are two kinds of layer."""

    def test_a_probe_layer_pairs_by_precondition(self) -> None:
        """No backchannel score exists without a backchannel."""
        assert (
            check_selection(
                ["selectivity.backchannel"], active_layers=["L5.backchannel"]
            )
            == []
        )

    def test_a_channel_layer_pairs_by_sensitivity(self) -> None:
        """WER is defined on a clean line; the codec is what makes it move."""
        assert (
            check_selection(["divergence.wer_said_heard"], active_layers=["L4.codec"])
            == []
        )

    def test_sensitivity_covers_every_perturbation_in_its_layer(self) -> None:
        """Declared per layer, so adding one does not go stale."""
        assert (
            check_selection(
                ["divergence.wer_said_heard"],
                active_layers=["L4.codec", "L4.frame_loss", "L3.background_bed"],
            )
            == []
        )

    def test_an_unrelated_metric_does_not_pair(self) -> None:
        problems = check_selection(
            ["validity.checks_passed"], active_layers=["L4.codec"]
        )
        assert [p.kind for p in problems] == ["unpaired_layer"]
        assert "sensitive_to=L4" in problems[0].detail

    def test_wer_stays_applicable_to_the_clean_baseline(self) -> None:
        """The reason sensitivity is not a precondition."""
        from harbor_voice.metrics import REGISTRY

        assert REGISTRY.get("divergence.wer_said_heard").requires == (
            "cap:exposes_asr",
        )
        outputs = score(a_trajectory(), metrics=["divergence.wer_said_heard"])
        assert outputs.reward == {"divergence.wer_said_heard": 0.0}
