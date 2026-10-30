"""Parity statistics on synthetic outputs of both sides, in their real layouts.

The sides are plain folders (``fixtures/compare``): harbor job folders as
harbor writes them and EVA run directories as EVA writes them. Nothing is
run.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from eva_adapter.compare import GRADED
from eva_adapter.compare import Trial
from eva_adapter.compare import add_missing
from eva_adapter.compare import agreement
from eva_adapter.compare import compare_folders
from eva_adapter.compare import describe
from eva_adapter.compare import eva_trials
from eva_adapter.compare import harbor_trials
from eva_adapter.compare import ks_2samp
from eva_adapter.compare import pass_k
from eva_adapter.compare import sign_flip_p
from eva_adapter.conditions import PerturbationConfig
from eva_adapter.main import main

sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "compare"))

from compare_fixtures import RECORDS  # noqa: E402
from compare_fixtures import metrics  # noqa: E402
from compare_fixtures import record  # noqa: E402
from compare_fixtures import sides  # noqa: E402

pytestmark = pytest.mark.unit


def _compare(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    """Compare the synthetic sides (made here unless ``runs`` is given; the
    condition they ran is also the one compared)."""
    made = {
        k: kwargs.pop(k) for k in ("noise", "model", "metric_configs") if k in kwargs
    }
    if "perturbation" in kwargs:
        made["perturbation"] = kwargs["perturbation"]
    runs = kwargs.pop("runs", None) or sides(tmp_path / "sides", **made)
    kwargs.setdefault("output", tmp_path / "parity_experiment.json")
    return compare_folders(runs, **kwargs)


def test_runs_are_evas_trial_indices_and_nothing_is_dropped(tmp_path: Path) -> None:
    report = _compare(tmp_path)
    overall = report["scopes"]["overall"]
    harbor, original = overall["harbor"], overall["original"]
    # run 0: 1.2.2, 2.2.5, 16; run 1: 2.2.5 errored; run 2: 2.2.5 not
    # validated and 16 missing, so 1.2.2 alone.
    a = harbor["composites"]["EVA-A_pass"]
    assert a["runs"] == [pytest.approx(66.6667), 100.0, 0.0]
    assert a["mean"] == pytest.approx(500 / 9)
    assert a["sem"] == pytest.approx(
        math.sqrt(
            ((200 / 3 - 500 / 9) ** 2 + (100 - 500 / 9) ** 2 + (500 / 9) ** 2) / 2
        )
        / math.sqrt(3)
    )
    assert original["composites"]["EVA-A_pass"]["runs"] == [
        pytest.approx(66.6667),
        pytest.approx(33.3333),
        100.0,
    ]
    assert harbor["status"] == {
        "graded": 6, "judge_errored": 0, "ungraded": 0, "not_validated": 1,
        "errored": 1, "missing": 1,
    }  # fmt: skip
    # Calls: every attempt, against one per trial planned.
    assert harbor["calls"] == {"planned": 9, "made": 9}  # 2.2.5/2 three, errored none
    assert original["calls"] == {"planned": 9, "made": 12}
    assert (
        original["status"]["graded"] == 8 and original["status"]["not_validated"] == 1
    )
    # EVA's pass^k: records with all 3 trials scored only.
    assert harbor["composites"]["EVA-A_pass"]["records_all_scored"] == 1
    assert harbor["composites"]["EVA-A_pass"]["pass_hat_k"][2] == pytest.approx(1 / 3)
    assert original["composites"]["EVA-A_pass"]["pass_hat_k"][3] == pytest.approx(0.5)
    # Components, per domain too.
    assert report["scopes"]["itsm"]["original"]["components"]["faithfulness"][
        "mean"
    ] == pytest.approx(0.9)
    errored = next(t for t in report["trials"] if t["status"] == "errored")
    assert (errored["domain"], errored["record_id"], errored["trial"]) == (
        "airline",
        "2.2.5",
        1,
    )
    assert (
        errored["error"]
        == "VoiceInfrastructureError: ProviderError: [throttled] quota exhausted"
    )


def test_parity_experiment_is_harbors_format(tmp_path: Path) -> None:
    out = tmp_path / "out" / "parity_experiment.json"
    _compare(tmp_path, notes="synthetic", output=out)
    entry = json.loads(out.read_text())
    assert len(entry) == 1
    head = entry[0]
    assert (
        head["adapter_name"] == "eva"
        and head["model"] == "openai.realtime/gpt-realtime-1.5"
    )
    assert head["adapted_benchmark_size"] == 213 and head["parity_benchmark_size"] == 3
    assert head["number_of_runs"] == 3 and head["notes"] == "synthetic"
    assert head["original_parity_repo"] == "https://github.com/ServiceNow/eva"
    rows = {(m["benchmark_name"], m["metric"]): m for m in head["metrics"]}
    assert set(rows) == {
        (f"eva/{scope}", name)
        for scope in ("airline", "itsm", "overall")
        for name in ("EVA-A_pass", "EVA-X_pass", "EVA-overall_pass")
    }
    overall = rows[("eva/overall", "EVA-A_pass")]
    assert overall["harbor"] == "55.56% +/- 29.40%"  # stdev 50.92 / sqrt 3
    assert overall["harbor_runs"] == [66.6667, 100.0, 0.0]
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["eva_summaries"] == {}
    assert report["judges"] == {"audio": "gemini-3-flash-preview@minimal", "run": None}


def test_diagnostics_count_every_attempt_on_both_sides(tmp_path: Path) -> None:
    report = _compare(tmp_path)
    harbor = report["scopes"]["overall"]["harbor"]
    original = report["scopes"]["overall"]["original"]
    # Original: 1.2.2 trial 1 reran once; 2.2.5 trial 2 used all three.
    assert original["attempts"] == {1: 7, 2: 1, 3: 1}
    assert original["rerun_rate"] == pytest.approx(2 / 9)
    assert original["end_reasons_all_attempts"] == {
        "goodbye": 8,
        "inactivity_timeout": 3,
        "error": 1,
    }
    assert original["end_reasons"] == {"goodbye": 8, "inactivity_timeout": 1}
    assert original["validation_rate"] == pytest.approx(8 / 9)
    assert harbor["validation_rate"] == pytest.approx(6 / 7)
    assert harbor["caller"] == {"turns_per_call": 2.0, "words_per_turn": 2.5}
    assert harbor["call_s"]["median"] == 120.0
    assert harbor["response_speed_s"]["turns"] == 12
    tokens = report["tokens"]
    # Attempts: harbor 9 (3 archived for 2.2.5/2), original 12.
    assert (
        tokens["harbor"]["agent_input"] == 900
        and tokens["original"]["agent_input"] == 1200
    )
    assert tokens["original"]["caller_input"] == 12 * 1000
    assert tokens["original"]["caller_initiated_input"] == 12 * 1200
    assert tokens["original"]["caller_seconds"] == 12 * 60
    assert (
        tokens["harbor"]["judge_calls"] == 7 and tokens["original"]["judge_calls"] == 2
    )
    ks = report["pairs"]["harbor-vs-original"]["overall"]["response_speed_ks"]
    assert ks["n"] == [12, 16] and 0 < ks["d"] <= 1


def test_per_record_agreement_is_tau_voices_analysis(tmp_path: Path) -> None:
    report = _compare(tmp_path)
    a = report["pairs"]["harbor-vs-original"]["overall"]["EVA-A_pass"]
    # 1.2.2: harbor 2/3, original 3/3; 2.2.5: 0/1 vs 0/2; 16: 2/2 vs 2/3.
    assert (a["records"], a["same"], a["harbor_more"], a["original_more"]) == (
        3,
        1,
        1,
        1,
    )
    assert a["count_difference"] == {"-1": 1}  # only 1.2.2 has n equal
    assert a["successes"] == {"harbor": 4, "original": 5}
    assert a["sign_flip_p"] == 1.0  # the differences cancel
    assert a["flipping_records"] == {"harbor": 1, "original": 1}
    assert a["ranges_overlap"] is True and a["lopsided"] == []
    assert a["gap_pts"] == pytest.approx(abs(500 / 9 - 200 / 3))


def test_the_noise_floor_is_original_against_original(tmp_path: Path) -> None:
    report = _compare(tmp_path, noise=True)
    assert set(report["pairs"]) == {
        "harbor-vs-original",
        "original-vs-original2",
        "harbor-vs-original2",
    }
    floor = report["noise_floor"]["overall"]["EVA-A_pass"]
    # original2's run 2: 1.2.2 fails, so [66.7, 33.3, 50].
    assert floor["original_vs_original2_pts"] == pytest.approx(200 / 3 - 50)
    assert floor["harbor_vs_original_pts"] == pytest.approx(200 / 3 - 500 / 9)
    assert floor["harbor_within_noise"] is True


def test_a_different_agent_on_the_two_sides_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="different agents"):
        _compare(tmp_path, model="gpt-realtime-mini")


def test_what_was_expected_and_absent_is_missing(tmp_path: Path) -> None:
    runs = sides(tmp_path / "sides")
    expected = {
        side: {(d, r, k) for d, ids in RECORDS.items() for r in ids for k in range(3)}
        | {("itsm", "20", 0)}
        for side in ("harbor", "original")
    }
    report = _compare(tmp_path, runs=runs, expected=expected, trials=3)
    for side in ("harbor", "original"):
        missing = {
            (t["domain"], t["record_id"], t["trial"])
            for t in report["trials"]
            if t["side"] == side and t["status"] == "missing"
        }
        assert ("itsm", "20", 0) in missing
    head = json.loads((tmp_path / "parity_experiment.json").read_text())[0]
    assert head["parity_benchmark_size"] == 4


# ------------------------------------------------------------------ units


def _t(side: str, record: str, k: int, a: float | None, status: str = GRADED) -> Trial:
    return Trial(side, "airline", record, k, status, composites={"EVA-A_pass": a})


def test_pass_k_is_evas_estimator() -> None:
    trials = [_t("h", "r", k, v) for k, v in enumerate([1.0, 1.0, 0.0])]
    found = pass_k(trials, "EVA-A_pass", 3)
    assert found["pass_hat_k"] == {
        1: pytest.approx(2 / 3),
        2: pytest.approx(1 / 3),
        3: 0.0,
    }
    assert found["pass_at_k"][2] == 1.0 and found["pass_at_k"][1] == pytest.approx(
        2 / 3
    )
    assert pass_k(trials[:2], "EVA-A_pass", 3)["records_all_scored"] == 0


def test_describe_leaves_unscored_runs_out_and_counts_values() -> None:
    trials = [
        _t("h", "r", 0, 1.0),
        _t("h", "r", 1, None),
        _t("h", "s", 1, 0.0, "missing"),
    ]
    found = describe(trials, lambda t: t.composites.get("EVA-A_pass"), scale=100)
    assert found["runs"] == [100.0] and found["sem"] is None and found["values"] == 1


def test_sign_flip_is_exact_when_small() -> None:
    assert sign_flip_p([1.0, 1.0, 1.0]) == pytest.approx(2 / 8)  # all +, or all -
    assert sign_flip_p([0.0, 0.0]) == 1.0
    assert sign_flip_p([1.0] * 20) == pytest.approx(2 / 2**20, abs=5e-4)  # sampled


def test_ks_matches_known_cases() -> None:
    same = ks_2samp([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert same is not None and same["d"] == 0.0 and same["p"] == 1.0
    apart = ks_2samp([1.0] * 30, [5.0] * 30)
    assert apart is not None and apart["d"] == 1.0 and apart["p"] < 1e-6
    assert ks_2samp([], [1.0]) is None


def test_lopsided_records_are_named() -> None:
    harbor = [_t("harbor", "r", k, 1.0) for k in range(3)]
    original = [_t("original", "r", k, 0.0) for k in range(3)]
    found = agreement(harbor, original, "EVA-A_pass", 3, ("harbor", "original"))
    assert found["lopsided"] == [
        {"domain": "airline", "record_id": "r", "harbor": "3/3", "original": "0/3"}
    ]
    assert found["sign_flip_p"] == 1.0  # one record: both signs equally far


def test_every_side_is_expected_to_have_what_any_side_has() -> None:
    trials = add_missing([_t("harbor", "r", 0, 1.0), _t("original", "s", 0, 1.0)])
    assert {(t.side, t.record_id, t.status) for t in trials} == {
        ("harbor", "r", GRADED),
        ("harbor", "s", "missing"),
        ("original", "r", "missing"),
        ("original", "s", GRADED),
    }


def test_a_one_trial_eva_run_is_trial_zero(tmp_path: Path) -> None:
    run_dir = tmp_path / "r1"
    record(run_dir / "records" / "16_failed_attempt_1", end="error")
    record(run_dir / "records" / "16", metrics=metrics(1.0, 1.0, [1.0]))
    (run_dir / "config.json").write_text(json.dumps({"domain": "itsm"}))
    (run_dir / "evaluation_summary.json").write_text(
        json.dumps(
            {"simulation": {"successful_record_ids": ["16"], "failed_record_ids": []}}
        )
    )
    (trial,) = eva_trials(run_dir, "original")
    assert (trial.record_id, trial.trial, trial.status, trial.attempts) == (
        "16",
        0,
        GRADED,
        2,
    )
    assert trial.end_reasons == ["error", "goodbye"]


# ------------------------------------------------------------- conditions

NOISY = PerturbationConfig("coffee_shop", 15.0)


def test_a_perturbed_comparison_says_so(tmp_path: Path) -> None:
    report = _compare(tmp_path, perturbation=NOISY, notes="synthetic")
    assert report["condition"] == "coffee_shop-15db"
    assert report["perturbation"] == NOISY.as_dict()
    # The same numbers as the clean synthetic run: the condition only names it.
    harbor = report["scopes"]["overall"]["harbor"]
    assert harbor["composites"]["EVA-A_pass"]["runs"] == [
        pytest.approx(66.6667),
        100.0,
        0.0,
    ]
    # The errored trial left no summary; its perturbed folder still names it.
    errored = next(t for t in report["trials"] if t["status"] == "errored")
    assert (errored["domain"], errored["record_id"]) == ("airline", "2.2.5")
    assert {t["condition"] for t in report["trials"] if t["status"] != "missing"} == {
        "coffee_shop-15db",
        None,  # the errored trial
    }
    (head,) = json.loads((tmp_path / "parity_experiment.json").read_text())
    assert head["notes"].startswith(
        "EVA perturbation condition coffee_shop-15db on both sides"
    )
    assert "statistically" in head["notes"] and head["notes"].endswith("synthetic")


def test_trials_of_another_condition_are_refused(tmp_path: Path) -> None:
    runs = sides(tmp_path / "sides", perturbation=NOISY)
    config = runs["original"][1] / "config.json"
    data = json.loads(config.read_text())
    data["perturbation"]["snr_db"] = 5.0  # EVA ran it at another SNR
    config.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="another condition than the run's"):
        _compare(tmp_path, runs=runs, perturbation=NOISY)
    # Compared as clean, the noisy trials are strays too.
    with pytest.raises(SystemExit, match="another condition than the run's"):
        _compare(tmp_path, runs=sides(tmp_path / "noisy", perturbation=NOISY))
    assert not (tmp_path / "parity_experiment.json").exists()


# ------------------------------------------------------------- the audio judges

LOW = {
    metric: {
        "audio_judge_model": "gemini-3.8-flash",
        "judge_params": {"reasoning_effort": "low"},
    }
    for metric in ("agent_speech_fidelity", "user_speech_fidelity")
}


def test_the_audio_judges_are_read_from_every_trial(tmp_path: Path) -> None:
    report = _compare(tmp_path, metric_configs=LOW)
    assert report["judges"]["audio"] == "gemini-3.8-flash@low"
    judged = {t["judges"] for t in report["trials"] if t["status"] != "missing"}
    assert judged <= {"gemini-3.8-flash@low", None}  # errored: no config left
    clean = _compare(tmp_path / "clean")
    assert clean["judges"]["audio"] == "gemini-3-flash-preview@minimal"


def test_trials_judged_apart_are_refused(tmp_path: Path) -> None:
    runs = sides(tmp_path / "sides", metric_configs=LOW)
    config = runs["original"][1] / "config.json"
    data = json.loads(config.read_text())
    data["metric_configs"] = {}  # EVA's own judges, among gemini-3.8-flash ones
    config.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="judged by different audio judges"):
        _compare(tmp_path, runs=runs)
    # Named, every trial is held to it.
    with pytest.raises(SystemExit, match="judged by other audio judges than the run's"):
        _compare(tmp_path, runs=runs, judge_label="gemini-3.8-flash@low")


# ------------------------------------------------------------- judge errors


def _judge_errored(tmp_path: Path) -> dict[str, list[Path]]:
    """The synthetic sides, harbor's 1.2.2 trial 2 graded with a judge error,
    and its 2.2.5 trial 2 never validated after validation judge errors."""
    runs = sides(tmp_path / "sides")
    job = runs["harbor"][0]
    (graded,) = [t for t in job.glob("*__2") if t.is_dir()]
    eva = graded / "agent" / "call" / "eva"
    for path in (eva / "metrics.json", eva / "run/records/1.2.2/trial_2/metrics.json"):
        data = json.loads(path.read_text())
        data["metrics"]["agent_speech_fidelity"] = {"error": "429 Too Many Requests"}
        data["aggregate_metrics"]["EVA-A_pass"] = None
        path.write_text(json.dumps(data))
    (refused,) = [t for t in job.glob("*__5") if t.is_dir()]
    for archived in (refused / "agent/call/eva/run/records/2.2.5").glob("*_failed_*"):
        (archived / "metrics.json").write_text(
            json.dumps({"metrics": {"user_speech_fidelity": {"error": "429"}}})
        )
    return runs


def test_judge_errors_are_results_and_said_in_the_notes(tmp_path: Path) -> None:
    report = _compare(tmp_path, runs=_judge_errored(tmp_path))
    harbor = report["scopes"]["overall"]["harbor"]
    assert harbor["status"]["judge_errored"] == 1 and harbor["status"]["graded"] == 5
    assert harbor["judge_errors"] == {
        "trials": 1,
        "metrics": {"agent_speech_fidelity": 1},
    }
    assert harbor["validation_judge_errors"] == {
        "failed_attempts": 3,
        "not_validated": 1,
        "not_validated_after_judge_error": 1,
        "not_validated_judge_error_every_attempt": 1,
        "metrics": {"user_speech_fidelity": 3},
    }
    # The judge-errored trial's EVA-X still counts; its EVA-A is unset.
    assert harbor["composites"]["EVA-A_pass"]["values"] == 5
    assert harbor["composites"]["EVA-X_pass"]["values"] == 6
    assert harbor["regraded"] == {"trials": 0, "fixed": 0, "metrics": {}}
    assert any("Still graded with a judge error" in n for n in report["notes"])
    assert any("does not undo a validation decision" in n for n in report["notes"])


def test_a_regrade_and_the_runs_own_notes_come_from_the_caller(tmp_path: Path) -> None:
    runs = _judge_errored(tmp_path)

    def fixed(trial_dir: Path) -> dict[str, Any] | None:
        return (
            {"metrics": ["agent_speech_fidelity"], "fixed": True}
            if trial_dir.name.endswith("__2")
            else None
        )

    report = _compare(
        tmp_path,
        runs=runs,
        notes="mine",
        run_notes=["How the run was judged."],
        judges={"audio": "as the run recorded it"},
        harbor_regrade=fixed,
        eva_regrades=lambda run_dir: {
            "16/trial_1": {"metrics": ["faithfulness"], "fixed": False}
        },
    )
    harbor = report["scopes"]["overall"]["harbor"]
    original = report["scopes"]["overall"]["original"]
    assert harbor["regraded"] == {
        "trials": 1,
        "fixed": 1,
        "metrics": {"agent_speech_fidelity": 1},
    }
    assert original["regraded"]["trials"] == 1 and original["regraded"]["fixed"] == 0
    assert report["judges"]["run"] == {"audio": "as the run recorded it"}
    (head,) = json.loads((tmp_path / "parity_experiment.json").read_text())
    assert head["notes"].startswith("How the run was judged. Re-graded from the")
    assert head["notes"].endswith(" mine")


def test_harbor_trials_take_the_task_folder_when_no_summary_is_left(
    tmp_path: Path,
) -> None:
    runs = sides(tmp_path / "sides")
    trials = harbor_trials(runs["harbor"][0])
    errored = [t for t in trials if t.status == "errored"]
    assert [(t.domain, t.record_id, t.trial) for t in errored] == [
        ("airline", "2.2.5", 1)
    ]


# ------------------------------------------------------------------ CLI


def test_the_cli_compares_plain_folders(tmp_path: Path) -> None:
    runs = sides(tmp_path / "sides", noise=True)
    records = tmp_path / "records.json"
    records.write_text(json.dumps({"records": RECORDS}))
    out = tmp_path / "out" / "parity.json"
    argv = ["compare", "--harbor", *map(str, runs["harbor"])]
    argv += ["--original", *map(str, runs["original"])]
    argv += ["--original2", *map(str, runs["original2"])]
    argv += ["--records", str(records), "--notes", "cli", "-o", str(out)]
    with pytest.raises(SystemExit) as done:
        main(argv)
    assert done.value.code == 0
    (head,) = json.loads(out.read_text())
    assert head["parity_benchmark_size"] == 3 and head["notes"] == "cli"
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert set(report["pairs"]) == {
        "harbor-vs-original",
        "original-vs-original2",
        "harbor-vs-original2",
    }
    assert report["scopes"]["overall"]["harbor"]["status"]["missing"] == 1


def test_the_cli_takes_the_condition_compared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = sides(tmp_path / "sides", perturbation=NOISY)
    out = tmp_path / "parity.json"
    argv = ["compare", "--harbor", *map(str, runs["harbor"])]
    argv += ["--original", *map(str, runs["original"]), "-o", str(out)]
    with pytest.raises(SystemExit, match="another condition"):
        main(argv)
    with pytest.raises(SystemExit) as done:
        main([*argv, "--background-noise", "coffee_shop", "--snr-db", "15"])
    assert done.value.code == 0
    assert json.loads(out.read_text())[0]["notes"].startswith(
        "EVA perturbation condition coffee_shop-15db"
    )
    with pytest.raises(SystemExit) as refused:
        main([*argv, "--snr-db", "5"])
    assert refused.value.code == 2 and "needs --background-noise" in (
        capsys.readouterr().err
    )
