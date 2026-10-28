"""Parity statistics: tau2's Pass^k, run means, and errored trials left out."""

from __future__ import annotations

import pytest

from tau_voice.compare import Record
from tau_voice.compare import INTERACTION
from tau_voice.compare import compare
from tau_voice.compare import interaction_table
from tau_voice.compare import pass_hat_k
from tau_voice.compare import run_means
from tau_voice.compare import summarise


def _runs(
    task: str, rewards: list[float | None], domain: str = "airline"
) -> list[Record]:
    return [Record(domain, task, run, reward) for run, reward in enumerate(rewards)]


def test_pass_hat_k_is_taus_estimator() -> None:
    # c = 2 successes in n = 3 runs: C(2,k) / C(3,k) for k = 1, 2, 3.
    records = _runs("a", [1.0, 1.0, 0.0])
    assert pass_hat_k(records, 1) == pytest.approx(2 / 3)
    assert pass_hat_k(records, 2) == pytest.approx(1 / 3)
    assert pass_hat_k(records, 3) == 0.0
    # Averaged over tasks.
    both = records + _runs("b", [1.0, 1.0, 1.0])
    assert pass_hat_k(both, 2) == pytest.approx((1 / 3 + 1) / 2)


def test_run_means_and_sem_over_runs() -> None:
    records = _runs("a", [1.0, 0.0, 1.0]) + _runs("b", [1.0, 1.0, 0.0])
    assert run_means(records) == [1.0, 0.5, 0.5]
    summary = summarise(records, runs=3)
    assert summary["mean"] == pytest.approx(200 / 3)
    assert summary["sem"] == pytest.approx(16.6667, abs=1e-3)  # stdev 28.87 / sqrt 3


def test_errored_trials_are_counted_not_averaged() -> None:
    records = _runs("a", [1.0, None, 1.0]) + _runs("b", [0.0, 1.0, 0.0])
    summary = summarise(records, runs=3)
    assert summary["errored"] == 1 and summary["graded"] == 5
    assert summary["runs"] == [50.0, 100.0, 50.0]  # run 1 averages task b alone


def test_overall_pools_domains() -> None:
    harbor = _runs("a", [1.0]) + _runs("x", [0.0], domain="retail")
    report = compare(harbor, harbor, runs=1)
    assert set(report) == {"airline", "retail", "overall"}
    assert report["overall"]["harbor"]["runs"] == [50.0]


def test_the_interaction_panel_reads_tau2s_metric_names() -> None:
    panel = {key: 0.5 for key in INTERACTION}
    metrics = {
        "original": {"domains": {"airline": panel}, "overall": panel},
        "harbor": {"domains": {}, "overall": None},
    }
    table = interaction_table(metrics).splitlines()
    assert table[0].split(" | ")[2:4] == ["L_R (s)", "L_Y (s)"]
    assert "| airline | original | 0.500 |" in table[2]
    assert table[3].startswith("| airline | harbor | n/a |")
    assert table[-1].startswith("| overall | harbor | n/a |")


def test_tokens_count_every_attempt_on_both_sides(tmp_path) -> None:
    import json

    from tau_voice.compare import harbor_tokens
    from tau_voice.compare import original_tokens

    call = tmp_path / "job" / "airline-2__x" / "agent" / "call"
    call.mkdir(parents=True)
    (call / "outcome.json").write_text(
        json.dumps(
            {
                "summary": {
                    "speech_complexity": "regular",
                    "usage": [
                        {"component": "agent", "input_tokens": 100, "cached_input_tokens": 80,
                         "output_tokens": 10, "input_audio_tokens": 20, "output_audio_tokens": 8},
                        {"component": "agent.backend", "input_tokens": 7, "output_tokens": 1},
                        {"component": "caller.llm", "input_tokens": 50, "output_tokens": 5},
                    ],
                }
            }
        )
    )  # fmt: skip
    harbor = harbor_tokens([tmp_path / "job"], "regular")
    # OpenAI Live's backend counts as the agent's.
    assert harbor["agent_input"] == 107 and harbor["caller_output"] == 5
    assert harbor_tokens([tmp_path / "job"], "control")["agent_input"] == 0

    def sim(agent_in: int) -> dict:
        return {
            "task_id": "2",
            "speech_environment": {"complexity": "regular"},
            "agent_usage": {"records": [{"input_tokens": agent_in, "output_tokens": 1}]},
            "ticks": [{"user_chunk": {"usage": {"prompt_tokens": 7, "completion_tokens": 1}}}],
        }  # fmt: skip

    save = tmp_path / "tau2"
    (save / "hallucination_discarded").mkdir(parents=True)
    (save / "results.json").write_text(json.dumps({"simulations": [sim(30)]}))
    (save / "hallucination_discarded" / "a.json").write_text(
        json.dumps({"simulations": [sim(20)]})
    )
    original = original_tokens([save], "regular")
    assert original["agent_input"] == 50  # the rerun set aside counts too
    assert original["caller_input"] == 14
