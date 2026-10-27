"""Parity statistics: tau2's Pass^k, run means, and errored trials left out."""

from __future__ import annotations

import pytest

from tau_voice.compare import Record
from tau_voice.compare import compare
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
