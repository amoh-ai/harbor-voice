"""Where harbor finds harbor-voice's published datasets: the checkout's
``registry.json``, or the repository's."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harbor.models.job.config import JobConfig

from harbor_voice import datasets

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]


def test_a_checkout_reads_its_own_registry() -> None:
    assert datasets.checkout_registry() == ROOT / "registry.json"
    assert datasets.registry_args() == ["--registry-path", str(ROOT / "registry.json")]


def test_an_install_reads_the_repositorys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets, "checkout_registry", lambda: None)
    assert datasets.registry_args() == ["--repo", datasets.REPO]
    entry = datasets.hub_dataset("tau-voice-control@0.1.0")
    assert entry == {
        "name": "tau-voice-control",
        "repo": datasets.REPO,
        "version": "0.1.0",
    }
    (dataset,) = JobConfig.model_validate({"datasets": [entry]}).datasets
    assert (dataset.name, dataset.version, dataset.repo) == (
        "tau-voice-control",
        "0.1.0",
        datasets.REPO,
    )
    assert "version" not in datasets.hub_dataset("eva")


def test_a_registry_file_checks_the_name_and_version(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            [{"name": "eva", "version": "0.1.0", "description": "", "tasks": []}]
        ),
        encoding="utf-8",
    )
    entry = datasets.hub_dataset("eva@0.1.0", registry)
    assert entry == {"name": "eva", "registry_path": str(registry), "version": "0.1.0"}
    with pytest.raises(ValueError, match="no version '9'"):
        datasets.hub_dataset("eva@9", registry)
    with pytest.raises(ValueError, match="not in registry.json"):
        datasets.hub_dataset("tau-voice-control", registry)
    with pytest.raises(ValueError, match="no registry.json"):
        datasets.hub_dataset("eva", tmp_path / "missing" / "registry.json")
    with pytest.raises(ValueError, match="name a dataset"):
        datasets.hub_dataset("@0.1.0", registry)
