"""scripts/hub.py: images recorded by digest, registry entries pinned to a
released commit, and the checks that keep published entries fixed."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from harbor.models.job.config import JobConfig

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = "ghcr.io/amoh-ai"
DIGEST = "sha256:" + "a" * 64


def _load_hub() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hub", ROOT / "scripts" / "hub.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["hub"] = module
    spec.loader.exec_module(module)
    return module


hub = _load_hub()

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
        ]
        + list(args),
        capture_output=True,
        text=True,
        check=True,
        env=GIT_ENV,
    ).stdout.strip()


@pytest.fixture
def records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Each adapter's digest record in tmp_path, holding its images as published."""
    from eva_adapter import images as eva
    from tau_voice import images as tau

    digests = {}
    for name, module in (("tau-voice", tau), ("eva", eva)):
        path = tmp_path / f"{name}-images.json"
        monkeypatch.setattr(module, "RECORD", path)
        pinned = {
            ref.removeprefix(REGISTRY + "/"): "sha256:" + f"{n:x}" * 64
            for n, ref in enumerate(module.refs(REGISTRY), start=len(digests) + 1)
        }
        path.write_text(json.dumps({REGISTRY: pinned}), encoding="utf-8")
        digests.update({f"{REGISTRY}/{key}": d for key, d in pinned.items()})
    return digests


@pytest.fixture
def datasets(tmp_path: Path, records: dict[str, str]) -> tuple[Path, Path]:
    """A datasets repo with converted tasks, its 'remote' (a bare repo), and
    one commit pushed to main and tagged v0.1.0."""
    from eva_adapter.main import write_task as write_eva
    from tau_voice.main import write_task as write_tau

    remote, repo = tmp_path / "remote.git", tmp_path / "datasets-repo"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(remote)], check=True,
                   capture_output=True, env=GIT_ENV)  # fmt: skip
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True,
                   capture_output=True, env=GIT_ENV)  # fmt: skip
    for task_id in ("7", "8"):
        write_tau(repo / "datasets" / "tau-voice-control", "airline", task_id,
                  complexity="control", seed=1, registry=REGISTRY)  # fmt: skip
    write_eva(repo / "datasets" / "eva", "airline", "1.1.2", registry=REGISTRY)
    (repo / "README.md").write_text("datasets\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "v0.1.0")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "tag", "v0.1.0")
    _git(repo, "push", "-q", "origin", "main", "v0.1.0")
    return repo, remote


def _register(repo: Path, remote: Path, registry: Path, *datasets: tuple[str, str]):
    return hub.register(
        repo, "v0.1.0", "0.1.0", list(datasets), registry=registry, git_url=str(remote)
    )


def test_a_release_registers_every_task_at_its_commit(
    datasets, tmp_path, monkeypatch
) -> None:
    repo, remote = datasets
    registry = tmp_path / "registry.json"
    registry.write_text("[]\n", encoding="utf-8")

    _register(repo, remote, registry, ("tau-voice-control", "tau-voice, control"),
              ("eva", "EVA, clean"))  # fmt: skip

    entries = json.loads(registry.read_text(encoding="utf-8"))
    sha = _git(repo, "rev-parse", "HEAD")
    tau = entries[0]
    assert (tau["name"], tau["version"]) == ("tau-voice-control", "0.1.0")
    assert len(tau["tasks"]) == 2
    assert {task["git_commit_id"] for task in tau["tasks"]} == {sha}
    assert (
        tau["tasks"][0]["path"]
        == f"datasets/tau-voice-control/{tau['tasks'][0]['name']}"
    )
    assert hub.check(registry=registry, repo=repo, git_url=str(remote)) == []

    # job-config names it as harbor reads a registry dataset.
    from tau_voice import job

    monkeypatch.setattr(job, "REGISTRY", registry)
    raw = job.job_config(hub="tau-voice-control@0.1.0", model="m")
    assert raw["datasets"] == [
        {
            "name": "tau-voice-control",
            "registry_path": str(registry),
            "version": "0.1.0",
        }
    ]
    assert JobConfig.model_validate(raw).datasets[0].is_registry()


def test_a_published_version_is_never_registered_again(datasets, tmp_path) -> None:
    repo, remote = datasets
    registry = tmp_path / "registry.json"
    registry.write_text("[]\n", encoding="utf-8")
    _register(repo, remote, registry, ("eva", "EVA"))
    with pytest.raises(hub.HubError, match="already published"):
        _register(repo, remote, registry, ("eva", "EVA"))


def test_only_a_commit_on_main_and_tagged_is_pinned(datasets, tmp_path) -> None:
    repo, remote = datasets
    registry = tmp_path / "registry.json"
    registry.write_text("[]\n", encoding="utf-8")
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "unpushed")
    _git(repo, "tag", "v0.2.0")  # tagged here, but neither pushed

    with pytest.raises(hub.HubError) as raised:
        hub.register(repo, "v0.2.0", "0.2.0", [("eva", "EVA")], registry=registry,
                     git_url=str(remote))  # fmt: skip
    assert "is not on" in str(raised.value)
    assert "no tag on" in str(raised.value)
    assert json.loads(registry.read_text(encoding="utf-8")) == []


def test_stray_files_and_unpinned_images_are_refused(datasets, tmp_path) -> None:
    repo, remote = datasets
    registry = tmp_path / "registry.json"
    registry.write_text("[]\n", encoding="utf-8")
    from eva_adapter import images

    eva = repo / "datasets" / "eva"
    # eva-adapter convert writes a job.yaml, with this machine's paths in it.
    (eva / "job.yaml").write_text("path: C:/Users/someone\n", encoding="utf-8")
    compose = next(eva.glob("*/environment/docker-compose.yaml"))
    text = compose.read_text(encoding="utf-8")
    pinned, by_tag = images.pinned_sidecar(REGISTRY), images.sidecar_image(REGISTRY)
    assert pinned in text
    compose.write_text(text.replace(pinned, by_tag), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "v0.1.1")
    _git(repo, "tag", "v0.1.1")
    _git(repo, "push", "-q", "origin", "main", "v0.1.1")

    with pytest.raises(hub.HubError) as raised:
        hub.register(repo, "v0.1.1", "0.1.1", [("eva", "EVA")], registry=registry,
                     git_url=str(remote))  # fmt: skip
    assert "datasets/eva/job.yaml is not part of a task folder" in str(raised.value)
    assert "is not pinned by digest" in str(raised.value)


def test_published_entries_only_ever_grow(tmp_path) -> None:
    entry = {
        "name": "eva",
        "version": "0.1.0",
        "description": "EVA",
        "tasks": [
            {
                "name": "t",
                "git_url": hub.DATASETS_GIT_URL,
                "git_commit_id": "1" * 40,
                "path": "datasets/eva/t",
            }
        ],
    }
    later = {**entry, "version": "0.2.0"}
    assert hub.registry_problems([entry, later]) == []
    assert hub.append_only_problems([entry], [entry, later]) == []
    assert hub.append_only_problems([entry], [later]) == ["eva@0.1.0 was removed"]
    edited = {**entry, "description": "EVA, fixed"}
    assert hub.append_only_problems([entry], [edited]) == [
        "eva@0.1.0 was changed; publish a new version"
    ]


def test_the_registry_rules(tmp_path) -> None:
    task = {"name": "t", "git_url": hub.DATASETS_GIT_URL, "path": "datasets/eva/t"}
    short = {"name": "eva", "version": "1", "description": "", "tasks": [
        {**task, "git_commit_id": "abc1234"}]}  # fmt: skip
    problems = hub.registry_problems([short, short])
    assert "eva@1 is listed twice" in problems
    assert any("not a full commit SHA" in problem for problem in problems)
    metric = {**short, "version": "2", "metrics": [{"type": "mean"}]}
    metric["tasks"] = [{**task, "git_commit_id": "1" * 40}]
    assert hub.registry_problems([metric]) == [
        "eva@2 names metrics; they belong in job-config"
    ]
    assert hub.registry_problems([{"name": "eva"}])[0].startswith(
        "entry 0: not a harbor"
    )
    with pytest.raises(hub.HubError, match="must be a list"):
        path = tmp_path / "registry.json"
        path.write_text('{"datasets": []}', encoding="utf-8")
        hub.load_registry(path)


def test_the_repos_registry_and_records_pass() -> None:
    assert hub.check() == []


class FakeRegistry:
    """What ``docker`` would do: a registry of tag -> digest, local builds."""

    def __init__(self, served: dict[str, str]) -> None:
        self.served, self.pushed, self.built = dict(served), [], 0

    def digest(self, ref: str) -> str | None:
        return self.served.get(ref)

    def push(self, ref: str) -> None:
        self.pushed.append(ref)
        self.served[ref] = "sha256:" + f"{len(self.pushed):x}" * 64


def _fake_docker(
    monkeypatch, images: ModuleType, served: dict[str, str]
) -> FakeRegistry:
    fake = FakeRegistry(served)
    monkeypatch.setattr(hub, "remote_digest", fake.digest)
    monkeypatch.setattr(hub, "push", fake.push)
    monkeypatch.setattr(hub, "local_labels", lambda ref: dict(images.LABELS))

    def build(**kwargs) -> list[str]:
        fake.built += 1
        return images.refs(kwargs.get("registry"))

    monkeypatch.setattr(images, "build", build)
    return fake


def test_images_push_only_what_the_registry_lacks(tmp_path, monkeypatch) -> None:
    from tau_voice import images

    monkeypatch.setattr(images, "RECORD", tmp_path / "images.json")
    runtime, main = images.refs(REGISTRY)
    fake = _fake_docker(monkeypatch, images, {runtime: DIGEST})

    assert hub.publish_images(images, REGISTRY, dry_run=True) == [main]
    assert not (tmp_path / "images.json").exists() and fake.built == 0

    assert hub.publish_images(images, REGISTRY) == [main]
    assert fake.pushed == [main]
    record = images.published()[REGISTRY]
    assert record[runtime.removeprefix(REGISTRY + "/")] == DIGEST
    assert record[main.removeprefix(REGISTRY + "/")] == fake.served[main]
    # Tasks now name both by digest.
    assert images.pinned_tau2(REGISTRY) == f"{runtime}@{DIGEST}"

    # Everything published: nothing is built or pushed again.
    assert hub.publish_images(images, REGISTRY) == []
    assert fake.built == 1 and fake.pushed == [main]


def test_a_tag_pushed_again_is_an_error(tmp_path, monkeypatch) -> None:
    from eva_adapter import images

    monkeypatch.setattr(images, "RECORD", tmp_path / "images.json")
    refs = images.refs(REGISTRY)
    record = {ref.removeprefix(REGISTRY + "/"): DIGEST for ref in refs}
    hub.write_record(images.RECORD, {REGISTRY: record})
    _fake_docker(monkeypatch, images, {ref: "sha256:" + "b" * 64 for ref in refs})
    with pytest.raises(hub.HubError, match="was pushed again"):
        hub.publish_images(images, REGISTRY)


def test_an_image_built_without_labels_is_not_pushed(tmp_path, monkeypatch) -> None:
    from tau_voice import images

    monkeypatch.setattr(images, "RECORD", tmp_path / "images.json")
    fake = _fake_docker(monkeypatch, images, {})
    monkeypatch.setattr(hub, "local_labels", lambda ref: {})
    with pytest.raises(hub.HubError, match="--force"):
        hub.publish_images(images, REGISTRY)
    assert fake.pushed == []
