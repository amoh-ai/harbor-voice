"""scripts/check_dist.py: the release refuses anything but the packages' own
code, metadata and licenses, at one version."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_dist", ROOT / "scripts" / "check_dist.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_dist"] = module
    spec.loader.exec_module(module)
    return module


check_dist = _load()


def _metadata(name: str, version: str, adapters: str) -> bytes:
    lines = ["Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}"]
    if name == "harbor-voice":
        lines += [
            f"Requires-Dist: harbor-voice-tau-voice-adapter=={adapters} ; extra == 'tau-voice'",
            f"Requires-Dist: harbor-voice-eva-adapter=={adapters} ; extra == 'eva'",
        ]
    return ("\n".join(lines) + "\n\n").encode()


def _wheel(
    dist: Path, name: str, version: str, files: dict[str, bytes], adapters: str
) -> None:
    stem = check_dist.normalized(name)
    module = check_dist.PACKAGES[name]
    with zipfile.ZipFile(dist / f"{stem}-{version}-py3-none-any.whl", "w") as wheel:
        wheel.writestr(f"{module}/__init__.py", b'"""A package."""\n')
        wheel.writestr(
            f"{stem}-{version}.dist-info/METADATA", _metadata(name, version, adapters)
        )
        for path, data in files.items():
            wheel.writestr(path, data)


def _sdist(dist: Path, name: str, version: str, files: dict[str, bytes]) -> None:
    stem = check_dist.normalized(name)
    module = check_dist.PACKAGES[name]
    contents = {
        "PKG-INFO": b"Metadata-Version: 2.4\n",
        "pyproject.toml": b"[project]\n",
        "README.md": b"# readme\n",
        "LICENSE": b"Apache License\n",
        f"src/{module}/__init__.py": b'"""A package."""\n',
        **files,
    }
    with tarfile.open(dist / f"{stem}-{version}.tar.gz", "w:gz") as sdist:
        for path, data in contents.items():
            info = tarfile.TarInfo(f"{stem}-{version}/{path}")
            info.size = len(data)
            sdist.addfile(info, io.BytesIO(data))


def _release(
    tmp_path: Path,
    *,
    versions: dict[str, str] | None = None,
    adapters: str = "0.1.0",
    wheel_files: dict[str, dict[str, bytes]] | None = None,
    sdist_files: dict[str, dict[str, bytes]] | None = None,
) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in check_dist.PACKAGES:
        version = (versions or {}).get(name, "0.1.0")
        _wheel(dist, name, version, (wheel_files or {}).get(name, {}), adapters)
        _sdist(dist, name, version, (sdist_files or {}).get(name, {}))
    return dist


def _gitignore(tmp_path: Path) -> Path:
    path = tmp_path / ".gitignore"
    path.write_bytes(b".env\n")
    return path


def test_a_clean_release_passes(tmp_path: Path) -> None:
    dist = _release(tmp_path)
    assert check_dist.check(dist, tag="v0.1.0", gitignore=_gitignore(tmp_path)) == []


@pytest.mark.parametrize(
    ("leak", "label"),
    [
        ("sk-" + "proj-" + "x" * 24, "an OpenAI key"),
        ("AKIA" + "Q" * 16, "an AWS access key id"),
        ("ghp_" + "a" * 36, "a GitHub token"),
        ("C:" + "\\Users\\someone\\repo", "a local user path"),
        ("/home/" + "someone/repo", "a local user path"),
        ("someone" + "@company.io", "an email address"),
        ("see docs/" + "internal/notes.md", "a reference to internal notes"),
    ],
)
def test_a_leak_in_any_file_fails(tmp_path: Path, leak: str, label: str) -> None:
    module = "tau_voice/config.py"
    dist = _release(
        tmp_path,
        wheel_files={
            "harbor-voice-tau-voice-adapter": {module: f"X = {leak!r}\n".encode()}
        },
    )
    problems = check_dist.check(dist, gitignore=_gitignore(tmp_path))
    assert any(label in p and module in p for p in problems), problems


def test_example_addresses_are_not_leaks(tmp_path: Path) -> None:
    module = "harbor_voice/doc.py"
    dist = _release(
        tmp_path, wheel_files={"harbor-voice": {module: b"t@example.com\n"}}
    )
    assert check_dist.check(dist, gitignore=_gitignore(tmp_path)) == []


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_x.py",
        "docs/notes.md",
        "src/tau_voice/__pycache__/x.pyc",
        ".env",
        "notes.txt",
    ],
)
def test_a_file_outside_the_package_fails(tmp_path: Path, path: str) -> None:
    dist = _release(
        tmp_path, sdist_files={"harbor-voice-tau-voice-adapter": {path: b"x\n"}}
    )
    problems = check_dist.check(dist, gitignore=_gitignore(tmp_path))
    assert any(f"{path} does not belong" in p for p in problems), problems


def test_another_module_in_a_wheel_fails(tmp_path: Path) -> None:
    dist = _release(
        tmp_path, wheel_files={"harbor-voice": {"tau_voice/__init__.py": b"\n"}}
    )
    problems = check_dist.check(dist, gitignore=_gitignore(tmp_path))
    assert any("tau_voice/__init__.py does not belong" in p for p in problems)


def test_an_sdists_gitignore_must_be_the_repos(tmp_path: Path) -> None:
    gitignore = _gitignore(tmp_path)
    for name, content, ok in (("same", b".env\n", True), ("other", b"*\n", False)):
        (tmp_path / name).mkdir()
        dist = _release(
            tmp_path / name,
            sdist_files={"harbor-voice-eva-adapter": {".gitignore": content}},
        )
        problems = check_dist.check(dist, gitignore=gitignore)
        assert (problems == []) is ok
        assert ok or any(".gitignore is not the repo's" in p for p in problems)


def test_versions_must_agree_and_match_the_tag(tmp_path: Path) -> None:
    gitignore = _gitignore(tmp_path)
    (tmp_path / "a").mkdir()
    mixed = _release(tmp_path / "a", versions={"harbor-voice-eva-adapter": "0.2.0"})
    assert any(
        "versions differ" in p for p in check_dist.check(mixed, gitignore=gitignore)
    )
    (tmp_path / "b").mkdir()
    release = _release(tmp_path / "b")
    problems = check_dist.check(release, tag="v0.2.0", gitignore=gitignore)
    assert any("tag v0.2.0" in p for p in problems)


def test_the_extras_pin_the_adapters_at_the_release_version(tmp_path: Path) -> None:
    dist = _release(tmp_path, adapters="0.0.9")
    problems = check_dist.check(dist, gitignore=_gitignore(tmp_path))
    assert any("extra 'tau-voice'" in p for p in problems)
    assert any("extra 'eva'" in p for p in problems)


def test_a_missing_archive_fails(tmp_path: Path) -> None:
    dist = _release(tmp_path)
    next(dist.glob("harbor_voice_eva_adapter-*.tar.gz")).unlink()
    problems = check_dist.check(dist, gitignore=_gitignore(tmp_path))
    assert problems == ["no sdist for harbor-voice-eva-adapter"]


def test_the_command_reports_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dist = _release(tmp_path, versions={"harbor-voice": "0.2.0"})
    assert check_dist.main([str(dist)]) == 1
    assert "versions differ" in capsys.readouterr().err
