"""What installs harbor-voice and a driver into ``main``: a wheel of local
source, the wheel it was installed from, or the same version from the index."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from harbor_voice import install

pytestmark = pytest.mark.unit


class _Distribution:
    def __init__(self, direct_url: dict[str, object] | None) -> None:
        self._direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        if name == "direct_url.json" and self._direct_url is not None:
            return json.dumps(self._direct_url)
        return None


class _Exec:
    return_code = 0
    stderr = ""


class _Environment:
    def __init__(self) -> None:
        self.execs: list[str] = []
        self.uploads: list[str] = []

    async def exec(self, command: str, **kw: object) -> _Exec:
        self.execs.append(command)
        return _Exec()

    async def upload_file(self, source: Path, target: str) -> None:
        self.uploads.append(target)


def _installed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    direct_url: dict[str, object] | None,
    distribution: str = "harbor-voice",
    version: str = "0.1.0",
) -> None:
    monkeypatch.setattr(install, "distribution_of", lambda module: distribution)
    monkeypatch.setattr(
        install.importlib.metadata,
        "distribution",
        lambda name: _Distribution(direct_url),
    )
    monkeypatch.setattr(install.importlib.metadata, "version", lambda name: version)


def test_a_source_install_is_built_into_a_wheel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _installed(monkeypatch, direct_url={"url": "file:///src", "dir_info": {}})
    built = install.Wheel("harbor-voice", "0.1.0", tmp_path / "harbor_voice.whl")
    monkeypatch.setattr(install, "build_wheels", lambda modules, out: [built])
    assert install.package_for("harbor_voice", tmp_path) == built


def test_a_wheel_file_install_reuses_that_wheel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel = tmp_path / "harbor_voice-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"")
    url = wheel.as_uri()
    _installed(monkeypatch, direct_url={"url": url, "archive_info": {}})
    package = install.package_for("harbor_voice", tmp_path / "out")
    assert package == install.Wheel("harbor-voice", "0.1.0", wheel)


def test_an_index_install_is_pinned_to_its_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _installed(monkeypatch, direct_url=None, version="0.3.1")
    assert install.package_for("harbor_voice", tmp_path) == install.Pinned(
        "harbor-voice", "0.3.1"
    )


def test_a_gone_wheel_file_falls_back_to_the_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gone = (tmp_path / "gone.whl").as_uri()
    _installed(monkeypatch, direct_url={"url": gone, "archive_info": {}})
    assert isinstance(install.package_for("harbor_voice", tmp_path), install.Pinned)


def test_a_known_driver_not_installed_here_is_pinned_to_harbor_voices_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def distribution_of(module: str) -> str:
        if module == "harbor_voice":
            return "harbor-voice"
        raise install.InstallError(f"no installed distribution provides {module!r}")

    monkeypatch.setattr(install, "distribution_of", distribution_of)
    monkeypatch.setattr(install.importlib.metadata, "version", lambda name: "0.2.0")
    assert install.package_for("tau_voice.trial", tmp_path) == install.Pinned(
        "harbor-voice-tau-voice-adapter", "0.2.0"
    )
    assert install.package_for("eva_adapter", tmp_path) == install.Pinned(
        "harbor-voice-eva-adapter", "0.2.0"
    )
    with pytest.raises(install.InstallError, match="no installed distribution"):
        install.package_for("someone_elses_bench.trial", tmp_path)


def _extras(monkeypatch: pytest.MonkeyPatch, declared: list[str]) -> None:
    monkeypatch.setattr(install, "distribution_of", lambda module: "harbor-voice")
    metadata = types.SimpleNamespace(get_all=lambda key: declared)
    monkeypatch.setattr(install.importlib.metadata, "metadata", lambda dist: metadata)


async def test_a_pinned_package_installs_from_the_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _extras(monkeypatch, ["live", "tau-voice"])
    monkeypatch.delenv(install.INDEX_URL_ENV, raising=False)
    environment = _Environment()
    await install.install_into(
        environment,
        [install.Pinned("harbor-voice", "0.1.0")],
        extras=["live"],
    )
    (command,) = environment.execs
    assert command == "uv pip install --system --quiet 'harbor-voice[live]==0.1.0'"
    assert environment.uploads == []


async def test_another_index_comes_after_pypi(monkeypatch: pytest.MonkeyPatch) -> None:
    _extras(monkeypatch, [])
    monkeypatch.setenv(install.INDEX_URL_ENV, "https://test.pypi.org/simple/")
    environment = _Environment()
    await install.install_into(
        environment, [install.Pinned("harbor-voice-eva-adapter", "0.1.0")]
    )
    (command,) = environment.execs
    assert "--index-url https://test.pypi.org/simple/" in command
    assert f"--extra-index-url {install.PYPI}" in command
    assert command.endswith("harbor-voice-eva-adapter==0.1.0")


async def test_wheels_alone_need_no_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _extras(monkeypatch, [])
    monkeypatch.setenv(install.INDEX_URL_ENV, "https://test.pypi.org/simple/")
    environment = _Environment()
    wheel = install.Wheel("harbor-voice", "0.1.0", tmp_path / "harbor_voice.whl")
    await install.install_into(environment, [wheel])
    mkdir, command = environment.execs
    assert mkdir == f"mkdir -p {install.WHEELS_IN_ENV}"
    assert environment.uploads == [f"{install.WHEELS_IN_ENV}/harbor_voice.whl"]
    assert "--index-url" not in command


async def test_a_failed_install_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    _extras(monkeypatch, [])

    class _Failed(_Exec):
        return_code = 1
        stderr = "No solution found: harbor-voice==9.9.9 is not available"

    class _Broken(_Environment):
        async def exec(self, command: str, **kw: object) -> _Exec:
            return _Failed()

    with pytest.raises(install.InstallError, match="9.9.9 is not available"):
        await install.install_into(_Broken(), [install.Pinned("harbor-voice", "9.9.9")])
