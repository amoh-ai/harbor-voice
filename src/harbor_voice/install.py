"""Installing the call runner into a task's environment — harbor's agent install.

Harbor keeps the agent out of the task: a task folder holds the benchmark,
and at trial start the agent is installed into the running ``main``
container at a pinned version (``agents/installed/base.py``: ``setup`` →
``install``). harbor-voice does the same with what runs a voice call —
harbor-voice itself and the benchmark adapter whose driver the task names.

``main`` gets exactly the code this orchestrator runs, at its version:

- installed from a source folder (a checkout): a wheel built from that
  source, uploaded;
- installed from a wheel file: that wheel, uploaded;
- installed from a package index: the same version, from the index.

A driver whose adapter isn't installed here at all is installed from the
index at harbor-voice's own version, since harbor-voice and its adapters are
released together (:data:`DRIVER_DISTRIBUTIONS`).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from urllib.parse import urlparse

DRIVER_DISTRIBUTIONS = {
    "tau_voice": "harbor-voice-tau-voice-adapter",
    "eva_adapter": "harbor-voice-eva-adapter",
}
"""The distribution of each benchmark driver's top-level module."""

INDEX_URL_ENV = "HARBOR_VOICE_INDEX_URL"
"""Another package index for harbor-voice's own packages (TestPyPI, a mirror).

PyPI is still tried first, so a package it has is never taken from here."""

PYPI = "https://pypi.org/simple"


class InstallError(RuntimeError):
    """The call runner could not be packaged for the environment."""


@dataclass(frozen=True, slots=True)
class Wheel:
    """A wheel file, uploaded into the environment and installed from there."""

    distribution: str
    version: str
    path: Path


@dataclass(frozen=True, slots=True)
class Pinned:
    """A distribution installed in the environment from the package index."""

    distribution: str
    version: str


Package = Wheel | Pinned


def distribution_of(module: str) -> str:
    """The installed distribution that provides top-level ``module``.

    The metadata index misses editable installs (their files are found via a
    ``.pth`` path, not a ``RECORD``), so fall back to locating the module and
    finding the local install whose source tree contains it.
    """
    top = module.split(".")[0]
    names = importlib.metadata.packages_distributions().get(top)
    if names:
        return names[0]
    spec = importlib.util.find_spec(top)
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    if origin is not None:
        # The deepest containing source wins: an adapter checked out inside
        # the repo of another package belongs to the adapter, not the repo.
        best: tuple[int, str] | None = None
        for dist in importlib.metadata.distributions():
            source = _local_source(dist)
            if source is not None and source.resolve() in origin.parents:
                depth = len(source.resolve().parts)
                if best is None or depth > best[0]:
                    best = (depth, dist.metadata["Name"])
        if best is not None:
            return best[1]
    raise InstallError(f"no installed distribution provides {top!r}")


def _direct_url(dist: importlib.metadata.Distribution) -> dict[str, Any] | None:
    raw = dist.read_text("direct_url.json")
    return json.loads(raw) if raw else None


def _file_path(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    if len(path) > 2 and path[0] == "/" and path[2] == ":":  # file:///C:/...
        path = path[1:]
    return Path(path)


def _local_source(dist: importlib.metadata.Distribution) -> Path | None:
    """The source folder ``dist`` was installed from, if it was one."""
    direct = _direct_url(dist)
    if direct is None or "dir_info" not in direct:
        return None
    return _file_path(direct.get("url", ""))


def _local_wheel(dist: importlib.metadata.Distribution) -> Path | None:
    """The wheel file ``dist`` was installed from, if it still exists."""
    direct = _direct_url(dist)
    if direct is None or "archive_info" not in direct:
        return None
    path = _file_path(direct.get("url", ""))
    if path is not None and path.suffix == ".whl" and path.is_file():
        return path
    return None


def _source_dir(distribution: str) -> Path:
    source = _local_source(importlib.metadata.distribution(distribution))
    if source is None:
        raise InstallError(f"{distribution} is not installed from a source folder")
    return source


def build_wheels(modules: list[str], out_dir: Path) -> list[Wheel]:
    """Wheels for the distributions providing ``modules``, built from source."""
    uv = shutil.which("uv")
    if uv is None:
        raise InstallError("uv is needed to build the call runner's wheels")
    out_dir.mkdir(parents=True, exist_ok=True)
    wheels: list[Wheel] = []
    seen: set[str] = set()
    for module in modules:
        distribution = distribution_of(module)
        if distribution in seen:
            continue
        seen.add(distribution)
        source = _source_dir(distribution)
        target = out_dir / distribution
        if target.exists():
            shutil.rmtree(target)
        result = subprocess.run(
            [uv, "build", "--wheel", "--out-dir", str(target), str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        built = sorted(target.glob("*.whl"))
        if result.returncode != 0 or not built:
            raise InstallError(
                f"building {distribution} failed: {result.stderr[-600:]}"
            )
        wheels.append(
            Wheel(
                distribution=distribution,
                version=importlib.metadata.version(distribution),
                path=built[-1],
            )
        )
    return wheels


def package_for(module: str, out_dir: Path) -> Package:
    """What installs ``module``'s distribution into an environment.

    A source install is built into a wheel under ``out_dir``; a wheel-file
    install reuses that wheel; an index install is pinned to its version. A
    known driver module with nothing installed here is pinned to
    harbor-voice's version.
    """
    try:
        distribution = distribution_of(module)
    except InstallError:
        known = DRIVER_DISTRIBUTIONS.get(module.split(".")[0])
        if known is None:
            raise
        return Pinned(
            known, importlib.metadata.version(distribution_of("harbor_voice"))
        )
    dist = importlib.metadata.distribution(distribution)
    if _local_source(dist) is not None:
        (wheel,) = build_wheels([module], out_dir)
        return wheel
    version = importlib.metadata.version(distribution)
    path = _local_wheel(dist)
    if path is not None:
        return Wheel(distribution, version, path)
    return Pinned(distribution, version)


WHEELS_IN_ENV = "/installed-agent/wheels"


def index_args() -> list[str]:
    """``uv pip install`` index flags: PyPI, plus :data:`INDEX_URL_ENV`'s index.

    uv tries ``--extra-index-url`` before ``--index-url``, so PyPI keeps
    priority for every package it has.
    """
    extra = os.environ.get(INDEX_URL_ENV, "").strip()
    if not extra:
        return []
    return ["--index-url", extra, "--extra-index-url", PYPI]


async def install_into(
    environment: Any, packages: Sequence[Package], *, extras: Sequence[str] = ()
) -> None:
    """Install ``packages`` into ``main`` with ``uv pip install``, as root.

    Wheels are uploaded first; pinned packages come from the index.
    harbor-voice's own package brings ``extras``, the ones the agent under
    test needs (``live`` for OpenAI Live), when its distribution declares them.
    """
    voice = distribution_of("harbor_voice")
    declared = importlib.metadata.metadata(voice).get_all("Provides-Extra") or []
    wanted = [e for e in dict.fromkeys(extras) if e in declared]
    targets: list[str] = []
    pinned = uploading = False
    for package in packages:
        extra = (
            f"[{','.join(wanted)}]" if package.distribution == voice and wanted else ""
        )
        if isinstance(package, Wheel):
            if not uploading:
                await environment.exec(f"mkdir -p {WHEELS_IN_ENV}", user="root")
                uploading = True
            target = f"{WHEELS_IN_ENV}/{package.path.name}"
            await environment.upload_file(package.path, target)
            targets.append(shlex.quote(target + extra))
        else:
            pinned = True
            targets.append(
                shlex.quote(f"{package.distribution}{extra}=={package.version}")
            )
    flags = [shlex.quote(arg) for arg in index_args()] if pinned else []
    result = await environment.exec(
        " ".join(["uv pip install --system --quiet", *flags, *targets]),
        user="root",
        timeout_sec=1800,
    )
    if result.return_code != 0:
        raise InstallError(
            f"installing the call runner failed: {(result.stderr or '')[-800:]}"
        )
