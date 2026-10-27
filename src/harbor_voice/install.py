"""Installing the call runner into a task's environment — harbor's agent install.

Harbor keeps the agent out of the task: a task folder holds the benchmark,
and at trial start the agent is installed into the running ``main``
container at a pinned version (``agents/installed/base.py``: ``setup`` →
``install``). harbor-voice does the same with what runs a voice call —
harbor-voice itself and the benchmark adapter whose driver the task names.

Both are installed from wheels built from the code this orchestrator is
running, so the environment runs exactly that code; the wheel versions go into
``trial.json``. Only local (editable or path) installs can be packaged this
way for now — enough for running from a checkout, which is how harbor-voice
is run today.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urlparse


class InstallError(RuntimeError):
    """The call runner could not be packaged for the environment."""


@dataclass(frozen=True, slots=True)
class Wheel:
    distribution: str
    version: str
    path: Path


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


def _local_source(dist: importlib.metadata.Distribution) -> Path | None:
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    parsed = urlparse(json.loads(raw).get("url", ""))
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    if len(path) > 2 and path[0] == "/" and path[2] == ":":  # file:///C:/...
        path = path[1:]
    return Path(path)


def _source_dir(distribution: str) -> Path:
    source = _local_source(importlib.metadata.distribution(distribution))
    if source is None:
        raise InstallError(
            f"{distribution} is not installed from a local path; packaging a "
            f"registry install for the environment is not supported yet"
        )
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
