"""The two images a tau-voice task runs on, for building once ahead of trials.

Tasks build their images by default, as harbor tasks do. ``tau-voice
build-image`` builds both into the local Docker image store for tasks
converted with ``--prebuilt``, which name them by ``image:`` only, so a trial
makes no registry or PyPI call before its call starts. Tags carry a digest of
what goes into each image, so
a change to the sidecar or to ``main`` gives a new tag and the converter's
tasks point at it — a stale image is never picked up silently.

Remote backends (Daytona, EC2, ...) pull from a registry instead:
``--registry <prefix> --push`` names the images under that prefix
(``ghcr.io/<org>``, ``docker.io/<user>``, ...) and pushes them, and tasks
converted with the same ``--registry`` name them there. Neither image holds a
secret (tau2-bench is public; keys arrive as environment variables at run
time), so a public registry works without credentials on the pulling side.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

TAU2_COMMIT = "2174a60"
SIDECAR = Path(__file__).resolve().parents[2] / "sidecar"
SIDECAR_FILES = ("Dockerfile", "server.py", "evaluate.py")

MAIN_DOCKERFILE = """\
# main: harbor-voice and the adapter are installed here at trial start, from
# wheels of the orchestrator's code.
FROM mirror.gcr.io/library/python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 UV_LINK_MODE=copy
WORKDIR /task
"""


def _digest(parts: list[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:10]


def _named(name: str, tag: str, registry: str | None) -> str:
    return f"{registry.rstrip('/')}/{name}:{tag}" if registry else f"{name}:{tag}"


def tau2_image(registry: str | None = None) -> str:
    """The sidecar: tau2 at a pinned commit, plus our server and grader."""
    files = [(SIDECAR / name).read_bytes() for name in SIDECAR_FILES]
    return _named("tau-voice-runtime", f"{TAU2_COMMIT}-{_digest(files)}", registry)


def main_image(registry: str | None = None) -> str:
    return _named("tau-voice-main", _digest([MAIN_DOCKERFILE.encode()]), registry)


def exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, check=False
    )
    return result.returncode == 0


def build(
    *, force: bool = False, registry: str | None = None, push: bool = False
) -> list[str]:
    """Build whichever image is missing (both with ``force``); returns the tags.

    With ``push``, both are pushed to ``registry`` (log in to it first).
    """
    if push and not registry:
        raise ValueError("push needs a registry")
    tau2, main = tau2_image(registry), main_image(registry)
    if force or not exists(tau2):
        subprocess.run(["docker", "build", "-t", tau2, str(SIDECAR)], check=True)
    if force or not exists(main):
        # No build context: the Dockerfile copies nothing from disk.
        subprocess.run(
            ["docker", "build", "-t", main, "-"],
            input=MAIN_DOCKERFILE.encode(),
            check=True,
        )
    if push:
        for image in (tau2, main):
            subprocess.run(["docker", "push", image], check=True)
    return [tau2, main]
