"""The two images a tau-voice task runs on, for building once ahead of trials.

Tasks build their images by default, as harbor tasks do. ``tau-voice
build-image`` builds both into the local Docker image store for tasks
converted with ``--prebuilt``, which name them by ``image:`` only, so a trial
makes no registry or PyPI call before its call starts. Tags carry a digest of
what goes into each image, so
a change to the sidecar or to ``main`` gives a new tag and the converter's
tasks point at it — a stale image is never picked up silently.

Remote backends will need these pushed to a registry they can pull from;
that is a later step.
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
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 UV_LINK_MODE=copy
WORKDIR /task
"""


def _digest(parts: list[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:10]


def tau2_image() -> str:
    """The sidecar: tau2 at a pinned commit, plus our server and grader."""
    files = [(SIDECAR / name).read_bytes() for name in SIDECAR_FILES]
    return f"tau-voice-runtime:{TAU2_COMMIT}-{_digest(files)}"


def main_image() -> str:
    return f"tau-voice-main:{_digest([MAIN_DOCKERFILE.encode()])}"


def exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, check=False
    )
    return result.returncode == 0


def build(*, force: bool = False) -> list[str]:
    """Build whichever image is missing (both with ``force``); returns the tags."""
    tau2, main = tau2_image(), main_image()
    if force or not exists(tau2):
        subprocess.run(["docker", "build", "-t", tau2, str(SIDECAR)], check=True)
    if force or not exists(main):
        # No build context: the Dockerfile copies nothing from disk.
        subprocess.run(
            ["docker", "build", "-t", main, "-"],
            input=MAIN_DOCKERFILE.encode(),
            check=True,
        )
    return [tau2, main]
