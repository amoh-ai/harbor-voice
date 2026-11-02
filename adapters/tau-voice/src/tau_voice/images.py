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

The images published for the harbor-voice hub are recorded by digest in
``adapters/tau-voice/images.json`` (:data:`RECORD`, written by ``scripts/hub.py
images``). Tasks converted with a registry recorded there name each image by
that digest (:func:`pinned`), so a published dataset pulls the same bytes for
good, even if a tag is pushed again.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

TAU2_COMMIT = "2174a60"
SIDECAR = Path(__file__).resolve().parents[2] / "sidecar"
SIDECAR_FILES = ("Dockerfile", "server.py", "evaluate.py")
RECORD = Path(__file__).resolve().parents[2] / "images.json"
"""Published images, ``{registry: {"name:tag": "sha256:..."}}``."""
PLATFORM = "linux/amd64"
"""What every backend runs; an ARM machine would otherwise build ARM images."""
SOURCE = "https://github.com/amoh-ai/harbor-voice"
LABELS = {
    "org.opencontainers.image.source": SOURCE,
    "org.opencontainers.image.licenses": "Apache-2.0 AND MIT",
}
"""OCI labels on both images; ``source`` links a registry package to the repo.
Labels are build arguments, not Dockerfile lines, so they leave tags alone."""
DESCRIPTIONS = {
    "tau-voice-runtime": f"tau-voice's sidecar: tau2-bench at {TAU2_COMMIT} "
    "(MIT License, Copyright (c) 2025 Sierra Research) with harbor-voice's "
    "server and grader (Apache-2.0)",
    "tau-voice-main": "tau-voice's main container: python-slim and uv; "
    "harbor-voice is installed at trial start",
}

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


def refs(registry: str | None = None) -> list[str]:
    """The images :func:`build` makes, by tag."""
    return [tau2_image(registry), main_image(registry)]


class UnpublishedImage(LookupError):
    """A registry with published images lacks the one a task would name."""


def published(path: Path | None = None) -> dict[str, dict[str, str]]:
    """``{registry: {"name:tag": digest}}``, empty before anything is published."""
    path = RECORD if path is None else path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pinned(image: str, registry: str | None) -> str:
    """``image`` as a task names it: by digest if ``registry`` is recorded.

    A registry in :data:`RECORD` must hold every image its tasks name, so a
    checkout whose sidecar files changed since the images were published
    fails here instead of naming a tag nobody pushed. Any other registry
    (your own) names images by tag, as before.
    """
    if registry is None:
        return image
    prefix = registry.rstrip("/")
    record = published().get(prefix)
    if record is None:
        return image
    digest = record.get(image.removeprefix(prefix + "/"))
    if digest is None:
        raise UnpublishedImage(
            f"{image} is not in {RECORD.name}, which records what was published "
            f"to {prefix}: publish it (scripts/hub.py images), or convert from "
            "the commit that recorded the images you mean"
        )
    return f"{image}@{digest}"


def pinned_tau2(registry: str | None = None) -> str:
    """The sidecar as a task names it (:func:`pinned`)."""
    return pinned(tau2_image(registry), registry)


def pinned_main(registry: str | None = None) -> str:
    """``main`` as a task names it (:func:`pinned`)."""
    return pinned(main_image(registry), registry)


def from_image(image: str) -> str:
    """``main``'s Dockerfile for a task that names a published ``main``.

    Harbor builds it only when a run forces builds; then it is that same
    image, pinned, rather than a fresh build of python-slim.
    """
    return (
        f"# main: the published image this task names (docker_image).\nFROM {image}\n"
    )


def _labels(name: str) -> list[str]:
    labels = {**LABELS, "org.opencontainers.image.description": DESCRIPTIONS[name]}
    return [
        arg for key, value in labels.items() for arg in ("--label", f"{key}={value}")
    ]


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
    tau2, main = refs(registry)
    platform = ["--platform", PLATFORM]
    if force or not exists(tau2):
        subprocess.run(
            ["docker", "build", *platform, *_labels("tau-voice-runtime"), "-t", tau2]
            + [str(SIDECAR)],
            check=True,
        )
    if force or not exists(main):
        # No build context: the Dockerfile copies nothing from disk.
        subprocess.run(
            ["docker", "build", *platform, *_labels("tau-voice-main"), "-t", main, "-"],
            input=MAIN_DOCKERFILE.encode(),
            check=True,
        )
    if push:
        for image in (tau2, main):
            subprocess.run(["docker", "push", image], check=True)
    return [tau2, main]
