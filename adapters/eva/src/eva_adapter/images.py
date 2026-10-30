"""The images an EVA task runs on: ``main`` and the ``eva`` sidecar.

The sidecar is EVA's own image at the pinned commit plus our server
(``adapters/eva/sidecar/``). Its Dockerfile takes the EVA image as
``EVA_BASE``, from either of two places, both built from ServiceNow's public
repo at :data:`EVA_COMMIT_FULL`:

- ``eva-base:039efa0a`` — **EVA's own Dockerfile**, run on EVA's repo at the
  commit. :func:`build` makes it with ``docker build <repo>#<commit>`` when
  it is missing (a dev checkout built with ``docker build -t
  eva-base:039efa0a eva`` is the same image). This is what ``build-image``
  builds the sidecar on, and what gets pushed.
- ``eva-from-source`` — a stage of the sidecar's Dockerfile that runs EVA's
  Dockerfile stage for stage on a checkout of the same commit. It needs
  nothing local and no registry, so a task built where it runs (Daytona,
  without ``--registry``) can always build it. Tasks converted without
  ``--prebuilt`` build this one.

**Which to use.** For runs of any size, ``eva-adapter build-image
--registry <prefix> --push`` once, and tasks converted with the same
``--registry``: every trial pulls the same bytes, under a tag that names the
EVA commit and a digest of our files. Building per trial makes each sandbox
fetch EVA, apt packages and EVA's ~200 locked wheels (minutes, and every
network failure a lost trial), and two builds of floating base images need
not be the same image. The from-source build is the fallback that makes a
task runnable anywhere, not the way to run 600 trials.

``main`` comes in two: clean (:data:`MAIN_DOCKERFILE`), and perturbed
(:data:`MAIN_PERTURBED_DOCKERFILE`) for tasks converted with one of EVA's
perturbation conditions: EVA's numpy and its noise WAVs, fetched from EVA's
repo at the commit and checked, because EVA's perturbator runs in ``main``.

Neither image holds a secret (EVA is public; keys arrive as environment
variables at run time), so a public registry needs no pull credentials.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from eva_adapter import EVA_COMMIT
from eva_adapter.conditions import NOISE_FILES

EVA_REPO = "https://github.com/ServiceNow/eva.git"
EVA_COMMIT_FULL = "039efa0a80916a2e59ca83b330f6ed6fd63ab4db"
"""``eva_adapter.EVA_COMMIT``, in full: what every image is built from."""
SIDECAR = Path(__file__).resolve().parents[2] / "sidecar"
SIDECAR_FILES = ("Dockerfile", "server.py", "aggregate.py")

EVA_BASE = f"eva-base:{EVA_COMMIT}"
"""EVA's own Dockerfile at the commit (the sidecar Dockerfile's default)."""
PLATFORM = "linux/amd64"
"""EVA builds its image for amd64 (``scripts/docker_build.sh``)."""
FROM_SOURCE = "eva-from-source"
"""The sidecar Dockerfile's own build of EVA's image (``EVA_BASE`` arg)."""

MAIN_DOCKERFILE = """\
# main: harbor-voice and the adapter are installed here at trial start, from
# wheels of the orchestrator's code.
FROM mirror.gcr.io/library/python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 UV_LINK_MODE=copy
WORKDIR /task
"""
"""tau-voice's ``main``: the call loop needs nothing the wheels don't bring."""

EVA_NUMPY = "2.2.6"
"""EVA's numpy (``uv.lock`` @039efa0a). numpy 2.3 changed how a float32 sum
of more than 8192 elements rounds (measured: the RMS of a 16 000-sample chunk
moves by 2 ulps), and EVA's perturbator takes each chunk's RMS that way."""
EVA_AUDIOOP_LTS = "0.2.2"
"""EVA's ``audioop-lts`` (``uv.lock`` @039efa0a), ``audioop`` on Python 3.13."""
NOISE_DIR = "/opt/eva/assets/noise"
"""Where ``main`` keeps EVA's noise WAVs: the vendored ``AudioPerturbator``'s
``_ASSETS_DIR`` (EVA's own is ``assets/noise`` in its repo)."""
NOISE_URL = (
    f"https://raw.githubusercontent.com/ServiceNow/eva/{EVA_COMMIT_FULL}/assets/noise/"
)
PERTURBATION_LAYER = (
    "# EVA's perturbations run here, in EVA's AudioPerturbator (eva_adapter's\n"
    "# copy), so with EVA's pins: numpy 2.3 rounds long float32 sums differently.\n"
    "# harbor-voice's install keeps them (it asks numpy>=2.0, audioop-lts>=0.2).\n"
    f"RUN uv pip install --system --no-cache numpy=={EVA_NUMPY} "
    f"audioop-lts=={EVA_AUDIOOP_LTS}\n"
    "# EVA's background-noise WAVs (assets/noise/ at ServiceNow/eva, the commit\n"
    "# above), where that copy reads them; fetched from EVA's repo, never\n"
    "# redistributed, each checked by SHA-256.\n"
    f"RUN mkdir -p {NOISE_DIR} && cd {NOISE_DIR} \\\n"
    "    && python -c 'import sys, urllib.request; "
    "[urllib.request.urlretrieve(sys.argv[1] + n, n) for n in sys.argv[2:]]' \\\n"
    f"        {NOISE_URL} \\\n"
    + "".join(f"        {name}.wav \\\n" for name in sorted(NOISE_FILES))
    + "    && printf '%s  %s\\n' \\\n"
    + "".join(
        f"        {digest} {name}.wav \\\n"
        for name, digest in sorted(NOISE_FILES.items())
    )
    + "     | sha256sum -c - \\\n"
    "    && chmod a+r *.wav\n"
)
"""What a perturbed task's ``main`` adds: EVA's numpy and the seven WAVs
(55 MB, fetched by the full commit id). Clean tasks keep
:data:`MAIN_DOCKERFILE` byte for byte."""
MAIN_PERTURBED_DOCKERFILE = MAIN_DOCKERFILE + PERTURBATION_LAYER
"""``main`` for every perturbation condition, noise or not: one image, so a
registry needs one more image, not one per condition."""


def main_dockerfile(perturbed: bool = False) -> str:
    """``main``'s Dockerfile; with EVA's numpy and WAVs for a perturbed task."""
    return MAIN_PERTURBED_DOCKERFILE if perturbed else MAIN_DOCKERFILE


def _digest(parts: list[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:10]


def _named(name: str, tag: str, registry: str | None) -> str:
    return f"{registry.rstrip('/')}/{name}:{tag}" if registry else f"{name}:{tag}"


def sidecar_image(registry: str | None = None, *, from_source: bool = False) -> str:
    """The sidecar: EVA at the commit, plus our server; the tag names both.

    ``from_source`` names the build a task makes for itself
    (:data:`FROM_SOURCE`), so it never shares a tag with the image built on
    EVA's own Dockerfile.
    """
    # Line ends normalised: a CRLF checkout (Windows) names the same image.
    files = [
        (SIDECAR / name).read_bytes().replace(b"\r\n", b"\n") for name in SIDECAR_FILES
    ]
    tag = f"{EVA_COMMIT}-{_digest(files)}" + ("-src" if from_source else "")
    return _named("eva-sidecar", tag, registry)


def main_image(registry: str | None = None, *, perturbed: bool = False) -> str:
    """``eva-main:<digest>``, or ``eva-main:perturbed-<digest>`` for a perturbed task."""
    digest = _digest([main_dockerfile(perturbed).encode()])
    return _named("eva-main", f"perturbed-{digest}" if perturbed else digest, registry)


def exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def build_eva_base(*, force: bool = False) -> str:
    """``eva-base:<commit>``: EVA's own Dockerfile on EVA's repo at the commit.

    As EVA's ``scripts/docker_build.sh`` builds it (``linux/amd64``, the
    ``GIT_*`` provenance args of a clean checkout), with the repo at the
    commit as the context instead of a local checkout.
    """
    if force or not exists(EVA_BASE):
        subprocess.run(
            [
                "docker",
                "build",
                "--platform",
                PLATFORM,
                "-t",
                EVA_BASE,
                "--build-arg",
                f"GIT_COMMIT_SHA={EVA_COMMIT_FULL}",
                "--build-arg",
                "GIT_BRANCH=main",
                "--build-arg",
                "GIT_DIRTY=false",
                f"{EVA_REPO}#{EVA_COMMIT_FULL}",
            ],
            check=True,
            stdin=subprocess.DEVNULL,
        )
    return EVA_BASE


def build(
    *,
    force: bool = False,
    registry: str | None = None,
    push: bool = False,
    from_source: bool = False,
) -> list[str]:
    """Build whichever image is missing (all with ``force``); returns the tags.

    The sidecar is built on :func:`build_eva_base` unless ``from_source``.
    ``main`` is built twice: as clean tasks use it, and with EVA's numpy and
    noise WAVs for tasks converted with a perturbation. With ``push``, all three are
    pushed to ``registry`` (log in to it first).
    """
    if push and not registry:
        raise ValueError("push needs a registry")
    sidecar = sidecar_image(registry, from_source=from_source)
    mains = {main_image(registry, perturbed=p): p for p in (False, True)}
    if force or not exists(sidecar):
        base = FROM_SOURCE if from_source else build_eva_base(force=force)
        subprocess.run(
            [
                "docker",
                "build",
                "--platform",
                PLATFORM,
                "-t",
                sidecar,
                "--build-arg",
                f"EVA_BASE={base}",
                str(SIDECAR),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
        )
    for main, perturbed in mains.items():
        if force or not exists(main):
            # No build context: the Dockerfile copies nothing from disk.
            subprocess.run(
                ["docker", "build", "-t", main, "-"],
                input=main_dockerfile(perturbed).encode(),
                check=True,
            )
    if push:
        for image in (sidecar, *mains):
            subprocess.run(
                ["docker", "push", image], check=True, stdin=subprocess.DEVNULL
            )
    return [sidecar, *mains]
