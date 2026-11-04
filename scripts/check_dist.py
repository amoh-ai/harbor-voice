"""Release checks on built packages: what goes to PyPI, and nothing else.

::

    uv build --all-packages --out-dir dist
    uv run python scripts/check_dist.py dist [--tag v0.1.0]

harbor-voice is released as three packages at one version (:data:`PACKAGES`).
The checks:

- one wheel and one sdist per package, all at one version, the tag's with
  ``--tag``;
- harbor-voice's ``tau-voice`` and ``eva`` extras pin the adapters at that
  version (:data:`EXTRAS`);
- every file in every archive is the package's own code, metadata or license
  (:func:`allowed`); an sdist's ``.gitignore`` is this repository's;
- no text file holds a credential, a local user path, an email address or a
  reference to internal notes (:data:`LEAKS`).
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = {
    "harbor-voice": "harbor_voice",
    "harbor-voice-tau-voice-adapter": "tau_voice",
    "harbor-voice-eva-adapter": "eva_adapter",
}
"""Each distribution, and the one top-level module it ships."""

EXTRAS = {
    "tau-voice": "harbor-voice-tau-voice-adapter",
    "eva": "harbor-voice-eva-adapter",
}
"""harbor-voice's extras that install an adapter at harbor-voice's version."""

SDIST_FILES = {"PKG-INFO", "pyproject.toml", "README.md", "LICENSE", ".gitignore"}
"""What an sdist holds beside its package's source folder."""

FORBIDDEN = re.compile(
    r"(^|/)(tests|docs|jobs|datasets|internal|__pycache__|\.venv)/"
    r"|(^|/)\.env(\.|$)|\.(pyc|log|wav|whl)$"
)
"""Paths no archive may hold, even inside an allowed folder."""

LEAKS = {
    "an OpenAI key": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}",
    "an ElevenLabs key": r"\bsk_[a-f0-9]{40,}\b",
    "an AWS access key id": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "an AWS secret key": r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+]{40}",
    "a Google API key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "a GitHub token": r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    "a private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "an AWS account id": r"\barn:aws[a-z-]*:[a-z0-9-]+:[a-z0-9-]*:\d{12}:",
    "a local user path": (
        r"(?i)\b[a-z]:[\\/]+users[\\/]+[a-z0-9._-]+"
        r"|/home/[a-z][a-z0-9_-]*/|/Users/[A-Za-z][A-Za-z0-9_-]*/"
    ),
    "an email address": (
        r"\b[A-Za-z0-9._%+-]+@(?!example\.(?:com|org)\b)"
        r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
    ),
    "a reference to internal notes": r"(?i)\bdocs/internal\b",
}
"""What no text file in an archive may hold."""

_LEAKS = {label: re.compile(pattern) for label, pattern in LEAKS.items()}
_WHEEL = re.compile(r"^(?P<name>[A-Za-z0-9_.]+)-(?P<version>[^-]+)-.+\.whl$")
_SDIST = re.compile(r"^(?P<name>[A-Za-z0-9_.]+)-(?P<version>[^-]+)\.tar\.gz$")


def normalized(distribution: str) -> str:
    """The name as archive file names spell it (``harbor_voice``)."""
    return re.sub(r"[-_.]+", "_", distribution).lower()


_BY_FILE_NAME = {normalized(name): name for name in PACKAGES}


def members(archive: Path) -> dict[str, bytes]:
    """Every file in a wheel or sdist, by path (an sdist's without its root)."""
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as zipped:
            return {
                name: zipped.read(name)
                for name in zipped.namelist()
                if not name.endswith("/")
            }
    with tarfile.open(archive) as tarred:
        files: dict[str, bytes] = {}
        for member in tarred.getmembers():
            if not member.isfile():
                continue
            extracted = tarred.extractfile(member)
            assert extracted is not None
            files[member.name.partition("/")[2]] = extracted.read()
        return files


def allowed(path: str, *, distribution: str, wheel: bool) -> bool:
    """Whether ``path`` belongs in ``distribution``'s wheel or sdist."""
    if FORBIDDEN.search(path):
        return False
    module = PACKAGES[distribution]
    if wheel:
        top = path.split("/", 1)[0]
        return path.startswith(f"{module}/") or (
            top.endswith(".dist-info")
            and top.startswith(f"{normalized(distribution)}-")
        )
    return path in SDIST_FILES or path.startswith(f"src/{module}/")


def leaks(text: str) -> list[str]:
    """What :data:`LEAKS` finds in ``text``, one line per kind."""
    found = []
    for label, pattern in _LEAKS.items():
        match = pattern.search(text)
        if match is not None:
            found.append(f"{label} ({match.group(0)[:24]!r})")
    return found


def metadata(files: dict[str, bytes]) -> dict[str, list[str]]:
    """A wheel's METADATA fields, each as a list of its values."""
    (name,) = [p for p in files if p.endswith(".dist-info/METADATA")]
    parsed = Parser().parsestr(files[name].decode("utf-8"), headersonly=True)
    fields: dict[str, list[str]] = {}
    for key, value in parsed.items():
        fields.setdefault(key, []).append(value)
    return fields


def check(
    dist: Path, *, tag: str | None = None, gitignore: Path | None = None
) -> list[str]:
    """Every problem with the archives in ``dist``; empty when all is well."""
    gitignore = ROOT / ".gitignore" if gitignore is None else gitignore
    problems: list[str] = []
    found: dict[tuple[str, bool], str] = {}
    for archive in sorted(dist.iterdir()):
        match = _WHEEL.match(archive.name) or _SDIST.match(archive.name)
        if match is None:
            if archive.suffix in (".whl", ".gz"):
                problems.append(f"{archive.name}: not a wheel or sdist name")
            continue
        distribution = _BY_FILE_NAME.get(match["name"].lower())
        if distribution is None:
            problems.append(f"{archive.name}: not one of {sorted(PACKAGES)}")
            continue
        wheel = archive.suffix == ".whl"
        if (distribution, wheel) in found:
            problems.append(f"{archive.name}: a second {'wheel' if wheel else 'sdist'}")
        found[(distribution, wheel)] = match["version"]
        files = members(archive)
        for path, data in sorted(files.items()):
            if not allowed(path, distribution=distribution, wheel=wheel):
                problems.append(f"{archive.name}: {path} does not belong in it")
                continue
            if path == ".gitignore":
                if data.replace(b"\r\n", b"\n") != gitignore.read_bytes().replace(
                    b"\r\n", b"\n"
                ):
                    problems.append(f"{archive.name}: .gitignore is not the repo's")
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            problems.extend(f"{archive.name}: {path}: {leak}" for leak in leaks(text))
        if wheel and distribution == "harbor-voice":
            problems.extend(
                f"{archive.name}: {issue}"
                for issue in _extras(metadata(files), match["version"])
            )

    for distribution in PACKAGES:
        for wheel in (True, False):
            if (distribution, wheel) not in found:
                kind = "wheel" if wheel else "sdist"
                problems.append(f"no {kind} for {distribution}")
    versions = set(found.values())
    if len(versions) > 1:
        problems.append(f"the packages' versions differ: {sorted(versions)}")
    if tag is not None:
        wanted = tag.removeprefix("v")
        if versions and versions != {wanted}:
            problems.append(f"tag {tag} but the packages are {sorted(versions)}")
    return problems


def _extras(fields: dict[str, list[str]], version: str) -> list[str]:
    requires = fields.get("Requires-Dist", [])
    issues = []
    for extra, distribution in EXTRAS.items():
        wanted = f"{distribution}=={version}"
        pins = [
            r.split(";")[0].strip()
            for r in requires
            if r.startswith(distribution) and f"extra == '{extra}'" in r
        ]
        if pins != [wanted]:
            issues.append(f"extra {extra!r} should require {wanted}, not {pins}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Release checks on built packages: what goes to PyPI."
    )
    parser.add_argument("dist", type=Path, help="the folder uv build wrote to")
    parser.add_argument("--tag", help="the release tag, e.g. v0.1.0")
    args = parser.parse_args(argv)
    problems = check(args.dist, tag=args.tag)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"ok: {len(PACKAGES)} packages, wheel and sdist each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
