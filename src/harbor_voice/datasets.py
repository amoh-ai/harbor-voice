"""harbor-voice's published datasets, and where harbor finds them.

``registry.json``, at the root of the repository, lists every dataset version
and pins its tasks and images. harbor resolves a dataset from it by name: from
the checkout's copy when harbor-voice runs from a checkout, and otherwise from
the repository itself, through harbor's git registry (``--repo``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = "amoh-ai/harbor-voice"
"""The repository harbor reads ``registry.json`` from."""


def checkout_registry() -> Path | None:
    """``registry.json`` of the checkout this code runs from.

    ``None`` when harbor-voice is installed from a package index: the file is
    then not beside the code, and harbor reads it from :data:`REPO`.
    """
    here = Path(__file__).resolve()
    registry = here.parents[2] / "registry.json"
    if here.parents[1].name == "src" and registry.is_file():
        return registry
    return None


def hub_dataset(spec: str, registry: Path | None = None) -> dict[str, Any]:
    """harbor's dataset entry for ``name@version``, or ``name`` for the latest.

    With a ``registry`` file (the checkout's by default), the name and version
    are checked against it and the entry reads it. Without one, the entry
    names :data:`REPO`, and harbor checks the name when the job starts. A name
    or version the file doesn't list is a ``ValueError``.
    """
    name, _, version = spec.partition("@")
    if not name:
        raise ValueError("name a dataset: NAME or NAME@VERSION")
    registry = checkout_registry() if registry is None else registry
    entry: dict[str, Any] = {"name": name}
    if registry is None:
        entry["repo"] = REPO
    else:
        if not registry.is_file():
            raise ValueError(f"no {registry.name} at {registry.parent}")
        entries = json.loads(registry.read_text(encoding="utf-8"))
        versions = [item["version"] for item in entries if item["name"] == name]
        if not versions:
            listed = sorted({item["name"] for item in entries})
            raise ValueError(f"{name!r} is not in {registry.name}; it lists {listed}")
        if version and version not in versions:
            raise ValueError(f"{name} has no version {version!r}; it has {versions}")
        entry["registry_path"] = str(registry)
    if version:
        entry["version"] = version
    return entry


def registry_args() -> list[str]:
    """The flags that point ``harbor jobs start -d`` at harbor-voice's registry."""
    registry = checkout_registry()
    if registry is not None:
        return ["--registry-path", str(registry)]
    return ["--repo", REPO]
