"""Lazy imports of numpy and websockets, with an error that says what to install.

Both are core dependencies of harbor-voice, so this only fires in a broken
environment. Importing them on first use keeps ``import harbor_voice`` light.
"""

from __future__ import annotations

from types import ModuleType


class MissingVoiceDependency(ImportError):
    """A dependency harbor-voice needs is not installed."""


def _require(module: str, *, used_for: str) -> ModuleType:
    """Import ``module`` or raise with the exact command to fix it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised by the message test
        raise MissingVoiceDependency(
            f"{module!r} is needed for {used_for} but is not installed. "
            f"Reinstall harbor-voice:  pip install --force-reinstall harbor-voice"
        ) from exc


def numpy() -> ModuleType:
    """NumPy, for audio DSP."""
    return _require("numpy", used_for="audio DSP")


def websockets() -> ModuleType:
    """The websockets client, for realtime provider transports."""
    return _require("websockets", used_for="realtime transports")
