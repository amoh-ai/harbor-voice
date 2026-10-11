"""Lazy access to the ``voice`` extra, with an error that says what to install.

harbor ships no numpy, scipy or soundfile, and resolves vendor SDKs lazily via
``(module, class_name, pip_extra)`` in ``environments/factory.py`` so a broken
or absent vendor package cannot break unrelated backends. harbor_voice follows
the same rule: nothing at import time depends on the ``voice`` extra, so a
plain ``import harbor_voice`` works without it.
"""

from __future__ import annotations

from types import ModuleType

_EXTRA = "voice"


class MissingVoiceDependency(ImportError):
    """A voice-only dependency is needed and not installed."""


def _require(module: str, *, used_for: str) -> ModuleType:
    """Import ``module`` or raise with the exact command to fix it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised by the message test
        raise MissingVoiceDependency(
            f"{module!r} is needed for {used_for} but is not installed. "
            f"Install the voice extra:  pip install 'harbor[{_EXTRA}]'"
        ) from exc


def numpy() -> ModuleType:
    """NumPy, for audio DSP."""
    return _require("numpy", used_for="audio DSP")


def websockets() -> ModuleType:
    """The websockets client, for realtime provider transports."""
    return _require("websockets", used_for="realtime transports")
