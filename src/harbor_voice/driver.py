"""Call drivers: a benchmark's own call loop, run by :class:`VoiceTrial`.

The built-in loop is turn-based — a content source, a simulator, a floor.
Some benchmarks define the call differently: tau-voice runs it in lockstep
0.2 s steps with a caller whose turn-taking is the benchmark's own. Rather
than grow core to fit each one, a task names a **driver** in ``voice.toml``::

    [call]
    driver = "tau_voice.trial:drive"
    config = { domain = "airline", task_id = "7" }

The driver is the adapter's code. Core supplies everything that is not
benchmark-specific through a :class:`DriverContext` — the agent under test
built from the task's ``[agent]``, the task's services, the verifier — and
writes the trial's outputs from the :class:`DriverOutcome` it returns.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any


if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.trajectory.model import VoiceTrajectory
    from harbor_voice.trial import VoiceTaskConfig
    from harbor_voice.transport.base import VoiceTransport


@dataclass(frozen=True, slots=True)
class VerifierResult:
    """What a task's verifier concluded."""

    reward: float | None
    """``logs/verifier/reward.txt`` — harbor's scalar reward."""
    result: Mapping[str, Any] = field(default_factory=dict)
    """``logs/verifier/result.json``, if the verifier wrote one."""


class DriverError(RuntimeError):
    """A driver could not be loaded, or broke its contract."""


@dataclass(slots=True)
class DriverContext:
    """What core hands a driver.

    Attributes:
        config: The task's ``voice.toml``.
        driver_config: Its ``[call].config`` table — the driver's own settings.
        task_dir: The task directory.
        out_dir: Where this trial's files go. The driver may write anything
            under it; core writes ``reward.json`` and ``trial.json`` beside.
        services: Each declared service's URL, by name.
        build_agent: Builds the agent under test from the task's ``[agent]``.
            Keyword arguments override that slot's config (a benchmark's
            system prompt, say); ``tools`` is the driver's router.
        verify: Runs the task's verifier; ``None`` if the task declares none.
            A driver may call it more than once — tau-voice grades every
            attempt to decide on a retry.
        progress: Reports a line of progress (an attempt started, graded).
            Printed with :data:`PROGRESS_PREFIX`; ``VoiceCallAgent`` relays
            such lines into harbor's ``trial.log``.
    """

    config: VoiceTaskConfig
    driver_config: Mapping[str, Any]
    task_dir: Path
    out_dir: Path
    services: Mapping[str, str]
    build_agent: Callable[..., VoiceTransport]
    verify: Callable[[], Awaitable[VerifierResult]] | None = None
    progress: Callable[[str], None] = field(default_factory=lambda: print_progress)


PROGRESS_PREFIX = "harbor-voice: "
"""Marks a progress line in a driver's output, so a runner can pick it out."""


def print_progress(message: str) -> None:
    print(f"{PROGRESS_PREFIX}{message}", flush=True)


@dataclass(slots=True)
class DriverOutcome:
    """What a driver hands back.

    Attributes:
        verified: The verifier result that counts. ``None`` lets core run the
            verifier once itself after the driver returns.
        summary: Driver-specific record of the call, written to ``trial.json``.
        trajectory: The call as ATIF-voice — the attempt that counts. Written
            as ATIF where harbor reads an agent's trajectory.
    """

    verified: VerifierResult | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    trajectory: VoiceTrajectory | None = None


CallDriver = Callable[[DriverContext], Awaitable[DriverOutcome]]


def load_driver(path: str) -> CallDriver:
    """Import ``"package.module:function"``."""
    module_name, sep, attribute = path.partition(":")
    if not sep or not module_name or not attribute:
        raise DriverError(f"driver must be 'module:function', got {path!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise DriverError(
            f"driver module {module_name!r} is not importable; is the adapter "
            f"installed in this environment? ({exc})"
        ) from exc
    driver = getattr(module, attribute, None)
    if not callable(driver):
        raise DriverError(f"{path!r} is not a callable driver")
    return driver
