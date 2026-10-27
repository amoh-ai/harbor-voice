"""Driven trials — a benchmark's own call loop, run by its driver.

A task whose ``voice.toml`` names a ``[call]`` driver runs it here, in this
process: ``harbor-voice drive`` calls :func:`drive_here` inside the task's
``main`` container, and offline tasks and tests call it directly.

Everything around the call is harbor's: harbor's trial builds and starts the
task's environment, ``VoiceCallAgent`` installs harbor-voice and the driver's
package into ``main`` and runs ``harbor-voice drive`` there
(:mod:`harbor_voice.harbor_agent`), and harbor's verifier grades the result.
The agent under test is never part of the task — the run supplies it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from harbor_voice.driver import DriverContext
from harbor_voice.driver import DriverOutcome
from harbor_voice.driver import load_driver

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.trial import VoiceTaskConfig
    from harbor_voice.transport.base import VoiceTransport

OUTCOME = "outcome.json"
"""Written by ``harbor-voice drive`` next to the driver's own files."""


class DrivenTrialError(RuntimeError):
    """A step of a driven trial failed; the message says which."""


async def drive_here(
    config: VoiceTaskConfig,
    *,
    task_dir: Path,
    out_dir: Path,
    build_agent: Callable[..., VoiceTransport],
    trajectory_path: Path | None = None,
) -> DriverOutcome:
    """Run the task's driver in this process and write ``outcome.json``.

    The driver's trajectory, if it returns one, is written as ATIF to
    ``trajectory_path`` (default ``out_dir/trajectory.json``) — harbor reads
    an agent's from ``/logs/agent/trajectory.json``.
    """
    if config.call is None:
        raise DrivenTrialError("the task names no [call] driver")
    driver = load_driver(config.call.driver)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = DriverContext(
        config=config,
        driver_config=config.call.config,
        task_dir=task_dir,
        out_dir=out_dir,
        services={},
        build_agent=build_agent,
        verify=None,
    )
    outcome = await driver(context)
    (out_dir / OUTCOME).write_text(
        json.dumps(
            {
                "verified": (
                    {
                        "reward": outcome.verified.reward,
                        "result": dict(outcome.verified.result),
                    }
                    if outcome.verified
                    else None
                ),
                "summary": outcome.summary,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if outcome.trajectory is not None:
        path = trajectory_path or out_dir / "trajectory.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            outcome.trajectory.to_atif().model_dump_json(indent=2), encoding="utf-8"
        )
    return outcome
