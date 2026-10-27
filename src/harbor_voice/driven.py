"""Driven trials — a benchmark's own call loop, run harbor's way.

A task whose ``voice.toml`` names a ``[call]`` driver runs in one of two
places:

**In an environment** (the task declares ``[environment]``). Harbor's model,
on any of harbor's backends (``docker`` locally, ``ec2``/``modal``/
``daytona``/… remotely), with nothing of ours copied:

1. harbor builds and starts the task's ``environment/`` — a ``main`` container
   plus whatever services the task's compose file adds (a benchmark sidecar);
2. harbor-voice and the driver's package are installed into ``main`` from
   wheels of the code this orchestrator runs — harbor's *agent install*,
   applied to the call runner (:mod:`harbor_voice.install`);
3. ``harbor-voice drive`` runs the call **inside** ``main``, so the call runs
   wherever the environment runs;
4. the task's verifier runs in the service ``[verifier]`` names, and its
   ``/logs/verifier`` is collected;
5. the environment is stopped and deleted.

**Here** (no ``[environment]``, or inside ``main`` as step 3): the driver is
called in this process. Offline tasks and tests use this directly.

The agent under test is never part of the task — the run supplies it, as
harbor's job config does.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from harbor_voice.driver import DriverContext
from harbor_voice.driver import DriverOutcome
from harbor_voice.driver import load_driver
from harbor_voice.driver import VerifierResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.trial import AgentSpec
    from harbor_voice.trial import VoiceTaskConfig
    from harbor_voice.transport.base import VoiceTransport

OUTCOME = "outcome.json"
"""Written by ``harbor-voice drive`` next to the driver's own files."""

TASK_IN_ENV = "/task"
CALL_OUT_IN_ENV = "/logs/agent/call"
WHEELS_IN_ENV = "/installed-agent/wheels"


class DrivenTrialError(RuntimeError):
    """A step of a driven trial failed; the message says which."""


@dataclass(frozen=True, slots=True)
class DrivenResult:
    outcome: DriverOutcome
    verified: VerifierResult | None
    installed: tuple[dict[str, str], ...] = ()


async def drive_here(
    config: VoiceTaskConfig,
    *,
    task_dir: Path,
    out_dir: Path,
    build_agent: Callable[..., VoiceTransport],
) -> DriverOutcome:
    """Run the task's driver in this process and write ``outcome.json``."""
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
    return outcome


async def drive_in_environment(
    config: VoiceTaskConfig,
    *,
    task_dir: Path,
    out_dir: Path,
    agent: AgentSpec,
    environment_type: str = "docker",
    environment_kwargs: dict[str, Any] | None = None,
    wheels_dir: Path | None = None,
) -> DrivenResult:
    """Harbor's trial shape for a driven task: start, install, drive, verify, stop."""
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.paths import TrialPaths

    from harbor_voice.install import build_wheels

    if config.call is None or config.environment is None:
        raise DrivenTrialError("an environment run needs [call] and [environment]")
    trial_id = config.trial_id or "trial"
    paths = TrialPaths(trial_dir=out_dir)
    paths.mkdir()
    environment = EnvironmentFactory.create_environment(
        EnvironmentType(environment_type),
        environment_dir=task_dir / "environment",
        environment_name=_name(config.task or task_dir.name),
        session_id=_name(trial_id),
        trial_paths=paths,
        task_env_config=config.environment,
        **(environment_kwargs or {}),
    )

    driver_module = config.call.driver.partition(":")[0]
    wheels = build_wheels(
        ["harbor_voice", driver_module], wheels_dir or out_dir / "wheels"
    )
    installed = tuple(
        {"distribution": w.distribution, "version": w.version, "wheel": w.path.name}
        for w in wheels
    )

    await environment.start(force_build=False)
    try:
        await _install(environment, wheels)
        await environment.exec(f"mkdir -p {TASK_IN_ENV} {CALL_OUT_IN_ENV}")
        await environment.upload_file(
            task_dir / "voice.toml", f"{TASK_IN_ENV}/voice.toml"
        )
        command = (
            f"harbor-voice drive {TASK_IN_ENV} --out {CALL_OUT_IN_ENV} "
            f"--agent-json {shlex.quote(agent.model_dump_json())} "
            f"--trial-id {shlex.quote(trial_id)}"
        )
        result = await environment.exec(
            command, timeout_sec=int(config.call.timeout_sec)
        )
        # Fetched, not mounted: harbor mounts /logs only when the caller passes
        # mounts, and a download works the same on every backend.
        await environment.download_dir("/logs/agent", paths.agent_dir)
        if result.return_code != 0:
            raise DrivenTrialError(
                f"the call failed in the environment ({result.return_code}): "
                f"{(result.stderr or result.stdout or '')[-800:]}"
            )
        outcome = _read_outcome(paths.agent_dir / "call" / OUTCOME)
        verified = await _verify(environment, config, paths)
    finally:
        await environment.stop(delete=True)
    return DrivenResult(outcome=outcome, verified=verified, installed=installed)


async def _install(environment: Any, wheels: list[Any]) -> None:
    """Harbor's agent install, for the call runner: upload wheels, pip install."""
    await environment.exec(f"mkdir -p {WHEELS_IN_ENV}", user="root")
    targets = []
    for wheel in wheels:
        target = f"{WHEELS_IN_ENV}/{wheel.path.name}"
        await environment.upload_file(wheel.path, target)
        # harbor-voice's own wheel brings its voice extra; the adapter's
        # dependencies come with it.
        extra = "[voice]" if wheel.distribution == "harbor" else ""
        targets.append(shlex.quote(target + extra))
    result = await environment.exec(
        "uv pip install --system --quiet " + " ".join(targets),
        user="root",
        timeout_sec=1800,
    )
    if result.return_code != 0:
        raise DrivenTrialError(
            f"installing the call runner failed: {(result.stderr or '')[-800:]}"
        )


async def _verify(
    environment: Any, config: VoiceTaskConfig, paths: Any
) -> VerifierResult | None:
    verifier = config.verifier
    if verifier is None:
        return None
    result = await environment.service_exec(
        " ".join(shlex.quote(part) for part in verifier.command),
        service=verifier.service,
        env=verifier.env or None,
        timeout_sec=int(verifier.timeout_sec),
    )
    await environment.service_download_dir(
        "/logs/verifier", paths.verifier_dir, service=verifier.service
    )
    reward_file = paths.verifier_dir / "reward.txt"
    if result.return_code != 0 or not reward_file.exists():
        raise DrivenTrialError(
            f"the verifier failed ({result.return_code}): "
            f"{(result.stderr or result.stdout or '')[-800:]}"
        )
    text = reward_file.read_text(encoding="utf-8").strip()
    result_file = paths.verifier_dir / "result.json"
    detail = (
        json.loads(result_file.read_text(encoding="utf-8"))
        if result_file.exists()
        else {}
    )
    return VerifierResult(reward=float(text) if text else None, result=detail)


def _read_outcome(path: Path) -> DriverOutcome:
    if not path.exists():
        raise DrivenTrialError(f"the driver wrote no {OUTCOME} at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    verified = data.get("verified")
    return DriverOutcome(
        verified=(
            VerifierResult(
                reward=verified["reward"], result=verified.get("result") or {}
            )
            if verified
            else None
        ),
        summary=data.get("summary") or {},
    )


def _name(raw: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in raw.lower())
    return cleaned.strip("-")[:60] or "voice"
