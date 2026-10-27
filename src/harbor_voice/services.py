"""A task's services: its ``environment/docker-compose.yaml``, run per trial.

Harbor's model, kept: a task directory carries the containers it needs and a
verifier that grades the trial. A voice trial's agent is reached over a
transport rather than run in a sandbox, so here the containers are the
*benchmark's* — a tool runtime, a judge — and the call runs harness-side
against them.

Per trial:

1. ``docker compose up --wait`` under a project name of its own, with
   ``HARBOR_VOICE_LOGS`` pointing at a host directory the compose file mounts
   (as ``/logs``, by convention), so whatever the services write survives
   them;
2. the driver reaches each declared service at its published host port;
3. the verifier runs *inside* a service (``docker compose exec``), where the
   benchmark's own code and state live, and writes ``logs/verifier/
   reward.txt`` and ``result.json``;
4. ``docker compose down``, always.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

LOGS_ENV = "HARBOR_VOICE_LOGS"

_WINDOWS_DOCKER = Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe")


class ServiceError(RuntimeError):
    """A compose command failed."""


def docker_binary() -> str:
    """``docker`` on PATH, or Docker Desktop's on Windows where it often is not."""
    found = shutil.which("docker")
    if found:
        return found
    if _WINDOWS_DOCKER.exists():
        return str(_WINDOWS_DOCKER)
    raise ServiceError("docker not found; install Docker or put it on PATH")


@dataclass(frozen=True, slots=True)
class VerifierResult:
    """What a task's verifier concluded."""

    reward: float | None
    """``logs/verifier/reward.txt`` — harbor's scalar reward."""
    result: Mapping[str, Any] = field(default_factory=dict)
    """``logs/verifier/result.json``, if the verifier wrote one."""


class ComposeEnvironment:
    """One trial's instance of a task's compose services.

    Args:
        compose_file: The task's ``environment/docker-compose.yaml``.
        project: A compose project name unique to this trial.
        logs_dir: Host directory mounted as the services' ``/logs``.
        env: Extra environment for compose — passed through to the services
            the compose file names it in, never written anywhere.
    """

    def __init__(
        self,
        compose_file: Path,
        *,
        project: str,
        logs_dir: Path,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.compose_file = compose_file
        self.project = project
        self.logs_dir = logs_dir
        self._env = {**os.environ, **(env or {}), LOGS_ENV: str(logs_dir.resolve())}
        self._docker = docker_binary()
        self.is_up = False

    async def _compose(self, *args: str, check: bool = True) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            self._docker,
            "compose",
            "-f",
            str(self.compose_file),
            "-p",
            self.project,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )
        out, err = await process.communicate()
        code = process.returncode if process.returncode is not None else -1
        stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
        if check and code != 0:
            raise ServiceError(
                f"docker compose {' '.join(args[:2])} failed ({code}): {stderr[-600:]}"
            )
        return code, stdout, stderr

    async def up(self, *, timeout_s: int = 300) -> None:
        """Start every service and wait until each reports healthy."""
        (self.logs_dir / "agent").mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "verifier").mkdir(parents=True, exist_ok=True)
        await self._compose("up", "-d", "--wait", "--wait-timeout", str(timeout_s))
        self.is_up = True

    async def url(self, service: str, port: int) -> str:
        """``http://host:port`` for a service's container port, as published."""
        _, out, _ = await self._compose("port", service, str(port))
        host, _, published = out.strip().splitlines()[-1].rpartition(":")
        host = "127.0.0.1" if host in ("0.0.0.0", "", "[::]", "::") else host
        return f"http://{host}:{published}"

    async def exec(self, service: str, command: Sequence[str]) -> tuple[int, str, str]:
        return await self._compose("exec", "-T", service, *command, check=False)

    async def verify(self, service: str, command: Sequence[str]) -> VerifierResult:
        """Run the verifier inside ``service`` and read what it wrote."""
        code, _, err = await self.exec(service, command)
        reward_file = self.logs_dir / "verifier" / "reward.txt"
        result_file = self.logs_dir / "verifier" / "result.json"
        if code != 0 or not reward_file.exists():
            raise ServiceError(f"verifier failed ({code}): {err[-600:]}")
        text = reward_file.read_text(encoding="utf-8").strip()
        result = (
            json.loads(result_file.read_text(encoding="utf-8"))
            if result_file.exists()
            else {}
        )
        return VerifierResult(reward=float(text) if text else None, result=result)

    async def down(self) -> None:
        if not self.is_up:
            return
        await self._compose("down", "-v", "--remove-orphans", check=False)
        self.is_up = False
