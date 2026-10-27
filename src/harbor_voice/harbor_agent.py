"""A voice call as a harbor agent, so harbor's own job runner runs voice tasks.

Harbor's job runner (``harbor jobs start``, mounted as ``harbor-voice jobs``)
gives concurrency, attempts, retries, resume, ``result.json`` and metrics —
for harbor ``Trial`` objects,
whose unit is a task plus an *agent*. A voice task is a harbor task whose
"agent" is the call runner: it installs harbor-voice and the task's driver
package into ``main``, then runs ``harbor-voice drive`` there. The voice agent
under test is this agent's model — the run's choice, never the task's::

    harbor-voice jobs start -p datasets/tau-voice \\
      -m openai.realtime/gpt-realtime-1.5

``-m`` is an S2S spec; ``--ak agent_json=…`` takes any ``AgentSpec`` (a
cascade). ``--ak seed=N`` overrides the task's seed: a job config listing
this agent once per seed runs each task under each, as tau2's trials do.

The task reaches ``main`` through its instruction: harbor hands an agent the
task's ``instruction.md`` and nothing else of the task folder, so a voice
task's instruction carries its ``voice.toml`` in a ``toml`` block.

A call that ran is never an exception — it is graded by the task's verifier.
A call that could not run (a provider down, the runner crashing) raises
:class:`VoiceInfrastructureError`, the type a job's ``retry`` targets; a
timeout stays harbor's ``AgentTimeoutError``, which harbor does not retry.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import tempfile
import threading
import tomllib
from pathlib import Path
from typing import Any
from typing import override

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

from harbor_voice import install
from harbor_voice.capabilities import Pipeline

TASK_IN_ENV = "/task"
CALL_OUT_IN_ENV = "/logs/agent/call"

_SPEC = re.compile(r"```toml\n(.*?)\n```", re.DOTALL)


class VoiceInfrastructureError(RuntimeError):
    """The call could not run; retryable, and not a graded result."""


class VoiceCallOptions(InstalledAgentOptions):
    agent_json: str | None = Field(
        default=None,
        description="Agent under test as AgentSpec JSON; overrides the model.",
    )
    seed: int | None = Field(default=None, description="Run the task under this seed.")


def call_spec(instruction: str) -> str:
    """The ``voice.toml`` a voice task's instruction carries."""
    found = _SPEC.search(instruction)
    if found is None:
        raise VoiceInfrastructureError(
            "the instruction carries no ```toml block: not a voice task"
        )
    return found.group(1)


class _Wheels:
    """Wheels of the running code, built once per process and shared."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._built: dict[str, install.Wheel] = {}
        self._dir = Path(tempfile.mkdtemp(prefix="harbor-voice-wheels-"))

    def get(self, module: str) -> install.Wheel:
        with self._lock:
            if module not in self._built:
                (wheel,) = install.build_wheels([module], self._dir / module)
                self._built[module] = wheel
            return self._built[module]


_wheels = _Wheels()


class VoiceCallAgent(BaseInstalledAgent):
    options_model = VoiceCallOptions

    @staticmethod
    @override
    def name() -> str:
        return "harbor-voice"

    @override
    def version(self) -> str | None:
        import importlib.metadata

        return importlib.metadata.version(install.distribution_of("harbor_voice"))

    def agent_spec_json(self) -> str:
        from harbor_voice.trial import AgentSpec

        options = self.options
        if isinstance(options, VoiceCallOptions) and options.agent_json:
            return AgentSpec.model_validate_json(options.agent_json).model_dump_json()
        if not self.model_name:
            raise ValueError(
                "no agent under test: pass -m <s2s spec> or --ak agent_json=..."
            )
        return AgentSpec(pipeline=Pipeline.S2S, s2s=self.model_name).model_dump_json()

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        wheel = await asyncio.to_thread(_wheels.get, "harbor_voice")
        await install.install_into(environment, [wheel])

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        from harbor_voice.trial import VoiceTaskConfig

        spec = call_spec(instruction)
        config = VoiceTaskConfig.model_validate(tomllib.loads(spec))
        if config.call is None:
            raise VoiceInfrastructureError("the task names no [call] driver")
        agent = self.agent_spec_json()

        try:
            driver = config.call.driver.partition(":")[0]
            wheel = await asyncio.to_thread(_wheels.get, driver)
            await install.install_into(environment, [wheel])
        except install.InstallError as exc:
            raise VoiceInfrastructureError(str(exc)) from exc

        await environment.exec(f"mkdir -p {TASK_IN_ENV} {CALL_OUT_IN_ENV}")
        with tempfile.TemporaryDirectory() as scratch:
            local = Path(scratch) / "voice.toml"
            local.write_text(spec, encoding="utf-8", newline="\n")
            await environment.upload_file(local, f"{TASK_IN_ENV}/voice.toml")

        options = self.options
        seed = options.seed if isinstance(options, VoiceCallOptions) else None
        command = (
            f"harbor-voice drive {TASK_IN_ENV} --out {CALL_OUT_IN_ENV} "
            f"--agent-json {shlex.quote(agent)}"
            + (f" --seed {seed}" if seed is not None else "")
        )
        result = await environment.exec(command)
        if result.return_code != 0:
            raise VoiceInfrastructureError(
                f"the call did not run ({result.return_code}): "
                f"{(result.stderr or result.stdout or '')[-800:]}"
            )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        outcome = self.logs_dir / "call" / "outcome.json"
        if not outcome.exists():
            return
        data: dict[str, Any] = json.loads(outcome.read_text(encoding="utf-8"))
        context.metadata = {**(context.metadata or {}), "call": data.get("summary")}
