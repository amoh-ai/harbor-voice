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
``--ak call_config='{…}'`` overrides settings of the task's driver
(``[call].config``) for the run.

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
import logging
import re
import shlex
import tempfile
import threading
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import override

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

from harbor_voice import install
from harbor_voice.capabilities import Pipeline
from harbor_voice.driver import PROGRESS_PREFIX

if TYPE_CHECKING:
    from harbor_voice.providers.registry import ProviderRegistry

TASK_IN_ENV = "/task"
CALL_OUT_IN_ENV = "/logs/agent/call"
TRAJECTORY_IN_ENV = "/logs/agent/trajectory.json"
"""Where harbor reads an agent's ATIF trajectory."""
KEYS_IN_ENV = "/task/.agent-keys"
"""The agent's credentials, uploaded as a file the drive command reads and
deletes before the call starts. Uploaded, not passed as ``exec`` env: harbor's
Docker backend turns that into ``docker exec -e KEY=value``, which anyone on
the host can read in the process list."""

_SPEC = re.compile(r"```toml\n(.*?)\n```", re.DOTALL)


def harbor_parses_model_names() -> bool:
    """Whether harbor resolves ``-m`` through litellm (and has the switch off).

    Our model names are harbor-voice provider specs
    (``openai.realtime/gpt-realtime-1.5``), not litellm ids; harbor 0.25+
    warns about each one unless the agent config sets ``skip_model_parsing``.
    Earlier harbor neither parses nor has the field.
    """
    from harbor.models.trial.config import AgentConfig

    return "skip_model_parsing" in AgentConfig.model_fields


class VoiceInfrastructureError(RuntimeError):
    """The call could not run; retryable, and not a graded result."""


class VoiceCallOptions(InstalledAgentOptions):
    agent_json: str | None = Field(
        default=None,
        description="Agent under test as AgentSpec JSON; overrides the model.",
    )
    seed: int | None = Field(default=None, description="Run the task under this seed.")
    call_config: dict[str, Any] | None = Field(
        default=None,
        description="Overrides for the task's [call].config, its driver's settings.",
    )


def call_spec(instruction: str) -> str:
    """The ``voice.toml`` a voice task's instruction carries."""
    found = _SPEC.search(instruction)
    if found is None:
        raise VoiceInfrastructureError(
            "the instruction carries no ```toml block: not a voice task"
        )
    return found.group(1)


class _ProgressRelay:
    """Driver progress lines (:data:`PROGRESS_PREFIX`) into the trial's log."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._partial = ""
        self.relayed = False

    async def feed(self, text: str, stream: Any) -> None:
        del stream
        *complete, self._partial = (self._partial + text).split("\n")
        for line in complete:
            if self._log(line):
                self.relayed = True

    def lines(self, text: str) -> None:
        for line in text.splitlines():
            self._log(line)

    def _log(self, line: str) -> bool:
        if not line.startswith(PROGRESS_PREFIX):
            return False
        self._logger.info(line.removeprefix(PROGRESS_PREFIX).strip())
        return True


class _Packages:
    """The running code's packages for ``main``, resolved once per process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._resolved: dict[str, install.Package] = {}
        self._dir = Path(tempfile.mkdtemp(prefix="harbor-voice-wheels-"))

    def get(self, module: str) -> install.Package:
        with self._lock:
            if module not in self._resolved:
                self._resolved[module] = install.package_for(module, self._dir / module)
            return self._resolved[module]


_packages = _Packages()


class VoiceCallAgent(BaseInstalledAgent):
    options_model = VoiceCallOptions
    capabilities = AgentCapabilities(atif=True)
    _registry: ProviderRegistry | None = None

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

    def agent_extras(self) -> list[str]:
        """harbor-voice extras the agent under test needs (``live`` for Live)."""
        from harbor_voice.trial import AgentSpec
        from harbor_voice.trial import required_extras

        spec = AgentSpec.model_validate_json(self.agent_spec_json())
        return required_extras(spec, self._providers())

    def agent_env(self) -> dict[str, str]:
        """The agent under test's credentials, for the call's environment."""
        from harbor_voice.trial import AgentSpec
        from harbor_voice.trial import required_env

        spec = AgentSpec.model_validate_json(self.agent_spec_json())
        return required_env(spec, self._providers())

    def _providers(self) -> ProviderRegistry:
        """The provider registry, built once per agent (it imports every
        vendor package) and shared by install and run."""
        from harbor_voice.providers.registry import provider_registry

        if self._registry is None:
            self._registry = provider_registry()
        return self._registry

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        started = time.monotonic()
        package = await asyncio.to_thread(_packages.get, "harbor_voice")
        await install.install_into(environment, [package], extras=self.agent_extras())
        self.logger.info(
            f"install: {package.distribution} {package.version} into main "
            f"({time.monotonic() - started:.0f} s)"
        )

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
            package = await asyncio.to_thread(_packages.get, driver)
            await install.install_into(environment, [package])
        except install.InstallError as exc:
            raise VoiceInfrastructureError(str(exc)) from exc

        await environment.exec(f"mkdir -p {TASK_IN_ENV} {CALL_OUT_IN_ENV}")
        keys = self.agent_env()
        with tempfile.TemporaryDirectory() as scratch:
            local = Path(scratch) / "voice.toml"
            local.write_text(spec, encoding="utf-8", newline="\n")
            await environment.upload_file(local, f"{TASK_IN_ENV}/voice.toml")
            if keys:
                secret = Path(scratch) / "keys.env"
                secret.touch(mode=0o600)
                secret.write_text(
                    "".join(f"{k}={shlex.quote(v)}\n" for k, v in keys.items()),
                    encoding="utf-8",
                )
                await environment.upload_file(secret, KEYS_IN_ENV)

        options = self.options
        seed = options.seed if isinstance(options, VoiceCallOptions) else None
        call_config = (
            options.call_config if isinstance(options, VoiceCallOptions) else None
        )
        self.logger.info(
            f"call: {config.task or 'task'} · seed {seed if seed is not None else config.seed}"
            f" · agent {self.model_name or 'agent_json'}"
        )
        # The file is removed whether or not it loaded: a failed source must
        # not leave plaintext keys in the sandbox (or a snapshot of it). A
        # POSIX shell exits outright when `.` fails, hence the exit trap.
        load_keys = (
            f"trap 'rm -f {KEYS_IN_ENV}' EXIT; "
            f"set -a && . {KEYS_IN_ENV} && set +a && rm -f {KEYS_IN_ENV} && "
            if keys
            else ""
        )
        command = load_keys + (
            f"harbor-voice drive {TASK_IN_ENV} --out {CALL_OUT_IN_ENV} "
            f"--trajectory {TRAJECTORY_IN_ENV} "
            f"--agent-json {shlex.quote(agent)}"
            + (f" --seed {seed}" if seed is not None else "")
            + (
                f" --call-config {shlex.quote(json.dumps(call_config))}"
                if call_config
                else ""
            )
        )
        started = time.monotonic()
        relay = _ProgressRelay(self.logger)
        with environment.scoped_output_callback(relay.feed):
            result = await environment.exec(command)
        if not relay.relayed:
            # Backends that cannot stream (Daytona) return the output at the end.
            relay.lines(result.stdout or "")
        self.logger.info(f"call: done in {(time.monotonic() - started) / 60:.1f} min")
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
