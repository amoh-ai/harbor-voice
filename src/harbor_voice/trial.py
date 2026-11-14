"""One voice trial, run in this process.

:class:`VoiceTrial` runs a task's ``voice.toml`` one of two ways:

- Driven (``[call]``): the task names a benchmark's call driver, and the
  call is the driver's (:mod:`harbor_voice.driven`). This is what
  ``harbor-voice drive`` runs inside a harbor environment's ``main``; harbor's
  trial, with ``VoiceCallAgent``, supplies the environment, the install and
  the verifier around it. Both benchmark adapters work this way.
- Built-in (``[caller]``): harbor-voice runs the call itself, in four
  steps::

      build     providers, simulator, transport, floor, pipeline
      call      turns, until the content source runs out
      record    an ATIF-voice trajectory
      score     measurement.json / reward.json / validity.json

  The call metrics are scored in-process, with the same
  :func:`~harbor_voice.metrics.outputs.build_outputs` a verifier would call.
  This path runs from Python only; harbor's runner needs a ``[call]`` driver.
"""

from __future__ import annotations

import contextlib

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Literal
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from harbor_voice.adapter import Turn
from harbor_voice.audio.clock import SessionClock
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.ids import PartyId
from harbor_voice.perturbation.base import SeedSet
from harbor_voice.perturbation.plan import PerturbationPipeline
from harbor_voice.perturbation.profiles import get_profile
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.credentials import resolve as resolve_credentials
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.providers.registry import Registration
from harbor_voice.providers.registry import credential_slug
from harbor_voice.providers.registry import provider_registry
from harbor_voice.providers.tools import ToolExecutor
from harbor_voice.providers.tools import ToolRouter
from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.config import VerifierConfig
from harbor_voice.driver import VerifierResult
from harbor_voice.simulator.simulator import UniversalSimulator
from harbor_voice.simulator.simulator import Utterance
from harbor_voice.trajectory.model import VoiceTrajectory

if TYPE_CHECKING:
    from harbor_voice.trajectory.recorder import Recorder

DEFAULT_CALLER = PartyId("caller")
DEFAULT_AGENT = PartyId("agent")
RECORDING = "call.wav"
"""The built-in loop's recording, written beside the trajectory in ``out_dir``."""


class TrialConfigError(ValueError):
    """A voice task is configured in a way that cannot produce a measurement.

    Note:
        Raised inside pydantic validators, so pydantic re-raises it wrapped in
        ``pydantic.ValidationError``. Catch that; the message is preserved.
        Same shape as :class:`~harbor_voice.capabilities.CapabilityError`.
    """


class AgentSpec(BaseModel):
    """The system under test — a cascade's three slots, or one S2S model.

    Exactly one shape, enforced here rather than discovered at connect time.
    A half-filled cascade that falls back to a default STT would produce a
    run attributing someone else's recogniser to the agent under test.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pipeline: Pipeline
    s2s: str | None = None
    stt: str | None = None
    llm: str | None = None
    tts: str | None = None
    omni: str | None = None
    """An omni model (audio in, text out) in place of ``stt`` and ``llm``: a
    cascade of two slots, ``omni`` and ``tts``."""
    config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per-slot provider settings, keyed by slot name.

    A spec string names *which* provider; most need settings as well — a
    region, a temperature, a voice, or for the offline set, the script it
    reads back. Keeping them here rather than folding them into the spec
    string means a task can say ``stt = "aws.voxtral"`` and nothing else when
    that is genuinely all it needs.
    """

    @model_validator(mode="after")
    def _one_shape(self) -> AgentSpec:
        cascade = (self.stt, self.llm, self.tts)
        if self.pipeline is Pipeline.S2S:
            if self.s2s is None:
                raise TrialConfigError("pipeline is s2s but no s2s provider was given")
            if any(cascade) or self.omni is not None:
                raise TrialConfigError(
                    "pipeline is s2s but cascade slots were also given; an S2S "
                    "model exposes no seam between hearing and reasoning, so "
                    "those slots would be silently ignored"
                )
            return self
        if self.omni is not None:
            if self.stt is not None or self.llm is not None:
                raise TrialConfigError(
                    "an omni cascade hears and answers in one model; stt and llm "
                    "would be silently ignored, so give omni and tts only"
                )
            if self.tts is None:
                raise TrialConfigError("an omni cascade needs a tts to speak its reply")
            if self.s2s is not None:
                raise TrialConfigError(
                    "pipeline is cascade but an s2s provider was given"
                )
            return self
        missing = [
            name
            for name, value in zip(("stt", "llm", "tts"), cascade, strict=True)
            if value is None
        ]
        if missing:
            raise TrialConfigError(
                f"pipeline is cascade but {missing} were not given; a default "
                f"would attribute someone else's component to the agent"
            )
        if self.s2s is not None:
            raise TrialConfigError("pipeline is cascade but an s2s provider was given")
        return self


class CallerSpec(BaseModel):
    """The simulated caller — its voice and where its words come from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tts: str = "offline.tone-tts"
    source: Literal["script", "persona", "corpus"] = "script"
    turns: tuple[str, ...] = ()
    brief: str = ""
    llm: str | None = None
    stt: str | None = None
    """How a persona caller hears the agent: an STT that transcribes each
    reply's audio. Without one, a cascade agent's own words (what its TTS
    was given) are used; a speech-to-speech agent's words exist only as
    audio, so a persona caller talking to one needs this."""
    config: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _source_has_what_it_needs(self) -> CallerSpec:
        if self.source == "script" and not self.turns:
            raise TrialConfigError("a script caller needs turns")
        if self.source == "persona" and not (self.brief and self.llm):
            raise TrialConfigError("a persona caller needs a brief and an llm")
        if self.source == "corpus":
            raise TrialConfigError(
                "corpus callers are not wired yet; they stay a "
                "first-class source, and the loader lands with the first "
                "benchmark that ships recordings"
            )
        return self


class CallSpec(BaseModel):
    """``[call]``: a benchmark's own call loop instead of the built-in one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver: str
    """``"package.module:function"`` - see :mod:`harbor_voice.driver`."""
    config: dict[str, Any] = Field(default_factory=dict)
    """Passed to the driver untouched."""
    timeout_sec: float = Field(default=3600.0, gt=0)
    """The call phase's limit — harbor's ``[agent].timeout_sec`` for a voice call."""


class VoiceTaskConfig(BaseModel):
    """What a voice task declares. Parsed from ``voice.toml`` beside the task.

    Its own file rather than a section of ``task.toml``, because the schema is
    ours: putting it in harbor's model would mean editing harbor to add a
    field, which is exactly what going standalone avoided.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent: AgentSpec | None = None
    """The agent under test. Optional, as in harbor, where the task never names
    its agent: the run supplies it (``-m`` on ``harbor-voice trials start``, a
    job's ``agents:``). A task may keep one as a default for convenience."""
    caller: CallerSpec | None = None
    """Required unless ``[call]`` names a driver, which brings its own caller."""
    profile: str = "clean"
    seed: int = 0
    duplex: Duplex = Duplex.HALF
    max_turns: int = Field(default=8, gt=0)
    idle_s: float | None = Field(default=None, gt=0)
    """Quiet period that ends the agent's turn. Unset, it is 0.2 s on a
    half-duplex call and 1.0 s on a full-duplex one, where the agent streams at
    speaking pace and pauses between sentences.

    Tunable because it is a property of the agent, not of the harness: a
    model that pauses mid-sentence needs a longer one, and a run that cuts
    its replies short reports a truncation as a short answer.
    """

    turn_timeout_s: float = Field(default=30.0, gt=0)
    reference_stt: str | None = None
    metrics: tuple[str, ...] | None = None
    task: str | None = None
    task_sha: str | None = None
    trial_id: str | None = None

    call: CallSpec | None = None
    environment: EnvironmentConfig | None = None
    """Harbor's ``EnvironmentConfig`` — the task's ``environment/`` folder is
    built and run by harbor's environment layer, on any of its backends."""
    verifier: VerifierConfig | None = None
    """Harbor's ``VerifierConfig``; harbor runs ``tests/test.sh`` in ``main``."""

    @model_validator(mode="after")
    def _caller_unless_driven(self) -> VoiceTaskConfig:
        if self.caller is None and self.call is None:
            raise TrialConfigError(
                "a task needs a [caller], unless [call] names a driver that "
                "brings its own"
            )
        return self

    @model_validator(mode="after")
    def _verifier_needs_an_environment(self) -> VoiceTaskConfig:
        if self.verifier is not None and self.environment is None:
            raise TrialConfigError(
                "a [verifier] runs inside the task's environment; declare one"
            )
        return self

    def with_seed(self, seed: int) -> VoiceTaskConfig:
        """This task under another seed — the task's and its driver's.

        A job runs one task under several seeds (tau2 gives trial k its own);
        the driver reads ``[call.config].seed`` when it has one.
        """
        update: dict[str, Any] = {"seed": seed}
        if self.call is not None and "seed" in self.call.config:
            update["call"] = self.call.model_copy(
                update={"config": {**self.call.config, "seed": seed}}
            )
        return self.model_copy(update=update)

    def with_call_config(self, overrides: Mapping[str, Any]) -> VoiceTaskConfig:
        """This task with its driver's settings overridden for one run.

        A job sets what the task leaves to the run, the same for every task
        it runs (tau-voice: whether a cascade agent may take the call).
        """
        if self.call is None:
            raise TrialConfigError("call settings given, but the task names no [call]")
        call = self.call.model_copy(
            update={"config": {**self.call.config, **overrides}}
        )
        return self.model_copy(update={"call": call})

    @classmethod
    def from_toml(cls, path: Path) -> VoiceTaskConfig:
        """Load ``voice.toml``."""
        import tomllib

        return cls.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class DrivenOutcome:
    """A trial whose call a benchmark's driver ran (``[call]``).

    The verifier's reward and the driver's record. The driver's ATIF-voice
    trajectory, when it returns one, is written to ``trajectory.json`` by
    :func:`~harbor_voice.driven.drive_here`, not held here.
    """

    reward: dict[str, float] | None
    verified: VerifierResult | None
    summary: dict[str, Any]
    written: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """What one trial produced."""

    trajectory: VoiceTrajectory
    turns: tuple[Turn, ...]
    written: tuple[Path, ...] = ()
    measurement: dict[str, Any] = field(default_factory=dict)
    reward: dict[str, float] | None = None

    @property
    def measurement_valid(self) -> bool:
        return self.trajectory.validity.measurement_valid


class VoiceTrial:
    """Build, call, record, score. One task, one agent, one trajectory.

    Args:
        config: What the task declared.
        registry: Providers. Defaults to the offline set plus AWS when its
            SDK is importable, so a task naming only offline providers runs
            with no credentials at all.
        out_dir: Where the three documents land. Nothing is written without
            one — a trial that only wants the trajectory should not have to
            clean up after itself.
    """

    def __init__(
        self,
        config: VoiceTaskConfig,
        *,
        registry: ProviderRegistry | None = None,
        out_dir: Path | None = None,
        tools: ToolExecutor | None = None,
        caller: PartyId = DEFAULT_CALLER,
        agent: PartyId = DEFAULT_AGENT,
        task_dir: Path | None = None,
        agent_spec: AgentSpec | None = None,
    ) -> None:
        self._config = config
        self._task_dir = task_dir
        self._agent_spec_override = agent_spec
        self._registry = registry if registry is not None else provider_registry()
        self._hearing: Any = None
        self._heard_lines: list[str] = []
        self._out_dir = out_dir
        self._caller = caller
        self._agent = agent
        self._clock = SessionClock()
        self._router = (
            None if tools is None else ToolRouter(tools, world=self.pipeline().world)
        )

    @property
    def config(self) -> VoiceTaskConfig:
        return self._config

    @property
    def tools(self) -> ToolRouter | None:
        """The tool router, when the task supplied an executor.

        Supplied by the caller rather than named in ``voice.toml``: a
        benchmark's tools are its own, and an import-path string in
        a config file would be a second, weaker way to say what an adapter
        already says in Python.
        """
        return self._router

    # --------------------------------------------------------------- build

    def pipeline(self) -> PerturbationPipeline:
        """The profile's layers, seeded from the task's root seed."""
        return get_profile(self._config.profile).pipeline(
            SeedSet.derive(self._config.seed)
        )

    def simulator(self, pipeline: PerturbationPipeline) -> UniversalSimulator:
        """The caller. Shares this trial's clock, which is not optional.

        Two ``SessionClock`` instances start at different instants, so a
        driver with its own reads a smaller ``now()`` than the simulator did
        when it stamped ``t_produced``, and the frame's monotonic guard then
        rejects a transmit that looks like it preceded its own synthesis.
        """
        spec = self._caller_spec()
        return UniversalSimulator(
            party=self._caller,
            content=self._content(),
            tts=self._registry.create(
                spec.tts, modality=Modality.TTS, **spec.config.get("tts", {})
            ),
            pipeline=pipeline,
            seeds=SeedSet.derive(self._config.seed),
            clock=self._clock,
        )

    def _caller_spec(self) -> CallerSpec:
        if self._config.caller is None:
            raise TrialConfigError(
                "this task has no [caller]; its driver runs the call"
            )
        return self._config.caller

    def _content(self) -> Any:
        from harbor_voice.simulator.content import PersonaContent
        from harbor_voice.simulator.content import ScriptContent

        spec = self._caller_spec()
        if spec.source == "script":
            return ScriptContent(turns=list(spec.turns))
        llm = self._registry.create(
            str(spec.llm), modality=Modality.LLM, **spec.config.get("llm", {})
        )
        return PersonaContent(brief=spec.brief, llm=llm)

    def transport(self) -> Any:
        """The agent under test, as a frame transport."""
        spec = self.agent_spec()
        if spec.pipeline is Pipeline.S2S:
            return self._registry.create(
                str(spec.s2s),
                modality=Modality.S2S,
                party=self._agent,
                clock=self._clock,
                tools=self._router,
                **spec.config.get("s2s", {}),
            )
        if spec.omni is not None:
            from harbor_voice.cascade.omni import OmniCascadeTransport

            return OmniCascadeTransport(
                omni=self._registry.create(
                    spec.omni, modality=Modality.OMNI, **spec.config.get("omni", {})
                ),
                tts=self._registry.create(
                    str(spec.tts), modality=Modality.TTS, **spec.config.get("tts", {})
                ),
                tools=self._router,
                party=self._agent,
                clock=self._clock,
            )
        from harbor_voice.cascade import CascadeTransport

        return CascadeTransport(
            tools=self._router,
            party=self._agent,
            clock=self._clock,
            **self._slots(spec),
        )

    def _slots(self, spec: AgentSpec) -> dict[str, Any]:
        """A cascade's three providers, each with its own settings."""
        return {
            slot: self._registry.create(
                str(getattr(spec, slot)), modality=modality, **spec.config.get(slot, {})
            )
            for slot, modality in (
                ("stt", Modality.STT),
                ("llm", Modality.LLM),
                ("tts", Modality.TTS),
            )
        }

    def floor(self) -> Any:
        """Half-duplex queues; full duplex measures who yielded.

        A full-duplex trial gets a :class:`~harbor_voice.floor.BargeInFloor`
        rather than a plain one, because a run that permits overlap and then
        cannot say whether the agent stopped has only made the audio harder.
        """
        from harbor_voice.floor import BargeInFloor
        from harbor_voice.floor import HalfDuplexFloor

        if self._config.duplex is Duplex.FULL:
            return BargeInFloor()
        return HalfDuplexFloor()

    # ----------------------------------------------------------------- run

    async def run(self) -> TrialOutcome | DrivenOutcome:
        """Place the call, record it, and score what came back."""
        if self._config.call is not None:
            return await self._run_driven(self._config.call)
        pipeline = self.pipeline()
        simulator = self.simulator(pipeline)
        transport = self.transport()
        floor = self.floor()

        try:
            turns, probes = await self._call(transport, simulator, floor)
        finally:
            await simulator.aclose()
            if (close := getattr(self._hearing, "aclose", None)) is not None:
                with contextlib.suppress(Exception):
                    await close()
            self._hearing = None
        trajectory = self._record(pipeline, simulator, transport, floor, turns, probes)
        return self._score(trajectory, turns)

    async def _agent_words(self, transport: Any, turns: Sequence[Turn]) -> list[str]:
        """What the agent has said so far, one line per turn, for a caller
        that reacts to it. A scripted caller reads nothing."""
        caller = self._config.caller
        if caller is None or caller.source != "persona":
            return []
        if caller.stt is not None:
            if self._hearing is None:
                self._hearing = self._registry.create(
                    caller.stt, modality=Modality.STT, **caller.config.get("stt", {})
                )
            while len(self._heard_lines) < len(turns):
                turn = turns[len(self._heard_lines)]
                # A half-duplex turn keeps the agent's frames in its TTS's
                # format; the STT expects the recording format.
                frames = [_canonical(frame) for frame in turn.received]
                text = ""
                async for partial in self._hearing.transcribe(frames):
                    if partial.is_final:
                        text = partial.text
                self._heard_lines.append(text)
            return list(self._heard_lines)
        said = getattr(transport, "said", None)
        if said is None:
            raise TrialConfigError(
                "a persona caller cannot hear this agent: a speech-to-speech "
                "agent's words exist only as audio, so set [caller] stt to "
                "transcribe them"
            )
        return list(said)

    async def _call(
        self, transport: Any, simulator: UniversalSimulator, floor: Any
    ) -> tuple[list[Turn], list[Any]]:
        """Run turns until the content source stops or the budget runs out."""
        from harbor_voice.transport.base import SessionConfig

        if self._config.duplex is Duplex.FULL:
            return await self._call_duplex(transport, simulator, floor)

        from harbor_voice.cascade import CascadeAdapter
        from harbor_voice.transport.drain import DEFAULT_IDLE_S

        adapter = CascadeAdapter(
            transport,
            party=self._agent,
            floor=floor,
            clock=self._clock,
            idle_s=self._config.idle_s or DEFAULT_IDLE_S,
            turn_timeout_s=self._config.turn_timeout_s,
        )
        await adapter.open(SessionConfig(party=self._agent))
        turns: list[Turn] = []
        try:
            for _ in range(self._config.max_turns):
                utterance = await simulator.next_utterance(
                    await self._agent_words(transport, turns)
                )
                if utterance is None or not utterance.frames:
                    break
                turns.append(await adapter.call(utterance.frames))
        finally:
            await adapter.close()
        return turns, []

    async def _call_duplex(
        self, transport: Any, simulator: UniversalSimulator, floor: Any
    ) -> tuple[list[Turn], list[Any]]:
        from harbor_voice.duplex import DUPLEX_IDLE_S
        from harbor_voice.duplex import DuplexDriver
        from harbor_voice.transport.base import SessionConfig

        driver = DuplexDriver(
            transport,
            simulator=simulator,
            party=self._agent,
            floor=floor,
            clock=self._clock,
            idle_s=self._config.idle_s or DUPLEX_IDLE_S,
            turn_timeout_s=self._config.turn_timeout_s,
        )
        await driver.open(SessionConfig(party=self._agent))
        turns: list[Turn] = []
        probes: list[Any] = []
        try:
            for _ in range(self._config.max_turns):
                utterance = await simulator.next_utterance(
                    await self._agent_words(transport, turns)
                )
                if utterance is None or not utterance.frames:
                    break
                exchange = await driver.exchange(utterance)
                turns.append(exchange.as_turn())
                probes.extend(exchange.probes)
        finally:
            await driver.close()
        return turns, probes

    def agent_spec(self) -> AgentSpec:
        """The agent under test: the run's choice, else the task's default."""
        spec = self._agent_spec_override or self._config.agent
        if spec is None:
            raise TrialConfigError(
                "no agent under test: pass one for this run (-m), as a harbor "
                "job names its agents; the task does not"
            )
        return spec

    async def _run_driven(self, call: CallSpec) -> DrivenOutcome:
        """A benchmark's own call loop, here.

        Inside an environment this is what ``harbor-voice drive`` runs in
        ``main``; harbor's trial (with ``VoiceCallAgent``) does the environment,
        the install and the verifier around it.
        """
        import json

        from harbor_voice import driven

        if self._task_dir is None or self._out_dir is None:
            raise TrialConfigError("a driven trial needs task_dir and out_dir")
        out_dir = self._out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        agent = self.agent_spec()

        outcome = await driven.drive_here(
            self._config,
            task_dir=self._task_dir,
            out_dir=out_dir,
            build_agent=self.build_agent,
            agent=agent,
        )
        verified = outcome.verified
        del call

        written: list[Path] = []
        reward = None
        if verified is not None and verified.reward is not None:
            reward = {"reward": verified.reward}
            path = out_dir / "reward.json"
            path.write_text(json.dumps(reward), encoding="utf-8")
            written.append(path)
        path = out_dir / "trial.json"
        path.write_text(
            json.dumps(
                {
                    "task": self._config.task,
                    "trial_id": self._config.trial_id,
                    "driver": self._config.call.driver if self._config.call else None,
                    "agent": agent.model_dump(mode="json"),
                    "reward": reward,
                    "verifier": dict(verified.result) if verified else None,
                    "summary": outcome.summary,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        written.append(path)
        return DrivenOutcome(
            reward=reward,
            verified=verified,
            summary=outcome.summary,
            written=tuple(written),
        )

    def build_agent(self, *, tools: ToolRouter | None = None, **overrides: Any) -> Any:
        """The agent under test, for a driver.

        ``overrides`` win over the agent's configured settings: a benchmark
        supplies its own system prompt, and that is the benchmark's call.

        A cascade is built for a full-duplex call
        (:class:`~harbor_voice.cascade.DuplexCascadeTransport`) and takes only
        a ``system_prompt``; any other override is a speech-to-speech
        provider's setting, and a driver that passes one is refused rather
        than given an agent that ignores it. Its STT slot must be
        ``deepgram.stt``, which finds the turns; another is refused.
        """
        spec = self.agent_spec()
        if spec.omni is not None:
            raise TrialConfigError(
                "an omni cascade runs on harbor-voice's own call loop only; a "
                "driven benchmark's cascade needs deepgram.stt to find the turns "
                "and an LLM that holds the conversation and its tools"
            )
        if spec.pipeline is Pipeline.CASCADE:
            from harbor_voice.cascade import DuplexCascadeTransport
            from harbor_voice.cascade.duplex import stt_refusal

            if (refusal := stt_refusal(str(spec.stt))) is not None:
                raise TrialConfigError(refusal)

            unknown = sorted(set(overrides) - {"system_prompt"})
            if unknown:
                raise TrialConfigError(
                    f"a cascade agent takes only a system_prompt from a driver; "
                    f"this driver also sets {unknown}, which only a "
                    f"speech-to-speech provider reads"
                )
            return DuplexCascadeTransport(
                tools=tools,
                party=self._agent,
                clock=self._clock,
                system_prompt=overrides.get("system_prompt", ""),
                **self._slots(spec),
            )
        return self._registry.create(
            str(spec.s2s),
            modality=Modality.S2S,
            party=self._agent,
            clock=self._clock,
            tools=tools,
            **{**spec.config.get("s2s", {}), **overrides},
        )

    # -------------------------------------------------------------- record

    def _record(
        self,
        pipeline: PerturbationPipeline,
        simulator: UniversalSimulator,
        transport: Any,
        floor: Any,
        turns: Sequence[Turn],
        probes: Sequence[Any],
    ) -> VoiceTrajectory:
        # From the module, not the package: `trajectory/__init__` resolves
        # these through a PEP 562 `__getattr__` that keeps the scoring import
        # cone narrow, and a lazy attribute is opaque to the type
        # checker. Importing inside the function is just as lazy.
        from harbor_voice.trajectory.recorder import Recorder
        from harbor_voice.trajectory.recorder import matrix_from
        from harbor_voice.trajectory.model import PartyRecord
        from harbor_voice.trajectory.model import PartyRole

        recorder = Recorder(
            trial_id=self._config.trial_id,
            task=self._config.task,
            task_sha=self._config.task_sha,
            agent=self._agent_name(),
        )
        recorder.add_party(
            PartyRecord(
                id=str(self._agent),
                role=PartyRole.SYSTEM_UNDER_TEST,
                capabilities=transport.capabilities,
            )
        )
        recorder.add_party(
            PartyRecord(
                id=str(self._caller),
                role=PartyRole.SIMULATED_USER,
                capabilities=simulator.capabilities,
            )
        )
        recorder.session_open(0.0)

        heard = list(getattr(transport, "heard", []))
        for index, turn in enumerate(turns):
            recorder.record_turn(
                turn,
                utterance=_utterance_for(simulator, index),
                heard=heard[index] if index < len(heard) else None,
            )

        # Tool calls, with the L6 latency separated from the tool's own, so a
        # trajectory can tell a slow backend from one we made slow.
        if self._router is not None:
            for call in self._router.records:
                recorder.tool_call(
                    self._clock.now(),
                    call.name,
                    injected_latency_ms=call.injected_latency_ms,
                )

        # The driver already reported every probe, so the floor must not write
        # a second BARGE_IN for the same interruption.
        recorder.absorb_floor(floor, include_yields=not probes)
        if probes:
            recorder.record_probes(probes, by=self._caller)
        recorder.session_close(self._clock.now())

        trajectory = recorder.finish(
            config_matrix=matrix_from(
                floor=floor,
                pipeline=pipeline,
                agent_pipeline=self.agent_spec().pipeline,
                profile=self._config.profile,
            ),
            pipeline=pipeline,
            fidelity=simulator.fidelity(),
            render=False,
        )
        if self._out_dir is None:
            return trajectory
        return self._save_recording(recorder, trajectory)

    def _save_recording(
        self, recorder: Recorder, trajectory: VoiceTrajectory
    ) -> VoiceTrajectory:
        """Write the call as heard, both parties on one timeline, beside the
        trajectory, and name it there by the file's hash, as the adapters name
        theirs. ``harbor-voice listen`` finds it by that hash."""
        from harbor_voice.trajectory.model import Artifacts
        from harbor_voice.trajectory.render import to_wav
        from harbor_voice.trajectory.stepped import audio_ref

        mixed = recorder.render()
        if not mixed.pcm:
            return trajectory
        assert self._out_dir is not None
        path = self._out_dir / RECORDING
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(to_wav(mixed.pcm, mixed.fmt))
        artifacts = Artifacts(
            mixed_timeline=audio_ref(path), segments=recorder.segments(), tier="T1"
        )
        return trajectory.model_copy(update={"artifacts": artifacts})

    def _agent_name(self) -> str:
        spec = self.agent_spec()
        if spec.pipeline is Pipeline.S2S:
            return str(spec.s2s)
        return f"{spec.stt}+{spec.llm}+{spec.tts}"

    # --------------------------------------------------------------- score

    def _score(
        self, trajectory: VoiceTrajectory, turns: Sequence[Turn]
    ) -> TrialOutcome:
        from harbor_voice.metrics import REGISTRY
        from harbor_voice.metrics import MetricContext
        from harbor_voice.metrics import build_outputs

        selected = list(self._config.metrics or REGISTRY.names())
        results = REGISTRY.evaluate(selected, MetricContext(trajectory=trajectory))
        outputs = build_outputs(
            results,
            trajectory.validity,
            trial_id=trajectory.trial_id,
            task=trajectory.task,
            agent=trajectory.agent,
            config_matrix=(
                trajectory.config_matrix.model_dump(mode="json")
                if trajectory.config_matrix
                else {}
            ),
            comparable_on=trajectory.comparable_on,
        )
        written: tuple[Path, ...] = ()
        if self._out_dir is not None:
            written = tuple(outputs.write(self._out_dir))
            if trajectory.artifacts is not None:  # the recording was written
                written += (self._out_dir / RECORDING,)
        return TrialOutcome(
            trajectory=trajectory,
            turns=tuple(turns),
            written=written,
            measurement=outputs.measurement,
            reward=outputs.reward,
        )


def required_extras(
    spec: AgentSpec, registry: ProviderRegistry | None = None
) -> list[str]:
    """harbor-voice extras the agent's providers declare (``requires_extra``)."""
    extras = [
        entry.spec.requires_extra
        for entry, _ in _resolved(spec, registry)
        if entry.spec.requires_extra
    ]
    return list(dict.fromkeys(extras))


def required_env(
    spec: AgentSpec, registry: ProviderRegistry | None = None
) -> dict[str, str]:
    """The credentials the agent's providers need, from this process's environment.

    A task's ``[environment] env`` names the keys its own caller and grader
    use (OpenAI, ElevenLabs, AWS). The agent under test can be any provider,
    so its keys are passed into the call here: Gemini Live needs
    ``GEMINI_API_KEY`` and Grok ``XAI_API_KEY``, which no task names. Unset
    variables are left out; the provider says what is missing when it opens.
    Variables that name something on this machine (:data:`HOST_ONLY_ENVS`)
    are left out too, and each provider's ``sandbox_env`` has the last word.
    """
    env: dict[str, str] = {}
    for entry, model in _resolved(spec, registry):
        found = resolve_credentials(credential_slug(entry.spec, model)).container_env()
        if entry.spec.sandbox_env:
            found = entry.spec.sandbox_env(found)
        env.update(found)
    return {k: v for k, v in env.items() if k not in HOST_ONLY_ENVS}


HOST_ONLY_ENVS = frozenset(
    {
        "AWS_PROFILE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
)
"""Variables naming a profile or a file on this machine, meaningless in a
sandbox: an ``AWS_PROFILE`` there fails with ProfileNotFound even when the
exported keys are present."""


def _canonical(frame: Any) -> Any:
    """``frame`` in :data:`~harbor_voice.audio.format.CANONICAL`."""
    from dataclasses import replace

    from harbor_voice.audio.convert import convert
    from harbor_voice.audio.format import CANONICAL

    if frame.fmt == CANONICAL:
        return frame
    return replace(frame, pcm=convert(frame.pcm, frame.fmt, CANONICAL), fmt=CANONICAL)


def _resolved(
    spec: AgentSpec, registry: ProviderRegistry | None = None
) -> list[tuple[Registration, str]]:
    """Each provider the agent names: its registration and the model.

    A provider the registry cannot resolve here is skipped: it needs nothing
    named, and building it will say what is missing.
    """
    registry = registry or provider_registry()
    found: list[tuple[Registration, str]] = []
    for name in (spec.s2s, spec.stt, spec.llm, spec.omni, spec.tts):
        if not name:
            continue
        try:
            found.append(registry.resolve(name))
        except Exception:  # noqa: BLE001 - see the docstring
            continue
    return found


def _utterance_for(simulator: UniversalSimulator, index: int) -> Utterance | None:
    """The caller's utterance for a turn, skipping probe interjections.

    A probe is the caller speaking, so it lands in ``utterances`` beside the
    content turns; pairing by position without filtering would attribute an
    agent reply to an "mm-hm".
    """
    content = [u for u in simulator.utterances if not u.is_probe]
    return content[index] if index < len(content) else None
