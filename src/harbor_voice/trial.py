"""Running one voice trial.

This is what `harbor run` would have dispatched to, had we taken the other
branch. We did not: harbor-voice is its own tool, so a trial runs here and
`src/harbor` is untouched.

**Why standalone rather than a `Trial` subclass.** Inheriting would have cost
roughly ten lines of harbor and bought the job machinery. It would also have
coupled us to a 2,127-line base class whose contract we do not control — an
upstream change to ``_run_agent_phase`` or ``_prepare`` breaks a subclass
silently, and upstream commits daily. Zero harbor edits keeps ``upstream``
merges trivial, which is the only real cost of pulling from it.

What we give up — ``TrialEvent`` hooks, artifact phasing, OTel spans,
harbor-format result recording — is reimplementable and none of it is on the
path to a number.

### The loop is four steps and the agent phase is the only odd one

```
  build     providers, simulator, transport, floor, pipeline
  call      turns, until the content source runs out
  record    an ATIF-voice trajectory
  score     measurement.json / reward.json / validity.json
```

harbor's agent phase launches a process in a sandbox. Ours does not: a voice
agent is reached over a transport, so the call runs **harness-side** and a
container is started only when a task declares tools. That is a real
divergence from harbor's execution model and the reason inheriting was always
going to be an awkward fit.

### Scoring runs here, for now

``test.sh`` is the scoring entry point so a benchmark adapter owns
its ``task.*`` metrics. Nothing declares tools yet, so there is no container
to run it in, and the call metrics are scored in-process — the same
:func:`~harbor_voice.metrics.outputs.build_outputs` the verifier would call.
The first task with a tool sidecar brings the upload path with it; until then
it would be machinery with no consumer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Literal

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
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.providers.registry import default_registry
from harbor_voice.simulator.simulator import UniversalSimulator
from harbor_voice.simulator.simulator import Utterance
from harbor_voice.trajectory.model import VoiceTrajectory

DEFAULT_CALLER = PartyId("caller")
DEFAULT_AGENT = PartyId("agent")


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
            if any(cascade):
                raise TrialConfigError(
                    "pipeline is s2s but cascade slots were also given; an S2S "
                    "model exposes no seam between hearing and reasoning, so "
                    "those slots would be silently ignored"
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


class VoiceTaskConfig(BaseModel):
    """What a voice task declares. Parsed from ``voice.toml`` beside the task.

    Its own file rather than a section of ``task.toml``, because the schema is
    ours: putting it in harbor's model would mean editing harbor to add a
    field, which is exactly what going standalone avoided.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent: AgentSpec
    caller: CallerSpec = Field(default_factory=CallerSpec)
    profile: str = "clean"
    seed: int = 0
    duplex: Duplex = Duplex.HALF
    max_turns: int = Field(default=8, gt=0)
    idle_s: float = Field(default=0.2, gt=0)
    """Quiet period that ends the agent's turn.

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

    @classmethod
    def from_toml(cls, path: Path) -> VoiceTaskConfig:
        """Load ``voice.toml``."""
        import tomllib

        return cls.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


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
        caller: PartyId = DEFAULT_CALLER,
        agent: PartyId = DEFAULT_AGENT,
    ) -> None:
        self._config = config
        self._registry = registry if registry is not None else _registry()
        self._out_dir = out_dir
        self._caller = caller
        self._agent = agent
        self._clock = SessionClock()

    @property
    def config(self) -> VoiceTaskConfig:
        return self._config

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
        spec = self._config.caller
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

    def _content(self) -> Any:
        from harbor_voice.simulator.content import PersonaContent
        from harbor_voice.simulator.content import ScriptContent

        spec = self._config.caller
        if spec.source == "script":
            return ScriptContent(turns=list(spec.turns))
        llm = self._registry.create(
            str(spec.llm), modality=Modality.LLM, **spec.config.get("llm", {})
        )
        return PersonaContent(brief=spec.brief, llm=llm)

    def transport(self) -> Any:
        """The agent under test, as a frame transport."""
        spec = self._config.agent
        if spec.pipeline is Pipeline.S2S:
            return self._registry.create(
                str(spec.s2s),
                modality=Modality.S2S,
                party=self._agent,
                clock=self._clock,
                **spec.config.get("s2s", {}),
            )
        from harbor_voice.cascade import CascadeTransport

        return CascadeTransport(
            stt=self._registry.create(
                str(spec.stt), modality=Modality.STT, **spec.config.get("stt", {})
            ),
            llm=self._registry.create(
                str(spec.llm), modality=Modality.LLM, **spec.config.get("llm", {})
            ),
            tts=self._registry.create(
                str(spec.tts), modality=Modality.TTS, **spec.config.get("tts", {})
            ),
            party=self._agent,
            clock=self._clock,
        )

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

    async def run(self) -> TrialOutcome:
        """Place the call, record it, and score what came back."""
        pipeline = self.pipeline()
        simulator = self.simulator(pipeline)
        transport = self.transport()
        floor = self.floor()

        turns, probes = await self._call(transport, simulator, floor)
        trajectory = self._record(pipeline, simulator, transport, floor, turns, probes)
        return self._score(trajectory, turns)

    async def _call(
        self, transport: Any, simulator: UniversalSimulator, floor: Any
    ) -> tuple[list[Turn], list[Any]]:
        """Run turns until the content source stops or the budget runs out."""
        from harbor_voice.transport.base import SessionConfig

        if self._config.duplex is Duplex.FULL:
            return await self._call_duplex(transport, simulator, floor)

        from harbor_voice.cascade import CascadeAdapter

        adapter = CascadeAdapter(
            transport,
            party=self._agent,
            floor=floor,
            clock=self._clock,
            idle_s=self._config.idle_s,
            turn_timeout_s=self._config.turn_timeout_s,
        )
        await adapter.open(SessionConfig(party=self._agent))
        turns: list[Turn] = []
        try:
            for _ in range(self._config.max_turns):
                utterance = await simulator.next_utterance(_heard(turns))
                if utterance is None or not utterance.frames:
                    break
                turns.append(await adapter.call(utterance.frames))
        finally:
            await adapter.close()
        return turns, []

    async def _call_duplex(
        self, transport: Any, simulator: UniversalSimulator, floor: Any
    ) -> tuple[list[Turn], list[Any]]:
        from harbor_voice.duplex import DuplexDriver
        from harbor_voice.transport.base import SessionConfig

        driver = DuplexDriver(
            transport,
            simulator=simulator,
            party=self._agent,
            floor=floor,
            clock=self._clock,
            idle_s=self._config.idle_s,
            turn_timeout_s=self._config.turn_timeout_s,
        )
        await driver.open(SessionConfig(party=self._agent))
        turns: list[Turn] = []
        probes: list[Any] = []
        try:
            for _ in range(self._config.max_turns):
                utterance = await simulator.next_utterance(_heard(turns))
                if utterance is None or not utterance.frames:
                    break
                exchange = await driver.exchange(utterance)
                turns.append(exchange.as_turn())
                probes.extend(exchange.probes)
        finally:
            await driver.close()
        return turns, probes

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

        # The driver already reported every probe, so the floor must not write
        # a second BARGE_IN for the same interruption.
        recorder.absorb_floor(floor, include_yields=not probes)
        if probes:
            recorder.record_probes(probes, by=self._caller)
        recorder.session_close(self._clock.now())

        return recorder.finish(
            config_matrix=matrix_from(
                floor=floor, pipeline=pipeline, profile=self._config.profile
            ),
            pipeline=pipeline,
            fidelity=simulator.fidelity(),
            render=False,
        )

    def _agent_name(self) -> str:
        spec = self._config.agent
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
        written = (
            tuple(outputs.write(self._out_dir)) if self._out_dir is not None else ()
        )
        return TrialOutcome(
            trajectory=trajectory,
            turns=tuple(turns),
            written=written,
            measurement=outputs.measurement,
            reward=outputs.reward,
        )


def _registry() -> ProviderRegistry:
    """Offline providers, plus AWS when its SDK is importable.

    Degrading rather than raising: a task naming only offline providers must
    run on a laptop with no cloud SDK, and a trial that refuses to start
    because an optional extra is missing is one nobody runs while authoring.
    """
    registry = default_registry()
    try:
        from harbor_voice.providers.aws import register_all

        register_all(registry)
    except Exception:  # noqa: BLE001 - optional extra; the task may not need it
        pass
    return registry


def _heard(turns: Sequence[Turn]) -> list[str]:
    """What the agent has said so far, for a content source that reacts."""
    return [str(t.index) for t in turns]


def _utterance_for(simulator: UniversalSimulator, index: int) -> Utterance | None:
    """The caller's utterance for a turn, skipping probe interjections.

    A probe is the caller speaking, so it lands in ``utterances`` beside the
    content turns; pairing by position without filtering would attribute an
    agent reply to an "mm-hm".
    """
    content = [u for u in simulator.utterances if not u.is_probe]
    return content[index] if index < len(content) else None
