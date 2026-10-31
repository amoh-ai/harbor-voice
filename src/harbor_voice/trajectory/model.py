"""ATIF-voice.

ATIF v1.8 already gives us a great deal: steps, content parts, subagent
trajectories, audio by reference (``AudioSource{media_type, path,
duration_sec}``, so referencing audio by hash costs nothing), and the RL fields
``prompt_token_ids`` / ``completion_token_ids`` / ``logprobs`` on
``Step.metrics``.

What it lacks is the one thing a call needs: a continuous timeline with
overlapping events. ATIF is step-based. Two parties talking over each other
is not a sequence of steps.

How this extends rather than replaces it: every ATIF model sets
``extra="forbid"``, so a subclass carrying new top-level fields would be
*rejected* by a plain ATIF reader. But ``Trajectory.extra`` is a declared
field. So :class:`VoiceTrajectory` is its own typed model, and
:meth:`VoiceTrajectory.to_atif` emits a genuine ATIF ``Trajectory`` with the
voice payload inside ``extra`` — valid ATIF for their consumers, typed voice
data for ours.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from harbor.models.trajectories.agent import Agent
from harbor.models.trajectories.final_metrics import FinalMetrics
from harbor.models.trajectories.metrics import Metrics
from harbor.models.trajectories.step import Step
from harbor.models.trajectories.trajectory import Trajectory
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline

VOICE_SCHEMA_VERSION = "ATIF-v1.8+voice.1"
VOICE_EXTRA_KEY = "voice"


class PartyRole(StrEnum):
    """Why a party is on the call."""

    SYSTEM_UNDER_TEST = "system_under_test"
    SIMULATED_USER = "simulated_user"
    THIRD_PARTY = "third_party"
    """An interpreter, a person in the room, a supervisor."""


class EventType(StrEnum):
    """What happened at a point on the timeline."""

    SESSION_OPEN = "session_open"
    SESSION_CLOSE = "session_close"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    STAGE = "stage"
    OVERLAP = "overlap"
    BARGE_IN = "barge_in"
    FLOOR_VIOLATION = "floor_violation"
    TRANSPORT_GAP = "transport_gap"
    TOOL_CALL = "tool_call"


class BargeInOutcome(StrEnum):
    """Whether an interruption actually landed.

    Recording *whether the perturbation fired* is the most often missing piece
    of metadata in interruption benchmarks: without it, a barge-in metric
    cannot distinguish "the agent handled it" from "it never happened".
    """

    FIRED_AFTER_SPEECH = "fired_after_speech"
    FIRED_BEFORE_SPEECH = "fired_before_speech"
    PENDING_DONE = "pending_done"
    NO_ADAPTER = "no_adapter"


class TimelineEvent(BaseModel):
    """One point on the call's timeline. Overlapping by nature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    t: float = Field(ge=0, description="Seconds since session open.")
    type: EventType
    party: str | None = None
    parties: tuple[str, ...] = ()
    duration_s: float | None = Field(default=None, ge=0)
    meta: dict[str, Any] = Field(default_factory=dict)


class TokenSpan(BaseModel):
    """Which tokens were emitted during a turn's audio.

    The voice-specific problem RL runs into: the timeline is time-indexed
    while turns are step-indexed, and credit assignment needs to know which
    tokens were produced during which audio interval — especially when a
    barge-in cuts a response mid-stream.

    Carried from v1 because it is the only RL affordance that is expensive
    to retrofit: everything else is either already present for other
    reasons or cheap to add later, but this is a schema migration on a format
    other people have adopted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_idx: int = Field(ge=0)
    end_idx: int = Field(ge=0)
    completion_token_ids: tuple[int, ...] = ()
    logprobs: tuple[float, ...] = ()

    @property
    def length(self) -> int:
        return max(0, self.end_idx - self.start_idx)


class AudioRef(BaseModel):
    """Audio by content hash. Never inlined.

    A three-minute call is ~8.6 MB per direction at 24 kHz PCM16. Base64 in
    JSON is a non-starter at a thousand trials.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    digest: str
    duration_s: float = Field(ge=0)
    bytes: int = Field(ge=0)
    media_type: str = "audio/wav"


class Clocks(BaseModel):
    """The four domains, side by side and never reconciled."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    produced: float | None = None
    tx: float | None = None
    rx: float | None = None
    voiced: float | None = None


class StageRecord(BaseModel):
    """One seam crossed, and in whose clock domain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str
    t: float = Field(ge=0)
    clock: str
    meta: dict[str, Any] = Field(default_factory=dict)


class VoiceTurn(BaseModel):
    """One turn, with both sides of said-vs-heard.

    ``said`` is the other party's own words; ``heard`` is what the system
    transcribed. Keeping both is what makes an alignment-free WER possible,
    and the pairing is explicitly *a channel measurement, not human
    transcription truth* — the only WER
    definition that survives perturbation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    idx: int = Field(ge=0)
    party: str
    intent: str | None = None

    said: str | None = None
    heard: str | None = None
    reply: str | None = None
    """The agent's own words in this exchange, when the call recorded them
    (driven calls do). ``heard`` is the agent's transcription of ``said``;
    this is what it said back."""

    audio_clean: AudioRef | None = None
    audio_transmitted: AudioRef | None = None

    clocks: Clocks = Field(default_factory=Clocks)
    stages: tuple[StageRecord, ...] = ()
    token_span: TokenSpan | None = None

    stop_reason: str | None = None
    dropped_frames: int = 0
    bled_frames: int = 0

    @property
    def has_both_sides(self) -> bool:
        """Whether a said-vs-heard comparison is possible for this turn."""
        return self.said is not None and self.heard is not None


class ConfigMatrix(BaseModel):
    """What was selected for this run. A pinned comparison dimension."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    duplex: Duplex
    pipeline: Pipeline
    floor_policy: str
    profile: str = "clean"
    seed_root: int | None = None


class PartyRecord(BaseModel):
    """A participant and what it declared it could do."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    role: PartyRole
    capabilities: Capabilities | None = None
    attested: tuple[str, ...] = ()


class ValidityCheck(BaseModel):
    """One named check on whether the measurement held."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    ok: bool
    detail: str | None = None
    score: float | None = None


class ValidityRecord(BaseModel):
    """Run validity — the quarantine channel.

    A failed run and an *invalid* run are different outcomes. harbor's
    ``aggregate_reward_dicts`` maps a ``None`` reward to ``0``, putting a
    crashed trial in the same bucket as a genuine zero; this exists so we do
    not inherit that.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    measurement_valid: bool = True
    checks: tuple[ValidityCheck, ...] = ()

    @property
    def failed(self) -> tuple[ValidityCheck, ...]:
        return tuple(c for c in self.checks if not c.ok)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(c.detail or c.name for c in self.failed)


class UsageRecord(BaseModel):
    """What one part of a call consumed, in the units it is billed in.

    Tokens, characters and billed seconds only, never dollars: prices change
    and differ by account, a count does not. ``None`` means the part does not
    meter in that unit, not that it used none.
    """

    model_config = ConfigDict(frozen=True)

    component: str
    """What consumed it, e.g. ``agent``, ``caller.llm``, ``caller.tts``."""
    model: str | None = None
    requests: int | None = None
    input_tokens: int | None = None
    """All input tokens, cached and audio included."""
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    """All output tokens, audio included."""
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None
    characters: int | None = None
    """For a TTS billed per character."""
    seconds: float | None = None
    """For a part billed by duration, as the provider reports it (OpenAI
    Live's voice session: ``usage.seconds``)."""


def merge_usage(records: Iterable[UsageRecord]) -> tuple[UsageRecord, ...]:
    """Sum records per (component, model), in first-seen order — e.g. a
    trial's attempts. A count stays ``None`` only if every record lacks it."""
    merged: dict[tuple[str, str | None], UsageRecord] = {}
    counts = (
        "requests",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "input_audio_tokens",
        "output_audio_tokens",
        "characters",
        "seconds",
    )
    for record in records:
        key = (record.component, record.model)
        seen = merged.get(key)
        if seen is None:
            merged[key] = record
            continue
        update: dict[str, int | float | None] = {}
        for name in counts:
            a, b = getattr(seen, name), getattr(record, name)
            update[name] = None if a is None and b is None else (a or 0) + (b or 0)
        merged[key] = seen.model_copy(update=update)
    return tuple(merged.values())


class Artifacts(BaseModel):
    """Where the audio went. Content-addressed, tier-gated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mixed_timeline: AudioRef | None = None
    per_party: dict[str, AudioRef] = Field(default_factory=dict)
    segments: tuple[dict[str, Any], ...] = ()
    tier: str = "T0"


class VoiceTrajectory(BaseModel):
    """A voice call, in ATIF-voice.

    Serialises to a genuine ATIF ``Trajectory`` via :meth:`to_atif`, with this
    payload under ``extra["voice"]``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = VOICE_SCHEMA_VERSION
    trial_id: str | None = None
    task: str | None = None
    task_sha: str | None = None
    agent: str | None = None
    model: str | None = None
    attempt: int = 0

    parties: tuple[PartyRecord, ...] = ()
    config_matrix: ConfigMatrix | None = None
    perturbation_plan: dict[str, Any] = Field(default_factory=dict)

    timeline: tuple[TimelineEvent, ...] = ()
    turns: tuple[VoiceTurn, ...] = ()
    artifacts: Artifacts = Field(default_factory=Artifacts)
    validity: ValidityRecord = Field(default_factory=ValidityRecord)

    simulator_fidelity: dict[str, Any] = Field(default_factory=dict)

    usage: tuple[UsageRecord, ...] = ()
    """What each part of the call consumed: the agent, the caller's LLM, its
    TTS. ATIF's ``final_metrics`` carries the token totals."""

    # ------------------------------------------------------------ queries

    @property
    def duration_s(self) -> float:
        """Wall time from the first event to the last."""
        if not self.timeline:
            return 0.0
        return max(e.t for e in self.timeline) - min(e.t for e in self.timeline)

    def events_of(self, kind: EventType) -> tuple[TimelineEvent, ...]:
        return tuple(e for e in self.timeline if e.type is kind)

    @property
    def is_degraded(self) -> bool:
        """Whether any perturbation layer was active."""
        return bool(self.perturbation_plan.get("layers"))

    @property
    def comparable_on(self) -> dict[str, Any]:
        """The pinned dimensions two runs must share to be compared.

        This is the comparison contract made concrete: a noisy run cannot be
        diffed against a clean one, and the reason is data rather than a
        convention somebody remembered.
        """
        matrix = self.config_matrix
        return {
            "task_sha": self.task_sha,
            "profile": matrix.profile if matrix else None,
            "duplex": matrix.duplex.value if matrix else None,
            "pipeline": matrix.pipeline.value if matrix else None,
            "seed_root": matrix.seed_root if matrix else None,
            "schema_version": self.schema_version,
        }

    # --------------------------------------------------------- conversion

    def to_atif(self) -> Trajectory:
        """Emit a valid ATIF ``Trajectory`` carrying this under ``extra``.

        Turns are mapped onto ATIF ``Step``s rather than left empty, for two
        reasons. ATIF requires at least one step, so an empty list is not a
        valid document. And a turn genuinely *is* two steps — the caller said
        something, the agent replied — so the mapping is faithful rather than
        a formality.

        The token fields land in ``Step.metrics``, which is where ATIF
        already puts ``completion_token_ids`` and ``logprobs``. An RL consumer
        reading plain ATIF finds them in the standard place.
        """
        steps: list[Step] = []
        for turn in self.turns:
            if turn.said is not None:
                steps.append(
                    Step(
                        step_id=len(steps) + 1,  # ATIF step_id is 1-indexed
                        source="user",
                        message=turn.said,
                        extra={"voice_turn": turn.idx, "intent": turn.intent},
                    )
                )
            metrics = None
            if turn.token_span is not None:
                metrics = Metrics(
                    completion_token_ids=list(turn.token_span.completion_token_ids)
                    or None,
                    logprobs=list(turn.token_span.logprobs) or None,
                    extra={
                        "token_span": {
                            "start_idx": turn.token_span.start_idx,
                            "end_idx": turn.token_span.end_idx,
                        }
                    },
                )
            steps.append(
                Step(
                    step_id=len(steps) + 1,  # ATIF step_id is 1-indexed
                    source="agent",
                    message=(
                        turn.reply
                        if turn.reply is not None
                        else turn.heard
                        if turn.heard is not None
                        else ""
                    ),
                    metrics=metrics,
                    extra={"voice_turn": turn.idx, "party": turn.party},
                )
            )

        if not steps:
            # A call that produced no turns is legitimate; an empty ATIF
            # document is not. Say so explicitly rather than fabricating one.
            steps.append(
                Step(
                    step_id=1,
                    source="system",
                    message="voice call produced no turns",
                )
            )

        return Trajectory(
            session_id=self.trial_id,
            agent=Agent(
                name=self.agent or "unknown",
                version="0",
                model_name=self.model,
            ),
            steps=steps,
            final_metrics=self._final_metrics(len(steps)),
            extra={VOICE_EXTRA_KEY: self.model_dump(mode="json")},
        )

    def _final_metrics(self, steps: int) -> FinalMetrics:
        """Token totals over every metered part of the call; no cost.

        A total is ``None`` when no part meters that unit, rather than a zero
        that would claim it was counted.
        """

        def total(field: str) -> int | None:
            counts = [getattr(u, field) for u in self.usage]
            known = [c for c in counts if c is not None]
            return sum(known) if known else None

        return FinalMetrics(
            total_prompt_tokens=total("input_tokens"),
            total_completion_tokens=total("output_tokens"),
            total_cached_tokens=total("cached_input_tokens"),
            total_steps=steps,
            extra={"usage": [u.model_dump(exclude_none=True) for u in self.usage]}
            if self.usage
            else None,
        )

    @classmethod
    def from_atif(cls, trajectory: Trajectory) -> VoiceTrajectory:
        """Recover a voice trajectory from an ATIF document.

        Raises:
            ValueError: if the document carries no voice payload.
        """
        payload = (trajectory.extra or {}).get(VOICE_EXTRA_KEY)
        if payload is None:
            raise ValueError(
                f"ATIF trajectory has no {VOICE_EXTRA_KEY!r} payload; it was "
                f"not produced by harbor-voice"
            )
        return cls.model_validate(payload)
