"""Assembling an ATIF-voice trajectory from a running call.

The recorder is a passive observer. It is handed events and turns and builds
the record; it never drives the call and never decides whether the run was
good. That split is deliberate: everything downstream of here reads the
trajectory rather than the live run, which is what buys harbor's
``regrade`` for voice — fix a metric definition and re-score every historical
trial for the cost of CPU.

The validity checks it runs are the ones that can only be known while the call
is happening: whether the clean twin survived, whether the simulator stayed in
persona, whether the transport was down too long. A failed check makes the run
``measurement_invalid`` — not a failed trial, because the agent was not
given a fair run and the result is not evidence about it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from harbor_voice.adapter import Turn
from harbor_voice.audio.clock import Clock
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.floor import BaseFloor
from harbor_voice.floor import BargeInFloor
from harbor_voice.ids import Digest
from harbor_voice.ids import PartyId
from harbor_voice.ids import sha256_digest
from harbor_voice.perturbation.conversation import Landing
from harbor_voice.perturbation.conversation import ProbeResult
from harbor_voice.perturbation.plan import CleanReferenceError
from harbor_voice.perturbation.plan import PerturbationPipeline
from harbor_voice.simulator.simulator import SimulatorFidelity
from harbor_voice.simulator.simulator import Utterance
from harbor_voice.trajectory.model import AudioRef
from harbor_voice.trajectory.model import Artifacts
from harbor_voice.trajectory.model import BargeInOutcome
from harbor_voice.trajectory.model import Clocks
from harbor_voice.trajectory.model import ConfigMatrix
from harbor_voice.trajectory.model import EventType
from harbor_voice.trajectory.model import PartyRecord
from harbor_voice.trajectory.model import StageRecord
from harbor_voice.trajectory.model import TimelineEvent
from harbor_voice.trajectory.model import TokenSpan
from harbor_voice.trajectory.model import ValidityCheck
from harbor_voice.trajectory.model import ValidityRecord
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.model import VoiceTurn
from harbor_voice.trajectory.render import PlacedAudio
from harbor_voice.trajectory.render import RenderResult
from harbor_voice.trajectory.render import render_timeline

DEFAULT_PERSONA_FLOOR = 0.85


@dataclass(slots=True)
class RecorderConfig:
    """Thresholds that decide whether a run counts as evidence."""

    require_in_persona: float | None = DEFAULT_PERSONA_FLOOR
    max_transport_gap_s: float = 1.0
    require_clean_reference: bool = True
    fmt: AudioFormat = CANONICAL


class Recorder:
    """Builds an ATIF-voice trajectory while a call runs."""

    def __init__(
        self,
        *,
        trial_id: str | None = None,
        task: str | None = None,
        task_sha: str | None = None,
        agent: str | None = None,
        model: str | None = None,
        config: RecorderConfig | None = None,
    ) -> None:
        self._trial_id = trial_id
        self._task = task
        self._task_sha = task_sha
        self._agent = agent
        self._model = model
        self._config = config or RecorderConfig()

        self._parties: list[PartyRecord] = []
        self._events: list[TimelineEvent] = []
        self._turns: list[VoiceTurn] = []
        self._placed: list[PlacedAudio] = []
        self._checks: list[ValidityCheck] = []
        self._gap_total_s = 0.0
        self._token_cursor = 0
        self._referenced: set[Digest] = set()
        self._unreferenced = 0
        self._saw_perturbed = False

    # ------------------------------------------------------------- setup

    def add_party(self, record: PartyRecord) -> None:
        self._parties.append(record)

    def session_open(self, t: float = 0.0) -> None:
        self._event(EventType.SESSION_OPEN, t)

    def session_close(self, t: float) -> None:
        self._event(EventType.SESSION_CLOSE, t)

    # ------------------------------------------------------------ events

    def _event(
        self,
        kind: EventType,
        t: float,
        *,
        party: str | None = None,
        parties: tuple[str, ...] = (),
        duration_s: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Append one timeline event. Explicitly typed rather than ``**kwargs``,
        so a mistyped field is a type error here instead of a validation error
        at the end of a six-hour run."""
        self._events.append(
            TimelineEvent(
                t=max(0.0, t),
                type=kind,
                party=party,
                parties=parties,
                duration_s=duration_s,
                meta=meta or {},
            )
        )

    def transport_gap(self, t: float, duration_s: float) -> None:
        """Record a connection drop. Budgeted against validity."""
        self._gap_total_s += duration_s
        self._event(EventType.TRANSPORT_GAP, t, duration_s=duration_s)

    def tool_call(
        self, t: float, name: str, *, injected_latency_ms: float | None = None
    ) -> None:
        """Record a tool invocation, including any L6 latency injected."""
        meta: dict[str, Any] = {"name": name}
        if injected_latency_ms is not None:
            meta["injected_latency_ms"] = injected_latency_ms
        self._event(EventType.TOOL_CALL, t, meta=meta)

    def barge_in(
        self,
        t: float,
        *,
        by: PartyId,
        outcome: BargeInOutcome,
        native: bool = False,
        yielded_in_time: bool | None = None,
        yield_latency_s: float | None = None,
        deadline_s: float | None = None,
    ) -> None:
        """Record an interruption and whether it landed.

        ``yielded_in_time=None`` means the holder never stopped, not that
        nobody looked: ``turns.barge_in_success_rate`` reads an absent key as
        *never*, which is why the key is written whenever it is known and
        omitted only when it genuinely is not.

        ``yield_latency_s`` is in reply-audio seconds — how much more of
        the agent's speech the caller heard after interrupting. It is a
        duration inside one domain rather than a difference between two
        clocks, which is what makes it comparable between a real-time call and
        a harness that streams faster.
        """
        meta: dict[str, Any] = {"outcome": outcome.value, "native": native}
        if yielded_in_time is not None or yield_latency_s is not None:
            meta["yielded_in_time"] = yielded_in_time
        if yield_latency_s is not None:
            meta["yield_latency_s"] = yield_latency_s
            meta["latency_basis"] = "reply_audio"
        if deadline_s is not None:
            meta["deadline_s"] = deadline_s
        self._event(EventType.BARGE_IN, t, party=by, meta=meta)

    def record_probe(self, result: ProbeResult, *, by: PartyId) -> None:
        """Record one L5 interjection and what the agent did about it.

        Both families read this one call. The selectivity family matches the
        ``SPEECH_START`` event by ``meta["probe"]``; the turn-taking family
        reads the ``BARGE_IN``, which is emitted only for probes that claimed
        the floor — a correctly-ignored "mm-hm" must never land in the
        barge-in denominator.
        """
        probe = result.probe
        self._event(
            EventType.SPEECH_START,
            result.t_injected or 0.0,
            party=by,
            meta={
                "probe": probe.kind.value,
                "layer": str(probe.key),
                "landing": result.landing.value,
                "bid": probe.bid.value,
                "ignored": result.ignored,
                "at_reply_s": result.at_reply_s,
            },
        )
        if not probe.expects_yield:
            return
        self.barge_in(
            result.t_injected or 0.0,
            by=by,
            outcome=_OUTCOME_OF[result.landing],
            yielded_in_time=result.yielded_in_time,
            yield_latency_s=result.yield_latency_s,
            deadline_s=result.deadline_s,
        )

    def record_probes(self, results: Sequence[ProbeResult], *, by: PartyId) -> None:
        for result in results:
            self.record_probe(result, by=by)

    def absorb_floor(self, floor: BaseFloor, *, include_yields: bool = True) -> None:
        """Pull overlaps, violations and yields off a floor policy.

        The floor counts during the call; the recorder turns those counts into
        timeline events afterwards. Keeping the two apart is what lets the same
        traffic be replayed under a different floor.

        Args:
            floor: The policy that ran.
            include_yields: Whether to emit a ``BARGE_IN`` per yield record.
                A run driven by :class:`~harbor_voice.duplex.DuplexDriver`
                passes ``False`` and uses :meth:`record_probe` instead, which
                knows which probe caused each yield and whether it landed
                anywhere measurable. Two sources writing the same event would
                double every interruption in the denominator.
        """
        for overlap in floor.overlaps:
            self._event(
                EventType.OVERLAP,
                overlap.t_start,
                parties=tuple(str(p) for p in overlap.parties),
                duration_s=overlap.duration_s,
            )
        for violation in floor.violations:
            self._event(
                EventType.FLOOR_VIOLATION,
                violation.t_start,
                party=violation.party,
                duration_s=violation.duration_s,
                meta={
                    "holder": str(violation.holder),
                    "decision": violation.decision.value,
                },
            )
        if include_yields and isinstance(floor, BargeInFloor):
            for record in floor.yields:
                self.barge_in(
                    record.t_interrupted,
                    by=PartyId(str(record.interrupter)),
                    outcome=(
                        BargeInOutcome.PENDING_DONE
                        if record.t_yielded is None
                        else BargeInOutcome.FIRED_AFTER_SPEECH
                    ),
                    yielded_in_time=record.yielded_in_time,
                    yield_latency_s=record.latency_s,
                    deadline_s=record.deadline_s,
                )

    # ------------------------------------------------------------- turns

    def record_turn(
        self,
        turn: Turn,
        *,
        utterance: Utterance | None = None,
        heard: str | None = None,
        clean_pcm: bytes | None = None,
        token_ids: Sequence[int] = (),
        logprobs: Sequence[float] = (),
    ) -> VoiceTurn:
        """Record one exchange, with both sides of said-vs-heard if available."""
        said = utterance.said if utterance is not None else None
        sent_frames = tuple(utterance.frames) if utterance is not None else turn.sent

        if sent_frames:
            self._event(
                EventType.SPEECH_START,
                _first(sent_frames, Clock.TX) or 0.0,
                party=sent_frames[0].party,
            )
            self._event(
                EventType.SPEECH_END,
                _last(sent_frames, Clock.TX) or 0.0,
                party=sent_frames[0].party,
            )
            # What the caller sent carries a send stamp only; placing it by a
            # receive stamp put every caller turn at zero.
            self._place(sent_frames, Clock.TX)

        if turn.received:
            self._event(EventType.SPEECH_START, turn.first_rx or 0.0, party=turn.party)
            self._event(EventType.SPEECH_END, turn.last_rx or 0.0, party=turn.party)
            self._place(turn.received, Clock.RX)

        for mark in turn.stages:
            self._event(
                EventType.STAGE,
                mark.t,
                party=mark.party,
                meta={"stage": mark.stage.value, **dict(mark.meta)},
            )

        span = None
        if token_ids:
            span = TokenSpan(
                start_idx=self._token_cursor,
                end_idx=self._token_cursor + len(token_ids),
                completion_token_ids=tuple(token_ids),
                logprobs=tuple(logprobs),
            )
            self._token_cursor += len(token_ids)

        record = VoiceTurn(
            idx=turn.index,
            party=str(turn.party),
            intent=utterance.intent if utterance is not None else None,
            said=said,
            heard=heard,
            audio_clean=_ref(clean_pcm, self._config.fmt),
            audio_transmitted=_ref(
                b"".join(f.pcm for f in turn.received) or None, self._config.fmt
            ),
            clocks=Clocks(
                produced=sent_frames[0].t_produced if sent_frames else None,
                tx=turn.last_tx,
                rx=turn.first_rx,
            ),
            stages=tuple(
                StageRecord(
                    stage=m.stage.value,
                    t=m.t,
                    clock=m.clock.value,
                    meta=dict(m.meta),
                )
                for m in turn.stages
            ),
            token_span=span,
            stop_reason=turn.stop.value,
            dropped_frames=turn.dropped,
            bled_frames=len(turn.bled),
        )
        self._turns.append(record)
        return record

    def _place(self, frames: Sequence[AudioFrame], clock: Clock) -> None:
        from harbor_voice.trajectory.render import placed_from_frames

        for frame in frames:
            if frame.clean_ref is not None:
                self._referenced.add(frame.clean_ref)
                self._saw_perturbed = True
            elif frame.plan_ref is not None:
                # A plan touched it but left no reference — the exact failure
                # the clean reference exists to prevent.
                self._unreferenced += 1
        self._placed.extend(placed_from_frames(frames, clock=clock))

    # ---------------------------------------------------------- validity

    def check_clean_reference(self, pipeline: PerturbationPipeline) -> None:
        """Verify every perturbed frame still has a retrievable clean twin.

        Frames are checked as they are recorded (see :meth:`_place`), because
        the recorder does not retain them. This reports what that check found.
        """
        if not self._config.require_clean_reference or pipeline.is_clean:
            self._checks.append(
                ValidityCheck(name="clean_tx_parity", ok=True, detail="clean profile")
            )
            return
        if self._unreferenced:
            self._checks.append(
                ValidityCheck(
                    name="clean_tx_parity",
                    ok=False,
                    score=float(self._unreferenced),
                    detail=(
                        f"{self._unreferenced} perturbed frame(s) carried no "
                        f"clean_ref; they cannot be compared to anything"
                    ),
                )
            )
            return
        missing = [
            digest for digest in self._referenced if not _retrievable(pipeline, digest)
        ]
        self._checks.append(
            ValidityCheck(
                name="clean_tx_parity",
                ok=not missing,
                score=float(len(self._referenced)),
                detail=(
                    f"{len(missing)} reference(s) missing from the store"
                    if missing
                    else f"{len(self._referenced)} reference(s) retrievable"
                ),
            )
        )

    def check_simulator(self, fidelity: SimulatorFidelity) -> None:
        """A caller that drifted means the run is not evidence."""
        floor = self._config.require_in_persona
        if floor is None:
            return
        if fidelity.in_persona is None:
            self._checks.append(
                ValidityCheck(
                    name="simulator_in_persona",
                    ok=True,
                    detail=(
                        "unmeasured — the run is not invalidated, but no claim "
                        "about caller fidelity may be published from it"
                    ),
                )
            )
            return
        self._checks.append(
            ValidityCheck(
                name="simulator_in_persona",
                ok=fidelity.in_persona >= floor,
                score=fidelity.in_persona,
                detail=f"floor is {floor}",
            )
        )

    def check_transport(self) -> None:
        budget = self._config.max_transport_gap_s
        self._checks.append(
            ValidityCheck(
                name="transport_gap_budget",
                ok=self._gap_total_s <= budget,
                score=self._gap_total_s,
                detail=f"budget is {budget}s",
            )
        )

    # ------------------------------------------------------------ finish

    def render(self) -> RenderResult:
        """Mix every placed source onto one timeline."""
        return render_timeline(self._placed, fmt=self._config.fmt)

    def segments(self) -> tuple[dict[str, Any], ...]:
        """Each party's placed audio: who, when, and its hash."""
        return tuple(
            {
                "party": str(p.party),
                "t0": p.start_s,
                "t1": p.end_s,
                "digest": str(sha256_digest(p.pcm)),
            }
            for p in self._placed
        )

    def finish(
        self,
        *,
        config_matrix: ConfigMatrix | None = None,
        pipeline: PerturbationPipeline | None = None,
        fidelity: SimulatorFidelity | None = None,
        render: bool = True,
    ) -> VoiceTrajectory:
        """Run the validity checks and produce the trajectory."""
        if pipeline is not None:
            self.check_clean_reference(pipeline)
        if fidelity is not None:
            self.check_simulator(fidelity)
        self.check_transport()

        artifacts = Artifacts(tier="T0")
        if render and self._placed:
            mixed = self.render()
            artifacts = Artifacts(
                mixed_timeline=AudioRef(
                    digest=sha256_digest(mixed.pcm),
                    duration_s=mixed.duration_s,
                    bytes=len(mixed.pcm),
                ),
                segments=self.segments(),
                tier="T1",
            )

        checks = tuple(self._checks)
        return VoiceTrajectory(
            trial_id=self._trial_id,
            task=self._task,
            task_sha=self._task_sha,
            agent=self._agent,
            model=self._model,
            parties=tuple(self._parties),
            config_matrix=config_matrix,
            perturbation_plan=(
                pipeline.as_record().as_dict() if pipeline is not None else {}
            ),
            timeline=tuple(sorted(self._events, key=lambda e: e.t)),
            turns=tuple(self._turns),
            artifacts=artifacts,
            validity=ValidityRecord(
                measurement_valid=all(c.ok for c in checks), checks=checks
            ),
            simulator_fidelity=(
                {
                    "utterances": fidelity.utterances,
                    "deterministic_source": fidelity.deterministic_source,
                    "source_kind": fidelity.source_kind.value,
                    "prerecorded_fraction": fidelity.prerecorded_fraction,
                    "in_persona": fidelity.in_persona,
                    "notes": list(fidelity.notes),
                }
                if fidelity is not None
                else {}
            ),
        )


_OUTCOME_OF: dict[Landing, BargeInOutcome] = {
    Landing.DURING_SPEECH: BargeInOutcome.FIRED_AFTER_SPEECH,
    Landing.REPLY_ALREADY_DONE: BargeInOutcome.PENDING_DONE,
    Landing.INDISTINGUISHABLE: BargeInOutcome.PENDING_DONE,
    Landing.NO_REPLY: BargeInOutcome.NO_ADAPTER,
}
"""Only ``DURING_SPEECH`` maps to an outcome the metric counts.

``INDISTINGUISHABLE`` deliberately shares ``PENDING_DONE``: in both cases the
reply had effectively finished, so neither is evidence about the agent.
"""


def _retrievable(pipeline: PerturbationPipeline, digest: Digest) -> bool:
    try:
        pipeline.clean_store.get(digest)
    except CleanReferenceError:
        return False
    return True


def _first(frames: Sequence[AudioFrame], clock: Clock) -> float | None:
    for frame in frames:
        t = frame.at(clock)
        if t is not None:
            return t
    return None


def _last(frames: Sequence[AudioFrame], clock: Clock) -> float | None:
    for frame in reversed(frames):
        t = frame.at(clock)
        if t is not None:
            return t
    return None


def _ref(pcm: bytes | None, fmt: AudioFormat) -> AudioRef | None:
    if not pcm:
        return None
    return AudioRef(
        digest=str(sha256_digest(pcm)),
        duration_s=fmt.duration_s(len(pcm)),
        bytes=len(pcm),
    )


def matrix_from(
    *, floor: BaseFloor, pipeline: PerturbationPipeline, profile: str = "clean"
) -> ConfigMatrix:
    """Build the pinned comparison dimensions from what actually ran."""
    return ConfigMatrix(
        duplex=floor.duplex,
        pipeline=Pipeline.CASCADE,
        floor_policy=floor.name,
        profile=profile,
        seed_root=pipeline.seeds.root,
    )


__all__ = [
    "DEFAULT_PERSONA_FLOOR",
    "Digest",
    "Duplex",
    "Recorder",
    "RecorderConfig",
    "matrix_from",
]
