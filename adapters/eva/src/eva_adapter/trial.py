"""The EVA call driver — what a converted task's ``voice.toml`` names.

``[call] driver = "eva_adapter.trial:drive"``. It runs inside the task's
``main`` container (harbor-voice installs it there at trial start), next to
the ``eva`` sidecar service, and runs one trial of one EVA record the way
EVA's runner does:

1. ``POST /open``: EVA's agent session (prompt, tools, greeting) and caller
   (prompt, gender) for this attempt.
2. The call: the agent under test built from the task's ``[agent]`` with
   EVA's Realtime settings (checked against EVA's own ``session.update``
   before anything connects), the caller on EVA's ElevenLabs agent, and
   :class:`~eva_adapter.call.RealtimeCall` between them. Tools go to
   ``POST /tool``.
3. ``POST /record``, always — with the error, if the call failed — then
   ``POST /validate``: EVA's ``ValidationRunner`` and rerun rule. While it
   says ``rerun``, back to 1 (at most ``max_attempts``, EVA's 3).
4. ``POST /grade`` if an attempt passed: EVA's metrics on the attempt that
   counts. ``GET /files`` either way: EVA's run directory, archived attempts
   included.

**Outputs**, under ``<out>/eva/``: ``summary.json`` (every attempt's end
reason, validation and timings; usage per component), ``metrics.json``
(EVA's, for the attempt that counts), ``run/`` (EVA's run directory as the
sidecar holds it), ``attempt-<n>/attempt.json``, and the ElevenLabs agent's
live config under ``elevenlabs/``. The ATIF-voice trajectory of the attempt
that counts is the outcome's; core writes it where harbor reads it.

**The reward** is one of EVA's record composites, ``[call.config].reward``:
``EVA-A_pass`` (accuracy; the default), ``EVA-X_pass`` (experience) or
``EVA-overall_pass`` (both). It is harbor's ``reward``; all three composites
are recorded beside it under their own names (:func:`rewards`). A record no
attempt validated has no metrics in EVA, and so no reward here: EVA leaves it
out of every mean, and so does the job's metric.

**Why the task keeps a verifier.** The driver grades through the sidecar
itself, because the trial's summary and trajectory need the grade, and
because ``main`` cannot exec into the sidecar. The task's ``tests/test.sh``
still runs afterwards and is the reward harbor records, as in tau-voice. It
calls the same ``/grade``, which costs nothing the second time: EVA's
``MetricsRunner`` computes only the metrics missing from the record's
``metrics.json`` (``run_and_save_record``, normal mode), so no judge is
called again and the numbers are the driver's.

**The trial index.** EVA seeds nothing per trial: its trials of a record are
``records/<id>/trial_<k>``, ``k`` = 0 … ``num_trials`` − 1, and pass@k is
taken over them. The task's seed is ``k``; a job runs the task once per
seed 0, 1, 2 (:func:`eva_adapter.main.job_config`).

**Perturbations.** ``[call.config.perturbation]`` is one of EVA's conditions
(:mod:`eva_adapter.conditions`). Every attempt's call runs EVA's own
``AudioPerturbator`` on the caller's audio, a new one per attempt as EVA
builds one per conversation; a noise WAV it cannot read fails the trial
before any attempt. The summary (``condition``, ``perturbation``) and the
trajectory (``config_matrix.profile``, ``perturbation_plan``) name it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Protocol

import numpy
from harbor_voice.driver import DriverContext
from harbor_voice.driver import DriverOutcome
from harbor_voice.driver import VerifierResult
from harbor_voice.providers.elevenlabs.agents import API_BASE
from harbor_voice.providers.elevenlabs.agents import AgentsAPI
from harbor_voice.providers.elevenlabs.agents import ElevenLabsAgentsTransport
from harbor_voice.providers.openai.realtime import RealtimeSettings
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.model import merge_usage
from harbor_voice.trial import AgentSpec

from eva_adapter import trajectory as eva_trajectory
from eva_adapter.agents import agent_id_from_env
from eva_adapter.agents import label_for
from eva_adapter.agents import snapshot
from eva_adapter.call import CallOutcome
from eva_adapter.call import RealtimeCall
from eva_adapter.caller import ElevenLabsCaller
from eva_adapter.client import Grade
from eva_adapter.client import Opened
from eva_adapter.client import Validation
from eva_adapter.images import EVA_NUMPY
from eva_adapter.conditions import ConditionError
from eva_adapter.conditions import PerturbationConfig
from eva_adapter.conditions import as_record
from eva_adapter.conditions import label_of
from eva_adapter.conditions import perturbator
from eva_adapter.tools import SidecarToolExecutor
from eva_adapter.tools import tool_specs

logger = logging.getLogger(__name__)

SIDECAR_URL = "http://eva:8000"
"""The sidecar by compose service name, inside the environment's network."""

OUT = "eva"
RUN = "run"
RECORD = "record"
"""``attempt-<n>/record/``: the call loop's files, as sent to ``/record``;
removed once ``run/`` holds EVA's copy."""

REWARDS = ("EVA-A_pass", "EVA-X_pass", "EVA-overall_pass")
DEFAULT_REWARD = "EVA-A_pass"
NUM_TRIALS = 3
"""EVA's ``num_trials`` (k of pass@k)."""

REALTIME = "openai.realtime"
"""The one agent provider whose EVA settings exist: EVA's reference system."""
REFERENCE_MODEL = "gpt-realtime-1.5"

SERVER_URL = "ws://localhost:10000/ws"
"""Logged with the caller's ``connected`` event, where EVA logs its
assistant server's URL (its port pool starts at 10000). Nothing reads it."""


def eva_realtime_settings(opening_text: str | None) -> RealtimeSettings:
    """EVA's OpenAI Realtime session (``openai_realtime_server.py:185-282``).

    ``_build_session_config`` with ``s2s_params`` holding only the model, and
    ``_handle_session``'s greeting, as the Realtime-session golden records.
    """
    return RealtimeSettings(
        voice="marin",
        vad_threshold=0.5,
        prefix_padding_ms=300,
        silence_duration_ms=200,
        transcription_model="whisper-1",
        language=None,
        noise_reduction=None,
        tool_choice=None,
        audio_format="pcm24k",
        opening_text=opening_text,
        tool_flow="serial",
        tool_output_ascii=False,
        extended_events=True,
    )


class EvaConfigError(RuntimeError):
    """The trial cannot run as configured; no attempt is spent on it."""


class EvaParityError(RuntimeError):
    """The agent would not be set up as EVA sets it up."""


def _diff(expected: Any, actual: Any, path: str = "session") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        out: list[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            where = f"{path}.{key}"
            if key not in actual:
                out.append(f"{where}: missing (EVA sends {expected[key]!r:.80})")
            elif key not in expected:
                out.append(f"{where}: extra (EVA sends none; got {actual[key]!r:.80})")
            else:
                out.extend(_diff(expected[key], actual[key], where))
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: {len(actual)} items, EVA sends {len(expected)}"]
        out = []
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            out.extend(_diff(e, a, f"{path}[{i}]"))
        return out
    if expected != actual:
        return [f"{path}: {actual!r:.80}, EVA sends {expected!r:.80}"]
    return []


def check_session(agent: Any, opened: Opened) -> str:
    """The live parity guard: the agent's ``session.update`` must be EVA's.

    EVA's own server rendered ``opened.agent.session`` for this attempt
    (``/open``); the transport renders its own from the same prompt, tools
    and :func:`eva_realtime_settings`. They must be equal, or the trial fails
    here, before anything connects.

    Returns ``"matched"``, or why the check does not apply. Raises
    :class:`EvaParityError` with every difference.
    """
    expected = opened.agent.session
    render = getattr(agent, "session_update", None)
    if expected is None:
        return "not applicable: the sidecar returned no Realtime session"
    if not callable(render):
        return f"not applicable: {type(agent).__name__} sends no session.update"
    actual = render().get("session")
    if actual != expected:
        diffs = _diff(expected, actual) or ["the payloads differ"]
        raise EvaParityError(
            "the agent's session.update is not EVA's for this record "
            f"({len(diffs)} difference(s)): " + "; ".join(diffs[:12])
        )
    return "matched"


# ------------------------------------------------------------- configuration


@dataclass(frozen=True, slots=True)
class TrialConfig:
    """``[call.config]``, read and checked."""

    domain: str
    record_id: str
    language: str = "en"
    trial: int = 0
    num_trials: int = NUM_TRIALS
    reward: str = DEFAULT_REWARD
    sidecar_url: str = SIDECAR_URL
    caller_api_base: str = API_BASE
    """ElevenLabs' API (tests point it at a fake)."""
    snapshot_caller: bool = True
    """Store the caller agent's live config with the trial (one ``GET``)."""
    perturbation: PerturbationConfig | None = None
    """``[call.config.perturbation]``: EVA's condition on the caller's audio
    (:mod:`eva_adapter.conditions`); none is EVA's clean run."""

    def __post_init__(self) -> None:
        if self.reward not in REWARDS:
            raise EvaConfigError(
                f"reward {self.reward!r} is not one of EVA's: {', '.join(REWARDS)}"
            )
        if not 0 <= self.trial < self.num_trials:
            raise EvaConfigError(
                f"trial {self.trial} is outside EVA's num_trials {self.num_trials} "
                "(the task's seed is the trial index k)"
            )

    @classmethod
    def from_context(cls, ctx: DriverContext) -> TrialConfig:
        c = ctx.driver_config
        perturbation = c.get("perturbation")
        if perturbation is not None and not isinstance(perturbation, Mapping):
            raise EvaConfigError("[call.config.perturbation] must be a table")
        try:
            condition = (
                None
                if perturbation is None
                else PerturbationConfig.from_mapping(perturbation)
            )
        except ConditionError as exc:
            raise EvaConfigError(str(exc)) from exc
        return cls(
            domain=str(c["domain"]),
            record_id=str(c["record_id"]),
            language=str(c.get("language", "en")),
            trial=int(c.get("trial", ctx.config.seed)),
            num_trials=int(c.get("num_trials", NUM_TRIALS)),
            reward=str(c.get("reward", DEFAULT_REWARD)),
            sidecar_url=str(c.get("sidecar_url", SIDECAR_URL)),
            caller_api_base=str(c.get("caller_api_base", API_BASE)),
            snapshot_caller=bool(c.get("snapshot_caller", True)),
            perturbation=condition,
        )

    @property
    def task(self) -> str:
        """The task's name (``voice.toml``'s ``task``), the condition included."""
        name = f"eva/{self.domain}/{self.record_id}"
        return name if self.perturbation is None else f"{name}@{self.condition}"

    @property
    def condition(self) -> str:
        return label_of(self.perturbation)

    @property
    def output_id(self) -> str:
        """EVA's output id: the record directory under ``records/``."""
        return f"{self.record_id}/trial_{self.trial}"


# ----------------------------------------------------------------- one call


@dataclass(slots=True)
class AttemptCall:
    """What one call produced beside its record files."""

    outcome: CallOutcome
    usage: tuple[UsageRecord, ...] = ()
    agent: dict[str, Any] = field(default_factory=dict)
    caller: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    session_check: str = "not run"


CallAttempt = Callable[[Opened, Path], Awaitable[AttemptCall]]
"""Runs one opened attempt's call, writing its record files into the path."""


class SidecarAPI(Protocol):
    """The part of :class:`~eva_adapter.client.Sidecar` a trial uses."""

    async def open(
        self,
        *,
        domain: str,
        record_id: str,
        language: str = "en",
        trial: int = 0,
        num_trials: int | None = None,
    ) -> Opened: ...

    async def record(self, record_dir: Path, call: Any) -> dict[str, Any]: ...

    async def validate(self, attempt: int) -> Validation: ...

    async def grade(self) -> Grade: ...

    async def fetch(self, dest: Path) -> Path: ...


def agent_usage(agent: Any) -> tuple[UsageRecord, ...]:
    """The agent transport's own meter, where it keeps one."""
    records = getattr(agent, "usage_records", None)
    record = getattr(agent, "usage_record", None)
    if callable(records):
        return tuple(records("agent"))
    if callable(record):
        return (record("agent"),)
    return ()


def agent_info(agent: Any) -> dict[str, Any]:
    ref = getattr(agent, "ref", None)
    return {
        "provider": getattr(ref, "provider", None),
        "model": getattr(ref, "model", None),
        "session_id": getattr(agent, "session_id", None),
        "errors": list(getattr(agent, "errors", []) or []),
        "pump_error": str(error)
        if (error := getattr(agent, "pump_error", None))
        else None,
    }


def caller_info(caller: ElevenLabsCaller, agent_id: str) -> dict[str, Any]:
    transport = caller.transport
    return {
        "agent_id": agent_id,
        "conversation_id": transport.conversation_id,
        "end_reason": caller.end_reason,
        "end_call_found": caller.end_call_found,
        "timed_out": caller.timed_out,
        "keepalives_sent": caller.keepalives_sent,
        "error": str(caller.error) if caller.error else None,
        "transport_errors": list(transport.errors),
        "ignored_messages": dict(transport.ignored),
        "audio_chunks_sent": transport.sent_chunks,
    }


def realtime_attempts(
    ctx: DriverContext,
    config: TrialConfig,
    sidecar: Any,
    out_dir: Path,
) -> CallAttempt:
    """Each attempt's call: the task's agent, EVA's caller, EVA's call loop."""
    snapped: set[str] = set()

    async def attempt(opened: Opened, record_dir: Path) -> AttemptCall:
        if opened.agent.system_prompt is None:
            raise EvaConfigError(
                "the sidecar returned no Realtime session for this record; the "
                "driver runs EVA's openai_realtime setup"
            )
        executor = SidecarToolExecutor(sidecar, tool_specs(opened.agent.tools))
        agent = ctx.build_agent(
            tools=ToolRouter(executor),
            system_prompt=opened.agent.system_prompt,
            settings=eva_realtime_settings(opened.agent.opening_text),
        )
        session_check = check_session(agent, opened)
        try:
            agent_id = agent_id_from_env(opened.caller.language, opened.caller.gender)
        except ValueError as exc:
            raise EvaConfigError(str(exc)) from exc
        label = label_for(opened.caller.language, opened.caller.gender)
        snapshot_note: dict[str, Any] = {}
        if config.snapshot_caller and label not in snapped:
            snapped.add(label)
            try:
                await snapshot(
                    {label: agent_id},
                    out_dir / "elevenlabs",
                    AgentsAPI(api_base=config.caller_api_base),
                )
                snapshot_note = {
                    "snapshot": f"elevenlabs/elevenlabs_agent_{label}.json"
                }
            except Exception as exc:  # noqa: BLE001 - a record of the caller, not the call
                logger.warning("Could not snapshot the caller agent: %s", exc)
                snapshot_note = {"snapshot_error": f"{type(exc).__name__}: {exc}"}
        caller_transport = ElevenLabsAgentsTransport(
            agent_id=agent_id, api_base=config.caller_api_base
        )
        model = (
            getattr(getattr(agent, "ref", None), "model", None)
            or opened.raw.get("agent", {}).get("model")
            or REFERENCE_MODEL
        )
        call = RealtimeCall(
            agent=agent,
            caller_transport=caller_transport,
            out_dir=record_dir,
            conversation_id=opened.conversation_id,
            model=str(model),
            guard_timeout_s=opened.caller_timeout_s,
            # A new one per attempt, as EVA builds one per conversation.
            perturbator=perturbator(config.perturbation),
        )
        caller = ElevenLabsCaller(
            transport=caller_transport,
            prompt=opened.caller.prompt,
            events=call.events,
            out_dir=record_dir,
            bridge=call.bridge,
            timeout_s=opened.caller_timeout_s,
            server_url=SERVER_URL,
        )
        outcome = await call.run(caller)
        return AttemptCall(
            outcome=outcome,
            usage=(*agent_usage(agent), *caller.usage_records()),
            agent=agent_info(agent),
            caller={**caller_info(caller, agent_id), "label": label, **snapshot_note},
            tools=executor.log,
            session_check=session_check,
        )

    return attempt


# ---------------------------------------------------------------- the trial


@dataclass(slots=True)
class TrialRun:
    """What a trial produced, for the driver's outcome."""

    reward: float | None
    result: dict[str, Any]
    summary: dict[str, Any]
    trajectory: VoiceTrajectory | None


def judge_usage(run_dir: Path) -> tuple[UsageRecord, ...]:
    """EVA's ``judge_token_usage.json``: per metric, the run's running totals
    (validation of every attempt, then grading)."""
    path = run_dir / "judge_token_usage" / "judge_token_usage.json"
    if not path.exists():
        return ()
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        UsageRecord(
            component=f"judge.{metric}",
            model=entry.get("model_id"),
            requests=entry.get("num_calls"),
            input_tokens=entry.get("total_input_tokens"),
            output_tokens=entry.get("total_output_tokens"),
        )
        for metric, entry in sorted(data.items())
        if isinstance(entry, Mapping)
    )


def scores(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each EVA metric's headline numbers, without its details."""
    return {
        name: {
            key: score.get(key)
            for key in ("score", "normalized_score", "skipped", "error")
        }
        for name, score in metrics.items()
        if isinstance(score, Mapping)
    }


def rewards(aggregates: Mapping[str, Any], reward_key: str) -> dict[str, float]:
    """harbor's rewards: ``reward`` (the chosen composite), then each of EVA's
    composites EVA set, under its own name. Empty when the chosen one is unset,
    so the trial is unscored (``tests/grade.py`` writes the same)."""
    if aggregates.get(reward_key) is None:
        return {}
    named = {
        key: float(aggregates[key])
        for key in REWARDS
        if aggregates.get(key) is not None
    }
    return {"reward": named[reward_key], **named}


def graded_result(grade: Grade, reward_key: str) -> dict[str, Any]:
    """The verifier's ``result.json`` (``tests/grade.py`` writes the same)."""
    value = grade.aggregates.get(reward_key)
    return {
        "graded": True,
        "reward_key": reward_key,
        "reward": None if value is None else float(value),
        "rewards": rewards(grade.aggregates, reward_key),
        "record_id": grade.record_id,
        "aggregates": dict(grade.aggregates),
        "metrics": scores(grade.metrics),
    }


def _validation(v: Validation) -> dict[str, Any]:
    return {
        "passed": v.passed,
        "gate": v.gate,
        "reason": v.reason,
        "scores": v.scores,
        "failed_metrics": v.failed_metrics,
        "rerun": v.rerun,
        "restored_attempt": v.restored_attempt,
    }


def _usage(records: Sequence[UsageRecord]) -> list[dict[str, Any]]:
    return [u.model_dump(exclude_none=True) for u in records]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def counted_record(run_dir: Path, output_id: str, last_attempt: int) -> Path | None:
    """The record EVA settled on, else the last attempt's archive."""
    record = run_dir / "records" / output_id
    if (record / "result.json").exists():
        return record
    archive = record.parent / f"{record.name}_failed_attempt_{last_attempt}"
    return archive if archive.exists() else None


async def run_trial(
    sidecar: SidecarAPI,
    config: TrialConfig,
    *,
    out_dir: Path,
    call: CallAttempt,
    progress: Callable[[str], None] = lambda _message: None,
) -> TrialRun:
    """EVA's attempt loop for one trial of one record; see the module docstring."""
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    usage_by_attempt: dict[int, tuple[UsageRecord, ...]] = {}
    started = time.monotonic()
    validation: Validation | None = None
    while True:
        opened = await sidecar.open(
            domain=config.domain,
            record_id=config.record_id,
            language=config.language,
            trial=config.trial,
            num_trials=config.num_trials,
        )
        n = opened.attempt
        attempt_dir = out_dir / f"attempt-{n}"
        record_dir = attempt_dir / RECORD
        progress(f"attempt {n}/{opened.max_attempts}: started ({config.output_id})")
        t0 = time.monotonic()
        made = await call(opened, record_dir)
        t1 = time.monotonic()
        outcome = made.outcome
        await sidecar.record(record_dir, outcome.call_info(n))
        t2 = time.monotonic()
        validation = await sidecar.validate(n)
        t3 = time.monotonic()
        usage_by_attempt[n] = made.usage
        entry = {
            "attempt": n,
            "conversation_id": opened.conversation_id,
            "end_reason": outcome.conversation_ended_reason,
            "error": outcome.error,
            "agent_error": outcome.agent_error,
            "started_at": outcome.started_at.isoformat(),
            "ended_at": outcome.ended_at.isoformat(),
            "timings": {
                "call_s": round(outcome.duration_seconds, 3),
                "call_wall_s": round(t1 - t0, 3),
                "record_s": round(t2 - t1, 3),
                "validate_s": round(t3 - t2, 3),
            },
            "stats": outcome.stats,
            "validation": _validation(validation),
            "session_check": made.session_check,
            "agent": made.agent,
            "caller": made.caller,
            "tool_calls": len(made.tools),
            "usage": _usage(made.usage),
        }
        attempts.append(entry)
        _write_json(attempt_dir / "attempt.json", {**entry, "tools": made.tools})
        progress(
            f"attempt {n}: {outcome.conversation_ended_reason} · "
            f"{outcome.duration_seconds:.0f} s · validation {validation.reason}"
            + (" → rerun" if validation.rerun else "")
        )
        if not validation.rerun:
            break

    assert validation is not None
    last = attempts[-1]["attempt"]
    counted = (
        (validation.restored_attempt or validation.attempt)
        if validation.passed
        else None
    )
    grade: Grade | None = None
    grade_error: Exception | None = None
    grade_s = None
    if validation.passed:
        t = time.monotonic()
        try:
            grade = await sidecar.grade()
        except Exception as exc:  # noqa: BLE001 - fetch first, then raise
            grade_error = exc
        grade_s = round(time.monotonic() - t, 3)

    run_dir = out_dir / RUN
    fetch_error = None
    try:
        await sidecar.fetch(run_dir)
    except Exception as exc:  # noqa: BLE001 - the local copies stay instead
        fetch_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Could not fetch EVA's run directory: %s", exc)
    if grade_error is not None:
        raise grade_error

    record_dir = counted_record(run_dir, config.output_id, last)
    if fetch_error is None:
        for entry in attempts:
            shutil.rmtree(
                out_dir / f"attempt-{entry['attempt']}" / RECORD, ignore_errors=True
            )
    elif record_dir is None:
        record_dir = out_dir / f"attempt-{counted or last}" / RECORD

    judges = judge_usage(run_dir)
    reward_value = grade.aggregates.get(config.reward) if grade else None
    reward = None if reward_value is None else float(reward_value)
    result: dict[str, Any]
    if grade is not None:
        result = graded_result(grade, config.reward)
        eva_metrics = record_dir / "metrics.json" if record_dir else None
        if eva_metrics is not None and eva_metrics.exists():
            shutil.copyfile(eva_metrics, out_dir / "metrics.json")
        else:
            _write_json(
                out_dir / "metrics.json",
                {
                    "record_id": grade.record_id,
                    "metrics": grade.metrics,
                    "aggregate_metrics": grade.aggregates,
                },
            )
    else:
        result = {
            "graded": False,
            "reward_key": config.reward,
            "reward": None,
            "detail": "no attempt passed EVA's validation; EVA scores no such record",
        }
    result["validation"] = {
        "passed": validation.passed,
        "reason": validation.reason,
        "attempts": len(attempts),
        "counted_attempt": counted,
    }

    used = merge_usage([*(u for a in usage_by_attempt.values() for u in a), *judges])
    summary: dict[str, Any] = {
        "task": config.task,
        "domain": config.domain,
        "record_id": config.record_id,
        "language": config.language,
        "trial": config.trial,
        "num_trials": config.num_trials,
        "output_id": config.output_id,
        "condition": config.condition,
        "perturbation": as_record(config.perturbation),
        "numpy": numpy.__version__,
        "reward_key": config.reward,
        "reward": reward,
        "aggregates": dict(grade.aggregates) if grade else None,
        "validated": validation.passed,
        "validation_reason": validation.reason,
        "attempts_used": len(attempts),
        "counted_attempt": counted,
        "rerun_history": validation.raw.get("rerun_history"),
        "attempts": attempts,
        "grade_s": grade_s,
        "wall_s": round(time.monotonic() - started, 3),
        "usage": _usage(used),
        "usage_by_component": {
            "agent": _usage(
                merge_usage(
                    u
                    for a in usage_by_attempt.values()
                    for u in a
                    if u.component == "agent" or u.component.startswith("agent.")
                )
            ),
            "caller": _usage(
                merge_usage(
                    u
                    for a in usage_by_attempt.values()
                    for u in a
                    if u.component == "caller" or u.component.startswith("caller.")
                )
            ),
            "judges": _usage(judges),
        },
        "files": {
            "run": RUN if fetch_error is None else None,
            "record": record_dir.relative_to(out_dir).as_posix()
            if record_dir is not None and record_dir.is_relative_to(out_dir)
            else None,
            "fetch_error": fetch_error,
        },
    }
    _write_json(out_dir / "summary.json", summary)

    voice = None
    if record_dir is not None and record_dir.exists():
        shown = counted or last
        agent = next((a["agent"] for a in attempts if a["attempt"] == shown), {})
        voice = eva_trajectory.build(
            record_dir,
            usage=(*usage_by_attempt.get(shown, ()), *judges),
            task=config.task,
            trial_id=config.output_id,
            agent_name=agent.get("provider"),
            model=agent.get("model"),
            attempt=shown,
            seed_root=config.trial,
            plan={**config.perturbation.plan(), "numpy": numpy.__version__}
            if config.perturbation
            else None,
        )
    progress(
        f"graded: {config.reward} {reward}"
        if grade is not None
        else f"not graded: {validation.reason} after {len(attempts)} attempt(s)"
    )
    return TrialRun(reward=reward, result=result, summary=summary, trajectory=voice)


# -------------------------------------------------------------- the driver


def _require_realtime(agent: AgentSpec | None) -> None:
    provider = ((agent.s2s if agent else None) or REALTIME).partition("/")[0]
    if provider != REALTIME:
        raise EvaConfigError(
            f"EVA's settings exist for {REALTIME} only (EVA's reference system); "
            f"the agent under test is {provider}"
        )


def _preflight(ctx: DriverContext, config: TrialConfig) -> None:
    """Keys a call cannot run without, checked before any attempt is spent.

    A missing key would otherwise end each call as an ``error``, which EVA
    reruns, until the attempts run out.
    """
    try:
        AgentsAPI(api_base=config.caller_api_base).require_key()
    except Exception as exc:
        raise EvaConfigError(f"the caller cannot connect: {exc}") from exc
    s2s = (ctx.agent.config.get("s2s", {}) if ctx.agent else {}) or {}
    if not s2s.get("api_key") and not os.environ.get("OPENAI_API_KEY"):
        raise EvaConfigError("the agent cannot connect: OPENAI_API_KEY is not set")
    if config.perturbation is not None and numpy.__version__ != EVA_NUMPY:
        logger.warning(
            "EVA's perturbator runs on numpy %s, not EVA's %s: chunks over 8192 "
            "samples can round differently (eva_adapter.images.EVA_NUMPY)",
            numpy.__version__,
            EVA_NUMPY,
        )
    try:
        perturbator(config.perturbation)  # EVA's, reading its noise WAV
    except (OSError, ValueError) as exc:
        raise EvaConfigError(
            f"EVA's perturbation {config.condition} cannot run here: {exc} "
            "(tasks converted with --background-noise build main with EVA's WAVs)"
        ) from exc


async def drive(ctx: DriverContext) -> DriverOutcome:
    from eva_adapter.client import Sidecar

    config = TrialConfig.from_context(ctx)
    _require_realtime(ctx.agent)
    _preflight(ctx, config)
    out_dir = ctx.out_dir / OUT
    sidecar = Sidecar(config.sidecar_url)
    try:
        run = await run_trial(
            sidecar,
            config,
            out_dir=out_dir,
            call=realtime_attempts(ctx, config, sidecar, out_dir),
            progress=ctx.progress,
        )
    finally:
        await sidecar.aclose()
    return DriverOutcome(
        verified=VerifierResult(reward=run.reward, result=run.result),
        summary=run.summary,
        trajectory=run.trajectory,
    )
