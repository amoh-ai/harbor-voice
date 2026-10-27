"""tau-voice runtime sidecar — tau2's environment, prompts and tool execution.

Runs inside the tau2 image (its own Python; tau2 pins ``litellm<1.82.7`` and
harbor-voice needs ``>=1.92``, so the two cannot share an environment). The
same split as upstream harbor's ``tau3-bench`` runtime server.

**It does not run a call.** harbor-voice runs the call — the caller, the clock,
the agent transport — and comes here for the things that define the
benchmark:

- ``POST /open``  — build the task: environment, agent prompt and tools, the
  caller's rendered system prompt and tools, and the caller's voice/effect
  configuration. Everything is built by tau2's own builders.
- ``POST /tool``  — execute one agent or caller tool call against the
  environment, exactly as ``Orchestrator._execute_tool_calls`` does.
- ``POST /user_context`` — the caller LLM request tau2 would send right now:
  harbor-voice's step history, seen from the caller's side, linearised,
  role-flipped and wrapped exactly as
  ``VoiceStreamingUserSimulator._generate_full_duplex_voice_message`` does,
  returned as the request body. Pure computation, and the step most likely to
  break parity if ported, so it is not ported.
- ``POST /grade`` — grade the call last closed (same code as
  ``tests/test.sh``), for the driver's per-attempt hallucination retry.
- ``POST /close`` — turn harbor-voice's step log plus the tool calls recorded
  here into tau2 ``Tick`` records and write a ``SimulationRun`` to the state
  log. Grading is **not** done here; ``tests/evaluate.py`` does it, so a call
  can be re-graded without being re-run.

One session at a time per container. A trial gets its own container; a
hallucination retry reopens it after ``/close``.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Literal

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from tau2.agent.base.streaming import LinearizationStrategy
from tau2.agent.base.streaming import ParticipantTick
from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.config import DEFAULT_LLM_USER
from tau2.data_model.message import AssistantMessage
from tau2.data_model.message import MultiToolMessage
from tau2.data_model.message import SystemMessage
from tau2.data_model.message import Tick
from tau2.data_model.message import ToolCall
from tau2.data_model.message import ToolMessage
from tau2.data_model.message import UserMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.simulation import VoiceRunConfig
from tau2.data_model.voice_personas import get_elevenlabs_voice_id
from tau2.runner.batch import make_voice_run_settings
from tau2.runner.build import _build_env_kwargs
from tau2.runner.build import build_agent
from tau2.runner.build import build_environment
from tau2.runner.build import build_voice_user
from tau2.runner.helpers import get_tasks
from tau2.utils.llm_utils import to_litellm_messages

STATE_PATH = Path(
    os.environ.get("TAU_VOICE_STATE_PATH", "/logs/agent/tau_voice_state.json")
)
MAX_ERRORS = int(os.environ.get("TAU_VOICE_MAX_ERRORS", "10"))

app = FastAPI(title="tau-voice runtime")


class OpenRequest(BaseModel):
    domain: str
    task_id: str
    speech_complexity: str = "control"
    seed: int = 42
    user_llm: str = DEFAULT_LLM_USER
    hallucination_feedback: str | None = None


class ToolRequest(BaseModel):
    side: Literal["agent", "user"]
    step: int = Field(ge=0, description="harbor-voice step index the call belongs to")
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Utterance(BaseModel):
    text: str | None = None
    speech: bool = False


class Step(BaseModel):
    step: int = Field(ge=0)
    t_s: float = Field(ge=0, description="call time at the start of the step")
    user: Utterance | None = None
    agent: Utterance | None = None


class CallerToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class CallerToolResult(BaseModel):
    id: str
    content: str
    error: bool = False


class CallerSaid(BaseModel):
    """What the caller emitted in one step: a text slice, and/or tool calls."""

    text: str | None = None
    speech: bool = False
    tool_calls: list[CallerToolCall] = Field(default_factory=list)


class CallerStep(BaseModel):
    """One step from the caller's side — tau2's ``ParticipantTick``."""

    step: int = Field(ge=0)
    said: CallerSaid | None = None
    heard: Utterance | None = None
    tool_results: list[CallerToolResult] = Field(default_factory=list)


class UserContextRequest(BaseModel):
    steps: list[CallerStep]
    pending: Utterance | None = Field(
        default=None,
        description="agent speech arriving now, not yet recorded as a step",
    )
    pending_tool_results: list[CallerToolResult] = Field(
        default_factory=list,
        description=(
            "caller tool results arriving now; tau2's _process_tool_result puts "
            "them, not speech, in the pending slot"
        ),
    )


ROLE_REMINDER = (
    "REMINDER: You are the CUSTOMER calling for help. Respond as the customer "
    "would - with questions, requests, or information about your issue. Do NOT "
    "respond as the customer service agent."
)
"""Verbatim from ``_generate_full_duplex_voice_message``."""


class CloseRequest(BaseModel):
    steps: list[Step]
    termination: Literal[
        "agent_stop",
        "user_stop",
        "max_steps",
        "too_many_errors",
        "agent_error",
        "user_error",
        "timeout",
    ]
    duration_s: float = Field(ge=0)
    tick_duration_s: float = 0.2


class _Session:
    """Everything one task needs between ``/open`` and ``/close``."""

    def __init__(self, request: OpenRequest) -> None:
        tasks = get_tasks(request.domain, task_ids=[request.task_id])
        if len(tasks) != 1:
            raise ValueError(f"task {request.task_id!r} not found in {request.domain}")
        self.request = request
        self.task = tasks[0]
        self.config = VoiceRunConfig(
            domain=request.domain,
            audio_native_config=AudioNativeConfig(),
            llm_user=request.user_llm,
            speech_complexity=request.speech_complexity,
            seed=request.seed,
        )
        self.env_kwargs = _build_env_kwargs(self.config, self.task)
        self.environment = build_environment(request.domain, env_kwargs=self.env_kwargs)
        # Mirrors FullDuplexOrchestrator.initialize(): state first, then sync.
        initial = self.task.initial_state
        self.environment.set_state(
            initialization_data=initial.initialization_data if initial else None,
            initialization_actions=initial.initialization_actions if initial else None,
            message_history=[],
        )
        self.environment.sync_tools()

        agent = build_agent(
            self.config.effective_agent,
            self.environment,
            audio_native_config=self.config.audio_native_config,
        )
        # Run-level settings derived exactly as tau2's batch runner does.
        voice_settings, persona = make_voice_run_settings(self.config)
        # Constructing the caller renders its prompt and samples its voice;
        # it opens no connection and synthesises nothing.
        user = build_voice_user(
            self.environment,
            self.task,
            self.config.audio_native_config,
            llm=self.config.llm_user,
            llm_args=self.config.llm_args_user,
            voice_settings=voice_settings,
            persona_config=persona,
            speech_complexity=self.config.speech_complexity,
            seed=request.seed,
            domain=request.domain,
            hallucination_feedback=request.hallucination_feedback,
        )
        # FullDuplexOrchestrator.initialize() seeds both participants. The
        # agent's set_seed is a no-op; the caller's puts ``seed`` into its LLM
        # arguments, so every caller request carries it.
        agent.set_seed(request.seed)
        user.set_seed(request.seed)
        self.agent = agent
        self.user = user
        self.closed = False
        # Built once: it loads noise files and effect schedulers, none of which
        # linearisation touches. Only its ticks and input buffer are replaced
        # per request.
        self.user_state = user.get_init_state()
        self.calls: list[tuple[int, str, ToolCall, ToolMessage]] = []
        self.num_errors = 0
        self.start = datetime.now()
        self.lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        anc = self.config.audio_native_config
        user = self.user
        return {
            "task_id": self.task.id,
            "domain": self.request.domain,
            "agent": {
                "system_prompt": self.agent.system_prompt,
                "tools": [t.openai_schema for t in self.agent.tools],
                "stop_tool": DiscreteTimeAudioNativeAgent.STOP_FUNCTION_NAME,
                "default_provider": anc.provider,
                "default_model": anc.model,
            },
            "user": {
                "system_prompt": user.system_prompt,
                "tools": [t.openai_schema for t in (user.tools or [])],
                "llm": self.config.llm_user,
                "llm_args": self.config.llm_args_user,
                "persona": user.persona_config.model_dump(mode="json"),
                "persona_name": user.voice_settings.speech_environment.persona_name,
                # Resolved here, by tau2, from TAU2_VOICE_ID_* in this container.
                "voice_id": get_elevenlabs_voice_id(
                    user.voice_settings.speech_environment.persona_name
                ),
                "voice_settings": user.voice_settings.model_dump(mode="json"),
                "turn_taking": {
                    "tick_duration_s": anc.tick_duration_seconds,
                    "wait_to_respond_other_ticks": user.wait_to_respond_threshold_other,
                    "wait_to_respond_self_ticks": user.wait_to_respond_threshold_self,
                    "yield_when_interrupted_ticks": user.yield_threshold_when_interrupted,
                    "yield_when_interrupting_ticks": user.yield_threshold_when_interrupting,
                    "backchannel_min_ticks": user.backchannel_min_threshold,
                    "backchannel_max_ticks": user.backchannel_max_threshold,
                    "backchannel_poisson_rate": user.backchannel_poisson_rate,
                    "use_llm_backchannel": user.use_llm_backchannel,
                    "interruption_check_interval_ticks": user.interruption_check_interval,
                    "integration_ticks": user.integration_ticks,
                    "silence_annotation_threshold_ticks": user.silence_annotation_threshold_ticks,
                    "max_steps_seconds": anc.max_steps_seconds,
                },
            },
        }

    def call_tool(self, request: ToolRequest) -> dict[str, Any]:
        call = ToolCall(
            id=request.id,
            name=request.name,
            arguments=request.arguments,
            requestor="assistant" if request.side == "agent" else "user",
        )
        with self.lock:
            result = self.environment.get_response(call)
            if result.error:
                self.num_errors += 1
            # The orchestrator syncs once per tick; syncing per call reaches
            # the same state because sync_tools is idempotent.
            self.environment.sync_tools()
            self.calls.append((request.step, request.side, call, result))
        return {
            "content": result.content,
            "error": result.error,
            "stop": request.side == "agent"
            and request.name == DiscreteTimeAudioNativeAgent.STOP_FUNCTION_NAME,
            "too_many_errors": self.num_errors >= MAX_ERRORS,
        }

    def user_context(self, request: UserContextRequest) -> dict[str, Any]:
        user = self.user
        state = self.user_state
        stamp = self.start.isoformat()
        ticks = []
        for step in request.steps:
            said = None
            if step.said is not None:
                calls = [
                    ToolCall(
                        id=c.id, name=c.name, arguments=c.arguments, requestor="user"
                    )
                    for c in step.said.tool_calls
                ]
                said = UserMessage(
                    role="user",
                    content=step.said.text or None,
                    contains_speech=step.said.speech,
                    tool_calls=calls or None,
                    timestamp=stamp,
                )
            heard = None
            if step.heard is not None:
                heard = AssistantMessage(
                    role="assistant",
                    content=step.heard.text or None,
                    contains_speech=step.heard.speech,
                    timestamp=stamp,
                )
            env = None
            results = [
                ToolMessage(
                    id=r.id,
                    role="tool",
                    content=r.content,
                    error=r.error,
                    requestor="user",
                )
                for r in step.tool_results
            ]
            if len(results) == 1:
                env = results[0]
            elif results:
                env = MultiToolMessage(role="tool", tool_messages=results)
            ticks.append(
                ParticipantTick(
                    tick_id=step.step,
                    timestamp=stamp,
                    self_chunk=said,
                    other_chunk=heard,
                    env_chunk=env,
                )
            )
        state.ticks = ticks
        state.input_turn_taking_buffer = []
        if request.pending_tool_results:
            results = [
                ToolMessage(
                    id=r.id,
                    role="tool",
                    content=r.content,
                    error=r.error,
                    requestor="user",
                )
                for r in request.pending_tool_results
            ]
            state.input_turn_taking_buffer = [
                results[0]
                if len(results) == 1
                else MultiToolMessage(role="tool", tool_messages=results)
            ]
        elif request.pending is not None:
            state.input_turn_taking_buffer = [
                AssistantMessage(
                    role="assistant",
                    content=request.pending.text or None,
                    contains_speech=request.pending.speech,
                    timestamp=stamp,
                )
            ]
        linearized = state.get_linearized_messages(
            strategy=LinearizationStrategy.CONTAINMENT_AWARE,
            include_pending_input=True,
            indicate_current_incomplete=True,
            integration_ticks=user.integration_ticks,
            silence_annotation_threshold_ticks=user.silence_annotation_threshold_ticks,
            tick_duration_seconds=user.tick_duration_seconds,
        )
        messages = (
            state.system_messages
            + user._flip_roles_for_llm(linearized)
            + [SystemMessage(role="system", content=ROLE_REMINDER)]
        )
        tools = [t.openai_schema for t in user.tools] if user.tools else None
        body: dict[str, Any] = {
            "model": user.llm,
            "messages": to_litellm_messages(messages),
            **(user.llm_args or {}),
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    def close(self, request: CloseRequest) -> SimulationRun:
        ticks: dict[int, Tick] = {}
        for step in request.steps:
            stamp = (self.start + timedelta(seconds=step.t_s)).isoformat()
            tick = Tick(
                tick_id=step.step,
                timestamp=stamp,
                tick_duration_seconds=request.tick_duration_s,
            )
            if step.user is not None:
                tick.user_chunk = UserMessage(
                    role="user",
                    content=step.user.text or None,
                    timestamp=stamp,
                    contains_speech=step.user.speech,
                )
            if step.agent is not None:
                tick.agent_chunk = AssistantMessage(
                    role="assistant",
                    content=step.agent.text or None,
                    timestamp=stamp,
                    contains_speech=step.agent.speech,
                )
            ticks[step.step] = tick
        for index, side, call, result in self.calls:
            tick = ticks.get(index)
            if tick is None:
                stamp = (
                    self.start + timedelta(seconds=index * request.tick_duration_s)
                ).isoformat()
                tick = ticks[index] = Tick(tick_id=index, timestamp=stamp)
            if side == "agent":
                tick.agent_tool_calls.append(call)
                tick.agent_tool_results.append(result)
                if call.name == DiscreteTimeAudioNativeAgent.STOP_FUNCTION_NAME:
                    # Same marker DiscreteTimeAudioNativeAgent puts on the
                    # chunk, so the communicate evaluator sees what it would.
                    chunk = tick.agent_chunk or AssistantMessage(
                        role="assistant",
                        content=None,
                        timestamp=tick.timestamp,
                        contains_speech=False,
                    )
                    stop = DiscreteTimeAudioNativeAgent.STOP_TOKEN
                    chunk.content = (
                        stop if chunk.content is None else f"{chunk.content} {stop}"
                    )
                    tick.agent_chunk = chunk
            else:
                tick.user_tool_calls.append(call)
                tick.user_tool_results.append(result)
        ordered = [ticks[i] for i in sorted(ticks)]
        end = self.start + timedelta(seconds=request.duration_s)
        return SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=self.start.isoformat(),
            end_time=end.isoformat(),
            duration=request.duration_s,
            termination_reason=TerminationReason(request.termination),
            ticks=ordered,
            seed=self.request.seed,
            mode="full_duplex",
            speech_environment=self.user.voice_settings.speech_environment,
        )


_session: _Session | None = None


def _require() -> _Session:
    if _session is None:
        raise HTTPException(status_code=409, detail="no open session; POST /open first")
    return _session


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/open")
def open_session(request: OpenRequest) -> dict[str, Any]:
    global _session
    if _session is not None and not _session.closed:
        raise HTTPException(
            status_code=409, detail="a session is open; close it before opening another"
        )
    try:
        _session = _Session(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _session.describe()


@app.post("/tool")
def tool(request: ToolRequest) -> dict[str, Any]:
    return _require().call_tool(request)


@app.post("/user_context")
def user_context(request: UserContextRequest) -> dict[str, Any]:
    return _require().user_context(request)


@app.post("/close")
def close(request: CloseRequest) -> dict[str, Any]:
    session = _require()
    run = session.close(request)
    session.closed = True
    state = {
        "open": session.request.model_dump(mode="json"),
        "env_kwargs_task_bound": bool(session.env_kwargs),
        "num_errors": session.num_errors,
        "simulation": run.model_dump(mode="json"),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return {"state_path": str(STATE_PATH), "ticks": len(run.ticks or [])}


class GradeRequest(BaseModel):
    hallucination_check: bool = True
    review_model: str | None = None


@app.post("/grade")
def grade_last(request: GradeRequest) -> dict[str, Any]:
    """Grade the call last closed — the same code as ``tests/test.sh``.

    The driver grades every attempt to decide on tau2's hallucination retry.
    It runs in the ``main`` container and cannot exec into this one, so it
    asks over the environment's internal network. The orchestrator's final
    ``tests/test.sh`` remains the official grade.
    """
    if not STATE_PATH.exists():
        raise HTTPException(
            status_code=409, detail="nothing to grade; POST /close first"
        )
    from evaluate import grade

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return grade(
        state,
        review_model=request.review_model or os.environ.get("TAU_VOICE_REVIEW_MODEL"),
        hallucination_check=request.hallucination_check,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
