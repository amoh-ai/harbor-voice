"""A tau-voice call as ATIF-voice: its steps through core's lockstep builder."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.stepped import Tick
from harbor_voice.trajectory.stepped import ToolEvent
from harbor_voice.trajectory.stepped import from_steps

from tau_voice.stepped_call import CallResult

AGENT = "agent"
CALLER = "caller"
FLOOR_POLICY = "tau2-patient"
"""tau2's ``control`` turn-taking (respond after 5 / 25 quiet ticks, yield
when interrupted) — what decided who spoke when."""


def ticks_of(result: CallResult) -> tuple[list[Tick], list[ToolEvent]]:
    """Both parties per step, and the caller's own tool calls."""
    ticks: list[Tick] = []
    tools: list[ToolEvent] = []
    # tau2 answers a caller's tool call at a later step than the call, so
    # results are matched by id across the whole call.
    results = {r.id: r for log in result.steps for r in log.user_tool_results}
    for log in result.steps:
        user = log.user
        if user.speech or (user.text and not user.tool_calls):
            ticks.append(
                Tick(log.step, CALLER, user.text if user.speech else None, user.speech)
            )
        if log.agent_text or log.agent_speech:
            ticks.append(
                Tick(
                    log.step,
                    AGENT,
                    log.agent_text,
                    log.agent_speech,
                    cut_off=bool(log.agent and log.agent.was_truncated),
                )
            )
        for call in user.tool_calls:
            answer = results.get(call.id)
            tools.append(
                ToolEvent(
                    step=log.step,
                    party=CALLER,
                    name=call.name,
                    arguments=dict(call.arguments),
                    result=answer.content if answer and not answer.error else None,
                    error=answer.content if answer and answer.error else None,
                    call_id=call.id,
                )
            )
    return ticks, tools


def agent_tools(log: Sequence[Mapping[str, Any]]) -> list[ToolEvent]:
    return [
        ToolEvent(
            step=int(entry["step"]),
            party=AGENT,
            name=str(entry["name"]),
            arguments=dict(entry.get("arguments") or {}),
            result=entry.get("result"),
            error=entry.get("error"),
            call_id=entry.get("call_id"),
        )
        for entry in log
    ]


def build(
    result: CallResult,
    *,
    tick_s: float,
    agent_tool_log: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    audio: Path | None,
    attempt: int = 0,
) -> VoiceTrajectory:
    ticks, caller_tools = ticks_of(result)
    tools = sorted([*caller_tools, *agent_tools(agent_tool_log)], key=lambda e: e.step)
    agent = summary.get("agent") or {}
    return from_steps(
        ticks,
        tick_s=tick_s,
        agent=AGENT,
        caller=CALLER,
        tools=tools,
        n_steps=len(result.steps),
        audio=audio,
        floor_policy=FLOOR_POLICY,
        plan=summary.get("caller_plan"),
        usage=[UsageRecord.model_validate(u) for u in summary.get("usage") or []],
        seed_root=summary.get("seed"),
        task=f"tau-voice/{summary.get('domain')}/{summary.get('task_id')}",
        trial_id=agent.get("session_id"),
        agent_name=agent.get("provider"),
        model=agent.get("model"),
        attempt=attempt,
    )
