"""A tau-voice call's steps become ATIF-voice through core's lockstep builder."""

from __future__ import annotations

import pytest

from harbor_voice.trajectory.model import EventType
from tau_voice.caller import CallerChunk
from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.stepped_call import CallResult
from tau_voice.stepped_call import StepLog
from tau_voice.trajectory import build


def _step(
    n: int,
    user: CallerChunk,
    agent_text: str | None = None,
    agent_speech: bool = False,
    **kw: object,
) -> StepLog:
    return StepLog(n, n * 0.2, user, agent_text, agent_speech, **kw)  # type: ignore[arg-type]


def test_a_call_with_both_sides_tools_becomes_a_trajectory() -> None:
    quiet = CallerChunk(audio=b"", speech=False)
    toggle = CallerToolCall(
        id="u1", name="toggle_airplane_mode", arguments={"on": False}
    )
    result = CallResult(
        steps=[
            _step(0, quiet, "Hi! How can I help you today?"),
            _step(1, CallerChunk(audio=b"", speech=True, text="My phone ")),
            _step(2, CallerChunk(audio=b"", speech=True, text="is offline.")),
            _step(3, quiet, "Let me ", True),
            _step(4, quiet, "check.", True),
            _step(5, CallerChunk(audio=b"", speech=False, tool_calls=(toggle,))),
            # tau2 answers the caller's tool call at the next step.
            _step(
                6, quiet, user_tool_results=(CallerToolResult(id="u1", content="done"),)
            ),
        ]
    )
    voice = build(
        result,
        tick_s=0.2,
        agent_tool_log=[
            {
                "step": 4,
                "name": "get_line",
                "arguments": {"id": 7},
                "result": "ok",
                "call_id": "a1",
            }
        ],
        summary={
            "domain": "telecom",
            "task_id": "t1",
            "seed": 626729,
            "agent": {"provider": "openai.realtime", "model": "gpt-realtime-1.5"},
            "caller_plan": {"plan_id": "tau-voice-control"},
        },
        audio=None,
    )
    assert [(t.said, t.reply) for t in voice.turns] == [
        (None, "Hi! How can I help you today?"),
        ("My phone is offline.", "Let me check."),
    ]
    tools = voice.events_of(EventType.TOOL_CALL)
    assert [(e.party, e.meta["name"], e.meta.get("result")) for e in tools] == [
        ("agent", "get_line", "ok"),
        ("caller", "toggle_airplane_mode", "done"),
    ]
    assert voice.task == "tau-voice/telecom/t1"
    assert voice.config_matrix is not None
    assert voice.config_matrix.floor_policy == "tau2-patient"
    assert voice.config_matrix.seed_root == 626729
    (close,) = voice.events_of(EventType.SESSION_CLOSE)
    assert close.t == pytest.approx(1.4)  # seven steps of 0.2 s
