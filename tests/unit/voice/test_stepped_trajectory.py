"""ATIF-voice for a lockstep call: ticks in, a VoiceTrajectory out."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
from harbor.models.trajectories.trajectory import Trajectory

from harbor_voice.trajectory.model import EventType
from harbor_voice.trajectory.model import PartyRole
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.stepped import Tick
from harbor_voice.trajectory.stepped import ToolEvent
from harbor_voice.trajectory.stepped import from_steps

pytestmark = pytest.mark.unit

A, C = "agent", "caller"


def _call() -> list[Tick]:
    return [
        Tick(0, A, "Hi! How can I help?"),  # a greeting: text, not speech
        # The caller speaks, pausing one tick mid-line (still one line).
        Tick(5, C, "I need to ", True),
        Tick(6, C, "cancel", True),
        Tick(8, C, " a flight.", True),
        # The agent replies and is cut off by the caller at step 14.
        Tick(12, A, "Sure, what is ", True),
        Tick(13, A, "your ", True),
        Tick(14, A, "booking", True, cut_off=True),
        Tick(14, C, "It's ", True),
        Tick(15, C, "ABC123.", True),
        Tick(20, A, "Done.", True),
    ]


def _build(**kwargs: object) -> VoiceTrajectory:
    return from_steps(_call(), tick_s=0.2, agent=A, caller=C, n_steps=25, **kwargs)  # type: ignore[arg-type]


def test_exchanges_pair_the_callers_words_with_the_agents_reply() -> None:
    turns = _build().turns
    assert [(t.said, t.reply) for t in turns] == [
        (None, "Hi! How can I help?"),
        ("I need to cancel a flight.", "Sure, what is your booking"),
        ("It's ABC123.", "Done."),
    ]
    assert turns[1].stop_reason == "cut_off"
    assert turns[1].clocks.tx == pytest.approx(1.0)
    assert turns[1].clocks.rx == pytest.approx(2.4)


def test_the_timeline_keeps_what_turns_hide() -> None:
    voice = _build()
    starts = voice.events_of(EventType.SPEECH_START)
    assert [(e.party, e.t) for e in starts] == [(C, 1.0), (A, 2.4), (C, 2.8), (A, 4.0)]
    (overlap,) = voice.events_of(EventType.OVERLAP)
    assert overlap.t == pytest.approx(2.8) and overlap.duration_s == pytest.approx(0.2)
    (barge,) = voice.events_of(EventType.BARGE_IN)
    assert barge.party == C and barge.meta["cut"] == A
    close = voice.events_of(EventType.SESSION_CLOSE)[0]
    assert close.t == pytest.approx(5.0)


def test_tools_land_on_the_timeline_with_their_results() -> None:
    voice = _build(
        tools=[
            ToolEvent(16, A, "get_reservation", {"id": "ABC123"}, result="{...}"),
            ToolEvent(17, C, "toggle_airplane_mode", error="no phone"),
        ]
    )
    calls = voice.events_of(EventType.TOOL_CALL)
    assert [(e.party, e.meta["name"]) for e in calls] == [
        (A, "get_reservation"),
        (C, "toggle_airplane_mode"),
    ]
    assert calls[0].meta["arguments"] == {"id": "ABC123"}
    assert calls[0].meta["result"] == "{...}" and calls[1].meta["error"] == "no phone"


def test_it_is_valid_atif_and_round_trips(tmp_path: Path) -> None:
    audio = tmp_path / "both.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(1)
        wav.setframerate(8000)
        wav.writeframes(b"\xff" * 2 * 8000 * 5)
    voice = _build(audio=audio, plan={"plan_id": "tau-voice-control"}, seed_root=7)

    atif = Trajectory.model_validate(json.loads(voice.to_atif().model_dump_json()))
    agent_steps = [s for s in atif.steps if s.source == "agent"]
    assert agent_steps[1].message == "Sure, what is your booking"
    back = VoiceTrajectory.from_atif(atif)
    assert back == voice
    assert back.artifacts.mixed_timeline is not None
    assert back.artifacts.mixed_timeline.duration_s == pytest.approx(5.0)
    assert back.config_matrix is not None
    assert back.config_matrix.profile == "tau-voice-control"
    assert {p.role for p in back.parties} == {
        PartyRole.SYSTEM_UNDER_TEST,
        PartyRole.SIMULATED_USER,
    }


def test_a_silent_call_is_still_a_document() -> None:
    voice = from_steps([], tick_s=0.2, agent=A, caller=C)
    assert voice.turns == ()
    Trajectory.model_validate(json.loads(voice.to_atif().model_dump_json()))
