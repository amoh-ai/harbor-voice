"""ATIF-voice for a real-time call: speech segments in wall time in, a
VoiceTrajectory out."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
from harbor.models.trajectories.trajectory import Trajectory

from harbor_voice.trajectory.model import AudioRef
from harbor_voice.trajectory.model import EventType
from harbor_voice.trajectory.model import PartyRole
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.timed import Segment
from harbor_voice.trajectory.timed import ToolEvent
from harbor_voice.trajectory.timed import from_timeline

pytestmark = pytest.mark.unit

A, C = "agent", "caller"


def _call() -> list[Segment]:
    return [
        Segment(A, 0.5, 2.0, "Hello! How can I help you today?"),
        # The caller pauses 0.4 s mid-sentence: one line, not two.
        Segment(C, 3.0, 4.2, "Hi, I need to change", heard="hi I need to change"),
        Segment(C, 4.6, 5.5, "my flight.", heard="my flight"),
        # The agent is cut off by the caller at 8.2 and stops at 9.0.
        Segment(A, 6.0, 9.0, "Sure, what is your confirmation", cut_off=True),
        Segment(C, 8.2, 10.0, "It's ZK3FFW."),
        # A tool call (10.5-10.7), then two agent segments 0.5 s apart.
        Segment(A, 11.0, 12.0, "Let me check."),
        Segment(A, 12.5, 14.0, "Found it."),
    ]


TOOLS = [
    ToolEvent(
        10.5,
        A,
        "get_reservation",
        {"confirmation_number": "ZK3FFW"},
        result={"status": "success"},
        call_id="call_1",
        end=10.7,
    ),
]


def _build(segments: list[Segment] | None = None, **kwargs: object) -> VoiceTrajectory:
    return from_timeline(
        _call() if segments is None else segments,
        agent=A,
        caller=C,
        **kwargs,  # type: ignore[arg-type]
    )


def test_exchanges_pair_the_callers_lines_with_the_agents_reply() -> None:
    turns = _build(tools=TOOLS, duration_s=15.0).turns
    assert [(t.said, t.reply) for t in turns] == [
        (None, "Hello! How can I help you today?"),
        ("Hi, I need to change my flight.", "Sure, what is your confirmation"),
        ("It's ZK3FFW.", "Let me check. Found it."),
    ]
    assert turns[1].heard == "hi I need to change my flight"
    assert turns[1].stop_reason == "cut_off" and turns[2].stop_reason is None
    assert (turns[1].clocks.tx, turns[1].clocks.rx) == (3.0, 6.0)
    assert (turns[0].clocks.tx, turns[0].clocks.rx) == (None, 0.5)
    assert all(t.party == A for t in turns)


def test_the_timeline_measures_what_turns_hide() -> None:
    voice = _build(tools=TOOLS, duration_s=15.0)
    starts = voice.events_of(EventType.SPEECH_START)
    assert [(e.party, e.t) for e in starts] == [
        (A, 0.5),
        (C, 3.0),
        (A, 6.0),
        (C, 8.2),
        (A, 11.0),
    ]
    ends = {(e.party, e.t): e.duration_s for e in voice.events_of(EventType.SPEECH_END)}
    assert ends[(C, 5.5)] == pytest.approx(2.5)  # the line spans the pause
    assert ends[(A, 14.0)] == pytest.approx(3.0)
    (overlap,) = voice.events_of(EventType.OVERLAP)
    assert (overlap.t, overlap.duration_s, overlap.parties) == (8.2, 0.8, (C, A))
    (barge,) = voice.events_of(EventType.BARGE_IN)
    assert (barge.t, barge.party) == (8.2, C)
    assert barge.meta["cut"] == A and barge.meta["yielded_in_time"] is True
    assert barge.meta["yield_latency_s"] == pytest.approx(0.8)
    (close,) = voice.events_of(EventType.SESSION_CLOSE)
    assert close.t == 15.0
    assert [e.t for e in voice.timeline] == sorted(e.t for e in voice.timeline)


def test_a_tool_call_between_turns_lands_on_the_timeline_with_its_time() -> None:
    voice = _build(
        tools=[
            *TOOLS,
            ToolEvent(12.2, A, "rebook_flight", error="no seats"),
        ]
    )
    calls = voice.events_of(EventType.TOOL_CALL)
    assert [(e.t, e.party, e.meta["name"]) for e in calls] == [
        (10.5, A, "get_reservation"),
        (12.2, A, "rebook_flight"),
    ]
    assert calls[0].duration_s == pytest.approx(0.2)
    assert calls[0].meta["result"] == {"status": "success"}
    assert calls[0].meta["call_id"] == "call_1"
    assert calls[1].meta["error"] == "no seats" and calls[1].duration_s is None
    # The tool answered between the caller's line and the agent's reply.
    ordered = [e.type for e in voice.timeline if 10.0 <= e.t <= 11.0]
    assert ordered == [
        EventType.SPEECH_END,
        EventType.TOOL_CALL,
        EventType.SPEECH_START,
    ]
    # Without a stated length, the call ends at its last segment.
    assert voice.events_of(EventType.SESSION_CLOSE)[0].t == 14.0


def test_the_other_party_speaking_in_a_pause_splits_the_line() -> None:
    voice = _build(
        [
            Segment(C, 1.0, 2.0, "I want to"),
            Segment(A, 2.1, 2.3, "Mm-hm."),
            Segment(C, 2.5, 3.0, "cancel."),
        ]
    )
    assert [(t.said, t.reply) for t in voice.turns] == [
        ("I want to", "Mm-hm."),
        ("cancel.", None),
    ]


def test_a_zero_gap_keeps_every_segment_its_own_line() -> None:
    voice = _build(_call(), max_gap_s=0.0)
    assert len(voice.events_of(EventType.SPEECH_START)) == 7


def test_overlapping_segments_of_one_party_are_one_stretch() -> None:
    voice = _build(
        [Segment(A, 0.0, 2.0, "One"), Segment(A, 1.5, 3.0, "two"), Segment(C, 1.0, 2.5)]
    )
    (start,) = [e for e in voice.events_of(EventType.SPEECH_START) if e.party == A]
    (overlap,) = voice.events_of(EventType.OVERLAP)
    assert (overlap.t, overlap.duration_s) == (1.0, 1.5)
    assert start.t == 0.0
    # The caller said nothing we know of, so no exchange starts there.
    assert [(t.said, t.reply) for t in voice.turns] == [(None, "One two")]


def test_an_interrupter_who_starts_just_after_the_cut_is_found() -> None:
    """A provider that stops on its own VAD can stop before the caller's
    audio is counted as started."""
    voice = _build(
        [
            Segment(A, 0.0, 2.0, "Long answer", cut_off=True),
            Segment(C, 2.3, 3.0, "Wait."),
        ]
    )
    (barge,) = voice.events_of(EventType.BARGE_IN)
    assert (barge.t, barge.party, barge.meta["yield_latency_s"]) == (2.3, C, 0.0)
    assert not voice.events_of(EventType.OVERLAP)


def test_a_cut_with_nobody_heard_is_still_a_barge_in() -> None:
    voice = _build([Segment(A, 0.0, 2.0, "Long answer", cut_off=True)])
    (barge,) = voice.events_of(EventType.BARGE_IN)
    assert (barge.t, barge.party) == (2.0, None)
    assert "yield_latency_s" not in barge.meta


def test_a_yield_deadline_judges_in_time() -> None:
    voice = _build(yield_deadline_s=0.5)
    (barge,) = voice.events_of(EventType.BARGE_IN)
    assert barge.meta["yielded_in_time"] is False and barge.meta["deadline_s"] == 0.5


def test_an_empty_call_is_a_valid_trajectory() -> None:
    voice = _build([])
    assert [e.type for e in voice.timeline] == [
        EventType.SESSION_OPEN,
        EventType.SESSION_CLOSE,
    ]
    assert voice.turns == ()
    atif = voice.to_atif()
    assert len(atif.steps) == 1 and atif.steps[0].source == "system"


def test_only_a_greeting_is_exchange_zero() -> None:
    voice = _build([Segment(A, 0.2, 1.0, "Hello!")])
    (turn,) = voice.turns
    assert (turn.idx, turn.said, turn.reply) == (0, None, "Hello!")


@pytest.mark.parametrize(
    "segment",
    [Segment("third", 0.0, 1.0), Segment(A, 2.0, 1.0), Segment(C, -0.5, 1.0)],
)
def test_a_bad_segment_is_refused(segment: Segment) -> None:
    with pytest.raises(ValueError):
        _build([segment])


def _wav(path: Path, frames: int) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(b"\x00\x00" * frames)
    return path


def test_audio_goes_by_reference_and_usage_passes_through(tmp_path: Path) -> None:
    mixed = _wav(tmp_path / "mixed.wav", 24_000)
    agent = _wav(tmp_path / "agent.wav", 12_000)
    given = AudioRef(digest="sha256:abc", duration_s=1.0, bytes=10)
    usage = [UsageRecord(component="agent", model="m", requests=2, input_tokens=100)]
    voice = _build(
        audio=mixed,
        per_party_audio={A: agent, C: given, "gone": tmp_path / "missing.wav"},
        usage=usage,
    )
    assert voice.artifacts.tier == "T1"
    assert voice.artifacts.mixed_timeline is not None
    assert voice.artifacts.mixed_timeline.duration_s == pytest.approx(1.0)
    assert voice.artifacts.per_party[A].duration_s == pytest.approx(0.5)
    assert voice.artifacts.per_party[C] == given
    assert "gone" not in voice.artifacts.per_party
    assert voice.usage == tuple(usage)
    assert _build().artifacts.tier == "T0"


def test_it_is_valid_atif_and_round_trips() -> None:
    voice = _build(
        tools=TOOLS,
        task="eva/airline/1.1.2",
        trial_id="t1",
        agent_name="openai.realtime",
        model="gpt-realtime-1.5",
        plan={"plan_id": "clean"},
        seed_root=7,
        floor_policy="eva-bridge",
    )
    assert {p.id: p.role for p in voice.parties} == {
        A: PartyRole.SYSTEM_UNDER_TEST,
        C: PartyRole.SIMULATED_USER,
    }
    assert voice.config_matrix is not None
    assert voice.config_matrix.floor_policy == "eva-bridge"
    atif = voice.to_atif()
    Trajectory.model_validate(atif.model_dump())
    assert [s.source for s in atif.steps] == ["agent", "user", "agent", "user", "agent"]
    assert VoiceTrajectory.from_atif(atif) == voice


def test_the_package_exports_it_lazily() -> None:
    import harbor_voice.trajectory as pkg
    from harbor_voice.trajectory import timed

    assert pkg.from_timeline is timed.from_timeline
    assert pkg.ToolEvent is timed.ToolEvent and pkg.Segment is timed.Segment
    assert "from_timeline" in dir(pkg)
