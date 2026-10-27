"""ATIF-voice for a lockstep call — one a benchmark's driver ran in fixed ticks.

The built-in call path records frames as they cross the wire
(:class:`~harbor_voice.trajectory.recorder.Recorder`). A driven call — tau-voice
today, any benchmark that steps both parties in lockstep — has a different
record: per tick, what each party said (a slice of text), whether it was
speech, whether it was cut off, plus the tool calls. This turns that record
into the same :class:`VoiceTrajectory`, so harbor's ATIF tooling and our
metrics read both alike.

Turns are **exchanges**, as the built-in path records them: the caller's
words (``said``) and the agent's words until the caller speaks again
(``reply``). Anything the agent says before the caller first speaks — a
greeting — is exchange 0, with no ``said``. Full duplex has no strict turns;
this is the cut a reader expects, and the timeline keeps the overlaps the cut
hides.

Runs of activity tolerate short silences (``max_gap_s``): a caller pausing
mid-sentence is one line, not two — unless the other party spoke in the
pause.
"""

from __future__ import annotations

import hashlib
import wave
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.trajectory.model import Artifacts
from harbor_voice.trajectory.model import AudioRef
from harbor_voice.trajectory.model import BargeInOutcome
from harbor_voice.trajectory.model import Clocks
from harbor_voice.trajectory.model import ConfigMatrix
from harbor_voice.trajectory.model import EventType
from harbor_voice.trajectory.model import PartyRecord
from harbor_voice.trajectory.model import PartyRole
from harbor_voice.trajectory.model import TimelineEvent
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.model import VoiceTurn


@dataclass(frozen=True, slots=True)
class Tick:
    """One party in one tick."""

    step: int
    party: str
    text: str | None = None
    """This tick's slice of what the party said; slices concatenate."""
    speech: bool = False
    cut_off: bool = False
    """The party's output was truncated here — it was interrupted."""


@dataclass(frozen=True, slots=True)
class ToolEvent:
    step: int
    party: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class _Run:
    party: str
    start: int
    end: int
    """Last active step, inclusive."""
    text: str
    speech: bool
    cut_off_at: int | None


def _runs(
    ticks: Sequence[Tick], party: str, max_gap: int, other: set[int]
) -> list[_Run]:
    """A party's lines: active ticks, joined across short silences unless the
    other party spoke in the silence — then it is a new line."""
    active = sorted(
        (t for t in ticks if t.party == party and (t.speech or t.text)),
        key=lambda t: t.step,
    )
    runs: list[_Run] = []
    group: list[Tick] = []
    for tick in active:
        last = group[-1].step if group else None
        if last is not None and (
            tick.step - last - 1 > max_gap
            or any(last < step < tick.step for step in other)
        ):
            runs.append(_close(party, group))
            group = []
        group.append(tick)
    if group:
        runs.append(_close(party, group))
    return runs


def _close(party: str, group: list[Tick]) -> _Run:
    cut = next((t.step for t in group if t.cut_off), None)
    return _Run(
        party=party,
        start=group[0].step,
        end=group[-1].step,
        text="".join(t.text or "" for t in group),
        speech=any(t.speech for t in group),
        cut_off_at=cut,
    )


def _spans(steps: Sequence[int]) -> list[tuple[int, int]]:
    """Consecutive steps as (first, last) spans."""
    spans: list[tuple[int, int]] = []
    for step in sorted(steps):
        if spans and step == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], step)
        else:
            spans.append((step, step))
    return spans


def audio_ref(path: Path) -> AudioRef:
    """A WAV by content hash — never inlined."""
    data = path.read_bytes()
    with wave.open(str(path), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    return AudioRef(
        digest="sha256:" + hashlib.sha256(data).hexdigest(),
        duration_s=duration,
        bytes=len(data),
    )


def from_steps(
    ticks: Sequence[Tick],
    *,
    tick_s: float,
    agent: str,
    caller: str,
    tools: Sequence[ToolEvent] = (),
    n_steps: int | None = None,
    audio: Path | None = None,
    max_gap_s: float = 1.0,
    floor_policy: str = "lockstep",
    plan: Mapping[str, Any] | None = None,
    seed_root: int | None = None,
    task: str | None = None,
    trial_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    attempt: int = 0,
) -> VoiceTrajectory:
    """Build the trajectory of a lockstep call from its per-tick record."""

    def t(step: float) -> float:
        return round(step * tick_s, 6)

    max_gap = max(0, round(max_gap_s / tick_s))
    last = max((tick.step for tick in ticks), default=0)
    n_steps = n_steps if n_steps is not None else last + 1
    active = {
        party: {x.step for x in ticks if x.party == party and (x.speech or x.text)}
        for party in (caller, agent)
    }
    caller_runs = _runs(ticks, caller, max_gap, active[agent])
    agent_runs = _runs(ticks, agent, max_gap, active[caller])

    events: list[TimelineEvent] = [
        TimelineEvent(t=0.0, type=EventType.SESSION_OPEN),
        TimelineEvent(t=t(n_steps), type=EventType.SESSION_CLOSE),
    ]
    for run in (*caller_runs, *agent_runs):
        if not run.speech:
            continue
        events.append(
            TimelineEvent(t=t(run.start), type=EventType.SPEECH_START, party=run.party)
        )
        events.append(
            TimelineEvent(
                t=t(run.end + 1),
                type=EventType.SPEECH_END,
                party=run.party,
                duration_s=t(run.end + 1 - run.start),
            )
        )
    speaking = {
        party: {tick.step for tick in ticks if tick.party == party and tick.speech}
        for party in (caller, agent)
    }
    for first, end in _spans(sorted(speaking[caller] & speaking[agent])):
        events.append(
            TimelineEvent(
                t=t(first),
                type=EventType.OVERLAP,
                parties=(caller, agent),
                duration_s=t(end + 1 - first),
            )
        )
    for run in agent_runs:
        if run.cut_off_at is not None:
            events.append(
                TimelineEvent(
                    t=t(run.cut_off_at),
                    type=EventType.BARGE_IN,
                    party=caller,
                    meta={
                        "outcome": BargeInOutcome.FIRED_AFTER_SPEECH.value,
                        "native": False,
                        "yielded_in_time": True,
                        "cut": agent,
                    },
                )
            )
    for tool in tools:
        meta: dict[str, Any] = {"name": tool.name, "arguments": dict(tool.arguments)}
        if tool.call_id is not None:
            meta["call_id"] = tool.call_id
        if tool.result is not None:
            meta["result"] = tool.result
        if tool.error is not None:
            meta["error"] = tool.error
        events.append(
            TimelineEvent(
                t=t(tool.step), type=EventType.TOOL_CALL, party=tool.party, meta=meta
            )
        )
    events.sort(key=lambda e: e.t)

    turns = _exchanges(ticks, caller_runs, agent, t)

    return VoiceTrajectory(
        trial_id=trial_id,
        task=task,
        agent=agent_name,
        model=model,
        attempt=attempt,
        parties=(
            PartyRecord(id=agent, role=PartyRole.SYSTEM_UNDER_TEST),
            PartyRecord(id=caller, role=PartyRole.SIMULATED_USER),
        ),
        config_matrix=ConfigMatrix(
            duplex=Duplex.FULL,
            pipeline=Pipeline.S2S,
            floor_policy=floor_policy,
            profile=str((plan or {}).get("plan_id", "clean")),
            seed_root=seed_root,
        ),
        perturbation_plan=dict(plan or {}),
        timeline=tuple(events),
        turns=turns,
        artifacts=(
            Artifacts(mixed_timeline=audio_ref(audio), tier="T1")
            if audio is not None and audio.exists()
            else Artifacts()
        ),
    )


def _exchanges(
    ticks: Sequence[Tick], caller_runs: list[_Run], agent: str, t: Any
) -> tuple[VoiceTurn, ...]:
    said_runs = [run for run in caller_runs if run.text]
    bounds = [run.start for run in said_runs]
    agent_ticks = sorted(
        (tick for tick in ticks if tick.party == agent and (tick.text or tick.speech)),
        key=lambda tick: tick.step,
    )

    def window(lo: int | None, hi: int | None) -> list[Tick]:
        # (lo, hi]: what the agent said on the step the caller starts a new
        # line was cut by it, so it ends the previous reply.
        return [
            x
            for x in agent_ticks
            if (lo is None or x.step > lo) and (hi is None or x.step <= hi)
        ]

    turns: list[VoiceTurn] = []
    lead = window(None, bounds[0] if bounds else None)
    if lead:
        turns.append(_turn(len(turns), agent, None, lead, t))
    for i, run in enumerate(said_runs):
        hi = bounds[i + 1] if i + 1 < len(bounds) else None
        turns.append(_turn(len(turns), agent, run, window(run.start, hi), t))
    return tuple(turns)


def _turn(
    idx: int, agent: str, said: _Run | None, reply: list[Tick], t: Any
) -> VoiceTurn:
    text = "".join(x.text or "" for x in reply) or None
    return VoiceTurn(
        idx=idx,
        party=agent,
        said=said.text if said is not None else None,
        reply=text,
        clocks=Clocks(
            tx=t(said.start) if said is not None else None,
            rx=t(reply[0].step) if reply else None,
        ),
        stop_reason="cut_off" if any(x.cut_off for x in reply) else None,
    )
