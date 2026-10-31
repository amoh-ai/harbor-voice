"""ATIF-voice for a real-time call — one that ran on the wall clock.

:mod:`~harbor_voice.trajectory.stepped` is for a call a driver ran in fixed
ticks. A real-time call (EVA's; Full-Duplex-Bench's later) has no ticks: its
record is per-party speech segments in seconds from call start — when
each party's audio started and stopped, what it said, whether it was cut off
— plus tool calls stamped on the same clock, usage, and the recordings. This
turns that record into the same :class:`VoiceTrajectory`, so harbor's ATIF
tooling and our metrics read a real-time call as they read the others.

The cuts are the lockstep builder's, carried over so the two read alike:

- Lines. A party's segments join into one line across silences up to
  ``max_gap_s``, unless the other party spoke in the silence. A turn
  detector splits a sentence at a breath; a reader does not.
- Exchanges are the turns: a caller line that has words (``said``) and
  every agent segment that starts after it and no later than the next such
  line (``reply``). What the agent says before the caller first speaks — a
  greeting — is exchange 0, with no ``said``. Full duplex has no strict
  turns; the timeline keeps the overlaps this cut hides.

What wall time adds over steps is measured, not inferred: overlaps are the
exact intersections of the two parties' speech, a barge-in carries how much
longer the cut party went on (``yield_latency_s``), and a tool call carries
how long it took.

Nothing here is benchmark-specific. The caller of :func:`from_timeline`
decides what a segment is (EVA's bridge decides it from audio, with its own
silence thresholds) and passes the segments in.
"""

from __future__ import annotations

from collections.abc import Iterable
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
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.model import VoiceTurn
from harbor_voice.trajectory.stepped import audio_ref


@dataclass(frozen=True, slots=True)
class Segment:
    """One stretch of one party's speech, in seconds from call start."""

    party: str
    start: float
    end: float
    text: str | None = None
    """What the party said in it, in its own words: an LLM caller's text, an
    S2S agent's own transcript. ``None`` when unknown."""
    cut_off: bool = False
    """It ended because the other party cut in."""
    heard: str | None = None
    """What the *other* party transcribed of it — an agent's speech-to-text
    of the caller — which makes said-vs-heard possible."""


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """A tool call on the call's clock."""

    t: float
    """When the call went to the tools."""
    party: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    call_id: str | None = None
    end: float | None = None
    """When its answer came back, if known."""


@dataclass(frozen=True, slots=True)
class _Line:
    party: str
    start: float
    end: float
    segments: tuple[Segment, ...]

    @property
    def text(self) -> str | None:
        return _join(s.text for s in self.segments)

    @property
    def heard(self) -> str | None:
        return _join(s.heard for s in self.segments)


def _join(parts: Iterable[str | None]) -> str | None:
    words = [p.strip() for p in parts if p and p.strip()]
    return " ".join(words) if words else None


def _r(t: float) -> float:
    return round(t, 6)


def _lines(mine: list[Segment], theirs: list[Segment], max_gap_s: float) -> list[_Line]:
    """A party's lines: segments joined across short silences, unless the
    other party spoke in the silence — then it is a new line."""
    lines: list[_Line] = []
    group: list[Segment] = []
    end = 0.0
    for seg in mine:
        silent = seg.start > end  # no silence between overlapping segments
        if group and (
            seg.start - end > max_gap_s
            or (silent and any(o.start < seg.start and o.end > end for o in theirs))
        ):
            lines.append(_Line(group[0].party, group[0].start, end, tuple(group)))
            group = []
        end = max(end, seg.end) if group else seg.end
        group.append(seg)
    if group:
        lines.append(_Line(group[0].party, group[0].start, end, tuple(group)))
    return lines


def _union(segments: list[Segment]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for seg in segments:
        if spans and seg.start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], seg.end))
        else:
            spans.append((seg.start, seg.end))
    return spans


def _overlaps(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _interrupter(
    cut: Segment, theirs: list[Segment], max_gap_s: float
) -> Segment | None:
    """Who cut ``cut`` off: the other party's last segment starting inside it,
    else their first starting just after it (a provider that stops on its
    own VAD can stop before the caller's audio is counted as started)."""
    inside = [o for o in theirs if cut.start <= o.start <= cut.end]
    if inside:
        return inside[-1]
    return next((o for o in theirs if cut.end < o.start <= cut.end + max_gap_s), None)


def from_timeline(
    segments: Sequence[Segment],
    *,
    agent: str,
    caller: str,
    tools: Sequence[ToolEvent] = (),
    duration_s: float | None = None,
    audio: Path | AudioRef | None = None,
    per_party_audio: Mapping[str, Path | AudioRef] | None = None,
    max_gap_s: float = 1.0,
    yield_deadline_s: float | None = None,
    floor_policy: str = "native",
    pipeline: Pipeline = Pipeline.S2S,
    plan: Mapping[str, Any] | None = None,
    seed_root: int | None = None,
    task: str | None = None,
    trial_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    attempt: int = 0,
    usage: Sequence[UsageRecord] = (),
) -> VoiceTrajectory:
    """Build the trajectory of a real-time call from its timeline.

    Args:
        segments: Both parties' speech, in any order.
        duration_s: The call's length; by default, its last segment or tool
            answer.
        audio: The mixed recording; ``per_party_audio`` each party's track.
            By content hash, never inlined; a path that does not
            exist is skipped, as the lockstep builder skips it.
        max_gap_s: The longest silence a line spans (see the module
            docstring). ``0`` keeps every segment its own line.
        yield_deadline_s: How soon a cut-off party must stop to have yielded
            *in time*. ``None``: stopping at all counts.
        floor_policy: What decided who spoke when. ``native`` by default:
            each party's own turn-taking, as on a real call.

    Raises:
        ValueError: a segment belongs to neither party, or runs backwards.
    """
    for seg in segments:
        if seg.party not in (agent, caller):
            raise ValueError(
                f"segment party {seg.party!r} is neither {agent!r} nor {caller!r}"
            )
        if seg.start < 0 or seg.end < seg.start:
            raise ValueError(f"segment runs backwards or before the call: {seg!r}")
    by_party = {
        party: sorted((s for s in segments if s.party == party), key=lambda s: s.start)
        for party in (caller, agent)
    }
    other = {caller: agent, agent: caller}
    lines = {
        party: _lines(by_party[party], by_party[other[party]], max_gap_s)
        for party in (caller, agent)
    }

    last = max(
        [0.0]
        + [s.end for s in segments]
        + [tool.end if tool.end is not None else tool.t for tool in tools]
    )
    events: list[TimelineEvent] = [
        TimelineEvent(t=0.0, type=EventType.SESSION_OPEN),
        TimelineEvent(
            t=_r(duration_s if duration_s is not None else last),
            type=EventType.SESSION_CLOSE,
        ),
    ]
    for line in (*lines[caller], *lines[agent]):
        events.append(
            TimelineEvent(
                t=_r(line.start), type=EventType.SPEECH_START, party=line.party
            )
        )
        events.append(
            TimelineEvent(
                t=_r(line.end),
                type=EventType.SPEECH_END,
                party=line.party,
                duration_s=_r(line.end - line.start),
            )
        )
    for lo, hi in _overlaps(_union(by_party[caller]), _union(by_party[agent])):
        events.append(
            TimelineEvent(
                t=_r(lo),
                type=EventType.OVERLAP,
                parties=(caller, agent),
                duration_s=_r(hi - lo),
            )
        )
    for party in (caller, agent):
        for seg in by_party[party]:
            if seg.cut_off:
                events.append(
                    _barge_in(seg, by_party[other[party]], max_gap_s, yield_deadline_s)
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
                t=_r(tool.t),
                type=EventType.TOOL_CALL,
                party=tool.party,
                duration_s=_r(tool.end - tool.t)
                if tool.end is not None and tool.end >= tool.t
                else None,
                meta=meta,
            )
        )
    events.sort(key=lambda e: e.t)

    per_party = {
        party: ref
        for party, source in (per_party_audio or {}).items()
        if (ref := _ref(source)) is not None
    }
    mixed = _ref(audio) if audio is not None else None
    artifacts = (
        Artifacts(mixed_timeline=mixed, per_party=per_party, tier="T1")
        if mixed is not None or per_party
        else Artifacts()
    )

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
            pipeline=pipeline,
            floor_policy=floor_policy,
            profile=str((plan or {}).get("plan_id", "clean")),
            seed_root=seed_root,
        ),
        perturbation_plan=dict(plan or {}),
        timeline=tuple(events),
        turns=_exchanges(lines[caller], by_party[agent], agent),
        artifacts=artifacts,
        usage=tuple(usage),
    )


def _ref(source: Path | AudioRef) -> AudioRef | None:
    if isinstance(source, AudioRef):
        return source
    return audio_ref(source) if source.exists() else None


def _barge_in(
    cut: Segment, theirs: list[Segment], max_gap_s: float, deadline_s: float | None
) -> TimelineEvent:
    meta: dict[str, Any] = {
        "outcome": BargeInOutcome.FIRED_AFTER_SPEECH.value,
        "native": False,
        "cut": cut.party,
    }
    by = _interrupter(cut, theirs, max_gap_s)
    if by is None:
        # Cut off by someone we never heard start: the cut is still a fact.
        meta["yielded_in_time"] = True
        return TimelineEvent(t=_r(cut.end), type=EventType.BARGE_IN, meta=meta)
    latency = max(0.0, cut.end - by.start)
    meta["yielded_in_time"] = True if deadline_s is None else latency <= deadline_s
    meta["yield_latency_s"] = _r(latency)
    meta["latency_basis"] = "reply_audio"
    if deadline_s is not None:
        meta["deadline_s"] = deadline_s
    return TimelineEvent(
        t=_r(by.start), type=EventType.BARGE_IN, party=by.party, meta=meta
    )


def _exchanges(
    caller_lines: list[_Line], agent_segments: list[Segment], agent: str
) -> tuple[VoiceTurn, ...]:
    said = [line for line in caller_lines if line.text]
    bounds = [line.start for line in said]

    def window(lo: float | None, hi: float | None) -> list[Segment]:
        # (lo, hi]: what the agent starts as the caller starts a new line is
        # cut by it, so it ends the previous reply — the lockstep rule.
        return [
            s
            for s in agent_segments
            if (lo is None or s.start > lo) and (hi is None or s.start <= hi)
        ]

    turns: list[VoiceTurn] = []
    lead = window(None, bounds[0] if bounds else None)
    if lead:
        turns.append(_turn(len(turns), agent, None, lead))
    for i, line in enumerate(said):
        hi = bounds[i + 1] if i + 1 < len(bounds) else None
        turns.append(_turn(len(turns), agent, line, window(line.start, hi)))
    return tuple(turns)


def _turn(idx: int, agent: str, said: _Line | None, reply: list[Segment]) -> VoiceTurn:
    return VoiceTurn(
        idx=idx,
        party=agent,
        said=said.text if said is not None else None,
        heard=said.heard if said is not None else None,
        reply=_join(s.text for s in reply),
        clocks=Clocks(
            tx=_r(said.start) if said is not None else None,
            rx=_r(reply[0].start) if reply else None,
        ),
        stop_reason="cut_off" if any(s.cut_off for s in reply) else None,
    )
