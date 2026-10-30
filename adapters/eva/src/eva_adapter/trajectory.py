"""An EVA call as ATIF-voice, read from its record directory.

The record directory is EVA's (``records/<id>/trial_<k>/``, as the sidecar's
``/files`` returns it, or the call loop's own copy before ``/record``). The
timeline comes from what EVA's metrics read, so the trajectory and EVA's
scores describe the same call:

- **Segments** are the bridge's ``audio_start`` / ``audio_end`` pairs in
  ``user_simulator_events.jsonl`` (``simulated_user`` is the caller,
  ``assistant`` the agent): EVA's own speech edges, back-dated as EVA's
  bridge back-dates them, in epoch seconds.
- **Words.** Each party's own words and what the other party heard:

  ====================  ===============================  ============================
  party                 ``text`` (its own words)         ``heard`` (the other's STT)
  ====================  ===============================  ============================
  caller                ``user_speech`` (its LLM's text) audit log ``user`` entries
                                                         (the agent's transcription)
  agent                 audit log ``assistant`` entries  ``assistant_speech``
                        (its own transcript)             (ElevenLabs' transcription)
  ====================  ===============================  ============================

  Each line is attached to that party's segment it belongs to: the last one
  starting no later than :data:`ATTACH_SLACK_S` after the line was stamped.
  EVA stamps a user entry at speech start and an assistant entry at its first
  audio, and the caller's text can arrive just before its audio does. An
  assistant entry ending ``[interrupted]`` (EVA's partial) marks its segment
  cut off.
- **Tools** are the audit log's ``tool_call`` / ``tool_response`` pairs, on
  the same clock.

Time zero is the earliest stamp in the record (the caller's ``connected``),
from epoch stamps only: ``result.json``'s times are naive local times of the
sidecar's clock and zone, which need not be this process's. Audio is
referenced by content hash, never inlined.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.timed import Segment
from harbor_voice.trajectory.timed import ToolEvent
from harbor_voice.trajectory.timed import from_timeline

from eva_adapter.events import EVENTS_FILE

AGENT = "agent"
CALLER = "caller"
ROLES = {"simulated_user": CALLER, "assistant": AGENT}
"""EVA's ``user`` field on audio events → our party ids."""
INTERRUPTED = "[interrupted]"
"""EVA's suffix on an assistant entry the caller cut off."""
ATTACH_SLACK_S = 0.5
FLOOR_POLICY = "native"
"""Who spoke when was each party's own doing: ElevenLabs' turn-taking for
the caller, the Realtime server's VAD for the agent."""

AUDIT_LOG = "audit_log.json"
AUDIO_MIXED = "audio_mixed.wav"
AUDIO_PER_PARTY = {CALLER: "audio_user.wav", AGENT: "audio_assistant.wav"}
"""EVA's aligned tracks (24 kHz): the caller as the agent heard it, the agent
before μ-law."""


@dataclass(slots=True)
class _Span:
    party: str
    start: float
    end: float
    text: list[str] = field(default_factory=list)
    heard: list[str] = field(default_factory=list)
    cut_off: bool = False


@dataclass(frozen=True, slots=True)
class _Line:
    party: str
    t: float
    text: str
    kind: str
    """``text`` (the party's own words) or ``heard``."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _ms(stamp: Any) -> float | None:
    try:
        return float(stamp) / 1000.0
    except (TypeError, ValueError):
        return None


def spans(events: Sequence[Mapping[str, Any]]) -> list[_Span]:
    """The bridge's speech segments, epoch seconds, in log order.

    A start while the role is already open is ignored, as an end with none
    open is; a segment still open when the log ends closes at the last stamp.
    """
    out: list[_Span] = []
    open_: dict[str, float] = {}
    last = 0.0
    for event in events:
        stamp = _ms(event.get("timestamp"))
        if stamp is not None:
            last = max(last, stamp)
        kind = event.get("event_type")
        party = ROLES.get(str(event.get("user")))
        if party is None or kind not in ("audio_start", "audio_end"):
            continue
        t = float(event["audio_timestamp"])
        last = max(last, t)
        if kind == "audio_start":
            open_.setdefault(party, t)
        elif party in open_:
            start = open_.pop(party)
            out.append(_Span(party, start, max(start, t)))
    for party, start in open_.items():
        out.append(_Span(party, start, max(start, last)))
    return out


def lines(
    events: Sequence[Mapping[str, Any]], transcript: Sequence[Mapping[str, Any]]
) -> list[_Line]:
    """Every transcript line of either party, stamped in epoch seconds."""
    out: list[_Line] = []
    for event in events:
        kind = event.get("type")
        text = (event.get("data") or {}).get("text")
        t = _ms(event.get("timestamp"))
        if t is None or not isinstance(text, str):
            continue
        if kind == "user_speech":
            out.append(_Line(CALLER, t, text, "text"))
        elif kind == "assistant_speech":
            out.append(_Line(AGENT, t, text, "heard"))
    for entry in transcript:
        kind = entry.get("message_type")
        value = entry.get("value")
        t = _ms(entry.get("timestamp"))
        if t is None or not isinstance(value, str):
            continue
        if kind == "user":
            out.append(_Line(CALLER, t, value, "heard"))
        elif kind == "assistant":
            out.append(_Line(AGENT, t, value, "text"))
    return out


def _attach(segments: list[_Span], line: _Line) -> None:
    mine = sorted((s for s in segments if s.party == line.party), key=lambda s: s.start)
    if not mine:
        # Words with no audio the bridge saw (cut before it played): a point.
        mine = [_Span(line.party, line.t, line.t)]
        segments.append(mine[0])
    target = mine[0]
    for seg in mine:
        if seg.start <= line.t + ATTACH_SLACK_S:
            target = seg
    text = line.text
    if line.kind == "text" and line.party == AGENT and text.endswith(INTERRUPTED):
        target.cut_off = True
        text = text[: -len(INTERRUPTED)].rstrip()
    (target.text if line.kind == "text" else target.heard).append(text)


def tool_events(transcript: Sequence[Mapping[str, Any]], t0: float) -> list[ToolEvent]:
    """The audit log's tool calls, each with the response that follows it."""
    out: list[ToolEvent] = []
    pending: list[tuple[float, Mapping[str, Any]]] = []
    for entry in transcript:
        kind = entry.get("message_type")
        value = entry.get("value")
        t = _ms(entry.get("timestamp"))
        if t is None or not isinstance(value, Mapping):
            continue
        if kind == "tool_call":
            pending.append((t, value))
        elif kind == "tool_response" and pending:
            index = next(
                (
                    i
                    for i, (_, v) in enumerate(pending)
                    if v.get("tool") == value.get("tool")
                ),
                0,
            )
            started, call = pending.pop(index)
            response = value.get("response")
            error = None
            if isinstance(response, Mapping) and response.get("status") == "error":
                error = str(response.get("message") or response.get("error") or "error")
            out.append(
                ToolEvent(
                    t=max(0.0, started - t0),
                    party=AGENT,
                    name=str(call.get("tool")),
                    arguments=dict(call.get("parameters") or {}),
                    result=response,
                    error=error,
                    end=max(0.0, t - t0),
                )
            )
    for started, call in pending:
        out.append(
            ToolEvent(
                t=max(0.0, started - t0),
                party=AGENT,
                name=str(call.get("tool")),
                arguments=dict(call.get("parameters") or {}),
                error="no response logged",
            )
        )
    return sorted(out, key=lambda e: e.t)


def _stamps(
    events: Iterable[Mapping[str, Any]], transcript: Iterable[Mapping[str, Any]]
) -> list[float]:
    out = [t for e in events if (t := _ms(e.get("timestamp"))) is not None]
    out += [float(e["audio_timestamp"]) for e in events if "audio_timestamp" in e]
    out += [t for e in transcript if (t := _ms(e.get("timestamp"))) is not None]
    return out


def build(
    record_dir: Path,
    *,
    usage: Sequence[UsageRecord] = (),
    task: str | None = None,
    trial_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    attempt: int = 0,
    seed_root: int | None = None,
    plan: Mapping[str, Any] | None = None,
) -> VoiceTrajectory:
    """The call in ``record_dir`` as ATIF-voice.

    ``plan`` is the call's perturbation (EVA's condition,
    :meth:`eva_adapter.conditions.PerturbationConfig.plan`); none is clean.
    """
    events = _read_jsonl(record_dir / EVENTS_FILE)
    transcript = list(_read_json(record_dir / AUDIT_LOG).get("transcript") or [])
    stamps = _stamps(events, transcript)
    t0 = min(stamps) if stamps else 0.0
    end = max(stamps) if stamps else 0.0

    segments = spans(events)
    for line in sorted(lines(events, transcript), key=lambda line: line.t):
        _attach(segments, line)
    timeline = [
        Segment(
            party=s.party,
            start=round(max(0.0, s.start - t0), 6),
            end=round(max(0.0, s.end - t0), 6),
            text=" ".join(s.text) if s.text else None,
            cut_off=s.cut_off,
            heard=" ".join(s.heard) if s.heard else None,
        )
        for s in segments
    ]
    per_party = {
        party: record_dir / name
        for party, name in AUDIO_PER_PARTY.items()
        if (record_dir / name).exists()
    }
    mixed = record_dir / AUDIO_MIXED
    return from_timeline(
        timeline,
        agent=AGENT,
        caller=CALLER,
        tools=tool_events(transcript, t0),
        duration_s=round(end - t0, 6),
        audio=mixed if mixed.exists() else None,
        per_party_audio=per_party,
        floor_policy=FLOOR_POLICY,
        plan=plan,
        seed_root=seed_root,
        task=task,
        trial_id=trial_id,
        agent_name=agent_name,
        model=model,
        attempt=attempt,
        usage=usage,
    )
