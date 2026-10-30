"""Compare two call-golden record directories: EVA's and harbor's.

Used by ``tests/test_eva_call_golden.py``, and from the command line::

    python call_compare.py compare EVA_DIR HARBOR_DIR          # print the report
    python call_compare.py fixture EVA_DIR FIXTURE_DIR          # check in EVA's record

A record is read as four ordered streams, each item a *key* (what happened)
and its times (when, in seconds from the call's start ``t0``):

- ``events``: ``user_simulator_events.jsonl``; ``audio_start`` / ``audio_end``
  keyed by role, with both the logged time and ``audio_timestamp``;
- ``audit``: ``audit_log.json``'s transcript (message type and value);
- ``framework``: ``framework_logs.jsonl``;
- ``metrics``: ``pipecat_metrics.jsonl`` (and each ``model_response`` value).

and three traces the harnesses keep in ``g5_meta.json``: every frame the
agent received (``agent_frames``: time, bytes, voiced), every chunk the caller
heard (``caller_heard``: time, bytes, non-silent bytes) and every alignment
pad the recorder made (``pads``: time, track, length before, target, and the
bot/user speaking flags it decided on).

The sequences must be equal key for key. Times are compared item by item.
The one text allowed to differ is an ``[interrupted]`` partial, whose length
is however far the agent had streamed when the caller cut in; its key is
normalised, and both partials are reported.

The fixture keeps EVA's logs and ``g5_meta.json`` but not its WAVs (about
3 MB): their lengths go in ``wavs.json``.
"""

from __future__ import annotations

import json
import shutil
import sys
import wave
from pathlib import Path
from typing import Any

LOGS = (
    "user_simulator_events.jsonl",
    "audit_log.json",
    "framework_logs.jsonl",
    "pipecat_metrics.jsonl",
    "transcript.jsonl",
    "response_latencies.json",
    "g5_meta.json",
)
WAVS = (
    "audio_user.wav",
    "audio_assistant.wav",
    "audio_mixed.wav",
    "audio_user_clean.wav",
)
INTERRUPTED = " [interrupted]"

Item = tuple[str, list[float]]


def _norm(value: Any) -> str:
    if isinstance(value, str):
        return "<partial>" + INTERRUPTED if value.endswith(INTERRUPTED) else value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def wav_lengths(record: Path) -> dict[str, list[int]]:
    """``{name: [frames, rate]}``, from the WAVs or from a fixture's ``wavs.json``."""
    stored = record / "wavs.json"
    if stored.exists():
        return json.loads(stored.read_text())
    out: dict[str, list[int]] = {}
    for name in WAVS:
        path = record / name
        if path.exists():
            with wave.open(str(path)) as f:
                out[name] = [f.getnframes(), f.getframerate()]
    return out


def load(record: Path) -> dict[str, Any]:
    meta = json.loads((record / "g5_meta.json").read_text())
    t0 = float(meta["t0"])

    def rel_ms(ms: Any) -> float:
        return int(ms) / 1000 - t0

    events: list[Item] = []
    partials: list[str] = []
    for e in _jsonl(record / "user_simulator_events.jsonl"):
        if "event_type" in e:
            events.append(
                (
                    f"{e['event_type']}:{e['user']}",
                    [rel_ms(e["timestamp"]), e["audio_timestamp"] - t0],
                )
            )
        else:
            events.append((f"{e['type']}:{_norm(e['data'])}", [rel_ms(e["timestamp"])]))
    audit_data = json.loads((record / "audit_log.json").read_text(encoding="utf-8"))
    audit: list[Item] = []
    for e in audit_data["transcript"]:
        value = e["value"]
        if isinstance(value, str) and value.endswith(INTERRUPTED):
            partials.append(value)
        audit.append((f"{e['message_type']}:{_norm(value)}", [rel_ms(e["timestamp"])]))
    framework: list[Item] = [
        (
            f"{e['type']}:{_norm(e['data'].get('frame') if e['type'] != 'tool_call' else e['data'])}"
            + (
                f":{e['data'].get('was_interrupted')}"
                if e["type"] == "turn_end"
                else ""
            ),
            [rel_ms(e["timestamp"])],
        )
        for e in _jsonl(record / "framework_logs.jsonl")
    ]
    metrics: list[Item] = []
    for e in _jsonl(record / "pipecat_metrics.jsonl"):
        if e["type"] == "LatencyMetric":
            metrics.append(
                (
                    f"LatencyMetric:{e['stage']}:{e['model']}",
                    [rel_ms(e["timestamp"]), e["value"]],
                )
            )
        else:
            metrics.append(
                (f"{e['type']}:{_norm(e['value'])}", [rel_ms(e["timestamp"])])
            )
    latencies_path = record / "response_latencies.json"
    latencies = (
        json.loads(latencies_path.read_text())["latencies"]
        if latencies_path.exists()
        else []
    )
    traces = {
        "agent_frames": [(f"{n}:{v}", [t]) for t, n, v in meta.get("agent_frames", [])],
        # Keyed by size: the non-silent count can differ by a byte or two,
        # since EVA's soxr conversion dithers (see call_audio_cases).
        "caller_heard": [(f"{n}", [t]) for t, n, _ in meta.get("caller_heard", [])],
        "pads": [
            (f"{k}:{a}:{b}:{bot}:{usr}", [t])
            for t, k, a, b, bot, usr in meta.get("pads", [])
        ],
    }
    return {
        "meta": meta,
        "streams": {
            "events": events,
            "audit": audit,
            "framework": framework,
            "metrics": metrics,
        },
        "traces": traces,
        "partials": partials,
        "response_latencies": latencies,
        "wavs": wav_lengths(record),
    }


def _compare_stream(a: list[Item], b: list[Item]) -> dict[str, Any]:
    keys_a, keys_b = [k for k, _ in a], [k for k, _ in b]
    out: dict[str, Any] = {"same_sequence": keys_a == keys_b, "count": [len(a), len(b)]}
    if keys_a != keys_b:
        first = next(
            (i for i, (x, y) in enumerate(zip(keys_a, keys_b)) if x != y),
            min(len(keys_a), len(keys_b)),
        )
        out["first_mismatch"] = {
            "index": first,
            "eva": keys_a[first : first + 3],
            "harbor": keys_b[first : first + 3],
        }
        return out
    worst: tuple[float, str, int] = (0.0, "", 0)
    deltas: list[list[float]] = []
    for (key, ta), (_, tb) in zip(a, b):
        d = [round((y - x) * 1000, 1) for x, y in zip(ta, tb)]
        deltas.append(d)
        for i, v in enumerate(d):
            if abs(v) > abs(worst[0]):
                worst = (v, key, i)
    out["max_abs_dt_ms"] = abs(worst[0])
    out["worst"] = {"dt_ms": worst[0], "key": worst[1], "field": worst[2]}
    out["deltas_ms"] = deltas
    return out


def compare(eva_dir: Path, harbor_dir: Path) -> dict[str, Any]:
    eva, harbor = load(eva_dir), load(harbor_dir)
    report: dict[str, Any] = {
        "end_reason": [
            eva["meta"]["conversation_ended_reason"],
            harbor["meta"]["conversation_ended_reason"],
        ],
        "stats": [eva["meta"].get("stats"), harbor["meta"].get("stats")],
        "streams": {
            name: _compare_stream(eva["streams"][name], harbor["streams"][name])
            for name in eva["streams"]
        },
        "traces": {
            name: _compare_stream(eva["traces"][name], harbor["traces"][name])
            for name in eva["traces"]
        },
        "caller_heard_voiced_max_diff": max(
            (
                abs(x[2] - y[2])
                for x, y in zip(
                    eva["meta"].get("caller_heard", []),
                    harbor["meta"].get("caller_heard", []),
                )
            ),
            default=0,
        ),
        "partials": [eva["partials"], harbor["partials"]],
        "response_latencies_ms": [
            round((y - x) * 1000, 1)
            for x, y in zip(eva["response_latencies"], harbor["response_latencies"])
        ]
        if len(eva["response_latencies"]) == len(harbor["response_latencies"])
        else None,
        "wav_frames": {
            name: [eva["wavs"].get(name), harbor["wavs"].get(name)]
            for name in sorted(set(eva["wavs"]) | set(harbor["wavs"]))
        },
    }
    return report


def compare_processor(eva: dict[str, Any], harbor: dict[str, Any]) -> dict[str, Any]:
    """EVA's processor output on both records: equal turns and texts; latency deltas."""
    exact = (
        "intended_user_turns",
        "transcribed_user_turns",
        "transcribed_assistant_turns",
        "intended_assistant_turns",
        "assistant_interrupted_turns",
        "user_interrupted_turns",
        "num_assistant_turns",
        "num_user_turns",
        "num_tool_calls",
        "tool_called",
        "tool_params",
        "tool_responses",
        "conversation_ended_reason",
    )
    out: dict[str, Any] = {"equal": {}, "latency_ms": {}, "segments_ms": {}}
    for name in exact:
        out["equal"][name] = _strip_partials(eva.get(name)) == _strip_partials(
            harbor.get(name)
        )
    out["equal"]["trace_roles"] = [
        (e.get("role"), e.get("type"), e.get("turn_id"))
        for e in eva.get("conversation_trace") or []
    ] == [
        (e.get("role"), e.get("type"), e.get("turn_id"))
        for e in harbor.get("conversation_trace") or []
    ]
    la, lb = (
        eva.get("latency_assistant_turns") or {},
        harbor.get("latency_assistant_turns") or {},
    )
    out["equal"]["latency_turns"] = sorted(la) == sorted(lb)
    out["latency_ms"] = {k: round((lb[k] - la[k]) * 1000, 1) for k in la if k in lb}
    for role in ("user", "assistant"):
        sa = eva.get(f"audio_timestamps_{role}_turns") or {}
        sb = harbor.get(f"audio_timestamps_{role}_turns") or {}
        out["equal"][f"segments_{role}_turns"] = {k: len(v) for k, v in sa.items()} == {
            k: len(v) for k, v in sb.items()
        }
        ta, tb = eva["t0"], harbor["t0"]
        out["segments_ms"][role] = {
            k: [
                [
                    round(((y0 - tb) - (x0 - ta)) * 1000, 1),
                    round(((y1 - tb) - (x1 - ta)) * 1000, 1),
                ]
                for (x0, x1), (y0, y1) in zip(sa[k], sb.get(k, []))
            ]
            for k in sa
        }
    return out


def _strip_partials(value: Any) -> Any:
    if isinstance(value, str):
        return _norm(value) if value.endswith(INTERRUPTED) else value
    if isinstance(value, dict):
        return {k: _strip_partials(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_partials(v) for v in value]
    return value


def write_fixture(eva_dir: Path, fixture: Path) -> None:
    fixture.mkdir(parents=True, exist_ok=True)
    for name in LOGS:
        if (eva_dir / name).exists():
            shutil.copyfile(eva_dir / name, fixture / name)
    (fixture / "wavs.json").write_text(json.dumps(wav_lengths(eva_dir), indent=1))


def main() -> None:
    command, a, b = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if command == "fixture":
        write_fixture(a, b)
        return
    report = compare(a, b)
    for group in ("streams", "traces"):
        for stream in report[group].values():
            stream.pop("deltas_ms", None)
    json.dump(report, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
