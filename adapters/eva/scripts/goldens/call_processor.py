"""EVA's ``MetricsContextProcessor`` over a call-golden record directory, in ``eva-base``.

::

    MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD/adapters/eva/scripts/goldens:/goldens" \\
        -v "$RECORDS:/records" --entrypoint /opt/venv/bin/python eva-base:039efa0a \\
        /goldens/call_processor.py /records/eva /records/harbor ...

For each directory, prints one JSON object (keyed by the directory's name)
with what EVA's metrics are computed from: the turns, transcripts, audio
segments, interrupt sets, per-turn latencies and the conversation trace,
as ``process_record`` builds them for an S2S pipeline. A directory without
``result.json`` (harbor's: the sidecar writes it) gets one built from
``g5_meta.json`` the way EVA's worker builds it (``worker.py:245-279``).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _result(record: Path) -> Any:
    from eva.models.results import ConversationResult

    path = record / "result.json"
    if path.exists():
        data = json.loads(path.read_text())
        data["output_dir"] = str(record)
        for key in ("audio_assistant_path", "audio_user_path", "audio_mixed_path"):
            if data.get(key):
                data[key] = str(record / Path(data[key]).name)
        for key in ("pipecat_logs_path", "user_simulator_logs_path"):
            if data.get(key):
                data[key] = str(record / Path(data[key]).name)
        return ConversationResult.model_validate(data)
    meta = json.loads((record / "g5_meta.json").read_text())
    stats = meta.get("stats") or {}
    now = datetime.now()
    return ConversationResult(
        record_id="g5",
        completed=True,
        started_at=now,
        ended_at=now,
        duration_seconds=0.0,
        output_dir=str(record),
        audio_assistant_path=str(record / "audio_assistant.wav"),
        audio_user_path=str(record / "audio_user_clean.wav"),
        audio_mixed_path=str(record / "audio_mixed.wav"),
        transcript_path=str(record / "transcript.jsonl"),
        audit_log_path=str(record / "audit_log.json"),
        pipecat_logs_path=str(record / "framework_logs.jsonl"),
        user_simulator_logs_path=str(record / "user_simulator_events.jsonl"),
        num_turns=stats.get("num_turns", 0),
        num_tool_calls=stats.get("num_tool_calls", 0),
        tools_called=stats.get("tools_called", []),
        conversation_ended_reason=meta.get("conversation_ended_reason"),
    )


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, set):
        return sorted(_plain(v) for v in value)
    return value


FIELDS = (
    "intended_user_turns",
    "transcribed_user_turns",
    "intended_assistant_turns",
    "transcribed_assistant_turns",
    "audio_timestamps_user_turns",
    "audio_timestamps_assistant_turns",
    "assistant_interrupted_turns",
    "user_interrupted_turns",
    "latency_assistant_turns",
    "num_assistant_turns",
    "num_user_turns",
    "num_tool_calls",
    "tool_called",
    "tool_params",
    "tool_responses",
    "conversation_trace",
    "conversation_ended_reason",
)


def process(record: Path) -> dict[str, Any]:
    from eva.metrics.processor import MetricsContextProcessor
    from eva.models.config import PipelineType

    meta = json.loads((record / "g5_meta.json").read_text())
    context = MetricsContextProcessor().process_record(
        _result(record), record, PipelineType.S2S
    )
    if context is None:
        return {"error": "process_record returned None"}
    out = {name: _plain(getattr(context, name, None)) for name in FIELDS}
    out["t0"] = meta["t0"]
    out["history_len"] = len(context.history)
    return out


def main() -> None:
    results = {Path(arg).name: process(Path(arg)) for arg in sys.argv[1:]}
    json.dump(results, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
