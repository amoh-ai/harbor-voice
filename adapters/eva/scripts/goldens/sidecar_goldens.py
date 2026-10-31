"""Sidecar goldens: EVA's own code, run natively, for the sidecar to match.

Each golden is produced here by EVA's code as EVA's orchestrator runs it
(no sidecar code involved), and checked in under
``tests/adapters/eva/fixtures/sidecar/``. ``tests/adapters/eva/test_eva_sidecar_goldens.py``
runs the sidecar image and compares its HTTP answers with them; harbor's
tests never import EVA.

- ``bundle``: the inputs. EVA's recorded run ``tests/artifacts/records/1.1.2``
  (verbatim; EVA is MIT-licensed) plus two short synthetic WAVs, so the
  audio judges have tracks to hear: ``eva_record_1.1.2.tar.gz``.

The recording predates today's scenario data, so its tool results are not
all what EVA's tools answer now. The goldens below use the native
record (``native_record``): the recording, with its tool calls run again
through EVA's tool path and EVA's ``AuditLog`` writing their results, as
EVA's server would write this call today.

- ``g1``: EVA's processor on the native record; it must equal EVA's own
  golden ``expected_processor_context.json`` except the tool results that
  changed (asserted here). Plus the latency stats EVA's worker computes from
  the record's files: ``g1_record.json``.
- ``g3``: EVA's own re-entry path (``BenchmarkRunner.validate_existing``,
  i.e. ``eva --run-id``) validates and grades the native record under the
  fake judges of ``sidecar_fake_judges.py``; every judge request is
  captured: ``g3_judges.json``.
- ``g4``: the record's tool calls through EVA's executor: each result as
  the model is sent it and as EVA's audit log holds it (serialised when the
  call ends), and the DB hashes: ``g4_tools.json``.
- ``g6``: EVA's ``--aggregate-only`` (``MetricsRunner.run_aggregate_only``)
  on a seeded synthetic set of ``metrics.json`` files, 6 records x 3 trials:
  ``g6_aggregation.json``.
- ``g8``: for all 213 records (English), what EVA sends at the start of a
  call: its Realtime server's ``session.update`` and greeting item (its own
  ``_handle_session`` against a recording client) and the caller's prompt and
  agent (its ElevenLabs simulator's own ``_run_elevenlabs_conversation``
  against a recording SDK): ``g8_open.json``.

Regenerate inside EVA's image, with EVA's ``tests/`` mounted (not in it)::

    MSYS_NO_PATHCONV=1 docker run --rm \\
        -v "$PWD/adapters/eva:/adapter" -v "$EVA_REPO/tests:/eva-tests:ro" \\
        -v "$PWD/tests/adapters/eva/fixtures:/fixtures" \\
        --entrypoint /opt/venv/bin/python eva-base:039efa0a \\
        /adapter/scripts/goldens/sidecar_goldens.py all
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import math
import os
import random
import shutil
import struct
import sys
import tarfile
import tempfile
import wave
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import sidecar_fake_judges  # noqa: E402

from eva.assistant.agentic.audit_log import AuditLog, ConversationMessage  # noqa: E402
from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer  # noqa: E402
from eva.assistant.tools.tool_executor import ToolExecutor, execute_and_log_tool  # noqa: E402
from eva.metrics.processor import MetricsContextProcessor  # noqa: E402
from eva.metrics.runner import MetricsRunner  # noqa: E402
from eva.models.config import RunConfig  # noqa: E402
from eva.models.record import EvaluationRecord  # noqa: E402
from eva.models.results import ConversationResult, MetricScore, RecordMetrics  # noqa: E402
from eva.orchestrator import runner as eva_runner  # noqa: E402
from eva.orchestrator.runner import BenchmarkRunner  # noqa: E402
from eva.orchestrator.worker import ConversationWorker  # noqa: E402
from eva.user_simulator import elevenlabs as eva_elevenlabs  # noqa: E402
from eva.utils.hash_utils import get_dict_hash  # noqa: E402

ADAPTER = Path(os.environ.get("EVA_ADAPTER_DIR", "/adapter"))
EVA_TESTS = Path(os.environ.get("EVA_TESTS_DIR", "/eva-tests"))
FIXTURES = Path(os.environ.get("EVA_FIXTURES_DIR", "/fixtures"))
OUT = FIXTURES / "sidecar"
BUNDLE = OUT / "eva_record_1.1.2.tar.gz"

RECORD = "1.1.2"
DOMAIN = "airline"
RECORD_FILES = (
    "audit_log.json",
    "elevenlabs_events.jsonl",
    "pipecat_logs.jsonl",
    "pipecat_metrics.jsonl",
    "response_latencies.json",
    "transcript.jsonl",
)
"""What harbor's call loop would hand ``/record`` (EVA's file formats)."""
EVA_FILES = (
    "result.json",
    "initial_scenario_db.json",
    "final_scenario_db.json",
    "expected_processor_context.json",
)
"""What EVA's own code wrote for the record, for comparison only."""

REFERENCE_MODEL = {
    "s2s": "gpt-realtime-1.5",
    "s2s_params": {"model": "gpt-realtime-1.5", "api_key": "unused"},
}
CASCADE_MODEL = {
    "llm": "gpt-5.2",
    "stt": "deepgram",
    "stt_params": {"model": "nova-3", "api_key": "unused"},
    "tts": "cartesia",
    "tts_params": {"model": "sonic", "api_key": "unused"},
}
"""A cascade pipeline, for EVA's processor golden (recorded on a cascade run)."""
PLACEHOLDER_MODEL_LIST = [
    {"model_name": "unconfigured", "litellm_params": {"model": "unconfigured"}}
]


def sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def wire(obj: Any) -> str:
    """JSON as it goes on the wire: key order kept, no ASCII escaping."""
    return json.dumps(obj, ensure_ascii=False)


def write(name: str, data: Any) -> None:
    path = OUT / name
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {path}")


def run_config(
    output_dir: Path,
    *,
    model: dict[str, Any],
    framework: str,
    run_id: str | None = None,
) -> RunConfig:
    fields: dict[str, Any] = {
        "domain": DOMAIN,
        "language": "en",
        "framework": framework,
        "model": model,
        "num_trials": 3,
        "output_dir": output_dir,
        "model_list": PLACEHOLDER_MODEL_LIST,
    }
    if run_id is not None:
        fields["run_id"] = run_id
    return RunConfig(**fields)


def records(domain: str) -> list[EvaluationRecord]:
    return EvaluationRecord.load_dataset(f"data/{domain}_dataset.json")


# ── bundle ──────────────────────────────────────────────────────────────────


def tone(rate: int, seconds: float, freq: float) -> bytes:
    """A deterministic mono PCM16 WAV: integer samples from ``math.sin``."""
    frames = b"".join(
        struct.pack("<h", int(round(8000 * math.sin(2 * math.pi * freq * n / rate))))
        for n in range(int(rate * seconds))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buffer.getvalue()


def bundle() -> None:
    source = EVA_TESTS / "artifacts" / "records" / RECORD
    entries: dict[str, bytes] = {
        f"record/{n}": (source / n).read_bytes() for n in RECORD_FILES
    }
    entries["record/audio_assistant.wav"] = tone(24000, 1.0, 440.0)
    entries["record/audio_user_clean.wav"] = tone(16000, 1.0, 330.0)
    entries.update({f"eva/{n}": (source / n).read_bytes() for n in EVA_FILES})
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(entries):
            info = tarfile.TarInfo(name)
            info.size = len(entries[name])
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(entries[name]))
    BUNDLE.write_bytes(gzip.compress(raw.getvalue(), mtime=0))
    print(f"wrote {BUNDLE}")


def unbundle(dest: Path) -> None:
    with tarfile.open(BUNDLE, "r:gz") as tar:
        tar.extractall(dest, filter="data")


async def native_record(dest: Path) -> dict[str, Any]:
    """EVA's recorded 1.1.2, as EVA would write it today for the same calls.

    The recording's tool calls run again on the scenario DB EVA's worker
    materialises today, through EVA's single tool path
    (``execute_and_log_tool``) into an EVA ``AuditLog``. The recording's
    audit log keeps its timestamps and turns, with each tool result replaced
    by the one EVA's log holds now, and ``AuditLog.save`` writes it: results
    are serialised when the call ends, as EVA's server serialises them. The
    DBs are written as the server writes them (``_save_scenario_dbs``).

    Returns the tool calls with both forms of each result: ``response``,
    what the model was sent (serialised at once), and ``logged``, what EVA's
    audit log holds.
    """
    with tempfile.TemporaryDirectory() as tmp:
        unbundle(Path(tmp))
        shutil.copytree(Path(tmp) / "record", dest)
        recorded = json.loads((Path(tmp) / "record" / "audit_log.json").read_text())
    config = run_config(
        dest.parent / "output", model=REFERENCE_MODEL, framework="openai_realtime"
    )
    runner = BenchmarkRunner(config)
    record = next(r for r in records(DOMAIN) if r.id == RECORD)
    worker = ConversationWorker(
        config=config,
        record=record,
        agent=runner.agent,
        agent_config_path=str(config.agent_config_path),
        scenario_base_path=str(config.tool_mocks_path),
        output_dir=dest.parent / "worker",
        port=0,
        output_id=f"{RECORD}/trial_0",
    )
    db_path = worker._materialize_resolved_scenario_db()
    # As AbstractAssistantServer builds it (base_server.py:81-87).
    executor = ToolExecutor(
        tool_config_path=str(config.agent_config_path),
        scenario_db_path=str(db_path),
        tool_module_path=runner.agent.tool_module_path,
        current_date_time=record.current_date_time,
    )
    log = AuditLog()
    calls = []
    for entry in (
        e for e in recorded["transcript"] if e["message_type"] == "tool_call"
    ):
        name, arguments = entry["value"]["tool"], entry["value"]["parameters"]
        result = await execute_and_log_tool(executor, log, name, arguments)
        # openai_realtime_server.py:621-627: what the model is sent, at once.
        calls.append(
            {
                "name": name,
                "arguments": arguments,
                "response": json.loads(json.dumps(result)),
            }
        )
    live = [
        e["value"]["response"]
        for e in log.transcript
        if e["message_type"] == "tool_response"
    ]
    results = [
        e for e in recorded["transcript"] if e["message_type"] == "tool_response"
    ]
    march = [e["value"]["response"] for e in results]
    for entry, result in zip(results, live, strict=True):
        entry["value"]["response"] = result
    native = AuditLog()
    native.transcript = recorded["transcript"]
    native.llm_prompts = recorded["llm_prompts"]
    native.conversation_messages = [
        ConversationMessage.model_validate(m) for m in recorded["conversation_messages"]
    ]
    native.save(dest / "audit_log.json")
    for name, db in (
        ("initial_scenario_db.json", executor.original_db),
        ("final_scenario_db.json", executor.db),
    ):
        with open(dest / name, "w") as f:
            json.dump(db, f, indent=2, sort_keys=True, default=str, ensure_ascii=False)
    logged = json.loads((dest / "audit_log.json").read_text())["transcript"]
    logged_results = [
        e["value"]["response"] for e in logged if e["message_type"] == "tool_response"
    ]
    for call, result, then in zip(calls, logged_results, march, strict=True):
        call["logged"] = result
        call["matches_recording"] = result == then
    return {
        "calls": calls,
        "initial_scenario_db_hash": get_dict_hash(
            json.loads((dest / "initial_scenario_db.json").read_text())
        ),
        "final_scenario_db_hash": get_dict_hash(
            json.loads((dest / "final_scenario_db.json").read_text())
        ),
    }


def processor_json(ctx: Any) -> dict[str, Any]:
    """A processor context as ``metrics.json`` stores it (JSON types)."""
    fields = (
        "num_assistant_turns",
        "num_user_turns",
        "num_tool_calls",
        "tool_called",
        "conversation_ended_reason",
        "transcribed_assistant_turns",
        "transcribed_user_turns",
        "intended_assistant_turns",
        "intended_user_turns",
        "audio_timestamps_assistant_turns",
        "audio_timestamps_user_turns",
        "tool_params",
        "tool_responses",
        "latency_assistant_turns",
        "conversation_trace",
    )
    out = {name: json.loads(json.dumps(getattr(ctx, name))) for name in fields}
    out["assistant_interrupted_turns"] = sorted(ctx.assistant_interrupted_turns)
    out["user_interrupted_turns"] = sorted(ctx.user_interrupted_turns)
    out["pipeline_type"] = ctx.pipeline_type.value
    return out


# ── g1 ──────────────────────────────────────────────────────────────────────


async def g1() -> None:
    """EVA's processor on the record, and the stats EVA's worker computes.

    The processor runs on the native record (cascade, as recorded). Its
    context must be EVA's own golden ``expected_processor_context.json``
    except where today's tool results differ from the recording's
    (``g4_tools.json``: ``matches_recording``); that is asserted here, and
    the context is written for the sidecar to match exactly.
    """
    with tempfile.TemporaryDirectory() as tmp:
        unbundle(Path(tmp) / "bundle")
        record_dir = Path(tmp) / "record"
        tools = await native_record(record_dir)
        result = json.loads((Path(tmp) / "bundle" / "eva" / "result.json").read_text())
        result.update(
            output_dir=str(record_dir),
            pipecat_logs_path=str(record_dir / "pipecat_logs.jsonl"),
            elevenlabs_logs_path=str(record_dir / "elevenlabs_events.jsonl"),
            audio_assistant_path=None,
            audio_user_path=None,
            audio_mixed_path=None,
        )
        ctx = MetricsContextProcessor().process_record(
            ConversationResult(**result), record_dir
        )
        assert ctx is not None
        context = processor_json(ctx)

        expected = json.loads(
            (
                Path(tmp) / "bundle" / "eva" / "expected_processor_context.json"
            ).read_text()
        )
        changed = [
            i for i, call in enumerate(tools["calls"]) if not call["matches_recording"]
        ]
        for i in changed:
            expected["tool_responses"][i]["tool_response"] = tools["calls"][i]["logged"]
        responses = [
            e
            for e in expected["conversation_trace"]
            if e.get("type") == "tool_response"
        ]
        for i in changed:
            responses[i]["tool_response"] = tools["calls"][i]["logged"]
        trace = [
            {k: v for k, v in e.items() if k != "timestamp"}
            for e in context["conversation_trace"]
        ]
        assert trace == expected["conversation_trace"]
        for name in expected:
            if name in context and name != "conversation_trace":
                assert context[name] == expected[name], name

        # The worker's own stats on these files (worker.py:239-243).
        worker = ConversationWorker.__new__(ConversationWorker)
        worker.output_dir = record_dir
        latency = {
            "llm_latency": worker._calculate_llm_latency(),
            "stt_latency": worker._calculate_stt_latency(),
            "tts_latency": worker._calculate_tts_latency(),
            "model_response_latency": worker._calculate_model_response_latency(),
        }
    write(
        "g1_record.json",
        {
            "record": f"{DOMAIN}/{RECORD}",
            "context": context,
            "latency": {
                k: v.model_dump(mode="json") if v else None for k, v in latency.items()
            },
            "eva_golden_changed_tool_results": changed,
        },
    )


# ── g3 ──────────────────────────────────────────────────────────────────────


async def g3() -> None:
    """EVA's re-entry path on the record, every judge request captured."""
    fake = sidecar_fake_judges.install()

    async def no_preflight(config: RunConfig) -> None:
        raise RuntimeError("the golden must not run a conversation")

    eva_runner.run_preflight = no_preflight  # type: ignore[assignment]
    with tempfile.TemporaryDirectory() as tmp:
        unbundle(Path(tmp) / "bundle")
        await native_record(Path(tmp) / "native")
        output = Path(tmp) / "output"
        config = run_config(
            output,
            model=REFERENCE_MODEL,
            framework="openai_realtime",
            run_id="g3-native",
        )
        await BenchmarkRunner(config).run([])  # writes config.json as EVA does
        run_dir = output / "g3-native"
        result = json.loads((Path(tmp) / "bundle" / "eva" / "result.json").read_text())
        for trial in range(config.num_trials):
            record_dir = run_dir / "records" / RECORD / f"trial_{trial}"
            shutil.copytree(Path(tmp) / "native", record_dir)
            # The paths EVA's worker writes today (worker.py:258-265); EVA's
            # recording predates the clean caller track's name.
            result.update(
                output_dir=str(record_dir),
                audio_assistant_path=str(record_dir / "audio_assistant.wav"),
                audio_user_path=str(record_dir / "audio_user_clean.wav"),
                audio_mixed_path=str(record_dir / "audio_mixed.wav"),
                transcript_path=str(record_dir / "transcript.jsonl"),
                audit_log_path=str(record_dir / "audit_log.json"),
                conversation_log_path=str(record_dir / "logs.log"),
                pipecat_logs_path=str(record_dir / "pipecat_logs.jsonl"),
                elevenlabs_logs_path=str(record_dir / "elevenlabs_events.jsonl"),
            )
            (record_dir / "result.json").write_text(json.dumps(result, indent=2))
        rerun = BenchmarkRunner.from_existing_run(run_dir)
        record = next(r for r in records(DOMAIN) if r.id == RECORD)
        await rerun.validate_existing(run_dir, [record], force_revalidation=True)

        summary = json.loads((run_dir / "evaluation_summary.json").read_text())
        assert summary["simulation"]["successful_records"] == config.num_trials, summary
        counts: dict[str, int] = {}
        for request in fake.requests:
            key = json.dumps(request, sort_keys=True)
            counts[key] = counts.get(key, 0) + 1
        # The three trials are copies: each request must come once per trial.
        assert all(n == config.num_trials for n in counts.values()), sorted(
            counts.values()
        )
        metrics = json.loads(
            (run_dir / "records" / RECORD / "trial_0" / "metrics.json").read_text()
        )
        judge_prompts = {
            name: sha(score["details"]["judge_prompt"])
            for name, score in metrics["metrics"].items()
            if "judge_prompt" in (score.get("details") or {})
        }
        write(
            "g3_judges.json",
            {
                "record": f"{DOMAIN}/{RECORD}",
                "requests": sorted(
                    (json.loads(k) for k in counts),
                    key=lambda r: json.dumps(r, sort_keys=True),
                ),
                "judge_prompts": dict(sorted(judge_prompts.items())),
                "metrics": sorted(metrics["metrics"]),
            },
        )


# ── g4 ──────────────────────────────────────────────────────────────────────


async def g4() -> None:
    """The record's tool calls through EVA's executor, on today's scenario DB:
    what the model is sent, what EVA's audit log holds, and the DB hashes."""
    with tempfile.TemporaryDirectory() as tmp:
        tools = await native_record(Path(tmp) / "record")
    write("g4_tools.json", {"record": f"{DOMAIN}/{RECORD}", **tools})


# ── g6 ──────────────────────────────────────────────────────────────────────


def synthetic_metrics(rng: random.Random, record_id: str) -> RecordMetrics:
    """One trial's metrics: every composite component, some errored or skipped."""

    def score(name: str, value: float | None, **extra: Any) -> MetricScore:
        return MetricScore(name=name, score=value, normalized_score=value, **extra)

    completed = rng.random() < 0.6
    metrics = {
        "task_completion": score("task_completion", 1.0 if completed else 0.0),
        "faithfulness": score("faithfulness", rng.choice([0.0, 0.5, 1.0])),
        "agent_speech_fidelity": score(
            "agent_speech_fidelity", rng.choice([0.9, 0.95, 1.0])
        ),
        "conversation_progression": score(
            "conversation_progression", rng.choice([0.0, 0.5, 1.0])
        ),
        "turn_taking": score("turn_taking", round(rng.uniform(0.5, 1.0), 4)),
        "conciseness": score("conciseness", round(rng.uniform(0.3, 1.0), 4)),
        "response_speed": MetricScore(
            name="response_speed", score=round(rng.uniform(0.5, 3.0), 4)
        ),
        "time_to_completion": MetricScore(
            name="time_to_completion",
            score=round(rng.uniform(120, 400), 2) if completed else None,
            skipped=not completed,
        ),
        "turns_to_completion": MetricScore(
            name="turns_to_completion",
            score=float(rng.randint(6, 20)) if completed else None,
            skipped=not completed,
        ),
    }
    roll = rng.random()
    if roll < 0.1:
        metrics["faithfulness"] = MetricScore(
            name="faithfulness", score=0.0, error="judge failed"
        )
    elif roll < 0.2:
        metrics["agent_speech_fidelity"] = MetricScore(
            name="agent_speech_fidelity", score=None, skipped=True
        )
    return RecordMetrics(record_id=record_id, context={}, metrics=metrics)


async def g6() -> None:
    rng = random.Random(6)
    run_id = "g6-aggregation"
    inputs: dict[str, Any] = {}
    for record_id in ("1.1.2", "1.2.1", "2.1.1", "3.2.1", "5.1.1", "7.4.1"):
        for trial in range(3):
            oid = f"{record_id}/trial_{trial}"
            inputs[oid] = json.loads(synthetic_metrics(rng, oid).model_dump_json())
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / run_id
        for oid, data in inputs.items():
            path = run_dir / "records" / oid / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(data, indent=2))
        await MetricsRunner.run_aggregate_only(run_dir, num_draws=3)
        summary = json.loads((run_dir / "metrics_summary.json").read_text())
        aggregates = {
            oid: json.loads((run_dir / "records" / oid / "metrics.json").read_text())[
                "aggregate_metrics"
            ]
            for oid in inputs
        }
    write(
        "g6_aggregation.json",
        {
            "run_id": run_id,
            "num_draws": 3,
            "inputs": inputs,
            "aggregates": aggregates,
            "summary": summary,
        },
    )


# ── g8 ──────────────────────────────────────────────────────────────────────


class _Recorder:
    """The OpenAI Realtime client and connection, recording EVA's calls."""

    def __init__(self) -> None:
        self.model: str | None = None
        self.sent: list[dict[str, Any]] = []
        self.realtime = self
        self.session = self
        self.conversation = self
        self.item = self
        self.response = self

    def connect(self, model: str) -> _Recorder:
        self.model = model
        return self

    async def __aenter__(self) -> _Recorder:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def update(self, session: dict[str, Any]) -> None:
        self.sent.append({"type": "session.update", "session": session})

    async def create(self, item: dict[str, Any] | None = None) -> None:
        if item is None:
            self.sent.append({"type": "response.create"})
        else:
            self.sent.append({"type": "conversation.item.create", "item": item})

    def __aiter__(self) -> _Recorder:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class _Socket:
    async def receive_text(self) -> str:
        return json.dumps({"event": "stop"})

    async def send_text(self, text: str) -> None:
        return None


class _Captured(Exception):
    pass


async def caller_start(simulator: Any) -> dict[str, Any]:
    """What EVA's ElevenLabs simulator opens the caller's conversation with."""
    seen: dict[str, Any] = {}

    class Interface:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def start_async(self) -> None:
            pass

        async def stop_async(self) -> None:
            pass

        def get_latencies(self) -> list[float]:
            return []

    class Conversation:
        def __init__(
            self, client: Any, agent_id: str, *, config: Any, **kwargs: Any
        ) -> None:
            seen["agent_id"] = agent_id
            seen["dynamic_variables"] = dict(config.dynamic_variables)
            raise _Captured

    async def no_sleep(seconds: float) -> None:
        return None

    with (
        mock.patch.object(eva_elevenlabs, "ElevenLabsAudioInterface", Interface),
        mock.patch.object(eva_elevenlabs, "ElevenLabs", lambda **kwargs: None),
        mock.patch.object(eva_elevenlabs, "Conversation", Conversation),
        mock.patch.object(eva_elevenlabs.asyncio, "sleep", no_sleep),
    ):
        try:
            await simulator._run_elevenlabs_conversation("unused")
        except _Captured:
            pass
    return seen


async def open_record(
    config: RunConfig, runner: BenchmarkRunner, record: EvaluationRecord, out: Path
) -> dict[str, Any]:
    worker = ConversationWorker(
        config=config,
        record=record,
        agent=runner.agent,
        agent_config_path=str(config.agent_config_path),
        scenario_base_path=str(config.tool_mocks_path),
        output_dir=out / record.id,
        port=0,
        output_id=f"{record.id}/trial_0",
    )

    async def no_start(self: Any) -> None:
        return None

    with mock.patch.object(OpenAIRealtimeAssistantServer, "start", no_start):
        await worker._start_assistant()
    await worker._start_user_simulator()
    server = worker._assistant_server
    assert isinstance(server, OpenAIRealtimeAssistantServer)
    recorder = _Recorder()
    with mock.patch.object(server, "_create_client", lambda: recorder):
        await server._handle_session(_Socket())  # type: ignore[arg-type]
    assert [e["type"] for e in recorder.sent] == [
        "session.update",
        "conversation.item.create",
        "response.create",
    ]
    session = recorder.sent[0]["session"]
    caller = await caller_start(worker._user_simulator)
    prompt = caller["dynamic_variables"]["prompt"]
    return {
        "domain": config.domain,
        "record_id": record.id,
        "model": recorder.model,
        "system_prompt": sha(session["instructions"]),
        "tools": sha(wire(session["tools"])),
        "tool_count": len(session["tools"]),
        "session": sha(wire(session)),
        "greeting": server.initial_message,
        "opening_text": recorder.sent[1]["item"]["content"][0]["text"],
        "caller_prompt": sha(prompt),
        "caller_prompt_chars": len(prompt),
        "caller_variables": sorted(caller["dynamic_variables"]),
        "gender": caller["agent_id"].removeprefix("agent-"),
        "language": config.language.value,
    }


async def g8() -> None:
    os.environ["EVA_EN_USER_F"] = "agent-F"
    os.environ["EVA_EN_USER_M"] = "agent-M"
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for domain in ("airline", "itsm", "medical_hr"):
            fields: dict[str, Any] = {
                "domain": domain,
                "language": "en",
                "framework": "openai_realtime",
                "model": REFERENCE_MODEL,
                "num_trials": 3,
                "output_dir": Path(tmp),
                "model_list": PLACEHOLDER_MODEL_LIST,
            }
            config = RunConfig(**fields)
            runner = BenchmarkRunner(config)
            for record in records(domain):
                rows.append(
                    await open_record(config, runner, record, Path(tmp) / domain)
                )
    counts = {
        d: sum(1 for r in rows if r["domain"] == d)
        for d in ("airline", "itsm", "medical_hr")
    }
    assert counts == {"airline": 50, "itsm": 80, "medical_hr": 83}, counts
    write("g8_open.json", {"language": "en", "records": rows})


def main(argv: list[str]) -> None:
    steps = argv[1:] or ["all"]
    if steps == ["all"]:
        steps = ["bundle", "g1", "g3", "g4", "g6", "g8"]
    OUT.mkdir(parents=True, exist_ok=True)
    for step in steps:
        if step == "bundle":
            bundle()
        elif step in ("g1", "g3", "g4", "g6", "g8"):
            asyncio.run({"g1": g1, "g3": g3, "g4": g4, "g6": g6, "g8": g8}[step]())
        else:
            raise SystemExit(f"unknown step {step!r}")


if __name__ == "__main__":
    main(sys.argv)
