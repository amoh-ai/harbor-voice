"""EVA sidecar: EVA's own code behind HTTP, for harbor-voice's EVA adapter.

Runs inside EVA's image at the pinned commit (``eva-base:039efa0a``, built
from EVA's Dockerfile; see ``Dockerfile``), with ``cwd`` at EVA's repo root:
EVA loads ``data/``, ``configs/`` and its tool modules by relative path.

It does not run a call. harbor-voice runs the call (the caller, the agent
under test, the audio path, the clock) and comes here for everything EVA
decides by code. Each endpoint drives EVA's own classes the way EVA's
orchestrator drives them for one record (``BenchmarkRunner._run`` and
``ConversationWorker.run``), never a re-implementation:

- ``POST /open``: build one attempt of one record. EVA's worker materialises
  the scenario DB with names resolved for the language and builds the
  assistant server (``_start_assistant``) and the caller
  (``_start_user_simulator``). Returned: what EVA's OpenAI Realtime server
  sends the model when the call starts (``session.update``: prompt and tools;
  the ``Say: '<greeting>'`` item), captured from its own ``_handle_session``,
  and the caller's prompt (``_build_prompt``) and gender, as EVA's ElevenLabs
  simulator picks its agent.
- ``POST /tool``: one tool call, through the assistant server's
  ``execute_tool`` (EVA's ``execute_and_log_tool`` on the scenario DB).
- ``POST /record``: the record files harbor's call loop wrote (EVA's
  formats) plus ``call.json``. The sidecar completes the record directory as
  the server's ``stop()`` and the worker do: the scenario DBs, the DB hashes,
  the latency stats and ``result.json``.
- ``POST /validate``: ``ValidationRunner.validate_one`` and the attempt
  bookkeeping of ``BenchmarkRunner._run``: the time-limit acceptance rule,
  the rerun decision, ``{id}_failed_attempt_N`` archives.
- ``POST /grade``: ``MetricsRunner.run_and_save_record`` on the attempt that
  counts, then EVA's final pass; returns the metrics and the EVA composites
  (``compute_record_aggregates``).
- ``GET /files``: a tar of the run directory, archived attempts included.

The run directory is a native EVA run directory (``config.json``,
``records/<id>/trial_<k>/``, ``evaluation_summary.json``,
``metrics_summary.json``): its ``config.json`` is the one EVA writes for the
reference system, ``framework=openai_realtime`` with ``model.s2s =
gpt-realtime-1.5``, so the S2S variants of EVA's metrics apply as in a native
EVA run. The run config is built as EVA's CLI builds it, so ``EVA_*``
variables in the container's environment (``EVA_MODEL_LIST`` for the judges,
...) apply as they do to EVA. The task passes only ``EVA_MODEL_LIST`` of EVA's
own (an empty ``EVA_*`` fails EVA's config); EVA's ``metric_configs`` comes
through the sidecar's own ``EVA_SIDECAR_METRIC_CONFIGS`` instead (JSON in
``EVA_METRIC_CONFIGS``' format, empty for EVA's defaults), which the sidecar
hands to ``RunConfig`` (another audio judge, say).

One session per container, as in tau-voice: a trial gets its own container.
Each attempt of the trial reopens the same session with ``POST /open``.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import posixpath
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal
from unittest import mock

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydantic import Field
from starlette.background import BackgroundTask

import eva
from eva.assistant.agentic.audit_log import ConversationMessage
from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer
from eva.metrics.runner import MetricsRunner
from eva.models.config import PipelineType
from eva.models.config import RunConfig
from eva.models.record import EvaluationRecord
from eva.models.results import ConversationResult
from eva.orchestrator import worker as eva_worker
from eva.orchestrator.runner import BenchmarkRunner
from eva.orchestrator.validation_runner import ValidationResult
from eva.orchestrator.validation_runner import ValidationRunner
from eva.orchestrator.worker import ConversationWorker
from eva.user_simulator.base import AbstractUserSimulator
from eva.user_simulator.elevenlabs import _PERSONA_GENDER
from eva.utils import router
from eva.utils.conversation_checks import check_conversation_finished
from eva.utils.conversation_checks import find_records_with_llm_generic_error
from eva.utils.error_handler import create_error_details
from eva.utils.hash_utils import get_dict_hash
from eva.utils.llm_client import LLMClient
from eva.utils.logging import add_record_log_file
from eva.utils.logging import current_record_id
from eva.utils.logging import get_logger
from eva.utils.logging import remove_record_log_file
from eva.utils.logging import setup_logging
from eva.utils.provenance import capture_provenance
from eva.utils.provenance import resolve_tool_module_file

logger = get_logger(__name__)

OUTPUT_DIR = Path(os.environ.get("EVA_SIDECAR_OUTPUT_DIR", "/tmp/eva-sidecar"))
"""Where run directories go (EVA's ``output/``). Container-local: harbor
fetches a trial's files with ``GET /files``."""

REFERENCE_FRAMEWORK = "openai_realtime"
REFERENCE_MODEL: dict[str, Any] = {
    "s2s": "gpt-realtime-1.5",
    "s2s_params": {"model": "gpt-realtime-1.5"},
}
"""EVA's reference S2S system, the agent under test in parity runs. Only the
pipeline type and the session settings come from it here: the sidecar never
calls the model."""

UNUSED_KEY = "unused-by-the-sidecar"
"""EVA requires an ``api_key`` in every service's params; the sidecar opens no
connection to the agent under test, so it never needs one. ``config.json``
redacts it like any key."""

PLACEHOLDER_MODEL_LIST = [
    {"model_name": "unconfigured", "litellm_params": {"model": "unconfigured"}}
]
"""``RunConfig.model_list`` when ``EVA_MODEL_LIST`` is unset: enough for
prompts and tools; a judge then fails with EVA's "Router not initialized"."""

METRIC_CONFIGS_ENV = "EVA_SIDECAR_METRIC_CONFIGS"
"""EVA's ``metric_configs`` for this trial, as JSON (``EVA_METRIC_CONFIGS``'
format); unset or empty is EVA's defaults. Not an ``EVA_*`` setting of EVA's
(EVA ignores it): the task passes it as ``${...:-}``, and an empty
``EVA_METRIC_CONFIGS`` would fail EVA's config."""


def _metric_configs() -> dict[str, dict[str, Any]] | None:
    """:data:`METRIC_CONFIGS_ENV`, parsed; ``None`` when unset or empty. A
    malformed value raises ``ValueError``: ``/open`` refuses the trial."""
    raw = os.environ.get(METRIC_CONFIGS_ENV, "").strip()
    if not raw:
        return None
    configs = json.loads(raw)
    if not isinstance(configs, dict) or not all(
        isinstance(v, dict) for v in configs.values()
    ):
        raise ValueError(
            f"{METRIC_CONFIGS_ENV} must be a JSON object of objects, keyed by metric"
        )
    return configs


CALL_FILE = "call.json"
OWN_FILES = frozenset(
    {
        "result.json",
        "metrics.json",
        "logs.log",
        "scenario_db.json",
        "initial_scenario_db.json",
        "final_scenario_db.json",
    }
)
"""Record files only EVA's code writes (the worker, the server, the metrics
runner); a ``/record`` upload carrying one is refused."""

API_VERSION = 1
"""The HTTP API this sidecar serves, reported by ``/health``. The driver
(``eva_adapter.client.SIDECAR_API``) checks it before a call: bump both when a
route or a request model changes (``tests/adapters/test_sidecar_api.py``)."""

app = FastAPI(title="EVA sidecar")


class OpenRequest(BaseModel):
    domain: Literal["airline", "itsm", "medical_hr"]
    record_id: str
    language: str = "en"
    trial: int = Field(
        0, ge=0, description="the trial's index k; its record dir is trial_k"
    )
    num_trials: int = Field(3, ge=1, description="EVA's num_trials (k of pass@k)")
    framework: str = REFERENCE_FRAMEWORK
    model: dict[str, Any] | None = Field(
        default=None,
        description="EVA's RunConfig.model; the reference system if omitted",
    )

    def key(self) -> str:
        return self.model_dump_json()


class ToolRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class CallInfo(BaseModel):
    """``call.json``: what EVA's worker knows of a call that harbor-voice ran."""

    attempt: int = Field(ge=1)
    conversation_ended_reason: str | None = None
    """The caller's end reason (EVA's ``run_conversation`` return value), or
    ``time_limit_exceeded`` when the worker's guard cut the call off."""
    started_at: datetime
    ended_at: datetime
    error: str | None = None
    """Set when the call raised: EVA's worker then reports ``error``."""


class ValidateRequest(BaseModel):
    attempt: int = Field(ge=1)


class _Unstarted:
    """Mixin: the assistant server is built, never started (nothing listens)."""

    async def start(self) -> None:
        return None


def _unstarted(
    server_class: type[AbstractAssistantServer],
) -> type[AbstractAssistantServer]:
    return type(server_class.__name__, (_Unstarted, server_class), {})


class _Namespace:
    def __init__(self, **attrs: Any) -> None:
        self.__dict__.update(attrs)


class _RecordingConnection:
    """Stands in for the OpenAI Realtime connection; records what EVA sends,
    as the client events the SDK would put on the socket.

    Iterating it yields no server event, so EVA's session handler ends as
    soon as it has configured the session and asked for the greeting.
    """

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.session = _Namespace(update=self._session_update)
        self.conversation = _Namespace(item=_Namespace(create=self._item_create))
        self.response = _Namespace(create=self._response_create)

    async def _session_update(self, session: dict[str, Any]) -> None:
        self.sent.append({"type": "session.update", "session": session})

    async def _item_create(self, item: dict[str, Any]) -> None:
        self.sent.append({"type": "conversation.item.create", "item": item})

    async def _response_create(self) -> None:
        self.sent.append({"type": "response.create"})

    def __aiter__(self) -> _RecordingConnection:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class _RecordingClient:
    """``AsyncOpenAI`` as EVA's server uses it: ``realtime.connect(model=...)``."""

    def __init__(self) -> None:
        self.connection = _RecordingConnection()
        self.model: str | None = None
        self.realtime = self

    def connect(self, model: str) -> _RecordingClient:
        self.model = model
        return self

    async def __aenter__(self) -> _RecordingConnection:
        return self.connection

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _StoppedSocket:
    """The caller's socket, already sending Twilio's ``stop``."""

    async def receive_text(self) -> str:
        return json.dumps({"event": "stop"})

    async def send_text(self, text: str) -> None:
        return None


async def _opening(server: OpenAIRealtimeAssistantServer) -> dict[str, Any]:
    """What EVA's Realtime server sends the model when a call starts.

    Its own ``_handle_session`` runs against a recording client: the
    ``session.update`` (``_build_session_config``: prompt, tools, voice,
    VAD), then the ``Say: '<greeting>'`` user item and ``response.create``.
    """
    client = _RecordingClient()
    server._create_client = lambda: client  # type: ignore[method-assign]
    try:
        await server._handle_session(_StoppedSocket())  # type: ignore[arg-type]
    finally:
        del server._create_client
    sent = client.connection.sent
    kinds = [event["type"] for event in sent]
    if kinds != ["session.update", "conversation.item.create", "response.create"]:
        raise RuntimeError(f"EVA's session start changed shape: {kinds}")
    return {"model": client.model, "events": sent}


def _with_placeholder_keys(model: dict[str, Any]) -> dict[str, Any]:
    model = json.loads(json.dumps(model))
    for name, params in model.items():
        if name.endswith("_params") and isinstance(params, dict):
            params.setdefault("api_key", UNUSED_KEY)
    return model


def _run_config(request: OpenRequest) -> RunConfig:
    """EVA's run config for this trial, built as EVA's CLI builds it (env and
    defaults), with the trial's own fields given explicitly."""
    fields: dict[str, Any] = {
        "domain": request.domain,
        "language": request.language,
        "framework": request.framework,
        "model": _with_placeholder_keys(request.model or REFERENCE_MODEL),
        "num_trials": request.num_trials,
        "output_dir": OUTPUT_DIR,
    }
    if "EVA_MODEL_LIST" not in os.environ:
        fields["model_list"] = PLACEHOLDER_MODEL_LIST
    metric_configs = _metric_configs()
    if metric_configs is not None:
        fields["metric_configs"] = metric_configs
    config = RunConfig(**fields)
    unrouted = _unrouted_judges(config)
    if unrouted:
        raise ValueError(
            "judge deployments whose endpoint variable is empty: "
            + ", ".join(unrouted)
            + " (LiteLLM would send them to its provider's default endpoint, "
            "OpenAI's for an OpenAI-compatible proxy, instead of failing)"
        )
    return config


def _unrouted_judges(config: RunConfig) -> list[str]:
    """The judge deployments this run's metrics ask for whose ``api_base`` is
    an ``os.environ/`` reference to an unset or empty variable (a deployment
    behind a proxy of your own, say, whose URL variable is not set)."""
    wanted = {
        name
        for metric in config.metric_configs.values()
        for name in (metric.get("audio_judge_model"), metric.get("judge_model"))
        if name
    }
    if os.environ.get("JUDGE_MODEL"):
        wanted.add(os.environ["JUDGE_MODEL"])
    found = []
    for deployment in config.model_list:
        base = (deployment.get("litellm_params") or {}).get("api_base")
        if deployment.get("model_name") in wanted and isinstance(base, str):
            name = base.removeprefix("os.environ/")
            if base.startswith("os.environ/") and not os.environ.get(name):
                found.append(f"{deployment['model_name']} ({name} is empty)")
    return found


def _write_run_config(runner: BenchmarkRunner) -> None:
    """``config.json``, as ``BenchmarkRunner.run`` writes it (runner.py:165-199)."""
    config = runner.config
    if not config.tool_module_path:
        config.tool_module_path = runner.agent.tool_module_path
    try:
        tool_module_file = resolve_tool_module_file(config.tool_module_path)
        config.provenance = capture_provenance(
            config, tool_module_file=tool_module_file
        )
    except Exception as e:
        logger.warning(f"Failed to capture provenance: {e}")
    (runner.output_dir / "records").mkdir(parents=True, exist_ok=True)
    if config.model.pipeline_type == PipelineType.CASCADE:
        stt_params = config.model.stt_params or {}
        tts_params = config.model.tts_params or {}
        config.resolved_models = {
            "stt_provider": config.model.stt,
            "stt_model": stt_params["model"],
            "stt_alias": stt_params.get("alias"),
            "tts_provider": config.model.tts,
            "tts_model": tts_params["model"],
            "tts_alias": tts_params.get("alias"),
            "llm": config.model.llm,
        }
    config_data = config.model_dump(mode="json")
    config_data["pipeline_parts"] = config.model.pipeline_parts
    (runner.output_dir / "config.json").write_text(
        json.dumps(config_data, indent=2, ensure_ascii=False)
    )


_RECORDS: dict[str, dict[str, EvaluationRecord]] = {}


def _record(config: RunConfig, record_id: str) -> EvaluationRecord:
    domain = config.domain
    if domain not in _RECORDS:
        _RECORDS[domain] = {
            r.id: r for r in EvaluationRecord.load_dataset(config.dataset_path)
        }
    record = _RECORDS[domain].get(record_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"record {record_id!r} not in {config.dataset_path}"
        )
    return record


def _local(moment: datetime) -> datetime:
    """EVA stamps ``result.json`` with naive local times (``datetime.now()``)."""
    return moment.astimezone().replace(tzinfo=None) if moment.tzinfo else moment


def _conversation_result(
    worker: ConversationWorker, call: CallInfo, stats: dict[str, Any]
) -> ConversationResult:
    """``ConversationWorker.run``'s result for a call harbor-voice ran.

    worker.py:203-305, with the worker's own helpers; only the times and the
    end reason come from ``call.json`` instead of the worker's clock.
    """
    output_dir = worker.output_dir
    record_id = worker.record.id
    started_at = _local(call.started_at)
    ended_at = _local(call.ended_at)
    duration = (ended_at - started_at).total_seconds()
    if call.error is not None:
        # worker.py:206-218: a conversation that raised gets no hashes or stats.
        return ConversationResult(
            record_id=record_id,
            completed=False,
            error=call.error,
            error_details=create_error_details(
                error=RuntimeError(call.error), retry_count=0, retry_succeeded=False
            ),
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration,
            output_dir=str(output_dir),
            conversation_ended_reason="error",
        )
    reason = call.conversation_ended_reason
    try:
        with open(output_dir / "initial_scenario_db.json") as f:
            initial_db = json.load(f)
        with open(output_dir / "final_scenario_db.json") as f:
            final_db = json.load(f)
        return ConversationResult(
            record_id=record_id,
            completed=reason != "error",
            error=None,
            error_details=None,
            llm_latency=worker._calculate_llm_latency(),
            stt_latency=worker._calculate_stt_latency(),
            tts_latency=worker._calculate_tts_latency(),
            model_response_latency=worker._calculate_model_response_latency(),
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration,
            output_dir=str(output_dir),
            audio_assistant_path=str(output_dir / "audio_assistant.wav"),
            audio_user_path=str(output_dir / "audio_user_clean.wav"),
            audio_mixed_path=str(output_dir / "audio_mixed.wav"),
            transcript_path=str(output_dir / "transcript.jsonl"),
            audit_log_path=str(output_dir / "audit_log.json"),
            conversation_log_path=str(output_dir / "logs.log"),
            pipecat_logs_path=worker._resolve_framework_logs_path(),
            user_simulator_logs_path=str(output_dir / "user_simulator_events.jsonl"),
            num_turns=stats.get("num_turns", 0),
            num_tool_calls=stats.get("num_tool_calls", 0),
            tools_called=stats.get("tools_called", []),
            conversation_ended_reason=reason,
            initial_scenario_db_hash=get_dict_hash(initial_db),
            final_scenario_db_hash=get_dict_hash(final_db),
        )
    except Exception as e:
        # worker.py:283-305: post-conversation processing failed.
        logger.error(
            f"Post-conversation processing failed for {record_id}: {e}", exc_info=True
        )
        return ConversationResult(
            record_id=record_id,
            completed=False,
            error=f"Post-conversation processing failed: {e}",
            error_details=create_error_details(
                error=e, retry_count=0, retry_succeeded=False
            ),
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration,
            output_dir=str(output_dir),
            conversation_ended_reason=reason,
            num_turns=stats.get("num_turns", 0),
            num_tool_calls=stats.get("num_tool_calls", 0),
            tools_called=stats.get("tools_called", []),
        )


def _member_name(member: tarfile.TarInfo) -> str:
    return posixpath.normpath(member.name)


def _tool_calls(transcript: list[dict[str, Any]]) -> list[tuple[Any, Any]]:
    return [
        (entry.get("value", {}).get("tool"), entry.get("value", {}).get("parameters"))
        for entry in transcript
        if entry.get("message_type") == "tool_call"
    ]


def _tool(entry: dict[str, Any]) -> Any:
    return entry.get("value", {}).get("tool")


def _validation(vr: ValidationResult | None) -> dict[str, Any] | None:
    if vr is None:
        return None
    return {
        "passed": vr.passed,
        "failed_metrics": list(vr.failed_metrics),
        "scores": dict(vr.scores),
        "details": vr.details,
    }


class _Session:
    """One trial of one record: EVA's run directory, its attempts, its grade."""

    def __init__(self, request: OpenRequest) -> None:
        self.request = request
        self.config = _run_config(request)
        if request.trial >= self.config.num_trials:
            raise HTTPException(
                status_code=422,
                detail=f"trial {request.trial} >= num_trials {self.config.num_trials}",
            )
        self.record = _record(self.config, request.record_id)
        # run_benchmark.py:26-27: the judges' router and their concurrency.
        if "EVA_MODEL_LIST" in os.environ:
            router.init(self.config.model_list)
        LLMClient.set_max_concurrency(self.config.max_concurrent_judges)
        self.runner = BenchmarkRunner(self.config)
        self.run_dir = self.runner.output_dir
        self.output_id = f"{self.record.id}/trial_{request.trial}"
        self.record_dir = self.run_dir / "records" / self.output_id
        _write_run_config(self.runner)
        self.validation = ValidationRunner(
            run_dir=self.run_dir,
            dataset=[self.record],
            thresholds=self.config.validation_thresholds,
            metric_configs=self.runner._metric_configs,
        )
        self.metrics_runner: MetricsRunner | None = None
        self.started_at = datetime.now()
        self.attempt = 0
        self.state: Literal["open", "recorded", "validated"] = "open"
        self.final: Literal["passed", "failed"] | None = None
        self.tool_calls = 0
        self.rerun_history: list[dict[str, Any]] = []
        self.time_limit_attempts = 0
        self.time_limit_validation: dict[int, ValidationResult] = {}
        self.time_limit_accepted = False
        self.last_validation: dict[str, Any] | None = None
        self.log_handler: Any = None
        self.worker: ConversationWorker
        self.server: AbstractAssistantServer
        self.simulator: AbstractUserSimulator
        self.opening: dict[str, Any] | None = None

    @property
    def pristine(self) -> bool:
        """Nothing has happened yet that a replacement session would lose."""
        return self.attempt == 1 and self.state == "open" and self.tool_calls == 0

    @property
    def max_attempts(self) -> int:
        return self.config.max_rerun_attempts

    async def start_attempt(self) -> None:
        """One attempt, set up as ``ConversationWorker.run`` sets one up."""
        if self.attempt:
            # runner.py:297-300: archive the previous attempt before the next.
            self.runner._archive_failed_attempt(self.output_id, self.attempt)
        self.attempt += 1
        self.state = "open"
        self.tool_calls = 0
        self.last_validation = None
        self.record_dir.mkdir(parents=True, exist_ok=True)
        current_record_id.set(self.output_id)
        self.log_handler = add_record_log_file(
            self.output_id, str(self.record_dir / "logs.log")
        )
        self.worker = ConversationWorker(
            config=self.config,
            record=self.record,
            agent=self.runner.agent,
            agent_config_path=str(self.config.agent_config_path),
            scenario_base_path=str(self.config.tool_mocks_path),
            output_dir=self.record_dir,
            port=0,
            output_id=self.output_id,
        )
        get_server_class = eva_worker._get_server_class
        with mock.patch.object(
            eva_worker, "_get_server_class", lambda fw: _unstarted(get_server_class(fw))
        ):
            await self.worker._start_assistant()
        await self.worker._start_user_simulator()
        server = self.worker._assistant_server
        simulator = self.worker._user_simulator
        assert server is not None and simulator is not None
        self.server = server
        self.simulator = simulator
        self.opening = (
            await _opening(server)
            if isinstance(server, OpenAIRealtimeAssistantServer)
            else None
        )

    def end_attempt_log(self) -> None:
        if self.log_handler is not None:
            remove_record_log_file(self.log_handler)
            self.log_handler = None

    def describe(self) -> dict[str, Any]:
        agent: dict[str, Any] = {
            "system_prompt": None,
            "tools": None,
            "greeting": self.server.initial_message,
            "opening_text": None,
            "session": None,
            "model": None,
        }
        if self.opening is not None:
            session_update, greeting_item, _ = self.opening["events"]
            session = session_update["session"]
            agent.update(
                system_prompt=session["instructions"],
                tools=session["tools"],
                opening_text=greeting_item["item"]["content"][0]["text"],
                session=session,
                model=self.opening["model"],
            )
        persona = self.simulator.persona_config
        return {
            "conversation_id": self.output_id,
            "output_id": self.output_id,
            "run_id": self.config.run_id,
            "domain": self.config.domain,
            "record_id": self.record.id,
            "trial": self.request.trial,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "time_limit_s": self.config.conversation_time_limit_seconds,
            "caller_timeout_s": self.worker._conversation_guard_timeout_seconds(),
            "agent": agent,
            "caller": {
                "prompt": self.simulator._build_prompt(),
                # elevenlabs.py:31,147-148: the persona picks EVA_<LANG>_USER_<F|M>.
                "gender": _PERSONA_GENDER[persona["user_persona_id"]],
                "language": self.config.language.value,
            },
        }

    async def call_tool(self, request: ToolRequest) -> dict[str, Any]:
        if self.state != "open":
            raise HTTPException(
                status_code=409,
                detail=f"attempt {self.attempt} is {self.state}; tools run during a call",
            )
        self.tool_calls += 1
        try:
            result = await self.server.execute_tool(request.name, request.arguments)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"EVA's tool executor raised: {e}"
            ) from e
        # openai_realtime_server.py:621-627: the function_call_output EVA sends.
        return {"result": result, "output": json.dumps(result, ensure_ascii=False)}

    def store(self, tar: tarfile.TarFile, call: CallInfo) -> dict[str, Any]:
        """Write the call loop's files, then complete the record as EVA does."""
        if self.state != "open":
            raise HTTPException(
                status_code=409,
                detail=f"attempt {self.attempt} is already {self.state}",
            )
        if call.attempt != self.attempt:
            raise HTTPException(
                status_code=409,
                detail=f"call.json is attempt {call.attempt}; this is attempt {self.attempt}",
            )
        members = [
            m for m in tar.getmembers() if m.isfile() and _member_name(m) != CALL_FILE
        ]
        own = sorted({_member_name(m) for m in members} & OWN_FILES)
        if own:
            raise HTTPException(
                status_code=422,
                detail=f"EVA's code writes these, not the call loop: {own}",
            )
        tar.extractall(self.record_dir, members=members, filter="data")

        server = self.server
        audit_log_path = self.record_dir / "audit_log.json"
        if audit_log_path.exists():
            self._merge_audit_log(json.loads(audit_log_path.read_text()))
        # As EVA's stop() writes it (base_server.py:233): the call loop's
        # entries, with the tool results EVA's executor holds; or, if the call
        # loop wrote none (it failed early), the server's own.
        server.audit_log.save(audit_log_path)
        if not (self.record_dir / "transcript.jsonl").exists():
            server._save_transcript()
        # base_server.py:308-322: the DBs, from the executor's state.
        server._save_scenario_dbs()
        result = _conversation_result(
            self.worker, call, server.get_conversation_stats()
        )
        (self.record_dir / "result.json").write_text(result.model_dump_json(indent=2))
        self.state = "recorded"
        self.end_attempt_log()
        return {
            "record_dir": str(self.record_dir),
            "output_id": self.output_id,
            "attempt": self.attempt,
            "result": json.loads(result.model_dump_json()),
        }

    def _merge_audit_log(self, uploaded: dict[str, Any]) -> None:
        """Put the call loop's audit log into the server's ``AuditLog``.

        The call loop holds the conversation (user and assistant turns, and
        the tool entries it wrote around each ``/tool``); the tools it logged
        must be the ones run here, in order. Each tool result is then the
        object EVA's executor returned, as in EVA's own audit log: EVA keeps
        the live result and serialises it when the call ends, so a result
        that shares objects with the scenario DB shows the DB as the call
        left it (``search_rebooking_options`` then shows the seat a later
        ``rebook_flight`` took), not what the model was sent.
        """
        server = self.server
        transcript = uploaded.get("transcript", [])
        own = server.audit_log.transcript
        logged, executed = _tool_calls(transcript), _tool_calls(own)
        results = [e for e in transcript if e.get("message_type") == "tool_response"]
        live = [e for e in own if e.get("message_type") == "tool_response"]
        if logged != executed or [_tool(e) for e in results] != [
            _tool(e) for e in live
        ]:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "audit_log.json's tool entries differ from the calls /tool ran",
                    "logged": [name for name, _ in logged],
                    "logged_results": [_tool(e) for e in results],
                    "executed": [name for name, _ in executed],
                },
            )
        for entry, result in zip(results, live, strict=True):
            entry["value"]["response"] = result["value"]["response"]
        try:
            messages = [
                ConversationMessage.model_validate(m)
                for m in uploaded.get("conversation_messages", [])
            ]
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=f"audit_log.json conversation_messages: {e}"
            ) from e
        server.audit_log.transcript = transcript
        server.audit_log.llm_prompts = uploaded.get("llm_prompts", [])
        server.audit_log.conversation_messages = messages

    async def _validate_one(
        self, *, skip_gate: bool = False
    ) -> ValidationResult | None:
        """runner.py:328-349: validation, with sample-data problems as not_finished."""
        try:
            return await self.validation.validate_one(
                self.output_id, skip_gate=skip_gate
            )
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning(f"Skipping {self.output_id} as invalid: {exc}")
        except Exception as exc:
            logger.error(f"Pipeline error for {self.output_id}: {exc}", exc_info=True)
        return None

    def _result(self) -> ConversationResult | None:
        path = self.record_dir / "result.json"
        if not path.exists():
            return None
        return ConversationResult(**json.loads(path.read_text()))

    async def validate(self, attempt: int) -> dict[str, Any]:
        """One attempt through ``BenchmarkRunner._run``'s validation step.

        runner.py:328-495 for a single output id: validate; on failure,
        classify it (not_finished / time_limit_exceeded / validation_failed),
        apply the time-limit acceptance rule, and decide on a rerun. A record
        that runs out of attempts is archived as EVA archives it.
        """
        if attempt != self.attempt:
            raise HTTPException(
                status_code=409,
                detail=f"attempt {attempt} is not the current attempt {self.attempt}",
            )
        if self.state == "validated" and self.last_validation is not None:
            return self.last_validation
        if self.state != "recorded":
            raise HTTPException(
                status_code=409,
                detail=f"attempt {attempt} is {self.state}; POST /record first",
            )
        oid = self.output_id
        result = self._result()
        vr = await self._validate_one()
        passed = vr is not None and vr.passed
        gate = vr is not None and (vr.passed or bool(vr.failed_metrics))
        reason = "passed"
        time_limit_vr: ValidationResult | None = None
        accepted: int | None = None
        if not passed:
            if not gate:
                # runner.py:387-403
                hit_time_limit = (
                    result is not None
                    and result.conversation_ended_reason == "time_limit_exceeded"
                )
                reason = "time_limit_exceeded" if hit_time_limit else "not_finished"
                if hit_time_limit:
                    self.time_limit_attempts += 1
                    time_limit_vr = await self.validation.validate_one(
                        oid, skip_gate=True
                    )
                    self.time_limit_validation[attempt] = time_limit_vr
                self.rerun_history.append({"attempt": attempt, "reason": reason})
            else:
                # runner.py:404-416
                assert vr is not None
                reason = "validation_failed"
                entry: dict[str, Any] = {
                    "attempt": attempt,
                    "reason": reason,
                    "failed_metrics": vr.failed_metrics,
                    "scores": vr.scores,
                }
                if vr.details:
                    failure_details = {
                        name: vr.details[name]
                        for name in vr.failed_metrics
                        if name in vr.details
                    }
                    if failure_details:
                        entry["failure_details"] = failure_details
                self.rerun_history.append(entry)
            accepted = self._accept_at_time_limit(attempt)
            if accepted is not None:
                passed = True
                reason = "time_limit_accepted"
        rerun = not passed and attempt < self.max_attempts
        if passed:
            self.final = "passed"
        elif not rerun:
            self.final = "failed"
        if self.final is not None:
            self._write_evaluation_summary()
        self.state = "validated"
        counted = time_limit_vr or vr
        self.last_validation = {
            "attempt": attempt,
            "passed": passed,
            "gate": gate,
            "reason": reason,
            "scores": dict(counted.scores) if counted else {},
            "failed_metrics": list(counted.failed_metrics) if counted else [],
            "validation": _validation(vr),
            "time_limit_validation": _validation(time_limit_vr),
            "restored_attempt": accepted if accepted not in (None, attempt) else None,
            "rerun": rerun,
            "attempts_left": self.max_attempts - attempt,
            "final": self.final,
            "rerun_history": self.rerun_history,
        }
        return self.last_validation

    def _accept_at_time_limit(self, attempt: int) -> int | None:
        """runner.py:418-489: a record that hit the time limit often enough is
        accepted if this attempt, or an archived one, passed the LLM
        validation with the gate bypassed. Returns the attempt accepted (an
        archived one is restored as the record), or None."""
        oid = self.output_id
        max_time_limit_attempts = int(
            self.config.validation_thresholds.get("max_time_limit_attempts", 1)
        )
        if self.time_limit_attempts < max_time_limit_attempts:
            return None
        cached = self.time_limit_validation
        current_vr = cached.get(attempt)
        if current_vr and current_vr.passed:
            logger.info(
                f"Record {oid} hit time limit {self.time_limit_attempts} times, "
                f"current attempt passed LLM validation - accepting"
            )
            self.runner._accept_time_limit_record(oid, [oid], [], None, [])
            self.time_limit_accepted = True
            return attempt
        records = self.run_dir / "records"
        for prev_attempt, prev_vr in cached.items():
            if prev_attempt == attempt or not prev_vr.passed:
                continue
            logger.info(
                f"Record {oid} hit time limit {self.time_limit_attempts} times, "
                f"restoring attempt {prev_attempt} which passed LLM validation"
            )
            archive_dir = records / f"{oid}_failed_attempt_{prev_attempt}"
            if not archive_dir.exists():
                logger.warning(
                    f"Record {oid}: archive dir for attempt {prev_attempt} not found at "
                    f"{archive_dir} - cannot restore, skipping"
                )
                continue
            if self.record_dir.exists():
                shutil.move(
                    str(self.record_dir),
                    str(records / f"{oid}_failed_attempt_{attempt}"),
                )
            shutil.move(str(archive_dir), str(self.record_dir))
            self.runner._accept_time_limit_record(oid, [oid], [], None, [])
            self.time_limit_accepted = True
            return prev_attempt
        logger.info(
            f"Record {oid} hit time limit {self.time_limit_attempts} times, "
            f"no attempt passed LLM validation - staying pending"
        )
        return None

    def _write_evaluation_summary(self) -> None:
        """``evaluation_summary.json`` for this trial (runner.py:500-592).

        A record out of attempts is archived after its failure is classified,
        as EVA does at the end of a run.
        """
        oid = self.output_id
        failed_ids = {oid} if self.final == "failed" else set()
        successful_ids = {oid} if self.final == "passed" else set()
        not_finished_count = sum(
            1
            for failed in failed_ids
            if not check_conversation_finished(self.run_dir / "records" / failed)
        )
        final_failures: dict[str, dict] = {}
        for failed in failed_ids:
            if self.rerun_history:
                final_failures[failed] = self.rerun_history[-1]
            elif check_conversation_finished(self.run_dir / "records" / failed):
                final_failures[failed] = {"reason": "validation_failed"}
            else:
                final_failures[failed] = {"reason": "not_finished"}
        for failed in failed_ids:
            self.runner._archive_failed_attempt(failed, self.attempt)
        generic_error_ids = find_records_with_llm_generic_error(
            self.run_dir, successful_ids
        )
        ended_at = datetime.now()
        summary = {
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": (ended_at - self.started_at).total_seconds(),
            "simulation": {
                "total_records": 1,
                "successful_records": len(successful_ids),
                "failed_records": len(failed_ids),
                "not_finished_count": not_finished_count,
                "validation_failed_count": len(failed_ids) - not_finished_count,
                "records_with_llm_generic_error": len(generic_error_ids),
                "llm_generic_error_record_ids": generic_error_ids,
                "failed_record_ids": sorted(failed_ids),
                "successful_record_ids": sorted(successful_ids),
                "time_limit_accepted_record_ids": [oid]
                if self.time_limit_accepted
                else [],
            },
            "rerun_history": {oid: self.rerun_history} if self.rerun_history else {},
            "final_failures": final_failures,
        }
        with open(self.run_dir / "evaluation_summary.json", "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    async def grade(self) -> dict[str, Any]:
        """The attempt that counts, through EVA's metrics (runner.py:331-335,
        522-541): the record's metrics, then the run's summary."""
        if self.final != "passed":
            detail = (
                "no attempt passed validation"
                if self.final == "failed"
                else "the record has not passed validation yet"
            )
            raise HTTPException(status_code=409, detail=f"nothing to grade: {detail}")
        if not self.config.metrics:
            raise HTTPException(
                status_code=409, detail="no metrics configured (EVA_METRICS is empty)"
            )
        if self.metrics_runner is None:
            self.metrics_runner = MetricsRunner(
                run_dir=self.run_dir,
                dataset=[self.record],
                metric_names=self.config.metrics,
                metric_configs=self.runner._metric_configs,
                num_draws=self.config.num_trials,
                force_rerun=self.config.force_rerun_metrics,
            )
        record_metrics = await self.metrics_runner.run_and_save_record(
            self.output_id, self.record_dir
        )
        if record_metrics is None:
            raise HTTPException(
                status_code=409, detail=f"{self.output_id} has no result.json"
            )
        self.metrics_runner.record_ids = {self.output_id}
        await self.metrics_runner.run()
        data = json.loads((self.record_dir / "metrics.json").read_text())
        return {
            "record_id": data["record_id"],
            "metrics": data["metrics"],
            "aggregates": data.get("aggregate_metrics", {}),
            "context": data.get("context"),
        }


_session: _Session | None = None
_lock = asyncio.Lock()


def _require() -> _Session:
    if _session is None:
        raise HTTPException(status_code=409, detail="no open session; POST /open first")
    current_record_id.set(_session.output_id)
    return _session


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "api": API_VERSION,
        "eva": eva.__version__,
        "commit": os.environ.get("GIT_COMMIT_SHA"),
    }


@app.post("/open")
async def open_session(request: OpenRequest) -> dict[str, Any]:
    """Open the trial, or its next attempt (the same request again)."""
    global _session
    async with _lock:
        session = _session
        if session is not None and session.request.key() == request.key():
            current_record_id.set(session.output_id)
            if session.final is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"the record is final ({session.final}); nothing to rerun",
                )
            if session.state == "open" and session.tool_calls:
                raise HTTPException(
                    status_code=409,
                    detail=f"attempt {session.attempt} ran tools but was never recorded; POST /record (with its error) first",
                )
            if session.state == "recorded":
                raise HTTPException(
                    status_code=409,
                    detail=f"attempt {session.attempt} is recorded; POST /validate first",
                )
            if session.state == "validated":
                await session.start_attempt()
            return session.describe()
        if session is not None:
            if not session.pristine:
                raise HTTPException(
                    status_code=409,
                    detail=f"a session for {session.output_id} is in use",
                )
            # Nothing happened in it: replace it.
            session.end_attempt_log()
            shutil.rmtree(session.run_dir, ignore_errors=True)
            _session = None
        try:
            session = _Session(request)
        except ValueError as e:  # EVA's RunConfig refused the trial's settings
            raise HTTPException(status_code=422, detail=str(e)) from e
        await session.start_attempt()
        _session = session
        return session.describe()


@app.post("/tool")
async def tool(request: ToolRequest) -> dict[str, Any]:
    async with _lock:
        return await _require().call_tool(request)


@app.post("/record")
async def record(request: Request) -> dict[str, Any]:
    """Body: a tar of the call loop's record files, with ``call.json``."""
    body = await request.body()
    async with _lock:
        session = _require()
        try:
            tar = tarfile.open(fileobj=io.BytesIO(body), mode="r:*")
        except tarfile.TarError as e:
            raise HTTPException(
                status_code=422, detail=f"body is not a tar: {e}"
            ) from e
        with tar:
            call_member = next(
                (
                    m
                    for m in tar.getmembers()
                    if m.isfile() and _member_name(m) == CALL_FILE
                ),
                None,
            )
            handle = tar.extractfile(call_member) if call_member is not None else None
            if handle is None:
                raise HTTPException(
                    status_code=422, detail=f"the tar has no {CALL_FILE}"
                )
            call = CallInfo.model_validate_json(handle.read())
            return session.store(tar, call)


@app.post("/validate")
async def validate(request: ValidateRequest) -> dict[str, Any]:
    async with _lock:
        return await _require().validate(request.attempt)


@app.post("/grade")
async def grade() -> dict[str, Any]:
    async with _lock:
        return await _require().grade()


@app.get("/files")
async def files() -> FileResponse:
    """The run directory as a tar: ``config.json``, the summaries,
    ``records/<id>/trial_<k>/`` and its ``_failed_attempt_N`` archives."""
    async with _lock:
        session = _require()
        handle, name = tempfile.mkstemp(suffix=".tar")
        os.close(handle)
        with tarfile.open(name, "w") as tar:
            for child in sorted(session.run_dir.iterdir()):
                tar.add(child, arcname=child.name)
    return FileResponse(
        name,
        media_type="application/x-tar",
        filename=f"{session.config.run_id}.tar",
        background=BackgroundTask(os.unlink, name),
    )


def main() -> None:
    setup_logging(level=os.environ.get("EVA_LOG_LEVEL", "INFO"))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
