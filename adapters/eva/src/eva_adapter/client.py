"""Client for the EVA sidecar (``adapters/eva/sidecar``).

The sidecar is EVA's own code in EVA's image; this is the harbor-voice side
of its HTTP interface. Wire types are plain JSON, so nothing
here imports EVA.

One sidecar session is one trial of one record. Its attempts follow EVA's
rerun loop: ``open`` → ``tool``\\* (during the call) → ``record`` →
``validate``; while ``validate`` says ``rerun``, ``open`` again for the next
attempt. Then ``grade`` and ``fetch``.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal

import httpx

CALL_FILE = "call.json"
"""What the call loop knows of a call, beside its record files."""

JUDGE_TIMEOUT_S = 1800.0
"""``/validate`` and ``/grade`` wait on EVA's judges: several models, one call
at a time per model by EVA's default, each retried with backoff."""


SIDECAR_API = 1
"""The sidecar API this driver speaks (``API_VERSION`` in ``sidecar/server.py``).

The driver is installed from the running checkout at every trial; the
sidecar is whatever image the task names, built when the task was converted.
:meth:`Sidecar.open` checks the two agree before anything is spent."""


class SidecarError(RuntimeError):
    """The sidecar refused a request or is unreachable."""


class SidecarMismatch(SidecarError):
    """The task's sidecar serves another API than this driver speaks."""


def mismatch(served: object, expected: int, adapter: str) -> str:
    """Why a trial stops: the two versions, and what to do about it."""
    have = (
        "no API version (it predates the check)" if served is None else f"API {served}"
    )
    return (
        f"the task's {adapter} sidecar serves {have}; this harbor-voice's driver "
        f"speaks API {expected}. Run the task with the harbor-voice it was "
        "converted with (for a hub dataset: the checkout whose registry.json lists "
        "it), or reconvert it from this checkout."
    )


@dataclass(frozen=True, slots=True)
class AgentSetup:
    """The agent under test as EVA sets it up (its OpenAI Realtime server)."""

    system_prompt: str | None
    tools: list[dict[str, Any]] | None
    """Realtime ``session.tools``: ``{type, name, description, parameters}``."""
    greeting: str
    """EVA's opening line for the language (``initial_messages.yaml``)."""
    opening_text: str | None
    """The user item EVA sends to start the call: ``Say: '<greeting>'``."""
    session: dict[str, Any] | None
    """EVA's whole ``session.update`` payload (``_build_session_config``)."""


@dataclass(frozen=True, slots=True)
class CallerSetup:
    """The caller as EVA's ElevenLabs simulator sets it up."""

    prompt: str
    """The ``prompt`` dynamic variable (``_build_prompt``)."""
    gender: Literal["F", "M"]
    """Picks the ElevenLabs agent (EVA's ``EVA_<LANG>_USER_<F|M>``)."""
    language: str


@dataclass(frozen=True, slots=True)
class Opened:
    """One attempt, opened."""

    conversation_id: str
    record_id: str
    trial: int
    attempt: int
    max_attempts: int
    time_limit_s: float
    """EVA's ``conversation_time_limit_seconds``."""
    caller_timeout_s: float
    """The timeout EVA's worker gives its simulator (limit + 20 s grace)."""
    agent: AgentSetup
    caller: CallerSetup
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolReply:
    result: dict[str, Any]
    """EVA's ``execute_tool`` result (it always carries ``status``)."""
    output: str
    """The ``function_call_output`` EVA sends the model: the result as JSON."""


@dataclass(frozen=True, slots=True)
class CallInfo:
    """``call.json``: what EVA's worker would know of the call."""

    attempt: int
    conversation_ended_reason: str | None
    started_at: datetime
    ended_at: datetime
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "conversation_ended_reason": self.conversation_ended_reason,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class Validation:
    """EVA's verdict on one attempt, and what its runner does next."""

    attempt: int
    passed: bool
    """The attempt counts: validated, or accepted at the time limit."""
    gate: bool
    """``conversation_valid_end`` passed (False is EVA's ``not_finished``)."""
    reason: str
    """``passed``, ``time_limit_accepted``, ``not_finished``,
    ``time_limit_exceeded`` or ``validation_failed``."""
    scores: dict[str, float]
    failed_metrics: list[str]
    rerun: bool
    """Run the record again (``open`` → a new attempt)."""
    restored_attempt: int | None
    """An earlier attempt EVA's time-limit rule restored as the record."""
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Grade:
    record_id: str
    metrics: dict[str, Any]
    """EVA's ``MetricScore`` per metric, as in ``metrics.json``."""
    aggregates: dict[str, float | None]
    """EVA's composites: ``EVA-A_pass``, ``EVA-X_pass``, ``EVA-overall_mean``, ..."""
    context: dict[str, Any] | None
    """The metric context EVA's processor built (``metrics.json``)."""


class Sidecar:
    """One sidecar session: one trial of one record, all its attempts."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 60.0,
        judge_timeout_s: float = JUDGE_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None
        self._judge_timeout_s = judge_timeout_s
        self._api_checked = False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(
                method, f"{self._base}{path}", **kwargs
            )
        except httpx.HTTPError as exc:
            raise SidecarError(
                f"sidecar unreachable at {self._base}{path}: {exc}"
            ) from exc
        if response.status_code != 200:
            raise SidecarError(
                f"sidecar {path} returned {response.status_code}: {response.text[:500]}"
            )
        return response

    async def _post(
        self, path: str, body: Mapping[str, Any], *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"json": dict(body)}
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s
        return (await self._send("POST", path, **kwargs)).json()

    async def health(self) -> bool:
        try:
            response = await self._client.get(f"{self._base}/health")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def open(
        self,
        *,
        domain: str,
        record_id: str,
        language: str = "en",
        trial: int = 0,
        num_trials: int | None = None,
        framework: str | None = None,
        model: Mapping[str, Any] | None = None,
    ) -> Opened:
        """Open the trial, or its next attempt (the same arguments again).

        ``framework`` and ``model`` are EVA's run config for the agent under
        test; the sidecar defaults to EVA's reference system
        (``openai_realtime``, ``gpt-realtime-1.5``).
        """
        body: dict[str, Any] = {
            "domain": domain,
            "record_id": record_id,
            "language": language,
            "trial": trial,
        }
        if num_trials is not None:
            body["num_trials"] = num_trials
        if framework is not None:
            body["framework"] = framework
        if model is not None:
            body["model"] = dict(model)
        await self.check_api()
        return _opened(await self._post("/open", body))

    async def check_api(self) -> None:
        """Refuse a sidecar whose API isn't :data:`SIDECAR_API` (once per session)."""
        if self._api_checked:
            return
        served = (await self._send("GET", "/health")).json().get("api")
        if served != SIDECAR_API:
            raise SidecarMismatch(mismatch(served, SIDECAR_API, "EVA"))
        self._api_checked = True

    async def tool(self, name: str, arguments: Mapping[str, Any]) -> ToolReply:
        """Run one tool call on the scenario DB, as EVA's server runs it."""
        reply = await self._post("/tool", {"name": name, "arguments": dict(arguments)})
        return ToolReply(result=reply["result"], output=reply["output"])

    async def record(self, record_dir: Path, call: CallInfo) -> dict[str, Any]:
        """Hand over the call loop's record files; the sidecar completes the
        record with EVA's code (scenario DBs, hashes, ``result.json``)."""
        response = await self._send(
            "POST",
            "/record",
            content=pack_record(record_dir, call),
            headers={"content-type": "application/x-tar"},
        )
        return response.json()

    async def validate(self, attempt: int) -> Validation:
        reply = await self._post(
            "/validate", {"attempt": attempt}, timeout_s=self._judge_timeout_s
        )
        return Validation(
            attempt=int(reply["attempt"]),
            passed=bool(reply["passed"]),
            gate=bool(reply["gate"]),
            reason=str(reply["reason"]),
            scores=dict(reply.get("scores") or {}),
            failed_metrics=list(reply.get("failed_metrics") or []),
            rerun=bool(reply["rerun"]),
            restored_attempt=reply.get("restored_attempt"),
            raw=reply,
        )

    async def grade(self) -> Grade:
        """EVA's metrics on the attempt that counts."""
        reply = await self._post("/grade", {}, timeout_s=self._judge_timeout_s)
        return Grade(
            record_id=str(reply["record_id"]),
            metrics=dict(reply["metrics"]),
            aggregates=dict(reply["aggregates"]),
            context=reply.get("context"),
        )

    async def files(self) -> bytes:
        """The run directory as a tar, archived attempts included."""
        return (
            await self._send("GET", "/files", timeout=self._judge_timeout_s)
        ).content

    async def fetch(self, dest: Path) -> Path:
        """Extract the run directory into ``dest``."""
        unpack(await self.files(), dest)
        return dest


def _opened(reply: dict[str, Any]) -> Opened:
    agent = reply["agent"]
    caller = reply["caller"]
    gender = caller["gender"]
    if gender not in ("F", "M"):
        raise SidecarError(f"sidecar returned caller gender {gender!r}")
    return Opened(
        conversation_id=str(reply["conversation_id"]),
        record_id=str(reply["record_id"]),
        trial=int(reply["trial"]),
        attempt=int(reply["attempt"]),
        max_attempts=int(reply["max_attempts"]),
        time_limit_s=float(reply["time_limit_s"]),
        caller_timeout_s=float(reply["caller_timeout_s"]),
        agent=AgentSetup(
            system_prompt=agent.get("system_prompt"),
            tools=agent.get("tools"),
            greeting=str(agent["greeting"]),
            opening_text=agent.get("opening_text"),
            session=agent.get("session"),
        ),
        caller=CallerSetup(
            prompt=str(caller["prompt"]),
            gender=gender,
            language=str(caller["language"]),
        ),
        raw=reply,
    )


def pack_record(record_dir: Path, call: CallInfo) -> bytes:
    """The ``/record`` body: every file under ``record_dir`` (paths relative
    to it) and ``call.json``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted(p for p in record_dir.rglob("*") if p.is_file()):
            name = path.relative_to(record_dir).as_posix()
            if name == CALL_FILE:
                raise ValueError(
                    f"{record_dir} has its own {CALL_FILE}; it is written from CallInfo"
                )
            tar.add(path, arcname=name, recursive=False)
        data = json.dumps(call.to_json(), indent=2).encode()
        info = tarfile.TarInfo(CALL_FILE)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def unpack(data: bytes, dest: Path) -> None:
    """Extract a sidecar tar into ``dest``, refusing paths that leave it."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        tar.extractall(dest, filter="data")
