"""Client for the tau-voice runtime sidecar (``adapters/tau-voice/sidecar``).

The sidecar is tau2 in its own container; this is the harbor-voice side of
its HTTP interface. Wire types are plain JSON so nothing here imports tau2.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.reactive import Heard
from tau_voice.reactive import Pending
from tau_voice.reactive import StepRecord


class SidecarError(RuntimeError):
    """The sidecar refused a request or is unreachable."""


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    content: str
    error: bool
    stop: bool
    """The agent called tau2's stop tool (``transfer_to_human_agents``)."""
    too_many_errors: bool


class Sidecar:
    """One sidecar session — one task, one trial."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(f"{self._base}{path}", json=body)
        except httpx.HTTPError as exc:
            raise SidecarError(
                f"sidecar unreachable at {self._base}{path}: {exc}"
            ) from exc
        if response.status_code != 200:
            raise SidecarError(
                f"sidecar {path} returned {response.status_code}: {response.text[:300]}"
            )
        return response.json()

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
        task_id: str,
        speech_complexity: str,
        seed: int,
        user_llm: str | None = None,
        hallucination_feedback: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "domain": domain,
            "task_id": task_id,
            "speech_complexity": speech_complexity,
            "seed": seed,
        }
        if user_llm is not None:
            body["user_llm"] = user_llm
        if hallucination_feedback is not None:
            body["hallucination_feedback"] = hallucination_feedback
        return await self._post("/open", body)

    async def tool(
        self,
        *,
        side: str,
        step: int,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
    ) -> ToolOutcome:
        reply = await self._post(
            "/tool",
            {
                "side": side,
                "step": step,
                "id": call_id,
                "name": name,
                "arguments": dict(arguments),
            },
        )
        return ToolOutcome(
            content=reply["content"],
            error=bool(reply["error"]),
            stop=bool(reply["stop"]),
            too_many_errors=bool(reply["too_many_errors"]),
        )

    async def user_context(
        self, steps: Sequence[StepRecord], pending: Pending
    ) -> dict[str, Any]:
        """The caller-LLM request body tau2 would send now."""
        body: dict[str, Any] = {"steps": [_caller_step(s) for s in steps]}
        if isinstance(pending, tuple):
            body["pending"] = None
            body["pending_tool_results"] = [_result(r) for r in pending]
        else:
            body["pending"] = _heard(pending)
        return await self._post("/user_context", body)

    async def close(self, body: Mapping[str, Any]) -> dict[str, Any]:
        return await self._post("/close", body)

    async def grade(self, *, hallucination_check: bool = True) -> dict[str, Any]:
        """Grade the call last closed — the verifier's code, run in the sidecar."""
        return await self._post("/grade", {"hallucination_check": hallucination_check})


def _heard(heard: Heard | None) -> dict[str, Any] | None:
    if heard is None:
        return None
    return {"text": heard.text, "speech": heard.speech}


def _call(call: CallerToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}


def _result(result: CallerToolResult) -> dict[str, Any]:
    return {"id": result.id, "content": result.content, "error": result.error}


def _caller_step(step: StepRecord) -> dict[str, Any]:
    said = None
    if step.said_text is not None or step.said_speech or step.said_tools:
        said = {
            "text": step.said_text,
            "speech": step.said_speech,
            "tool_calls": [_call(c) for c in step.said_tools],
        }
    return {
        "step": step.step,
        "said": said,
        "heard": _heard(step.heard),
        "tool_results": [_result(r) for r in step.tool_results],
    }
