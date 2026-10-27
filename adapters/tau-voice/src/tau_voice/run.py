"""Run one tau-voice task: sidecar + our caller + an agent transport.

The sidecar (tau2 in its container) supplies the task — prompts, tools, the
caller's persona and voice — and executes every tool call. harbor-voice runs
the call: :class:`~tau_voice.stepped_call.SteppedCall` steps a
:class:`~tau_voice.caller.ReactiveCaller` (``Tau2Content`` for
the words, ElevenLabs + tau2's phone line for the voice) against the agent.
The step log goes back to the sidecar's ``/close``; grading is the verifier's.
"""

from __future__ import annotations

import json
import wave
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor_voice.audio.format import AudioFormat
from tau_voice.content import Tau2Content
from tau_voice.client import Sidecar
from tau_voice.telephony import TelephonyConverter
from tau_voice.telephony import ulaw_to_pcm16
from harbor_voice.ids import PartyId
from harbor_voice.providers.cache import BlobCache
from harbor_voice.providers.elevenlabs import ElevenLabsTTS
from harbor_voice.providers.openai.llm import OpenAILLM
from harbor_voice.providers.openai.realtime import TELEPHONY
from harbor_voice.providers.openai.realtime import OpenAIRealtimeTransport
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.providers.tools import ToolSpec
from tau_voice.caller import PatientPolicy
from tau_voice.caller import ReactiveCaller
from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.stepped_call import CallResult
from tau_voice.stepped_call import SteppedCall
from tau_voice.stepped_call import Termination
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport

AgentFactory = Callable[[ToolRouter, str], VoiceTransport]
"""``(tools, system_prompt) -> agent transport``."""

SYNTHESIS = AudioFormat(sample_rate_hz=16_000)
"""tau2's ``DEFAULT_PCM_SAMPLE_RATE``: the caller is synthesised at 16 kHz."""


class Tau2Voice:
    """ElevenLabs at 16 kHz, then tau2's phone line. A ``CallerVoice``."""

    def __init__(self, tts: ElevenLabsTTS, *, step_s: float = 0.2) -> None:
        self.tts = tts
        self.line = TelephonyConverter(input_rate_hz=SYNTHESIS.sample_rate_hz)
        self.step_bytes = int(SYNTHESIS.sample_rate_hz * step_s) * 2

    async def synthesize(self, text: str) -> bytes:
        return await self.tts.synthesize_pcm(text)

    def channel(self, pcm: bytes) -> bytes:
        return self.line.convert(pcm)


@dataclass(slots=True)
class SidecarToolExecutor:
    """The agent's tools: executed by tau2, answered at the next step.

    Executed by tau2 after the step's receive window closes, answered at the
    start of the next step — see ``SteppedCall.tool_window``.
    """

    sidecar: Sidecar
    specs: Sequence[ToolSpec]
    call: SteppedCall | None = None

    async def execute(self, invocation: ToolInvocation) -> Mapping[str, Any]:
        if self.call is None:
            raise RuntimeError("executor used before the call was attached")
        step = await self.call.tool_window()
        try:
            outcome = await self.sidecar.tool(
                side="agent",
                step=step,
                call_id=invocation.tool_use_id,
                name=invocation.name,
                arguments=invocation.arguments,
            )
        finally:
            await self.call.tool_done(step)
        if outcome.stop:
            self.call.request_stop(Termination.AGENT_STOP)
        if outcome.too_many_errors:
            self.call.request_stop(Termination.TOO_MANY_ERRORS)
        await self.call.gate.after(step)
        # tau2 sends the tool's string verbatim; ``{"text": ...}`` is how a
        # ToolResult carries that (see ToolResult.as_text).
        return {"text": outcome.content}


def _specs(tools: Sequence[Mapping[str, Any]]) -> list[ToolSpec]:
    specs = []
    for tool in tools:
        function = tool["function"]
        specs.append(
            ToolSpec(
                name=function["name"],
                description=function.get("description") or function["name"],
                input_schema=function.get("parameters") or {"type": "object"},
            )
        )
    return specs


async def run_task(
    *,
    sidecar_url: str,
    domain: str,
    task_id: str,
    out_dir: Path,
    speech_complexity: str = "control",
    seed: int = 42,
    agent_model: str | None = None,
    agent_factory: AgentFactory | None = None,
    synthesis_cache: Path | None = None,
    hallucination_feedback: str | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run one task end to end; returns the run summary written to ``out_dir``.

    ``agent_factory`` builds the agent under test from the tools and the
    benchmark's system prompt; harbor-voice's driver passes one built from the
    task's ``[agent]``. Without one, it is OpenAI Realtime — tau2's default.
    """
    if speech_complexity != "control":
        raise NotImplementedError(
            "only the control preset is wired so far; regular needs the reactive "
            "policy and tau2's effects"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = Sidecar(sidecar_url)
    try:
        task = await sidecar.open(
            domain=domain,
            task_id=task_id,
            speech_complexity=speech_complexity,
            seed=seed,
            hallucination_feedback=hallucination_feedback,
        )
        agent_info, user_info = task["agent"], task["user"]
        turns = user_info["turn_taking"]

        executor = SidecarToolExecutor(sidecar, _specs(agent_info["tools"]))
        router = ToolRouter(executor)
        if agent_factory is not None:
            transport = agent_factory(router, agent_info["system_prompt"])
        else:
            transport = OpenAIRealtimeTransport(
                party=PartyId("agent"),
                model=agent_model or agent_info["default_model"],
                system_prompt=agent_info["system_prompt"],
                tools=router,
            )
        content = Tau2Content(sidecar, OpenAILLM(model=user_info["llm"]))
        voice = Tau2Voice(
            ElevenLabsTTS(
                voice_id=user_info["voice_id"],
                audio_format=SYNTHESIS,
                cache=BlobCache(synthesis_cache) if synthesis_cache else None,
            ),
            step_s=turns["tick_duration_s"],
        )
        caller = ReactiveCaller(
            content=content,
            voice=voice,
            policy=PatientPolicy(
                wait_other=turns["wait_to_respond_other_ticks"],
                wait_self=turns["wait_to_respond_self_ticks"],
                yield_when_interrupted=turns["yield_when_interrupted_ticks"],
                yield_when_interrupting=turns["yield_when_interrupting_ticks"],
            ),
        )

        async def user_tools(
            calls: Sequence[CallerToolCall], step: int
        ) -> tuple[CallerToolResult, ...]:
            return await content.run_tools(calls, step=step)

        await transport.open(
            SessionConfig(
                party=PartyId("agent"), audio_in=TELEPHONY, audio_out=TELEPHONY
            )
        )
        try:
            call = SteppedCall(
                caller=caller,
                transport=transport,
                run_user_tools=user_tools,
                audio_format=TELEPHONY,
                step_s=turns["tick_duration_s"],
                max_steps=max_steps
                or int(turns["max_steps_seconds"] / turns["tick_duration_s"]),
            )
            executor.call = call
            result = await call.run()
        finally:
            await transport.close()

        closed = await sidecar.close(_close_body(result, turns["tick_duration_s"]))
        summary = {
            "domain": domain,
            "task_id": task_id,
            "speech_complexity": speech_complexity,
            "seed": seed,
            "persona": user_info["persona_name"],
            "agent": {
                "provider": getattr(transport.ref, "provider", None),
                "model": getattr(transport.ref, "model", None),
                "session_id": getattr(transport, "session_id", None),
                "errors": list(getattr(transport, "errors", [])),
            },
            "termination": result.termination.value,
            "steps": len(result.steps),
            "call_s": round(len(result.steps) * turns["tick_duration_s"], 1),
            "wall_s": round(result.wall_s, 1),
            "truncations": len(result.truncations),
            "caller_llm_calls": len(content.exchanges),
            "tts_characters_billed": voice.tts.characters_billed,
            "sidecar_state": closed,
        }
        _write(out_dir, result, summary)
        return summary
    finally:
        await sidecar.aclose()


Grader = Callable[[Path], Awaitable[Mapping[str, Any]]]
"""Grades the attempt just closed; returns the verifier's ``result.json``.

It runs the verifier where the sidecar's state lives (``tests/evaluate.py``
in the sidecar image) and copies what it needs into the attempt directory.
"""

HALLUCINATION_RETRIES = 3
"""tau2's ``RunConfig.hallucination_retries`` default."""


async def run_trial(
    *,
    grade: Grader,
    out_dir: Path,
    seed: int = 42,
    hallucination_retries: int = HALLUCINATION_RETRIES,
    attempt: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    **task: Any,
) -> dict[str, Any]:
    """One trial with tau2's hallucination retry (``runner/batch.py::run_unit``).

    After each attempt the verifier's hallucination check runs. If the caller
    invented facts, the attempt is set aside and the task is rerun with tau2's
    feedback appended to the caller's instructions and seed ``seed + n·1000``,
    up to ``hallucination_retries`` times. The last attempt is the result;
    every attempt stays on disk under ``attempt-<n>/``.
    """
    run = attempt or run_task
    feedback: str | None = None
    attempts: list[dict[str, Any]] = []
    graded: Mapping[str, Any] = {}
    for n in range(hallucination_retries + 1):
        attempt_dir = out_dir / f"attempt-{n}"
        summary = await run(
            out_dir=attempt_dir,
            seed=seed + n * 1000,
            hallucination_feedback=feedback,
            **task,
        )
        graded = await grade(attempt_dir)
        found = bool((graded.get("hallucination") or {}).get("found"))
        attempts.append(
            {
                "attempt": n,
                "seed": seed + n * 1000,
                "reward": graded.get("reward"),
                "hallucination": found,
                "termination": summary.get("termination"),
                "call_s": summary.get("call_s"),
            }
        )
        if not found or n == hallucination_retries:
            break
        feedback = (graded.get("hallucination") or {}).get("feedback")
    trial = {
        **{
            k: v
            for k, v in task.items()
            if k in ("domain", "task_id", "speech_complexity")
        },
        "seed": seed,
        "reward": graded.get("reward"),
        "reward_info": graded.get("reward_info"),
        "final_attempt": attempts[-1]["attempt"],
        "hallucination_retries_used": len(attempts) - 1,
        "attempts": attempts,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trial.json").write_text(json.dumps(trial, indent=2), encoding="utf-8")
    return trial


def _close_body(result: CallResult, step_s: float) -> dict[str, Any]:
    steps = []
    for log in result.steps:
        user = log.user
        steps.append(
            {
                "step": log.step,
                "t_s": log.t_s,
                "user": {"text": user.text, "speech": user.speech}
                if user.is_meaningful
                else None,
                "agent": {"text": log.agent_text, "speech": log.agent_speech}
                if (log.agent_text or log.agent_speech)
                else None,
            }
        )
    return {
        "steps": steps,
        "termination": result.termination.value,
        "duration_s": len(result.steps) * step_s,
        "tick_duration_s": step_s,
    }


def _write(out_dir: Path, result: CallResult, summary: dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log = [
        {
            "step": s.step,
            "user": s.user.text if s.user.is_meaningful else None,
            "user_action": s.user.action.value if s.user.action else None,
            "user_tools": [c.name for c in s.user.tool_calls],
            "agent": s.agent_text,
            "agent_speech": s.agent_speech,
            "truncated": bool(s.agent and s.agent.was_truncated),
        }
        for s in result.steps
        if s.user.is_meaningful or s.agent_text or s.agent_speech or s.user.tool_calls
    ]
    (out_dir / "steps.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    # Stereo, 8 kHz: left the caller as the agent received it, right the agent
    # as the caller heard it — tau2's both.wav layout.
    import numpy as np

    left = b"".join(s.user.audio or b"\xff" * 1600 for s in result.steps)
    right = b"".join(
        s.agent.played if s.agent else b"\x7f" * 1600 for s in result.steps
    )
    a = np.frombuffer(ulaw_to_pcm16(left), "<i2")
    b = np.frombuffer(ulaw_to_pcm16(right), "<i2")
    n = max(len(a), len(b))
    stereo = np.zeros((n, 2), dtype="<i2")
    stereo[: len(a), 0] = a
    stereo[: len(b), 1] = b
    with wave.open(str(out_dir / "both.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(stereo.tobytes())
