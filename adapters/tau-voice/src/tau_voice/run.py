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
import tempfile
import wave
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from harbor_voice.audio.format import AudioFormat
from tau_voice.content import Tau2Content
from tau_voice.content import Tau2Listener
from harbor_voice.providers.openai.llm import usage_record as chat_usage
from harbor.models.trajectories.trajectory import Trajectory
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.model import merge_usage
from tau_voice.client import Sidecar
from tau_voice.client import SidecarError
from tau_voice.effects import Tau2CallerEffects
from tau_voice.main import HALLUCINATION_RETRIES
from tau_voice.main import PRESETS
from tau_voice.telephony import Tau2Telephony
from tau_voice.telephony import ulaw_to_pcm16
from harbor_voice.ids import PartyId
from harbor_voice.ids import PlanId
from harbor_voice.perturbation import PerturbationPipeline
from harbor_voice.perturbation import SeedSet
from harbor_voice.perturbation.persona import PersonaPlan
from harbor_voice.perturbation.persona import VoiceSelection
from harbor_voice.simulator import SourceKind
from harbor_voice.simulator import UniversalSimulator
from tau_voice.voice import SimulatorVoice
from harbor_voice.providers.cache import BlobCache
from harbor_voice.providers.elevenlabs import ElevenLabsTTS
from harbor_voice.providers.google.live import GeminiLiveTransport
from harbor_voice.providers.openai.live import LIVE_AUDIO
from harbor_voice.providers.openai.live import LiveSettings
from harbor_voice.providers.openai.live import OpenAILiveTransport
from harbor_voice.providers.openai.llm import OpenAILLM
from harbor_voice.providers.openai.realtime import TELEPHONY
from harbor_voice.providers.openai.realtime import OpenAIRealtimeTransport
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.providers.tools import ToolSpec
from tau_voice.caller import ReactiveCaller
from tau_voice.caller import ReactivePolicy
from tau_voice.caller import tau2_backchannels
from tau_voice.agent_steps import AgentStepper
from tau_voice.agent_steps import CascadeAgentStepper
from tau_voice.agent_steps import GeminiAgentStepper
from tau_voice.agent_steps import LiveAgentStepper
from tau_voice.cascade import cascade_prompt
from harbor_voice.capabilities import Pipeline
from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.stepped_call import CallResult
from tau_voice.trajectory import build as build_trajectory
from tau_voice.stepped_call import SteppedCall
from tau_voice.stepped_call import Termination
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport

AgentFactory = Callable[..., VoiceTransport]
"""``(tools, system_prompt, **overrides) -> agent transport``; for OpenAI Live
the overrides are tau2's backend prompt and voice."""

SYNTHESIS = AudioFormat(sample_rate_hz=16_000)
"""tau2's ``DEFAULT_PCM_SAMPLE_RATE``: the caller is synthesised at 16 kHz."""


def caller_simulator(
    tts: ElevenLabsTTS, *, voice_id: str, seed: int, complexity: str
) -> UniversalSimulator:
    """The caller's voice: a ``UniversalSimulator`` with tau2's phone line.

    16 kHz (tau2's synthesis rate); the persona's ElevenLabs voice through our
    L1 layer; no signal layers; tau2's telephony as the wire encoder. tau2's
    effects (every preset but ``control``) run per step around this chain,
    in :class:`~tau_voice.voice.SimulatorVoice` — see :mod:`tau_voice.effects`.
    """
    pipeline = PerturbationPipeline(
        (),
        seeds=SeedSet.derive(seed),
        plan_id=PlanId(f"tau-voice-{complexity}"),
        fmt=SYNTHESIS,
        persona=PersonaPlan(voice=VoiceSelection(voice_id=voice_id)),
        encoder=Tau2Telephony(),
    )
    return UniversalSimulator(
        party=PartyId("caller"),
        tts=tts,
        pipeline=pipeline,
        seeds=SeedSet.derive(seed),
        fmt=SYNTHESIS,
        source_kind=SourceKind.LLM,
    )


async def tau2_effects(
    sidecar: Sidecar, user_info: Mapping[str, Any], *, step_s: float, folder: Path
) -> Tau2CallerEffects:
    """tau2's effects for this call, its noise files fetched into ``folder``."""
    effects = Tau2CallerEffects(
        user_info["voice_settings"],
        turn_burst_files=user_info["tau2_burst_files"],
        step_s=step_s,
    )
    for kind, name in effects.needed_assets:
        if Path(name).name != name:
            raise SidecarError(f"unexpected tau2 noise file name {name!r}")
        path = folder / kind / name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(await sidecar.asset(kind, name))
        effects.assets[name] = path
    return effects


@dataclass(slots=True)
class SidecarToolExecutor:
    """The agent's tools: executed by tau2, answered at the next step.

    Executed by tau2 after the step's receive window closes, answered at the
    start of the next step — see ``SteppedCall.tool_window``.
    """

    sidecar: Sidecar
    specs: Sequence[ToolSpec]
    call: SteppedCall | None = None
    log: list[dict[str, Any]] = field(default_factory=list)
    """Every agent tool call, in order: step, name, arguments, result or error."""

    async def execute(self, invocation: ToolInvocation) -> Mapping[str, Any]:
        if self.call is None:
            raise RuntimeError("executor used before the call was attached")
        step = await self.call.tool_window()
        entry: dict[str, Any] = {
            "step": step,
            "party": "agent",
            "call_id": invocation.tool_use_id,
            "name": invocation.name,
            "arguments": dict(invocation.arguments),
        }
        self.log.append(entry)
        try:
            outcome = await self.sidecar.tool(
                side="agent",
                step=step,
                call_id=invocation.tool_use_id,
                name=invocation.name,
                arguments=invocation.arguments,
            )
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            await self.call.tool_done(step)
        entry["result"] = outcome.content
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
    user_llm: str | None = None,
    max_steps: int | None = None,
    attempt_index: int = 0,
    live_config: Mapping[str, Any] | None = None,
    cascade: bool = False,
) -> dict[str, Any]:
    """Run one task end to end; returns the run summary written to ``out_dir``.

    ``agent_factory`` builds the agent under test from the tools and the
    benchmark's system prompt; harbor-voice's driver passes one built from the
    task's ``[agent]``. Without one, it is OpenAI Realtime — tau2's default.

    ``live_config`` (tau2's ``--live-config``) says the agent is OpenAI Live:
    tau2 renders its two prompts, and the call steps as tau2's Live adapter
    does (:class:`~tau_voice.agent_steps.LiveAgentStepper`, real-time pacing).
    A Gemini Live agent steps as tau2's Gemini adapter does
    (:class:`~tau_voice.agent_steps.GeminiAgentStepper`).

    ``cascade`` says ``agent_factory`` builds a duplex cascade
    (:class:`~harbor_voice.cascade.DuplexCascadeTransport`): it gets tau2's
    cascade prompt, and the call steps as tau2's cascaded adapter does
    (:class:`~tau_voice.agent_steps.CascadeAgentStepper`).
    """
    if cascade and (agent_factory is None or live_config is not None):
        raise ValueError("a cascade is built by agent_factory, and is not OpenAI Live")
    if speech_complexity not in PRESETS:
        raise ValueError(
            f"unknown speech complexity {speech_complexity!r}; tau2's: {', '.join(PRESETS)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = Sidecar(sidecar_url)
    scratch = tempfile.TemporaryDirectory(prefix="tau-voice-")
    try:
        task = await sidecar.open(
            domain=domain,
            task_id=task_id,
            speech_complexity=speech_complexity,
            seed=seed,
            user_llm=user_llm,
            hallucination_feedback=hallucination_feedback,
            live_config=live_config,
            agent_model=agent_model,
        )
        agent_info, user_info = task["agent"], task["user"]
        live = agent_info.get("live")
        turns = user_info["turn_taking"]

        executor = SidecarToolExecutor(sidecar, _specs(agent_info["tools"]))
        router = ToolRouter(executor)
        if live is not None and agent_factory is not None:
            transport = agent_factory(
                router,
                live["frontend_prompt"],
                backend_instructions=live["backend_prompt"],
                voice=live["voice"],
            )
        elif live is not None:
            transport = OpenAILiveTransport(
                party=PartyId("agent"),
                model=agent_model or agent_info["default_model"],
                system_prompt=live["frontend_prompt"],
                tools=router,
                settings=LiveSettings(
                    backend_model=live["backend_model"],
                    voice=live["voice"],
                    backend_instructions=live["backend_prompt"],
                ),
            )
        elif cascade and agent_factory is not None:
            transport = agent_factory(
                router, cascade_prompt(agent_info["system_prompt"])
            )
        elif agent_factory is not None:
            transport = agent_factory(router, agent_info["system_prompt"])
        else:
            transport = OpenAIRealtimeTransport(
                party=PartyId("agent"),
                model=agent_model or agent_info["default_model"],
                system_prompt=agent_info["system_prompt"],
                tools=router,
            )
        content = Tau2Content(sidecar, OpenAILLM(model=user_info["llm"]))
        tts = ElevenLabsTTS(
            audio_format=SYNTHESIS,
            cache=BlobCache(synthesis_cache) if synthesis_cache else None,
        )
        simulator = caller_simulator(
            tts,
            voice_id=user_info["voice_id"],
            seed=seed,
            complexity=speech_complexity,
        )
        effects = None
        if speech_complexity != "control":
            effects = await tau2_effects(
                sidecar,
                user_info,
                step_s=turns["tick_duration_s"],
                folder=(synthesis_cache or Path(scratch.name)) / "tau2-noise",
            )

            async def synthesize(text: str) -> bytes:
                return (await simulator.prepare(text)).pcm

            await effects.open(synthesize, voice_id=user_info["voice_id"])
        voice = SimulatorVoice(
            simulator, step_s=turns["tick_duration_s"], effects=effects
        )
        if not turns["use_llm_backchannel"] and turns["backchannel_min_ticks"]:
            raise NotImplementedError(
                "tau2's Poisson backchannels (no preset uses them)"
            )
        listener = None
        if turns["enable_user_initiated_interruption"] or turns["use_llm_backchannel"]:
            listener = Tau2Listener(sidecar)
        voice_seed = user_info["voice_settings"]["speech_environment"]["voice_seed"]
        caller = ReactiveCaller(
            content=content,
            voice=voice,
            policy=ReactivePolicy(
                wait_other=turns["wait_to_respond_other_ticks"],
                wait_self=turns["wait_to_respond_self_ticks"],
                yield_when_interrupted=turns["yield_when_interrupted_ticks"],
                yield_when_interrupting=turns["yield_when_interrupting_ticks"],
                listener=listener,
                check_interval=turns["interruption_check_interval_ticks"],
            ),
            backchannels=tau2_backchannels(voice_seed),
        )

        async def user_tools(
            calls: Sequence[CallerToolCall], step: int
        ) -> tuple[CallerToolResult, ...]:
            return await content.run_tools(calls, step=step)

        step_s = turns["tick_duration_s"]
        session, stepping = _stepping(transport, live=live is not None, step_s=step_s)
        await transport.open(session)
        try:
            call = SteppedCall(
                caller=caller,
                transport=transport,
                run_user_tools=user_tools,
                audio_format=TELEPHONY,
                step_s=step_s,
                max_steps=max_steps or int(turns["max_steps_seconds"] / step_s),
                **stepping,
            )
            executor.call = call
            result = await call.run()
        finally:
            await transport.close()

        close = _close_body(result, turns["tick_duration_s"])
        if effects is not None:
            close["effect_timeline"] = effects.close()
        closed = await sidecar.close(close)
        usage = call_usage(
            transport=transport,
            content=content,
            listener=listener,
            tts=tts,
            caller_llm=user_info["llm"],
        )
        summary = {
            "domain": domain,
            "task_id": task_id,
            "speech_complexity": speech_complexity,
            "seed": seed,
            "persona": user_info["persona_name"],
            "agent": {
                # ``ref`` is a provider transport's, not every transport's.
                "provider": getattr(getattr(transport, "ref", None), "provider", None),
                "model": getattr(getattr(transport, "ref", None), "model", None),
                "session_id": getattr(transport, "session_id", None),
                "errors": list(getattr(transport, "errors", [])),
            },
            "termination": result.termination.value,
            "steps": len(result.steps),
            "call_s": round(len(result.steps) * turns["tick_duration_s"], 1),
            "wall_s": round(result.wall_s, 1),
            "truncations": len(result.truncations),
            "caller_llm_calls": len(content.exchanges),
            "usage": [u.model_dump(exclude_none=True) for u in usage],
            "caller_plan": simulator.pipeline.as_record().as_dict(),
            "caller_lines": len(simulator.utterances),
            "caller_lines_cut_off": sum(1 for u in simulator.utterances if u.truncated),
            "caller_effects": effects.summary() if effects is not None else None,
            "caller_reactions": listener.summary() if listener is not None else None,
            "sidecar_state": closed,
        }
        _write(out_dir, result, summary)
        # tau2's own record of the call, beside ours; then the call as ATIF.
        try:
            state = await sidecar.state()
        except SidecarError as exc:
            state = {"unavailable": str(exc)}
        (out_dir / "tau2_state.json").write_text(
            json.dumps(state, indent=1), encoding="utf-8"
        )
        trajectory = build_trajectory(
            result,
            tick_s=turns["tick_duration_s"],
            agent_tool_log=executor.log,
            summary=summary,
            audio=out_dir / "both.wav",
            attempt=attempt_index,
        )
        (out_dir / "trajectory.json").write_text(
            trajectory.to_atif_json(), encoding="utf-8"
        )
        return summary
    finally:
        await sidecar.aclose()
        scratch.cleanup()


def _stepping(
    transport: VoiceTransport, *, live: bool, step_s: float
) -> tuple[SessionConfig, dict[str, Any]]:
    """The agent's session and how its call steps, per tau2 adapter.

    OpenAI Realtime takes the phone line's μ-law both ways, steps with
    SteppedCall's defaults. OpenAI Live, Gemini Live and a duplex cascade
    speak PCM: their steppers convert the line as tau2's adapters do. Live
    and a cascade get each step's caller audio whole, Gemini in paced
    packets; Live is also paced as tau2 paces it.
    """
    agent = PartyId("agent")
    gemini = isinstance(transport, GeminiLiveTransport)
    pipeline = transport.capabilities.pipeline
    if not live and not gemini and pipeline is not Pipeline.CASCADE:
        return SessionConfig(party=agent, audio_in=TELEPHONY, audio_out=TELEPHONY), {}
    ulaw_step = int(TELEPHONY.sample_rate_hz * step_s)
    stepper: AgentStepper
    if gemini:
        audio_in = transport.capabilities.audio_in
        audio_out = transport.capabilities.audio_out
        stepper = GeminiAgentStepper(
            bytes_per_step=ulaw_step,
            bytes_per_second=TELEPHONY.sample_rate_hz,
            input_rate=audio_in.sample_rate_hz,
            output_rate=audio_out.sample_rate_hz,
        )
    elif live:
        audio_in = audio_out = LIVE_AUDIO
        stepper = LiveAgentStepper(
            bytes_per_step=ulaw_step, bytes_per_second=TELEPHONY.sample_rate_hz
        )
    else:
        audio_in = transport.capabilities.audio_in
        audio_out = transport.capabilities.audio_out
        if audio_out.encoding != "pcm16" or audio_out.channels != 1:
            raise ValueError(
                f"the cascade's TTS gives {audio_out}; tau2's line converter "
                f"takes mono PCM16"
            )
        stepper = CascadeAgentStepper(
            bytes_per_step=ulaw_step,
            bytes_per_second=TELEPHONY.sample_rate_hz,
            input_rate=audio_in.sample_rate_hz,
            output_rate=audio_out.sample_rate_hz,
        )
    return SessionConfig(party=agent, audio_in=audio_in, audio_out=audio_out), {
        "stepper": stepper,
        "realtime_pacing": live,
        "send_format": audio_in,
    }


def call_usage(
    *,
    transport: Any,
    content: Tau2Content,
    listener: Tau2Listener | None,
    tts: Any,
    caller_llm: str,
) -> list[UsageRecord]:
    """What the call consumed, part by part, in tokens and characters.

    The agent's comes from its transport, where the transport meters it (the
    OpenAI Realtime one does); the caller's LLM, its decisions and its TTS
    from the caller's side.
    """
    usage: list[UsageRecord] = []
    records = getattr(transport, "usage_records", None)
    record = getattr(transport, "usage_record", None)
    if callable(records):
        usage.extend(records("agent"))
    elif callable(record):
        usage.append(record("agent"))
    usage.append(
        chat_usage(
            (e.reply for e in content.exchanges),
            component="caller.llm",
            model=caller_llm,
        )
    )
    if listener is not None:
        usage.append(listener.usage_record())
    usage.append(
        UsageRecord(
            component="caller.tts",
            model=getattr(tts, "model", None),
            characters=tts.characters_billed,
        )
    )
    return usage


Grader = Callable[[Path], Awaitable[Mapping[str, Any]]]
"""Grades the attempt just closed; returns the verifier's ``result.json``.

It runs the verifier where the sidecar's state lives (``tests/evaluate.py``
in the sidecar image) and copies what it needs into the attempt directory.
"""


async def run_trial(
    *,
    grade: Grader,
    out_dir: Path,
    seed: int = 42,
    hallucination_retries: int = HALLUCINATION_RETRIES,
    attempt: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    on_progress: Callable[[str], None] | None = None,
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
    summaries: list[Mapping[str, Any]] = []
    graded: Mapping[str, Any] = {}
    report = on_progress or (lambda message: None)
    for n in range(hallucination_retries + 1):
        attempt_dir = out_dir / f"attempt-{n}"
        report(f"attempt {n}: started, seed {seed + n * 1000}")
        summary = await run(
            out_dir=attempt_dir,
            seed=seed + n * 1000,
            hallucination_feedback=feedback,
            attempt_index=n,
            **task,
        )
        graded = await grade(attempt_dir)
        grading = [UsageRecord.model_validate(u) for u in graded.get("usage") or []]
        if grading:
            summary = _add_usage(attempt_dir, summary, grading)
        summaries.append(summary)
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
        call_s = summary.get("call_s")
        report(
            f"attempt {n}: reward {graded.get('reward')} · "
            f"hallucination {'yes' if found else 'no'} · "
            f"{summary.get('termination')} · "
            + (f"{call_s:.0f} s audio" if call_s is not None else "no audio")
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
        # Every attempt consumed tokens, the set-aside ones included.
        "usage": [
            u.model_dump(exclude_none=True)
            for u in merge_usage(
                UsageRecord.model_validate(record)
                for summary in summaries
                for record in summary.get("usage") or []
            )
        ],
        "hallucination_retries_used": len(attempts) - 1,
        "attempts": attempts,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trial.json").write_text(json.dumps(trial, indent=2), encoding="utf-8")
    return trial


def _add_usage(
    attempt_dir: Path, summary: Mapping[str, Any], records: Sequence[UsageRecord]
) -> dict[str, Any]:
    """Grading's own LLM calls, added to the attempt's record once graded:
    its ``summary.json`` and its ATIF-voice trajectory, as far as they exist."""
    summary = {
        **summary,
        "usage": [
            *(summary.get("usage") or []),
            *(u.model_dump(exclude_none=True) for u in records),
        ],
    }
    path = attempt_dir / "summary.json"
    if path.exists():
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    atif = attempt_dir / "trajectory.json"
    if atif.exists():
        voice = VoiceTrajectory.from_atif(
            Trajectory.model_validate_json(atif.read_text(encoding="utf-8"))
        )
        voice = voice.model_copy(update={"usage": (*voice.usage, *records)})
        atif.write_text(voice.to_atif_json(), encoding="utf-8")
    return summary


def _close_body(result: CallResult, step_s: float) -> dict[str, Any]:
    """The step log for ``/close``: both chunks of every step, as tau2 records
    them, with the fields its interaction metrics read."""
    steps = []
    for log in result.steps:
        user = log.user
        effects = user.effects or {}
        agent = log.agent
        steps.append(
            {
                "step": log.step,
                "t_s": log.t_s,
                "user": {
                    "text": user.text if user.is_meaningful else None,
                    "speech": user.speech,
                    "action": user.action.value if user.action else None,
                    "utterance_id": user.utterance_id,
                    "speech_effects": effects.get("speech_effects"),
                    "source_effects": effects.get("source_effects"),
                    "channel_effects": effects.get("channel_effects"),
                },
                "agent": {
                    "text": log.agent_text,
                    "speech": log.agent_speech,
                    "utterance_ids": list(agent.utterances) if agent else None,
                    "raw_data": agent.raw_data() if agent else None,
                },
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
