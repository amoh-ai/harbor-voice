"""A scripted call with listener reactions, run by tau2's caller or by ours.

``control_behavior``: tau2's caller interrupts and backchannels when its
decision model says so, and inserts vocal tics and out-of-turn phrases. Here
the decision model, the caller's LLM and synthesis are scripted — decisions
are answered from fixed lists, one per kind, in the order asked — and
everything else is real: the policy, the check cadence, the counters, the
backchannel phrases, the effects, the phone line.

Generate the golden inside the sidecar image, with the current ``server.py``::

    docker run --rm -v "$PWD/tests/adapters/tau-voice:/cases" \\
        -v "$PWD/adapters/tau-voice/sidecar/server.py:/app/server.py" \\
        tau-voice-runtime:<tag> python /cases/tau2_caller_behavior_cases.py --tau2 \\
        > tests/adapters/tau-voice/fixtures/tau2_caller_behavior.json

The golden also records every decision prompt tau2 built, and tau2's
interaction metrics for the call (its caller's chunks, with a scripted agent
record). Two checks replay *our* side through the sidecar in the image: the
``/decide`` requests must give tau2's prompts, and the ``/close`` body must
give tau2's metrics::

    python tests/adapters/tau-voice/tau2_caller_behavior_cases.py --ours \\
        > ours.json                # harbor-voice's environment
    docker run ... -v "$PWD/ours.json:/ours.json" tau-voice-runtime:<tag> \\
        python /cases/tau2_caller_behavior_cases.py --check /ours.json
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

TASK = ("airline", "7")
SEED = 42
REPLIES = [
    {"text": "I need to change the date of my flight, please."},
    {"text": "Sorry to cut in, but is that change fee refundable?"},
    {"tool": "check_status", "text": None},
    {"text": "Okay, it says it is confirmed now."},
    {"text": "Great, thank you so much. ###STOP###"},
]
ANSWERS = {
    "interrupt": ["NO", "NO", "NO", "NO", "YES"] + ["NO"] * 60,
    "backchannel": ["NO", "NO", "YES"] + ["NO"] * 60,
}
TOOL_RESULT = "status: confirmed"
GREETING = "Hi! Thanks for calling. How can I help you today?"
MONOLOGUE = (
    "Sure. Changing a date is possible. There is a change fee of seventy five "
    "dollars, and the fare difference applies. I can look at options for you. "
    "Let me pull up your reservation and check the flights on the new date."
)


def line_pcm(text: str) -> bytes:
    """Scripted synthesis: 50 ms per character, a tone per text."""
    n = 800 * len(text)
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    t = np.arange(n) / 16000
    signal = 0.25 * np.sin(2 * np.pi * (300 + seed % 500) * t)
    return (signal * 32767).astype("<i2").tobytes()


def spoken(text: str, steps: int) -> list[tuple[str | None, bool]]:
    """An agent utterance as speech steps, its words spread over them."""
    words = text.split()
    per = [[] for _ in range(steps)]
    for index, word in enumerate(words):
        per[index * steps // len(words)].append(word)
    return [(" ".join(w) + " " if w else None, True) for w in per]


def agent_script() -> list[tuple[str | None, bool]]:
    silent: list[tuple[str | None, bool]] = [(None, False)]
    return (
        silent
        + spoken(GREETING, 12)
        + silent * 30  # the caller's first line
        + spoken(MONOLOGUE, 60)  # backchannel, then an interruption
        + silent * 40  # the caller asks for a tool; its result
        + silent * 30
        + spoken("Your new departure is at nine in the morning.", 8)
        + silent * 40
    )


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


NO_EFFECTS = {"speech_effects": None, "source_effects": None, "channel_effects": None}


def agent_outputs(steps: int) -> list[dict[str, Any]]:
    """The agent's own chunk per step, for the record: what the caller hears a
    step later, with the ``TickResult`` fields tau2's agent keeps."""
    script = agent_script()
    out: list[dict[str, Any]] = []
    utterance, previous = 0, False
    for step in range(steps):
        text, speech = script[step + 1] if step + 1 < len(script) else (None, False)
        if speech and not previous:
            utterance += 1
        previous = speech
        out.append(
            {
                "text": text,
                "speech": speech,
                "utterance_ids": [f"item_{utterance}"] if speech else None,
                "raw_data": {
                    "vad_events": [],
                    "proportional_transcript": text or "",
                    "was_truncated": False,
                    "truncated_audio_bytes": 0,
                    "interruption_audio_start_ms": None,
                    "cumulative_user_audio_at_tick_start_ms": step * 200,
                },
            }
        )
    return out


def metrics_panel(ticks: list[Any]) -> dict[str, Any]:
    """tau2's interaction metrics for one call."""
    from types import SimpleNamespace

    from tau2.metrics.voice_interaction_metrics import InteractionMetricsConfig
    from tau2.metrics.voice_interaction_metrics import (
        compute_interaction_metrics_for_experiment,
    )

    sim = SimpleNamespace(ticks=ticks, id="scripted", task_id=TASK[1])
    return compute_interaction_metrics_for_experiment(
        SimpleNamespace(simulations=[sim]), InteractionMetricsConfig()
    )


class Utterances:
    """Utterance ids are random; compare which steps share one."""

    def __init__(self) -> None:
        self._index: dict[str, int] = {}

    def __call__(self, ids: list[str] | None) -> list[int] | None:
        if not ids:
            return None
        return [self._index.setdefault(i, len(self._index)) for i in ids]


# ------------------------------------------------------------------- tau2


def _session() -> Any:
    sys.path.insert(0, "/app")  # the sidecar's server.py
    import server

    domain, task_id = TASK
    return server, server._Session(
        server.OpenRequest(
            domain=domain,
            task_id=task_id,
            speech_complexity="control_behavior",
            seed=SEED,
        )
    )


def _scripted_decisions(prompts: list[dict[str, Any]]) -> Any:
    from tau2.data_model.message import AssistantMessage

    answers = {kind: iter(values) for kind, values in ANSWERS.items()}
    lock = threading.Lock()
    current = {"step": -1}

    def decide(kind: str, messages: list[Any]) -> Any:
        with lock:
            answer = next(answers[kind])
            prompts.append(
                {"step": current["step"], "kind": kind, "prompt": messages[0].content}
            )
        reply = AssistantMessage(role="assistant", content=answer, cost=0.0)
        reply.generation_time_seconds = 0.0
        return reply

    return decide, current


def run_tau2() -> dict[str, Any]:
    import tau2.agent.base.voice as voice_mixin
    import tau2.user.user_simulator_streaming as uss
    import tau2.voice.synthesis.audio_effects.speech_generator as speech_generator
    from tau2.data_model.audio import AudioData
    from tau2.data_model.audio import AudioEncoding
    from tau2.data_model.audio import AudioFormat
    from tau2.data_model.message import AssistantMessage
    from tau2.data_model.message import ToolCall
    from tau2.data_model.message import ToolMessage

    prompts: list[dict[str, Any]] = []
    decide, current = _scripted_decisions(prompts)
    replies = iter(REPLIES)

    def generate(
        model: str, messages: list[Any], call_name: str | None = None, **_: Any
    ) -> Any:
        if call_name == "interruption_decision":
            return decide("interrupt", messages)
        if call_name == "backchannel_decision":
            return decide("backchannel", messages)
        reply = next(replies)
        calls = None
        if "tool" in reply:
            calls = [ToolCall(id="u1", name=reply["tool"], arguments={})]
        message = AssistantMessage(
            role="assistant", content=reply["text"], tool_calls=calls
        )
        message.generation_time_seconds = 0.0
        return message

    def synthesize(*, text: str, provider: str, provider_config: Any) -> AudioData:
        fmt = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=16000, channels=1
        )
        return AudioData(data=line_pcm(text), format=fmt)

    uss.generate = generate
    voice_mixin.synthesize_voice = synthesize
    speech_generator.synthesize_voice = synthesize

    _, session = _session()
    user = session.user
    described = session.describe()["user"]
    state = user.get_init_state()
    out, pending_tool, said = [], None, []
    utterances = Utterances()

    def dump(value: Any) -> Any:
        return value.model_dump(mode="json") if value is not None else None

    for step, (text, speech) in enumerate(agent_script()):
        current["step"] = step
        heard = AssistantMessage(role="assistant", content=text, contains_speech=speech)
        tool_results = None
        if pending_tool is not None:
            tool_results = ToolMessage(
                id=pending_tool, role="tool", content=TOOL_RESULT, requestor="user"
            )
            pending_tool = None
        msg, state = user.get_next_chunk(
            state, participant_chunk=heard, tool_results=tool_results
        )
        said.append(msg)
        calls = [c.name for c in (msg.tool_calls or [])]
        if calls:
            pending_tool = msg.tool_calls[0].id
        audio = msg.get_audio_bytes() or b""
        action = msg.turn_taking_action
        out.append(
            {
                "action": action.action if action else None,
                "speech": bool(msg.contains_speech),
                "text": msg.content,
                "tool_calls": calls,
                "audio": digest(audio),
                "audio_len": len(audio),
                "stop": user.is_stop(msg),
                "utterance": utterances(msg.utterance_ids),
                "effects": {
                    "speech_effects": dump(msg.speech_effects),
                    "source_effects": dump(msg.source_effects),
                    "channel_effects": dump(msg.channel_effects),
                },
            }
        )
        if user.is_stop(msg):
            break
    prompts.sort(key=lambda p: (p["step"], p["kind"]))
    from tau2.data_model.message import Tick

    ticks = [
        Tick(
            tick_id=step,
            timestamp="2026-01-01T00:00:00",
            user_chunk=msg,
            agent_chunk=AssistantMessage(
                role="assistant",
                content=agent["text"],
                contains_speech=agent["speech"],
                utterance_ids=agent["utterance_ids"],
                raw_data=agent["raw_data"],
            ),
        )
        for step, (msg, agent) in enumerate(zip(said, agent_outputs(len(said))))
    ]
    return {
        "source": "tau2-bench@2174a60 VoiceStreamingUserSimulator, control_behavior, "
        "scripted LLM, decisions and synthesis",
        "metrics": metrics_panel(ticks),
        "user": {
            "voice_settings": described["voice_settings"],
            "tau2_burst_files": described["tau2_burst_files"],
            "turn_taking": described["turn_taking"],
        },
        "steps": out,
        "decisions": prompts,
    }


def check(ours_path: str) -> None:
    """Replay our ``/decide`` requests and ``/close`` body through the sidecar.

    The prompts must be tau2's, and tau2's interaction metrics on the ticks
    the sidecar writes must equal those on tau2's own caller's chunks.
    """
    import tau2.user.user_simulator_streaming as uss

    server, session = _session()
    golden = json.loads(
        (Path(__file__).parent / "fixtures" / "tau2_caller_behavior.json").read_text()
    )
    data = json.loads(Path(ours_path).read_text())
    prompts: list[dict[str, Any]] = []
    decide, current = _scripted_decisions(prompts)
    uss.generate = lambda model, messages, call_name=None, **_: decide(
        "interrupt" if call_name == "interruption_decision" else "backchannel", messages
    )
    for request in data["requests"]:
        current["step"] = request.pop("step")
        session.decide(server.UserContextRequest(**request))
    prompts.sort(key=lambda p: (p["step"], p["kind"]))
    run = session.close(server.CloseRequest(**data["close"]))
    metrics = metrics_panel(run.ticks or [])
    ok = True
    for name, got, want in (
        ("decision prompts", prompts, golden["decisions"]),
        ("interaction metrics", metrics, golden["metrics"]),
    ):
        same = got == want
        ok &= same
        print(f"{name}: {'IDENTICAL' if same else 'DIFFER'}")
        if not same:
            print("--- ours", json.dumps(got, indent=1)[:3000])
            print("--- tau2", json.dumps(want, indent=1)[:3000])
    raise SystemExit(0 if ok else 1)


# -------------------------------------------------------------------- ours


async def run_ours(golden: dict[str, Any]) -> dict[str, Any]:
    """Our caller on the same script; returns its steps and decide requests."""
    from collections.abc import AsyncIterator
    from collections.abc import Sequence

    from harbor_voice.ids import PartyId
    from harbor_voice.perturbation.persona import VoiceSelection
    from harbor_voice.simulator.content import SourceKind
    from tau_voice.caller import ReactiveCaller
    from tau_voice.caller import ReactivePolicy
    from tau_voice.caller import tau2_backchannels
    from tau_voice.client import _context
    from tau_voice.content import Tau2Listener
    from tau_voice.effects import Tau2CallerEffects
    from tau_voice.reactive import CallerToolCall
    from tau_voice.reactive import CallerToolResult
    from tau_voice.reactive import HangUp
    from tau_voice.reactive import Heard
    from tau_voice.reactive import Say
    from tau_voice.agent_steps import AgentStep
    from tau_voice.reactive import UseTools
    from tau_voice.run import SYNTHESIS
    from tau_voice.run import _close_body
    from tau_voice.run import caller_simulator
    from tau_voice.stepped_call import CallResult
    from tau_voice.stepped_call import StepLog
    from tau_voice.stepped_call import Termination
    from tau_voice.voice import SimulatorVoice

    class ScriptedTTS:
        name = "scripted.tts"
        audio_format = SYNTHESIS

        def select_voice(self, selection: VoiceSelection) -> bool:
            return selection.voice_id is not None

        async def synthesize(
            self, text: str, *, party: PartyId
        ) -> AsyncIterator[bytes]:
            yield line_pcm(text)

    class ScriptedContent:
        kind = SourceKind.SCRIPT
        is_deterministic = True

        def __init__(self) -> None:
            self._replies = iter(REPLIES)

        async def respond(self, steps: Sequence[Any], pending: Any) -> Any:
            reply = next(self._replies)
            if "tool" in reply:
                return UseTools(
                    (CallerToolCall("u1", reply["tool"], {}),), text=reply["text"]
                )
            if "###STOP###" in reply["text"]:
                return HangUp(text=reply["text"], reason="###STOP###")
            return Say(reply["text"])

        async def run_tools(self, calls: Sequence[Any], *, step: int) -> Any:
            raise AssertionError("the script delivers tool results itself")

    turns = golden["user"]["turn_taking"]
    answers = {kind: iter(values) for kind, values in ANSWERS.items()}
    requests: list[dict[str, Any]] = []

    class ScriptedSidecar:
        """``/decide`` answered from the script; the requests kept."""

        async def decide(self, steps: Sequence[Any], pending: Any) -> dict[str, Any]:
            requests.append({"step": len(steps), **_context(steps, pending)})
            interrupt = backchannel = False
            if turns["enable_user_initiated_interruption"]:
                interrupt = next(answers["interrupt"]) == "YES"
            if turns["use_llm_backchannel"]:
                backchannel = next(answers["backchannel"]) == "YES"
            return {"should_interrupt": interrupt, "should_backchannel": backchannel}

    settings = golden["user"]["voice_settings"]
    effects = Tau2CallerEffects(
        settings, turn_burst_files=golden["user"]["tau2_burst_files"]
    )
    simulator = caller_simulator(
        ScriptedTTS(), voice_id="voice", seed=SEED, complexity="control_behavior"
    )

    async def synthesize(text: str) -> bytes:
        return (await simulator.prepare(text)).pcm

    await effects.open(synthesize, voice_id="voice")
    listener = Tau2Listener(ScriptedSidecar())  # type: ignore[arg-type]
    caller = ReactiveCaller(
        content=ScriptedContent(),
        voice=SimulatorVoice(simulator, step_s=0.2, effects=effects),
        policy=ReactivePolicy(
            wait_other=turns["wait_to_respond_other_ticks"],
            wait_self=turns["wait_to_respond_self_ticks"],
            yield_when_interrupted=turns["yield_when_interrupted_ticks"],
            yield_when_interrupting=turns["yield_when_interrupting_ticks"],
            listener=listener,
            check_interval=turns["interruption_check_interval_ticks"],
        ),
        backchannels=tau2_backchannels(settings["speech_environment"]["voice_seed"]),
    )
    out, pending_tool, chunks = [], None, []
    utterances = Utterances()
    for text, speech in agent_script():
        results = None
        if pending_tool is not None:
            results = (CallerToolResult(pending_tool, TOOL_RESULT),)
            pending_tool = None
        chunk = await caller.next_chunk(Heard(text, speech), results)
        chunks.append(chunk)
        if chunk.tool_calls:
            pending_tool = chunk.tool_calls[0].id
        out.append(
            {
                "action": chunk.action.value if chunk.action else None,
                "speech": chunk.speech,
                "text": chunk.text,
                "tool_calls": [c.name for c in chunk.tool_calls],
                "audio": digest(chunk.audio),
                "audio_len": len(chunk.audio),
                "stop": chunk.hang_up is not None,
                "utterance": utterances(
                    [chunk.utterance_id] if chunk.utterance_id else None
                ),
                "effects": dict(chunk.effects or NO_EFFECTS),
            }
        )
        if chunk.hang_up is not None:
            break
    result = CallResult(termination=Termination.USER_STOP)
    for step, (chunk, agent) in enumerate(zip(chunks, agent_outputs(len(chunks)))):
        result.steps.append(
            StepLog(
                step=step,
                t_s=step * 0.2,
                user=chunk,
                agent_text=agent["text"],
                agent_speech=agent["speech"],
                agent=AgentStep(
                    played=b"",
                    raw_bytes=1 if agent["speech"] else 0,
                    transcript=agent["text"] or "",
                    was_truncated=False,
                    truncated_bytes=0,
                    utterances=tuple(agent["utterance_ids"] or ()),
                    user_ms_at_start=step * 200,
                ),
            )
        )
    return {
        "steps": out,
        "requests": requests,
        "checks": listener.summary(),
        "close": _close_body(result, 0.2),
    }


if __name__ == "__main__":
    if "--tau2" in sys.argv:
        print(json.dumps(run_tau2(), indent=1))
    elif "--check" in sys.argv:
        check(sys.argv[sys.argv.index("--check") + 1])
    elif "--ours" in sys.argv:
        golden = json.loads(
            (
                Path(__file__).parent / "fixtures" / "tau2_caller_behavior.json"
            ).read_text()
        )
        ours = asyncio.run(run_ours(golden))
        print(json.dumps({"requests": ours["requests"], "close": ours["close"]}))
    else:
        raise SystemExit("--tau2 | --ours | --check <ours.json>")
