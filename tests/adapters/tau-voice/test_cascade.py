"""A cascade agent on tau-voice: off by default, and stepped as tau2's cascaded adapter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from harbor.models.job.config import JobConfig

from harbor_voice.cascade import DuplexCascadeTransport
from harbor_voice.driver import DriverContext
from harbor_voice.ids import PartyId
from harbor_voice.providers import ScriptedSTT
from harbor_voice.providers import ToneTTS
from harbor_voice.providers.offline import ToolCallingEchoLLM
from harbor_voice.providers.openai.realtime import TELEPHONY
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.providers.tools import ToolSpec
from harbor_voice.simulator import PreparedLine
from harbor_voice.simulator.content import SourceKind
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import InputTranscript
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.trial import AgentSpec
from harbor_voice.trial import TrialConfigError
from harbor_voice.trial import VoiceTaskConfig
from tau_voice.agent_steps import CascadeAgentStepper
from tau_voice.caller import PatientPolicy
from tau_voice.caller import ReactiveCaller
from tau_voice.cascade import AUDIO_NATIVE_VOICE_INSTRUCTION
from tau_voice.cascade import CASCADED_MODEL_INSTRUCTION
from tau_voice.cascade import cascade_prompt
from tau_voice.cascade import is_cascade
from tau_voice.client import ToolOutcome
from tau_voice.job import job_config
from tau_voice.reactive import CallerReply
from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.reactive import HangUp
from tau_voice.reactive import Pending
from tau_voice.reactive import Say
from tau_voice.reactive import StepRecord
from tau_voice.run import SidecarToolExecutor
from tau_voice.run import _stepping
from tau_voice.stepped_call import SteppedCall
from tau_voice.stepped_call import Termination
from tau_voice.trial import drive

pytestmark = pytest.mark.unit

CASCADE = AgentSpec.model_validate(
    {
        "pipeline": "cascade",
        "stt": "offline.scripted-stt",
        "llm": "offline.tool-echo-llm",
        "tts": "offline.tone-tts",
    }
)
S2S = AgentSpec.model_validate(
    {"pipeline": "s2s", "s2s": "openai.realtime/gpt-realtime-1.5"}
)


# ------------------------------------------------------------------ opt-in


async def test_a_cascade_is_refused_by_default(tmp_path: Path) -> None:
    config = VoiceTaskConfig.model_validate(
        {
            "call": {
                "driver": "tau_voice.trial:drive",
                "config": {"domain": "airline", "task_id": "0"},
            }
        }
    )

    def build_agent(**_: Any) -> Any:
        pytest.fail("a refused cascade must not be built")

    context = DriverContext(
        config=config,
        driver_config=config.call.config if config.call else {},
        task_dir=tmp_path,
        out_dir=tmp_path,
        build_agent=build_agent,
        agent=CASCADE,
    )
    with pytest.raises(TrialConfigError, match="--allow-cascade") as refused:
        await drive(context)
    assert '"allow_cascade": true' in str(refused.value)


def test_a_run_that_opts_in_takes_a_cascade() -> None:
    assert is_cascade(CASCADE, {"allow_cascade": True}) is True
    assert is_cascade(S2S, {}) is False
    assert is_cascade(None, {}) is False


def test_job_config_needs_the_opt_in_and_passes_it_to_the_driver(
    tmp_path: Path,
) -> None:
    slots = ("deepgram.stt/nova-3", "openai.llm/gpt-4.1", "deepgram.tts/aura-2")
    with pytest.raises(ValueError, match="--allow-cascade"):
        job_config(dataset=tmp_path, cascade=slots)

    raw = job_config(
        dataset=tmp_path,
        cascade=slots,
        allow_cascade=True,
        agent_config={"llm": {"temperature": 0}},
    )
    (agent,) = JobConfig.model_validate(raw).agents
    assert (
        agent.model_name == "deepgram.stt/nova-3+openai.llm/gpt-4.1+deepgram.tts/aura-2"
    )
    assert agent.kwargs["call_config"] == {"allow_cascade": True}
    spec = AgentSpec.model_validate_json(agent.kwargs["agent_json"])
    assert (spec.stt, spec.llm, spec.tts) == slots
    assert spec.config == {"llm": {"temperature": 0}}


def test_a_cascade_gets_tau2s_cascade_instructions_over_the_same_policy() -> None:
    policy = "# Airline policy\n\nBe kind."
    prompt = cascade_prompt(f"{AUDIO_NATIVE_VOICE_INSTRUCTION}\n\n{policy}")
    assert prompt == f"{CASCADED_MODEL_INSTRUCTION}\n\n{policy}"
    with pytest.raises(ValueError, match="instructions"):
        cascade_prompt("Some other prompt")


# ---------------------------------------------------------------- barge-in


def _stepper() -> CascadeAgentStepper:
    """0.2 s steps on the 8 kHz line; the cascade's TTS at 16 kHz."""
    return CascadeAgentStepper(
        bytes_per_step=1600,
        bytes_per_second=8000,
        input_rate=16_000,
        output_rate=16_000,
    )


TONE = b"\x00\x10" * 3200
"""0.2 s of loud 16 kHz PCM16: one step of agent speech once converted."""


def test_barge_in_drops_all_unplayed_agent_audio_and_its_transcript() -> None:
    stepper = _stepper()
    stepper.begin()
    stepper.feed(OutputTranscript("utt_0", "Your flight is booked for Monday.", 0.0))
    for _ in range(4):  # 0.8 s of reply, delivered at once as a TTS does
        stepper.feed(OutputAudio(TONE, "utt_0", 0.0))
    first = stepper.end(user_bytes=1600)
    assert first.contains_speech and first.transcript
    assert not first.was_truncated

    # The caller starts speaking: this step's share and the carried rest go.
    stepper.begin()
    stepper.feed(InputSpeech(True, 200, 0.0))
    cut = stepper.end(user_bytes=1600)
    assert cut.vad == ("speech_started", "interrupted")
    assert cut.raw_bytes == 0 and not cut.contains_speech
    assert cut.played == b"\x7f" * 1600
    assert cut.transcript == ""
    # tau2 cuts nothing in the provider: no truncation is reported.
    assert not cut.was_truncated and cut.truncated_bytes == 0

    stepper.begin()
    assert stepper.end(user_bytes=1600).raw_bytes == 0


def test_a_transcript_is_a_barge_in_too_and_late_audio_plays_unannotated() -> None:
    """tau2 also treats the recogniser's transcripts as barge-in, and plays the
    rest of a forgotten utterance without its text."""
    stepper = _stepper()
    stepper.begin()
    stepper.feed(OutputTranscript("utt_0", "Let me check that for you.", 0.0))
    stepper.feed(OutputAudio(TONE, "utt_0", 0.0))
    stepper.feed(InputTranscript(None, "wait", 0.0))
    step = stepper.end(user_bytes=1600)
    assert step.vad == ("interrupted",) and step.raw_bytes == 0

    stepper.begin()
    stepper.feed(OutputAudio(TONE, "utt_0", 0.0))
    late = stepper.end(user_bytes=1600)
    assert late.contains_speech and late.transcript == ""


def test_speech_ending_or_starting_over_silence_cuts_nothing() -> None:
    stepper = _stepper()
    stepper.begin()
    stepper.feed(InputSpeech(True, 0, 0.0))
    stepper.feed(InputSpeech(False, 900, 0.0))
    stepper.feed(OutputTranscript("utt_0", "Sure.", 0.0))
    stepper.feed(OutputAudio(TONE, "utt_0", 0.0))
    step = stepper.end(user_bytes=1600)
    assert step.vad == ("speech_started", "speech_stopped")
    assert step.contains_speech and step.transcript == "Sure."


# ------------------------------------------------------------- a whole call

SPEECH = b"\x10"
"""A loud μ-law byte; ``\\xff`` is silence."""


class Voice:
    """The caller's voice: 20 ms of loud μ-law per 5 characters said."""

    step_bytes = 640  # 20 ms at 16 kHz
    last_effects = None

    async def prepare(self, text: str) -> PreparedLine:
        pcm = b"\x01\x01" * (320 * max(1, len(text) // 5))
        return PreparedLine(said=text, pcm=pcm, clean_digest=None)

    def emit_step(
        self, pcm: bytes, *, is_speech: bool, line: PreparedLine | None = None
    ) -> bytes:
        return (SPEECH if any(pcm) else b"\xff") * (len(pcm) // 4)

    def finish(self, line: PreparedLine, *, truncated: bool) -> None:
        pass


class Script:
    def __init__(self, replies: list[CallerReply]) -> None:
        self.replies = iter(replies)
        self.heard: list[str] = []

    @property
    def kind(self) -> SourceKind:
        return SourceKind.SCRIPT

    @property
    def is_deterministic(self) -> bool:
        return True

    async def respond(
        self, steps: Sequence[StepRecord], pending: Pending
    ) -> CallerReply:
        return next(self.replies)

    async def run_tools(
        self, calls: Sequence[CallerToolCall], *, step: int
    ) -> tuple[CallerToolResult, ...]:
        return ()


class Sidecar:
    """tau2's tool endpoint: records each call and answers it."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def tool(self, **call: Any) -> ToolOutcome:
        self.calls.append(call)
        return ToolOutcome(
            content="user u1: Jane Doe", error=False, stop=False, too_many_errors=False
        )


async def test_a_cascade_takes_a_whole_call_with_a_tool_at_a_step_boundary() -> None:
    sidecar = Sidecar()
    executor = SidecarToolExecutor(
        sidecar,  # ty: ignore[invalid-argument-type]
        [ToolSpec("find_user", "Look a customer up.")],
    )
    agent = DuplexCascadeTransport(
        stt=ScriptedSTT(turns=["I need to change a flight."]),
        llm=ToolCallingEchoLLM(tool="find_user", arguments={"user_id": "u1"}),
        tts=ToneTTS(),
        party=PartyId("agent"),
        system_prompt=CASCADED_MODEL_INSTRUCTION,
        tools=ToolRouter(executor),
    )
    caller = ReactiveCaller(
        content=Script(
            [Say("I need to change a flight."), HangUp("Bye ###STOP###", "###STOP###")]
        ),
        voice=Voice(),  # ty: ignore[invalid-argument-type]
        policy=PatientPolicy(wait_other=2, wait_self=60),
    )

    async def user_tools(calls: Any, step: int) -> tuple[CallerToolResult, ...]:
        return ()

    step_s = 0.02
    session, stepping = _stepping(agent, live=False, step_s=step_s)
    await agent.open(session)
    call = SteppedCall(
        caller=caller,
        transport=agent,
        run_user_tools=user_tools,
        audio_format=TELEPHONY,
        step_s=step_s,
        max_steps=300,
        **stepping,
    )
    executor.call = call
    try:
        result = await call.run()
    finally:
        await agent.close()

    assert result.termination is Termination.USER_STOP
    assert agent.errors == []
    # The tool ran once, at the step its call arrived in, through the sidecar.
    (entry,) = executor.log
    assert (entry["name"], entry["arguments"]) == ("find_user", {"user_id": "u1"})
    assert entry["result"] == "user u1: Jane Doe"
    assert [c["step"] for c in sidecar.calls] == [entry["step"]]
    # Its result reached the model, which answered aloud after that step; the
    # caller heard the answer in pieces as its audio played.
    spoken = [s for s in result.steps if s.agent_speech]
    assert spoken and min(s.step for s in spoken) > entry["step"]
    heard = "".join(s.agent_text or "" for s in spoken)
    assert heard == "the tool said: user u1: Jane Doe"
    assert {
        "role": "tool",
        "tool_call_id": "call-0",
        "content": "user u1: Jane Doe",
    } in (agent.messages)
