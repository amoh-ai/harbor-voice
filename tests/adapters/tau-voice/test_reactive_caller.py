"""ReactiveCaller + PatientPolicy against tau2's own caller, step for step.

The fixture is tau2's ``VoiceStreamingUserSimulator`` (control) driven with a
scripted LLM and scripted synthesis; everything else — the policy, the
counters, the chunking, the phone line — is tau2's real code. Ours is given
the same script and must produce the same action, speech flag, text slice,
tool calls and phone-line audio bytes at every step.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from collections.abc import AsyncIterator

from harbor_voice.ids import PartyId
from harbor_voice.perturbation.persona import VoiceSelection
from tau_voice.run import SYNTHESIS
from tau_voice.run import caller_simulator
from tau_voice.voice import SimulatorVoice
from tau_voice.caller import PatientPolicy
from tau_voice.caller import ReactiveCaller
from tau_voice.caller import split_text
from harbor_voice.simulator.content import SourceKind
from tau_voice.reactive import CallerReply
from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.reactive import HangUp
from tau_voice.reactive import Heard
from tau_voice.reactive import Pending
from tau_voice.reactive import Say
from tau_voice.reactive import StepRecord
from tau_voice.reactive import UseTools

pytestmark = pytest.mark.unit

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "tau2_caller_control.json").read_text(
        encoding="utf-8"
    )
)


def line_pcm(text: str) -> bytes:
    """The generator's scripted synthesis."""
    n = 800 * len(text)
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    t = np.arange(n) / 16000
    signal = 0.25 * np.sin(2 * np.pi * (300 + seed % 500) * t)
    return (signal * 32767).astype("<i2").tobytes()


class ScriptedContent:
    def __init__(self, replies: list[dict]) -> None:
        self._replies = iter(replies)
        self.calls: list[tuple[int, Pending]] = []

    @property
    def kind(self) -> SourceKind:
        return SourceKind.SCRIPT

    @property
    def is_deterministic(self) -> bool:
        return True

    async def respond(
        self, steps: Sequence[StepRecord], pending: Pending
    ) -> CallerReply:
        self.calls.append((len(steps), pending))
        reply = next(self._replies)
        if "tool" in reply:
            return UseTools(
                (CallerToolCall("u1", reply["tool"], {}),), text=reply["text"]
            )
        text = reply["text"]
        if "###STOP###" in text:
            return HangUp(text=text, reason="###STOP###")
        return Say(text)

    async def run_tools(
        self, calls: Sequence[CallerToolCall], *, step: int
    ) -> tuple[CallerToolResult, ...]:
        raise AssertionError("the test delivers tool results itself")


class ScriptedTTS:
    """The generator's scripted synthesis, as a TTS provider."""

    name = "scripted.tts"
    audio_format = SYNTHESIS
    selected: str | None = None

    def select_voice(self, selection: VoiceSelection) -> bool:
        self.selected = selection.voice_id
        return selection.voice_id is not None

    async def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        yield line_pcm(text)


def _voice() -> tuple[SimulatorVoice, ScriptedTTS]:
    """The real path: UniversalSimulator + L1 voice + tau2's phone line encoder."""
    tts = ScriptedTTS()
    simulator = caller_simulator(
        tts, voice_id="persona-voice", seed=626729, complexity="control"
    )
    return SimulatorVoice(simulator, step_s=0.2), tts


async def test_matches_tau2_step_for_step() -> None:
    content = ScriptedContent(GOLDEN["replies"])
    voice, tts = _voice()
    caller = ReactiveCaller(content=content, voice=voice, policy=PatientPolicy())
    pending_tool: str | None = None
    for step, ((text, speech), expected) in enumerate(
        zip(GOLDEN["agent"], GOLDEN["expected"], strict=True)
    ):
        results = None
        if pending_tool is not None:
            results = (CallerToolResult(pending_tool, GOLDEN["tool_result"]),)
            pending_tool = None
        chunk = await caller.next_chunk(Heard(text, speech), results)
        if chunk.tool_calls:
            pending_tool = chunk.tool_calls[0].id

        where = f"step {step}"
        assert (chunk.action.value if chunk.action else None) == expected["action"], (
            where
        )
        assert chunk.speech is expected["speech"], where
        assert (chunk.text or "") == (expected["text"] or ""), where
        assert [c.name for c in chunk.tool_calls] == expected["tool_calls"], where
        assert len(chunk.audio) == expected["audio_len"], where
        assert hashlib.sha256(chunk.audio).hexdigest() == expected["audio_sha256"], (
            where
        )
        assert (chunk.hang_up is not None) is expected["stop"], where

    assert len(caller.records) == len(GOLDEN["expected"])

    # It really went through the simulator: the L1 voice was selected, every
    # line is an utterance (the one the agent talked over is marked cut off),
    # and the phone line is in the plan record as the L4 encoder.
    simulator = voice.simulator
    assert tts.selected == "persona-voice"
    assert [u.truncated for u in simulator.utterances] == [False, False, True]
    layers = simulator.pipeline.as_record().as_dict()["layers"]
    assert "L4.tau2-telephony" in layers


def test_split_text_is_character_even() -> None:
    assert split_text("abcdef", 3) == ["ab", "cd", "ef"]
    assert split_text("ab", 4) == ["a", "", "b", ""]
    assert split_text("", 2) == ["", ""]
