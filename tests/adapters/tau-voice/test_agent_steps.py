"""AgentStepper against outputs recorded from tau2's own adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tau_voice.agent_steps import AgentStepper
from tau_voice.agent_steps import Truncation
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript

pytestmark = pytest.mark.unit

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "tau2_agent_steps.json").read_text(
        encoding="utf-8"
    )
)


def _event(raw: dict):
    if raw["kind"] == "audio":
        return OutputAudio(bytes.fromhex(raw["pcm_hex"]), raw["utterance"], 0.0)
    if raw["kind"] == "transcript":
        return OutputTranscript(raw["utterance"], raw["delta"], 0.0)
    return InputSpeech(raw["started"], raw["audio_ms"], 0.0)


@pytest.mark.parametrize("name", sorted(GOLDEN["scenarios"]))
def test_matches_tau2(name: str) -> None:
    scenario = GOLDEN["scenarios"][name]
    stepper = AgentStepper(
        bytes_per_step=GOLDEN["bytes_per_step"],
        bytes_per_second=GOLDEN["bytes_per_second"],
        silence=bytes.fromhex(GOLDEN["silence_byte"]),
    )
    for events, expected in zip(scenario["steps"], scenario["expected"], strict=True):
        stepper.begin()
        truncations = [t for t in (stepper.feed(_event(e)) for e in events) if t]
        step = stepper.end(user_bytes=GOLDEN["bytes_per_step"])

        assert hashlib.sha256(step.played).hexdigest() == expected["played_sha256"]
        assert len(step.played) == expected["played_len"]
        assert step.raw_bytes == expected["raw_len"]
        assert step.contains_speech is expected["contains_speech"]
        assert step.transcript == expected["transcript"]
        assert step.was_truncated is expected["was_truncated"]
        assert step.truncated_bytes == expected["truncated_audio_bytes"]
        assert list(step.utterances) == expected["utterances"]
        assert step.raw_data() == expected["raw_data"]
        assert [
            {"utterance": t.utterance, "audio_end_ms": t.audio_end_ms}
            for t in truncations
        ] == expected["truncations"]


def test_feed_outside_a_step_is_an_error() -> None:
    stepper = AgentStepper(bytes_per_step=1600, bytes_per_second=8000)
    with pytest.raises(RuntimeError, match="begin"):
        stepper.feed(OutputTranscript("a", "x", 0.0))


def test_the_truncation_point_is_always_one_full_step() -> None:
    """tau2 reports the padded step length — kept deliberately (see module doc)."""
    stepper = AgentStepper(bytes_per_step=1600, bytes_per_second=8000)
    stepper.begin()
    stepper.feed(OutputAudio(b"\x01" * 100, "a", 0.0))
    assert stepper.feed(InputSpeech(True, 10, 0.0)) == Truncation("a", 200)


LIVE_GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "tau2_live_steps.json").read_text(
        encoding="utf-8"
    )
)


def test_the_live_stepper_matches_tau2s_live_adapter() -> None:
    """tau2's DiscreteTimeOpenAILiveAdapter: converted audio both ways, the
    transcript released by audio position, speech by amplitude, no truncation."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from tau2_live_steps_cases import run_ours

    ours = run_ours()
    for n, (got, want) in enumerate(zip(ours, LIVE_GOLDEN["steps"], strict=True)):
        assert got == want, f"step {n}"
