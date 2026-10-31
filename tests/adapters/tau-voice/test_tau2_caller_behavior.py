"""ReactivePolicy + backchannels against tau2's own caller, step for step.

The golden is tau2's ``VoiceStreamingUserSimulator`` (``control_behavior``)
on a scripted call (``tau2_caller_behavior_cases.py --tau2``): decisions are
answered from fixed lists, so both sides must ask at the same steps, act the
same way on each answer, and send the same audio.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from tau2_caller_behavior_cases import run_ours  # noqa: E402

pytestmark = pytest.mark.unit

GOLDEN = json.loads(
    (HERE / "fixtures" / "tau2_caller_behavior.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
async def ours() -> dict:
    return await run_ours(GOLDEN)


async def test_matches_tau2_step_for_step(ours: dict) -> None:
    expected = GOLDEN["steps"]
    for step, (got, want) in enumerate(zip(ours["steps"], expected)):
        assert got == want, f"step {step}"
    assert len(ours["steps"]) == len(expected)


async def test_asks_the_listener_when_tau2_does(ours: dict) -> None:
    tau2_steps = sorted({d["step"] for d in GOLDEN["decisions"]})
    assert [r["step"] for r in ours["requests"]] == tau2_steps
    assert ours["checks"]["interrupt"] == 1
    assert ours["checks"]["backchannel"] == 1


def test_the_case_backchannels_and_interrupts() -> None:
    actions = [s["action"] for s in GOLDEN["steps"]]
    assert "backchannel" in actions
    # The interruption: a line generated while the agent is still speaking.
    texts = [s["text"] or "" for s in GOLDEN["steps"]]
    assert any(t.startswith("Sorr") for t in texts)
