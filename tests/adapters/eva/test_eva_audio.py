"""``eva_adapter.audio`` produces EVA's bytes, case for case.

Goldens come from EVA's own functions, run in ``eva-base:039efa0a`` by
``adapters/eva/scripts/goldens/call_audio_cases.py --eva``; this runs the same cases on
our port. soxr's int16 output is dithered, so EVA does not reproduce it run
to run: those cases are compared the way EVA compares with itself (see the
cases module).
"""

from __future__ import annotations

import audioop
import base64
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
ADAPTER = HERE.parents[2] / "adapters" / "eva"
GOLDENS_DIR = ADAPTER / "scripts" / "goldens"
sys.path.insert(0, str(GOLDENS_DIR))

from call_audio_cases import run_cases  # noqa: E402

GOLDENS = json.loads((HERE / "fixtures" / "call" / "eva_audio.json").read_text())
OURS = run_cases(use_eva=False)

DITHERED = sorted(k for k, v in GOLDENS.items() if isinstance(v, dict) and "b64" in v)
EXACT = sorted(
    k for k in GOLDENS if k not in DITHERED and not k.startswith("soxr_core")
)

MAX_DITHER_SHARE = 0.03
"""EVA against itself differs on 0.8 % of codes (123 of 16 000); allow 3 %."""


def _ulaw_rank() -> dict[int, int]:
    """Each μ-law code's position on the decoded-value scale."""
    values = {
        code: int.from_bytes(audioop.ulaw2lin(bytes([code]), 2), "little", signed=True)
        for code in range(256)
    }
    scale = sorted(set(values.values()))
    return {code: scale.index(value) for code, value in values.items()}


RANK = _ulaw_rank()


@pytest.mark.parametrize("case", EXACT)
def test_conversion_matches_eva_byte_for_byte(case: str) -> None:
    assert OURS[case] == GOLDENS[case]


@pytest.mark.parametrize("case", DITHERED)
def test_soxr_output_matches_eva_up_to_its_dither(case: str) -> None:
    ours, theirs = OURS[case], GOLDENS[case]
    assert ours["len"] == theirs["len"]
    a = base64.b64decode(ours["b64"])
    b = base64.b64decode(theirs["b64"])
    steps = [abs(RANK[x] - RANK[y]) for x, y in zip(a, b, strict=True)]
    assert max(steps, default=0) <= 1
    assert sum(1 for s in steps if s) <= MAX_DITHER_SHARE * max(len(steps), 1) + 1


def test_soxr_resampler_core_matches_eva() -> None:
    ours, theirs = OURS["soxr_core/size_4800"], GOLDENS["soxr_core/size_4800"]
    assert len(ours) == len(theirs)
    assert max(abs(x - y) for x, y in zip(ours, theirs, strict=True)) < 1e-6


def test_inputs_are_the_same_on_both_sides() -> None:
    for key in ("input/caller_16k", "input/agent_24k", "input/mulaw_like"):
        assert OURS[key] == GOLDENS[key]
