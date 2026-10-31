"""The vendored tau2 audio code produces tau2's bytes, case for case.

Goldens come from tau2 itself, run inside the tau2 image
(``tau2_audio_cases.py --tau2``); this runs the same cases on
``tau_voice.tau2_audio``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from tau2_audio_cases import run_cases  # noqa: E402

GOLDENS = json.loads((HERE / "fixtures" / "tau2_audio_goldens.json").read_text())
OURS = run_cases(use_tau2=False)


@pytest.mark.parametrize("case", sorted(GOLDENS))
def test_vendored_effects_match_tau2(case: str) -> None:
    assert OURS[case] == GOLDENS[case]
