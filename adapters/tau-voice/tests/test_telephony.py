"""tau-voice phone line, bit-exact against tau2's StreamingTelephonyConverter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tau_voice.telephony import TelephonyConverter
from tau_voice.telephony import ulaw_to_pcm16

pytestmark = pytest.mark.unit

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "tau2_telephony.json").read_text(
        encoding="utf-8"
    )
)


def _input() -> bytes:
    """The generator's signal, rebuilt; its hash is checked before use."""
    rng = np.random.default_rng(GOLDEN["seed"])
    t = np.arange(16000 * 2) / 16000
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 2500 * t)
        + 0.05 * rng.standard_normal(t.size)
    )
    return (np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes()


def test_matches_tau2_chunk_for_chunk() -> None:
    pcm = _input()
    assert hashlib.sha256(pcm).hexdigest() == GOLDEN["input_sha256"]
    converter = TelephonyConverter(input_rate_hz=16_000)
    position = 0
    for expected in GOLDEN["chunks"]:
        chunk = pcm[position : position + expected["in_len"]]
        position += expected["in_len"]
        out = converter.convert(chunk)
        assert len(out) == expected["out_len"]
        assert hashlib.sha256(out).hexdigest() == expected["sha256"]


def test_output_is_real_g711() -> None:
    """Digital silence must encode to G.711 silence, not to a full-scale code."""
    out = TelephonyConverter().convert(bytes(3200))
    assert set(out) <= {0xFF, 0x7F}
    assert max(abs(s) for s in np.frombuffer(ulaw_to_pcm16(out), "<i2")) == 0
