"""G.711 μ-law: real wire bytes, not the simplified effect codec."""

from __future__ import annotations

import numpy as np
import pytest

from harbor_voice.audio import g711
from harbor_voice.perturbation import dsp

pytestmark = pytest.mark.unit


def test_silence_is_g711_silence() -> None:
    assert g711.encode(bytes(320)) == g711.SILENCE * 160


def test_round_trip_is_close() -> None:
    tone = (0.5 * np.sin(np.linspace(0, 40 * np.pi, 800)) * 32767).astype("<i2")
    back = np.frombuffer(g711.decode(g711.encode(tone.tobytes())), "<i2")
    assert np.max(np.abs(back.astype(int) - tone.astype(int))) < 1100


def test_the_effect_codec_is_not_wire_format() -> None:
    """The reason g711 exists: dsp's mu-law encodes silence as 0x00."""
    assert int(dsp.linear_to_mulaw(np.zeros(1))[0]) == 0x00
    assert g711.encode(bytes(2)) != bytes([0x00])
