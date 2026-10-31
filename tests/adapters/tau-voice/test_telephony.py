"""tau-voice phone line, bit-exact against tau2's StreamingTelephonyConverter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from harbor_voice.audio.format import AudioFormat
from harbor_voice.perturbation.base import SeedSet

from tau_voice.telephony import Tau2Telephony
from tau_voice.telephony import ulaw_to_pcm16

pytestmark = pytest.mark.unit

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "tau2_telephony.json").read_text(
        encoding="utf-8"
    )
)
PCM16_16K = AudioFormat(sample_rate_hz=16_000)


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


def _in_tau2_chunks() -> list[bytes]:
    pcm, position, out = _input(), 0, []
    encoder = Tau2Telephony()
    for expected in GOLDEN["chunks"]:
        chunk = pcm[position : position + expected["in_len"]]
        position += expected["in_len"]
        out.append(encoder.encode(chunk, PCM16_16K))
    return out


def test_matches_tau2_chunk_for_chunk() -> None:
    assert hashlib.sha256(_input()).hexdigest() == GOLDEN["input_sha256"]
    for out, expected in zip(_in_tau2_chunks(), GOLDEN["chunks"], strict=True):
        assert len(out) == expected["out_len"]
        assert hashlib.sha256(out).hexdigest() == expected["sha256"]


def test_20ms_frames_give_the_same_bytes_as_tau2s_chunks() -> None:
    """The streaming simulator feeds 20 ms frames; tau2 feeds 0.2 s chunks."""
    pcm = _input()
    encoder = Tau2Telephony()
    framed = b"".join(
        encoder.encode(pcm[i : i + 640], PCM16_16K) for i in range(0, len(pcm), 640)
    )
    assert framed == b"".join(_in_tau2_chunks())


def test_reset_starts_a_fresh_line() -> None:
    encoder = Tau2Telephony()
    first = encoder.encode(_input()[:6400], PCM16_16K)
    encoder.reset(SeedSet.derive(0))
    assert encoder.encode(_input()[:6400], PCM16_16K) == first


def test_output_is_real_g711() -> None:
    """Digital silence must encode to G.711 silence, not to a full-scale code."""
    out = Tau2Telephony().encode(bytes(3200), PCM16_16K)
    assert set(out) <= {0xFF, 0x7F}
    assert max(abs(s) for s in np.frombuffer(ulaw_to_pcm16(out), "<i2")) == 0
