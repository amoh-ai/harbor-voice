"""EVA's call-path audio conversions, case by case, run on EVA or on our port.

Run inside EVA's image to (re)generate the goldens::

    MSYS_NO_PATHCONV=1 docker run --rm \\
        -v "$PWD/adapters/eva/scripts/goldens:/goldens" \\
        --entrypoint /opt/venv/bin/python eva-base:039efa0a \\
        /goldens/call_audio_cases.py --eva \\
        > tests/adapters/eva/fixtures/call/eva_audio.json

``tests/adapters/eva/test_eva_audio.py`` runs the same cases on ``eva_adapter.audio`` and
compares. Inputs are built here with integer arithmetic only, so both sides
get the same bytes on any platform and numpy version; each input's SHA-256 is
in the golden too. Outputs are compared as SHA-256 of their bytes plus their
length: any one-sample difference fails.

**Except soxr's int16 output, which EVA itself does not reproduce.** libsoxr
applies TPDF dither when it writes int16 and seeds it per resampler instance,
so ``pcm16_24k_to_mulaw_8k`` gives different bytes from one EVA process to
the next (measured: 123 of 16 000 μ-law codes, each one quantisation step).
Those cases keep EVA's bytes (base64) and are compared as EVA compares with
itself: same length, every code within one μ-law step, a small share
differing. The resampler underneath has no dither, so ``soxr_core/*`` checks
it exactly (float64 in and out).

The cases cover what the call loop feeds these functions: 20 ms caller
chunks (each converted with fresh ``ratecv`` state, as EVA's bridge does),
the odd sizes its partial-chunk and shutdown paths produce, silence, and the
agent's 24 kHz deltas in sizes OpenAI sends and sizes it might.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from array import array
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


def _tri(phase: int, period: int, amp: int) -> int:
    """An integer triangle wave: 0 → amp → -amp → 0 over ``period``."""
    q = (4 * amp * (phase % period)) // period
    if q < amp:
        return q
    if q < 3 * amp:
        return 2 * amp - q
    return q - 4 * amp


def speech_like(n_samples: int, rate: int, seed: int = 1, level: int = 6000) -> bytes:
    """Deterministic speech-like PCM16-LE, integer arithmetic only.

    A gliding fundamental (≈110–220 Hz) with two harmonics, a syllable-rate
    envelope that never drops below 40 %, and a little LCG noise. Loud enough
    for any VAD, never silent for long enough to split an utterance.
    """
    out = array("h")
    state = seed & 0x7FFFFFFF
    step = max(1, rate // 50)  # the pitch moves every 20 ms
    syllable = max(1, rate // 4)  # ≈4 syllables a second
    for i in range(n_samples):
        f0 = 110 + ((i // step) * 7 + seed * 13) % 110
        period = max(2, rate // f0)
        env = 400 + (600 * abs(_tri(i, syllable, 1000))) // 1000  # 400..1000
        s = (
            _tri(i, period, level)
            + _tri(i, max(2, period // 2), level // 2)
            + _tri(i, max(2, period // 3), level // 4)
        )
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        noise = (state >> 16) % 401 - 200
        v = (s * env) // 1000 + noise
        out.append(max(-32768, min(32767, v)))
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def mulaw_like(n: int, seed: int = 3) -> bytes:
    """Deterministic μ-law bytes covering the whole code range."""
    state = seed
    out = bytearray()
    for _ in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        out.append((state >> 16) & 0xFF)
    return bytes(out)


def digest(data: bytes) -> dict[str, Any]:
    return {"len": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def modules(use_eva: bool) -> SimpleNamespace:
    """The functions under test, from EVA or from our port, under one set of names."""
    if use_eva:
        from eva.user_simulator import audio_bridge
        from eva.utils import audio_utils

        return SimpleNamespace(
            pcm16k_to_mulaw8k=audio_bridge.BotToBotAudioBridge._convert_pcm_to_mulaw,
            mulaw_8k_to_pcm16_24k=audio_utils.mulaw_8k_to_pcm16_24k,
            pcm16_24k_to_mulaw_8k=audio_utils.pcm16_24k_to_mulaw_8k,
            pcm16_mix=audio_utils.pcm16_mix,
            white_noise_pcm=audio_bridge._white_noise_pcm,
            chunk=audio_bridge.SEND_CHUNK_SIZE_PCM,
        )
    from eva_adapter import audio

    return SimpleNamespace(
        pcm16k_to_mulaw8k=audio.pcm16k_to_mulaw8k,
        mulaw_8k_to_pcm16_24k=audio.mulaw_8k_to_pcm16_24k,
        pcm16_24k_to_mulaw_8k=audio.pcm16_24k_to_mulaw_8k,
        pcm16_mix=audio.pcm16_mix,
        white_noise_pcm=audio.white_noise_pcm,
        chunk=audio.SEND_CHUNK_SIZE_PCM,
    )


def _run(fn: Callable[[bytes], bytes], data: bytes) -> dict[str, Any]:
    try:
        return digest(fn(data))
    except Exception:  # noqa: BLE001 - both sides must fail alike
        return {"error": True}


def _dithered(fn: Callable[[bytes], bytes], data: bytes) -> dict[str, Any]:
    """A soxr int16 output: its bytes, for a dither-tolerant comparison."""
    try:
        out = fn(data)
    except Exception:  # noqa: BLE001 - both sides must fail alike
        return {"error": True}
    return {"len": len(out), "b64": base64.b64encode(out).decode("ascii")}


def _soxr_core(pcm: bytes) -> list[float]:
    """soxr VHQ 24 → 8 kHz on float64: the resampler without int16 dither."""
    import numpy as np
    import soxr

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return [float(v) for v in soxr.resample(x, 24000, 8000, quality="VHQ")]


def run_cases(use_eva: bool) -> dict[str, Any]:
    m = modules(use_eva)
    cases: dict[str, Any] = {}

    caller = speech_like(16000 * 3 // 2, 16000, seed=1)  # 1.5 s caller speech
    cases["input/caller_16k"] = digest(caller)
    chunk = m.chunk
    cases["chunk_size_pcm"] = chunk

    # The bridge converts 640-byte chunks one at a time, state reset each time.
    wire = b"".join(
        m.pcm16k_to_mulaw8k(caller[i : i + chunk]) for i in range(0, len(caller), chunk)
    )
    cases["pcm16k_to_mulaw8k/chunked_640"] = digest(wire)
    for size in (0, 1, 2, 638, 640, 642, 1280, 7000, 6400):
        cases[f"pcm16k_to_mulaw8k/size_{size}"] = _run(
            m.pcm16k_to_mulaw8k, caller[:size]
        )
    cases["pcm16k_to_mulaw8k/silence_640"] = _run(m.pcm16k_to_mulaw8k, b"\x00" * chunk)
    padded = caller[:300] + b"\x00" * (chunk - 300)  # the partial-chunk pad
    cases["pcm16k_to_mulaw8k/padded_partial"] = _run(m.pcm16k_to_mulaw8k, padded)
    cases["pcm16k_to_mulaw8k/whole_utterance"] = _run(m.pcm16k_to_mulaw8k, caller)

    # Wire → agent, frame by frame as the server forwards it, and odd sizes.
    to_agent = b"".join(
        m.mulaw_8k_to_pcm16_24k(wire[i : i + 160]) for i in range(0, len(wire), 160)
    )
    cases["mulaw_8k_to_pcm16_24k/chunked_160"] = digest(to_agent)
    noise = mulaw_like(2000)
    cases["input/mulaw_like"] = digest(noise)
    for size in (1, 7, 159, 160, 161, 2000):
        cases[f"mulaw_8k_to_pcm16_24k/size_{size}"] = _run(
            m.mulaw_8k_to_pcm16_24k, noise[:size]
        )
    cases["mulaw_8k_to_pcm16_24k/silence_160"] = _run(
        m.mulaw_8k_to_pcm16_24k, b"\xff" * 160
    )

    # Agent → wire, delta by delta.
    agent = speech_like(24000 * 2, 24000, seed=5)  # 2 s of agent speech
    cases["input/agent_24k"] = digest(agent)
    for size in (2, 4, 6, 480, 960, 2398, 4800, 9600, 48000, 7202):
        cases[f"pcm16_24k_to_mulaw_8k/size_{size}"] = _dithered(
            m.pcm16_24k_to_mulaw_8k, agent[:size]
        )
    cases["pcm16_24k_to_mulaw_8k/deltas_4800"] = _dithered(
        lambda pcm: b"".join(
            m.pcm16_24k_to_mulaw_8k(pcm[i : i + 4800]) for i in range(0, len(pcm), 4800)
        ),
        agent,
    )
    cases["soxr_core/size_4800"] = _soxr_core(agent[:4800])

    # Recording helpers.
    cases["pcm16_mix/unequal"] = _run(
        lambda a: m.pcm16_mix(a, agent[: len(a) // 2 * 2 // 3 * 2]), agent[:30000]
    )
    loud = speech_like(4000, 24000, seed=9, level=30000)
    cases["pcm16_mix/clipping"] = _run(lambda a: m.pcm16_mix(a, a), loud)
    cases["white_noise_pcm/default"] = digest(m.white_noise_pcm())
    return cases


def main() -> None:
    use_eva = "--eva" in sys.argv[1:]
    json.dump(run_cases(use_eva), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
