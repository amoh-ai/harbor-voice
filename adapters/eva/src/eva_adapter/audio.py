"""EVA's audio conversions on the call path, byte for byte.

Every function here is a port of one in EVA @039efa0a, with the same
resamplers and the same quirks, because the bytes they produce are what the
agent under test hears and what EVA's judges listen to:

- :func:`pcm16k_to_mulaw8k`: the caller's 16 kHz PCM to the wire's 8 kHz
  μ-law (``BotToBotAudioBridge._convert_pcm_to_mulaw``,
  ``user_simulator/audio_bridge.py:332-358``). ``audioop.ratecv`` with no
  anti-alias filter and no state carried: EVA passes ``None`` as the state
  for every 20 ms chunk, so each chunk is resampled as if the stream started
  there. Ported as is.
- :func:`mulaw_8k_to_pcm16_24k`: the wire to OpenAI Realtime's 24 kHz PCM
  (``utils/audio_utils.py:31-44``), ``ratecv`` again, stateless, clamped to
  exactly 3x the input length.
- :func:`pcm16_24k_to_mulaw_8k`: the agent's 24 kHz PCM back to the wire
  (``utils/audio_utils.py:64-82``), soxr VHQ, clamped to round(n / 3).
- :func:`pcm16_mix`, :func:`sync_buffer_to_position`, :func:`save_audio_track`:
  the recording helpers (``utils/audio_utils.py:119-237``).

``tests/test_eva_audio.py`` checks each against EVA's own functions, run in
``eva-base:039efa0a`` by ``scripts/goldens/call_audio_cases.py``.
"""

from __future__ import annotations

import audioop
import functools
import wave
from pathlib import Path

import numpy as np
import soxr

PCM_SAMPLE_WIDTH = 2
"""16-bit PCM (``audio_bridge.py:35``)."""

ASSISTANT_SAMPLE_RATE = 8_000
"""The wire between caller and assistant: 8 kHz μ-law (``audio_bridge.py:36``)."""

ELEVENLABS_OUTPUT_RATE = 16_000
"""The caller's speech: ElevenLabs agents output 16 kHz PCM (``audio_bridge.py:37``)."""

OPENAI_SAMPLE_RATE = 24_000
"""OpenAI Realtime's PCM, and the assistant-side recording rate
(``openai_realtime_server.py:34``, ``base_server.py:30``)."""

SEND_CHUNK_DURATION_MS = 20
SEND_CHUNK_SIZE_PCM = (
    int(ELEVENLABS_OUTPUT_RATE * SEND_CHUNK_DURATION_MS / 1000) * PCM_SAMPLE_WIDTH
)
"""640 bytes: one 20 ms chunk of caller PCM (``audio_bridge.py:41-42``)."""

MULAW_CHUNK_SIZE = 160
"""One 20 ms μ-law chunk at 8 kHz (``openai_realtime_server.py:38``)."""

MULAW_SILENCE = b"\xff"
"""μ-law silence, as the bridge pads with it (``audio_bridge.py:655-659``)."""

USER_TRAILING_NOISE_SNR_DB = 45.0
_FULL_SCALE_RMS = 32768.0
"""``audio_bridge.py:65-66``; used only by the (disabled) trailing-noise block."""


def pcm16k_to_mulaw8k(pcm_data: bytes) -> bytes:
    """The bridge's caller PCM → wire μ-law (``audio_bridge.py:332-358``).

    ``ratecv`` 16 kHz → 8 kHz with ``state=None`` on every call, then G.711
    μ-law. Any error (an odd byte count, say) gives ``b""``, as EVA's
    ``except Exception`` does, which its senders then treat as nothing to send.
    """
    try:
        pcm_8khz, _ = audioop.ratecv(
            pcm_data,
            PCM_SAMPLE_WIDTH,
            1,
            ELEVENLABS_OUTPUT_RATE,
            ASSISTANT_SAMPLE_RATE,
            None,
        )
        return audioop.lin2ulaw(pcm_8khz, PCM_SAMPLE_WIDTH)
    except Exception:  # noqa: BLE001 - EVA swallows every conversion error
        return b""


def mulaw_8k_to_pcm16_24k(mulaw_bytes: bytes) -> bytes:
    """8 kHz μ-law → 24 kHz PCM16 (``audio_utils.py:31-44``)."""
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    pcm_24k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)
    # ratecv can produce ±2 samples; clamp to exactly 3x the input length.
    expected_bytes = len(pcm_8k) * 3
    if len(pcm_24k) < expected_bytes:
        pcm_24k = pcm_24k + b"\x00" * (expected_bytes - len(pcm_24k))
    elif len(pcm_24k) > expected_bytes:
        pcm_24k = pcm_24k[:expected_bytes]
    return pcm_24k


def pcm16_24k_to_mulaw_8k(pcm_bytes: bytes) -> bytes:
    """24 kHz PCM16 → 8 kHz μ-law, soxr VHQ (``audio_utils.py:64-82``)."""
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
    resampled = soxr.resample(audio_data, 24000, 8000, quality="VHQ")
    # round(), so 2399 input samples give 800, not 799 (EVA's comment).
    expected_samples = round(len(audio_data) * 8000 / 24000)
    if len(resampled) < expected_samples:
        resampled = np.pad(resampled, (0, expected_samples - len(resampled)))
    elif len(resampled) > expected_samples:
        resampled = resampled[:expected_samples]
    pcm_8k = resampled.astype(np.int16).tobytes()
    return audioop.lin2ulaw(pcm_8k, 2)


def sync_buffer_to_position(buffer: bytearray, target_position: int) -> None:
    """Pad ``buffer`` with zero bytes up to ``target_position`` (``audio_utils.py:119-128``)."""
    current_len = len(buffer)
    if current_len < target_position:
        buffer.extend(b"\x00" * (target_position - current_len))


def pcm16_mix(track_a: bytes, track_b: bytes) -> bytes:
    """Sample-wise sum with clipping, the shorter track zero-padded (``audio_utils.py:131-152``).

    EVA does this with ``struct`` over a generator; numpy gives the same
    bytes (int16 + int16 cannot overflow int32, and the clip is the same)
    in a fraction of the time on a ten-minute call.
    """
    max_len = max(len(track_a), len(track_b))
    track_a = track_a + b"\x00" * (max_len - len(track_a))
    track_b = track_b + b"\x00" * (max_len - len(track_b))
    n_bytes = (max_len // 2) * 2
    a = np.frombuffer(track_a[:n_bytes], dtype="<i2").astype(np.int32)
    b = np.frombuffer(track_b[:n_bytes], dtype="<i2").astype(np.int32)
    return np.clip(a + b, -32768, 32767).astype("<i2").tobytes()


def save_pcm_as_wav(
    audio_data: bytes,
    file_path: Path,
    sample_rate: int,
    num_channels: int,
    sample_width: int = 2,
) -> None:
    """Raw PCM to a WAV file (``audio_utils.py:203-219``)."""
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)


def save_audio_track(
    audio_bytes: bytes,
    file_path: Path,
    sample_rate: int,
    num_channels: int = 1,
) -> bool:
    """One track to WAV, skipping empty audio (``audio_utils.py:222-237``)."""
    if not audio_bytes:
        return False
    save_pcm_as_wav(audio_bytes, file_path, sample_rate, num_channels)
    return True


@functools.cache
def white_noise_pcm(
    chunk_size: int = SEND_CHUNK_SIZE_PCM,
    snr_db: float = USER_TRAILING_NOISE_SNR_DB,
) -> bytes:
    """EVA's cached trailing white-noise chunk (``audio_bridge.py:79-95``).

    Only reachable when ``USER_TRAILING_SILENCE_CHUNKS > 0``, which EVA
    @039efa0a has at 0; ported so the branch that uses it is complete.
    """
    n_samples = chunk_size // PCM_SAMPLE_WIDTH
    target_rms = _FULL_SCALE_RMS * (10 ** (-snr_db / 20.0))
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, target_rms, size=n_samples)
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
