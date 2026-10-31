"""The same audio-effect cases, run against tau2 itself or against our vendored copy.

Run inside the tau2 image to (re)generate the goldens::

    docker run --rm -v "$PWD/tests/adapters/tau-voice:/cases" tau-voice-runtime:<tag> \\
        python /cases/tau2_audio_cases.py --tau2 > tests/adapters/tau-voice/fixtures/tau2_audio_goldens.json

``test_tau2_audio.py`` runs the same cases on ``tau_voice.tau2_audio`` and
compares. Inputs are generated here, deterministically (including the noise
and burst WAVs), so neither side needs tau2's data files. Outputs are compared
as SHA-256 of their bytes: any one-sample difference fails.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import sys
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TICK_16K = 3200  # 0.2 s at 16 kHz, tau2's user chunk


def modules(use_tau2: bool) -> SimpleNamespace:
    def load(tau2: str, vendored: str):  # noqa: ANN202
        return importlib.import_module(
            tau2 if use_tau2 else f"tau_voice.tau2_audio.{vendored}"
        )

    return SimpleNamespace(
        audio=load("tau2.data_model.audio", "audio"),
        fx=load("tau2.data_model.audio_effects", "audio_effects"),
        effects=load("tau2.voice.synthesis.audio_effects.effects", "effects"),
        noise=load(
            "tau2.voice.synthesis.audio_effects.noise_generator", "noise_generator"
        ),
        scheduler=load("tau2.voice.synthesis.audio_effects.scheduler", "scheduler"),
        text=load("tau2.voice.utils.text_effects", "text_effects"),
        pre=load("tau2.voice.utils.audio_preprocessing", "audio_preprocessing"),
    )


def speech(seconds: float, seed: int = 7, rate: int = 16000) -> bytes:
    """Deterministic speech-like PCM16: a few harmonics with a slow envelope."""
    t = np.arange(int(seconds * rate)) / rate
    rng = np.random.default_rng(seed)
    signal = sum(
        0.3 / (k + 1) * np.sin(2 * np.pi * f * t)
        for k, f in enumerate((180.0, 360.0, 720.0, 1450.0, 2900.0))
    )
    signal = signal * (
        0.6 + 0.4 * np.sin(2 * np.pi * 1.7 * t)
    ) + 0.02 * rng.standard_normal(len(t))
    return (np.clip(signal, -1, 1) * 20000).astype(np.int16).tobytes()


def write_wav(path: Path, pcm: bytes, rate: int) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pcm(m: SimpleNamespace, data: bytes, rate: int = 16000):  # noqa: ANN201
    fmt = m.audio.AudioFormat(
        encoding=m.audio.AudioEncoding.PCM_S16LE, sample_rate=rate, channels=1
    )
    return m.audio.AudioData(data=data, format=fmt)


def regular_configs(m: SimpleNamespace) -> SimpleNamespace:
    """tau2's REGULAR_CONFIG effect settings, as sample_voice_config merges them."""
    return SimpleNamespace(
        channel=m.fx.ChannelEffectsConfig(
            enable_frame_drops=True,
            frame_drop_rate=0.02,
            frame_drop_burst_duration_ms=100,
        ),
        source=m.fx.SourceEffectsConfig(
            enable_background_noise=True,
            enable_burst_noise=True,
            burst_noise_events_per_minute=1.0,
        ),
        speech=m.fx.SpeechEffectsConfig(
            enable_dynamic_muffling=True,
            enable_vocal_tics=True,
            enable_non_directed_phrases=True,
            speech_insert_events_per_minute=0.7,
        ),
    )


def run_cases(use_tau2: bool) -> dict[str, object]:
    m = modules(use_tau2)
    out: dict[str, object] = {}
    work = Path(tempfile.mkdtemp())
    noise_wav = write_wav(work / "bed.wav", speech(3.1, seed=11, rate=22050), 22050)
    burst_wav = write_wav(work / "burst.wav", speech(0.9, seed=12, rate=44100), 44100)
    utterance = pcm(m, speech(2.4))

    # Muffling, as regular applies it to an utterance (one 500 ms segment).
    cfg = regular_configs(m)
    muffled = m.effects.apply_dynamic_muffling(
        utterance,
        segment_count=cfg.speech.muffle_segment_count,
        segment_duration_ms=cfg.speech.muffle_segment_duration_ms,
        cutoff_freq=cfg.speech.muffle_cutoff_freq,
        transition_ms=cfg.speech.muffle_transition_ms,
    )
    out["muffling"] = digest(muffled.data)
    out["constant_muffling"] = digest(m.effects.apply_constant_muffling(utterance).data)

    # Background bed + a burst, chunk by chunk, as the streaming path mixes them.
    generator = m.noise.create_background_noise_generator(
        config=cfg.source,
        sample_rate=16000,
        background_noise_file=noise_wav,
        burst_noise_files=[burst_wav],
    )
    mixed = []
    for i in range(40):
        if i == 9:
            generator.add_burst(burst_wav)
        chunk = pcm(
            m, utterance.data[(i % 12) * TICK_16K * 2 : ((i % 12) + 1) * TICK_16K * 2]
        )
        if len(chunk.data) < TICK_16K * 2:
            chunk = pcm(m, chunk.data + b"\x00" * (TICK_16K * 2 - len(chunk.data)))
        mixed.append(m.noise.apply_background_noise(chunk, generator).data)
    out["noise_bed_with_burst"] = digest(b"".join(mixed))

    # The scheduler: which effects fire, over 10 minutes of 0.2 s chunks.
    scheduler = m.scheduler.EffectScheduler(
        seed=677354 + 300,
        source_config=cfg.source,
        speech_config=cfg.speech,
        channel_config=cfg.channel,
        burst_noise_files=[Path("a.wav"), Path("b.wav"), Path("c.wav")],
    )
    fired = []
    for i in range(3000):
        effects = scheduler.check_for_effects(
            chunk_duration_ms=200,
            is_silence=(i // 25) % 2 == 1,
            current_time_ms=i * 200,
            has_active_burst=False,
        )
        for e in effects:
            fired.append(
                [
                    i,
                    e.effect_type,
                    getattr(e, "burst_file", None) and Path(str(e.burst_file)).name,
                    getattr(e, "frame_drop_duration_ms", None),
                    getattr(getattr(e, "speech_item", None), "text", None),
                ]
            )
    out["scheduler"] = digest(json.dumps(fired).encode())
    out["scheduler_count"] = len(fired)

    # Per-turn effect draws (muffle / burst / frame drop / vocal tic).
    synthesis = SimpleNamespace(
        channel_effects_config=cfg.channel,
        source_effects_config=cfg.source,
        speech_effects_config=cfg.speech,
    )
    turns = []
    for turn in range(60):
        speech_fx, source_fx, channel_fx = m.scheduler.generate_turn_effects(
            seed=677654,
            turn_idx=turn,
            synthesis_config=synthesis,
            burst_noise_files=[Path("a.wav")],
        )
        turns.append(
            [
                turn,
                speech_fx.dynamic_muffling_enabled,
                speech_fx.speech_insert.text if speech_fx.speech_insert else None,
                source_fx.burst_noise_file and Path(source_fx.burst_noise_file).name,
                channel_fx.frame_drops_enabled,
            ]
        )
    out["turn_effects"] = digest(json.dumps(turns).encode())

    # Vocal-tic insertion into the utterance text.
    texts = []
    for turn in range(30):
        insert = cfg.speech.vocal_tics[turn % len(cfg.speech.vocal_tics)]
        texts.append(
            m.text.insert_speech_text(
                "I need to change the flight I booked for next Tuesday morning please",
                insert,
                rng=random.Random(677654 + turn),
                min_words=cfg.speech.min_words_for_vocal_tics,
                in_turn=True,
            )
        )
    out["vocal_tic_text"] = digest(json.dumps(texts).encode())

    # Frame drops on telephony bytes, and the phone line itself.
    ulaw = m.effects.convert_to_telephony(utterance)
    out["telephony"] = digest(ulaw.data)
    out["frame_drops"] = digest(
        m.effects.apply_frame_drops(ulaw, drop_count=1, drop_duration_ms=150).data
    )
    return out


if __name__ == "__main__":
    print(
        json.dumps(run_cases(use_tau2="--tau2" in sys.argv), indent=2, sort_keys=True)
    )
