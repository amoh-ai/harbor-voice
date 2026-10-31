"""One scripted call through tau2's caller effects, run by tau2 or by us.

Run inside the sidecar image, with the current ``server.py``, to (re)generate
the golden::

    docker run --rm -v "$PWD/tests/adapters/tau-voice:/cases" \\
        -v "$PWD/adapters/tau-voice/sidecar/server.py:/app/server.py" \\
        tau-voice-runtime:<tag> python /cases/tau2_caller_effects_cases.py --tau2 \\
        > tests/adapters/tau-voice/fixtures/tau2_caller_effects.json

The tau2 side is tau2's real caller (``VoiceStreamingUserSimulator``, built by
the sidecar for a ``regular`` task): its ``synthesize_voice`` per line, its
chunking, its ``_apply_chunk_effects`` per step. Ours is
:class:`~tau_voice.voice.SimulatorVoice` with
:class:`~tau_voice.effects.Tau2CallerEffects`, built from the voice settings
the tau2 side recorded. Both get the same script: lines (some cut off part
way, as a yield does) and silent steps between.

Synthesis is scripted, and the noise files are generated here, so the case
needs neither TTS nor tau2's data. Event rates are raised over ``regular``'s
so that a short call exercises every effect.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

STEP_SAMPLES = 3200
TASK = ("airline", "18")
SEED = 300
BED = "fx_bed.wav"
BURSTS = ["fx_horn.wav", "fx_door.wav", "fx_dog.wav"]
RATES = {
    "source_effects_config": {"burst_noise_events_per_minute": 6.0},
    "speech_effects_config": {"speech_insert_events_per_minute": 6.0},
    "channel_effects_config": {"frame_drop_rate": 0.05},
}
WORDS = (
    "I need to change my flight to next Tuesday morning please and also check "
    "whether the reservation includes a checked bag because my daughter is "
    "traveling with me and we have two suitcases"
).split()


def line_pcm(text: str) -> bytes:
    """Scripted synthesis: 40 ms per character of speech-like PCM16 at 16 kHz."""
    n = 640 * len(text)
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    t = np.arange(n) / 16000
    f0 = 140 + seed % 120
    signal = sum(
        0.25 / (k + 1) * np.sin(2 * np.pi * f0 * (k + 1) * t) for k in range(5)
    )
    signal = signal * (0.6 + 0.4 * np.sin(2 * np.pi * 2.3 * t))
    signal = signal + 0.01 * np.random.default_rng(seed).standard_normal(n)
    return (np.clip(signal, -1, 1) * 20000).astype("<i2").tobytes()


def noise_wav(path: Path, seconds: float, rate: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    signal = 0.2 * rng.standard_normal(len(t)) + 0.1 * np.sin(2 * np.pi * 90 * t)
    pcm = (np.clip(signal, -1, 1) * 20000).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return path


def write_noise(continuous: Path, bursts: Path) -> dict[str, Path]:
    paths = {BED: noise_wav(continuous / BED, 4.3, 22050, 1)}
    for index, name in enumerate(BURSTS):
        paths[name] = noise_wav(bursts / name, 0.7 + 0.4 * index, 44100, 2 + index)
    return paths


def script() -> list[tuple[str, Any]]:
    """``("line", (text, steps_spoken or None))`` and ``("silence", n)``."""
    rng = random.Random(5)
    events: list[tuple[str, Any]] = [("silence", 3)]
    for index in range(24):
        start = rng.randrange(len(WORDS) - 6)
        text = " ".join(WORDS[start : start + rng.randrange(3, 14)])
        cut = rng.randrange(2, 6) if index % 5 == 3 else None
        events.append(("line", (text, cut)))
        events.append(("silence", rng.randrange(4, 40)))
    return events


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_effects(chunk: Any) -> dict[str, Any]:
    """tau2's effect results on a chunk, as JSON; a burst by file name (the
    sidecar puts tau2's data folder back)."""

    def dump(value: Any) -> Any:
        return value.model_dump(mode="json") if value is not None else None

    source = dump(chunk.source_effects)
    if source and source["burst_noise_file"]:
        source["burst_noise_file"] = Path(source["burst_noise_file"]).name
    return {
        "speech_effects": dump(chunk.speech_effects),
        "source_effects": source,
        "channel_effects": dump(chunk.channel_effects),
    }


# ------------------------------------------------------------------- tau2


def run_tau2() -> dict[str, Any]:
    sys.path.insert(0, "/app")  # the sidecar's server.py
    import server
    import tau2.agent.base.voice as voice_mixin
    import tau2.voice.synthesis.audio_effects.speech_generator as speech_generator
    from tau2.data_model.audio import AudioData
    from tau2.data_model.audio import AudioEncoding
    from tau2.data_model.audio import AudioFormat
    from tau2.data_model.message import UserMessage
    from tau2.voice_config import BACKGROUND_NOISE_CONTINUOUS_DIR
    from tau2.voice_config import BURST_NOISE_DIR

    def synthesize(*, text: str, provider: str, provider_config: Any) -> AudioData:
        fmt = AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=16000, channels=1
        )
        return AudioData(data=line_pcm(text), format=fmt)

    voice_mixin.synthesize_voice = synthesize
    speech_generator.synthesize_voice = synthesize
    write_noise(BACKGROUND_NOISE_CONTINUOUS_DIR, BURST_NOISE_DIR)

    domain, task_id = TASK
    session = server._Session(
        server.OpenRequest(
            domain=domain, task_id=task_id, speech_complexity="regular", seed=SEED
        )
    )
    user = session.user
    env = user.voice_settings.speech_environment
    env.background_noise_file = BED
    env.burst_noise_files = list(BURSTS)
    synthesis = user.voice_settings.synthesis_config
    for config, values in RATES.items():
        for key, value in values.items():
            setattr(getattr(synthesis, config), key, value)
    described = session.describe()["user"]
    state = user.get_init_state()

    wire, texts, lines, fx = [], [], [], []
    for kind, arg in script():
        if kind == "silence":
            for _ in range(arg):
                out = user._apply_chunk_effects(None, state, is_speech=False)
                wire.append(out.get_audio_bytes())
                fx.append(chunk_effects(out))
            continue
        text, cut = arg
        message = user.synthesize_voice(
            UserMessage(role="user", content=text),
            state,
            effects_turn_idx=state.user_utterance_count,
            add_background_noise=False,
            add_burst_noise=False,
            add_telephony_format=False,
            add_channel_effects=False,
        )
        state.user_utterance_count += 1
        lines.append(
            [message.audio_script_gold, message.speech_effects.dynamic_muffling_enabled]
        )
        chunks = user._create_chunk_messages(message)
        for chunk in chunks[:cut]:
            texts.append(chunk.content)
            out = user._apply_chunk_effects(chunk, state, is_speech=True)
            wire.append(out.get_audio_bytes())
            fx.append(chunk_effects(out))
    user._effect_timeline.close_all_open(int(state.elapsed_samples * 1000 / 16000))
    return {
        "voice_settings": described["voice_settings"],
        "tau2_burst_files": described["tau2_burst_files"],
        "result": result(
            wire, texts, lines, user.get_effect_timeline().model_dump(mode="json"), fx
        ),
    }


# -------------------------------------------------------------------- ours


async def run_ours(golden: dict[str, Any], noise_dir: Path) -> dict[str, Any]:
    from collections.abc import AsyncIterator

    from harbor_voice.ids import PartyId
    from harbor_voice.perturbation.persona import VoiceSelection
    from tau_voice.caller import _pad
    from tau_voice.caller import split_text
    from tau_voice.effects import Tau2CallerEffects
    from tau_voice.run import SYNTHESIS
    from tau_voice.run import caller_simulator
    from tau_voice.voice import SimulatorVoice

    class ScriptedTTS:
        name = "scripted.tts"
        audio_format = SYNTHESIS

        def select_voice(self, selection: VoiceSelection) -> bool:
            return selection.voice_id is not None

        async def synthesize(
            self, text: str, *, party: PartyId
        ) -> AsyncIterator[bytes]:
            yield line_pcm(text)

    (noise_dir / "c").mkdir(parents=True, exist_ok=True)
    (noise_dir / "b").mkdir(parents=True, exist_ok=True)
    effects = Tau2CallerEffects(
        golden["voice_settings"],
        turn_burst_files=golden["tau2_burst_files"],
        assets=write_noise(noise_dir / "c", noise_dir / "b"),
    )
    simulator = caller_simulator(
        ScriptedTTS(), voice_id="voice", seed=SEED, complexity="regular"
    )

    async def synthesize(text: str) -> bytes:
        return (await simulator.prepare(text)).pcm

    await effects.open(synthesize, voice_id="voice")
    voice = SimulatorVoice(simulator, step_s=0.2, effects=effects)

    wire, texts, lines, fx = [], [], [], []
    for kind, arg in script():
        if kind == "silence":
            for _ in range(arg):
                wire.append(voice.emit_step(bytes(voice.step_bytes), is_speech=False))
                fx.append(voice.last_effects)
            continue
        text, cut = arg
        line = await voice.prepare(text)
        lines.append([line.said, effects.lines[-1].muffled])
        size = voice.step_bytes
        steps = [
            _pad(line.pcm[i : i + size], size) for i in range(0, len(line.pcm), size)
        ]
        pieces = split_text(line.said, len(steps))
        spoken = steps[:cut]
        for index, step in enumerate(spoken):
            texts.append(pieces[index])
            wire.append(voice.emit_step(step, is_speech=True, line=line))
            fx.append(voice.last_effects)
        voice.finish(line, truncated=len(spoken) < len(steps))
    return result(wire, texts, lines, effects.close(), fx)


def result(
    wire: list[bytes],
    texts: list[str],
    lines: list[list[Any]],
    timeline: dict[str, Any],
    fx: list[dict[str, Any] | None],
) -> dict[str, Any]:
    events: dict[str, int] = {}
    for event in timeline["events"]:
        events[event["effect_type"]] = events.get(event["effect_type"], 0) + 1
    carried = {
        kind: sum(1 for f in fx if f and f[kind])
        for kind in ("speech_effects", "source_effects", "channel_effects")
    }
    return {
        "step_effects": digest(json.dumps(fx, sort_keys=True).encode()),
        "step_effects_carried": carried,
        "steps": len(wire),
        "wire": digest(b"".join(wire)),
        "step_digests": [digest(w)[:12] for w in wire],
        "texts": digest(json.dumps(texts).encode()),
        "lines": lines,
        "timeline": timeline,
        "event_counts": events,
    }


if __name__ == "__main__":
    if "--tau2" not in sys.argv:
        raise SystemExit("run with --tau2 inside the sidecar image")
    print(json.dumps(run_tau2(), indent=1, sort_keys=True))
