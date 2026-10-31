"""tau2's caller effects, step for step: the *how it sounds* beyond control.

tau2 degrades its caller in two places, and this runs both with tau2's own
code (vendored verbatim in :mod:`tau_voice.tau2_audio`):

- per line (``VoiceMixin.synthesize_voice``): draw the turn's effects,
  insert a vocal tic into the text, synthesise, muffle;
- per 0.2 s step (``VoiceStreamingUserSimulator._apply_chunk_effects``):
  the scheduler fires bursts, out-of-turn speech and frame drops; the noise
  bed, burst and out-of-turn tracks are mixed under the step; the phone line
  encodes it; frame drops zero the start of the encoded step.

The step is the unit because tau2's effects are defined per step: the SNR
envelope spans one step, and the scheduler draws once per step. Run per
20 ms frame they would sound different and fire at different rates.

Everything here is tau2's, line for line, except the glue: tau2's state
object and message types become fields and bytes, and the out-of-turn sounds
are synthesised up front through harbor-voice's TTS (and cache).
"""

from __future__ import annotations

import random
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tau_voice.tau2_audio._shim import SynthesisConfig
from tau_voice.tau2_audio._shim import synthesizer
from tau_voice.tau2_audio.audio import AudioData
from tau_voice.tau2_audio.audio import AudioEncoding
from tau_voice.tau2_audio.audio import AudioFormat
from tau_voice.tau2_audio.audio_effects import ChannelEffectsConfig
from tau_voice.tau2_audio.audio_effects import ChannelEffectsResult
from tau_voice.tau2_audio.audio_effects import EffectTimeline
from tau_voice.tau2_audio.audio_effects import SourceEffectsConfig
from tau_voice.tau2_audio.audio_effects import SourceEffectsResult
from tau_voice.tau2_audio.audio_effects import SpeechEffectsConfig
from tau_voice.tau2_audio.audio_effects import SpeechEffectsResult
from tau_voice.tau2_audio.noise_generator import BackgroundNoiseGenerator
from tau_voice.tau2_audio.processor import BatchAudioEffectsMixin
from tau_voice.tau2_audio.processor import PendingEffectState
from tau_voice.tau2_audio.processor import StreamingAudioEffectsMixin
from tau_voice.tau2_audio.scheduler import EffectScheduler
from tau_voice.tau2_audio.scheduler import generate_turn_effects
from tau_voice.tau2_audio.speech_generator import OutOfTurnSpeechGenerator
from tau_voice.tau2_audio.speech_generator import create_streaming_audio_generators
from tau_voice.tau2_audio.text_effects import insert_speech_text

SAMPLE_RATE = 16_000
"""tau2's ``PCM_SAMPLE_RATE``: the caller is synthesised and mixed at 16 kHz."""
TELEPHONY_RATE = 8_000

Synthesize = Callable[[str], Awaitable[bytes]]
"""Text to 16 kHz PCM16 — harbor-voice's TTS, in the persona's voice."""


@dataclass(frozen=True, slots=True)
class LineEffects:
    """tau2's per-turn draw for one line."""

    turn: int
    said: str
    """The line as synthesised: the text, with any vocal tic inserted."""
    muffled: bool
    tic: str | None
    speech_effects: dict[str, Any]
    """tau2's ``SpeechEffectsResult``, as its chunks of the line carry it."""
    channel_effects: dict[str, Any]
    """tau2's ``ChannelEffectsResult`` for the turn (before any frame drop)."""


@dataclass(frozen=True, slots=True)
class StepMix:
    """One step, mixed, before the phone line."""

    pcm: bytes
    frame_drops_ms: tuple[int, ...]
    """Frame drops to apply to the step after the phone line, in order."""
    source_effects: dict[str, Any] | None = None
    """tau2's ``SourceEffectsResult`` for the step: a burst or an out-of-turn
    sound that started in it. A burst is named by file name only; tau2 keeps
    the path in its data folder, which the sidecar restores."""


class _Processor(BatchAudioEffectsMixin, StreamingAudioEffectsMixin):
    """tau2's two effect mixins, without the simulator they are mixed into."""


class Tau2CallerEffects:
    """tau2's caller effects for one call. See the module docstring.

    Built from the sidecar's ``user.voice_settings`` (tau2's ``VoiceSettings``
    as JSON). ``turn_burst_files`` is tau2's own burst list
    (``BURST_NOISE_FILES``), in its order — per-turn draws pick from it, which
    consumes randomness by its length, and the scheduler falls back to it when
    a task names none. ``assets`` maps tau2's noise file names to local copies.
    """

    def __init__(
        self,
        voice_settings: Mapping[str, Any],
        *,
        turn_burst_files: Sequence[str],
        assets: Mapping[str, Path] | None = None,
        step_s: float = 0.2,
    ) -> None:
        env = voice_settings["speech_environment"]
        self.step_samples = int(SAMPLE_RATE * step_s)
        synthesis = voice_settings["synthesis_config"]
        if not env.get("telephony_enabled", True):
            raise NotImplementedError("tau2 effects without the phone line")
        self.voice_seed: int = env["voice_seed"]
        self.persona_name: str = env["persona_name"]
        self.assets: dict[str, Path] = dict(assets or {})
        """Local copies of tau2's noise files, by name; see :attr:`needed_assets`."""
        self.background_noise_file: str | None = env.get("background_noise_file")
        self.burst_noise_files: list[str] = list(env.get("burst_noise_files") or [])
        self.turn_burst_files = list(turn_burst_files)
        self.config = SynthesisConfig(
            provider=synthesis.get("provider", "elevenlabs"),
            provider_config=SimpleNamespace(**(synthesis.get("provider_config") or {})),
            channel_effects_config=ChannelEffectsConfig.model_validate(
                synthesis["channel_effects_config"]
            ),
            source_effects_config=SourceEffectsConfig.model_validate(
                synthesis["source_effects_config"]
            ),
            speech_effects_config=SpeechEffectsConfig.model_validate(
                synthesis["speech_effects_config"]
            ),
        )
        self.timeline = EffectTimeline()
        self.lines: list[LineEffects] = []
        self._processor = _Processor()
        self._turn = 0
        self._elapsed_samples = 0
        self._pending: PendingEffectState | None = None
        self._noise: BackgroundNoiseGenerator | None = None
        self._out_of_turn: OutOfTurnSpeechGenerator | None = None
        self._scheduler: EffectScheduler | None = None

    @property
    def needed_assets(self) -> list[tuple[str, str]]:
        """``(kind, name)`` of every noise file this call can play."""
        needed = [("bursts", name) for name in self._scheduler_bursts()]
        if self.background_noise_file:
            needed.insert(0, ("continuous", self.background_noise_file))
        return needed

    def _scheduler_bursts(self) -> list[str]:
        source = self.config.source_effects_config
        if not source.enable_burst_noise:
            return []
        # tau2 passes None when the task names no bursts; the scheduler then
        # takes BURST_NOISE_FILES.
        return self.burst_noise_files or self.turn_burst_files

    def _asset(self, name: str) -> Path:
        try:
            return self.assets[name]
        except KeyError:
            raise KeyError(f"tau2 noise file {name!r} was not fetched") from None

    async def open(self, synthesize: Synthesize, *, voice_id: str) -> None:
        """tau2's ``get_init_state``: noise bed, out-of-turn sounds, scheduler.

        The out-of-turn sounds (vocal tics, non-directed phrases) are
        synthesised here, through ``synthesize``, then handed to tau2's own
        generator, which muffles the phrases exactly as it does.
        """
        items = self.config.speech_effects_config.get_out_of_turn_speech_inserts()
        clips = {item.text: await synthesize(item.text) for item in items}

        def from_clips(*, text: str, provider: str, provider_config: Any) -> AudioData:
            return pcm16(clips[text])

        background = (
            self._asset(self.background_noise_file)
            if self.background_noise_file
            else None
        )
        with synthesizer(from_clips, voice_id=voice_id):
            self._noise, self._out_of_turn = create_streaming_audio_generators(
                synthesis_config=self.config,
                persona_name=self.persona_name,
                sample_rate=SAMPLE_RATE,
                background_noise_file=background,
            )
        bursts = self._scheduler_bursts()
        self._scheduler = EffectScheduler(
            seed=self.voice_seed,
            source_config=self.config.source_effects_config,
            speech_config=self.config.speech_effects_config,
            channel_config=self.config.channel_effects_config,
            sample_rate=SAMPLE_RATE,
            burst_noise_files=[self._asset(n) for n in bursts] if bursts else [],
        )

    # ------------------------------------------------------------- per line

    def next_line(self, text: str) -> LineEffects:
        """``synthesize_voice``'s text half: this turn's draw, and its tic."""
        turn = self._turn
        self._turn += 1
        speech_fx, _, channel_fx = generate_turn_effects(
            seed=self.voice_seed,
            turn_idx=turn,
            synthesis_config=self.config,
            burst_noise_files=[Path(n) for n in self.turn_burst_files],
        )
        said = text
        insert = speech_fx.speech_insert
        if insert:
            said = insert_speech_text(
                text,
                insert,
                rng=random.Random(self.voice_seed + turn),
                min_words=self.config.speech_effects_config.min_words_for_vocal_tics,
                in_turn=True,
            )
        line = LineEffects(
            turn=turn,
            said=said,
            muffled=speech_fx.dynamic_muffling_enabled,
            tic=insert.text if insert else None,
            speech_effects=speech_fx.model_dump(mode="json"),
            channel_effects=channel_fx.model_dump(mode="json"),
        )
        self.lines.append(line)
        return line

    def muffle(self, line: LineEffects, pcm: bytes) -> bytes:
        """``synthesize_voice``'s audio half: tau2's batch effects, speech only."""
        if not line.muffled or not pcm:
            return pcm
        result = self._processor.apply_batch_effects(
            audio=pcm16(pcm),
            speech_effects=SpeechEffectsResult(dynamic_muffling_enabled=True),
            source_effects=SourceEffectsResult(),
            channel_effects=ChannelEffectsResult(),
            speech_config=self.config.speech_effects_config,
            channel_config=self.config.channel_effects_config,
            noise_generator=None,
            apply_source_effects=False,
            apply_channel_effects=False,
            apply_telephony=False,
        )
        return result.data

    # ------------------------------------------------------------- per step

    def mix(self, speech: bytes | None) -> StepMix:
        """``_apply_chunk_effects`` up to the phone line, timeline included.

        ``speech`` is one step of the line being spoken, or ``None`` for a
        step without speech (waiting, a tool call, a yield).
        """
        noise, scheduler = self._noise, self._scheduler
        if noise is None or scheduler is None:
            raise RuntimeError("open() the effects before mixing")
        timeline = self.timeline
        num_samples = self.step_samples
        if speech is not None and len(speech) != 2 * num_samples:
            raise ValueError(f"a step is {2 * num_samples} bytes, got {len(speech)}")
        current_time_ms = int(self._elapsed_samples * 1000 / SAMPLE_RATE)

        # 1. Scheduled effects
        scheduled = scheduler.check_for_effects(
            chunk_duration_ms=int(num_samples * 1000 / SAMPLE_RATE),
            is_silence=speech is None,
            current_time_ms=current_time_ms,
            has_active_burst=noise.has_active_burst(),
        )

        # 3. The multitrack pipeline
        result = self._processor.process_streaming_chunk(
            speech_audio=pcm16(speech) if speech is not None else None,
            noise_generator=noise,
            num_samples=num_samples,
            scheduled_effects=scheduled,
            out_of_turn_generator=self._out_of_turn,
            pending_effect=self._pending,
        )
        self._pending = result.pending_effect

        # 4. Timeline
        if result.burst_noise_file:
            burst_params: dict[str, Any] = {"file": Path(result.burst_noise_file).name}
            if "burst_snr_db" in result.tracks.metadata:
                burst_params["snr_db"] = result.tracks.metadata["burst_snr_db"]
            if "burst_scale" in result.tracks.metadata:
                burst_params["scale"] = result.tracks.metadata["burst_scale"]
            timeline.open_event(
                effect_type="burst_noise",
                start_ms=current_time_ms,
                participant="user",
                params=burst_params,
            )
        if result.triggered_speech_insert:
            timeline.open_event(
                effect_type="out_of_turn_speech",
                start_ms=current_time_ms,
                participant="user",
                params={
                    "type": result.triggered_speech_insert.type,
                    "text": result.triggered_speech_insert.text,
                },
            )
        if result.pending_effect_completed:
            timeline.close_event(
                effect_type="out_of_turn_speech",
                end_ms=current_time_ms,
                participant="user",
            )
        if result.pending_effect_discarded and timeline.has_open_event(
            "out_of_turn_speech", "user"
        ):
            timeline.close_event(
                effect_type="out_of_turn_speech",
                end_ms=current_time_ms,
                participant="user",
            )
        if not noise.has_active_burst() and timeline.has_open_event(
            "burst_noise", "user"
        ):
            timeline.close_event(
                effect_type="burst_noise", end_ms=current_time_ms, participant="user"
            )

        # 6. Sum the tracks
        mixed = result.to_mixed_audio_data().data

        # 8. Frame drops, recorded now, applied by the caller after the phone line
        drops: list[int] = []
        for effect in scheduled:
            if effect.effect_type == "frame_drop" and effect.frame_drop_duration_ms:
                timeline.open_event(
                    effect_type="frame_drop",
                    start_ms=current_time_ms,
                    participant="user",
                    params={"duration_ms": effect.frame_drop_duration_ms},
                )
                timeline.close_event(
                    effect_type="frame_drop",
                    end_ms=current_time_ms + effect.frame_drop_duration_ms,
                    participant="user",
                )
                drops.append(effect.frame_drop_duration_ms)

        self._elapsed_samples += num_samples
        source = (
            result.source_effects.model_dump(mode="json")
            if result.source_effects
            else None
        )
        if source and source["burst_noise_file"]:
            source["burst_noise_file"] = Path(source["burst_noise_file"]).name
        return StepMix(pcm=mixed, frame_drops_ms=tuple(drops), source_effects=source)

    def drop_frames(self, wire: bytes, frame_drops_ms: Sequence[int]) -> bytes:
        """Step 8: tau2's ``_apply_frame_drop`` on the encoded (μ-law) step."""
        for duration_ms in frame_drops_ms:
            wire = self._processor.apply_streaming_frame_drop(
                audio_bytes=wire,
                sample_rate=TELEPHONY_RATE,
                bytes_per_sample=1,
                drop_duration_ms=duration_ms,
                is_ulaw=True,
            )
        return wire

    # ---------------------------------------------------------------- after

    def close(self) -> dict[str, Any]:
        """tau2's ``stop``: close what is still open; the timeline as JSON."""
        self.timeline.close_all_open(int(self._elapsed_samples * 1000 / SAMPLE_RATE))
        return self.timeline.model_dump(mode="json")

    def summary(self) -> dict[str, Any]:
        """What the call got: the draws per line and the events per kind."""
        events: dict[str, int] = {}
        for event in self.timeline.events:
            events[event.effect_type] = events.get(event.effect_type, 0) + 1
        return {
            "voice_seed": self.voice_seed,
            "background_noise_file": self.background_noise_file,
            "burst_noise_files": self.burst_noise_files,
            "lines": len(self.lines),
            "lines_muffled": sum(1 for line in self.lines if line.muffled),
            "lines_with_tic": sum(1 for line in self.lines if line.tic),
            "events": events,
        }


def pcm16(data: bytes) -> AudioData:
    """16 kHz mono PCM16 as tau2's ``AudioData``."""
    return AudioData(
        data=data,
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=SAMPLE_RATE, channels=1
        ),
    )
