"""The tau-voice caller's voice: a ``UniversalSimulator``, streamed per step.

tau2's caller works in 0.2 s steps; the simulator's pipeline works in frames
(20 ms by default). This bridges the two: a step of clean audio is split into
frames and each is emitted through the simulator's signal chain — layers, then
the wire encoder — and the encoded frames are joined back into the step's wire
audio. Silent steps go through too, so a filter's tail and any noise bed run
on exactly as on a line.

Lines are recorded on the simulator as utterances: whole when the caller
finished them, partial (``truncated``) when it yielded.

With :class:`~tau_voice.effects.Tau2CallerEffects` (every preset but
``control``), tau2's effects run around that chain at tau2's own unit, the
step: a line gets its vocal tic and muffling when prepared; a step is mixed
with the noise bed, bursts and out-of-turn sounds before the frames are
emitted, and frame drops zero the start of the encoded step after.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from harbor_voice.audio.frame import AudioFrame
from harbor_voice.simulator import PreparedLine
from harbor_voice.simulator import UniversalSimulator
from tau_voice.effects import LineEffects
from tau_voice.effects import Tau2CallerEffects


class SimulatorVoice:
    """A :class:`~tau_voice.caller.CallerVoice` backed by a ``UniversalSimulator``."""

    def __init__(
        self,
        simulator: UniversalSimulator,
        *,
        step_s: float = 0.2,
        effects: Tau2CallerEffects | None = None,
    ) -> None:
        fmt = simulator.audio_format
        self.simulator = simulator
        self.effects = effects
        self.step_bytes = int(fmt.sample_rate_hz * step_s) * fmt.bytes_per_sample
        self._frame_bytes = fmt.frame_bytes()
        if self.step_bytes % self._frame_bytes:
            raise ValueError(
                f"a {step_s} s step is not a whole number of "
                f"{fmt.frame_bytes()}-byte frames"
            )
        self._sent: dict[int, list[AudioFrame]] = {}
        self._drawn: dict[int, LineEffects] = {}
        self.last_effects: dict[str, Any] | None = None
        """tau2's effect results for the step last emitted, as its chunk
        carries them: ``speech_effects`` (the line's), ``source_effects`` (the
        step's) and ``channel_effects``. ``None`` without effects."""

    async def prepare(self, text: str) -> PreparedLine:
        if self.effects is None:
            return await self.simulator.prepare(text)
        drawn = self.effects.next_line(text)
        line = await self.simulator.prepare(drawn.said)
        if drawn.muffled:
            line = dataclasses.replace(line, pcm=self.effects.muffle(drawn, line.pcm))
        self._drawn[id(line)] = drawn
        return line

    def emit_step(
        self, pcm: bytes, *, is_speech: bool, line: PreparedLine | None = None
    ) -> bytes:
        drops: tuple[int, ...] = ()
        self.last_effects = None
        if self.effects is not None:
            mixed = self.effects.mix(pcm if is_speech else None)
            pcm, drops = mixed.pcm, mixed.frame_drops_ms
            drawn = self._drawn.get(id(line)) if is_speech and line else None
            # tau2's _apply_frame_drop: each drop adds to what the chunk has.
            channel = dict(drawn.channel_effects) if drawn else None
            for ms in drops:
                total = ms + (channel["frame_drop_ms"] if channel else 0)
                channel = {"frame_drops_enabled": True, "frame_drop_ms": total}
            self.last_effects = {
                "speech_effects": drawn.speech_effects if drawn else None,
                "source_effects": mixed.source_effects,
                "channel_effects": channel,
            }
        frames = [
            self.simulator.emit(pcm[i : i + self._frame_bytes], is_speech=is_speech)
            for i in range(0, len(pcm), self._frame_bytes)
        ]
        if drops and self.effects is not None:
            frames = self._drop(frames, drops)
        if line is not None:
            self._sent.setdefault(id(line), []).extend(frames)
        return b"".join(frame.pcm for frame in frames)

    def finish(self, line: PreparedLine, *, truncated: bool) -> None:
        self.simulator.finish(line, self._sent.pop(id(line), []), truncated=truncated)
        self._drawn.pop(id(line), None)

    def _drop(
        self, frames: list[AudioFrame], drops: tuple[int, ...]
    ) -> list[AudioFrame]:
        """Frame drops apply to the encoded step as a whole, as in tau2."""
        assert self.effects is not None
        wire = self.effects.drop_frames(b"".join(f.pcm for f in frames), drops)
        out, offset = [], 0
        for frame in frames:
            size = len(frame.pcm)
            out.append(dataclasses.replace(frame, pcm=wire[offset : offset + size]))
            offset += size
        return out
