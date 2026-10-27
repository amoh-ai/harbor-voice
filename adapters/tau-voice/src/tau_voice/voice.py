"""The tau-voice caller's voice: a ``UniversalSimulator``, streamed per step.

tau2's caller works in 0.2 s steps; the simulator's pipeline works in frames
(20 ms by default). This bridges the two: a step of clean audio is split into
frames and each is emitted through the simulator's signal chain — layers, then
the wire encoder — and the encoded frames are joined back into the step's wire
audio. Silent steps go through too, so a filter's tail and any noise bed run
on exactly as on a line.

Lines are recorded on the simulator as utterances: whole when the caller
finished them, partial (``truncated``) when it yielded.
"""

from __future__ import annotations

from harbor_voice.simulator import PreparedLine
from harbor_voice.simulator import UniversalSimulator


class SimulatorVoice:
    """A :class:`~tau_voice.caller.CallerVoice` backed by a ``UniversalSimulator``."""

    def __init__(self, simulator: UniversalSimulator, *, step_s: float = 0.2) -> None:
        fmt = simulator.audio_format
        self.simulator = simulator
        self.step_bytes = int(fmt.sample_rate_hz * step_s) * fmt.bytes_per_sample
        self._frame_bytes = fmt.frame_bytes()
        if self.step_bytes % self._frame_bytes:
            raise ValueError(
                f"a {step_s} s step is not a whole number of "
                f"{fmt.frame_bytes()}-byte frames"
            )
        self._sent: dict[int, list] = {}

    async def prepare(self, text: str) -> PreparedLine:
        return await self.simulator.prepare(text)

    def emit_step(
        self, pcm: bytes, *, is_speech: bool, line: PreparedLine | None = None
    ) -> bytes:
        frames = [
            self.simulator.emit(pcm[i : i + self._frame_bytes], is_speech=is_speech)
            for i in range(0, len(pcm), self._frame_bytes)
        ]
        if line is not None:
            self._sent.setdefault(id(line), []).extend(frames)
        return b"".join(frame.pcm for frame in frames)

    def finish(self, line: PreparedLine, *, truncated: bool) -> None:
        self.simulator.finish(line, self._sent.pop(id(line), []), truncated=truncated)
