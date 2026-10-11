"""Rendering a call to audio.

**Mix on a timeline. Do not concatenate.**

Concatenating segments into one file is not a recording. The consequences
are not cosmetic: overlapping barge-in audio lands back-to-back so an interruption
sounds like an orderly exchange, inter-turn silence is dropped so the pacing is
wrong, and the saved duration does not match the recorded duration.

A saved recording that is not a faithful timeline rendering is **worse than no
recording**, because it will be used to adjudicate a disagreement and it will
lie. If two people were talking at once, the file has to say so.
"""

from __future__ import annotations

import io
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from harbor_voice._deps import numpy as _numpy
from harbor_voice.audio.clock import Clock
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.perturbation import dsp

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


@dataclass(frozen=True, slots=True)
class PlacedAudio:
    """Audio with a position on the call's timeline."""

    party: PartyId
    start_s: float
    pcm: bytes
    fmt: AudioFormat = CANONICAL

    @property
    def duration_s(self) -> float:
        return self.fmt.duration_s(len(self.pcm))

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True, slots=True)
class RenderResult:
    """A rendered mix and what it is made of."""

    pcm: bytes
    fmt: AudioFormat
    duration_s: float
    sources: int
    overlap_s: float
    """Audio time during which more than one party was speaking."""

    clipped_fraction: float

    @property
    def has_overlap(self) -> bool:
        return self.overlap_s > 0


def placed_from_frames(
    frames: Sequence[AudioFrame],
    *,
    clock: Clock = Clock.RX,
    fallback_start_s: float = 0.0,
) -> list[PlacedAudio]:
    """Group contiguous frames of one party into placed runs.

    Position comes from the frame's own clock stamp, so a gap in transmission
    becomes a gap in the render rather than being closed up.
    """
    placed: list[PlacedAudio] = []
    run: list[AudioFrame] = []
    run_start: float | None = None

    def _flush() -> None:
        nonlocal run, run_start
        if not run:
            return
        placed.append(
            PlacedAudio(
                party=run[0].party,
                start_s=run_start if run_start is not None else fallback_start_s,
                pcm=b"".join(f.pcm for f in run),
                fmt=run[0].fmt,
            )
        )
        run, run_start = [], None

    for frame in frames:
        t = frame.at(clock)
        if run and (frame.party != run[0].party or frame.fmt != run[0].fmt):
            _flush()
        if not run:
            run_start = t if t is not None else fallback_start_s
        run.append(frame)
    _flush()
    return placed


def render_timeline(
    sources: Sequence[PlacedAudio],
    *,
    fmt: AudioFormat = CANONICAL,
    t0: float | None = None,
    duration_s: float | None = None,
) -> RenderResult:
    """Mix placed audio onto one timeline.

    Overlapping sources are **summed**, so two parties talking at once sound
    like two parties talking at once. Silence between turns is preserved,
    because its length is a measurement.

    Args:
        sources: Audio with positions.
        fmt: Output format. Sources in another format are refused rather than
            silently resampled — a resample is a declared event.
        t0: Timeline origin. Defaults to the earliest source.
        duration_s: Output length. Defaults to covering every source.

    Returns:
        The mix, plus the overlap and clipping it contains.
    """
    np = _numpy()
    if not sources:
        return RenderResult(
            pcm=b"",
            fmt=fmt,
            duration_s=0.0,
            sources=0,
            overlap_s=0.0,
            clipped_fraction=0.0,
        )

    mismatched = {str(s.fmt) for s in sources if s.fmt != fmt}
    if mismatched:
        raise ValueError(
            f"cannot mix {sorted(mismatched)} into {fmt} without a declared "
            f"resample; convert at the adapter edge instead"
        )

    origin = t0 if t0 is not None else min(s.start_s for s in sources)
    span = (
        duration_s if duration_s is not None else max(s.end_s for s in sources) - origin
    )
    rate = fmt.sample_rate_hz
    total = max(1, int(round(span * rate)))

    mix = np.zeros(total, dtype="float64")
    voices = np.zeros(total, dtype="int32")

    for source in sources:
        samples = dsp.to_array(source.pcm)
        start = int(round((source.start_s - origin) * rate))
        if start < 0:
            samples = samples[-start:]
            start = 0
        end = min(total, start + samples.size)
        if end <= start:
            continue
        mix[start:end] += samples[: end - start]
        voices[start:end] += 1

    overlap_samples = int(np.count_nonzero(voices > 1))
    return RenderResult(
        pcm=dsp.to_pcm(mix),
        fmt=fmt,
        duration_s=total / rate,
        sources=len(sources),
        overlap_s=overlap_samples / rate,
        clipped_fraction=dsp.clipped_fraction(mix),
    )


def to_wav(pcm: bytes, fmt: AudioFormat = CANONICAL) -> bytes:
    """Wrap PCM16 in a WAV container, so a human can play it."""
    if fmt.encoding != "pcm16":
        raise ValueError(f"WAV export needs pcm16, got {fmt.encoding}")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(fmt.channels)
        handle.setsampwidth(fmt.bytes_per_sample)
        handle.setframerate(fmt.sample_rate_hz)
        handle.writeframes(pcm)
    return buffer.getvalue()


def downsample_to_fit(
    pcm: bytes,
    *,
    fmt: AudioFormat = CANONICAL,
    max_bytes: int,
    floor_hz: int = 3_000,
) -> tuple[bytes, AudioFormat]:
    """Step the rate down until the audio fits, **preserving duration**.

    The step-down keeps duration intact, so every duration-based assertion still
    holds against a shrunk artifact. A render that fits by being *truncated*
    would quietly break them.

    Returns:
        The audio and the format it is now in. If even ``floor_hz`` does not
        fit, the floor is returned anyway — shrinking further would start
        destroying the thing the artifact exists to preserve.
    """
    np = _numpy()
    rate = fmt.sample_rate_hz
    current = pcm
    current_fmt = fmt
    while len(current) > max_bytes and rate > floor_hz:
        rate = max(floor_hz, int(rate * 0.75) // 1000 * 1000 or floor_hz)
        samples = dsp.resample(dsp.to_array(pcm), fmt.sample_rate_hz, rate)
        current = dsp.to_pcm(np.asarray(samples))
        current_fmt = AudioFormat(
            sample_rate_hz=rate, channels=fmt.channels, encoding=fmt.encoding
        )
    return current, current_fmt
