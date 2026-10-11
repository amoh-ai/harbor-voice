"""Stimulus helpers for the conformance suite."""

from __future__ import annotations

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import CANONICAL_FRAME_MS
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId


def silence_stimulus(
    *,
    seconds: float,
    party: PartyId,
    fmt: AudioFormat = CANONICAL,
    frame_ms: int = CANONICAL_FRAME_MS,
    t0: float = 0.0,
    paced: bool = False,
) -> tuple[AudioFrame, ...]:
    """Frames of digital silence, correctly sized and timestamped.

    Args:
        seconds: Total duration to generate.
        party: Who the frames are attributed to.
        fmt: Audio format; defaults to canonical.
        frame_ms: Frame duration.
        t0: Base timestamp.
        paced: When false (the default) every frame carries ``t0`` as its
            ``t_produced``, modelling **batch** synthesis — the whole utterance
            came into existence at one instant. When true, frames are stamped
            on a real-time schedule, modelling **streaming** synthesis. Only
            use ``paced`` if the caller genuinely emits that fast: a frame
            cannot be transmitted before it exists, and the monotonic guard in
            :class:`~harbor_voice.audio.frame.AudioFrame` enforces it.

    Returns:
        Frames enough to exercise lifecycle and framing. **Not** enough to make
        a provider answer, so stream checks needing a reply still skip — the
        honest outcome rather than a vacuous pass.
    """
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    n_bytes = fmt.frame_bytes(frame_ms)
    step = frame_ms / 1000
    count = max(1, round(seconds / step))
    silence = bytes(n_bytes)
    return tuple(
        AudioFrame(
            pcm=silence,
            seq=i,
            party=party,
            t_produced=t0 + i * step if paced else t0,
            fmt=fmt,
        )
        for i in range(count)
    )
