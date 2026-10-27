"""G.711 μ-law — the wire format of telephone audio.

Real G.711 bytes, via ``audioop`` (``audioop-lts`` on Python 3.13+, where it
left the stdlib). Use this for anything that goes on a wire as ``pcmu`` /
``g711_ulaw``: OpenAI Realtime's ``audio/pcmu``, Twilio media streams, any
telephony transport.

Not to be confused with :func:`harbor_voice.perturbation.dsp.linear_to_mulaw`,
which is a simplified μ-law used to *simulate* the codec's character inside a
perturbation layer. That one is sign plus a 7-bit magnitude, encodes silence
as ``0x00`` — full-scale negative in G.711 — and must never reach a wire.
"""

from __future__ import annotations

import audioop

SILENCE = b"\xff"
"""Positive zero. ``0x7f`` (negative zero) also decodes to silence."""


def encode(pcm16: bytes) -> bytes:
    """PCM16-LE → μ-law, one byte per sample, same rate."""
    return audioop.lin2ulaw(pcm16, 2)


def decode(ulaw: bytes) -> bytes:
    """μ-law → PCM16-LE, same rate."""
    return audioop.ulaw2lin(ulaw, 2)
