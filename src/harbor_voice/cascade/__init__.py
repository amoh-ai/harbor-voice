"""The reference cascade — STT -> LLM -> TTS, composed in-process.

harbor-voice ships this so the ablation exists: hold the
LLM fixed, swap the STT, and mishearing separates from misreasoning. No
speech-to-speech provider permits that, because it exposes no seam between
hearing and reasoning.

Two transports: :class:`CascadeTransport` takes turns it is handed (input
stops arriving), :class:`DuplexCascadeTransport` finds them on an open line,
with tau2's cascaded turn-taking, for a benchmark that runs a full-duplex call.

The slots it composes live in :mod:`harbor_voice.providers`, not here. The
cascade consumes providers; it does not own them — the simulator
and the metrics layer hold the same protocols for different purposes.
"""

from __future__ import annotations

from harbor_voice.cascade.adapter import CascadeAdapter
from harbor_voice.cascade.duplex import DuplexCascadeTransport
from harbor_voice.cascade.transport import DEFAULT_INPUT_IDLE_S
from harbor_voice.cascade.transport import CascadeTransport

__all__ = [
    "DEFAULT_INPUT_IDLE_S",
    "CascadeAdapter",
    "CascadeTransport",
    "DuplexCascadeTransport",
]
