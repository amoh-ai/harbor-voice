"""What the three Deepgram providers share: the key, the header and the
encodings.

Every Deepgram API takes ``Authorization: Token <key>``
(https://developers.deepgram.com/reference/authentication).
"""

from __future__ import annotations

from collections.abc import Mapping

from harbor_voice.audio.format import AudioFormat
from harbor_voice.providers.base import require_key

API_KEY_ENV = "DEEPGRAM_API_KEY"

ENCODINGS: Mapping[str, str] = {"pcm16": "linear16", "mulaw": "mulaw", "alaw": "alaw"}
"""Our encodings as Deepgram names them. All three are raw, headerless audio."""


def encoding(fmt: AudioFormat) -> str:
    """Deepgram's name for a mono format; a setup mistake raises ``ValueError``."""
    if fmt.channels != 1:
        raise ValueError(f"Deepgram providers here take mono audio, got {fmt}")
    return ENCODINGS[fmt.encoding]


def auth_header(api_key: str | None, *, provider: str, model: str) -> dict[str, str]:
    """The ``Authorization`` header, from ``api_key`` or ``DEEPGRAM_API_KEY``."""
    key = require_key(api_key, API_KEY_ENV, provider=provider, model=model)
    return {"Authorization": f"Token {key}"}
