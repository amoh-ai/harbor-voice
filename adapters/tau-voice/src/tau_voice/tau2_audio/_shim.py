"""What the vendored tau2 audio code imports from the rest of tau2.

Hand-written (not vendored). Each name stands in for one tau2 import, with
the same value or the same behaviour on the paths the effects use.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("tau_voice.tau2_audio")
"""For ``loguru.logger``: tau2 logs with pre-formatted f-strings."""

# tau2.config
DEFAULT_PCM_SAMPLE_RATE = 16000
DEFAULT_TELEPHONY_RATE = 8000
DEFAULT_TICK_DURATION_SECONDS = 0.20
DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE = 24000

# tau2.utils.utils: tau2's data folder. Noise files are always passed as
# explicit paths here (fetched from the sidecar), so this is never read.
DATA_DIR = Path(os.environ.get("TAU2_DATA_DIR", "tau2-data"))

# tau2.voice.utils.utils: tau2 scans its data folder for burst files when
# none are passed. Callers here always pass them.
BURST_NOISE_FILES: list[Path] = []


class SynthesisConfig(BaseModel):
    """For ``tau2.data_model.voice.SynthesisConfig``: the fields the effects read."""

    provider: str = "elevenlabs"
    provider_config: Any = None
    channel_effects_config: Any = None
    source_effects_config: Any = None
    speech_effects_config: Any = None


Synthesizer = Callable[..., Any]
_synthesizer: Synthesizer | None = None
_voice_id: str | None = None


@contextmanager
def synthesizer(synthesize: Synthesizer, *, voice_id: str) -> Iterator[None]:
    """Route tau2's ``synthesize_voice`` to harbor-voice's TTS, for a block.

    ``voice_id`` is the persona's voice as the sidecar resolved it (tau2,
    from ``TAU2_VOICE_ID_*`` in its container).
    """
    global _synthesizer, _voice_id
    previous = _synthesizer, _voice_id
    _synthesizer, _voice_id = synthesize, voice_id
    try:
        yield
    finally:
        _synthesizer, _voice_id = previous


def get_elevenlabs_voice_id(persona_name: str) -> str:
    """For ``tau2.data_model.voice_personas.get_elevenlabs_voice_id``."""
    if _voice_id is None:
        raise RuntimeError("no voice: use tau2_audio's synthesizer() block")
    return _voice_id


def synthesize_voice(*, text: str, provider: str, provider_config: Any) -> Any:
    """For ``tau2.voice.synthesis.synthesize.synthesize_voice``."""
    if _synthesizer is None:
        raise RuntimeError("no synthesizer: use tau2_audio's synthesizer() block")
    return _synthesizer(text=text, provider=provider, provider_config=provider_config)
