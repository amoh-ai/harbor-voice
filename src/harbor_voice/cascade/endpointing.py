"""Where speech starts and ends on a line that never goes quiet.

:class:`~harbor_voice.cascade.transport.CascadeTransport` ends a turn when no
frame has arrived for a while. On a phone line frames never stop: silence is
sent too, so that rule never fires. Here a turn ends the way tau2's cascaded
agent ends one, on the audio itself: speech, then ``endpointing_ms`` of
non-speech (Deepgram's ``endpointing``, 350 ms in tau2's config).

tau2 gets both edges from Deepgram's streaming recogniser, which runs a
trained voice activity model. Our recognisers take a finished utterance, so
the edges are found here, by level: a 20 ms window is speech when it stands
``margin_db`` above the line's noise floor (and above ``min_dbfs``). The
floor is the quietest window of the last ``floor_ms``: speech has gaps
between words, and in them the line falls back to its noise, so steady
background noise (tau2's noisy presets mix it under the whole call) sets the
floor instead of reading as one endless utterance. It is a level detector,
not a speech model: a loud burst of noise opens it just as speech does. tau2's Deepgram VAD is fooled by bursts too,
less often.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from harbor_voice.audio.format import AudioFormat
from harbor_voice.perturbation import dsp

WINDOW_MS = 20

DEFAULT_ENDPOINTING_MS = 350
"""tau2's ``DeepgramSTTConfig.endpointing_ms``: non-speech that ends a turn."""


@dataclass(frozen=True, slots=True)
class SpeechEdge:
    """Speech started or ended, at ``at_ms`` into the audio fed so far."""

    started: bool
    at_ms: int
    audio: bytes = b""
    """On an end: the utterance, from a little before it started to now."""


class Endpointer:
    """Turns a continuous PCM16 stream into speech start and end edges.

    Args:
        fmt: Of the audio fed; PCM16 mono.
        endpointing_ms: Non-speech after speech that ends it.
        start_ms: Speech needed before a start is reported, so a click is
            not a turn.
        margin_db: How far above the noise floor speech must be.
        min_dbfs: Quieter than this is never speech, however quiet the line.
        floor_ms: How far back the floor looks for the line's quietest
            window; longer than any stretch of speech without a gap.
        preroll_ms: Audio kept from before the start, so the recogniser
            hears a word's first sound, which is quieter than its vowel.
    """

    def __init__(
        self,
        fmt: AudioFormat,
        *,
        endpointing_ms: int = DEFAULT_ENDPOINTING_MS,
        start_ms: int = 60,
        margin_db: float = 10.0,
        min_dbfs: float = -50.0,
        floor_ms: int = 1500,
        preroll_ms: int = 200,
    ) -> None:
        if fmt.encoding != "pcm16" or fmt.channels != 1:
            raise ValueError(f"endpointing needs PCM16 mono, got {fmt}")
        self._window = fmt.frame_bytes(WINDOW_MS)
        self._end_windows = math.ceil(endpointing_ms / WINDOW_MS)
        self._start_windows = max(1, math.ceil(start_ms / WINDOW_MS))
        self._margin = margin_db
        self._min = min_dbfs
        self._levels: deque[float] = deque(maxlen=math.ceil(floor_ms / WINDOW_MS))
        self._pending = b""
        self._windows = 0
        self._speaking = False
        self._run = 0
        """Consecutive windows on the other side of the current state."""
        self._recent: deque[bytes] = deque(
            maxlen=self._start_windows + math.ceil(preroll_ms / WINDOW_MS)
        )
        self._utterance: list[bytes] = []

    def feed(self, pcm: bytes) -> list[SpeechEdge]:
        """Consume audio; return the edges it completed, in order."""
        self._pending += pcm
        edges: list[SpeechEdge] = []
        while len(self._pending) >= self._window:
            window = self._pending[: self._window]
            self._pending = self._pending[self._window :]
            self._windows += 1
            edge = self._step(window)
            if edge is not None:
                edges.append(edge)
        return edges

    def _step(self, window: bytes) -> SpeechEdge | None:
        level = dsp.db(dsp.rms(dsp.to_array(window)))
        self._levels.append(level)
        voiced = level > max(min(self._levels) + self._margin, self._min)
        if not self._speaking:
            self._recent.append(window)
            self._run = self._run + 1 if voiced else 0
            if self._run < self._start_windows:
                return None
            self._speaking, self._run = True, 0
            self._utterance = list(self._recent)
            self._recent.clear()
            started = self._windows - self._start_windows
            return SpeechEdge(started=True, at_ms=started * WINDOW_MS)
        self._utterance.append(window)
        self._run = 0 if voiced else self._run + 1
        if self._run < self._end_windows:
            return None
        self._speaking, self._run = False, 0
        audio, self._utterance = b"".join(self._utterance), []
        return SpeechEdge(started=False, at_ms=self._windows * WINDOW_MS, audio=audio)
