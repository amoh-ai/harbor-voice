"""Sarvam speech to text: Saaras, over the realtime websocket.

``/speech-to-text-realtime/ws`` rather than the legacy ``/speech-to-text/ws``:
only the realtime endpoint sends partial transcripts. And ``endpointing=manual``
rather than Sarvam's VAD: the cascade has already decided the turn is over when
it hands audio here, so one ``speech_start`` / ``speech_end`` pair makes the
whole turn one utterance, and so exactly one ``transcript.final``.

The session, as the API reference (docs.sarvam.ai/api-reference/speech-to-text/
transcribe/realtime/ws) runs it:

- connect to ``wss://api.sarvam.ai/speech-to-text-realtime/ws?model=…&
  language_code=…&endpointing=manual&stream_type=fast&encoding=…&
  sample_rate=…`` with the ``Api-Subscription-Key`` header;
- send ``speech_start``, the audio as ``audio_input`` JSON messages of base64,
  ``speech_end``, then ``end``. Audio goes out in 100 ms chunks paced to real
  time, as Sarvam's own example sends it, so a turn takes at least its own
  length to transcribe;
- read ``transcript.partial`` (the utterance so far), ``transcript.final``,
  and ``session.end``, whose ``audio_duration_s`` is the billed quantity.

The endpoint takes 8 or 16 kHz only, so 24 kHz audio is resampled to 16 kHz
with the anti-aliased converter. ``stream_type=fast`` is the one Sarvam
recommends for conversational agents.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import urllib.parse
from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import Any

from harbor_voice._deps import numpy as _numpy
from harbor_voice._deps import websockets as _websockets
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.perturbation import dsp
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.protocols import Transcript
from harbor_voice.providers.sarvam.tts import DEFAULT_LANGUAGE
from harbor_voice.providers.sarvam.tts import api_key
from harbor_voice.providers.sarvam.tts import error

NAME = "sarvam.stt"
DEFAULT_MODEL = "saaras:v4"
WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
CHUNK_MS = 100
"""Sarvam's example sends 100 ms chunks, one per 100 ms."""

SAMPLE_RATES = frozenset({8_000, 16_000})
ENCODINGS = {"pcm16": "linear16", "mulaw": "mulaw", "alaw": "alaw"}
"""Our encodings, by Sarvam's name."""

_CLOSE_CODES = {
    1003: ProviderFailure.THROTTLED,
    1008: ProviderFailure.TIMEOUT,
    1011: ProviderFailure.STREAM_BROKEN,
    4000: ProviderFailure.MALFORMED,
}
"""Sarvam's documented close codes. 1003 also covers an invalid key, but a
rejected key normally arrives first as an ``error`` event with its 403."""


class SarvamSTT:
    """One websocket session per turn.

    Args:
        model: ``saaras:v4`` or ``saaras:v3-realtime``.
        language_code: BCP-47 code of the audio, or ``auto`` to detect it.
        api_key: Defaults to ``SARVAM_API_KEY``.
        connect: Replaces ``websockets.connect``; for tests.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        language_code: str = DEFAULT_LANGUAGE,
        api_key: str | None = None,
        url: str = WS_URL,
        connect: Any = None,
    ) -> None:
        self.name = NAME
        self.model = model
        self.language_code = language_code
        self._api_key = api_key
        self._url = url
        self._connect = connect
        self.usage = Usage(audio_seconds=0.0)
        """The last turn's audio seconds, the unit Sarvam bills."""

    async def transcribe(
        self, frames: Sequence[AudioFrame]
    ) -> AsyncIterator[Transcript]:
        """Yield partials while the audio is sent, then one final."""
        pcm = b"".join(f.pcm for f in frames)
        if not pcm:
            yield Transcript(text="", is_final=True, audio_duration_s=0.0)
            return
        fmt = frames[0].fmt
        duration = fmt.duration_s(len(pcm))
        pcm, fmt = _sendable(pcm, fmt)
        query = urllib.parse.urlencode(
            {
                "model": self.model,
                "language_code": self.language_code,
                "endpointing": "manual",
                "stream_type": "fast",
                "encoding": ENCODINGS[fmt.encoding],
                "sample_rate": fmt.sample_rate_hz,
            }
        )
        headers = {"Api-Subscription-Key": api_key(self._api_key, NAME, self.model)}
        connect = self._connect or _websockets().connect
        try:
            ws = await connect(f"{self._url}?{query}", additional_headers=headers)
        except Exception as exc:
            raise _classify(exc, self.model) from exc

        async def send() -> None:
            await ws.send(json.dumps({"event": "speech_start"}))
            step = fmt.frame_bytes(CHUNK_MS)
            for start in range(0, len(pcm), step):
                chunk = base64.b64encode(pcm[start : start + step]).decode("ascii")
                await ws.send(json.dumps({"event": "audio_input", "audio": chunk}))
                await asyncio.sleep(CHUNK_MS / 1000)
            await ws.send(json.dumps({"event": "speech_end"}))
            await ws.send(json.dumps({"event": "end"}))

        sender = asyncio.create_task(send())
        finals: list[str] = []
        billed: float | None = None
        try:
            async for raw in ws:
                event = json.loads(raw)
                kind = event.get("event")
                if kind == "transcript.partial":
                    yield Transcript(text=event.get("text", ""), is_final=False)
                elif kind == "transcript.final":
                    finals.append(event.get("text", ""))
                elif kind == "error" and event.get("is_fatal", True):
                    raise _error_event(event, self.model)
                elif kind == "session.end":
                    if event.get("audio_duration_s") is not None:
                        billed = float(event["audio_duration_s"])
                    break
            else:
                if not finals:
                    raise ProviderError(
                        ProviderFailure.STREAM_BROKEN,
                        "socket closed before a final transcript",
                        provider=NAME,
                        model=self.model,
                    )
            await sender
        except ProviderError:
            raise
        except Exception as exc:
            raise _classify(exc, self.model) from exc
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender
            with contextlib.suppress(Exception):  # keep the error that got us here
                await ws.close()
        self.usage = Usage(audio_seconds=duration if billed is None else billed)
        text = " ".join(t.strip() for t in finals if t.strip())
        yield Transcript(text=text, is_final=True, audio_duration_s=duration)


def _sendable(pcm: bytes, fmt: AudioFormat) -> tuple[bytes, AudioFormat]:
    """The audio at a rate the endpoint accepts: 16 kHz unless already 8 or 16."""
    if fmt.channels != 1:
        raise ValueError(f"{NAME}: mono audio only, got {fmt}")
    if fmt.sample_rate_hz in SAMPLE_RATES:
        return pcm, fmt
    if fmt.encoding != "pcm16":
        raise ValueError(f"{NAME}: {fmt} must be at 8 or 16 kHz")
    target = AudioFormat(sample_rate_hz=16_000)
    samples = dsp.resample_clean(dsp.to_array(pcm), fmt.sample_rate_hz, 16_000)
    return dsp.to_pcm(_numpy().asarray(samples)), target


def _error_event(event: dict[str, Any], model: str) -> ProviderError:
    message = f"{event.get('code', 'error')}: {event.get('message', '')}"
    status = event.get("status_code")
    if isinstance(status, int):
        return error(status, message, NAME, model)
    return ProviderError(
        ProviderFailure.STREAM_BROKEN, message[:300], provider=NAME, model=model
    )


def _classify(exc: Exception, model: str) -> ProviderError:
    """A failed handshake by HTTP status, a closed socket by close code."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return error(status, str(exc), NAME, model)
    code = getattr(getattr(exc, "rcvd", None), "code", None)
    if code in _CLOSE_CODES:
        kind = _CLOSE_CODES[code]
    elif isinstance(exc, (TimeoutError, OSError)):
        kind = ProviderFailure.TIMEOUT
    elif isinstance(exc, json.JSONDecodeError):
        kind = ProviderFailure.MALFORMED
    else:
        kind = ProviderFailure.STREAM_BROKEN
    return ProviderError(
        kind, f"{type(exc).__name__}: {str(exc)[:300]}", provider=NAME, model=model
    )


SARVAM_STT_SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.STT,
    capabilities=ProviderCapabilities(
        modality=Modality.STT,
        streams=True,
        emits_partials=True,
        word_timings=False,
        diarizes=False,
    ),
    credential_provider="sarvam",
    requires_extra="voice",
    default_model=DEFAULT_MODEL,
)
"""Partials are ``transcript.partial`` events. The endpoint has no word
timings (only segment ``start_s``/``end_s``) and no diarization."""


def build_stt(*, model: str = DEFAULT_MODEL, **config: Any) -> SarvamSTT:
    return SarvamSTT(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SARVAM_STT_SPEC, build_stt),)
