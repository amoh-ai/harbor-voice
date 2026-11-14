"""Meta speech to text: Muse Voice Transcribe, over the realtime websocket.

Muse Voice Transcribe is the only speech model on Meta Model API. Meta's docs
say it "does not synthesize speech or provide a speech-to-speech conversation
API" (dev.meta.ai/docs/models), so this is the whole Meta voice surface.

``PUSH_TO_TALK`` rather than ``ENDPOINTING``: the cascade has already decided
the turn is over when it hands audio here, and push-to-talk is the mode where
the client delimits the turn. The session, as the API reference
(dev.meta.ai/docs/api-reference/voice/realtime) runs it:

- connect to ``wss://api.meta.ai/v1/asr/realtime``. The key travels in the
  handshake, not in a header: the endpoint ignores ``Authorization``;
- send the handshake as the first text frame and wait for its acknowledgement
  (``{"sessionId": …}``, the only frame without a ``type``);
- send raw PCM as binary frames of 80 ms, paced to real time. The server
  rejects streams that run ahead of real time, so a turn takes at least its
  own length to transcribe;
- send ``{"type": "endStream"}`` and read ``transcript`` events: each partial
  replaces the last (``CUMULATIVE``), and ``final: true`` is the turn.

The model's native rate is 24 kHz; 16 kHz is also accepted. Anything else is
resampled to 24 kHz. Billed per hour of audio, in whole seconds rounded down.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import Any

from harbor_voice._deps import numpy as _numpy
from harbor_voice._deps import websockets as _websockets
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.perturbation import dsp
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.base import require_key
from harbor_voice.providers.base import socket_failure
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.protocols import Transcript

NAME = "meta.stt"
DEFAULT_MODEL = "muse-voice-transcribe-1.0"
WS_URL = "wss://api.meta.ai/v1/asr/realtime"
API_KEY_ENVS = ("META_API_KEY", "MODEL_API_KEY")
"""harbor's ``meta`` entry; Meta's own docs use ``MODEL_API_KEY``."""
CHUNK_MS = 80
"""The API reference's example frame, sent once per ``CHUNK_MS``."""

ENCODINGS = {24_000: "PCM_24KHZ", 16_000: "PCM_16KHZ"}

_CLOSE_CODES = {
    1008: ProviderFailure.MALFORMED,
    1011: ProviderFailure.STREAM_BROKEN,
    1013: ProviderFailure.THROTTLED,
}
"""Meta's documented close codes. 1008 is a bad request or bad pacing, which
the docs say fails again on retry."""

_ERROR_TYPES = {
    "authentication_error": ProviderFailure.AUTH,
    "rate_limit_error": ProviderFailure.THROTTLED,
    "server_error": ProviderFailure.STREAM_BROKEN,
}
"""Model API's error taxonomy, as an ``error`` event's ``errorType``."""


class MetaSTT:
    """One websocket session per turn.

    Args:
        model: Meta model id.
        language_bias: Language names to steer toward, e.g. ``["English"]``.
            Empty lets the model detect the language.
        api_key: Defaults to ``META_API_KEY``, then ``MODEL_API_KEY``.
        connect: Replaces ``websockets.connect``; for tests.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        language_bias: Sequence[str] = (),
        api_key: str | None = None,
        url: str = WS_URL,
        connect: Any = None,
    ) -> None:
        self.name = NAME
        self.model = model
        self.language_bias = list(language_bias)
        self._api_key = api_key
        self._url = url
        self._connect = connect
        self.usage = Usage(audio_seconds=0.0)
        """The last turn's billed audio seconds."""

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
        key = require_key(self._api_key, *API_KEY_ENVS, provider=NAME, model=self.model)
        handshake: dict[str, Any] = {
            "authorization": {"accessToken": f"Bearer {key}"},
            "audioEncoding": ENCODINGS[fmt.sample_rate_hz],
            "model": self.model,
            "mode": "PUSH_TO_TALK",
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": False,
        }
        if self.language_bias:
            handshake["languageBias"] = self.language_bias
        connect = self._connect or _websockets().connect
        try:
            ws = await connect(self._url)
        except Exception as exc:
            raise _socket_failure(exc, self.model) from exc

        async def send() -> None:
            step = fmt.frame_bytes(CHUNK_MS)
            for start in range(0, len(pcm), step):
                await ws.send(pcm[start : start + step])
                await asyncio.sleep(CHUNK_MS / 1000)
            await ws.send(json.dumps({"type": "endStream"}))

        sender: asyncio.Task[None] | None = None
        final: str | None = None
        try:
            await ws.send(json.dumps(handshake))
            ack = json.loads(await ws.recv())
            if ack.get("type") == "error":
                raise _error_event(ack, self.model)
            if "sessionId" not in ack:
                raise ProviderError(
                    ProviderFailure.MALFORMED,
                    f"unexpected handshake reply: {str(ack)[:300]}",
                    provider=NAME,
                    model=self.model,
                )
            sender = asyncio.create_task(send())
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                event = json.loads(raw)
                kind = event.get("type")
                if kind == "transcript":
                    if event.get("final"):
                        final = event.get("transcript", "")
                        break
                    yield Transcript(text=event.get("transcript", ""), is_final=False)
                elif kind == "error":
                    raise _error_event(event, self.model)
            else:
                raise ProviderError(
                    ProviderFailure.STREAM_BROKEN,
                    "socket closed before the final transcript",
                    provider=NAME,
                    model=self.model,
                )
            await sender
        except ProviderError:
            raise
        except Exception as exc:
            raise _socket_failure(exc, self.model) from exc
        finally:
            if sender is not None:
                sender.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await sender
            with contextlib.suppress(Exception):  # keep the error that got us here
                await ws.close()
        self.usage = Usage(audio_seconds=float(math.floor(duration)))
        yield Transcript(text=final.strip(), is_final=True, audio_duration_s=duration)


def _sendable(pcm: bytes, fmt: AudioFormat) -> tuple[bytes, AudioFormat]:
    """16-bit mono at 24 or 16 kHz, the two rates the endpoint takes."""
    if fmt.channels != 1 or fmt.encoding != "pcm16":
        raise ValueError(f"{NAME}: pcm16 mono audio only, got {fmt}")
    if fmt.sample_rate_hz in ENCODINGS:
        return pcm, fmt
    samples = dsp.resample_clean(dsp.to_array(pcm), fmt.sample_rate_hz, 24_000)
    return dsp.to_pcm(_numpy().asarray(samples)), CANONICAL


def _error_event(event: dict[str, Any], model: str) -> ProviderError:
    kind = _ERROR_TYPES.get(str(event.get("errorType")), ProviderFailure.MALFORMED)
    message = f"{event.get('message', '')} (session {event.get('sessionId', '')})"
    return ProviderError(kind, message[:300], provider=NAME, model=model)


def _socket_failure(exc: Exception, model: str) -> ProviderError:
    return socket_failure(exc, provider=NAME, model=model, close_codes=_CLOSE_CODES)


META_STT_SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.STT,
    capabilities=ProviderCapabilities(
        modality=Modality.STT,
        streams=True,
        emits_partials=True,
        word_timings=False,
        diarizes=False,
    ),
    credential_provider="meta",
    default_model=DEFAULT_MODEL,
)
"""Partials are the non-final ``transcript`` events. Meta documents turn-level
timestamps only, and diarization is a different mode from push-to-talk."""


def build_stt(*, model: str = DEFAULT_MODEL, **config: Any) -> MetaSTT:
    return MetaSTT(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((META_STT_SPEC, build_stt),)
