"""Cartesia speech to text: Ink, over the manual-finalize websocket.

``/stt/websocket`` rather than ``/stt/turns/websocket``: the cascade has
already decided the turn is over when it hands audio here, so Cartesia's own
turn detection has nothing left to decide. Batch ``/stt`` serves only the
older ``ink-whisper`` (docs.cartesia.ai/use-the-api/stt/compare-endpoints).

The session, as the API reference and Cartesia's own example run it:

- connect to ``wss://api.cartesia.ai/stt/websocket?model=…&encoding=…&
  sample_rate=…&cartesia_version=2026-08-14`` with ``X-API-Key``;
- send the audio as binary messages of 100 ms, paced to real time. The
  docs are explicit that realtime endpoints take about one second of audio
  per second, and that pushing a batch at once can surface as a server error.
  So a turn takes at least its own length to transcribe;
- send ``finalize``, then ``close``, and read until ``done``.

``transcript`` messages carry deltas. Each ``is_final`` delta extends the
committed text; a non-final one is a guess at what follows it. Both are
yielded as partials of the whole turn, then the committed text once, as the
final. Ink bills per second of audio sent, silence included.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import urllib.parse
from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import Any

from harbor_voice._deps import websockets as _websockets
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.cartesia.tts import API_VERSION
from harbor_voice.providers.cartesia.tts import ENCODINGS
from harbor_voice.providers.cartesia.tts import api_key
from harbor_voice.providers.cartesia.tts import error
from harbor_voice.providers.protocols import Transcript

NAME = "cartesia.stt"
DEFAULT_MODEL = "ink-2"
WS_URL = "wss://api.cartesia.ai/stt/websocket"
CHUNK_MS = 100
"""The API reference's suggested chunk, sent once per ``CHUNK_MS``."""


class CartesiaSTT:
    """One websocket session per turn.

    Args:
        model: Cartesia STT model id.
        api_key: Defaults to ``CARTESIA_API_KEY``.
        connect: Replaces ``websockets.connect``; for tests.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        url: str = WS_URL,
        connect: Any = None,
    ) -> None:
        self.name = NAME
        self.model = model
        self._api_key = api_key
        self._url = url
        self._connect = connect
        self.usage = Usage(audio_seconds=0.0)
        """The last turn's audio seconds, the unit Cartesia bills."""

    async def transcribe(
        self, frames: Sequence[AudioFrame]
    ) -> AsyncIterator[Transcript]:
        """Yield partials while the audio is sent, then one final."""
        pcm = b"".join(f.pcm for f in frames)
        if not pcm:
            yield Transcript(text="", is_final=True, audio_duration_s=0.0)
            return
        fmt = frames[0].fmt
        if fmt.channels != 1:
            raise ValueError(f"{NAME}: mono audio only, got {fmt}")
        duration = fmt.duration_s(len(pcm))
        query = urllib.parse.urlencode(
            {
                "model": self.model,
                "encoding": ENCODINGS[fmt.encoding],
                "sample_rate": fmt.sample_rate_hz,
                "cartesia_version": API_VERSION,
            }
        )
        headers = {"X-API-Key": api_key(self._api_key, NAME, self.model)}
        connect = self._connect or _websockets().connect
        try:
            ws = await connect(f"{self._url}?{query}", additional_headers=headers)
        except Exception as exc:
            raise _classify(exc, self.model) from exc

        async def send() -> None:
            step = fmt.frame_bytes(CHUNK_MS)
            for start in range(0, len(pcm), step):
                await ws.send(pcm[start : start + step])
                await asyncio.sleep(CHUNK_MS / 1000)
            await ws.send("finalize")
            await ws.send("close")

        sender = asyncio.create_task(send())
        committed = ""
        try:
            async for raw in ws:
                event = json.loads(raw)
                kind = event.get("type")
                if kind == "transcript":
                    text = event.get("text", "").strip()
                    if event.get("is_final"):
                        committed = f"{committed} {text}".strip()
                        yield Transcript(text=committed, is_final=False)
                    else:
                        guess = f"{committed} {text}".strip()
                        yield Transcript(text=guess, is_final=False)
                elif kind == "error":
                    raise error(
                        int(event.get("status_code", 0)),
                        str(event.get("message", event)),
                        NAME,
                        self.model,
                    )
                elif kind == "done":
                    break
            else:
                raise ProviderError(
                    ProviderFailure.STREAM_BROKEN,
                    "socket closed before 'done'",
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
        self.usage = Usage(audio_seconds=duration)
        yield Transcript(
            text=committed.strip(), is_final=True, audio_duration_s=duration
        )


def _classify(exc: Exception, model: str) -> ProviderError:
    """A failed handshake by its HTTP status; anything else by its type."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return error(status, str(exc), NAME, model)
    if isinstance(exc, (TimeoutError, OSError)):
        kind = ProviderFailure.TIMEOUT
    elif isinstance(exc, json.JSONDecodeError):
        kind = ProviderFailure.MALFORMED
    else:
        kind = ProviderFailure.STREAM_BROKEN
    return ProviderError(
        kind, f"{type(exc).__name__}: {str(exc)[:300]}", provider=NAME, model=model
    )


CARTESIA_STT_SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.STT,
    capabilities=ProviderCapabilities(
        modality=Modality.STT,
        streams=True,
        emits_partials=True,
        word_timings=False,
        diarizes=False,
    ),
    credential_provider="cartesia",
    requires_extra="voice",
    default_model=DEFAULT_MODEL,
)
"""Partials are what the API sends as audio arrives; Ink's word timestamps
are not surfaced, because :class:`Transcript` has nowhere to put them."""


def build_stt(*, model: str = DEFAULT_MODEL, **config: Any) -> CartesiaSTT:
    return CartesiaSTT(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((CARTESIA_STT_SPEC, build_stt),)
