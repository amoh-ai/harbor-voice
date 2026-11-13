"""Deepgram streaming speech to text: Nova-3 over the live websocket.

One utterance per connection to ``wss://api.deepgram.com/v1/listen``
(https://developers.deepgram.com/reference/speech-to-text/listen-streaming):

- the frames go up as binary messages, then ``{"type": "CloseStream"}``,
  which makes the server finish the audio, send its last results and a
  ``Metadata`` summary, and close (https://developers.deepgram.com/docs/close-stream);
- ``interim_results=true``: each ``Results`` with ``is_final=false`` is a
  partial, yielded as the finished segments so far plus the tentative one;
  ``is_final=true`` segments join into the final transcript;
- ``smart_format=true``, as Deepgram's own streaming examples set it.

Nova-3, not Flux. Deepgram recommends Flux for voice agents because its
model decides when the speaker's turn ended
(https://developers.deepgram.com/docs/models-languages-overview). In this slot
the cascade has already decided that: it hands over one finished turn. What
is left is plain recognition, which is what Nova-3 is documented for, on the
``/v1/listen`` API with interim results. Flux lives on ``/v2/listen`` with a
different message set, and is what ``deepgram.agent`` listens with.

Word timings come back (``words[].start``/``end``), but
:class:`~harbor_voice.providers.protocols.Transcript` has no field for them,
so the spec does not claim ``word_timings``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlencode

from harbor_voice._deps import websockets as _websockets
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import socket_failure
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.deepgram._common import auth_header
from harbor_voice.providers.deepgram._common import encoding
from harbor_voice.providers.protocols import Transcript

NAME = "deepgram.stt"
DEFAULT_MODEL = "nova-3"
URL = "wss://api.deepgram.com/v1/listen"


class DeepgramSTT:
    """One finished turn in, partials then one final transcript out.

    Args:
        model: A ``/v1/listen`` model, ``nova-3`` by default.
        language: Sent as ``language``; Deepgram's default is ``en``.
        api_key: Defaults to ``DEEPGRAM_API_KEY``.
        timeout_s: Longest wait for any one server message.
        connect: ``websockets.connect`` unless given (tests pass a fake).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        language: str = "en",
        api_key: str | None = None,
        url: str = URL,
        timeout_s: float = 30.0,
        connect: Any = None,
    ) -> None:
        self.name = NAME
        self.model = model
        self.language = language
        self._api_key = api_key
        self._url = url
        self._timeout_s = timeout_s
        self._connect = connect

    async def transcribe(
        self, frames: Sequence[AudioFrame]
    ) -> AsyncIterator[Transcript]:
        if not frames:
            yield Transcript(text="", is_final=True, audio_duration_s=0.0)
            return
        fmt = frames[0].fmt
        if any(frame.fmt != fmt for frame in frames):
            raise ValueError(f"{NAME}: one utterance must be in one format")
        query = urlencode(
            {
                "model": self.model,
                "language": self.language,
                "encoding": encoding(fmt),
                "sample_rate": fmt.sample_rate_hz,
                "channels": 1,
                "interim_results": "true",
                "smart_format": "true",
            }
        )
        headers = auth_header(self._api_key, provider=NAME, model=self.model)
        connect = self._connect or _websockets().connect
        try:
            ws = await connect(f"{self._url}?{query}", additional_headers=headers)
        except Exception as exc:
            raise socket_failure(exc, provider=NAME, model=self.model) from exc

        async def send() -> None:
            for frame in frames:
                await ws.send(frame.pcm)
            await ws.send(json.dumps({"type": "CloseStream"}))

        sender = asyncio.create_task(send())
        finals: list[str] = []
        confidences: list[float] = []
        duration = sum(frame.duration_s for frame in frames)
        try:
            messages = aiter(ws)
            while True:
                try:
                    raw = await asyncio.wait_for(anext(messages), self._timeout_s)
                except StopAsyncIteration:
                    break
                data = json.loads(raw)
                kind = data.get("type")
                if kind == "Results":
                    best = ((data.get("channel") or {}).get("alternatives") or [{}])[0]
                    text = (best.get("transcript") or "").strip()
                    if data.get("is_final"):
                        if text:
                            finals.append(text)
                            if best.get("confidence") is not None:
                                confidences.append(float(best["confidence"]))
                    elif text:
                        yield Transcript(text=" ".join([*finals, text]), is_final=False)
                elif kind == "Metadata":
                    duration = float(data.get("duration") or duration)
                elif kind == "Error":
                    raise ProviderError(
                        ProviderFailure.MALFORMED,
                        f"{data.get('variant')}: {data.get('description')}",
                        provider=NAME,
                        model=self.model,
                    )
            await sender
        except ProviderError:
            raise
        except Exception as exc:
            raise socket_failure(exc, provider=NAME, model=self.model) from exc
        finally:
            sender.cancel()
            with contextlib.suppress(BaseException):
                await sender
            with contextlib.suppress(Exception):
                await ws.close()
        yield Transcript(
            text=" ".join(finals),
            is_final=True,
            confidence=sum(confidences) / len(confidences) if confidences else None,
            audio_duration_s=duration,
        )


SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.STT,
    capabilities=ProviderCapabilities(
        modality=Modality.STT,
        streams=True,
        emits_partials=True,
        word_timings=False,
        diarizes=False,
    ),
    credential_provider="deepgram",
    default_model=DEFAULT_MODEL,
)
"""``streams`` and ``emits_partials``: interim results are on. Billed per
second of audio, which each final :class:`Transcript` carries as
``audio_duration_s`` (the server's ``Metadata.duration`` when it sent one)."""


def build(*, model: str = DEFAULT_MODEL, **config: Any) -> DeepgramSTT:
    return DeepgramSTT(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
