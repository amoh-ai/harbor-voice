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

:class:`DeepgramCallStream` (from :meth:`DeepgramSTT.listen`) is the other
way to use the slot: one connection for a whole call, which finds where the
caller's speech starts and ends as well as what was said. It is how tau2's
cascaded agent listens, and how a duplex cascade uses this slot. See its
docstring.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlencode

from harbor_voice._deps import websockets as _websockets
from harbor_voice.audio.format import AudioFormat
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

    def listen(
        self, fmt: AudioFormat, settings: ListenSettings | None = None
    ) -> DeepgramCallStream:
        """A connection for a whole call, with this slot's model and key.

        Not opened yet: call :meth:`DeepgramCallStream.open`.
        """
        return DeepgramCallStream(
            model=self.model,
            fmt=fmt,
            settings=settings or TAU2_LISTENING,
            api_key=self._api_key,
            url=self._url,
            connect=self._connect,
        )


# ------------------------------------------------------------ a whole call


@dataclass(frozen=True, slots=True)
class ListenSettings:
    """How a whole-call stream listens. The defaults are tau2's.

    They are tau2's ``DeepgramSTTConfig`` (``tau2/voice/audio_native/livekit/
    config.py`` at ``2174a60``), plus the defaults LiveKit's Deepgram plugin
    (``livekit-plugins-deepgram`` 1.5.1, the version tau2 locks) sends for
    the options tau2 does not set: ``no_delay``, ``filler_words``,
    ``profanity_filter``, ``numerals`` and ``mip_opt_out``.
    """

    language: str = "en-US"
    endpointing_ms: int = 350
    """Silence after speech before Deepgram calls the speech final. ``0``
    sends ``endpointing=false``."""
    utterance_end_ms: int = 2000
    """Not sent to Deepgram, as tau2 does not send it. tau2's own fallback:
    a final transcript left waiting this long with no further transcript
    starts the reply anyway. The duplex cascade applies it."""
    interim_results: bool = True
    vad_events: bool = True
    smart_format: bool = False
    punctuate: bool = True
    no_delay: bool = True
    filler_words: bool = True
    profanity_filter: bool = False
    numerals: bool = False
    mip_opt_out: bool = False


TAU2_LISTENING = ListenSettings()

KEEPALIVE_S = 5.0
"""Seconds between ``KeepAlive`` messages, as LiveKit's plugin sends them.
Deepgram closes a stream that receives nothing for about 10 s."""

CHUNK_MS = 50
"""Audio goes up in 50 ms messages, as LiveKit's plugin sends it."""


class Heard(StrEnum):
    """What a whole-call stream reports, in LiveKit's terms."""

    SPEECH_STARTED = "speech_started"
    """``START_OF_SPEECH``."""
    INTERIM = "interim"
    """``INTERIM_TRANSCRIPT``: a tentative transcript of the current segment."""
    FINAL = "final"
    """``FINAL_TRANSCRIPT``: a segment's transcript, which will not change."""
    SPEECH_ENDED = "speech_ended"
    """``END_OF_SPEECH``."""


@dataclass(frozen=True, slots=True)
class ListenEvent:
    kind: Heard
    text: str = ""
    audio_ms: int | None = None
    """Where in the audio sent it happened, as Deepgram counts it: a
    ``SpeechStarted`` timestamp, or the end of a ``Results`` segment."""


class DeepgramCallStream:
    """One ``/v1/listen`` connection for a whole call: turns and transcripts.

    This is how tau2's cascaded agent hears the caller (LiveKit's Deepgram
    plugin, which tau2's ``CascadedVoiceProvider`` streams every tick into):

    - the query is :class:`ListenSettings` (tau2's settings) with the slot's
      model, ``encoding`` and ``sample_rate`` of the audio sent, and
      ``channels=1``;
    - audio goes up in 50 ms binary messages; a ``KeepAlive`` goes up as the
      stream opens and every 5 s after, so a pause in the audio does not end
      the stream;
    - messages map to :class:`ListenEvent` exactly as the plugin maps them:
      ``SpeechStarted`` starts speech unless it already started; a
      ``Results`` with text starts speech if nothing did, and is an
      :attr:`Heard.INTERIM` or (``is_final``) :attr:`Heard.FINAL`
      transcript; ``speech_final`` then ends speech, if it started. Deepgram
      has no "speech ended" message: ``speech_final`` is it.

    The stream does not reconnect. LiveKit's plugin retries a dropped
    connection; here a dropped stream ends the call (see
    :class:`~harbor_voice.cascade.duplex.DuplexCascadeTransport`), so a call
    with a hole in its hearing is not scored.

    Args:
        model: A ``/v1/listen`` model.
        fmt: Of the audio sent: mono, PCM16, μ-law or A-law.
        settings: Everything else in the query.
        keepalive_s: Seconds between ``KeepAlive`` messages.
    """

    def __init__(
        self,
        *,
        model: str,
        fmt: AudioFormat,
        settings: ListenSettings = TAU2_LISTENING,
        api_key: str | None = None,
        url: str = URL,
        connect: Any = None,
        keepalive_s: float = KEEPALIVE_S,
    ) -> None:
        encoding(fmt)
        self.model = model
        self.fmt = fmt
        self.settings = settings
        self._api_key = api_key
        self._url = url
        self._connect = connect
        self._keepalive_s = keepalive_s
        self._chunk = fmt.frame_bytes(CHUNK_MS)
        self._pending = b""
        self._ws: Any = None
        self._events: asyncio.Queue[ListenEvent | None] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._speaking = False
        self._closing = False
        self.audio_seconds = 0.0
        """Audio sent so far, which is what Deepgram bills."""
        self.failure: ProviderError | None = None
        """Why the stream ended, if the server or the network ended it."""

    def query(self) -> dict[str, str | int]:
        """The ``/v1/listen`` query, booleans lowercased as LiveKit sends them."""
        s = self.settings
        options: dict[str, str | int | bool] = {
            "model": self.model,
            "punctuate": s.punctuate,
            "smart_format": s.smart_format,
            "no_delay": s.no_delay,
            "interim_results": s.interim_results,
            "encoding": encoding(self.fmt),
            "vad_events": s.vad_events,
            "sample_rate": self.fmt.sample_rate_hz,
            "channels": 1,
            "endpointing": s.endpointing_ms or False,
            "filler_words": s.filler_words,
            "profanity_filter": s.profanity_filter,
            "numerals": s.numerals,
            "mip_opt_out": s.mip_opt_out,
            "language": s.language,
        }
        return {
            k: str(v).lower() if isinstance(v, bool) else v for k, v in options.items()
        }

    async def open(self) -> None:
        """Connect, and start receiving and keeping the stream alive."""
        headers = auth_header(self._api_key, provider=NAME, model=self.model)
        connect = self._connect or _websockets().connect
        try:
            self._ws = await connect(
                f"{self._url}?{urlencode(self.query())}", additional_headers=headers
            )
        except Exception as exc:
            raise socket_failure(exc, provider=NAME, model=self.model) from exc
        self._tasks = [
            asyncio.create_task(self._receive()),
            asyncio.create_task(self._keep_alive()),
        ]

    async def send(self, pcm: bytes) -> None:
        """Audio from the call, sent on in 50 ms messages.

        Raises the stream's failure once it has one, so a dead stream is not
        fed in silence.
        """
        if self.failure is not None:
            raise self.failure
        if self._ws is None:
            raise RuntimeError(f"{NAME}: the stream is not open")
        self.audio_seconds += self.fmt.duration_s(len(pcm))
        self._pending += pcm
        try:
            while len(self._pending) >= self._chunk:
                chunk, self._pending = (
                    self._pending[: self._chunk],
                    self._pending[self._chunk :],
                )
                await self._ws.send(chunk)
        except Exception as exc:
            raise self._fail(exc) from exc

    async def events(self) -> AsyncIterator[ListenEvent]:
        """What was heard, in order; raises the failure that ended the stream."""
        while (event := await self._events.get()) is not None:
            yield event
        if self.failure is not None:
            raise self.failure

    async def close(self) -> None:
        """``CloseStream``, then the connection. Safe to call more than once."""
        if self._ws is None:
            return
        self._closing = True
        with contextlib.suppress(Exception):
            await self._ws.send(json.dumps({"type": "CloseStream"}))
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        with contextlib.suppress(Exception):
            await self._ws.close()
        self._ws = None
        self._events.put_nowait(None)

    # ----------------------------------------------------------- internals

    def _fail(self, exc: Exception) -> ProviderError:
        if self.failure is None:
            self.failure = (
                exc
                if isinstance(exc, ProviderError)
                else socket_failure(exc, provider=NAME, model=self.model)
            )
        return self.failure

    async def _keep_alive(self) -> None:
        """LiveKit's ``keepalive_task``: one now, then one every 5 s."""
        while self._ws is not None:
            with contextlib.suppress(Exception):  # a dead socket fails elsewhere
                await self._ws.send(json.dumps({"type": "KeepAlive"}))
            await asyncio.sleep(self._keepalive_s)

    async def _receive(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, str):
                    self._dispatch(json.loads(raw))
            if not self._closing:
                raise ProviderError(
                    ProviderFailure.STREAM_BROKEN,
                    "the server closed the stream during the call",
                    provider=NAME,
                    model=self.model,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - raised from events() and send()
            if not self._closing:
                self._fail(exc)
        finally:
            self._events.put_nowait(None)

    def _dispatch(self, data: dict[str, Any]) -> None:
        """LiveKit's ``_process_stream_event``."""
        kind = data.get("type")
        if kind == "SpeechStarted":
            if not self._speaking:
                self._speaking = True
                self._emit(Heard.SPEECH_STARTED, audio_ms=_ms(data.get("timestamp")))
        elif kind == "Results":
            best = ((data.get("channel") or {}).get("alternatives") or [{}])[0]
            text = best.get("transcript") or ""
            start, duration = data.get("start"), data.get("duration")
            end_ms = (
                _ms(float(start) + float(duration))
                if start is not None and duration is not None
                else None
            )
            if text:
                if not self._speaking:
                    self._speaking = True
                    self._emit(Heard.SPEECH_STARTED, audio_ms=_ms(start))
                final = bool(data.get("is_final"))
                self._emit(Heard.FINAL if final else Heard.INTERIM, text, end_ms)
            if data.get("speech_final") and self._speaking:
                self._speaking = False
                self._emit(Heard.SPEECH_ENDED, audio_ms=end_ms)
        elif kind == "Error":
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"{data.get('variant')}: {data.get('description')}",
                provider=NAME,
                model=self.model,
            )
        # Metadata, UtteranceEnd (never asked for): nothing to report.

    def _emit(self, kind: Heard, text: str = "", audio_ms: int | None = None) -> None:
        self._events.put_nowait(ListenEvent(kind, text, audio_ms))


def _ms(seconds: Any) -> int | None:
    return None if seconds is None else round(float(seconds) * 1000)


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
