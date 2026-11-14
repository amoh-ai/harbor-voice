"""Fakes shared by the provider tests: a websocket and the ways one fails."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from typing import Self

import httpx


class FakeSocket:
    """A websocket that hands out queued messages and records what it is sent.

    A ``None`` in the queue ends iteration, or raises ``close_with`` when set.
    A test for one vendor subclasses it to answer the way that server does.
    """

    def __init__(self, *messages: Any, close_with: Exception | None = None) -> None:
        self.inbox: asyncio.Queue[str | bytes | None] = asyncio.Queue()
        self.sent: list[Any] = []
        self.closed = False
        self.close_with = close_with
        for message in messages:
            self.push(message)

    def push(self, message: Any) -> None:
        """Queue a message: bytes, text and ``None`` as they are, the rest as JSON."""
        if message is not None and not isinstance(message, (bytes, str)):
            message = json.dumps(message)
        self.inbox.put_nowait(message)

    async def send(self, data: str | bytes) -> None:
        """Record text as parsed JSON and audio as bytes."""
        self.sent.append(data if isinstance(data, bytes) else json.loads(data))

    async def recv(self) -> Any:
        raw = await self.inbox.get()
        assert raw is not None
        return raw

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Any:
        raw = await self.inbox.get()
        if raw is None:
            if self.close_with is not None:
                raise self.close_with
            raise StopAsyncIteration
        return raw

    async def close(self) -> None:
        self.closed = True
        self.inbox.put_nowait(None)


class Closed(Exception):
    """Stands in for ``websockets.ConnectionClosedError``: a close code."""

    def __init__(self, code: int) -> None:
        super().__init__(f"closed {code}")
        self.rcvd = type("Close", (), {"code": code})()


class Rejected(Exception):
    """Stands in for ``websockets.InvalidStatus``: a refused handshake."""

    def __init__(self, status: int) -> None:
        super().__init__(f"server rejected WebSocket connection: HTTP {status}")
        self.response = httpx.Response(status)


def refusing(exc: Exception) -> Any:
    """A ``connect`` that fails with ``exc`` before any socket exists."""

    async def connect(url: str, **_: Any) -> Any:
        raise exc

    return connect


class FakeDeepgram(FakeSocket):
    """Deepgram's ``/v1/listen`` for a whole call, enough to hear a tone.

    Takes 16 kHz PCM16 in 50 ms messages, as :class:`DeepgramCallStream`
    sends it. A loud message after quiet is ``SpeechStarted`` and an interim
    ``Results`` (the transcript's first word); ``endpointing_ms`` of quiet
    after speech is a final ``Results`` with ``speech_final``, carrying the
    next of ``transcripts`` (empty once they run out).
    """

    def __init__(
        self,
        transcripts: list[str] | None = None,
        *,
        endpointing_ms: int = 350,
        close_with: Exception | None = None,
    ) -> None:
        super().__init__(close_with=close_with)
        self.transcripts = list(transcripts or [])
        self.endpointing_ms = endpointing_ms
        self.url = ""
        self.headers: dict[str, str] = {}
        self._ms = 0
        self._speaking = False
        self._quiet_ms = 0
        self._started_ms = 0

    def connect(self) -> Any:
        """A ``websockets.connect`` that hands out this socket."""

        async def connect(url: str, **kwargs: Any) -> Any:
            self.url = url
            self.headers = dict(kwargs.get("additional_headers") or {})
            return self

        connect.socket = self  # ty: ignore[unresolved-attribute]
        return connect

    @property
    def audio(self) -> list[bytes]:
        return [m for m in self.sent if isinstance(m, bytes)]

    @property
    def controls(self) -> list[dict[str, Any]]:
        return [m for m in self.sent if not isinstance(m, bytes)]

    async def send(self, data: str | bytes) -> None:
        await super().send(data)
        if isinstance(data, bytes):
            self._hear(data)

    def results(self, text: str, *, final: bool, speech_final: bool) -> dict:
        start = self._started_ms / 1000
        return {
            "type": "Results",
            "is_final": final,
            "speech_final": speech_final,
            "start": start,
            "duration": self._ms / 1000 - start,
            "channel": {"alternatives": [{"transcript": text, "confidence": 0.9}]},
            "metadata": {"request_id": "r1"},
        }

    def _hear(self, pcm: bytes) -> None:
        samples = memoryview(pcm).cast("h")
        loud = max((abs(x) for x in samples), default=0) > 1000
        ms = len(samples) * 1000 // 16_000
        if loud and not self._speaking:
            self._speaking, self._started_ms = True, self._ms
            self.push({"type": "SpeechStarted", "timestamp": self._ms / 1000})
            first = (self.transcripts[0] if self.transcripts else "").split(" ")[0]
            self._ms += ms
            if first:
                self.push(self.results(first, final=False, speech_final=False))
            self._quiet_ms = 0
            return
        self._ms += ms
        if not self._speaking:
            return
        self._quiet_ms = 0 if loud else self._quiet_ms + ms
        if self._quiet_ms >= self.endpointing_ms:
            self._speaking = False
            text = self.transcripts.pop(0) if self.transcripts else ""
            self.push(self.results(text, final=True, speech_final=True))
