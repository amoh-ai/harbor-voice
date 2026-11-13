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
