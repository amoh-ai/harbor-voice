"""The two outside services an EVA trial talks to, faked on localhost.

Both are real servers on real sockets, so the trial's own clients are the
ones under test: harbor's ``openai.realtime`` transport connects to
:class:`FakeRealtime` by URL, and harbor's ``elevenlabs.agents`` transport
and REST client (signed URL, conversation record, server audio, the agent's
config) reach :class:`FakeElevenLabs` through its ``api_base``.

- :class:`FakeRealtime` hears what it is sent: an energy VAD over each
  appended frame (server VAD's 200 ms of silence), a transcript of each
  caller turn, and scripted responses that stream audio and transcript
  faster than real time; one ends in a tool call and the next answers after
  the tool's output. Modelled on the call golden's agent (``adapters/eva/scripts/goldens/call_scenario.py``).
- :class:`FakeElevenLabs` is the caller: on a timeline from the socket's
  start it sends the caller's speech (16 kHz PCM, in bursts, as ElevenLabs
  does), the ``agent_response`` cue, its transcript of the agent, and then
  closes the socket as after ``end_call``. Whether the conversation record
  then shows ``end_call`` is set per conversation: without it the caller's
  end reason is ``session_ended``, which EVA's gate rejects, so the attempt
  is rerun.

The call is about 21 s (see :data:`UTTERANCES` for why the gaps are wide).
Nothing here reaches the network.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import sys
import time
import urllib.parse
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from websockets.asyncio.server import Server
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve
from websockets.http11 import Request
from websockets.http11 import Response

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[5] / "adapters" / "eva" / "scripts" / "goldens"
    ),
)

from call_audio_cases import speech_like  # noqa: E402

AGENT_RATE = 24_000
CALLER_RATE = 16_000
DELTA_BYTES = 4_800
"""100 ms of 24 kHz PCM per audio delta."""
VAD_RMS = 500.0
VAD_SILENCE_MS = 200
TRANSCRIPTION_DELAY_S = 0.25

TOOL_NAME = "get_reservation"
TOOL_ARGUMENTS = {"confirmation_number": "ABC123", "last_name": "Smith"}


@dataclass(frozen=True)
class Reply:
    """One agent response."""

    text: str
    seconds: float
    latency_s: float = 0.3
    interval_s: float = 0.05
    """Between 100 ms deltas: twice real time."""
    tool: tuple[str, dict[str, Any]] | None = None


REPLIES: tuple[Reply, ...] = (
    Reply("Hello! How can I help you today?", 1.5),
    Reply("Let me look that up.", 1.0, tool=(TOOL_NAME, TOOL_ARGUMENTS)),
    Reply("I found your reservation. Is there anything else I can help with?", 2.0),
    Reply("You're welcome. Goodbye!", 1.0),
)
HEARD: tuple[str, ...] = (
    "Hi, I need to check my reservation. Confirmation ABC123, last name Smith.",
    "No, that's all. Thanks, goodbye.",
)
"""The agent's transcription of each caller turn."""


@dataclass(frozen=True)
class Utterance:
    text: str
    start_s: float
    seconds: float
    seed: int


UTTERANCES: tuple[Utterance, ...] = (
    Utterance(HEARD[0], 5.0, 1.6, seed=31),
    Utterance(HEARD[1], 15.0, 1.2, seed=32),
)
"""Each starts at least ~1 s after the agent's audio before it has played out
even if 20 ms pacing runs at 0.6x (a throttled Windows timer), so the caller
never speaks before EVA's bridge has registered the agent's end. If it does,
EVA's bridge (``audio_bridge.py:482-485``, ported as is) clears the agent's
end time while the agent stays "speaking": its end is never detected, no
silence reaches the agent, and the call stalls until the caller hangs up."""
TRANSCRIPTS: tuple[tuple[float, str], ...] = (
    (4.0, "Hello, how can I help you today?"),
    (14.0, "Let me look that up. I found your reservation. Is there anything else?"),
    (19.8, "You're welcome. Goodbye."),
)
"""ElevenLabs' transcript of the agent (``user_transcript``)."""
END_S = 20.5
CHUNK_BYTES = 3_200
"""100 ms of 16 kHz PCM per ``audio`` message, sent every 50 ms."""


def _rms(pcm: bytes) -> float:
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


# ------------------------------------------------------------- the agent


class _Agent:
    """One Realtime session's server side."""

    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self.emit = emit
        self.next = 0
        self.speaking = False
        self.silence_ms = 0.0
        self.audio_ms = 0.0
        self.turns = 0
        self.active: asyncio.Task[None] | None = None
        self.tasks: set[asyncio.Task[None]] = set()
        self.client_events: list[str] = []
        self.tool_outputs: list[dict[str, Any]] = []
        self.session: dict[str, Any] | None = None

    def spawn(self, coro: Any) -> asyncio.Task[None]:
        task = asyncio.get_running_loop().create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def close(self) -> None:
        for task in list(self.tasks):
            task.cancel()

    def on_client(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type"))
        if kind != "input_audio_buffer.append":
            self.client_events.append(kind)
        if kind == "session.update":
            self.session = event.get("session")
            self.emit({"type": "session.updated", "session": self.session})
        elif kind == "response.create":
            self.respond()
        elif kind == "conversation.item.create":
            item = event.get("item") or {}
            if item.get("type") == "function_call_output":
                self.tool_outputs.append(item)
        elif kind == "input_audio_buffer.append":
            self.hear(base64.b64decode(event.get("audio") or ""))

    def hear(self, pcm: bytes) -> None:
        frame_ms = len(pcm) / 2 / AGENT_RATE * 1000
        self.audio_ms += frame_ms
        voiced = _rms(pcm) > VAD_RMS
        if not self.speaking:
            if voiced:
                self.speaking = True
                self.silence_ms = 0.0
                self.emit(
                    {
                        "type": "input_audio_buffer.speech_started",
                        "audio_start_ms": int(self.audio_ms),
                        "item_id": f"in_{self.turns}",
                    }
                )
                if self.active is not None and not self.active.done():
                    self.active.cancel()
            return
        if voiced:
            self.silence_ms = 0.0
            return
        self.silence_ms += frame_ms
        if self.silence_ms >= VAD_SILENCE_MS:
            self.speaking = False
            turn = self.turns
            self.turns += 1
            self.emit(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "audio_end_ms": int(self.audio_ms - self.silence_ms),
                    "item_id": f"in_{turn}",
                }
            )
            self.spawn(self.transcribe(turn))
            self.respond()

    async def transcribe(self, turn: int) -> None:
        await asyncio.sleep(TRANSCRIPTION_DELAY_S)
        self.emit(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": f"in_{turn}",
                "content_index": 0,
                "transcript": HEARD[turn] if turn < len(HEARD) else "",
            }
        )

    def respond(self) -> None:
        index = self.next
        self.next += 1
        if index < len(REPLIES):
            self.active = self.spawn(self.speak(index))

    async def speak(self, index: int) -> None:
        reply = REPLIES[index]
        rid, item = f"resp_{index}", f"item_{index}"
        sent: list[str] = []
        streaming = False
        try:
            await asyncio.sleep(reply.latency_s)
            streaming = True
            self.emit({"type": "response.created", "response": {"id": rid}})
            pcm = speech_like(
                int(reply.seconds * AGENT_RATE), AGENT_RATE, seed=50 + index
            )
            words = reply.text.split(" ")
            n = math.ceil(len(pcm) / DELTA_BYTES)
            for i in range(n):
                self.emit(
                    {
                        "type": "response.output_audio.delta",
                        "response_id": rid,
                        "item_id": item,
                        "delta": base64.b64encode(
                            pcm[i * DELTA_BYTES : (i + 1) * DELTA_BYTES]
                        ).decode("ascii"),
                    }
                )
                lo, hi = (i * len(words)) // n, ((i + 1) * len(words)) // n
                if hi > lo:
                    delta = (" " if lo else "") + " ".join(words[lo:hi])
                    sent.append(delta)
                    self.emit(
                        {
                            "type": "response.output_audio_transcript.delta",
                            "response_id": rid,
                            "item_id": item,
                            "delta": delta,
                        }
                    )
                await asyncio.sleep(reply.interval_s)
            streaming = False
            self.emit(
                {
                    "type": "response.output_audio_transcript.done",
                    "item_id": item,
                    "transcript": reply.text,
                }
            )
            output: list[dict[str, Any]] = [
                {
                    "type": "message",
                    "id": item,
                    "content": [{"type": "output_audio", "transcript": reply.text}],
                }
            ]
            if reply.tool is not None:
                name, arguments = reply.tool
                call = {
                    "type": "function_call",
                    "id": f"fc_{index}",
                    "call_id": f"call_{index}",
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
                self.emit(
                    {
                        "type": "response.function_call_arguments.done",
                        "response_id": rid,
                        "item_id": call["id"],
                        "call_id": call["call_id"],
                        "name": name,
                        "arguments": call["arguments"],
                    }
                )
                output.append(call)
            self.emit(
                {
                    "type": "response.done",
                    "response": {
                        "id": rid,
                        "status": "completed",
                        "output": output,
                        "usage": {"input_tokens": 100 + index, "output_tokens": 40},
                    },
                }
            )
        except asyncio.CancelledError:
            if streaming:
                partial = "".join(sent)
                self.emit(
                    {
                        "type": "response.output_audio_transcript.done",
                        "item_id": item,
                        "transcript": partial,
                    }
                )
                self.emit(
                    {
                        "type": "response.done",
                        "response": {
                            "id": rid,
                            "status": "cancelled",
                            "output": [],
                            "usage": {"input_tokens": 100 + index, "output_tokens": 5},
                        },
                    }
                )


@dataclass
class FakeRealtime:
    """OpenAI Realtime on localhost; one :class:`_Agent` per connection."""

    sessions: list[_Agent] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    _server: Server | None = None
    port: int = 0

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v1/realtime"

    async def __aenter__(self) -> FakeRealtime:
        self._server = await serve(self._handle, "127.0.0.1", 0, max_size=None)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, ws: ServerConnection) -> None:
        self.urls.append(ws.request.path if ws.request else "")
        outbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        agent = _Agent(outbox.put_nowait)
        self.sessions.append(agent)

        async def send() -> None:
            while (event := await outbox.get()) is not None:
                await ws.send(json.dumps(event))

        sender = asyncio.create_task(send())
        outbox.put_nowait({"type": "session.created", "session": {"id": "sess_e2e"}})
        try:
            async for raw in ws:
                agent.on_client(json.loads(raw))
        except Exception:  # noqa: BLE001 - the client hung up
            pass
        finally:
            agent.close()
            outbox.put_nowait(None)
            sender.cancel()


# ------------------------------------------------------------ the caller


def caller_script() -> list[tuple[float, dict[str, Any]]]:
    """ElevenLabs' server messages after the metadata, from the socket's start."""
    items: list[tuple[float, int, dict[str, Any]]] = []
    event_id = 0
    for u in UTTERANCES:
        pcm = speech_like(int(u.seconds * CALLER_RATE), CALLER_RATE, seed=u.seed)
        for k, offset in enumerate(range(0, len(pcm), CHUNK_BYTES)):
            event_id += 1
            audio = base64.b64encode(pcm[offset : offset + CHUNK_BYTES]).decode()
            items.append(
                (
                    u.start_s + k * 0.05,
                    0,
                    {
                        "type": "audio",
                        "audio_event": {"audio_base_64": audio, "event_id": event_id},
                    },
                )
            )
        items.append(
            (
                u.start_s + 0.1,
                1,
                {
                    "type": "agent_response",
                    "agent_response_event": {"agent_response": u.text},
                },
            )
        )
    for at, text in TRANSCRIPTS:
        items.append(
            (
                at,
                2,
                {
                    "type": "user_transcript",
                    "user_transcription_event": {"user_transcript": text},
                },
            )
        )
    items.sort(key=lambda item: (item[0], item[1]))
    return [(at, message) for at, _, message in items]


@dataclass
class Conversation:
    conversation_id: str
    agent_id: str
    end_call: bool
    initiation: dict[str, Any] = field(default_factory=dict)
    heard_bytes: int = 0
    user_activity: int = 0

    def record(self) -> dict[str, Any]:
        """``GET /v1/convai/conversations/{id}``, once the call is over."""
        return {
            "agent_id": self.agent_id,
            "conversation_id": self.conversation_id,
            "status": "done",
            "transcript": [
                {"role": "agent", "message": u.text, "tool_results": []}
                for u in UTTERANCES
            ],
            "metadata": {
                "termination_reason": (
                    "end_call tool was called."
                    if self.end_call
                    else "Client disconnected: 1000"
                ),
                "call_duration_secs": int(END_S),
                "charging": {
                    "llm_usage": {
                        "irreversible_generation": {
                            "model_usage": {
                                "gpt-5.1": {
                                    "input": {"tokens": 1200},
                                    "output_total": {"tokens": 60},
                                }
                            }
                        }
                    }
                },
            },
        }


@dataclass
class FakeElevenLabs:
    """ElevenLabs Agents on localhost: the conversation socket and its REST API.

    ``end_call[n]`` says whether conversation ``n``'s record shows the
    caller's ``end_call``.
    """

    end_call: list[bool]
    conversations: list[Conversation] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)
    _server: Server | None = None
    port: int = 0

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def __aenter__(self) -> FakeElevenLabs:
        self._server = await serve(
            self._handle,
            "127.0.0.1",
            0,
            process_request=self._rest,
            max_size=None,
        )
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    def _conversation(self, conversation_id: str) -> Conversation | None:
        return next(
            (c for c in self.conversations if c.conversation_id == conversation_id),
            None,
        )

    def _rest(self, connection: ServerConnection, request: Request) -> Response | None:
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        url = urllib.parse.urlsplit(request.path)
        query = dict(urllib.parse.parse_qsl(url.query))
        path = url.path.rstrip("/")
        self.requests.append(path)
        if request.headers.get("xi-api-key") != "el-fake":
            return connection.respond(401, json.dumps({"detail": "bad key"}))
        parts = path.split("/")
        if path == "/v1/convai/conversation/get-signed-url":
            signed = (
                f"ws://127.0.0.1:{self.port}/v1/convai/conversation?"
                + urllib.parse.urlencode(
                    {"agent_id": query["agent_id"], "conversation_signature": "sig"}
                )
            )
            return connection.respond(200, json.dumps({"signed_url": signed}))
        if parts[:4] == ["", "v1", "convai", "agents"] and len(parts) == 5:
            return connection.respond(
                200,
                json.dumps(
                    {
                        "agent_id": parts[4],
                        "name": "EVA EN caller (fake)",
                        "conversation_config": {
                            "agent": {"prompt": {"prompt": "{{prompt}}"}}
                        },
                        "access_info": {"creator_email": "nobody@example.com"},
                    }
                ),
            )
        if parts[:4] == ["", "v1", "convai", "conversations"] and len(parts) >= 5:
            conversation = self._conversation(parts[4])
            if conversation is None:
                return connection.respond(404, json.dumps({"detail": "no such id"}))
            if len(parts) == 6 and parts[5] == "audio":
                return connection.respond(200, "ID3-fake-audio")
            return connection.respond(200, json.dumps(conversation.record()))
        return connection.respond(404, json.dumps({"detail": f"no route {path}"}))

    async def _handle(self, ws: ServerConnection) -> None:
        n = len(self.conversations)
        query = dict(
            urllib.parse.parse_qsl(
                urllib.parse.urlsplit(ws.request.path if ws.request else "").query
            )
        )
        conversation = Conversation(
            conversation_id=f"conv_e2e_{n}",
            agent_id=query.get("agent_id", "?"),
            end_call=self.end_call[min(n, len(self.end_call) - 1)],
        )
        self.conversations.append(conversation)
        conversation.initiation = json.loads(await ws.recv())
        await ws.send(
            json.dumps(
                {
                    "type": "conversation_initiation_metadata",
                    "conversation_initiation_metadata_event": {
                        "conversation_id": conversation.conversation_id,
                        "agent_output_audio_format": "pcm_16000",
                        "user_input_audio_format": "ulaw_8000",
                    },
                }
            )
        )
        t0 = time.monotonic()

        async def listen() -> None:
            async for raw in ws:
                message = json.loads(raw)
                if "user_audio_chunk" in message:
                    conversation.heard_bytes += len(
                        base64.b64decode(message["user_audio_chunk"])
                    )
                elif message.get("type") == "user_activity":
                    conversation.user_activity += 1

        listener = asyncio.create_task(listen())
        try:
            for at, message in caller_script():
                delay = t0 + at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                await ws.send(json.dumps(message))
            delay = t0 + END_S - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        finally:
            listener.cancel()
            await ws.close()
