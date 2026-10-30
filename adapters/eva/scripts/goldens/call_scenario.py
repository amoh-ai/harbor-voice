"""The call golden's scripted call: one caller script and one fake agent, for EVA and for harbor.

Both sides of the call golden run this module, unchanged: EVA's own bridge,
Realtime server and ElevenLabs simulator in ``eva-base`` (``call_eva_golden.py``),
and harbor's port with harbor's transports (``call_harbor.py``). Nothing here
is either side's code; it plays the two services the call talks to.

**The caller** (:data:`CALLER_SCRIPT`) is ElevenLabs' side of the socket, as
server messages on a wall-clock timeline from the call's start: the
conversation metadata, the caller's speech as ``audio`` events (16 kHz PCM,
delivered faster than real time in chunks, as ElevenLabs sends them), the
``agent_response`` text cue, the agent's speech as ElevenLabs transcribed it
(``user_transcript``), and finally the socket closing, as it does after the
caller's ``end_call``. It is time-scripted rather than reactive so that both
sides get the same input at the same moments; the agent is the reactive part.

**The agent** (:class:`FakeRealtime`) is an OpenAI Realtime server that hears
the audio it is sent: an energy VAD over each appended frame (server VAD's
threshold, 300 ms prefix, 200 ms silence), transcripts 250 ms after the
caller stops, scripted responses that stream audio and transcript deltas
faster than real time, one response that speaks and then calls a tool, a
response created by the client after the tool output, and a barge-in: the
caller starts talking mid-response, which cancels it (``response.done`` with
status ``cancelled``) while the audio already sent keeps playing. Because the
VAD runs on what the bridge actually sent — including the 600 ms gap before a
partial chunk and the 18-frame silence bursts — the agent's timing is a
function of the bridge's, which is what the call golden compares.

The caller's utterances are four (one ends on a partial chunk, one exactly on
a chunk boundary so only the end-of-utterance cue ends it, one barges in, one
says goodbye); the call lasts about 28 s plus EVA's 6 s of teardown.
"""

from __future__ import annotations

import asyncio
import base64
import math
import time
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from call_audio_cases import speech_like

GREETING = "Hello! How can I help you today?"
OPENING_TEXT = f"Say: '{GREETING}'"
CONVERSATION_ID = "g5-1.1.2"
ELEVENLABS_CONVERSATION_ID = "conv_g5"
MODEL = "gpt-realtime-1.5"

TOOL_NAME = "get_reservation"
TOOL_CALL_ID = "call_g5_1"
TOOL_ARGUMENTS = {"confirmation_number": "ABC123", "last_name": "Smith"}
TOOL_RESULT = {
    "status": "success",
    "reservation": {
        "confirmation_number": "ABC123",
        "flight": "100",
        "destination": "Boston",
        "date": "Monday",
        "passenger": "José Smith",
    },
}
TOOL_DESCRIPTION = "Look up a reservation by confirmation number and last name."
TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "confirmation_number": {"type": "string"},
        "last_name": {"type": "string"},
    },
    "required": ["confirmation_number", "last_name"],
}

AGENT_RATE = 24_000
CALLER_RATE = 16_000
DELTA_BYTES = 4_800
"""100 ms of 24 kHz PCM per audio delta."""


@dataclass(frozen=True)
class Response:
    """One agent response: what it says, how long, how fast it streams."""

    text: str
    seconds: float
    latency_s: float
    """Trigger (``response.create``, or the caller's speech stopping) → first delta."""
    interval_s: float
    """Between deltas; 100 ms of audio each, so 0.05 s is twice real time."""
    function_call: bool = False


RESPONSES: tuple[Response, ...] = (
    Response(GREETING, 2.0, 0.4137, 0.0503),
    Response("Let me check that for you.", 1.2, 0.5171, 0.0507, function_call=True),
    Response(
        "I found your booking: flight 100 to Boston, Monday at 9 AM. "
        "Would you like to change it?",
        3.0,
        0.4093,
        0.0501,
    ),
    Response(
        "Sure. I can move you to Friday at 9 AM, at 2 PM, or at 6 PM, "
        "all with no change fee.",
        5.0,
        0.5233,
        0.0809,
    ),
    Response(
        "The evening flight leaves at 6 PM. Shall I book it?", 2.5, 0.5059, 0.0511
    ),
    Response("You're all set on the 6 PM flight. Goodbye!", 1.5, 0.4967, 0.0497),
)
"""Latencies and delta intervals are deliberately off the bridge's 20 ms and
250 ms grids: on the virtual clock an event that lands on the same instant
as a bridge tick is a tie, and a tie's order is an accident of each side's
task structure rather than of its logic."""

INPUT_TRANSCRIPTS: tuple[str, ...] = (
    "I need to change my flight.",
    "Yes, I want to move it to Friday.",
    "Wait, what about the evening?",
    "Yes, please. Thanks, goodbye.",
)
"""The agent's transcription of each caller turn (whisper-1's role)."""

TRANSCRIPTION_DELAY_S = 0.2617
VAD_RMS = 500.0
VAD_PREFIX_MS = 300
VAD_SILENCE_MS = 200


@dataclass(frozen=True)
class Utterance:
    """One caller utterance as ElevenLabs delivers it."""

    text: str
    start_s: float
    """First ``audio`` message, from the call's start."""
    n_bytes: int
    """16 kHz PCM16 length; not a multiple of 640 leaves a partial chunk."""
    chunk_bytes: int
    cue_s: float
    """The ``agent_response`` message (EVA's end-of-utterance cue)."""
    seed: int
    every_s: float = 0.05


UTTERANCES: tuple[Utterance, ...] = (
    # Ends on a partial chunk: the bridge's 600 ms partial-chunk path.
    Utterance(
        "I need to change my flight.",
        3.5031,
        48_320,
        7_000,
        3.6013,
        seed=21,
        every_s=0.0497,
    ),
    # A whole number of 640-byte chunks: only the cue ends it.
    Utterance(
        "Yes, I want to move it to Friday.",
        11.5047,
        51_200,
        6_400,
        11.7029,
        seed=22,
        every_s=0.0493,
    ),
    # Barges in on the agent's long answer.
    Utterance(
        "Wait, what about the evening?",
        16.6073,
        38_720,
        7_000,
        16.7011,
        seed=23,
        every_s=0.0511,
    ),
    Utterance(
        "Yes, please. Thanks, goodbye.",
        22.6019,
        45_120,
        7_000,
        22.7037,
        seed=24,
        every_s=0.0489,
    ),
)

ASSISTANT_TRANSCRIPTS: tuple[tuple[float, str], ...] = (
    (3.1043, "Hello, how can I help you today?"),
    (
        11.3507,
        "Let me check that for you. I found your booking, flight 100 to Boston, "
        "Monday at 9 AM. Would you like to change it?",
    ),
    (18.6061, "Sure, I can move you to Friday at 9 AM."),
    (22.4033, "The evening flight leaves at 6 PM. Shall I book it?"),
    (27.5079, "You're all set on the 6 PM flight. Goodbye."),
)
"""ElevenLabs' transcript of the agent (``user_transcript``), from the call's start.

Each lands at least ~100 ms clear of where the bridge can detect the end of
the agent's audio: that detection runs on 250 ms ticks, and a transcript
logged within a tick of it could swap places with it from run to run."""

END_S = 28.0071
"""ElevenLabs closes the socket (the caller's ``end_call``), from the call's start."""

SIGNED_URL_DELAY_S = 0.3017
"""The signed-URL request before the caller's socket connects."""


def caller_pcm(utterance: Utterance) -> bytes:
    return speech_like(utterance.n_bytes // 2, CALLER_RATE, seed=utterance.seed)


def agent_pcm(index: int) -> bytes:
    response = RESPONSES[index]
    return speech_like(int(response.seconds * AGENT_RATE), AGENT_RATE, seed=40 + index)


def build_caller_script() -> list[tuple[float, dict[str, Any]]]:
    """ElevenLabs' server messages, in time order, from the call's start."""
    items: list[tuple[float, int, dict[str, Any]]] = []
    event_id = 0
    for u in UTTERANCES:
        pcm = caller_pcm(u)
        for k, offset in enumerate(range(0, len(pcm), u.chunk_bytes)):
            event_id += 1
            items.append(
                (
                    u.start_s + k * u.every_s,
                    0,
                    {
                        "type": "audio",
                        "audio_event": {
                            "audio_base_64": base64.b64encode(
                                pcm[offset : offset + u.chunk_bytes]
                            ).decode("ascii"),
                            "event_id": event_id,
                        },
                    },
                )
            )
        items.append(
            (
                u.cue_s,
                1,
                {
                    "type": "agent_response",
                    "agent_response_event": {"agent_response": u.text},
                },
            )
        )
    for at, text in ASSISTANT_TRANSCRIPTS:
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


CALLER_SCRIPT = build_caller_script()

METADATA = {
    "type": "conversation_initiation_metadata",
    "conversation_initiation_metadata_event": {
        "conversation_id": ELEVENLABS_CONVERSATION_ID,
        "agent_output_audio_format": "pcm_16000",
        "user_input_audio_format": "ulaw_8000",
    },
}

CONVERSATION_RECORD = {
    "conversation_id": ELEVENLABS_CONVERSATION_ID,
    "status": "done",
    "metadata": {"termination_reason": "Client disconnected: end_call tool was called"},
    "transcript": [],
}
"""What the conversations API returns after the call: ``end_call`` ran."""


async def sleep_until(t: float) -> None:
    delay = t - time.time()
    if delay > 0:
        await asyncio.sleep(delay)


async def play_caller_script(
    t0: float, deliver: Callable[[dict[str, Any]], Any]
) -> None:
    """Deliver every caller message at its time; return at :data:`END_S`."""
    for at, message in CALLER_SCRIPT:
        await sleep_until(t0 + at)
        result = deliver(message)
        if asyncio.iscoroutine(result):
            await result
    await sleep_until(t0 + END_S)


# ----------------------------------------------------------------- the agent


def _rms(pcm: bytes) -> float:
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class FakeRealtime:
    """An OpenAI Realtime server, as far as the call golden needs one.

    Client events go in as dicts (:meth:`on_client`); server events come out
    as dicts through ``emit``, in order. Responses run as tasks on the
    caller's event loop.
    """

    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self._emit = emit
        self._audio_ms = 0.0
        self._speaking = False
        self._silence_ms = 0.0
        self._stops = 0
        self._next_response = 0
        self._active: _ActiveResponse | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self.log: list[tuple[float, str]] = []
        self.frames: list[tuple[float, int, bool]] = []
        """Every appended frame: when, how many bytes, voiced."""

    def connected(self) -> None:
        self._out({"type": "session.created", "session": {"id": "sess_g5"}})

    def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()

    def _out(self, event: dict[str, Any]) -> None:
        self.log.append((time.time(), str(event.get("type"))))
        self._emit(event)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def on_client(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "session.update":
            self._out({"type": "session.updated", "session": event.get("session", {})})
        elif kind == "response.create":
            self._start_response()
        elif kind == "input_audio_buffer.append":
            self._on_audio(base64.b64decode(event.get("audio") or ""))
        elif kind == "conversation.item.create":
            item = event.get("item") or {}
            self.log.append((time.time(), f"client:{item.get('type')}"))

    def _on_audio(self, pcm: bytes) -> None:
        frame_ms = len(pcm) / 2 / AGENT_RATE * 1000
        self._audio_ms += frame_ms
        voiced = _rms(pcm) > VAD_RMS
        self.frames.append((time.time(), len(pcm), voiced))
        if not self._speaking:
            if voiced:
                self._speaking = True
                self._silence_ms = 0.0
                start = max(0, int(self._audio_ms - frame_ms - VAD_PREFIX_MS))
                self._out(
                    {
                        "type": "input_audio_buffer.speech_started",
                        "audio_start_ms": start,
                        "item_id": f"in_{self._stops}",
                    }
                )
                if self._active is not None:
                    self._active.cancel()
            return
        if voiced:
            self._silence_ms = 0.0
            return
        self._silence_ms += frame_ms
        if self._silence_ms >= VAD_SILENCE_MS:
            self._speaking = False
            index = self._stops
            self._stops += 1
            self._out(
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "audio_end_ms": int(self._audio_ms - self._silence_ms),
                    "item_id": f"in_{index}",
                }
            )
            self._spawn(self._transcribe(index))
            self._start_response()

    async def _transcribe(self, index: int) -> None:
        await asyncio.sleep(TRANSCRIPTION_DELAY_S)
        self._out(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": f"in_{index}",
                "content_index": 0,
                "transcript": INPUT_TRANSCRIPTS[index],
            }
        )

    def _start_response(self) -> None:
        index = self._next_response
        self._next_response += 1
        if index >= len(RESPONSES):
            return
        active = _ActiveResponse(self, index)
        self._active = active
        self._spawn(active.run())

    def _finished(self, active: _ActiveResponse) -> None:
        if self._active is active:
            self._active = None


class _ActiveResponse:
    def __init__(self, server: FakeRealtime, index: int) -> None:
        self.server = server
        self.index = index
        self.spec = RESPONSES[index]
        self.id = f"resp_{index}"
        self.item = f"item_{index}"
        self.sent_text: list[str] = []
        self.task: asyncio.Task[None] | None = None
        self.cancelled = False
        self.streaming = False

    def cancel(self) -> None:
        """Server VAD's ``interrupt_response``: stop, close what was said, report it cancelled."""
        if self.cancelled or not self.streaming:
            return
        self.cancelled = True
        if self.task is not None:
            self.task.cancel()
        partial = "".join(self.sent_text)
        out = self.server._out
        out(
            {
                "type": "response.output_audio_transcript.done",
                "item_id": self.item,
                "transcript": partial,
            }
        )
        out(
            {
                "type": "response.done",
                "response": {
                    "id": self.id,
                    "status": "cancelled",
                    "output": [
                        {
                            "type": "message",
                            "id": self.item,
                            "content": [
                                {"type": "output_audio", "transcript": partial}
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 100 + self.index,
                        "output_tokens": 10 + self.index,
                    },
                },
            }
        )
        self.server._finished(self)

    async def run(self) -> None:
        self.task = asyncio.current_task()
        out = self.server._out
        try:
            await asyncio.sleep(self.spec.latency_s)
            self.streaming = True
            out({"type": "response.created", "response": {"id": self.id}})
            pcm = agent_pcm(self.index)
            words = self.spec.text.split(" ")
            n = math.ceil(len(pcm) / DELTA_BYTES)
            for i in range(n):
                out(
                    {
                        "type": "response.output_audio.delta",
                        "response_id": self.id,
                        "item_id": self.item,
                        "delta": base64.b64encode(
                            pcm[i * DELTA_BYTES : (i + 1) * DELTA_BYTES]
                        ).decode("ascii"),
                    }
                )
                lo, hi = (i * len(words)) // n, ((i + 1) * len(words)) // n
                if hi > lo:
                    text = " ".join(words[lo:hi])
                    delta = text if lo == 0 else " " + text
                    self.sent_text.append(delta)
                    out(
                        {
                            "type": "response.output_audio_transcript.delta",
                            "response_id": self.id,
                            "item_id": self.item,
                            "delta": delta,
                        }
                    )
                await asyncio.sleep(self.spec.interval_s)
            self.streaming = False
            out(
                {
                    "type": "response.output_audio_transcript.done",
                    "item_id": self.item,
                    "transcript": self.spec.text,
                }
            )
            output: list[dict[str, Any]] = [
                {
                    "type": "message",
                    "id": self.item,
                    "content": [{"type": "output_audio", "transcript": self.spec.text}],
                }
            ]
            if self.spec.function_call:
                import json

                arguments = json.dumps(TOOL_ARGUMENTS)
                out(
                    {
                        "type": "response.function_call_arguments.done",
                        "response_id": self.id,
                        "item_id": f"fc_{self.index}",
                        "call_id": TOOL_CALL_ID,
                        "name": TOOL_NAME,
                        "arguments": arguments,
                    }
                )
                output.append(
                    {
                        "type": "function_call",
                        "id": f"fc_{self.index}",
                        "call_id": TOOL_CALL_ID,
                        "name": TOOL_NAME,
                        "arguments": arguments,
                    }
                )
            out(
                {
                    "type": "response.done",
                    "response": {
                        "id": self.id,
                        "status": "completed",
                        "output": output,
                        "usage": {
                            "input_tokens": 100 + self.index,
                            "output_tokens": 50 + self.index,
                        },
                    },
                }
            )
            self.server._finished(self)
        except asyncio.CancelledError:
            return
