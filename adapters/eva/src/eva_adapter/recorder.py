"""EVA's OpenAI Realtime assistant server, ported: what it records and when.

EVA's metrics read the agent's side of a call from files its assistant
server writes: ``audit_log.json`` (who said what, stamped on the wall clock),
``framework_logs.jsonl``, ``pipecat_metrics.jsonl`` (the ``model_response``
latency and token usage) and the 24 kHz ``audio_{user,assistant,mixed}.wav``.
This module is that server, ``eva/assistant/openai_realtime_server.py`` plus
the recording half of ``base_server.py`` @039efa0a, driven by harbor's
transport events instead of OpenAI's SDK. Comments cite EVA's file:line.

What carries over unchanged:

- **Stamps.** Every timestamp is ``time.time()`` read when the event is
  handled, as EVA's ``_wall_ms()`` is. A user entry is stamped at the
  bridge's ``user_speech_start`` (else at ``speech_started``), an assistant
  entry at its first audio delta, a tool call before it runs and its
  response after.
- **The turn records**: a new user turn only when the last one was flushed,
  ``[interrupted]`` partials flushed at ``speech_started``, cancelled
  responses skipped, the three-way transcript fallback in ``response.done``.
- **Buffer alignment.** Each track is padded up to the other before it
  grows (``sync_buffer_to_position``), the user track only when the user is
  not speaking and has sent nothing for 300 ms, the assistant track only
  when the bot is not speaking. ``_bot_speaking`` is cleared only by a reset
  of a response that streamed audio, so after an ``[interrupted]`` flush it
  stays set until the next such reset, exactly as in EVA.
- **Pacing.** Agent audio is converted to μ-law per delta and handed to the
  bridge in 160-byte chunks, 20 ms apart, by a pacer that never catches up
  after falling behind. Nothing is cleared on barge-in: EVA sends no
  ``truncate`` and keeps playing what it had queued.

What harbor replaces: the WebSocket to the bridge is the :class:`Recorder`
itself (an :class:`~eva_adapter.bridge.AssistantWire`), and the OpenAI
connection is harbor's ``openai.realtime`` transport, which the driver opens
with EVA's settings (``audio_format="pcm24k"``, ``opening_text``,
``tool_flow="serial"``, ``tool_output_ascii=False``,
``extended_events=True``). Tool calls are logged from the transport's
:class:`~harbor_voice.transport.stream.ToolCall` event, stamped at the
moments EVA's ``execute_and_log_tool`` stamps them: before the tool ran and
after it answered.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import InputTranscript
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import OutputTranscriptDone
from harbor_voice.transport.stream import ResponseDone
from harbor_voice.transport.stream import ToolCall

from eva_adapter.audio import MULAW_CHUNK_SIZE
from eva_adapter.audio import OPENAI_SAMPLE_RATE
from eva_adapter.audio import mulaw_8k_to_pcm16_24k
from eva_adapter.audio import pcm16_24k_to_mulaw_8k
from eva_adapter.audio import pcm16_mix
from eva_adapter.audio import save_audio_track
from eva_adapter.audio import sync_buffer_to_position

logger = logging.getLogger(__name__)

MULAW_CHUNK_DURATION_S = 0.02
"""``openai_realtime_server.py:39``."""
USER_ACTIVE_GUARD_S = 0.3
"""Don't pad the user track if real user audio arrived this recently
(``openai_realtime_server.py:40-46``)."""
MODEL_RESPONSE_MAX_MS = 30_000
"""Latencies outside (0, 30 s) are not written (``openai_realtime_server.py:587``)."""
METRICS_PROCESSOR_NAME = "openai_realtime"
"""``_metrics_processor_name`` (``openai_realtime_server.py:85``)."""
TRANSCRIPTION_UNAVAILABLE = "[user speech - transcription unavailable]"
"""The placeholder for a failed input transcription (``openai_realtime_server.py:462``)."""

AUDIT_LOG_FILE = "audit_log.json"
TRANSCRIPT_FILE = "transcript.jsonl"
FRAMEWORK_LOGS_FILE = "framework_logs.jsonl"
METRICS_FILE = "pipecat_metrics.jsonl"
AUDIO_MIXED_FILE = "audio_mixed.wav"
AUDIO_USER_FILE = "audio_user.wav"
AUDIO_ASSISTANT_FILE = "audio_assistant.wav"


def _tool_result(content: dict[str, Any]) -> Any:
    """What EVA logs as a tool's response: its result object.

    A driver may hand the transport the sidecar's exact ``function_call_output``
    string as ``{"text": output}`` so the model gets EVA's bytes; the result
    is the JSON object inside it.
    """
    if set(content) == {"text"} and isinstance(content["text"], str):
        try:
            return json.loads(content["text"])
        except ValueError:
            return dict(content)
    return dict(content)


def _wall_ms() -> str:
    """Wall-clock epoch milliseconds as a string (``openai_realtime_server.py:49-51``)."""
    return str(int(round(time.time() * 1000)))


current_timestamp_ms = _wall_ms
"""``agentic/audit_log.py:17-19``: the same expression."""


# ----------------------------------------------------------------- audit log


class AuditLog:
    """The part of EVA's ``AuditLog`` an S2S server writes (``agentic/audit_log.py:78-420``).

    ``transcript`` entries and ``conversation_messages`` in EVA's shapes;
    ``llm_prompts`` stays empty, as it does for the Realtime server.
    """

    def __init__(self) -> None:
        self.transcript: list[dict[str, Any]] = []
        self.llm_prompts: list[dict[str, Any]] = []
        self.conversation_messages: list[dict[str, Any]] = []
        self._tool_calls_count = 0
        self._tools_called: list[str] = []
        self._last_tool_call: str | None = None

    def append_user_input(
        self,
        content: str,
        timestamp_ms: str | None = None,
        turn_id: int | None = None,
        message_type: str = "user",
        llm_content: str | None = None,
    ) -> None:
        """``audit_log.py:96-149``."""
        entry: dict[str, Any] = {
            "value": content,
            "displayName": "User",
            "type": "text",
            "isBotMessage": False,
            "timestamp": timestamp_ms or current_timestamp_ms(),
            "message_type": message_type,
        }
        if turn_id is not None:
            entry["turn_id"] = turn_id
        if llm_content is not None:
            entry["llm_content"] = llm_content
        self.transcript.append(entry)
        # ConversationMessage.to_dict(): None fields and turn_id dropped.
        self.conversation_messages.append(
            {
                "role": "user",
                "content": llm_content if llm_content is not None else content,
            }
        )

    def append_assistant_output(
        self, content: str, timestamp_ms: str | None = None
    ) -> None:
        """``audit_log.py:212-251``, without tool calls or reasoning (S2S has neither)."""
        if content:
            self.transcript.append(
                {
                    "value": content,
                    "displayName": "Bot",
                    "type": "text",
                    "isBotMessage": True,
                    "timestamp": timestamp_ms or current_timestamp_ms(),
                    "message_type": "assistant",
                }
            )
        self.conversation_messages.append({"role": "assistant", "content": content})

    def append_tool_call(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        timestamp_ms: str | None = None,
    ) -> None:
        """``audit_log.py:296-324``.

        EVA stamps the entry as it appends it, just before the tool runs;
        ``timestamp_ms`` carries that moment when the entry is written after.
        """
        self.transcript.append(
            {
                "value": {"tool": tool_name, "parameters": parameters},
                "displayName": "Tool",
                "type": "tool_call",
                "isBotMessage": True,
                "timestamp": timestamp_ms or current_timestamp_ms(),
                "message_type": "tool_call",
            }
        )
        self._tool_calls_count += 1
        if tool_name not in self._tools_called:
            self._tools_called.append(tool_name)
        self._last_tool_call = tool_name

    def append_tool_response(
        self, tool_name: str, response: Any, timestamp_ms: str | None = None
    ) -> None:
        """``audit_log.py:326-337``; ``timestamp_ms`` as in :meth:`append_tool_call`."""
        self.transcript.append(
            {
                "value": {"tool": tool_name, "response": response},
                "displayName": "Tool Response",
                "type": "tool_response",
                "isBotMessage": True,
                "timestamp": timestamp_ms or current_timestamp_ms(),
                "message_type": "tool_response",
            }
        )

    def get_stats(self) -> dict[str, Any]:
        """``audit_log.py:376-385``."""
        num_turns = sum(1 for e in self.transcript if e.get("message_type") == "user")
        return {
            "num_turns": num_turns,
            "num_tool_calls": self._tool_calls_count,
            "tools_called": self._tools_called.copy(),
            "total_transcript_entries": len(self.transcript),
            "total_llm_calls": len(self.llm_prompts),
        }

    def save(self, path: Path) -> None:
        """``audit_log.py:387-411``: the transcript sorted by timestamp, then written."""
        self.transcript.sort(key=lambda e: int(e.get("timestamp", "0")))
        data = {
            "transcript": self.transcript,
            "llm_prompts": self.llm_prompts,
            "conversation_messages": self.conversation_messages,
        }
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_transcript_jsonl(self, path: Path) -> None:
        """``audit_log.py:413-420``: user and assistant entries only."""
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for entry in self.transcript:
                if entry.get("message_type") in ["user", "assistant"]:
                    record = {
                        "timestamp": entry["timestamp"],
                        "type": entry["message_type"],
                        "content": entry["value"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------- log writers


class _JsonlAppender:
    def __init__(self, path: Path) -> None:
        self.log_file = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, entry: dict[str, Any]) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001 - EVA logs and carries on
            logger.error("Error writing %s: %s", self.log_file.name, e)


class FrameworkLogWriter(_JsonlAppender):
    """``framework_logs.jsonl``, appended as it happens (``pipeline/observers.py:360-415``)."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir / FRAMEWORK_LOGS_FILE)

    def write(
        self, event_type: str, data: dict[str, Any], timestamp_ms: int | None = None
    ) -> None:
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        self._append({"timestamp": timestamp_ms, "type": event_type, "data": data})

    def turn_start(self, timestamp_ms: int | None = None) -> None:
        self.write("turn_start", {"frame": "turn_start"}, timestamp_ms)

    def turn_end(
        self, was_interrupted: bool = False, timestamp_ms: int | None = None
    ) -> None:
        self.write(
            "turn_end",
            {"frame": "turn_end", "was_interrupted": was_interrupted},
            timestamp_ms,
        )

    def llm_response(self, text: str, timestamp_ms: int | None = None) -> None:
        self.write("llm_response", {"frame": text}, timestamp_ms)

    def s2s_transcript(self, text: str, timestamp_ms: int | None = None) -> None:
        self.write("s2s_transcript", {"frame": text}, timestamp_ms)


class MetricsLogWriter(_JsonlAppender):
    """``pipecat_metrics.jsonl`` for a non-Pipecat server (``pipeline/observers.py:418-476``)."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir / METRICS_FILE)

    def write_latency(self, stage: str, value_seconds: float, model: str = "") -> None:
        self._append(
            {
                "timestamp": int(time.time() * 1000),
                "type": "LatencyMetric",
                "stage": stage,
                "model": model,
                "value": value_seconds,
            }
        )

    def write_token_usage(
        self, processor: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        self._append(
            {
                "timestamp": int(time.time() * 1000),
                "type": "LLMTokenUsageMetricsData",
                "processor": processor,
                "model": model,
                "value": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )


# ------------------------------------------------------------------- state


@dataclass
class _UserTurnRecord:
    """``openai_realtime_server.py:54-61``."""

    speech_started_wall_ms: str = ""
    speech_stopped_wall_ms: str = ""
    transcript: str = ""
    flushed: bool = False


@dataclass
class _AssistantResponseState:
    """``openai_realtime_server.py:64-72``."""

    transcript_parts: list[str] = field(default_factory=list)
    transcript_done_text: str = ""
    first_audio_wall_ms: str | None = None
    responding: bool = False
    has_function_calls: bool = False


# ---------------------------------------------------------------- recorder


_Message = tuple[str, Any]


class Recorder:
    """Port of ``OpenAIRealtimeAssistantServer`` (``openai_realtime_server.py:75-826``).

    It is the bridge's :class:`~eva_adapter.bridge.AssistantWire`: the
    bridge's messages queue here in order, as on EVA's socket, and
    :meth:`serve` runs EVA's three session tasks over them: forward user
    audio to the agent, handle the agent's events, pace its audio back.

    Args:
        output_dir: The record directory; logs are appended there as they
            happen, the audit log and WAVs written by :meth:`save_outputs`.
        model: The S2S model id, written into ``pipecat_metrics.jsonl``.
        party: The party stamped on frames sent to the agent.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        model: str,
        party: PartyId = PartyId("caller"),
        metrics_processor_name: str = METRICS_PROCESSOR_NAME,
    ) -> None:
        self.output_dir = Path(output_dir)
        self._model = model
        self._party = party
        self._metrics_processor_name = metrics_processor_name

        self.audit_log = AuditLog()
        self.fw_log: FrameworkLogWriter | None = None
        self.metrics_log: MetricsLogWriter | None = None

        # base_server.py:89-93.
        self._audio_buffer = bytearray()
        self.user_audio_buffer = bytearray()
        self.assistant_audio_buffer = bytearray()
        self._audio_sample_rate = OPENAI_SAMPLE_RATE

        # openai_realtime_server.py:90-109.
        self._last_user_audio_mono = 0.0
        self._user_turn: _UserTurnRecord | None = None
        self._assistant_state = _AssistantResponseState()
        self._user_speaking = False
        self._bot_speaking = False
        self._user_frame_count = 0
        self._delta_count = 0
        self._audio_interface_speech_start_ts: str | None = None

        self._inbound: asyncio.Queue[_Message] = asyncio.Queue()
        self._wire_open = True
        self._frames_sent = 0

    # ------------------------------------------------------------- set-up

    def start(self) -> None:
        """Create the log writers (``openai_realtime_server.py:120-122``)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fw_log = FrameworkLogWriter(self.output_dir)
        self.metrics_log = MetricsLogWriter(self.output_dir)

    # ------------------------------------------------- the wire from the bridge

    @property
    def is_open(self) -> bool:
        return self._wire_open

    def send_media(self, mulaw: bytes) -> None:
        if self._wire_open:
            self._inbound.put_nowait(("media", mulaw))

    def send_user_speech_start(self, timestamp_ms: str) -> None:
        if self._wire_open:
            self._inbound.put_nowait(("user_speech_start", timestamp_ms))

    def send_user_speech_stop(self, timestamp_ms: str) -> None:
        if self._wire_open:
            self._inbound.put_nowait(("user_speech_stop", timestamp_ms))

    def send_stop(self) -> None:
        if self._wire_open:
            self._inbound.put_nowait(("stop", None))

    def close(self) -> None:
        """The bridge closed the socket."""
        self._wire_open = False
        self._inbound.put_nowait(("closed", None))

    # ------------------------------------------------------------ session

    async def serve(
        self,
        agent: VoiceTransport,
        to_caller: Callable[[bytes], None],
    ) -> None:
        """Run the session until one of its tasks ends (``openai_realtime_server.py:284-304``).

        ``agent`` must be open, with ``extended_events``. ``to_caller`` takes
        each paced 160-byte μ-law chunk (the bridge's ``receive_media``).
        The wire is closed when this returns.
        """
        audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()
        tasks = [
            asyncio.create_task(self._forward_user_audio(agent)),
            asyncio.create_task(self._process_agent_events(agent, audio_output_queue)),
            asyncio.create_task(self._pace_audio_output(to_caller, audio_output_queue)),
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for task in done:
                if not task.cancelled() and task.exception():
                    logger.error("Session task failed: %s", task.exception())
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            self._wire_open = False

    def end_session(self) -> None:
        """The session is over; the bridge's sends go nowhere from now on."""
        self._wire_open = False

    async def _pace_audio_output(
        self,
        to_caller: Callable[[bytes], None],
        audio_output_queue: asyncio.Queue[bytes],
    ) -> None:
        """Agent audio to the bridge at 20 ms a chunk (``openai_realtime_server.py:313-342``).

        When it falls behind it restarts the schedule from now rather than
        catching up, as EVA's does.
        """
        next_send_time = time.monotonic()
        while True:
            try:
                chunk = await asyncio.wait_for(audio_output_queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            try:
                to_caller(chunk)
            except Exception as e:  # noqa: BLE001 - EVA ends the session on a send error
                logger.error("Error sending audio to the bridge: %s", e)
                return
            now = time.monotonic()
            if next_send_time <= now:
                next_send_time = now
            next_send_time += MULAW_CHUNK_DURATION_S
            sleep_duration = next_send_time - time.monotonic()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

    async def _forward_user_audio(self, agent: VoiceTransport) -> None:
        """Bridge messages → agent (``openai_realtime_server.py:346-409``)."""
        loop = asyncio.get_running_loop()
        try:
            while True:
                kind, payload = await self._inbound.get()
                if kind in ("stop", "closed"):
                    break
                if kind == "user_speech_start":
                    # The bridge's timestamp of when user audio really started.
                    self._audio_interface_speech_start_ts = payload
                    continue
                if kind != "media":
                    continue  # user_speech_stop: EVA's Realtime server ignores it

                pcm16_24k = mulaw_8k_to_pcm16_24k(payload)
                if not self._bot_speaking:
                    sync_buffer_to_position(
                        self.assistant_audio_buffer, len(self.user_audio_buffer)
                    )
                self.user_audio_buffer.extend(pcm16_24k)
                self._last_user_audio_mono = time.monotonic()
                self._user_frame_count += 1

                frame = AudioFrame(
                    pcm=pcm16_24k,
                    seq=self._frames_sent,
                    party=self._party,
                    t_produced=loop.time(),
                    fmt=CANONICAL,
                )
                self._frames_sent += 1
                await agent.tx(frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - EVA logs and ends the session
            logger.error("Error forwarding user audio: %s", e)

    async def _process_agent_events(
        self, agent: VoiceTransport, audio_output_queue: asyncio.Queue[bytes]
    ) -> None:
        """Agent events → handlers (``openai_realtime_server.py:413-487``)."""
        async for event in agent.events():
            try:
                await self._handle_event(event, audio_output_queue)
            except Exception as e:  # noqa: BLE001 - EVA logs and keeps reading
                logger.error("Error handling %s: %s", type(event).__name__, e)

    async def _handle_event(
        self, event: object, audio_output_queue: asyncio.Queue[bytes]
    ) -> None:
        match event:
            case InputSpeech(started=True):
                self._on_speech_started()
            case InputSpeech(started=False):
                self._on_speech_stopped()
            case InputTranscript(error=None, text=text):
                self._on_transcription_completed(text)
            case InputTranscript():
                self._on_transcription_failed()
            case OutputAudio(pcm=pcm):
                await self._on_audio_delta(pcm, audio_output_queue)
            case OutputTranscript(delta=delta):
                self._on_transcript_delta(delta)
            case OutputTranscriptDone(text=text):
                self._on_transcript_done(text)
            case ToolCall():
                self._on_tool_call(event)
            case ResponseDone():
                self._on_response_done(event)
            case _:
                logger.debug("Unhandled agent event: %s", type(event).__name__)

    # ------------------------------------------------------------ handlers

    def _on_speech_started(self) -> None:
        """``openai_realtime_server.py:491-530``."""
        self._user_speaking = True
        wall = _wall_ms()

        # The agent was responding: flush what it said as interrupted.
        if self._assistant_state.responding and self._assistant_state.transcript_parts:
            partial_text = (
                "".join(self._assistant_state.transcript_parts) + " [interrupted]"
            )
            self.audit_log.append_assistant_output(
                partial_text, timestamp_ms=self._assistant_state.first_audio_wall_ms
            )
            if self.fw_log:
                self.fw_log.s2s_transcript(partial_text)
                self.fw_log.turn_end(was_interrupted=True)
            # _bot_speaking is deliberately left as it is (EVA's).
            self._assistant_state = _AssistantResponseState()

        # A new user turn only if the last was flushed: VAD can fire several
        # speech_started in one utterance, and the first stamp is kept.
        if not self._user_turn or self._user_turn.flushed:
            start_ts = self._audio_interface_speech_start_ts or wall
            self._user_turn = _UserTurnRecord(speech_started_wall_ms=start_ts)
            if self.fw_log:
                self.fw_log.turn_start(timestamp_ms=int(start_ts))
            self._audio_interface_speech_start_ts = None

    def _on_speech_stopped(self) -> None:
        """``openai_realtime_server.py:532-548``."""
        self._user_speaking = False
        wall = _wall_ms()
        if self._user_turn:
            self._user_turn.speech_stopped_wall_ms = wall
        else:
            self._user_turn = _UserTurnRecord(speech_stopped_wall_ms=wall)

    def _on_transcription_completed(self, transcript: str) -> None:
        """``openai_realtime_server.py:550-566``."""
        transcript = (transcript or "").strip()
        if not transcript:
            return
        timestamp_ms = None
        if self._user_turn:
            timestamp_ms = self._user_turn.speech_started_wall_ms or None
            self._user_turn.transcript = transcript
            self._user_turn.flushed = True
        self.audit_log.append_user_input(transcript, timestamp_ms=timestamp_ms)

    def _on_transcription_failed(self) -> None:
        """``openai_realtime_server.py:453-465``: keep the turn with a placeholder."""
        if self._user_turn and not self._user_turn.flushed:
            timestamp_ms = self._user_turn.speech_started_wall_ms or None
            self.audit_log.append_user_input(
                TRANSCRIPTION_UNAVAILABLE, timestamp_ms=timestamp_ms
            )
            self._user_turn.flushed = True

    async def _on_audio_delta(
        self, pcm16_bytes: bytes, audio_output_queue: asyncio.Queue[bytes]
    ) -> None:
        """``openai_realtime_server.py:568-621``."""
        if not pcm16_bytes:
            return

        if self._assistant_state.first_audio_wall_ms is None:
            self._assistant_state.first_audio_wall_ms = _wall_ms()
            self._assistant_state.responding = True
            self._bot_speaking = True

            # user speech end → first audio; absent on the greeting.
            if (
                self._user_turn
                and self._user_turn.speech_stopped_wall_ms
                and self.metrics_log
            ):
                latency_ms = int(self._assistant_state.first_audio_wall_ms) - int(
                    self._user_turn.speech_stopped_wall_ms
                )
                if 0 < latency_ms < MODEL_RESPONSE_MAX_MS:
                    self.metrics_log.write_latency(
                        "model_response", latency_ms / 1000, self._model
                    )

        user_recently_active = (
            time.monotonic() - self._last_user_audio_mono
        ) <= USER_ACTIVE_GUARD_S
        if not self._user_speaking and not user_recently_active:
            sync_buffer_to_position(
                self.user_audio_buffer, len(self.assistant_audio_buffer)
            )
        self.assistant_audio_buffer.extend(pcm16_bytes)
        self._delta_count += 1

        try:
            mulaw_bytes = pcm16_24k_to_mulaw_8k(pcm16_bytes)
            offset = 0
            while offset < len(mulaw_bytes):
                chunk = mulaw_bytes[offset : offset + MULAW_CHUNK_SIZE]
                offset += MULAW_CHUNK_SIZE
                await audio_output_queue.put(chunk)
        except Exception as e:  # noqa: BLE001
            logger.error("Error converting audio for output queue: %s", e)

    def _on_transcript_delta(self, delta: str) -> None:
        """``openai_realtime_server.py:623-627``."""
        if delta:
            self._assistant_state.transcript_parts.append(delta)

    def _on_transcript_done(self, transcript: str) -> None:
        """``openai_realtime_server.py:629-638``."""
        if transcript:
            self._assistant_state.transcript_done_text = transcript.strip()

    def _on_tool_call(self, event: ToolCall) -> None:
        """``_on_function_call_done`` (``openai_realtime_server.py:640-678``), once answered.

        EVA marks the response as having a call, then ``execute_and_log_tool``
        appends ``tool_call`` (stamped just before the tool runs) and
        ``tool_response`` (just after), then ``framework_logs.jsonl`` gets a
        ``tool_call`` line. The transport ran the call and reports it here
        right after it answered, with how long it took on its own clock, so
        the response is stamped now and the call that long before.

        A call whose tools raised (the sidecar's ``/tool`` failing) gets the
        ``tool_call`` entry only: EVA's ``execute_and_log_tool`` raises before
        its ``tool_response``, and the sidecar checks the two logs agree.
        """
        self._assistant_state.has_function_calls = True
        answered = time.time()
        started = answered - max(0.0, event.t - event.started)
        arguments = dict(event.arguments)
        result = _tool_result(event.result)
        self.audit_log.append_tool_call(
            event.name, arguments, timestamp_ms=str(int(round(started * 1000)))
        )
        if event.is_error:
            return
        self.audit_log.append_tool_response(
            event.name, result, timestamp_ms=str(int(round(answered * 1000)))
        )
        if self.fw_log:
            self.fw_log.write(
                "tool_call",
                {
                    "frame": "tool_call",
                    "tool_name": event.name,
                    "arguments": arguments,
                    "result": result,
                },
            )

    def _on_response_done(self, event: ResponseDone) -> None:
        """``openai_realtime_server.py:680-755``."""
        if event.input_tokens is not None and self.metrics_log:
            self.metrics_log.write_token_usage(
                processor=self._metrics_processor_name,
                model=self._model,
                prompt_tokens=event.input_tokens or 0,
                completion_tokens=event.output_tokens or 0,
            )

        # Cancelled: interrupted, never fully spoken.
        if event.status == "cancelled":
            self._reset_assistant_state()
            return

        has_function_calls = event.has_function_calls

        # Best text: transcript.done, else the deltas, else the output items.
        content = self._assistant_state.transcript_done_text
        if not content:
            content = "".join(self._assistant_state.transcript_parts).strip()
        if not content:
            content = event.text

        audio_was_streamed = self._assistant_state.first_audio_wall_ms is not None

        if not content and has_function_calls:
            self._reset_assistant_state()  # tool call only, nothing spoken
            return
        if content and not audio_was_streamed and has_function_calls:
            self._reset_assistant_state()  # mixed, but no audio
            return
        if not content and audio_was_streamed:
            self._reset_assistant_state()  # spoken, but no transcript
            return
        if not content:
            self._reset_assistant_state()
            return

        timestamp = self._assistant_state.first_audio_wall_ms or _wall_ms()
        self.audit_log.append_assistant_output(content, timestamp_ms=timestamp)
        if self.fw_log:
            self.fw_log.llm_response(content)
            self.fw_log.turn_end(was_interrupted=False)
        self._reset_assistant_state()

    def _reset_assistant_state(self) -> None:
        """``openai_realtime_server.py:759-771``."""
        if self._assistant_state.first_audio_wall_ms is not None:
            self._bot_speaking = False
        self._assistant_state = _AssistantResponseState()

    # ------------------------------------------------------------ outputs

    def get_conversation_stats(self) -> dict[str, Any]:
        """``base_server.py:190-195``."""
        return self.audit_log.get_stats()

    def _ensure_mixed_audio(self) -> None:
        """``base_server.py:251-279``."""
        if self._audio_buffer:
            return
        if self.user_audio_buffer and self.assistant_audio_buffer:
            diff_bytes = abs(
                len(self.user_audio_buffer) - len(self.assistant_audio_buffer)
            )
            diff_ms = diff_bytes / (2 * self._audio_sample_rate) * 1000
            if diff_ms > 500:
                logger.warning(
                    "Audio buffer length mismatch: user=%d assistant=%d diff=%.0fms",
                    len(self.user_audio_buffer),
                    len(self.assistant_audio_buffer),
                    diff_ms,
                )
            self._audio_buffer = bytearray(
                pcm16_mix(
                    bytes(self.user_audio_buffer), bytes(self.assistant_audio_buffer)
                )
            )
        elif self.user_audio_buffer:
            self._audio_buffer = bytearray(self.user_audio_buffer)
        elif self.assistant_audio_buffer:
            self._audio_buffer = bytearray(self.assistant_audio_buffer)

    def save_outputs(self) -> None:
        """The audit log, the transcript and the three 24 kHz tracks.

        ``base_server.py:138-177`` and ``:223-293``. The scenario databases
        are the tool executor's and are written where it runs (the sidecar).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_mixed_audio()
        mixed_audio = bytes(self._audio_buffer)
        user_audio = bytes(self.user_audio_buffer)
        assistant_audio = bytes(self.assistant_audio_buffer)
        self.audit_log.save(self.output_dir / AUDIT_LOG_FILE)
        self.audit_log.save_transcript_jsonl(self.output_dir / TRANSCRIPT_FILE)
        rate = self._audio_sample_rate
        save_audio_track(mixed_audio, self.output_dir / AUDIO_MIXED_FILE, rate)
        save_audio_track(user_audio, self.output_dir / AUDIO_USER_FILE, rate)
        save_audio_track(assistant_audio, self.output_dir / AUDIO_ASSISTANT_FILE, rate)
