"""The call loop's rules the call golden does not drive: EVA's branches one at a time.

The call golden (``test_eva_call_golden.py``) runs a whole scripted call against EVA's
own record. These pin the paths that call does not take: the agent's side
closing mid-call, audio after the caller hung up, the worker's time limit, a
failed transcription, cancelled and tool-only responses, a tool that raised,
and the event log's format.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import InputTranscript
from harbor_voice.transport.stream import ResponseDone
from harbor_voice.transport.stream import ToolCall

from eva_adapter.bridge import Bridge
from eva_adapter.call import TIME_LIMIT_EXCEEDED
from eva_adapter.call import RealtimeCall
from eva_adapter.events import EventLog
from eva_adapter.recorder import TRANSCRIPTION_UNAVAILABLE
from eva_adapter.recorder import Recorder

# ------------------------------------------------------------------ events


def test_event_log_writes_eva_jsonl(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl", provider="elevenlabs")
    log.log_connection_state("connected", {"server_url": "ws://x"})
    log.log_audio_start("simulated_user", 1700000000.25)
    log.log_audio_end("assistant", 0.0)
    log.log_audio_start("assistant", 0.0)  # falsy: "now", as EVA's `or`
    log.log_event("user_speech", {"text": "héllo", "source": "simulated_user"})
    log.save()

    raw = (tmp_path / "events.jsonl").read_bytes()
    assert b"\r\n" not in raw
    lines = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert [e["sequence"] for e in lines] == [1, 2, 3, 4, 5]
    assert lines[0]["type"] == "connection_state"
    assert lines[0]["data"] == {
        "state": "connected",
        "details": {"server_url": "ws://x"},
    }
    start = lines[1]
    assert set(start) == {
        "timestamp",
        "sequence",
        "event_type",
        "user",
        "audio_timestamp",
        "provider",
    }
    assert start["audio_timestamp"] == 1700000000.25
    assert isinstance(start["timestamp"], int)
    assert lines[2]["audio_timestamp"] == 0.0  # end: only None means now
    assert lines[3]["audio_timestamp"] > 1e9
    assert "héllo" in raw.decode("utf-8")


# ------------------------------------------------------------------ bridge


class Wire:
    def __init__(self) -> None:
        self.is_open = True
        self.sent: list[tuple[str, Any]] = []

    def send_media(self, mulaw: bytes) -> None:
        self.sent.append(("media", mulaw))

    def send_user_speech_start(self, timestamp_ms: str) -> None:
        self.sent.append(("start", timestamp_ms))

    def send_user_speech_stop(self, timestamp_ms: str) -> None:
        self.sent.append(("stop_speech", timestamp_ms))

    def send_stop(self) -> None:
        self.sent.append(("stop", None))

    def close(self) -> None:
        self.is_open = False


async def test_agent_side_closing_mid_call_ends_it_as_a_disconnect(
    tmp_path: Path,
) -> None:
    log = EventLog(tmp_path / "e.jsonl", provider="elevenlabs")
    reasons: list[str] = []
    bridge = Bridge(
        Wire(), "c", event_logger=log, conversation_done_callback=reasons.append
    )
    await bridge.start_async()
    bridge.receive_media(b"\x00" * 160)
    assert bridge.is_assistant_playing()
    bridge.wire_closed()
    assert reasons == ["elevenlabs_disconnect"]
    kinds = [(e.get("event_type"), e.get("user")) for e in log.get_events()]
    assert kinds == [("audio_start", "assistant"), ("audio_end", "assistant")]
    await bridge.stop_async()
    assert reasons == ["elevenlabs_disconnect"]


async def test_audio_after_the_caller_hung_up_ends_the_receive_loop(
    tmp_path: Path,
) -> None:
    log = EventLog(tmp_path / "e.jsonl", provider="elevenlabs")
    reasons: list[str] = []
    wire = Wire()
    bridge = Bridge(
        wire, "c", event_logger=log, conversation_done_callback=reasons.append
    )
    await bridge.start_async()
    bridge.receive_media(b"\x00" * 160)
    bridge.stop()  # the caller's session ended: running = False
    assert reasons == ["session_ended"]
    bridge.receive_media(b"\x00" * 160)  # EVA's loop breaks here; its finally runs
    bridge.receive_media(b"\x00" * 160)
    ends = [e for e in log.get_events() if e.get("event_type") == "audio_end"]
    assert len(ends) == 1
    assert bridge.audio_buffer.qsize() == 1
    await bridge.stop_async()
    assert ("stop", None) in wire.sent and not wire.is_open


# ---------------------------------------------------------------- recorder


@pytest.fixture
def recorder(tmp_path: Path) -> Recorder:
    rec = Recorder(tmp_path, model="m")
    rec.start()
    return rec


def _audit(recorder: Recorder) -> list[tuple[str, Any]]:
    return [(e["message_type"], e["value"]) for e in recorder.audit_log.transcript]


async def test_barge_in_flushes_interrupted_and_leaves_bot_speaking(
    recorder: Recorder,
) -> None:
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    await recorder._on_audio_delta(b"\x01\x00" * 2640, queue)
    recorder._on_transcript_delta("Sure. I can")
    recorder._on_transcript_delta(" move you")
    first_audio = recorder._assistant_state.first_audio_wall_ms
    recorder._audio_interface_speech_start_ts = "1700000000123"
    recorder._on_speech_started()

    assert _audit(recorder) == [("assistant", "Sure. I can move you [interrupted]")]
    assert recorder.audit_log.transcript[0]["timestamp"] == first_audio
    assert recorder._bot_speaking  # EVA leaves it set after the flush
    assert recorder._user_turn is not None
    assert recorder._user_turn.speech_started_wall_ms == "1700000000123"
    # 110 ms: 880 μ-law bytes, paced out in 160-byte chunks.
    assert [len(queue.get_nowait()) for _ in range(queue.qsize())] == [160] * 5 + [80]

    recorder._on_response_done(ResponseDone("r", "cancelled", False, "", 0.0, 10, 2))
    assert recorder._bot_speaking  # the reset after a flush streamed nothing
    fw = (recorder.output_dir / "framework_logs.jsonl").read_text().splitlines()
    assert [json.loads(line)["type"] for line in fw] == [
        "s2s_transcript",
        "turn_end",
        "turn_start",
    ]


async def test_failed_transcription_keeps_the_turn(recorder: Recorder) -> None:
    await recorder._handle_event(InputSpeech(True, 0, 0.0), asyncio.Queue())
    await recorder._handle_event(
        InputTranscript("i", "", 0.0, error="no access"), asyncio.Queue()
    )
    await recorder._handle_event(
        InputTranscript("i", "late text", 0.0), asyncio.Queue()
    )
    assert _audit(recorder) == [
        ("user", TRANSCRIPTION_UNAVAILABLE),
        ("user", "late text"),
    ]


async def test_blank_transcription_is_skipped(recorder: Recorder) -> None:
    await recorder._handle_event(InputTranscript("i", "   ", 0.0), asyncio.Queue())
    assert _audit(recorder) == []


@pytest.mark.parametrize(
    ("audio", "transcript", "done", "logged"),
    [
        (False, "", ResponseDone("r", "completed", True, "", 0.0), []),
        (False, "words", ResponseDone("r", "completed", True, "", 0.0), []),
        (True, "", ResponseDone("r", "completed", False, "", 0.0), []),
        (
            True,
            "",
            ResponseDone("r", "completed", False, "from items", 0.0),
            ["from items"],
        ),
        (True, " Hello. ", ResponseDone("r", "completed", False, "", 0.0), ["Hello."]),
        (True, "Hi", ResponseDone("r", "cancelled", False, "", 0.0), []),
    ],
)
async def test_response_done_logs_what_eva_logs(
    recorder: Recorder,
    audio: bool,
    transcript: str,
    done: ResponseDone,
    logged: list[str],
) -> None:
    if audio:
        await recorder._on_audio_delta(b"\x01\x00" * 480, asyncio.Queue())
    if transcript:
        recorder._on_transcript_done(transcript)
    recorder._on_response_done(done)
    assert [v for kind, v in _audit(recorder) if kind == "assistant"] == logged
    assert recorder._assistant_state.first_audio_wall_ms is None


def test_token_usage_only_when_reported(recorder: Recorder) -> None:
    recorder._on_response_done(ResponseDone("a", "completed", False, "", 0.0))
    recorder._on_response_done(ResponseDone("b", "completed", False, "", 0.0, 7, 3))
    lines = (recorder.output_dir / "pipecat_metrics.jsonl").read_text().splitlines()
    assert [json.loads(line)["value"] for line in lines] == [
        {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    ]


@pytest.mark.parametrize(("stopped_ago_s", "written"), [(0.5, True), (31.0, False)])
async def test_model_response_latency_window(
    recorder: Recorder, stopped_ago_s: float, written: bool
) -> None:
    import time

    recorder._on_speech_started()
    recorder._on_speech_stopped()
    assert recorder._user_turn is not None
    recorder._user_turn.speech_stopped_wall_ms = str(
        int(round((time.time() - stopped_ago_s) * 1000))
    )
    await recorder._on_audio_delta(b"\x01\x00" * 480, asyncio.Queue())
    path = recorder.output_dir / "pipecat_metrics.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    assert bool(lines) is written


def _tool(result: dict[str, Any], *, is_error: bool = False) -> ToolCall:
    return ToolCall("c1", "lookup", {"id": "A1"}, result, is_error, 1.0, 1.25)


def test_tool_call_is_logged_as_eva_logs_it(recorder: Recorder) -> None:
    recorder._on_tool_call(_tool({"status": "success", "name": "José"}))
    entries = recorder.audit_log.transcript
    assert _audit(recorder) == [
        ("tool_call", {"tool": "lookup", "parameters": {"id": "A1"}}),
        (
            "tool_response",
            {"tool": "lookup", "response": {"status": "success", "name": "José"}},
        ),
    ]
    assert int(entries[1]["timestamp"]) - int(entries[0]["timestamp"]) in (
        249,
        250,
        251,
    )
    assert recorder.audit_log.get_stats()["tools_called"] == ["lookup"]
    fw = json.loads(
        (recorder.output_dir / "framework_logs.jsonl").read_text(encoding="utf-8")
    )
    assert fw["type"] == "tool_call" and fw["data"]["result"]["name"] == "José"


def test_tool_output_sent_as_text_is_logged_as_its_json(recorder: Recorder) -> None:
    output = json.dumps({"status": "success", "seats": [1, 2]}, ensure_ascii=False)
    recorder._on_tool_call(_tool({"text": output}))
    assert _audit(recorder)[1][1]["response"] == {"status": "success", "seats": [1, 2]}


def test_tool_that_raised_gets_no_response_entry(recorder: Recorder) -> None:
    recorder._on_tool_call(
        _tool({"error": "HTTPStatusError", "message": "500"}, is_error=True)
    )
    assert [kind for kind, _ in _audit(recorder)] == ["tool_call"]


def test_outputs_are_eva_files(recorder: Recorder) -> None:
    recorder.user_audio_buffer.extend(b"\x01\x00" * 100)
    recorder.assistant_audio_buffer.extend(b"\x02\x00" * 60)
    recorder.audit_log.append_user_input("b", timestamp_ms="2000")
    recorder.audit_log.append_assistant_output("a", timestamp_ms="1000")
    recorder.save_outputs()
    out = recorder.output_dir
    audit = json.loads((out / "audit_log.json").read_text(encoding="utf-8"))
    assert [e["value"] for e in audit["transcript"]] == ["a", "b"]
    assert audit["conversation_messages"] == [
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "a"},
    ]
    for name in ("audio_user.wav", "audio_assistant.wav", "audio_mixed.wav"):
        assert (out / name).exists()
    assert len((out / "transcript.jsonl").read_text().splitlines()) == 2


# --------------------------------------------------------------- the call


class FakeAgent:
    def __init__(self, fail_open: bool = False) -> None:
        self.is_open = False
        self.fail_open = fail_open
        self._inbox: asyncio.Queue[object | None] = asyncio.Queue()
        self.frames = 0

    async def open(self, config: object) -> None:
        if self.fail_open:
            raise ConnectionError("refused")
        self.is_open = True

    async def tx(self, frame: object) -> None:
        self.frames += 1

    async def events(self):  # noqa: ANN201
        while (event := await self._inbox.get()) is not None:
            yield event

    async def close(self) -> None:
        self.is_open = False
        self._inbox.put_nowait(None)


class FakeCallerTransport:
    async def events(self):  # noqa: ANN201
        await asyncio.Event().wait()
        yield None

    async def close(self) -> None:
        pass


class FakeCaller:
    def __init__(self, events: EventLog, bridge: Bridge) -> None:
        self.events = events
        self.bridge = bridge
        self.ended: list[str] = []
        self.finished = False
        self._done = asyncio.Event()

    @property
    def should_end(self) -> bool:
        return self._done.is_set()

    @property
    def end_reason(self) -> str:
        return self.ended[0] if self.ended else "unknown"

    async def start(self) -> None:
        self.events.log_connection_state("connected", {})

    def on_event(self, event: object) -> None:
        pass

    def end(self, reason: str) -> None:
        self.ended.append(reason)
        self._done.set()

    def record_audio(self, source: str, audio: bytes) -> None:
        pass

    async def wait(self) -> None:
        await self._done.wait()

    async def finish(self) -> str:
        self.finished = True
        self.bridge.stop()
        return self.end_reason


async def test_time_limit_ends_the_call_and_still_writes_the_record(
    tmp_path: Path,
) -> None:
    call = RealtimeCall(
        agent=FakeAgent(),  # ty: ignore[invalid-argument-type]
        caller_transport=FakeCallerTransport(),
        out_dir=tmp_path,
        conversation_id="c",
        guard_timeout_s=0.3,
    )
    caller = FakeCaller(call.events, call.bridge)
    outcome = await call.run(caller)
    assert outcome.conversation_ended_reason == TIME_LIMIT_EXCEEDED
    assert caller.finished
    assert (tmp_path / "user_simulator_events.jsonl").exists()
    assert (tmp_path / "audit_log.json").exists()
    info = outcome.call_info(attempt=2).to_json()
    assert (
        info["attempt"] == 2
        and info["conversation_ended_reason"] == TIME_LIMIT_EXCEEDED
    )


async def test_agent_that_cannot_connect_ends_the_call_as_a_disconnect(
    tmp_path: Path,
) -> None:
    call = RealtimeCall(
        agent=FakeAgent(fail_open=True),  # ty: ignore[invalid-argument-type]
        caller_transport=FakeCallerTransport(),
        out_dir=tmp_path,
        conversation_id="c",
        guard_timeout_s=5.0,
    )
    caller = FakeCaller(call.events, call.bridge)
    outcome = await call.run(caller)
    assert caller.ended[0] == "elevenlabs_disconnect"
    assert outcome.conversation_ended_reason == "elevenlabs_disconnect"
    assert outcome.agent_error is not None and "refused" in outcome.agent_error
