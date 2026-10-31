"""The caller: EVA's ElevenLabs simulator session logic, against fakes.

Runs the real ``elevenlabs.agents`` transport over a fake socket and a
``httpx.MockTransport`` for ElevenLabs' REST API, with a fake event log and
bridge, and a sleep the test drives — so 10 s keep-alives and 300 s polls take
no time. Every expectation cites the EVA line (``eva/user_simulator/elevenlabs.py``
@039efa0a unless named) it reproduces.
"""

from __future__ import annotations

import asyncio
import json
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from eva_adapter import caller as c
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.elevenlabs.agents import AgentResponse
from harbor_voice.providers.elevenlabs.agents import AgentResponseCorrection
from harbor_voice.providers.elevenlabs.agents import AgentsAPI
from harbor_voice.providers.elevenlabs.agents import ClientToolCall
from harbor_voice.providers.elevenlabs.agents import ElevenLabsAgentsTransport
from harbor_voice.providers.elevenlabs.agents import Interruption
from harbor_voice.providers.elevenlabs.agents import UserTranscript

pytestmark = pytest.mark.unit

PROMPT = "You are calling your airline. Say: 'Hi, I need to change my flight.'"

METADATA = json.dumps(
    {
        "type": "conversation_initiation_metadata",
        "conversation_initiation_metadata_event": {
            "conversation_id": "conv_1",
            "agent_output_audio_format": "pcm_16000",
            "user_input_audio_format": "ulaw_8000",
        },
    }
)

PROCESSING: dict[str, Any] = {"status": "processing", "transcript": [], "metadata": {}}
TERMINATED_BY_END_CALL: dict[str, Any] = {
    "status": "processing",
    "transcript": [],
    "metadata": {"termination_reason": "end_call tool was called."},
}
DONE_WITH_TOOL: dict[str, Any] = {
    "status": "done",
    "metadata": {"termination_reason": "Client disconnected"},
    "transcript": [
        {"role": "agent", "message": "Bye!", "tool_results": None},
        {
            "role": "agent",
            "message": None,
            "tool_results": [{"tool_name": "end_call", "type": "system"}],
        },
    ],
}
DONE_WITHOUT_END_CALL: dict[str, Any] = {
    "status": "done",
    "metadata": {"termination_reason": "Maximum duration reached"},
    "transcript": [{"role": "user", "message": "Hello?", "tool_results": []}],
}
DONE_EMPTY: dict[str, Any] = {"status": "done", "transcript": [], "metadata": {}}


# ------------------------------------------------------------------- fakes


class FakeWS:
    """A websocket: the test pushes server messages; ``None`` is the server hanging up."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.inbox: asyncio.Queue[str | None] = asyncio.Queue()
        self.close_code: int | None = None
        self.closed = False
        self.fail_on: str | None = None

    async def send(self, message: str) -> None:
        if self.closed:
            raise ConnectionError("socket closed")
        if self.fail_on and self.fail_on in message:
            raise ConnectionError("send failed")
        self.sent.append(message)

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.close_code = 1000
            self.inbox.put_nowait(None)

    def push(self, message: str) -> None:
        self.inbox.put_nowait(message)

    def hang_up(self) -> None:
        self.close_code = 1000
        self.inbox.put_nowait(None)

    def __aiter__(self) -> FakeWS:
        return self

    async def __anext__(self) -> str:
        message = await self.inbox.get()
        if message is None:
            raise StopAsyncIteration
        return message

    def sent_types(self) -> list[str]:
        return [json.loads(m).get("type", "user_audio_chunk") for m in self.sent]


class FakeEleven:
    """ElevenLabs: the signed URL, the conversation records in order, the audio."""

    def __init__(
        self,
        records: list[dict[str, Any] | int] | None = None,
        audio: list[int] | None = None,
        signed_url_status: int = 200,
    ) -> None:
        self.ws = FakeWS()
        self.records = list(records or [DONE_WITH_TOOL])
        self.audio = list(audio or [200])
        self.signed_url_status = signed_url_status
        self.requests: list[httpx.Request] = []
        self.urls: list[str] = []

    def _next(self, queue: list[Any]) -> Any:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def http(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/get-signed-url"):
            if self.signed_url_status != 200:
                return httpx.Response(self.signed_url_status, json={"detail": "no"})
            return httpx.Response(
                200,
                json={
                    "signed_url": "wss://el.test/v1/convai/conversation?agent_id=a&conversation_signature=s"
                },
            )
        if path.endswith("/audio"):
            status = self._next(self.audio)
            return httpx.Response(
                status, content=b"ID3-mp3" if status == 200 else b"no"
            )
        if path.startswith("/v1/convai/conversations/"):
            record = self._next(self.records)
            if isinstance(record, int):
                return httpx.Response(record, json={"detail": "error"})
            return httpx.Response(200, json=record)
        return httpx.Response(404)

    def polls(self) -> int:
        return sum(
            1
            for r in self.requests
            if r.url.path.startswith("/v1/convai/conversations/")
            and not r.url.path.endswith("/audio")
        )

    def audio_fetches(self) -> int:
        return sum(1 for r in self.requests if r.url.path.endswith("/audio"))

    def transport(self, *, api_key: str | None = "k") -> ElevenLabsAgentsTransport:
        api = AgentsAPI(
            api_key=api_key,
            client=httpx.AsyncClient(transport=httpx.MockTransport(self.http)),
        )

        async def connect(url: str, max_size: int) -> FakeWS:
            self.urls.append(url)
            return self.ws

        return ElevenLabsAgentsTransport(agent_id="agent_t", api=api, connect=connect)


class FakeEvents:
    """EVA's event logger, reduced to the JSONL it would write."""

    def __init__(self, order: list[str] | None = None) -> None:
        self.lines: list[dict[str, Any]] = []
        self.order = order if order is not None else []

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        self.lines.append({"type": event_type, "data": data, "provider": "elevenlabs"})
        self.order.append(f"log:{event_type}")

    def log_connection_state(
        self, state: str, details: dict[str, Any] | None = None
    ) -> None:
        self.log_event("connection_state", {"state": state, "details": details or {}})

    def log_error(self, error: str, details: dict[str, Any] | None = None) -> None:
        self.log_event("error", {"error": error, "details": details or {}})

    def types(self) -> list[str]:
        out = []
        for line in self.lines:
            if line["type"] == "connection_state":
                out.append(f"connection_state:{line['data']['state']}")
            else:
                out.append(line["type"])
        return out

    def valid_end(self) -> bool:
        """``conversation_valid_end`` on the last line (validation/conversation_valid_end.py:85-103)."""
        last = self.lines[-1]
        return (
            last["type"] == "connection_state"
            and last["data"]["details"].get("reason") == "goodbye"
        )


class FakeBridge:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.playing = False
        self.latencies: list[float] = [0.5, 1.5]
        self.stops = 0
        self.input_callback: Callable[[bytes], Any] | None = None

    def start(self, input_callback: Callable[[bytes], Any]) -> None:
        self.input_callback = input_callback

    def notify_user_utterance_complete(self) -> None:
        self.order.append("bridge:utterance_complete")

    def is_assistant_playing(self) -> bool:
        return self.playing

    def get_latencies(self) -> list[float]:
        return list(self.latencies)

    def stop(self) -> None:
        self.stops += 1


class Sleeper:
    """The keep-alive's sleeps wait for :meth:`tick`; listed durations block
    forever; every other sleep returns at once. The keep-alive is told apart
    by its task, not its 10 s: the poll and audio backoffs reach 10 s too."""

    def __init__(self, block: set[float] | None = None) -> None:
        self.slept: list[float] = []
        self.keepalive: list[float] = []
        self.ticks: asyncio.Queue[None] = asyncio.Queue()
        self.block = block or set()
        self.caller: c.ElevenLabsCaller | None = None

    async def __call__(self, seconds: float) -> None:
        caller = self.caller
        if caller is not None and asyncio.current_task() is caller._keepalive_task:
            self.keepalive.append(seconds)
            await self.ticks.get()
            return
        self.slept.append(seconds)
        if seconds in self.block:
            await asyncio.Event().wait()
        else:
            await asyncio.sleep(0)

    async def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.ticks.put_nowait(None)
            await settle()

    def others(self) -> list[float]:
        return list(self.slept)


async def settle(rounds: int = 30) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


class Call:
    """One caller session wired as the call loop wires it."""

    def __init__(self, tmp_path: Path, eleven: FakeEleven, **kw: Any) -> None:
        self.eleven = eleven
        self.order: list[str] = []
        self.events = FakeEvents(self.order)
        self.bridge = FakeBridge(self.order)
        self.sleep = Sleeper(kw.pop("block", None))
        self.signals: list[str | None] = []
        self.caller = c.ElevenLabsCaller(
            transport=kw.pop("transport", None) or eleven.transport(),
            prompt=PROMPT,
            events=self.events,
            out_dir=tmp_path,
            bridge=self.bridge,
            server_url="openai.realtime/gpt-realtime-1.5",
            on_conversation_ending=self.signals.append,
            sleep=self.sleep,
            **kw,
        )
        self.sleep.caller = self.caller
        self.pump: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self.caller.start()
        self.pump = asyncio.create_task(self._feed())

    async def _feed(self) -> None:
        async for event in self.caller.transport.events():
            self.caller.on_event(event)

    async def hang_up_after_metadata(self) -> None:
        self.eleven.ws.push(METADATA)
        self.eleven.ws.hang_up()
        await asyncio.wait_for(self.caller.wait(), 2)

    async def finish(self) -> str:
        reason = await self.caller.finish()
        if self.pump is not None:
            await asyncio.wait_for(self.pump, 2)
        return reason


# ---------------------------------------------------------------- start


async def test_start_logs_connected_then_session_started_and_sends_the_prompt(
    tmp_path: Path,
) -> None:
    call = Call(tmp_path, FakeEleven())
    await call.start()
    assert call.events.lines[:2] == [
        {
            "type": "connection_state",
            "data": {
                "state": "connected",
                "details": {"server_url": "openai.realtime/gpt-realtime-1.5"},
            },
            "provider": "elevenlabs",
        },
        {
            "type": "connection_state",
            "data": {"state": "session_started", "details": {}},
            "provider": "elevenlabs",
        },
    ]
    (initiation,) = call.eleven.ws.sent
    assert json.loads(initiation)["dynamic_variables"] == {"prompt": PROMPT}  # :144
    assert call.eleven.urls[0].endswith("&source=python_sdk&version=2.53.0")
    # The SDK hands the bridge its input callback once connected (conversation.py:845-846).
    assert call.bridge.input_callback == call.caller.transport.send_audio
    assert not call.caller.should_end
    await call.hang_up_after_metadata()
    await call.finish()


async def test_a_missing_key_fails_the_job_before_anything_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    eleven = FakeEleven()
    call = Call(tmp_path, eleven, transport=eleven.transport(api_key=None))
    with pytest.raises(ProviderError) as info:
        await call.caller.start()
    assert info.value.kind is ProviderFailure.AUTH
    assert call.events.lines == []


async def test_a_failed_start_is_evas_error_path(tmp_path: Path) -> None:
    call = Call(tmp_path, FakeEleven(signed_url_status=500))
    await call.caller.start()  # does not raise
    assert call.caller.should_end
    assert await call.caller.finish() == c.ERROR
    # connected, then (after the teardown) the error; no session_started (:104-110, :229-232).
    assert call.events.types() == ["connection_state:connected", "error"]
    assert "500" in call.events.lines[-1]["data"]["error"]
    assert call.sleep.others() == [c.STT_GRACE_S]
    assert call.signals == [c.ERROR]
    assert not call.events.valid_end()


# ------------------------------------------------------------- callbacks


async def test_callbacks_log_what_evas_do(tmp_path: Path) -> None:
    call = Call(tmp_path, FakeEleven())
    await call.start()
    caller = call.caller
    caller.on_event(AgentResponse("I need to change my flight.", 1.0))
    caller.on_event(AgentResponseCorrection("I need to change", "I need", 1.1))
    caller.on_event(UserTranscript("Sure, what's your code?", 2.0))
    caller.on_event(Interruption(2.5, 7))
    caller.on_event(ClientToolCall("t1", "x", {}, 2.6))
    assert call.events.lines[2:] == [
        {
            "type": "user_speech",
            "data": {"text": "I need to change my flight.", "source": "simulated_user"},
            "provider": "elevenlabs",
        },
        {
            "type": "user_speech_correction",
            "data": {"original": "I need to change", "corrected": "I need"},
            "provider": "elevenlabs",
        },
        {
            "type": "assistant_speech",
            "data": {"text": "Sure, what's your code?", "source": "assistant"},
            "provider": "elevenlabs",
        },
    ]
    # The bridge hears the end-of-utterance cue before the event is logged (:414-426).
    assert call.order[2:4] == ["bridge:utterance_complete", "log:user_speech"]
    await call.hang_up_after_metadata()
    await call.finish()


# ------------------------------------------------------------ end reasons


async def test_goodbye_from_termination_reason_and_the_files_eva_writes(
    tmp_path: Path,
) -> None:
    eleven = FakeEleven(records=[TERMINATED_BY_END_CALL])
    call = Call(tmp_path, eleven)
    await call.start()
    call.caller.record_audio("user_clean", b"\x01\x00" * 1600)
    call.caller.record_audio("assistant", b"\x02\x00" * 1600)  # not kept (base.py:173)
    await call.hang_up_after_metadata()
    assert call.caller.end_reason == c.SESSION_ENDED  # the SDK's stop() on hang-up
    assert call.bridge.stops == 1

    assert await call.finish() == c.GOODBYE
    assert call.events.lines[-1]["data"] == {
        "state": "session_ended",
        "details": {"reason": "goodbye"},
    }
    assert call.events.valid_end()
    assert call.signals == [c.SESSION_ENDED]
    assert call.bridge.stops == 2  # end_session runs stop() again (:209)
    # Poll after 2 s, found; then the 4 s grace (:289-296, :262).
    assert call.sleep.others() == [2.0, c.STT_GRACE_S]
    assert eleven.polls() == 1 and eleven.audio_fetches() == 1
    details = json.loads((tmp_path / c.DETAILS_FILE).read_text(encoding="utf-8"))
    assert details == TERMINATED_BY_END_CALL
    assert (tmp_path / c.AUDIO_FILE).read_bytes() == b"ID3-mp3"
    latencies = json.loads((tmp_path / c.LATENCIES_FILE).read_text(encoding="utf-8"))
    assert latencies == {"latencies": [0.5, 1.5], "mean": 1.0, "max": 1.5, "count": 2}
    with wave.open(str(tmp_path / c.CLEAN_AUDIO_FILE), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (
            16000,
            1,
            2,
        )
        assert wav.readframes(wav.getnframes()) == b"\x01\x00" * 1600
    assert call.eleven.ws.closed
    assert call.caller.usage_records()[0].component == "caller"


async def test_goodbye_from_an_end_call_tool_result(tmp_path: Path) -> None:
    eleven = FakeEleven(records=[PROCESSING, DONE_WITH_TOOL])
    call = Call(tmp_path, eleven)
    await call.start()
    await call.hang_up_after_metadata()
    assert await call.finish() == c.GOODBYE
    assert eleven.polls() == 2
    assert call.sleep.others() == [2.0, 4.0, c.STT_GRACE_S]


async def test_session_ended_when_the_finished_record_has_no_end_call(
    tmp_path: Path,
) -> None:
    eleven = FakeEleven(records=[DONE_WITHOUT_END_CALL])
    call = Call(tmp_path, eleven)
    await call.start()
    await call.hang_up_after_metadata()
    assert await call.finish() == c.SESSION_ENDED
    assert call.events.lines[-1]["data"]["details"] == {"reason": "session_ended"}
    assert not call.events.valid_end()


async def test_end_call_unverified_when_the_record_never_finishes(
    tmp_path: Path,
) -> None:
    eleven = FakeEleven(records=[PROCESSING])
    call = Call(tmp_path, eleven)
    await call.start()
    await call.hang_up_after_metadata()
    assert await call.finish() == c.END_CALL_UNVERIFIED
    # 2, 4, 8, then 10 s until 300 s of sleeps are spent: 32 polls (:289-327).
    polls = call.sleep.others()[:-1]
    assert polls[:4] == [2.0, 4.0, 8.0, 10.0] and sum(polls) == 304.0
    assert eleven.polls() == len(polls) == 32
    assert call.caller.end_call_found is None


async def test_a_done_record_with_an_empty_transcript_keeps_polling_as_in_eva(
    tmp_path: Path,
) -> None:
    eleven = FakeEleven(records=[DONE_EMPTY])
    call = Call(tmp_path, eleven)
    await call.start()
    await call.hang_up_after_metadata()
    assert await call.finish() == c.END_CALL_UNVERIFIED
    assert eleven.polls() == 32


async def test_a_failed_check_leaves_the_reason_and_audio_is_retried(
    tmp_path: Path,
) -> None:
    eleven = FakeEleven(records=[500], audio=[500])
    call = Call(tmp_path, eleven)
    await call.start()
    await call.hang_up_after_metadata()
    assert await call.finish() == c.SESSION_ENDED  # :219-220
    # One poll (raised), then 5 audio tries with 2, 4, 8, 10 s between (:334-352).
    assert call.sleep.others() == [2.0, 2.0, 4.0, 8.0, 10.0, c.STT_GRACE_S]
    assert eleven.polls() == 1 and eleven.audio_fetches() == 5
    assert not (tmp_path / c.AUDIO_FILE).exists()
    assert not (tmp_path / c.DETAILS_FILE).exists()
    assert call.events.types()[-1] == "connection_state:session_ended"


async def test_audio_arrives_on_a_retry(tmp_path: Path) -> None:
    eleven = FakeEleven(records=[DONE_WITH_TOOL], audio=[404, 404, 200])
    call = Call(tmp_path, eleven)
    await call.start()
    await call.hang_up_after_metadata()
    await call.finish()
    assert eleven.audio_fetches() == 3
    assert (tmp_path / c.AUDIO_FILE).read_bytes() == b"ID3-mp3"


async def test_no_conversation_id_skips_the_check(tmp_path: Path) -> None:
    eleven = FakeEleven()
    call = Call(tmp_path, eleven)
    await call.start()
    eleven.ws.hang_up()  # before any metadata
    await asyncio.wait_for(call.caller.wait(), 2)
    assert await call.finish() == c.SESSION_ENDED  # :214-215
    assert eleven.polls() == 0 and eleven.audio_fetches() == 0


# -------------------------------------------------------------- inactivity


async def test_twelve_idle_keepalives_end_the_call(tmp_path: Path) -> None:
    eleven = FakeEleven(records=[DONE_WITHOUT_END_CALL])
    call = Call(tmp_path, eleven)
    await call.start()
    eleven.ws.push(METADATA)
    await call.sleep.tick(11)
    assert not call.caller.should_end
    await call.sleep.tick()
    assert call.caller.should_end
    assert call.caller.end_reason == c.INACTIVITY_TIMEOUT
    assert eleven.ws.sent_types().count("user_activity") == 12  # :375
    assert set(call.sleep.keepalive) == {c.KEEPALIVE_INTERVAL_S}  # :369
    assert call.signals == [c.INACTIVITY_TIMEOUT]
    assert await call.finish() == c.INACTIVITY_TIMEOUT
    assert call.events.lines[-1]["data"]["details"] == {"reason": "inactivity_timeout"}


async def test_inactivity_then_end_call_is_still_goodbye(tmp_path: Path) -> None:
    eleven = FakeEleven(records=[DONE_WITH_TOOL])
    call = Call(tmp_path, eleven)
    await call.start()
    eleven.ws.push(METADATA)
    await call.sleep.tick(12)
    assert await call.finish() == c.GOODBYE  # _end_reason_after_check (:267-274)


async def test_speech_and_agent_audio_reset_the_count(tmp_path: Path) -> None:
    eleven = FakeEleven()
    call = Call(tmp_path, eleven)
    await call.start()
    eleven.ws.push(METADATA)
    await call.sleep.tick(11)
    call.caller.on_event(AgentResponse("Still there?", 1.0))  # :411
    await call.sleep.tick(11)
    call.caller.on_event(UserTranscript("Yes.", 2.0))  # :455
    await call.sleep.tick(11)
    call.bridge.playing = True  # :380-382
    await call.sleep.tick(3)
    call.bridge.playing = False
    await call.sleep.tick(11)
    call.caller.on_event(
        AgentResponseCorrection("a", "b", 3.0)
    )  # corrections do not reset
    assert not call.caller.should_end
    await call.sleep.tick()
    assert call.caller.end_reason == c.INACTIVITY_TIMEOUT
    await call.finish()


async def test_a_keepalive_that_cannot_be_sent_neither_counts_nor_resets(
    tmp_path: Path,
) -> None:
    eleven = FakeEleven()
    call = Call(tmp_path, eleven)
    await call.start()
    eleven.ws.push(METADATA)
    await call.sleep.tick(5)
    eleven.ws.fail_on = "user_activity"
    await call.sleep.tick(20)
    assert not call.caller.should_end  # :398-399
    eleven.ws.fail_on = None
    await call.sleep.tick(6)
    assert not call.caller.should_end
    await call.sleep.tick()
    assert call.caller.end_reason == c.INACTIVITY_TIMEOUT
    assert call.caller.keepalives_sent == 12
    await call.finish()


# -------------------------------------------------------------- time limits


async def test_the_simulators_timeout_is_logged_then_overwritten_as_in_eva(
    tmp_path: Path,
) -> None:
    eleven = FakeEleven(records=[DONE_WITHOUT_END_CALL])
    call = Call(tmp_path, eleven, timeout_s=0.05)
    await call.start()
    eleven.ws.push(METADATA)
    await asyncio.wait_for(call.caller.wait(), 2)
    assert call.caller.timed_out and call.caller.end_reason == c.TIMEOUT
    assert call.events.lines[-1] == {
        "type": "timeout",
        "data": {"duration": 0.05},
        "provider": "elevenlabs",
    }  # :190-192
    assert call.signals == [c.TIMEOUT]
    # end_session's stop() reports session_ended into a conversation not yet
    # done (audio_bridge.py:224-239), so "timeout" does not survive.
    assert await call.finish() == c.SESSION_ENDED
    assert call.signals == [c.TIMEOUT]


async def test_timeout_then_end_call_is_goodbye(tmp_path: Path) -> None:
    eleven = FakeEleven(records=[TERMINATED_BY_END_CALL])
    call = Call(tmp_path, eleven, timeout_s=0.05)
    await call.start()
    eleven.ws.push(METADATA)
    await asyncio.wait_for(call.caller.wait(), 2)
    assert await call.finish() == c.GOODBYE


async def test_the_guard_cancelling_finish_mid_poll_still_tears_down(
    tmp_path: Path,
) -> None:
    eleven = FakeEleven(records=[PROCESSING])
    call = Call(tmp_path, eleven, block={2.0})
    await call.start()
    await call.hang_up_after_metadata()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(call.caller.finish(), 0.1)
    # EVA's finally still ran: latencies, the grace; no session_ended (worker.py:166-176).
    assert call.sleep.slept[-1] == c.STT_GRACE_S
    assert (tmp_path / c.LATENCIES_FILE).exists()
    assert "connection_state:session_ended" not in call.events.types()
    assert not call.events.valid_end()


async def test_an_agent_disconnect_keeps_its_reason_unless_end_call_shows(
    tmp_path: Path,
) -> None:
    eleven = FakeEleven(records=[DONE_WITHOUT_END_CALL])
    call = Call(tmp_path, eleven)
    await call.start()
    eleven.ws.push(METADATA)
    await settle()
    call.caller.end(
        c.ELEVENLABS_DISCONNECT
    )  # the bridge's done callback (audio_bridge.py:738-744)
    await asyncio.wait_for(call.caller.wait(), 2)
    assert await call.finish() == c.ELEVENLABS_DISCONNECT
    assert call.signals == [c.ELEVENLABS_DISCONNECT]

    eleven = FakeEleven(records=[DONE_WITH_TOOL])
    call = Call(tmp_path / "b", eleven)
    await call.start()
    eleven.ws.push(METADATA)
    await settle()
    call.caller.end(c.ELEVENLABS_DISCONNECT)
    assert await call.finish() == c.GOODBYE


# ------------------------------------------------------------------ misc


async def test_nothing_is_taken_after_the_session_is_ended(tmp_path: Path) -> None:
    call = Call(tmp_path, FakeEleven())
    await call.start()
    await call.hang_up_after_metadata()
    await call.finish()
    before = list(call.events.lines)
    call.caller.on_event(AgentResponse("late", 9.0))
    assert call.events.lines == before
    assert await call.caller.finish() == c.GOODBYE  # idempotent


def test_end_reason_after_check() -> None:
    f = c.end_reason_after_check
    assert f(c.SESSION_ENDED, True) == c.GOODBYE
    assert f(c.INACTIVITY_TIMEOUT, True) == c.GOODBYE
    assert f(c.SESSION_ENDED, None) == c.END_CALL_UNVERIFIED
    assert f(c.INACTIVITY_TIMEOUT, None) == c.INACTIVITY_TIMEOUT
    assert f(c.SESSION_ENDED, False) == c.SESSION_ENDED
    assert f(c.ELEVENLABS_DISCONNECT, False) == c.ELEVENLABS_DISCONNECT


def test_end_call_verdict() -> None:
    v = c.end_call_verdict
    assert v(TERMINATED_BY_END_CALL) is True
    assert v(DONE_WITH_TOOL) is True
    assert v(DONE_WITHOUT_END_CALL) is False
    assert v(PROCESSING) is None
    assert v(DONE_EMPTY) is None
    assert v({"status": "in-progress", "transcript": [{"tool_results": None}]}) is None
    assert v({"metadata": None}) is None


def test_constants_are_evas() -> None:
    assert c.END_CALL_CHECK_BUDGET_SECONDS == 300.0  # elevenlabs.py:29
    assert (c.KEEPALIVE_INTERVAL_S, c.MAX_CONSECUTIVE_KEEPALIVES) == (
        10.0,
        12,
    )  # :369, :85
    assert c.STT_GRACE_S == 4.0  # :262
    assert c.AUDIO_FETCH_ATTEMPTS == 5  # :335
    assert c.SIMULATOR_TIMEOUT_S == c.GUARD_TIMEOUT_S == 620  # worker.py:23, :398-400
    assert c.ELEVENLABS_OUTPUT_RATE == 16_000  # audio_bridge.py:37
