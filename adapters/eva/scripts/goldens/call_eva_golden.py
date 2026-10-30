"""The call golden, EVA's side: EVA's own bridge, Realtime server and simulator on the scripted call.

Run inside EVA's image (cwd ``/app``, EVA's repo root)::

    MSYS_NO_PATHCONV=1 docker run --rm \\
        -v "$PWD/adapters/eva/scripts/goldens:/goldens" -v "$OUT:/out" \\
        --entrypoint /opt/venv/bin/python eva-base:039efa0a \\
        /goldens/call_eva_golden.py /out/eva [--real-time]

By default the call runs on ``call_vtime.VirtualClock`` (see there for why);
``--real-time`` runs it on the wall clock, for measuring EVA's own run-to-run
spread.

What runs is EVA's code, unchanged, wired as EVA's worker wires it
(``orchestrator/worker.py:134-424``): ``OpenAIRealtimeAssistantServer`` on a
local port, ``ElevenLabsUserSimulator`` (and so ``ElevenLabsAudioInterface``,
the bridge) connecting to it over EVA's WebSocket, the worker's 620 s guard,
then ``server.stop()`` and ``result.json``. Four things are replaced, at
their edges only:

- the OpenAI connection (``_create_client``): :class:`FakeConn` wraps
  ``call_scenario.FakeRealtime``, the same fake agent harbor's side talks to;
- the ElevenLabs SDK ``Conversation``: :class:`FakeConversation` delivers
  ``call_scenario.CALLER_SCRIPT`` through the SDK's own dispatch
  (``_handle_message_core``: ``output``, ``callback_agent_response``,
  ``callback_user_transcript``) and ends the session as the SDK does when
  the socket closes. It calls back on the event loop, not from a thread:
  EVA's SDK thread can delay a queued chunk by up to the loop's next wake
  (≤ 10 ms here), which is noise, not EVA's logic;
- the ElevenLabs REST client, whose conversation record says ``end_call`` ran;
- the tool's result (``ToolExecutor.execute``): a fixed answer, so both
  sides log the same one. Logging is EVA's ``execute_and_log_tool``.

Writes EVA's record directory to the given path, plus ``g5_meta.json``: the
call's start (``t0``, epoch seconds), its end reason, and what the caller
heard (every chunk handed to the SDK's ``input_callback``).
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import call_scenario as sc  # noqa: E402
import call_vtime  # noqa: E402

PORT = 10123
T0 = 0.0
VIRTUAL = True


def ns(value: Any) -> Any:
    """A server event as the OpenAI SDK hands it over: attributes, not keys."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [ns(v) for v in value]
    return value


class FakeConn:
    """``client.realtime.connect(...)`` over :class:`call_scenario.FakeRealtime`."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.engine = sc.FakeRealtime(self.queue.put_nowait)
        self.client_events: list[tuple[float, str]] = []
        self.appended = 0
        self.session = SimpleNamespace(update=self._session_update)
        self.conversation = SimpleNamespace(
            item=SimpleNamespace(create=self._item_create)
        )
        self.response = SimpleNamespace(create=self._response_create)
        self.input_audio_buffer = SimpleNamespace(append=self._append)

    def _client(self, event: dict[str, Any]) -> None:
        if event["type"] == "input_audio_buffer.append":
            self.appended += 1
        else:
            self.client_events.append((time.time() - T0, event["type"]))
        self.engine.on_client(event)

    async def _session_update(self, session: dict[str, Any]) -> None:
        self._client({"type": "session.update", "session": session})

    async def _item_create(self, item: dict[str, Any]) -> None:
        self._client({"type": "conversation.item.create", "item": item})

    async def _response_create(self) -> None:
        self._client({"type": "response.create"})

    async def _append(self, audio: str) -> None:
        self._client({"type": "input_audio_buffer.append", "audio": audio})

    def connect(self, model: str) -> FakeConn:
        return self

    async def __aenter__(self) -> FakeConn:
        self.engine.connected()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.engine.close()

    def __aiter__(self) -> FakeConn:
        return self

    async def __anext__(self) -> Any:
        return ns(await self.queue.get())


class FakeConversation:
    """The ElevenLabs SDK's ``Conversation``, playing the scripted caller."""

    instances: list[FakeConversation] = []

    def __init__(
        self,
        client: Any,
        agent_id: str,
        config: Any = None,
        requires_auth: bool = True,
        audio_interface: Any = None,
        callback_agent_response: Any = None,
        callback_agent_response_correction: Any = None,
        callback_user_transcript: Any = None,
        **_: Any,
    ) -> None:
        self.loop = asyncio.get_running_loop()
        self.audio_interface = audio_interface
        self.callback_agent_response = callback_agent_response
        self.callback_user_transcript = callback_user_transcript
        self._conversation_id: str | None = None
        self._last_interrupt_id = 0
        self._should_stop = False
        self.heard: list[tuple[float, int, int]] = []
        FakeConversation.instances.append(self)

    def start_session(self) -> None:
        """In the executor, as EVA runs it: the signed URL, then the socket thread.

        On the virtual clock the executor runs this inline and resolves it
        :data:`call_scenario.SIGNED_URL_DELAY_S` later (``call_vtime``), so the
        socket "thread" starts then.
        """
        if VIRTUAL:
            self.loop.call_later(
                sc.SIGNED_URL_DELAY_S, lambda: self.loop.create_task(self._run())
            )
            return
        time.sleep(sc.SIGNED_URL_DELAY_S)
        asyncio.run_coroutine_threadsafe(self._run(), self.loop)

    async def _run(self) -> None:
        self.audio_interface.start(self._input_callback)
        self._deliver(sc.METADATA)
        await sc.play_caller_script(T0, self._deliver)
        if not self._should_stop:
            self.end_session()  # ConnectionClosedOK → end_session (conversation.py:853)

    def _deliver(self, message: dict[str, Any]) -> None:
        """``_handle_message_core`` (conversation.py:475-548), the cases the script uses."""
        if self._should_stop:
            return
        kind = message["type"]
        if kind == "conversation_initiation_metadata":
            event = message["conversation_initiation_metadata_event"]
            self._conversation_id = event["conversation_id"]
        elif kind == "audio":
            event = message["audio_event"]
            if int(event["event_id"]) <= self._last_interrupt_id:
                return
            self.audio_interface.output(base64.b64decode(event["audio_base_64"]))
        elif kind == "agent_response":
            event = message["agent_response_event"]
            self.callback_agent_response(event["agent_response"].strip())
        elif kind == "user_transcript":
            event = message["user_transcription_event"]
            self.callback_user_transcript(event["user_transcript"].strip())

    def _input_callback(self, audio: bytes) -> None:
        if self._should_stop:
            return  # the socket is closed: the SDK's send fails, nothing is heard
        voiced = sum(1 for b in audio if b not in (0xFF, 0x7F))
        self.heard.append((time.time() - T0, len(audio), voiced))

    def end_session(self) -> None:
        if self.audio_interface is not None:
            self.audio_interface.stop()
        self._should_stop = True

    def register_user_activity(self) -> None:
        pass


class FakeDetails:
    def __init__(self) -> None:
        record = copy.deepcopy(sc.CONVERSATION_RECORD)
        self._record = record
        self.metadata = SimpleNamespace(**record["metadata"])
        self.transcript = record["transcript"]
        self.status = record["status"]

    def model_dump(self) -> dict[str, Any]:
        return copy.deepcopy(self._record)


class FakeElevenLabs:
    """The REST client: the record says ``end_call`` ran; the audio is a stub."""

    def __init__(self, **_: Any) -> None:
        self.conversational_ai = SimpleNamespace(
            conversations=SimpleNamespace(
                get=lambda conversation_id: FakeDetails(),
                audio=SimpleNamespace(get=lambda conversation_id: iter([b"ID3-g5"])),
            )
        )


async def main(out_dir: Path) -> None:
    global T0
    os.environ["ELEVENLABS_API_KEY"] = "g5-fake"
    os.environ["EVA_EN_USER_F"] = "agent_g5_fake"

    import eva.user_simulator.elevenlabs as eleven
    from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer
    from eva.models.agents import AgentConfig
    from eva.models.config import ModelConfig
    from eva.models.results import ConversationResult
    from eva.user_simulator.elevenlabs import ElevenLabsUserSimulator
    from eva.utils.hash_utils import get_dict_hash

    eleven.Conversation = FakeConversation  # ty: ignore
    eleven.ElevenLabs = FakeElevenLabs  # ty: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    agent_config_path = "configs/agents/airline_agent.yaml"
    agent = AgentConfig.from_yaml(agent_config_path)
    server = OpenAIRealtimeAssistantServer(
        current_date_time="2026-03-17 10:45 CST",
        pipeline_config=ModelConfig(
            s2s=sc.MODEL, s2s_params={"model": sc.MODEL, "api_key": "g5-fake"}
        ),
        agent=agent,
        agent_config_path=agent_config_path,
        scenario_db_path="data/airline_scenarios/1.1.2.json",
        output_dir=out_dir,
        port=PORT,
        conversation_id=sc.CONVERSATION_ID,
        language="en",
    )
    conn = FakeConn()
    server._create_client = lambda: SimpleNamespace(realtime=conn)  # ty: ignore

    async def execute(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(sc.TOOL_RESULT)

    server.tool_handler.execute = execute  # ty: ignore

    import eva.assistant.openai_realtime_server as realtime_server

    pads: list[tuple[float, str, int, int, bool, bool]] = []
    real_sync = realtime_server.sync_buffer_to_position

    def traced_sync(buffer: bytearray, target: int) -> None:
        track = "user" if buffer is server.user_audio_buffer else "assistant"
        pads.append(
            (
                round(time.time() - T0, 4),
                track,
                len(buffer),
                target,
                server._bot_speaking,
                server._user_speaking,
            )
        )
        real_sync(buffer, target)

    realtime_server.sync_buffer_to_position = traced_sync  # ty: ignore

    started_at = datetime.now()
    await server.start()
    goal = {
        "high_level_user_goal": "Change a flight.",
        "decision_tree": {
            key: "-"
            for key in (
                "must_have_criteria",
                "escalation_behavior",
                "nice_to_have_criteria",
                "negotiation_behavior",
                "resolution_condition",
                "failure_condition",
                "edge_cases",
            )
        },
        "information_required": "-",
        "starting_utterance": sc.UTTERANCES[0].text,
    }
    sim = ElevenLabsUserSimulator(
        current_date_time="2026-03-17 10:45 CST",
        persona_config={"user_persona_id": 1},
        goal=goal,
        server_url=f"ws://localhost:{PORT}/ws",
        output_dir=out_dir,
        agent_id=agent.id,
        timeout=620,
        language="en",
    )
    sim.on_conversation_ending = server.notify_conversation_ending

    T0 = time.time()
    try:
        reason = await asyncio.wait_for(sim.run_conversation(), timeout=620)
    except TimeoutError:
        reason = "time_limit_exceeded"
    stats = server.get_conversation_stats()
    deferred = await server.stop()
    if deferred is not None:
        await deferred
    ended_at = datetime.now()

    initial = json.loads((out_dir / "initial_scenario_db.json").read_text())
    final = json.loads((out_dir / "final_scenario_db.json").read_text())
    result = ConversationResult(
        record_id=sc.CONVERSATION_ID,
        completed=True,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=(ended_at - started_at).total_seconds(),
        output_dir=str(out_dir),
        audio_assistant_path=str(out_dir / "audio_assistant.wav"),
        audio_user_path=str(out_dir / "audio_user_clean.wav"),
        audio_mixed_path=str(out_dir / "audio_mixed.wav"),
        transcript_path=str(out_dir / "transcript.jsonl"),
        audit_log_path=str(out_dir / "audit_log.json"),
        pipecat_logs_path=str(out_dir / "framework_logs.jsonl"),
        user_simulator_logs_path=str(out_dir / "user_simulator_events.jsonl"),
        num_turns=stats.get("num_turns", 0),
        num_tool_calls=stats.get("num_tool_calls", 0),
        tools_called=stats.get("tools_called", []),
        conversation_ended_reason=reason,
        initial_scenario_db_hash=get_dict_hash(initial),
        final_scenario_db_hash=get_dict_hash(final),
    )
    (out_dir / "result.json").write_text(result.model_dump_json(indent=2))
    heard = FakeConversation.instances[0].heard if FakeConversation.instances else []
    meta = {
        "side": "eva",
        "clock": "virtual" if VIRTUAL else "wall",
        "t0": T0,
        "conversation_ended_reason": reason,
        "stats": stats,
        "caller_heard": heard,
        "agent_frames_received": conn.appended,
        "agent_client_events": conn.client_events,
        "pads": pads,
        "agent_frames": [(round(t - T0, 4), n, v) for t, n, v in conn.engine.frames],
        "agent_server_events": [(t - T0, kind) for t, kind in conn.engine.log],
    }
    (out_dir / "g5_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"eva side done: {reason}, {time.time() - T0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    if "--real-time" in sys.argv[2:]:
        VIRTUAL = False
        asyncio.run(main(out))
    else:
        call_vtime.EXECUTOR_DELAYS["start_session"] = sc.SIGNED_URL_DELAY_S
        with call_vtime.VirtualClock() as clock:
            clock.run(main(out))
