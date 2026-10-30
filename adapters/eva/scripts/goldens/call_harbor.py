"""The call golden, harbor's side: the same scripted call through harbor's port.

``RealtimeCall`` (bridge, recorder) with harbor's real transports, A2's
``openai.realtime`` and A3's ``elevenlabs.agents``, and A3's
``ElevenLabsCaller``, each with only its socket replaced:

- the Realtime socket carries ``call_scenario.FakeRealtime``, the fake agent
  EVA's side talks to, as JSON;
- the ElevenLabs socket plays ``call_scenario.CALLER_SCRIPT``, and the REST
  client (signed URL, conversation record, server audio) is a stub whose
  record says ``end_call`` ran.

The agent is opened with EVA's Realtime settings, the ones the driver will
use. ``tests/test_eva_call_golden.py`` runs :func:`run_harbor` and compares
it with EVA's record in ``tests/fixtures/call/g5``; run directly, this writes
the record directory for a look (or for EVA's processor)::

    uv run python adapters/eva/scripts/goldens/call_harbor.py OUT_DIR [--real-time]

By default on ``call_vtime.VirtualClock``, as EVA's side; ``--real-time`` for
the wall clock.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import call_scenario as sc  # noqa: E402
import call_vtime  # noqa: E402
from harbor_voice.ids import PartyId  # noqa: E402
from harbor_voice.providers.elevenlabs.agents import ElevenLabsAgentsTransport  # noqa: E402
from harbor_voice.providers.openai.realtime import OpenAIRealtimeTransport  # noqa: E402
from harbor_voice.providers.openai.realtime import RealtimeSettings  # noqa: E402
from harbor_voice.providers.tools import MappingExecutor  # noqa: E402
from harbor_voice.providers.tools import ToolRouter  # noqa: E402
from harbor_voice.providers.tools import ToolSpec  # noqa: E402

from eva_adapter.call import RealtimeCall  # noqa: E402
from eva_adapter.caller import ElevenLabsCaller  # noqa: E402

PORT = 10123
EVA_REALTIME = RealtimeSettings(
    voice="marin",
    vad_threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=200,
    transcription_model="whisper-1",
    language=None,
    noise_reduction=None,
    tool_choice=None,
    audio_format="pcm24k",
    opening_text=sc.OPENING_TEXT,
    tool_flow="serial",
    tool_output_ascii=False,
    extended_events=True,
)
"""EVA's session (``openai_realtime_server.py:185-226``) in harbor's settings."""


class _Socket:
    """A websocket: ``send``, ``recv``, async iteration, ``close``."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.close_code: int | None = None

    def push(self, event: dict[str, Any]) -> None:
        self.queue.put_nowait(json.dumps(event))

    async def recv(self) -> str:
        item = await self.queue.get()
        if item is None:
            raise ConnectionError("closed")
        return item

    def __aiter__(self) -> _Socket:
        return self

    async def __anext__(self) -> str:
        item = await self.queue.get()
        if item is None:
            self.queue.put_nowait(None)
            raise StopAsyncIteration
        return item

    async def close(self) -> None:
        if self.close_code is None:
            self.close_code = 1000
        self.queue.put_nowait(None)


class RealtimeSocket(_Socket):
    """The Realtime API's socket, with :class:`call_scenario.FakeRealtime` behind it."""

    def __init__(self, t0: float) -> None:
        super().__init__()
        self.t0 = t0
        self.engine = sc.FakeRealtime(self.push)
        self.client_events: list[tuple[float, str]] = []
        self.appended = 0
        self.engine.connected()

    async def send(self, data: str) -> None:
        event = json.loads(data)
        if event["type"] == "input_audio_buffer.append":
            self.appended += 1
        else:
            self.client_events.append((time.time() - self.t0, event["type"]))
        self.engine.on_client(event)

    async def close(self) -> None:
        self.engine.close()
        await super().close()


class ElevenLabsSocket(_Socket):
    """ElevenLabs' side of the caller's socket: the scripted caller."""

    def __init__(self, t0: float) -> None:
        super().__init__()
        self.t0 = t0
        self.heard: list[tuple[float, int, int]] = []
        self._script: asyncio.Task[None] | None = None

    async def send(self, data: str) -> None:
        message = json.loads(data)
        if "user_audio_chunk" in message:
            import base64

            audio = base64.b64decode(message["user_audio_chunk"])
            voiced = sum(1 for b in audio if b not in (0xFF, 0x7F))
            self.heard.append((time.time() - self.t0, len(audio), voiced))
        elif message.get("type") == "conversation_initiation_client_data":
            self.push(sc.METADATA)
            self._script = asyncio.create_task(self._play())

    async def _play(self) -> None:
        await sc.play_caller_script(self.t0, self.push)
        self.close_code = 1000
        self.queue.put_nowait(None)  # the server closes the socket

    async def close(self) -> None:
        if self._script is not None:
            self._script.cancel()
        await super().close()


class AgentsAPIStub:
    """The ElevenLabs REST calls the caller makes."""

    api_base = "https://api.elevenlabs.io"

    def require_key(self) -> str:
        return "g5-fake"

    async def signed_url(self, agent_id: str) -> str:
        await asyncio.sleep(sc.SIGNED_URL_DELAY_S)
        return "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=g5&conversation_signature=sig"

    async def conversation(self, conversation_id: str) -> dict[str, Any]:
        return copy.deepcopy(sc.CONVERSATION_RECORD)

    async def conversation_audio(self, conversation_id: str) -> bytes:
        return b"ID3-g5"


def run_harbor_virtual(out_dir: Path, *, epoch0: float | None = None) -> dict[str, Any]:
    """:func:`run_harbor` on the virtual clock, starting at ``epoch0`` if given.

    Pass the ``t0`` of EVA's record being compared with (``g5_meta.json``):
    the call then starts at EVA's instant, to the sub-millisecond, and
    same-millisecond ties fall as they fell for EVA (see ``VirtualClock``).
    """
    with call_vtime.VirtualClock(epoch0) as clock:
        return clock.run(run_harbor(out_dir, clock="virtual"))


async def run_harbor(out_dir: Path, *, clock: str = "wall") -> dict[str, Any]:
    """Run the scripted call through harbor's port; return ``g5_meta.json``'s content."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sockets: dict[str, Any] = {}

    async def connect_realtime(url: str, **_: Any) -> RealtimeSocket:
        sockets["realtime"] = RealtimeSocket(t0)
        return sockets["realtime"]

    async def connect_elevenlabs(url: str, **_: Any) -> ElevenLabsSocket:
        sockets["elevenlabs"] = ElevenLabsSocket(t0)
        return sockets["elevenlabs"]

    tools = ToolRouter(
        MappingExecutor(
            specs=[ToolSpec(sc.TOOL_NAME, sc.TOOL_DESCRIPTION, sc.TOOL_SCHEMA)],
            handlers={sc.TOOL_NAME: lambda **_: copy.deepcopy(sc.TOOL_RESULT)},
        )
    )
    agent = OpenAIRealtimeTransport(
        party=PartyId("agent"),
        model=sc.MODEL,
        system_prompt="G5",
        tools=tools,
        settings=EVA_REALTIME,
        api_key="g5-fake",
        connect=connect_realtime,
    )
    caller_transport = ElevenLabsAgentsTransport(
        agent_id="agent_g5_fake",
        api=AgentsAPIStub(),  # ty: ignore[invalid-argument-type]
        connect=connect_elevenlabs,
    )
    call = RealtimeCall(
        agent=agent,
        caller_transport=caller_transport,
        out_dir=out_dir,
        conversation_id=sc.CONVERSATION_ID,
        model=sc.MODEL,
    )
    import eva_adapter.recorder as recorder_module

    pads: list[tuple[float, str, int, int, bool, bool]] = []
    real_sync = recorder_module.sync_buffer_to_position
    recorder = call.recorder

    def traced_sync(buffer: bytearray, target: int) -> None:
        track = "user" if buffer is recorder.user_audio_buffer else "assistant"
        pads.append(
            (
                round(time.time() - t0, 4),
                track,
                len(buffer),
                target,
                recorder._bot_speaking,
                recorder._user_speaking,
            )
        )
        real_sync(buffer, target)

    recorder_module.sync_buffer_to_position = traced_sync  # ty: ignore
    caller = ElevenLabsCaller(
        transport=caller_transport,
        prompt="G5",
        events=call.events,
        out_dir=out_dir,
        bridge=call.bridge,
        server_url=f"ws://localhost:{PORT}/ws",
    )
    t0 = time.time()
    try:
        outcome = await call.run(caller)
    finally:
        recorder_module.sync_buffer_to_position = real_sync
    realtime = sockets.get("realtime")
    elevenlabs = sockets.get("elevenlabs")
    meta = {
        "side": "harbor",
        "clock": clock,
        "t0": t0,
        "conversation_ended_reason": outcome.conversation_ended_reason,
        "stats": outcome.stats,
        "error": outcome.error,
        "agent_error": outcome.agent_error,
        "caller_heard": elevenlabs.heard if elevenlabs else [],
        "agent_frames_received": realtime.appended if realtime else 0,
        "agent_client_events": realtime.client_events if realtime else [],
        "pads": pads,
        "agent_frames": [(round(t - t0, 4), n, v) for t, n, v in realtime.engine.frames]
        if realtime
        else [],
        "agent_server_events": (
            [(t - t0, kind) for t, kind in realtime.engine.log] if realtime else []
        ),
    }
    (out_dir / "g5_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    target = Path(sys.argv[1])
    if "--real-time" in sys.argv[2:]:
        result = asyncio.run(run_harbor(target))
    else:
        result = run_harbor_virtual(target)
    print(result["conversation_ended_reason"], file=sys.stderr)
