"""ElevenLabs Agents' transport against a fake server on localhost.

The wire must be the ElevenLabs Python SDK 2.53's (EVA's pin). The ``sdk_*``
helpers below are that SDK's message builders transcribed from
``elevenlabs/conversational_ai/conversation.py`` — kept verbatim, including
the ``str``-``Enum`` message type — and every test compares the bytes our
transport sends with what they produce. Nothing touches the real API.
"""

from __future__ import annotations

import asyncio
import base64
import json
import urllib.parse
from collections.abc import Awaitable
from collections.abc import Callable
from enum import Enum
from typing import Any

import httpx
import pytest
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve

from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import OptionalOp
from harbor_voice.ids import PartyId
from harbor_voice.providers import MappingExecutor
from harbor_voice.providers import ToolRouter
from harbor_voice.providers import ToolSpec
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.elevenlabs import agents as el
from harbor_voice.providers.elevenlabs import register_all
from harbor_voice.providers.registry import ProviderRegistry
from harbor_voice.transport.stream import OutputAudio

pytestmark = pytest.mark.unit

AGENT = "agent_test"
KEY = "test-key"


# ----------------------------------------------- SDK 2.53, transcribed verbatim


class ClientToOrchestratorEvent(str, Enum):  # conversation.py:25-38
    PONG = "pong"
    CLIENT_TOOL_RESULT = "client_tool_result"
    CONVERSATION_INITIATION_CLIENT_DATA = "conversation_initiation_client_data"
    FEEDBACK = "feedback"
    CONTEXTUAL_UPDATE = "contextual_update"
    USER_MESSAGE = "user_message"
    USER_ACTIVITY = "user_activity"
    MULTIMODAL_MESSAGE = "multimodal_message"


def sdk_initiation(
    extra_body: dict[str, Any],
    override: dict[str, Any],
    dynamic_variables: dict[str, Any],
    user_id: str | None = None,
) -> str:  # _create_initiation_message, conversation.py:460-473
    return json.dumps(
        {
            "type": "conversation_initiation_client_data",
            "custom_llm_extra_body": extra_body,
            "conversation_config_override": override,
            "dynamic_variables": dynamic_variables,
            "source_info": {
                "source": "python_sdk",
                "version": "2.53.0",
            },
            **({"user_id": user_id} if user_id else {}),
        }
    )


def sdk_audio(audio: bytes) -> str:  # input_callback, conversation.py:830-838
    return json.dumps(
        {
            "user_audio_chunk": base64.b64encode(audio).decode(),
        }
    )


def sdk_pong(event: dict[str, Any]) -> str:  # handle_ping, conversation.py:898-906
    return json.dumps(
        {
            "type": "pong",
            "event_id": event["event_id"],
        }
    )


def sdk_user_activity() -> str:  # UserActivityClientToOrchestratorEvent.to_dict, :58-65
    return json.dumps({"type": ClientToOrchestratorEvent.USER_ACTIVITY})


def sdk_contextual_update(text: str) -> str:  # :68-76
    return json.dumps(
        {"type": ClientToOrchestratorEvent.CONTEXTUAL_UPDATE, "text": text}
    )


def sdk_user_message(text: str) -> str:  # :47-55
    return json.dumps({"type": ClientToOrchestratorEvent.USER_MESSAGE, "text": text})


def sdk_tool_error(
    tool_name: str, parameters: dict[str, Any]
) -> str:  # ClientTools, :281-323
    try:
        raise ValueError(f"Tool '{tool_name}' is not registered")
    except Exception as e:
        response = {
            "type": "client_tool_result",
            "tool_call_id": parameters.get("tool_call_id"),
            "result": str(e),
            "is_error": True,
        }
    return json.dumps(response)


def sdk_signed_url(signed_url: str) -> str:  # _get_signed_url, conversation.py:440-445
    parsed = urllib.parse.urlparse(signed_url)
    existing_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    existing_params.extend([("source", "python_sdk"), ("version", "2.53.0")])
    return urllib.parse.urlunparse(
        parsed._replace(
            query=urllib.parse.urlencode(existing_params, quote_via=urllib.parse.quote)
        )
    )


# ------------------------------------------------------------- server messages


def metadata(conversation_id: str = "conv_1") -> str:
    return json.dumps(
        {
            "type": "conversation_initiation_metadata",
            "conversation_initiation_metadata_event": {
                "conversation_id": conversation_id,
                "agent_output_audio_format": "pcm_16000",
                "user_input_audio_format": "ulaw_8000",
            },
        }
    )


def audio(event_id: int, pcm: bytes) -> str:
    return json.dumps(
        {
            "type": "audio",
            "audio_event": {
                "audio_base_64": base64.b64encode(pcm).decode(),
                "event_id": event_id,
            },
        }
    )


def ping(event_id: int, ping_ms: int | None = None) -> str:
    return json.dumps(
        {"type": "ping", "ping_event": {"event_id": event_id, "ping_ms": ping_ms}}
    )


def typed(kind: str, key: str, **fields: Any) -> str:
    return json.dumps({"type": kind, key: fields})


# ------------------------------------------------------------------ the rig

Script = Callable[[ServerConnection, list[str]], Awaitable[None]]


class Rig:
    """A fake ElevenLabs: signed-URL REST via MockTransport, a websocket on 127.0.0.1."""

    def __init__(self, script: Script, *, signature: str = "sig%2Fa b") -> None:
        self.script = script
        self.signature = signature
        self.received: list[str] = []
        self.requests: list[httpx.Request] = []
        self.paths: list[str] = []
        self.close_codes: list[int | None] = []
        self.port = 0
        self.done = asyncio.Event()
        self.rest: dict[str, Callable[[httpx.Request], httpx.Response]] = {}

    async def _handler(self, ws: ServerConnection) -> None:
        self.paths.append(ws.request.path if ws.request else "")
        try:
            await self.script(ws, self.received)
        finally:
            self.done.set()

    def _http(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/convai/conversation/get-signed-url":
            url = (
                f"ws://127.0.0.1:{self.port}/v1/convai/conversation"
                f"?agent_id={AGENT}&conversation_signature={self.signature}"
            )
            return httpx.Response(200, json={"signed_url": url})
        for prefix, handler in self.rest.items():
            if path.startswith(prefix):
                return handler(request)
        return httpx.Response(404, json={"detail": "not found"})

    def transport(self, **kw: Any) -> el.ElevenLabsAgentsTransport:
        client = httpx.AsyncClient(transport=httpx.MockTransport(self._http))
        api = el.AgentsAPI(api_key=KEY, client=client)
        kw.setdefault("dynamic_variables", {"prompt": "You are a caller."})
        return el.ElevenLabsAgentsTransport(agent_id=AGENT, api=api, **kw)

    async def __aenter__(self) -> Rig:
        self._server = await serve(self._handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._server.close()
        await self._server.wait_closed()


async def collect(transport: el.ElevenLabsAgentsTransport) -> list[Any]:
    out: list[Any] = []
    async for event in transport.events():
        out.append(event)
    return out


async def recv_json(ws: ServerConnection, received: list[str]) -> dict[str, Any]:
    raw = await ws.recv()
    assert isinstance(raw, str)
    received.append(raw)
    return json.loads(raw)


# ------------------------------------------------------------------- connect


class TestConnect:
    async def test_signed_url_request_is_the_sdks(self) -> None:
        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)

        async with Rig(script) as rig:
            transport = rig.transport()
            await transport.open()
            await rig.done.wait()
            await transport.close()

        (request,) = rig.requests
        assert request.method == "GET"
        assert request.url.path == "/v1/convai/conversation/get-signed-url"
        # The SDK sends agent_id and drops its unset params (remove_none_from_dict).
        assert dict(request.url.params) == {"agent_id": AGENT}
        assert request.headers["xi-api-key"] == KEY
        assert request.headers["user-agent"] == "elevenlabs/2.53.0"
        assert request.headers["x-fern-sdk-name"] == "elevenlabs"
        assert request.headers["x-fern-sdk-version"] == "2.53.0"
        assert request.headers["x-fern-language"] == "Python"

    async def test_websocket_url_appends_source_and_version_as_the_sdk_does(
        self,
    ) -> None:
        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)

        async with Rig(script) as rig:
            transport = rig.transport()
            await transport.open()
            await rig.done.wait()
            await transport.close()
            signed = (
                f"ws://127.0.0.1:{rig.port}/v1/convai/conversation"
                f"?agent_id={AGENT}&conversation_signature={rig.signature}"
            )
        (path,) = rig.paths
        expected = sdk_signed_url(signed)
        assert path == expected[expected.index("/v1/") :]
        assert path.endswith("&source=python_sdk&version=2.53.0")
        # Re-encoding is the SDK's too: "%2F" decodes and "a b" becomes "a%20b".
        assert "conversation_signature=sig%2Fa%20b" in path

    def test_with_sdk_source_matches_the_sdk_on_awkward_queries(self) -> None:
        for url in (
            "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=a&conversation_signature=x+y%2B",
            "wss://h/p?blank=&agent_id=a",
            "wss://h/p",
        ):
            assert el.with_sdk_source(url) == sdk_signed_url(url)

    def test_public_url_is_the_sdks_wss_url(self) -> None:
        assert el.public_url("agent x") == (
            "wss://api.elevenlabs.io/v1/convai/conversation"
            "?agent_id=agent%20x&source=python_sdk&version=2.53.0"
        )

    async def test_initiation_message_is_byte_identical(self) -> None:
        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)

        prompt = "Tu es un client — «pressé».\nSpell codes one by one."
        async with Rig(script) as rig:
            transport = rig.transport(dynamic_variables={"prompt": prompt})
            await transport.open()
            await rig.done.wait()
            await transport.close()
        assert rig.received[0] == sdk_initiation({}, {}, {"prompt": prompt})

    async def test_initiation_carries_overrides_and_user_id_as_the_sdk_does(
        self,
    ) -> None:
        transport = el.ElevenLabsAgentsTransport(
            agent_id=AGENT,
            dynamic_variables={"prompt": "p"},
            conversation_config_override={"tts": {"voice_id": "v"}},
            custom_llm_extra_body={"x": 1},
            user_id="u1",
            api=el.AgentsAPI(api_key=KEY),
        )
        assert transport.initiation_message() == sdk_initiation(
            {"x": 1}, {"tts": {"voice_id": "v"}}, {"prompt": "p"}, "u1"
        )

    async def test_missing_key_is_an_auth_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        transport = el.ElevenLabsAgentsTransport(agent_id=AGENT)
        with pytest.raises(ProviderError) as info:
            await transport.open()
        assert info.value.kind is ProviderFailure.AUTH

    async def test_rejected_signed_url_is_classified(self) -> None:
        def reject(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": "bad key"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
        transport = el.ElevenLabsAgentsTransport(
            agent_id=AGENT, api=el.AgentsAPI(api_key=KEY, client=client)
        )
        with pytest.raises(ProviderError) as info:
            await transport.open()
        assert info.value.kind is ProviderFailure.AUTH
        assert KEY not in str(info.value)


# -------------------------------------------------------------------- stream


class TestStream:
    async def test_full_sequence_in_arrival_order(self) -> None:
        pcm_a, pcm_b, pcm_c = b"\x01\x00" * 160, b"\x02\x00" * 160, b"\x03\x00" * 160

        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)  # initiation
            await ws.send(metadata("conv_42"))
            await ws.send(ping(1, 25))
            assert await recv_json(ws, received) == {"type": "pong", "event_id": 1}
            await ws.send(audio(1, pcm_a))
            await ws.send(
                typed(
                    "agent_response",
                    "agent_response_event",
                    agent_response="  Hi there. ",
                )
            )
            await ws.send(
                typed(
                    "user_transcript",
                    "user_transcription_event",
                    user_transcript=" Hello! ",
                )
            )
            await ws.send(
                typed(
                    "agent_response_correction",
                    "agent_response_correction_event",
                    original_agent_response=" Hi there. ",
                    corrected_agent_response=" Hi ",
                )
            )
            await ws.send(typed("interruption", "interruption_event", event_id=3))
            await ws.send(audio(2, pcm_b))  # at or below the interruption: dropped
            await ws.send(audio(3, pcm_b))  # dropped
            await ws.send(audio(4, pcm_c))
            await ws.send(typed("vad_score", "vad_score_event", vad_score=0.9))
            await ws.send(
                json.dumps(
                    {
                        "type": "client_tool_call",
                        "client_tool_call": {
                            "tool_name": "lookup",
                            "tool_call_id": "tc_1",
                            "parameters": {"q": "x"},
                        },
                    }
                )
            )
            await recv_json(ws, received)  # the tool result
            await ws.close()

        async with Rig(script) as rig:
            transport = rig.transport()
            await transport.open()
            events = await collect(transport)

        kinds = [type(e).__name__ for e in events]
        assert kinds == [
            "OutputAudio",
            "AgentResponse",
            "UserTranscript",
            "AgentResponseCorrection",
            "Interruption",
            "OutputAudio",
            "ClientToolCall",
            "SessionEnded",
        ]
        first, response, transcript, correction, interruption, second, tool, ended = (
            events
        )
        assert (
            isinstance(first, OutputAudio)
            and first.pcm == pcm_a
            and first.utterance == "1"
        )
        assert isinstance(second, OutputAudio) and second.pcm == pcm_c
        assert response.text == "Hi there."
        assert transcript.text == "Hello!"
        assert (correction.original, correction.corrected) == ("Hi there.", "Hi")
        assert interruption.event_id == 3
        assert (tool.tool_call_id, tool.name, tool.parameters) == (
            "tc_1",
            "lookup",
            {"q": "x"},
        )
        assert ended.reason == "server_closed"
        assert ended.code == 1000
        assert transport.conversation_id == "conv_42"
        assert transport.pings == 1 and transport.ping_ms == [25]
        assert transport.ignored == {"vad_score": 1}
        assert transport.errors == []
        # Unregistered client tool: the SDK's error result, byte for byte.
        assert rig.received[-1] == sdk_tool_error(
            "lookup", {"tool_call_id": "tc_1", "q": "x"}
        )
        assert rig.received[1] == sdk_pong({"event_id": 1, "ping_ms": 25})

    async def test_pong_goes_out_before_the_next_message_is_handled(self) -> None:
        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)
            for n in (1, 2, 3):
                await ws.send(ping(n))
            for n in (1, 2, 3):
                assert await recv_json(ws, received) == {"type": "pong", "event_id": n}
            await ws.close()

        async with Rig(script) as rig:
            transport = rig.transport()
            await transport.open()
            await collect(transport)
        assert rig.received[1:] == [sdk_pong({"event_id": n}) for n in (1, 2, 3)]

    async def test_client_messages_are_the_sdks(self) -> None:
        sent_audio = bytes(range(256)) * 8

        async def script(ws: ServerConnection, received: list[str]) -> None:
            for _ in range(5):
                await recv_json(ws, received)
            await ws.close()

        async with Rig(script) as rig:
            transport = rig.transport()
            await transport.open()
            await transport.send_audio(sent_audio)
            await transport.send_user_activity()
            await transport.send_contextual_update("ctx")
            await transport.send_text("hello")
            await collect(transport)
        assert rig.received[1:] == [
            sdk_audio(sent_audio),
            sdk_user_activity(),
            sdk_contextual_update("ctx"),
            sdk_user_message("hello"),
        ]
        assert transport.sent_chunks == 1

    async def test_tx_takes_mulaw_frames_only(self) -> None:
        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)
            await recv_json(ws, received)
            await ws.close()

        mulaw = AudioFormat(sample_rate_hz=8_000, encoding="mulaw")
        async with Rig(script) as rig:
            transport = rig.transport()
            await transport.open()
            with pytest.raises(ProviderError):
                await transport.tx(
                    AudioFrame(
                        pcm=bytes(320), seq=0, party=PartyId("a"), t_produced=0.0
                    )
                )
            frame = AudioFrame(
                pcm=b"\xff" * 160, seq=0, party=PartyId("a"), t_produced=0.0, fmt=mulaw
            )
            await transport.tx(frame)
            await collect(transport)
        assert rig.received[1] == sdk_audio(b"\xff" * 160)

    async def test_audio_after_the_end_is_dropped_not_raised(self) -> None:
        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)
            await ws.close()

        async with Rig(script) as rig:
            transport = rig.transport()
            await transport.open()
            await collect(transport)
            await transport.send_audio(b"\xff" * 160)
        assert transport.dropped_chunks == 1
        with pytest.raises(Exception, match="not connected"):
            await transport.send_user_activity()

    async def test_a_message_the_sdk_would_fail_on_ends_the_session(self) -> None:
        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)
            await ws.send(
                json.dumps({"type": "agent_response", "agent_response_event": {}})
            )
            await ws.wait_closed()

        async with Rig(script) as rig:
            transport = rig.transport()
            await transport.open()
            events = await collect(transport)
        (ended,) = events
        assert ended.reason == "protocol_error"
        assert isinstance(transport.pump_error, KeyError)

    async def test_close_is_a_normal_closure_and_ends_the_stream(self) -> None:
        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)
            await ws.send(metadata())
            await ws.wait_closed()
            rig.close_codes.append(ws.close_code)

        async with Rig(script) as rig:
            transport = rig.transport()
            await transport.open()
            reader = asyncio.create_task(collect(transport))
            while transport.conversation_id is None:
                await asyncio.sleep(0.01)
            await transport.close()
            await transport.close()  # idempotent
            events = await reader
            await rig.done.wait()
        assert [type(e).__name__ for e in events] == ["SessionEnded"]
        assert events[0].reason == "client_closed"
        assert rig.close_codes == [1000]
        assert not transport.is_open

    async def test_registered_client_tool_is_answered_by_the_router(self) -> None:
        spec = ToolSpec(name="lookup", description="d", input_schema={"type": "object"})
        router = ToolRouter(
            MappingExecutor(
                specs=[spec], handlers={"lookup": lambda q: {"text": f"found {q}"}}
            ),
            sleep=lambda _s: asyncio.sleep(0),
        )

        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)
            await ws.send(
                json.dumps(
                    {
                        "type": "client_tool_call",
                        "client_tool_call": {
                            "tool_name": "lookup",
                            "tool_call_id": "tc_9",
                            "parameters": {"q": "x"},
                        },
                    }
                )
            )
            await recv_json(ws, received)
            await ws.close()

        async with Rig(script) as rig:
            transport = rig.transport(tools=router)
            await transport.open()
            await collect(transport)
        assert json.loads(rig.received[-1]) == {
            "type": "client_tool_result",
            "tool_call_id": "tc_9",
            "result": "found x",
            "is_error": False,
        }

    async def test_a_format_the_server_disagrees_on_is_recorded(self) -> None:
        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation_initiation_metadata",
                        "conversation_initiation_metadata_event": {
                            "conversation_id": "c",
                            "agent_output_audio_format": "pcm_24000",
                            "user_input_audio_format": "ulaw_8000",
                        },
                    }
                )
            )
            await ws.close()

        async with Rig(script) as rig:
            transport = rig.transport()
            await transport.open()
            await collect(transport)
        assert transport.errors == [
            "server says agent_output_audio_format='pcm_24000', transport expects 'pcm_16000'"
        ]


# ------------------------------------------------------------ declaration


def test_declares_what_it_implements() -> None:
    transport = el.ElevenLabsAgentsTransport(
        agent_id=AGENT, api=el.AgentsAPI(api_key=KEY)
    )
    caps = transport.capabilities
    assert transport.check_contract() == []
    assert caps.supports(OptionalOp.EVENT_STREAM) and caps.supports(
        OptionalOp.SEND_TEXT
    )
    assert caps.audio_in == AudioFormat(sample_rate_hz=8_000, encoding="mulaw")
    assert caps.audio_out == AudioFormat(sample_rate_hz=16_000)


def test_registered_and_built_from_a_spec_string() -> None:
    registry = ProviderRegistry()
    register_all(registry)
    registration = registry.lookup("elevenlabs.agents")
    assert registration.spec.credential_provider == "elevenlabs"
    transport = registry.create(
        "elevenlabs.agents/agent_abc", modality=Modality.S2S, party="caller", tools=None
    )
    assert isinstance(transport, el.ElevenLabsAgentsTransport)
    assert transport.agent_id == "agent_abc"


# ---------------------------------------------------------------- the record


class TestPoll:
    async def test_evas_schedule_until_the_budget(self) -> None:
        slept: list[float] = []

        async def sleep(s: float) -> None:
            slept.append(s)

        async def fetch() -> dict[str, Any]:
            return {"status": "processing"}

        outcome = await el.poll_conversation(fetch, sleep=sleep)
        assert slept[:4] == [2.0, 4.0, 8.0, 10.0]
        assert set(slept[3:]) == {10.0}
        assert outcome.attempts == len(slept) == 32
        assert outcome.waited_s == 304.0
        assert outcome.verdict is None

    async def test_stops_when_decide_answers_and_shows_every_record(self) -> None:
        records = iter(
            [{"status": "processing"}, {"status": "processing"}, {"status": "done"}]
        )
        seen: list[dict[str, Any]] = []

        async def fetch() -> dict[str, Any]:
            return next(records)

        async def sleep(_s: float) -> None:
            return None

        outcome = await el.poll_conversation(fetch, sleep=sleep, on_record=seen.append)
        assert outcome.verdict is True and outcome.attempts == 3
        assert len(seen) == 3

    async def test_record_and_audio_requests(self) -> None:
        async def script(ws: ServerConnection, received: list[str]) -> None:
            await recv_json(ws, received)
            await ws.send(metadata("conv_7"))
            await ws.close()

        async with Rig(script) as rig:
            rig.rest["/v1/convai/conversations/conv_7/audio"] = lambda r: (
                httpx.Response(200, content=b"ID3mp3")
            )
            rig.rest["/v1/convai/conversations/conv_7"] = lambda r: httpx.Response(
                200, json={"status": "done", "transcript": []}
            )
            transport = rig.transport()
            await transport.open()
            await collect(transport)

            async def no_sleep(_s: float) -> None:
                return None

            outcome = await transport.conversation_record(sleep=no_sleep)
            assert await transport.fetch_audio() == b"ID3mp3"
        assert outcome.record == {"status": "done", "transcript": []}
        paths = [r.url.path for r in rig.requests[1:]]
        assert paths == [
            "/v1/convai/conversations/conv_7",
            "/v1/convai/conversations/conv_7/audio",
        ]
        assert all(dict(r.url.params) == {} for r in rig.requests[1:])


def test_usage_records_from_a_conversation_record() -> None:
    record = {
        "agent_id": "agent_x",
        "metadata": {
            "call_duration_secs": 344,
            "charging": {
                "llm_usage": {
                    "irreversible_generation": {
                        "model_usage": {
                            "gpt-5.1": {
                                "input": {"tokens": 1000, "price": 0.1},
                                "input_cache_read": {"tokens": 400, "price": 0.01},
                                "input_cache_write": {"tokens": 0},
                                "output_total": {"tokens": 120, "price": 0.2},
                            },
                            "gemini-2.5-flash": {
                                "input": {"tokens": 50},
                                "output_total": {"tokens": 5},
                            },
                        }
                    },
                    "initiated_generation": {
                        "model_usage": {
                            "gpt-5.1": {
                                "input": {"tokens": 1500},
                                "output_total": {"tokens": 200},
                            }
                        }
                    },
                },
                "tts_usage": {
                    "primary_tts_model": "eleven_v3_conversational",
                    "total_characters": 900,
                    "total_audio_output_seconds": 61.5,
                },
                "asr_usage": {
                    "asr_model": "scribe_realtime",
                    "total_transcription_calls": 14,
                    "total_audio_input_seconds": 300.0,
                },
            },
        },
    }
    records = {(r.component, r.model): r for r in el.usage_records(record)}
    call = records[("caller", "agent_x")]
    assert call.seconds == 344.0 and call.requests == 1
    gpt = records[("caller.llm", "gpt-5.1")]
    assert (gpt.input_tokens, gpt.cached_input_tokens, gpt.output_tokens) == (
        1400,
        400,
        120,
    )
    assert records[("caller.llm", "gemini-2.5-flash")].input_tokens == 50
    assert records[("caller.llm.initiated", "gpt-5.1")].output_tokens == 200
    assert records[("caller.tts", "eleven_v3_conversational")].characters == 900
    assert records[("caller.asr", "scribe_realtime")].requests == 14


def test_usage_records_tolerate_a_bare_record() -> None:
    (only,) = el.usage_records({"metadata": {}})
    assert only.component == "caller" and only.seconds is None
