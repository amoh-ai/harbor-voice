"""Deepgram's Voice Agent API as a :class:`~harbor_voice.transport.base.VoiceTransport`.

One websocket to ``wss://agent.deepgram.com/v1/agent/converse``
(https://developers.deepgram.com/reference/voice-agent/voice-agent):

- wait for ``Welcome``, send ``Settings``, wait for ``SettingsApplied``;
  nothing else may be sent before it
  (https://developers.deepgram.com/docs/voice-agent-message-flow);
- audio goes both ways as binary messages, raw, in one format
  (:attr:`AgentSettings.audio_format`, 24 kHz PCM16 by default, as the
  ``Settings`` example sends it);
- Deepgram runs the whole agent: it listens with Flux (``flux-general-en``,
  which decides when the caller's turn ended), thinks with a managed LLM
  (the ``model``; ``open_ai`` ``gpt-5.6-luna`` by default) and speaks with
  Aura (``aura-2-thalia-en``, the voice ``deepgram.tts`` uses)
  (https://developers.deepgram.com/docs/configure-voice-agent);
- tools are declared as ``agent.think.functions`` with no ``endpoint``, so
  each call comes to us as a ``FunctionCallRequest`` with
  ``client_side: true``. It runs through the :class:`ToolRouter` beside the
  receive loop and goes back as a ``FunctionCallResponse`` whose ``content``
  is :meth:`ToolResult.as_text`. Deepgram continues on its own; there is no
  ``response.create``. A ``FunctionCallCancelled`` call is dropped unanswered,
  as documented (https://developers.deepgram.com/docs/voice-agent-function-call-request).

It declares ``EVENT_STREAM``:

- binary audio is :class:`OutputAudio`, and an assistant
  ``ConversationText`` is :class:`OutputTranscript`, both tagged with the
  agent turn they belong to. Deepgram sends no turn IDs, so a turn starts at
  the first of either after ``AgentAudioDone`` or ``UserStartedSpeaking``;
- ``UserStartedSpeaking`` (barge-in) is ``InputSpeech(started=True)``. There
  is no position in our audio and no "stopped" event, so ``audio_ms`` is
  ``None`` and nothing ever reports speech stopping.

With :attr:`AgentSettings.extended_events`, a user ``ConversationText`` is an
:class:`InputTranscript` and an answered call a :class:`ToolCall`.

Not ``TRUNCATE``: the API takes no "this much was heard". Usage is the
session's length in seconds, which is how the API is billed; Deepgram sends
no usage message, so it is measured here from ``SettingsApplied`` to close.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import override

from harbor_voice._deps import websockets as _websockets
from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import OptionalOp
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.ids import PartyId
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderRef
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.deepgram._common import auth_header
from harbor_voice.providers.deepgram._common import encoding
from harbor_voice.providers.deepgram._common import socket_error
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import InputTranscript
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import StreamEvent
from harbor_voice.transport.stream import ToolCall

NAME = "deepgram.agent"
DEFAULT_MODEL = "gpt-5.6-luna"
URL = "wss://agent.deepgram.com/v1/agent/converse"


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """What goes into ``Settings`` besides the prompt, the tools and the model."""

    think_provider: str = "open_ai"
    """``agent.think.provider.type`` for the managed LLM named by ``model``:
    ``open_ai``, ``anthropic``, ``google``, ``deepgram``…"""
    listen_model: str = "flux-general-en"
    voice: str = "aura-2-thalia-en"
    """``agent.speak.provider.model``: an Aura model, or a Flux TTS one."""
    audio_format: AudioFormat = CANONICAL
    """Both directions. Linear16, μ-law or A-law, mono."""
    greeting: str | None = None
    """Spoken by the agent as the call opens, so it talks first."""
    extended_events: bool = False
    """Also yield :class:`InputTranscript` and :class:`ToolCall` from
    ``events()``. Off by default: a stepped consumer rejects kinds it does not
    know."""


DEFAULT_SETTINGS = AgentSettings()


def _version(model: str) -> str:
    """Flux models (listen and speak) are on Deepgram's ``v2``, the rest ``v1``."""
    return "v2" if model.startswith("flux") else "v1"


KEEPALIVE_S = 5.0
"""Seconds between ``KeepAlive`` messages; Deepgram's idle limit is about 10."""


class DeepgramAgentTransport(VoiceTransport):
    """One Deepgram Voice Agent session over a websocket."""

    def __init__(
        self,
        *,
        party: PartyId,
        model: str = DEFAULT_MODEL,
        system_prompt: str = "",
        tools: ToolRouter | None = None,
        settings: AgentSettings = DEFAULT_SETTINGS,
        api_key: str | None = None,
        url: str = URL,
        clock: SessionClock | None = None,
        connect: Any = None,
    ) -> None:
        encoding(settings.audio_format)
        self._party = party
        self._model = model
        self._system_prompt = system_prompt
        self._tools = tools
        self._settings = settings
        self._format = settings.audio_format
        self._api_key = api_key
        self._url = url
        self._clock = clock or SessionClock()
        self._connect = connect
        self._ws: Any = None
        self._open = False
        self._pump: asyncio.Task[None] | None = None
        self._keepalive: asyncio.Task[None] | None = None
        self._inbox: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._calls: dict[str, asyncio.Task[None]] = {}
        self._turns = 0
        self._turn: str | None = None
        self._turn_spoke = False
        self._seq = 0
        self._started: float | None = None
        self._seconds = 0.0
        self.request_id: str | None = None
        self.errors: list[dict[str, Any]] = []
        """``Error`` messages the server sent after the session opened."""
        self.pump_error: Exception | None = None

    # ------------------------------------------------------------ declaration

    @property
    @override
    def capabilities(self) -> Capabilities:
        return Capabilities(
            duplex=frozenset({Duplex.HALF, Duplex.FULL}),
            pipeline=Pipeline.S2S,
            turn_control=TurnControl.NATIVE_VAD,
            exposes_asr=False,
            exposes_partials=False,
            can_be_interrupted=True,
            optional_ops=frozenset({OptionalOp.EVENT_STREAM}),
            audio_in=self._format,
            audio_out=self._format,
        )

    @property
    @override
    def is_open(self) -> bool:
        return self._open

    @property
    def ref(self) -> ProviderRef:
        return ProviderRef.build(
            provider=NAME,
            model=self._model,
            config={**asdict(self._settings), "system_prompt": self._system_prompt},
            capabilities=SPEC.capabilities,
        )

    def usage_record(self, component: str = "agent") -> UsageRecord:
        """Seconds the session has been open, the unit the API is billed in."""
        seconds = self._seconds
        if self._started is not None:
            seconds += self._clock.now() - self._started
        return UsageRecord(component=component, model=self._model, seconds=seconds)

    # ------------------------------------------------------------- lifecycle

    def settings_message(self) -> dict[str, Any]:
        """The ``Settings`` message for this session."""
        s = self._settings
        wire = {
            "encoding": encoding(self._format),
            "sample_rate": self._format.sample_rate_hz,
        }
        agent: dict[str, Any] = {
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": s.listen_model,
                    "version": _version(s.listen_model),
                }
            },
            "think": {
                "provider": {"type": s.think_provider, "model": self._model},
                "prompt": self._system_prompt,
                "functions": [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.input_schema or {"type": "object"},
                    }
                    for spec in (self._tools.specs if self._tools else ())
                ],
            },
            "speak": {
                "provider": {
                    "type": "deepgram",
                    "model": s.voice,
                    "version": _version(s.voice),
                }
            },
        }
        if s.greeting is not None:
            agent["greeting"] = s.greeting
        return {
            "type": "Settings",
            "audio": {"input": wire, "output": {**wire, "container": "none"}},
            "agent": agent,
        }

    @override
    async def open(self, config: SessionConfig) -> None:
        if self._open:
            return
        headers = auth_header(self._api_key, provider=NAME, model=self._model)
        connect = self._connect or _websockets().connect
        try:
            self._ws = await connect(self._url, additional_headers=headers)
            welcome = json.loads(await self._ws.recv())
            if welcome.get("type") != "Welcome":
                raise self._malformed(f"expected Welcome, got {str(welcome)[:300]}")
            self.request_id = welcome.get("request_id")
            await self._ws.send(json.dumps(self.settings_message()))
            while True:
                reply = json.loads(await self._ws.recv())
                if reply.get("type") == "SettingsApplied":
                    break
                if reply.get("type") == "Error":
                    raise self._malformed(
                        f"Settings rejected: {reply.get('code')}: "
                        f"{reply.get('description')}"
                    )
        except ProviderError:
            await self._drop()
            raise
        except Exception as exc:
            await self._drop()
            raise socket_error(exc, provider=NAME, model=self._model) from exc
        self._open = True
        self._started = self._clock.now()
        self._pump = asyncio.create_task(self._receive())
        self._keepalive = asyncio.create_task(self._keep_alive())

    @override
    async def tx(self, frame: AudioFrame) -> None:
        """Send one frame, which must already be in the session's format."""
        self.require_open()
        if frame.fmt != self._format:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"session audio is {self._format}, got a {frame.fmt} frame",
                provider=NAME,
                model=self._model,
            )
        await self._ws.send(frame.pcm)

    @override
    async def events(self) -> AsyncIterator[StreamEvent]:
        while True:
            event = await self._inbox.get()
            if event is None:
                return
            yield event

    @override
    async def rx(self) -> AsyncIterator[AudioFrame]:
        async for event in self.events():
            if isinstance(event, OutputAudio) and event.pcm:
                frame = AudioFrame(
                    pcm=event.pcm,
                    seq=self._seq,
                    party=self._party,
                    t_produced=event.t,
                    fmt=self._format,
                )
                self._seq += 1
                yield frame

    @override
    async def close(self) -> None:
        if not self._open:
            return
        self._open = False
        if self._started is not None:
            self._seconds += self._clock.now() - self._started
            self._started = None
        pending = list(self._calls.values())
        self._calls.clear()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in (self._keepalive, self._pump):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        await self._drop()
        await self._inbox.put(None)

    # ------------------------------------------------------------- internals

    def _malformed(self, message: str) -> ProviderError:
        return ProviderError(
            ProviderFailure.MALFORMED, message, provider=NAME, model=self._model
        )

    async def _keep_alive(self) -> None:
        """A ``KeepAlive`` every :data:`KEEPALIVE_S`.

        Deepgram closes a session that receives nothing for about 10 s. The
        caller's line normally streams, silence included, but a long agent
        turn or tool call on a driver that pauses its audio would otherwise
        end the call with no error.
        """
        while self._ws is not None:
            await asyncio.sleep(KEEPALIVE_S)
            with contextlib.suppress(
                Exception
            ):  # a dead socket ends the call elsewhere
                await self._ws.send(json.dumps({"type": "KeepAlive"}))

    async def _drop(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

    async def _receive(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    self._inbox.put_nowait(
                        OutputAudio(raw, self._agent_turn(), self._clock.now())
                    )
                else:
                    self._dispatch(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, surfaced at close
            self.pump_error = exc
        finally:
            await self._inbox.put(None)

    def _agent_turn(self) -> str:
        """The agent turn now speaking, starting one if none is."""
        if self._turn is None:
            self._turns += 1
            self._turn = f"agent-{self._turns}"
            self._turn_spoke = False
        return self._turn

    def _dispatch(self, data: dict[str, Any]) -> None:
        kind = data.get("type")
        now = self._clock.now()
        if kind == "ConversationText":
            content = data.get("content") or ""
            if data.get("role") == "assistant":
                if content:
                    turn = self._agent_turn()
                    # Deltas concatenate; a turn's second message needs its space.
                    delta = f" {content}" if self._turn_spoke else content
                    self._turn_spoke = True
                    self._inbox.put_nowait(OutputTranscript(turn, delta, now))
            elif self._settings.extended_events:
                self._inbox.put_nowait(InputTranscript(None, content, now))
        elif kind == "UserStartedSpeaking":
            self._turn = None
            self._inbox.put_nowait(InputSpeech(True, None, now))
        elif kind == "AgentAudioDone":
            self._turn = None
        elif kind == "FunctionCallRequest":
            for call in data.get("functions") or ():
                if call.get("client_side") and call.get("id") and call.get("name"):
                    invocation = ToolInvocation.from_json(
                        tool_use_id=call["id"],
                        name=call["name"],
                        raw=call.get("arguments") or "",
                    )
                    self._calls[call["id"]] = asyncio.create_task(
                        self._answer(invocation)
                    )
        elif kind == "FunctionCallCancelled":
            for call in data.get("functions") or ():
                task = self._calls.pop(call.get("id") or "", None)
                if task is not None:
                    task.cancel()
        elif kind == "Error":
            self.errors.append(data)

    async def _answer(self, invocation: ToolInvocation) -> None:
        """Run one call and send its ``FunctionCallResponse``."""
        started = self._clock.now()
        if self._tools is None:
            content: dict[str, Any] = {
                "error": "no tools are configured for this session"
            }
            is_error = True
            output = json.dumps(content)
        else:
            result = await self._tools.invoke(invocation, at_s=started)
            content, is_error, output = (
                result.content,
                result.is_error,
                result.as_text(),
            )
        self._calls.pop(invocation.tool_use_id, None)
        if self._ws is None:
            return
        if self._settings.extended_events:
            self._inbox.put_nowait(
                ToolCall(
                    call_id=invocation.tool_use_id,
                    name=invocation.name,
                    arguments=dict(invocation.arguments),
                    result=dict(content),
                    is_error=is_error,
                    started=started,
                    t=self._clock.now(),
                )
            )
        await self._ws.send(
            json.dumps(
                {
                    "type": "FunctionCallResponse",
                    "id": invocation.tool_use_id,
                    "name": invocation.name,
                    "content": output,
                }
            )
        )


SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.S2S,
    capabilities=ProviderCapabilities(
        modality=Modality.S2S,
        streams=True,
        emits_partials=False,
        word_timings=False,
        diarizes=False,
    ),
    credential_provider="deepgram",
    requires_extra="voice",
    default_model=DEFAULT_MODEL,
)


def build(*, model: str = DEFAULT_MODEL, **config: Any) -> DeepgramAgentTransport:
    party = config.pop("party", None) or PartyId("agent")
    return DeepgramAgentTransport(party=PartyId(str(party)), model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
