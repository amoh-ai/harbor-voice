"""OpenAI Realtime as a :class:`~harbor_voice.transport.base.VoiceTransport`.

tau-voice's default agent, so the session is **tau2's session**
(``tau2/voice/audio_native/openai/provider.py::configure_session``) unless a
caller says otherwise:

- ``gpt-realtime-1.5``, voice ``alloy``, no reasoning effort set;
- ``audio/pcmu`` (8 kHz μ-law) in and out — tau-voice is a phone benchmark;
- server VAD at threshold 0.5, 300 ms prefix padding, 500 ms silence;
- input transcription ``gpt-4o-transcribe`` in English, ``near_field`` noise
  reduction;
- tools offered with ``tool_choice=auto``; a call's output goes back as a
  ``function_call_output`` and one ``response.create`` follows the last
  outstanding output, as tau2 sends it.

It declares ``EVENT_STREAM`` (agent audio by utterance, the agent's own
transcript, provider VAD — in arrival order) and ``TRUNCATE``, which is what a
stepped caller needs. ``rx()`` is the same inbox reduced to audio frames.

Every other session is reached through :class:`RealtimeSettings`, never by
naming a benchmark here; the adapter passes its benchmark's values in. EVA's
(``openai_realtime_server.py``) differs from tau2's in ways each setting
names: 24 kHz PCM both ways, fields it never sends (``tool_choice``, the
transcription language, noise reduction), an opening user message, and how
tool answers are sequenced (:attr:`RealtimeSettings.tool_flow`). With every
default, the ``session.update`` is byte for byte tau2's and the session
behaves exactly as before.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from collections.abc import AsyncIterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from typing import Any
from typing import Literal
from typing import override

from harbor.models.metric import UsageInfo

from harbor_voice._deps import numpy as _numpy
from harbor_voice._deps import websockets as _websockets
from harbor_voice.audio import g711
from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import OptionalOp
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.ids import PartyId
from harbor_voice.perturbation import dsp
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderRef
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolResult
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import InputTranscript
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import OutputTranscriptDone
from harbor_voice.transport.stream import ResponseDone
from harbor_voice.transport.stream import StreamEvent
from harbor_voice.transport.stream import ToolCall

NAME = "openai.realtime"
TOKEN_COUNTS = (
    "input",
    "input_cached",
    "input_text",
    "input_audio",
    "output",
    "output_text",
    "output_audio",
)
"""The token counts kept per session, as :func:`token_counts` names them."""


def token_counts(usage: Mapping[str, Any]) -> dict[str, int]:
    """One ``response.done`` usage block as flat counts (cached is part of input)."""
    details_in = usage.get("input_token_details") or {}
    details_out = usage.get("output_token_details") or {}
    return {
        "input": int(usage.get("input_tokens") or 0),
        "input_cached": int(details_in.get("cached_tokens") or 0),
        "input_text": int(details_in.get("text_tokens") or 0),
        "input_audio": int(details_in.get("audio_tokens") or 0),
        "output": int(usage.get("output_tokens") or 0),
        "output_text": int(details_out.get("text_tokens") or 0),
        "output_audio": int(details_out.get("audio_tokens") or 0),
    }


DEFAULT_MODEL = "gpt-realtime-1.5"
URL = "wss://api.openai.com/v1/realtime"

TELEPHONY = AudioFormat(sample_rate_hz=8_000, channels=1, encoding="mulaw")
"""``audio/pcmu``. tau2's ``TELEPHONY_AUDIO_FORMAT``."""

PCM24K = AudioFormat(sample_rate_hz=24_000, channels=1, encoding="pcm16")
"""``audio/pcm`` at 24 kHz, the API's own rate."""

AudioFormatName = Literal["pcmu", "pcm24k"]
AUDIO_FORMATS: Mapping[str, AudioFormat] = {"pcmu": TELEPHONY, "pcm24k": PCM24K}


def wire_format(name: str) -> dict[str, Any]:
    """A format as ``session.update`` names it, for both directions."""
    if name == "pcm24k":
        return {"type": "audio/pcm", "rate": PCM24K.sample_rate_hz}
    return {"type": "audio/pcmu"}


@dataclass(frozen=True, slots=True)
class RealtimeSettings:
    """Session settings. Every default is tau2's.

    ``None`` on a field the API defaults for itself (``tool_choice``,
    ``language``, ``noise_reduction``) leaves the field **out** of
    ``session.update``, so the server's default applies. That is how a
    session that never sends a field — EVA's sends none of these three — is
    reproduced field for field, rather than approximated by sending the value
    the default happens to have today.
    """

    voice: str = "alloy"
    vad_threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    transcription_model: str = "gpt-4o-transcribe"
    language: str | None = "en"
    """The transcription language; ``None`` lets the model detect it."""
    noise_reduction: str | None = "near_field"
    reasoning_effort: str | None = None
    audio_format: AudioFormatName = "pcmu"
    """Both directions. ``pcmu``: μ-law 8 kHz, a phone line, as tau2 runs it.
    ``pcm24k``: PCM16 at 24 kHz, passed through unconverted — for a caller
    that converts to and from the phone line itself, so that its conversion,
    not ours, is what the model hears (EVA's bridge)."""
    tool_choice: str | None = "auto"
    parallel_tool_calls: bool | None = None
    """Sent only when set."""
    opening_text: str | None = None
    """A user text message sent verbatim right after ``session.update``,
    then ``response.create``, so the agent speaks first. EVA opens every call
    with ``Say: '<greeting>'``."""
    tool_flow: Literal["batch", "serial"] = "batch"
    """How tool calls are answered.

    ``batch`` (tau2): each call runs beside the receive loop as it arrives,
    and one ``response.create`` follows the last outstanding output — a
    response's parallel calls get one continuation between them.

    ``serial`` (EVA): the receive loop runs each call itself, in arrival
    order, reading nothing else from the server until that call is answered,
    and sends ``response.create`` after **every** output. That is EVA's
    ``_on_function_call_done``, awaited inside its event loop. With two calls
    in one response, the first ``response.create`` can reach the server
    while that response is still open, and the server then answers it with
    an ``error`` event (kept in :attr:`OpenAIRealtimeTransport.errors`)
    rather than a second response — exactly as it would for EVA.
    """
    tool_output_ascii: bool = True
    """Whether a JSON tool output escapes non-ASCII, ``json.dumps``' default.
    EVA sends ``json.dumps(result, ensure_ascii=False)``, so an accented name
    reaches its model as itself, not as ``\\u00e9``."""
    extended_events: bool = False
    """Also yield :class:`~harbor_voice.transport.stream.InputTranscript`,
    :class:`~harbor_voice.transport.stream.OutputTranscriptDone`,
    :class:`~harbor_voice.transport.stream.ResponseDone` and
    :class:`~harbor_voice.transport.stream.ToolCall` from ``events()``. Off by
    default: a stepped consumer rejects kinds it does not know."""

    def as_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "voice": self.voice,
            "vad_threshold": self.vad_threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "transcription_model": self.transcription_model,
            "language": self.language,
            "noise_reduction": self.noise_reduction,
            "reasoning_effort": self.reasoning_effort,
        }
        # Settings added after tau2's join only when changed, so a tau2
        # session keeps the config digest it always had.
        for spec in fields(self):
            value = getattr(self, spec.name)
            if spec.name not in config and value != spec.default:
                config[spec.name] = value
        return config


TAU2_SETTINGS = RealtimeSettings()


class OpenAIRealtimeTransport(VoiceTransport):
    """One OpenAI Realtime session over a websocket."""

    def __init__(
        self,
        *,
        party: PartyId,
        model: str = DEFAULT_MODEL,
        system_prompt: str = "",
        tools: ToolRouter | None = None,
        settings: RealtimeSettings = TAU2_SETTINGS,
        api_key: str | None = None,
        url: str = URL,
        clock: SessionClock | None = None,
        connect: Any = None,
    ) -> None:
        self._party = party
        self._model = model
        self._system_prompt = system_prompt
        self._tools = tools
        self._settings = settings
        self._format = AUDIO_FORMATS[settings.audio_format]
        self._api_key = api_key
        self._url = url
        self._clock = clock or SessionClock()
        self._connect = connect
        self._ws: Any = None
        self._open = False
        self._pump: asyncio.Task[None] | None = None
        self._inbox: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._tool_tasks: list[asyncio.Task[None]] = []
        self._tools_outstanding = 0
        self._usage = dict.fromkeys(TOKEN_COUNTS, 0)
        self._responses = 0
        self._seq = 0
        self.session_id: str | None = None
        self.errors: list[dict[str, Any]] = []
        """``error`` events the server sent without closing the session."""
        self.pump_error: Exception | None = None
        self.converted_frames = 0
        """Frames converted to the session's format at this edge. A declared
        event."""

    # ------------------------------------------------------------ declaration

    @property
    @override
    def capabilities(self) -> Capabilities:
        return Capabilities(
            duplex=frozenset({Duplex.HALF, Duplex.FULL}),
            pipeline=Pipeline.S2S,
            turn_control=TurnControl.NATIVE_VAD,
            # Input transcription is configured (as tau2 configures it). Its
            # result surfaces only as an opt-in InputTranscript, with no ASR
            # stage timing, so nothing here may claim ASR.
            exposes_asr=False,
            exposes_partials=False,
            can_be_interrupted=True,
            optional_ops=frozenset({OptionalOp.EVENT_STREAM, OptionalOp.TRUNCATE}),
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
            config={**self._settings.as_config(), "system_prompt": self._system_prompt},
            capabilities=SPEC.capabilities,
        )

    @property
    def usage(self) -> Usage:
        u = self._usage
        return Usage(
            tokens=UsageInfo(
                prompt_tokens=u["input"],
                completion_tokens=u["output"],
                cache_tokens=u["input_cached"],
                cost_usd=0.0,
            ),
            speech_tokens=u["input_audio"] + u["output_audio"],
            text_tokens=u["input_text"] + u["output_text"],
        )

    def usage_record(self, component: str = "agent") -> UsageRecord:
        """The session's tokens so far, summed over every ``response.done``."""
        u = self._usage
        return UsageRecord(
            component=component,
            model=self._model,
            requests=self._responses,
            input_tokens=u["input"],
            cached_input_tokens=u["input_cached"],
            output_tokens=u["output"],
            input_audio_tokens=u["input_audio"],
            output_audio_tokens=u["output_audio"],
        )

    # ------------------------------------------------------------- lifecycle

    def session_update(self) -> dict[str, Any]:
        """The ``session.update`` for these settings.

        With the defaults it is tau2's, field for field and in tau2's key
        order, so the bytes on the wire are tau2's too.
        """
        s = self._settings
        session: dict[str, Any] = {
            "type": "realtime",
            "instructions": self._system_prompt,
            "output_modalities": ["audio"],
            "tools": [
                {
                    "type": "function",
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema or {"type": "object"},
                }
                for spec in (self._tools.specs if self._tools else ())
            ],
        }
        if s.tool_choice is not None:
            session["tool_choice"] = s.tool_choice
        if s.reasoning_effort is not None:
            session["reasoning"] = {"effort": s.reasoning_effort}
        if s.parallel_tool_calls is not None:
            session["parallel_tool_calls"] = s.parallel_tool_calls
        transcription: dict[str, Any] = {"model": s.transcription_model}
        if s.language is not None:
            transcription["language"] = s.language
        audio_input: dict[str, Any] = {
            "format": wire_format(s.audio_format),
            "transcription": transcription,
        }
        if s.noise_reduction:
            audio_input["noise_reduction"] = {"type": s.noise_reduction}
        audio_input["turn_detection"] = {
            "type": "server_vad",
            "threshold": s.vad_threshold,
            "prefix_padding_ms": s.prefix_padding_ms,
            "silence_duration_ms": s.silence_duration_ms,
        }
        session["audio"] = {
            "input": audio_input,
            "output": {"format": wire_format(s.audio_format), "voice": s.voice},
        }
        return {"type": "session.update", "session": session}

    def opening(self) -> list[dict[str, Any]]:
        """What follows ``session.update`` when an opening text is set.

        A user message carrying the text, then a request to answer it — the
        two events EVA sends (``openai_realtime_server.py:270-282``), in its
        order and shape.
        """
        text = self._settings.opening_text
        if text is None:
            return []
        return [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            },
            {"type": "response.create"},
        ]

    @override
    async def open(self, config: SessionConfig) -> None:
        if self._open:
            return
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError(
                ProviderFailure.AUTH,
                "OPENAI_API_KEY is not set",
                provider=NAME,
                model=self._model,
            )
        connect = self._connect or _websockets().connect
        try:
            self._ws = await connect(
                f"{self._url}?model={self._model}",
                additional_headers={"Authorization": f"Bearer {key}"},
            )
            created = json.loads(await self._ws.recv())
            if created.get("type") != "session.created":
                raise ProviderError(
                    ProviderFailure.MALFORMED,
                    f"expected session.created, got {created.get('type')}: {str(created)[:300]}",
                    provider=NAME,
                    model=self._model,
                )
            self.session_id = (created.get("session") or {}).get("id")
            await self._ws.send(json.dumps(self.session_update()))
            # Sent before session.updated comes back, as EVA sends it: the
            # server applies client events in order, so the update still
            # lands first, and the agent starts speaking one round trip
            # sooner than if we waited.
            for event in self.opening():
                await self._ws.send(json.dumps(event))
            while True:
                reply = json.loads(await self._ws.recv())
                kind = reply.get("type")
                if kind == "session.updated":
                    break
                if kind == "error":
                    raise ProviderError(
                        ProviderFailure.MALFORMED,
                        f"session.update rejected: {reply.get('error')}",
                        provider=NAME,
                        model=self._model,
                    )
        except ProviderError:
            await self._drop()
            raise
        except Exception as exc:
            await self._drop()
            raise _classify(exc, self._model) from exc
        self._open = True
        self._pump = asyncio.create_task(self._receive())

    @override
    async def tx(self, frame: AudioFrame) -> None:
        """Send one frame: the session's format as is, others converted here.

        Conversion is anti-aliased resampling (and real G.711, either way),
        counted in :attr:`converted_frames` so it shows up as a
        declared event. A caller that owns its own phone line sends the
        session's format and nothing is converted: tau-voice's sends μ-law to
        a ``pcmu`` session, EVA's 24 kHz PCM it converted itself to a
        ``pcm24k`` one.
        """
        self.require_open()
        if frame.fmt == self._format:
            payload = frame.pcm
        elif self._format == TELEPHONY:
            if frame.fmt.encoding != "pcm16" or frame.fmt.channels != 1:
                raise ProviderError(
                    ProviderFailure.MALFORMED,
                    f"{NAME} takes μ-law 8 kHz or mono PCM16 frames, got {frame.fmt}",
                    provider=NAME,
                    model=self._model,
                )
            payload = self._to_pcmu(frame)
        elif frame.fmt.channels == 1 and frame.fmt.encoding in ("pcm16", "mulaw"):
            payload = self._to_pcm24k(frame)
        else:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"{NAME} takes mono PCM16 or μ-law frames, got {frame.fmt}",
                provider=NAME,
                model=self._model,
            )
        await self._ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(payload).decode("ascii"),
                }
            )
        )

    def _to_pcmu(self, frame: AudioFrame) -> bytes:
        pcm = frame.pcm
        if frame.fmt.sample_rate_hz != TELEPHONY.sample_rate_hz:
            resampled = dsp.resample_clean(
                dsp.to_array(pcm), frame.fmt.sample_rate_hz, TELEPHONY.sample_rate_hz
            )
            pcm = dsp.to_pcm(_numpy().asarray(resampled))
        self.converted_frames += 1
        return g711.encode(pcm)

    def _to_pcm24k(self, frame: AudioFrame) -> bytes:
        pcm = g711.decode(frame.pcm) if frame.fmt.encoding == "mulaw" else frame.pcm
        if frame.fmt.sample_rate_hz != PCM24K.sample_rate_hz:
            resampled = dsp.resample_clean(
                dsp.to_array(pcm), frame.fmt.sample_rate_hz, PCM24K.sample_rate_hz
            )
            pcm = dsp.to_pcm(_numpy().asarray(resampled))
        self.converted_frames += 1
        return pcm

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
    async def truncate(self, utterance: str, audio_end_ms: int) -> None:
        self.require_open()
        await self._ws.send(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": utterance,
                    "content_index": 0,
                    "audio_end_ms": audio_end_ms,
                }
            )
        )

    @override
    async def close(self) -> None:
        if not self._open:
            return
        self._open = False
        if self._tool_tasks:
            pending, self._tool_tasks = self._tool_tasks, []
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        await self._drop()
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump
        await self._inbox.put(None)

    # ------------------------------------------------------------- internals

    async def _drop(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

    async def _receive(self) -> None:
        try:
            async for raw in self._ws:
                serial = self._dispatch(json.loads(raw))
                if serial is not None:
                    # tool_flow="serial": answered here, so nothing behind it
                    # in the socket is read until it is — EVA's order.
                    await self._answer(serial)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, surfaced at close
            self.pump_error = exc
        finally:
            await self._inbox.put(None)

    def _dispatch(self, data: dict[str, Any]) -> ToolInvocation | None:
        """Apply one server event. Returns a tool call for the receive loop
        to answer itself (``tool_flow="serial"``), else ``None``."""
        kind = data.get("type", "")
        now = self._clock.now()
        extended = self._settings.extended_events
        if kind == "response.output_audio.delta":
            pcm = base64.b64decode(data.get("delta") or "")
            self._inbox.put_nowait(OutputAudio(pcm, data.get("item_id"), now))
        elif kind == "response.output_audio_transcript.delta":
            item = data.get("item_id")
            if item and data.get("delta"):
                self._inbox.put_nowait(OutputTranscript(item, data["delta"], now))
        elif kind == "response.output_audio_transcript.done":
            if extended:
                self._inbox.put_nowait(
                    OutputTranscriptDone(
                        data.get("item_id"), data.get("transcript") or "", now
                    )
                )
        elif kind == "conversation.item.input_audio_transcription.completed":
            if extended:
                self._inbox.put_nowait(
                    InputTranscript(
                        data.get("item_id"), data.get("transcript") or "", now
                    )
                )
        elif kind == "conversation.item.input_audio_transcription.failed":
            if extended:
                error = data.get("error") or {}
                reason = (
                    str(error.get("message") or error)
                    if isinstance(error, dict)
                    else str(error)
                )
                self._inbox.put_nowait(
                    InputTranscript(
                        data.get("item_id"), "", now, error=reason or "failed"
                    )
                )
        elif kind == "input_audio_buffer.speech_started":
            self._inbox.put_nowait(InputSpeech(True, data.get("audio_start_ms"), now))
        elif kind == "input_audio_buffer.speech_stopped":
            self._inbox.put_nowait(InputSpeech(False, data.get("audio_end_ms"), now))
        elif kind == "response.function_call_arguments.done":
            if data.get("call_id") and data.get("name"):
                invocation = ToolInvocation.from_json(
                    tool_use_id=data["call_id"],
                    name=data["name"],
                    raw=data.get("arguments") or "",
                )
                self._tools_outstanding += 1
                if self._settings.tool_flow == "serial":
                    return invocation
                self._tool_tasks.append(asyncio.create_task(self._answer(invocation)))
        elif kind == "response.done":
            response = data.get("response") or {}
            usage = response.get("usage") or {}
            self._responses += 1
            for key, value in token_counts(usage).items():
                self._usage[key] += value
            if extended:
                reported = response.get("usage")
                self._inbox.put_nowait(
                    ResponseDone(
                        response_id=response.get("id"),
                        status=str(response.get("status") or ""),
                        has_function_calls=response_has_function_calls(response),
                        text=response_text(response),
                        t=now,
                        input_tokens=(
                            int(reported.get("input_tokens") or 0)
                            if isinstance(reported, dict)
                            else None
                        ),
                        output_tokens=(
                            int(reported.get("output_tokens") or 0)
                            if isinstance(reported, dict)
                            else None
                        ),
                    )
                )
        elif kind == "error":
            self.errors.append(data.get("error") or data)
        return None

    async def _answer(self, invocation: ToolInvocation) -> None:
        """Answer one call; the last outstanding answer asks for a response.

        tau2 flushes a tick's tool results together and sends
        ``response.create`` after the last of them. Results here resolve at
        the same step boundary, so counting outstanding calls reproduces
        that: one ``response.create`` per batch. Under ``tool_flow="serial"``
        only one call is ever outstanding, so the same count asks for a
        response after every answer, as EVA does.
        """
        started = self._clock.now()
        if self._tools is None:
            content: dict[str, Any] = {
                "error": "no tools are configured for this session"
            }
            is_error = True
            output = json.dumps(content)
        else:
            result = await self._tools.invoke(invocation, at_s=started)
            content, is_error = result.content, result.is_error
            output = self._output(result)
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
        self._tools_outstanding -= 1
        if self._ws is None:
            return
        await self._ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": invocation.tool_use_id,
                        "output": output,
                    },
                }
            )
        )
        if self._tools_outstanding == 0:
            await self._ws.send(json.dumps({"type": "response.create"}))

    def _output(self, result: ToolResult) -> str:
        """The ``function_call_output`` text: :meth:`ToolResult.as_text`,
        unless non-ASCII is to go as itself (``tool_output_ascii=False``)."""
        content = result.content
        if self._settings.tool_output_ascii or (
            set(content) == {"text"} and isinstance(content["text"], str)
        ):
            return result.as_text()
        return json.dumps(content, ensure_ascii=False)


def response_has_function_calls(response: Mapping[str, Any]) -> bool:
    """Whether a ``response.done`` response holds a tool call.

    EVA's ``_response_has_function_calls``: any output item of type
    ``function_call``.
    """
    return any(
        isinstance(item, Mapping) and item.get("type") == "function_call"
        for item in response.get("output") or ()
    )


def response_text(response: Mapping[str, Any]) -> str:
    """The text a ``response.done`` response carries in its output items.

    EVA's ``_extract_response_text``, kept literally: content parts typed
    ``audio`` or ``text``, their ``transcript`` else their ``text``, joined
    and stripped. Those are the beta API's part types; the GA API, which
    EVA's SDK speaks, types them ``output_audio`` / ``output_text``, so for a
    GA response this is empty (the Realtime-session golden shows EVA computing exactly
    that). EVA reaches for it only when no transcript arrived, and a port of
    its recorder must log what EVA logs, not what it meant to.
    """
    parts: list[str] = []
    for item in response.get("output") or ():
        if not isinstance(item, Mapping):
            continue
        for part in item.get("content") or ():
            if isinstance(part, Mapping) and part.get("type") in ("audio", "text"):
                text = part.get("transcript") or part.get("text") or ""
                if text:
                    parts.append(text)
    return "".join(parts).strip()


def _classify(exc: Exception, model: str) -> ProviderError:
    text = str(exc)
    if "401" in text or "403" in text:
        kind = ProviderFailure.AUTH
    elif "404" in text:
        kind = ProviderFailure.NOT_ENABLED
    elif "429" in text:
        kind = ProviderFailure.THROTTLED
    elif isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        kind = ProviderFailure.TIMEOUT
    else:
        kind = ProviderFailure.STREAM_BROKEN
    return ProviderError(
        kind, f"{type(exc).__name__}: {text[:300]}", provider=NAME, model=model
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
    credential_provider="openai",
    requires_extra="voice",
    default_model=DEFAULT_MODEL,
)


def build(*, model: str = DEFAULT_MODEL, **config: Any) -> OpenAIRealtimeTransport:
    party = config.pop("party", None) or PartyId("agent")
    return OpenAIRealtimeTransport(party=PartyId(str(party)), model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
