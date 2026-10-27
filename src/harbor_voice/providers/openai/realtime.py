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
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
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
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import StreamEvent

NAME = "openai.realtime"
DEFAULT_MODEL = "gpt-realtime-1.5"
URL = "wss://api.openai.com/v1/realtime"

TELEPHONY = AudioFormat(sample_rate_hz=8_000, channels=1, encoding="mulaw")
"""``audio/pcmu``. tau2's ``TELEPHONY_AUDIO_FORMAT``."""


@dataclass(frozen=True, slots=True)
class RealtimeSettings:
    """Session settings. Every default is tau2's."""

    voice: str = "alloy"
    vad_threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    transcription_model: str = "gpt-4o-transcribe"
    language: str = "en"
    noise_reduction: str | None = "near_field"
    reasoning_effort: str | None = None

    def as_config(self) -> dict[str, Any]:
        return {
            "voice": self.voice,
            "vad_threshold": self.vad_threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "transcription_model": self.transcription_model,
            "language": self.language,
            "noise_reduction": self.noise_reduction,
            "reasoning_effort": self.reasoning_effort,
        }


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
        self._usage = {"input": 0, "output": 0, "audio_in": 0, "audio_out": 0}
        self._seq = 0
        self.session_id: str | None = None
        self.errors: list[dict[str, Any]] = []
        """``error`` events the server sent without closing the session."""
        self.pump_error: Exception | None = None
        self.converted_frames = 0
        """PCM16 frames converted to μ-law at this edge. A declared event."""

    # ------------------------------------------------------------ declaration

    @property
    @override
    def capabilities(self) -> Capabilities:
        return Capabilities(
            duplex=frozenset({Duplex.HALF, Duplex.FULL}),
            pipeline=Pipeline.S2S,
            turn_control=TurnControl.NATIVE_VAD,
            # Input transcription is configured (as tau2 configures it) but its
            # events are not surfaced, so nothing here may claim ASR.
            exposes_asr=False,
            exposes_partials=False,
            can_be_interrupted=True,
            optional_ops=frozenset({OptionalOp.EVENT_STREAM, OptionalOp.TRUNCATE}),
            audio_in=TELEPHONY,
            audio_out=TELEPHONY,
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
        return Usage(
            tokens=UsageInfo(
                prompt_tokens=self._usage["input"],
                completion_tokens=self._usage["output"],
                cache_tokens=0,
                cost_usd=0.0,
            ),
            speech_tokens=self._usage["audio_in"] + self._usage["audio_out"],
        )

    # ------------------------------------------------------------- lifecycle

    def session_update(self) -> dict[str, Any]:
        """The ``session.update`` tau2 sends, field for field."""
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
            "tool_choice": "auto",
        }
        if s.reasoning_effort is not None:
            session["reasoning"] = {"effort": s.reasoning_effort}
        audio_format = {"type": "audio/pcmu"}
        session["audio"] = {
            "input": {
                "format": audio_format,
                "transcription": {
                    "model": s.transcription_model,
                    "language": s.language,
                },
                "noise_reduction": (
                    {"type": s.noise_reduction} if s.noise_reduction else None
                ),
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": s.vad_threshold,
                    "prefix_padding_ms": s.prefix_padding_ms,
                    "silence_duration_ms": s.silence_duration_ms,
                },
            },
            "output": {"format": audio_format, "voice": s.voice},
        }
        return {"type": "session.update", "session": session}

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
        """Send one frame: μ-law 8 kHz as is, PCM16 converted at this edge.

        Conversion is anti-aliased resampling then real G.711, counted
        in :attr:`converted_frames` so it shows up as a declared event. A
        caller that owns its own phone line (tau-voice's does) sends μ-law
        and nothing is converted.
        """
        self.require_open()
        if frame.fmt == TELEPHONY:
            payload = frame.pcm
        elif frame.fmt.encoding == "pcm16" and frame.fmt.channels == 1:
            payload = self._to_pcmu(frame)
        else:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"{NAME} takes μ-law 8 kHz or mono PCM16 frames, got {frame.fmt}",
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
                    fmt=TELEPHONY,
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
                self._dispatch(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, surfaced at close
            self.pump_error = exc
        finally:
            await self._inbox.put(None)

    def _dispatch(self, data: dict[str, Any]) -> None:
        kind = data.get("type", "")
        now = self._clock.now()
        if kind == "response.output_audio.delta":
            pcm = base64.b64decode(data.get("delta") or "")
            self._inbox.put_nowait(OutputAudio(pcm, data.get("item_id"), now))
        elif kind == "response.output_audio_transcript.delta":
            item = data.get("item_id")
            if item and data.get("delta"):
                self._inbox.put_nowait(OutputTranscript(item, data["delta"], now))
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
                self._tool_tasks.append(asyncio.create_task(self._answer(invocation)))
        elif kind == "response.done":
            usage = (data.get("response") or {}).get("usage") or {}
            self._usage["input"] += int(usage.get("input_tokens") or 0)
            self._usage["output"] += int(usage.get("output_tokens") or 0)
            details_in = usage.get("input_token_details") or {}
            details_out = usage.get("output_token_details") or {}
            self._usage["audio_in"] += int(details_in.get("audio_tokens") or 0)
            self._usage["audio_out"] += int(details_out.get("audio_tokens") or 0)
        elif kind == "error":
            self.errors.append(data.get("error") or data)

    async def _answer(self, invocation: ToolInvocation) -> None:
        """Answer one call; the last outstanding answer asks for a response.

        tau2 flushes a tick's tool results together and sends
        ``response.create`` after the last of them. Results here resolve at
        the same step boundary, so counting outstanding calls reproduces
        that: one ``response.create`` per batch.
        """
        if self._tools is None:
            output = json.dumps({"error": "no tools are configured for this session"})
        else:
            result = await self._tools.invoke(invocation, at_s=self._clock.now())
            output = result.as_text()
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
