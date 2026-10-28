"""OpenAI Live as a :class:`~harbor_voice.transport.base.VoiceTransport`.

Live splits the agent in two: a speech frontend (``gpt-live-1``) that listens,
speaks and decides when to delegate, and a Responses backend (``delegation``)
that reasons and calls tools. The session is tau2's
(``tau2/voice/audio_native/openai/live_provider.py``):

- frontend ``instructions`` are the system prompt, voice ``marin``;
- the backend gets its own instructions (the system prompt unless given),
  the tools with ``tool_choice=auto`` and ``parallel_tool_calls=False``;
- media is WebRTC (:mod:`~harbor_voice.providers.openai.live_webrtc`, the
  ``live`` extra): caller audio is clocked out in 20 ms frames, agent audio
  arrives continuously — silence included — as 24 kHz PCM16;
- events ride the data channel. The agent's transcript is placed on the audio
  timeline (``OutputTranscript.audio_position``: agent audio bytes received
  when the delta arrived); the caller's input transcript is the only sign of
  their speech (``InputSpeech``);
- a backend response's tool calls are answered with ``response.item.create``
  and continued with one ``response.create`` once the whole batch is back and
  the response has completed — never on speech.

tau2 sends an ``OpenAI-Alpha: quicksilver=v3`` header on session creation; the
documented API takes none, and with it a project without that preview is
refused (``live_api_access_denied``). This sends only ``Authorization``.

It declares ``EVENT_STREAM`` but not ``TRUNCATE``: Live never truncates.
Usage is the voice session's billed seconds (``usage.seconds``: each
``session.usage.updated`` replaces the last, ``session.closed`` is final) and
the backend's tokens per completed response; tau2 records no Live usage.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import override

from harbor.models.metric import UsageInfo

from harbor_voice._deps import numpy as _numpy
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
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import StreamEvent

NAME = "openai.live"
DEFAULT_MODEL = "gpt-live-1"
URL = "https://api.openai.com/v1/live/sessions"

LIVE_AUDIO = AudioFormat(sample_rate_hz=24_000, channels=1, encoding="pcm16")
"""Both ways at this edge: WebRTC media resampled to and from 24 kHz PCM16."""

START_TIMEOUT_S = 20.0
CLOSE_TIMEOUT_S = 5.0


def backend_usage(usage: Mapping[str, Any]) -> dict[str, int]:
    """One backend ``response.completed`` usage block (Responses API) as counts."""
    details = usage.get("input_tokens_details") or {}
    return {
        "input": int(usage.get("input_tokens") or 0),
        "input_cached": int(details.get("cached_tokens") or 0),
        "output": int(usage.get("output_tokens") or 0),
    }


@dataclass(frozen=True, slots=True)
class LiveSettings:
    """Session settings. Defaults are tau2's ``LiveConfig``."""

    backend_model: str
    voice: str = "marin"
    backend_instructions: str | None = None
    """The backend's instructions; the system prompt when ``None``."""
    reasoning_effort: str | None = None

    def as_config(self) -> dict[str, Any]:
        return {
            "backend_model": self.backend_model,
            "voice": self.voice,
            "backend_instructions": self.backend_instructions,
            "reasoning_effort": self.reasoning_effort,
        }


class LiveConnection(Protocol):
    """The media and data channel under a session; WebRTC by default."""

    async def start(
        self, session: dict[str, Any], *, api_key: str, url: str
    ) -> dict[str, Any]:
        """Create the session and connect; returns the created ``session``."""
        ...

    def send(self, event: dict[str, Any]) -> None:
        """Send one client event on the data channel."""
        ...

    def append(self, pcm: bytes) -> None:
        """Queue caller audio (``LIVE_AUDIO``) for the clocked input track."""
        ...

    async def close(self) -> None: ...


Connect = Callable[
    [Callable[[dict[str, Any]], None], Callable[[bytes], None], Callable[[str], None]],
    LiveConnection,
]
"""``(on_event, on_audio, on_error) -> LiveConnection``."""


def _webrtc(
    on_event: Callable[[dict[str, Any]], None],
    on_audio: Callable[[bytes], None],
    on_error: Callable[[str], None],
) -> LiveConnection:
    from harbor_voice.providers.openai.live_webrtc import WebRTCConnection

    return WebRTCConnection(on_event, on_audio, on_error)


class OpenAILiveTransport(VoiceTransport):
    """One OpenAI Live session: frontend speech, delegated backend, tools."""

    def __init__(
        self,
        *,
        party: PartyId,
        settings: LiveSettings,
        model: str = DEFAULT_MODEL,
        system_prompt: str = "",
        tools: ToolRouter | None = None,
        api_key: str | None = None,
        url: str = URL,
        clock: SessionClock | None = None,
        connect: Connect | None = None,
    ) -> None:
        self._party = party
        self._model = model
        self._system_prompt = system_prompt
        self._tools = tools
        self._settings = settings
        self._api_key = api_key
        self._url = url
        self._clock = clock or SessionClock()
        self._connect = connect or _webrtc
        self._conn: LiveConnection | None = None
        self._open = False
        self._closing = False
        self._started = asyncio.Event()
        self._ready = asyncio.Event()
        """Started, or failed before it could."""
        self._closed = asyncio.Event()
        self._inbox: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._tool_tasks: list[asyncio.Task[None]] = []
        self._received_bytes = 0
        # Tool batches, as tau2 tracks them: a backend response's calls must
        # all be answered, and the response completed, before it continues.
        self._response_ids: dict[str, str] = {}
        self._tool_calls: dict[str, set[str]] = {}
        self._pending: dict[str, set[str]] = {}
        self._returned: set[str] = set()
        self._continued: set[str] = set()
        self._surfaced: set[str] = set()
        self._usage = {"input": 0, "input_cached": 0, "output": 0}
        self._responses = 0
        self._seconds: float | None = None
        self._seq = 0
        self.session_id: str | None = None
        self.delegations = 0
        self.errors: list[dict[str, Any]] = []
        """Errors the session ended on (Live's are fatal, as tau2 treats them)."""
        self.pump_error: ProviderError | None = None
        self.converted_frames = 0
        """Caller frames converted to ``LIVE_AUDIO`` at this edge."""

    # ------------------------------------------------------------ declaration

    @property
    @override
    def capabilities(self) -> Capabilities:
        return Capabilities(
            duplex=frozenset({Duplex.FULL}),
            pipeline=Pipeline.S2S,
            turn_control=TurnControl.NATIVE_VAD,
            exposes_asr=False,
            exposes_partials=False,
            can_be_interrupted=True,
            optional_ops=frozenset({OptionalOp.EVENT_STREAM}),
            audio_in=LIVE_AUDIO,
            audio_out=LIVE_AUDIO,
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
            text_tokens=u["input"] + u["output"],
        )

    def usage_records(self, component: str = "agent") -> list[UsageRecord]:
        """The frontend (delegations, billed seconds) and the backend's tokens."""
        u = self._usage
        return [
            UsageRecord(
                component=component,
                model=self._model,
                requests=self.delegations,
                seconds=self._seconds,
            ),
            UsageRecord(
                component=f"{component}.backend",
                model=self._settings.backend_model,
                requests=self._responses,
                input_tokens=u["input"],
                cached_input_tokens=u["input_cached"],
                output_tokens=u["output"],
            ),
        ]

    # ------------------------------------------------------------- lifecycle

    def session_start(self) -> dict[str, Any]:
        """The ``session`` tau2 creates, field for field."""
        s = self._settings
        responses: dict[str, Any] = {
            "model": s.backend_model,
            "instructions": (
                self._system_prompt
                if s.backend_instructions is None
                else s.backend_instructions
            ),
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
            "parallel_tool_calls": False,
        }
        if s.reasoning_effort is not None:
            responses["reasoning"] = {"effort": s.reasoning_effort}
        return {
            "model": self._model,
            "instructions": self._system_prompt,
            "audio": {"output": {"voice": s.voice}},
            "delegation": {"type": "responses", "responses": responses},
        }

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
        self._conn = self._connect(self._accept, self._audio, self._lost)
        try:
            created = await self._conn.start(
                self.session_start(), api_key=key, url=self._url
            )
            self.session_id = created.get("id")
            await asyncio.wait_for(self._ready.wait(), START_TIMEOUT_S)
        except ProviderError:
            await self._drop()
            raise
        except Exception as exc:
            await self._drop()
            raise _classify(exc, self._model) from exc
        if self.pump_error is not None:
            await self._drop()
            raise self.pump_error
        self._open = True

    @override
    async def tx(self, frame: AudioFrame) -> None:
        """Queue one frame: ``LIVE_AUDIO`` as is, μ-law or other PCM16 converted."""
        self.require_open()
        if self.pump_error is not None:
            raise self.pump_error
        if frame.fmt == LIVE_AUDIO:
            payload = frame.pcm
        elif frame.fmt.channels == 1 and frame.fmt.encoding in ("pcm16", "mulaw"):
            payload = self._to_live(frame)
        else:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"{NAME} takes mono PCM16 or μ-law frames, got {frame.fmt}",
                provider=NAME,
                model=self._model,
            )
        assert self._conn is not None
        self._conn.append(payload)

    def _to_live(self, frame: AudioFrame) -> bytes:
        pcm = g711.decode(frame.pcm) if frame.fmt.encoding == "mulaw" else frame.pcm
        if frame.fmt.sample_rate_hz != LIVE_AUDIO.sample_rate_hz:
            resampled = dsp.resample_clean(
                dsp.to_array(pcm), frame.fmt.sample_rate_hz, LIVE_AUDIO.sample_rate_hz
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
                    fmt=LIVE_AUDIO,
                )
                self._seq += 1
                yield frame

    @override
    async def close(self) -> None:
        if not self._open:
            return
        self._open = False
        self._closing = True
        if self._tool_tasks:
            pending, self._tool_tasks = self._tool_tasks, []
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if (
            self._conn is not None
            and self._started.is_set()
            and not self._closed.is_set()
        ):
            # An unconfirmed close must not fail a finished call (tau2 logs it).
            with contextlib.suppress(Exception):
                self._conn.send({"type": "session.close"})
                await asyncio.wait_for(self._closed.wait(), CLOSE_TIMEOUT_S)
        await self._drop()
        await self._inbox.put(None)

    # ------------------------------------------------------------- internals

    async def _drop(self) -> None:
        self._closing = True
        if self._conn is not None:
            with contextlib.suppress(Exception):
                await self._conn.close()
            self._conn = None

    def _audio(self, pcm: bytes) -> None:
        """Agent audio off the media track, in arrival order with the events."""
        self._received_bytes += len(pcm)
        self._inbox.put_nowait(OutputAudio(pcm, None, self._clock.now()))

    def _lost(self, message: str) -> None:
        """The media or peer connection failed."""
        self._fail({"code": "webrtc_connection_lost", "message": message})

    def _fail(self, error: dict[str, Any]) -> None:
        if self._closing:
            return
        self.errors.append(error)
        if self.pump_error is None:
            self.pump_error = ProviderError(
                ProviderFailure.STREAM_BROKEN,
                f"Live session failed: {str(error)[:300]}",
                provider=NAME,
                model=self._model,
            )
            self._inbox.put_nowait(None)
        self._ready.set()

    def _accept(self, data: dict[str, Any]) -> None:
        """One data-channel event (tau2's ``_accept_event`` + ``_normalize_event``)."""
        kind = data.get("type", "")
        now = self._clock.now()
        if kind == "session.started":
            self._started.set()
            self._ready.set()
        elif kind == "session.usage.updated":
            self._voice_seconds(data)
        elif kind == "session.closed":
            self._voice_seconds(data)
            self._closed.set()
            self._fail({"code": "session_closed", "message": self.session_id})
        elif kind == "session.output_transcript.delta":
            # Placed when it arrives, not when a step reads it.
            if data.get("delta"):
                self._inbox.put_nowait(
                    OutputTranscript(None, data["delta"], now, self._received_bytes)
                )
        elif kind == "session.input_transcript.delta":
            self._inbox.put_nowait(InputSpeech(True, data.get("start_ms"), now))
        elif kind == "session.delegation.created":
            self.delegations += 1
        elif kind == "error":
            self._fail(data.get("error") or data)
        elif kind == "response.event":
            self._backend(data.get("delegation_id") or "", data.get("event") or {})

    def _voice_seconds(self, data: dict[str, Any]) -> None:
        """A cumulative snapshot of the session's billed seconds; never summed."""
        seconds = (data.get("usage") or {}).get("seconds")
        if seconds is not None:
            self._seconds = float(seconds)

    def _backend(self, delegation: str, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "response.created":
            self._response_ids[delegation] = (event.get("response") or {}).get(
                "id", delegation
            )
        elif kind == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                return
            response_id = self._response_ids.get(delegation, delegation)
            call_id = item["call_id"]
            self._tool_calls.setdefault(response_id, set()).add(call_id)
            if call_id in self._surfaced:
                return
            self._surfaced.add(call_id)
            invocation = ToolInvocation.from_json(
                tool_use_id=call_id,
                name=item.get("name") or "",
                raw=item.get("arguments") or "",
            )
            self._tool_tasks.append(asyncio.create_task(self._answer(invocation)))
        elif kind == "response.completed":
            response = event.get("response") or {}
            response_id = response.get("id") or self._response_ids.get(delegation, "")
            self._responses += 1
            for key, value in backend_usage(response.get("usage") or {}).items():
                self._usage[key] += value
            # Live compacts response.output; the item.done events are the batch.
            required = self._tool_calls.pop(response_id, set())
            if required and response_id not in self._continued:
                self._pending[response_id] = required
                self._continue()
        elif kind in ("response.failed", "response.incomplete", "error"):
            self._fail({"code": f"backend_{kind}", "message": str(event)[:500]})

    async def _answer(self, invocation: ToolInvocation) -> None:
        if self._tools is None:
            output = json.dumps({"error": "no tools are configured for this session"})
        else:
            result = await self._tools.invoke(invocation, at_s=self._clock.now())
            output = result.as_text()
        if self._conn is None or self._closing:
            return
        self._send(
            {
                "type": "response.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": invocation.tool_use_id,
                    "output": output,
                },
            }
        )
        self._returned.add(invocation.tool_use_id)
        self._continue()

    def _continue(self) -> None:
        """``response.create`` for each completed response whose calls are all back."""
        for response_id, required in list(self._pending.items()):
            if required <= self._returned:
                self._send({"type": "response.create"})
                self._continued.add(response_id)
                del self._pending[response_id]

    def _send(self, event: dict[str, Any]) -> None:
        assert self._conn is not None
        try:
            self._conn.send(event)
        except Exception as exc:  # noqa: BLE001 - the session is gone
            self._fail({"code": "send_failed", "message": str(exc)})


def _classify(exc: Exception, model: str) -> ProviderError:
    text = str(exc)
    if "401" in text or "403" in text or "access" in text.lower():
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
    requires_extra="live",
    default_model=DEFAULT_MODEL,
)


def build(*, model: str = DEFAULT_MODEL, **config: Any) -> OpenAILiveTransport:
    """``backend_model`` (required), ``voice``, ``backend_instructions`` and
    ``reasoning_effort`` make the :class:`LiveSettings`; the rest go through."""
    party = config.pop("party", None) or PartyId("agent")
    backend = config.pop("backend_model", None)
    if not backend:
        raise ValueError(f"{NAME} needs backend_model (the delegated Responses model)")
    settings = LiveSettings(
        backend_model=backend,
        **{
            k: config.pop(k)
            for k in ("voice", "backend_instructions", "reasoning_effort")
            if k in config
        },
    )
    return OpenAILiveTransport(
        party=PartyId(str(party)), model=model, settings=settings, **config
    )


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
