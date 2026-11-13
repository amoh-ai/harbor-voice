"""Gemini Live as a :class:`~harbor_voice.transport.base.VoiceTransport`.

One websocket, one JSON message per frame, signed in either of two ways:

Google AI Studio
    An API key (``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``), sent as the
    ``x-goog-api-key`` header, as the ``google-genai`` SDK sends it, so the key
    never sits in a URL. Endpoint
    ``wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent``,
    model ``models/<id>``.
Vertex AI
    A project and location, and an OAuth token from Application Default
    Credentials (``google-auth``, the ``google`` extra). Endpoint
    ``wss://<location>-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent``,
    model ``projects/<p>/locations/<l>/publishers/google/models/<id>``.

Picked with ``vertex=True`` or, as the SDK reads it,
``GOOGLE_GENAI_USE_VERTEXAI=true``. Everything after the handshake is the
same protocol on both:

- the first message is ``setup`` and the server answers ``setupComplete``;
- audio goes in as ``realtimeInput.audio``, 16 kHz PCM16, and comes back in
  ``serverContent.modelTurn.parts[].inlineData``, 24 kHz PCM16;
- the server runs VAD itself (reported in ``voiceActivity``, which the docs
  define but do not say when it is sent) and cuts its own
  answer short on barge-in (``serverContent.interrupted``);
- tools arrive as ``toolCall.functionCalls`` and are answered with one
  ``toolResponse`` per ``toolCall``, through the same
  :class:`~harbor_voice.providers.tools.ToolRouter` every transport uses;
- ``usageMetadata`` reports tokens split by modality.

A connection lives about ten minutes; that is declared as ``max_session_s``.
Session resumption, which would stretch it, is not implemented.

Docs: https://ai.google.dev/api/live,
https://ai.google.dev/gemini-api/docs/live-api,
https://cloud.google.com/vertex-ai/generative-ai/docs/live-api/start-manage-session.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import json
import os
from collections.abc import AsyncIterator
from collections.abc import Mapping
from typing import Any
from typing import override

from harbor.models.metric import UsageInfo

from harbor_voice._deps import numpy as _numpy
from harbor_voice._deps import websockets as _websockets
from harbor_voice.audio import g711
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

NAME = "google.live"
DEFAULT_MODEL = "gemini-3.8-live"
"""Stable on AI Studio and generally available on Vertex AI, under one ID."""

API_KEY_ENVS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY")
VERTEX_ENV = "GOOGLE_GENAI_USE_VERTEXAI"
"""harbor's ``gemini`` entry, in its order (the SDK also prefers
``GOOGLE_API_KEY`` when both are set)."""

STUDIO_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
VERTEX_PATH = "/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
DEFAULT_LOCATION = "us-central1"
"""The region Vertex lists for ``gemini-3.8-live``."""

PCM16K = AudioFormat(sample_rate_hz=16_000, channels=1, encoding="pcm16")
"""What Live takes. Lower than our canonical, so input is resampled here."""

PCM24K = CANONICAL
"""What Live sends back: 24 kHz PCM16, our canonical, so no conversion."""

MAX_SESSION_S = 600.0
"""Both docs give a connection "around 10 minutes"; audio sessions get 15."""

TOKEN_COUNTS = (
    "input",
    "input_cached",
    "input_text",
    "input_audio",
    "output",
    "output_text",
    "output_audio",
)


def token_counts(usage: Mapping[str, Any]) -> dict[str, int]:
    """One ``usageMetadata`` as flat counts.

    Input is the prompt plus any tool-use prompt; output is the response plus
    thoughts, both billed at the output rate. Cached is part of input.
    """

    def by_modality(details: Any, modality: str) -> int:
        return sum(
            int(d.get("tokenCount") or 0)
            for d in details or ()
            if isinstance(d, Mapping) and d.get("modality") == modality
        )

    prompt = usage.get("promptTokensDetails")
    response = usage.get("responseTokensDetails")
    return {
        "input": int(usage.get("promptTokenCount") or 0)
        + int(usage.get("toolUsePromptTokenCount") or 0),
        "input_cached": int(usage.get("cachedContentTokenCount") or 0),
        "input_text": by_modality(prompt, "TEXT"),
        "input_audio": by_modality(prompt, "AUDIO"),
        "output": int(usage.get("responseTokenCount") or 0)
        + int(usage.get("thoughtsTokenCount") or 0),
        "output_text": by_modality(response, "TEXT"),
        "output_audio": by_modality(response, "AUDIO"),
    }


def _duration_ms(value: Any) -> int | None:
    """A protobuf ``Duration`` in JSON (``"1.250s"``) as milliseconds."""
    if not isinstance(value, str) or not value.endswith("s"):
        return None
    try:
        return round(float(value[:-1]) * 1000)
    except ValueError:
        return None


def _use_vertex() -> bool:
    return os.environ.get(VERTEX_ENV, "").lower() in ("1", "true")


def _adc_token() -> str:
    """An OAuth access token from Application Default Credentials. Blocking."""
    # Imported by name: google-auth is an optional extra, absent from the
    # environment the type checker sees unless that extra is installed.
    try:
        auth = importlib.import_module("google.auth")
        transport = importlib.import_module("google.auth.transport.requests")
    except ImportError as exc:
        raise ProviderError(
            ProviderFailure.MALFORMED,
            "Vertex AI sign-in needs google-auth. Install it:  "
            "pip install 'harbor-voice[google]'",
            provider=NAME,
        ) from exc
    try:
        credentials, _ = auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(transport.Request())
    except Exception as exc:  # noqa: BLE001 - every ADC failure is a sign-in failure
        raise ProviderError(
            ProviderFailure.AUTH,
            f"no usable Application Default Credentials ({type(exc).__name__}); "
            f"run `gcloud auth application-default login`",
            provider=NAME,
        ) from exc
    return credentials.token


class GeminiLiveTransport(VoiceTransport):
    """One Gemini Live session over a websocket, AI Studio or Vertex AI."""

    def __init__(
        self,
        *,
        party: PartyId,
        model: str = DEFAULT_MODEL,
        system_prompt: str = "",
        voice: str | None = None,
        tools: ToolRouter | None = None,
        vertex: bool | None = None,
        project: str | None = None,
        location: str | None = None,
        clock: SessionClock | None = None,
        connect: Any = None,
    ) -> None:
        self._party = party
        self._model = model
        self._system_prompt = system_prompt
        self._voice = voice
        self._tools = tools
        self._vertex = _use_vertex() if vertex is None else vertex
        self._project = project
        self._location = location
        self._clock = clock or SessionClock()
        self._connect = connect
        self._ws: Any = None
        self._open = False
        self._pump: asyncio.Task[None] | None = None
        self._inbox: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._tool_tasks: list[asyncio.Task[None]] = []
        self._turn = 0
        self._seq = 0
        self._usage = dict.fromkeys(TOKEN_COUNTS, 0)
        self._reports = 0
        self.pump_error: Exception | None = None
        self.converted_frames = 0
        """Frames resampled to 16 kHz at this edge. A declared event."""

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
            optional_ops=frozenset({OptionalOp.EVENT_STREAM, OptionalOp.SEND_TEXT}),
            audio_in=PCM16K,
            audio_out=PCM24K,
            max_session_s=MAX_SESSION_S,
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
            region=self._resolved_location() if self._vertex else None,
            config={"voice": self._voice, "system_prompt": self._system_prompt},
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
        """The session's tokens so far, summed over every ``usageMetadata``."""
        u = self._usage
        return UsageRecord(
            component=component,
            model=self._model,
            requests=self._reports,
            input_tokens=u["input"],
            cached_input_tokens=u["input_cached"],
            output_tokens=u["output"],
            input_audio_tokens=u["input_audio"],
            output_audio_tokens=u["output_audio"],
        )

    # ------------------------------------------------------------- lifecycle

    def setup(self) -> dict[str, Any]:
        """The first message: model, audio out, prompt, voice, tools."""
        generation: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        if self._voice:
            generation["speechConfig"] = {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice}}
            }
        setup: dict[str, Any] = {
            "model": self._model_path(),
            "generationConfig": generation,
            "outputAudioTranscription": {},
        }
        if self._system_prompt:
            setup["systemInstruction"] = {"parts": [{"text": self._system_prompt}]}
        if self._tools is not None and self._tools.specs:
            setup["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "parametersJsonSchema": spec.input_schema
                            or {"type": "object"},
                        }
                        for spec in self._tools.specs
                    ]
                }
            ]
        return {"setup": setup}

    @override
    async def open(self, config: SessionConfig) -> None:
        if self._open:
            return
        if config.max_session_s is not None and config.max_session_s > MAX_SESSION_S:
            raise self._error(
                ProviderFailure.MALFORMED,
                f"session of {config.max_session_s}s exceeds the documented "
                f"{MAX_SESSION_S}s connection lifetime; session resumption is "
                f"not implemented",
            )
        url, headers = await self._endpoint()
        connect = self._connect or _websockets().connect
        try:
            self._ws = await connect(url, additional_headers=headers)
            await self._ws.send(json.dumps(self.setup()))
            reply = json.loads(await self._ws.recv())
            if "setupComplete" not in reply:
                raise self._error(
                    ProviderFailure.MALFORMED,
                    f"expected setupComplete, got {str(reply)[:300]}",
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
        """Send one frame as 16 kHz PCM16, resampling anything else here."""
        self.require_open()
        if frame.fmt.channels != 1 or frame.fmt.encoding not in ("pcm16", "mulaw"):
            raise self._error(
                ProviderFailure.MALFORMED,
                f"{NAME} takes mono PCM16 or μ-law frames, got {frame.fmt}",
            )
        pcm = g711.decode(frame.pcm) if frame.fmt.encoding == "mulaw" else frame.pcm
        if frame.fmt != PCM16K:
            if frame.fmt.sample_rate_hz != PCM16K.sample_rate_hz:
                resampled = dsp.resample_clean(
                    dsp.to_array(pcm), frame.fmt.sample_rate_hz, PCM16K.sample_rate_hz
                )
                pcm = dsp.to_pcm(_numpy().asarray(resampled))
            self.converted_frames += 1
        await self._ws.send(
            json.dumps(
                {
                    "realtimeInput": {
                        "audio": {
                            "data": base64.b64encode(pcm).decode("ascii"),
                            "mimeType": f"audio/pcm;rate={PCM16K.sample_rate_hz}",
                        }
                    }
                }
            )
        )

    @override
    async def send_text(self, text: str) -> None:
        """Send text as realtime input, as the user would speak it."""
        self.require_open()
        await self._ws.send(json.dumps({"realtimeInput": {"text": text}}))

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
                yield AudioFrame(
                    pcm=event.pcm,
                    seq=self._seq,
                    party=self._party,
                    t_produced=event.t,
                    fmt=PCM24K,
                )
                self._seq += 1

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

    def _resolved_location(self) -> str:
        return (
            self._location
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or DEFAULT_LOCATION
        )

    def _model_path(self) -> str:
        if not self._vertex:
            return f"models/{self._model}"
        project = self._project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        return (
            f"projects/{project}/locations/{self._resolved_location()}"
            f"/publishers/google/models/{self._model}"
        )

    async def _endpoint(self) -> tuple[str, dict[str, str]]:
        """The URL and sign-in header for this mode. Never logged."""
        if not self._vertex:
            key = next(filter(None, map(os.environ.get, API_KEY_ENVS)), None)
            if not key:
                raise self._error(
                    ProviderFailure.AUTH,
                    f"none of {', '.join(API_KEY_ENVS)} is set (or set "
                    f"GOOGLE_GENAI_USE_VERTEXAI=true for Vertex AI)",
                )
            return STUDIO_URL, {"x-goog-api-key": key}
        if not (self._project or os.environ.get("GOOGLE_CLOUD_PROJECT")):
            raise self._error(
                ProviderFailure.AUTH,
                "Vertex AI needs a project: pass project= or set GOOGLE_CLOUD_PROJECT",
            )
        token = await asyncio.to_thread(_adc_token)
        host = f"{self._resolved_location()}-aiplatform.googleapis.com"
        return f"wss://{host}{VERTEX_PATH}", {"Authorization": f"Bearer {token}"}

    def _error(self, kind: ProviderFailure, message: str) -> ProviderError:
        return ProviderError(kind, message, provider=NAME, model=self._model)

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
        """Apply one server message. One message can carry several parts."""
        now = self._clock.now()
        utterance = f"turn-{self._turn}"
        content = data.get("serverContent") or {}
        for part in (content.get("modelTurn") or {}).get("parts") or ():
            inline = part.get("inlineData") or {}
            if inline.get("data"):
                pcm = base64.b64decode(inline["data"])
                self._inbox.put_nowait(OutputAudio(pcm, utterance, now))
        said = (content.get("outputTranscription") or {}).get("text")
        if said:
            self._inbox.put_nowait(OutputTranscript(utterance, said, now))
        if content.get("turnComplete") or content.get("interrupted"):
            self._turn += 1

        activity = data.get("voiceActivity") or {}
        if activity.get("type") in ("ACTIVITY_START", "ACTIVITY_END"):
            self._inbox.put_nowait(
                InputSpeech(
                    activity["type"] == "ACTIVITY_START",
                    _duration_ms(activity.get("audioOffset")),
                    now,
                )
            )

        calls = (data.get("toolCall") or {}).get("functionCalls") or ()
        if calls:
            self._tool_tasks.append(asyncio.create_task(self._answer(calls)))

        usage = data.get("usageMetadata")
        if isinstance(usage, Mapping):
            self._reports += 1
            for key, value in token_counts(usage).items():
                self._usage[key] += value

    async def _answer(self, calls: Any) -> None:
        """Run one ``toolCall``'s calls and send their answers together.

        Every call gets an answer: the router never returns nothing, and with
        no router the answer is an error, because an unanswered blocking call
        leaves the model waiting.
        """
        started = self._clock.now()
        invocations = [
            ToolInvocation(
                tool_use_id=str(call.get("id") or ""),
                name=str(call.get("name") or ""),
                arguments=dict(call.get("args") or {}),
            )
            for call in calls
        ]
        if self._tools is None:
            contents = [
                {"error": "no tools are configured for this session"}
                for _ in invocations
            ]
        else:
            results = await asyncio.gather(
                *(self._tools.invoke(i, at_s=started) for i in invocations)
            )
            contents = [r.content for r in results]
        if self._ws is None:
            return
        await self._ws.send(
            json.dumps(
                {
                    "toolResponse": {
                        "functionResponses": [
                            {"id": i.tool_use_id, "name": i.name, "response": content}
                            for i, content in zip(invocations, contents, strict=True)
                        ]
                    }
                }
            )
        )


def _classify(exc: Exception, model: str) -> ProviderError:
    """A handshake or setup failure as a :class:`ProviderFailure`.

    A refused handshake carries an HTTP status; a refused ``setup`` closes the
    socket with a reason naming the gRPC status, which is matched here too.
    """
    text = str(exc)
    if any(s in text for s in ("401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED")):
        kind = ProviderFailure.AUTH
    elif "API key not valid" in text:
        kind = ProviderFailure.AUTH
    elif "404" in text or "NOT_FOUND" in text or "not found" in text:
        kind = ProviderFailure.NOT_ENABLED
    elif "429" in text or "RESOURCE_EXHAUSTED" in text:
        kind = ProviderFailure.THROTTLED
    elif isinstance(exc, TimeoutError):
        kind = ProviderFailure.TIMEOUT
    else:
        kind = ProviderFailure.STREAM_BROKEN
    return ProviderError(
        kind, f"{type(exc).__name__}: {text[:300]}", provider=NAME, model=model
    )


def _sandbox_env(env: dict[str, str]) -> dict[str, str]:
    """Gemini over its API key in a sandbox, even where this machine uses
    Vertex AI: Vertex signs in with Application Default Credentials, a file
    here that is never copied into a sandbox."""
    if any(env.get(name) for name in API_KEY_ENVS) and env.get(VERTEX_ENV):
        return {**env, VERTEX_ENV: "false"}
    return env


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
    credential_provider="gemini",
    requires_extra="google",
    default_model=DEFAULT_MODEL,
    sandbox_env=_sandbox_env,
)


def build(*, model: str = DEFAULT_MODEL, **config: Any) -> GeminiLiveTransport:
    party = config.pop("party", None) or PartyId("agent")
    return GeminiLiveTransport(party=PartyId(str(party)), model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
