"""Amazon Nova Sonic — speech-to-speech over Bedrock bidirectional streaming.

An S2S provider is a transport, not a
slot. Frames in, frames out, four stages instead of ten.

The event protocol is read from current AWS documentation rather than
reconstructed, because it postdates this code's authorship and a
plausible-looking wrong schema would *look* right::

    sessionStart
      promptStart                       (output config: voice, 24 kHz LPCM)
        contentStart type=TEXT role=SYSTEM -> textInput -> contentEnd
        contentStart type=AUDIO role=USER
          audioInput ... audioInput      (base64 16 kHz LPCM)
        contentEnd
      promptEnd
    sessionEnd

Output arrives as ``contentStart`` (carrying ``role`` and a possible
``generationStage``), then ``textOutput`` / ``audioOutput``, ending at
``completionEnd``.

Three facts from the docs that shape this file:

Input is 16 kHz, output is 24 kHz. The asymmetry is the provider's, not
ours. Our canonical is 24 kHz, so output needs no conversion and input is
resampled at the edge — declared, and with the anti-aliased
converter rather than the deliberately-aliasing perturbation one.

It emits the transcript of its own input (``textOutput`` with
``role: USER``). That is why ``Stage.ASR_FINAL`` is now an S2S-optional stage.
Said-vs-heard works against Nova Sonic; the reference-cascade ablation still does not,
because you cannot swap the recogniser.

Sessions are capped at 8 minutes. Declared as ``max_session_s`` so a task
demanding a longer call fails at load rather than mid-measurement.
"""

from __future__ import annotations

import asyncio
import contextlib
import base64
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from typing import override

from harbor_voice._deps import numpy as _numpy
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.audio.stages import StageMark
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
from harbor_voice.providers.base import Usage
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolRouter
from harbor.models.metric import UsageInfo
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport

MODEL_ID = "amazon.nova-2-sonic-v1:0"
NOVA_INPUT_FORMAT = AudioFormat(sample_rate_hz=16_000, channels=1, encoding="pcm16")
"""What Nova Sonic accepts. Lower than our canonical, hence the edge convert."""

NOVA_OUTPUT_FORMAT = AudioFormat(sample_rate_hz=24_000, channels=1, encoding="pcm16")
"""What Nova Sonic emits — identical to :data:`CANONICAL`, so no conversion."""

MAX_SESSION_S = 480.0
"""AWS documents an 8-minute connection limit on the bidirectional stream."""

DEFAULT_VOICE = "matthew"
_PUMP_DRAIN_S = 3.0
"""How long close() lets the receive pump end by itself before cancelling."""

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly assistant. Keep your responses short, "
    "generally one or two sentences."
)

_AWS_EXTRA = "aws"


def _require_sdk() -> Any:
    """Import the bidirectional-streaming SDK, or say how to get it.

    Deliberately not boto3: boto3's HTTP transport does not do HTTP/2
    bidirectional event streams. This needs ``aws-sdk-bedrock-runtime`` with
    the AWS CRT client, which is a separate, Developer-Preview package.
    """
    try:
        from aws_sdk_bedrock_runtime import client as _client
        from aws_sdk_bedrock_runtime import config as _config
        from aws_sdk_bedrock_runtime import models as _models
        from smithy_http.aio.crt import AWSCRTHTTPClient
    except ImportError as exc:  # pragma: no cover - exercised by the message test
        raise ProviderError(
            ProviderFailure.MALFORMED,
            "needs the bidirectional-streaming SDK, which boto3 does not "
            f"provide. Install it:  pip install 'harbor-voice[{_AWS_EXTRA}]'  "
            "(or: pip install 'aws-sdk-bedrock-runtime[awscrt]')",
            provider="aws.nova-sonic",
            model=MODEL_ID,
        ) from exc
    return _client, _config, _models, AWSCRTHTTPClient


def _resolve_region(region: str | None) -> str | None:
    """Same fallback as the Bedrock client — see `bedrock.resolve_region`."""
    import os

    return region or os.environ.get("AWS_REGION") or None


def _tool_config(router: ToolRouter | None) -> dict[str, Any]:
    """The `promptStart` tool fields, or nothing when no tools are configured.

    Omitted entirely rather than sent empty: a `toolConfiguration` with no
    tools is a different declaration from no declaration at all, and there is
    no reason to find out which way this provider reads it.
    """
    if router is None or not router.specs:
        return {}
    return {
        "toolUseOutputConfiguration": {"mediaType": "application/json"},
        "toolConfiguration": {
            "tools": [spec.as_nova() for spec in router.specs],
            "toolChoice": {"auto": {}},
        },
    }


def _event(payload: dict[str, Any]) -> str:
    return json.dumps({"event": payload})


@dataclass(slots=True)
class NovaSonicTranscript:
    """One text output, with who it belongs to and whether it may be revised."""

    text: str
    role: str
    speculative: bool = False

    @property
    def is_heard(self) -> bool:
        """Whether this is Nova Sonic's transcript of what *we* sent."""
        return self.role == "USER"


class NovaSonicTransport(VoiceTransport):
    """Nova Sonic as a :class:`VoiceTransport`.

    Deliberately thin. The base class owns drain-to-silence, recording, stage
    marking and timing; this owns connect, frame in, frame out, disconnect and
    its capability declaration.
    """

    def __init__(
        self,
        *,
        party: PartyId,
        model: str = MODEL_ID,
        region: str | None = None,
        voice: str = DEFAULT_VOICE,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        endpointing_sensitivity: str = "HIGH",
        tools: ToolRouter | None = None,
        clock: SessionClock | None = None,
    ) -> None:
        self._party = party
        # The call's clock, as every transport stamps with: frames placed on
        # another timebase put the agent hours away from the caller.
        self._clock = clock or SessionClock()
        self._model = model
        self._region = region
        self._voice = voice
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._endpointing = endpointing_sensitivity
        self._tools = tools

        self._client: Any = None
        self._stream: Any = None
        self._pump: asyncio.Task[None] | None = None
        self._inbox: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._open = False
        self._audio_content_open = False
        self._tool_tasks: list[asyncio.Task[None]] = []
        self._seq = 0

        self._prompt_name = str(uuid.uuid4())
        self._system_content = str(uuid.uuid4())
        self._audio_content = str(uuid.uuid4())

        self.transcripts: list[NovaSonicTranscript] = []
        self.stages: list[StageMark] = []
        self.resampled_frames = 0
        self.pump_error: Exception | None = None
        """What killed the receive pump, if anything. Never swallowed."""
        self._usage: dict[str, Any] = {}
        self._saw_audio = False
        self.seen_events: list[str] = []
        """Every output event key, in order — the protocol as it arrived."""
        """Frames converted 24 kHz -> 16 kHz at this edge. A declared event."""

    # ------------------------------------------------------------ declaration

    @property
    @override
    def capabilities(self) -> Capabilities:
        """What Nova Sonic can do.

        ``exposes_asr`` is true and ``exposes_partials`` is false: it hands
        back a final transcript of the input, and nothing verified says it
        streams partials. Claiming them anyway would hand the applicability
        gate a capability it then trusts.

        Nothing is ``attested``: no conformance run has verified any of it,
        so every number resting on these claims reports as ``unattested``
        until one does. That is rule 3 working, not an omission.
        """
        return Capabilities(
            duplex=frozenset({Duplex.HALF, Duplex.FULL}),
            pipeline=Pipeline.S2S,
            turn_control=TurnControl.NATIVE_VAD,
            exposes_asr=True,
            exposes_partials=False,
            exposes_stages=frozenset(
                {Stage.RESPONSE_START, Stage.RESPONSE_DONE, Stage.ASR_FINAL}
            ),
            can_be_interrupted=True,
            optional_ops=frozenset({OptionalOp.SEND_TEXT}),
            audio_in=CANONICAL,
            audio_out=NOVA_OUTPUT_FORMAT,
            max_session_s=MAX_SESSION_S,
        )

    @property
    @override
    def is_open(self) -> bool:
        return self._open

    @property
    def ref(self) -> ProviderRef:
        """What a published record says about this provider."""
        return ProviderRef.build(
            provider="aws.nova-sonic",
            model=self._model,
            region=self._region,
            config={
                "voice": self._voice,
                "temperature": self._temperature,
                "top_p": self._top_p,
                "max_tokens": self._max_tokens,
                "endpointing_sensitivity": self._endpointing,
                "system_prompt": self._system_prompt,
            },
            capabilities=SPEC.capabilities,
        )

    # ------------------------------------------------------------- lifecycle

    @override
    async def open(self, config: SessionConfig) -> None:
        """Open the stream and send the init events, in protocol order.

        Returns only once audio can flow, per the transport contract — a
        connect that succeeds by assigning a sentinel and defers failure to
        first audio is the thing harbor-voice refuses to ship.
        """
        if self._open:
            return
        client_mod, config_mod, models_mod, crt = _require_sdk()
        self._models = models_mod

        if config.max_session_s is not None and config.max_session_s > MAX_SESSION_S:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"session of {config.max_session_s}s exceeds the documented "
                f"{MAX_SESSION_S}s limit on this stream; a longer call needs "
                f"the session-continuation pattern, which this adapter does "
                f"not implement",
                provider="aws.nova-sonic",
                model=self._model,
            )

        try:
            resolved = await config_mod.AsyncBedrockRuntimeConfig.resolve(
                region=_resolve_region(self._region), transport=crt()
            )
            self._client = client_mod.AsyncBedrockRuntimeClient(config=resolved)
            self._stream = await self._client.invoke_model_with_bidirectional_stream(
                models_mod.InvokeModelWithBidirectionalStreamOperationInput(
                    model_id=self._model
                )
            )
        except Exception as exc:
            raise _classify(exc, self._model) from exc

        for payload in self._init_events():
            await self._send(payload)
        self._audio_content_open = True
        self._open = True
        self._pump = asyncio.create_task(self._receive())

    def _init_events(self) -> list[str]:
        return [
            _event(
                {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": self._max_tokens,
                            "topP": self._top_p,
                            "temperature": self._temperature,
                        },
                        "turnDetectionConfiguration": {
                            "endpointingSensitivity": self._endpointing
                        },
                    }
                }
            ),
            _event(
                {
                    "promptStart": {
                        "promptName": self._prompt_name,
                        "textOutputConfiguration": {"mediaType": "text/plain"},
                        **_tool_config(self._tools),
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": NOVA_OUTPUT_FORMAT.sample_rate_hz,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "voiceId": self._voice,
                            "encoding": "base64",
                            "audioType": "SPEECH",
                        },
                    }
                }
            ),
            _event(
                {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": self._system_content,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "SYSTEM",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            ),
            _event(
                {
                    "textInput": {
                        "promptName": self._prompt_name,
                        "contentName": self._system_content,
                        "content": self._system_prompt,
                    }
                }
            ),
            _event(
                {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": self._system_content,
                    }
                }
            ),
            _event(
                {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": NOVA_INPUT_FORMAT.sample_rate_hz,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "audioType": "SPEECH",
                            "encoding": "base64",
                        },
                    }
                }
            ),
        ]

    async def _send(self, event_json: str) -> None:
        chunk = self._models.InvokeModelWithBidirectionalStreamInputChunk(
            value=self._models.BidirectionalInputPayloadPart(
                bytes_=event_json.encode("utf-8")
            )
        )
        await self._stream.input_stream.send(chunk)

    # ------------------------------------------------------------------- i/o

    @override
    async def tx(self, frame: AudioFrame) -> None:
        """Send one frame, converting 24 kHz -> 16 kHz at this edge.

        The conversion uses the anti-aliased resampler. The perturbation
        resampler folds out-of-band energy back into the signal on purpose —
        that is a simulated telephone line — and using it here would corrupt
        what the model hears and then blame the model for it.
        """
        if not self._open:
            raise ProviderError(
                ProviderFailure.STREAM_BROKEN,
                "tx on a closed session",
                provider="aws.nova-sonic",
                model=self._model,
            )
        pcm = self._to_nova_rate(frame)
        await self._send(
            _event(
                {
                    "audioInput": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content,
                        "content": base64.b64encode(pcm).decode("utf-8"),
                    }
                }
            )
        )

    def _to_nova_rate(self, frame: AudioFrame) -> bytes:
        from harbor_voice.perturbation import dsp

        if frame.fmt.sample_rate_hz == NOVA_INPUT_FORMAT.sample_rate_hz:
            return frame.pcm
        converted = dsp.resample_clean(
            dsp.to_array(frame.pcm),
            frame.fmt.sample_rate_hz,
            NOVA_INPUT_FORMAT.sample_rate_hz,
        )
        self.resampled_frames += 1
        return dsp.to_pcm(_numpy().asarray(converted))

    @override
    async def rx(self) -> AsyncIterator[AudioFrame]:
        """Yield inbound frames until the session closes."""
        while True:
            frame = await self._inbox.get()
            if frame is None:
                return
            yield frame

    async def _receive(self) -> None:
        """Pump output events into the inbox. One long-lived task."""
        try:
            _, output = await self._stream.await_output()
            if output is None:
                raise ProviderError(
                    ProviderFailure.STREAM_BROKEN,
                    "the service returned no output stream",
                    provider="aws.nova-sonic",
                    model=self._model,
                )
            role = "ASSISTANT"
            speculative = False
            async for event in output:
                payload = getattr(getattr(event, "value", None), "bytes_", None)
                if not payload:
                    continue
                data = json.loads(payload.decode("utf-8")).get("event", {})
                self.seen_events.extend(data)

                if "completionStart" in data:
                    self._mark(Stage.RESPONSE_START)
                elif "contentStart" in data:
                    start = data["contentStart"]
                    role = start.get("role", role)
                    speculative = _is_speculative(start)
                elif "textOutput" in data:
                    self.transcripts.append(
                        NovaSonicTranscript(
                            text=data["textOutput"].get("content", ""),
                            role=role,
                            speculative=speculative,
                        )
                    )
                    if role == "USER":
                        self._mark(Stage.ASR_FINAL)
                elif "audioOutput" in data:
                    self._saw_audio = True
                    await self._emit(base64.b64decode(data["audioOutput"]["content"]))
                elif "toolUse" in data:
                    # Nova Sonic drives its own loop: it asks mid-stream and
                    # keeps speaking while we work. Handled in a task so the
                    # pump keeps draining audio -- awaiting here would stall
                    # the very speech that makes asynchronous tool calling
                    # worth having.
                    self._tool_tasks.append(
                        asyncio.create_task(self._answer_tool(data["toolUse"]))
                    )
                elif "usageEvent" in data:
                    self._usage = data["usageEvent"]
                elif "contentEnd" in data:
                    # `completionEnd` does not arrive while the prompt stays
                    # open, so the close of the audio block is what actually
                    # ends a response. Observed, not assumed.
                    if self._saw_audio:
                        self._mark(Stage.RESPONSE_DONE)
                        self._saw_audio = False
                elif "completionEnd" in data:
                    self._mark(Stage.RESPONSE_DONE)
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - kept, not swallowed
            # A bare `pass` here hid a real protocol fault during bring-up:
            # audio arrived, text did not, and there was nothing to look at.
            # The pump cannot raise into the caller, so it records instead.
            self.pump_error = exc
        finally:
            await self._inbox.put(None)

    async def _emit(self, pcm: bytes) -> None:
        """Split a provider chunk into canonical frames.

        Nova Sonic's chunk size is its own business and does not line up with
        our 20 ms frame, so a chunk is cut to frame boundaries and any
        remainder is zero-padded rather than dropped — a dropped tail would
        shorten the measured response by up to a frame.
        """
        size = NOVA_OUTPUT_FORMAT.frame_bytes()
        arrived = self._clock.now()
        for start in range(0, len(pcm), size):
            piece = pcm[start : start + size]
            if len(piece) < size:
                piece = piece + bytes(size - len(piece))
            self._seq += 1
            await self._inbox.put(
                AudioFrame(
                    pcm=piece,
                    party=self._party,
                    seq=self._seq,
                    t_produced=arrived,
                    t_rx=arrived,
                    fmt=NOVA_OUTPUT_FORMAT,
                )
            )

    def _mark(self, stage: Stage) -> None:
        self.stages.append(
            StageMark(stage=stage, t=self._clock.now(), party=self._party)
        )

    @override
    async def send_text(self, text: str) -> None:
        """Inject a text turn — declared via :data:`OptionalOp.SEND_TEXT`."""
        content = str(uuid.uuid4())
        for payload in (
            {
                "contentStart": {
                    "promptName": self._prompt_name,
                    "contentName": content,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "USER",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            },
            {
                "textInput": {
                    "promptName": self._prompt_name,
                    "contentName": content,
                    "content": text,
                }
            },
            {"contentEnd": {"promptName": self._prompt_name, "contentName": content}},
        ):
            await self._send(_event(payload))

    async def _answer_tool(self, event: dict[str, Any]) -> None:
        """Run one tool and send the three-part reply.

        Always sends a result. The docs are explicit that Nova Sonic
        waits for one after every ``toolUse``, and that failing to answer --
        *even on error* -- leaves the model in a waiting state. So an
        exception here becomes an error payload rather than silence: a hung
        agent would be our bug wearing the agent's score.
        """
        content_name = str(uuid.uuid4())
        invocation = ToolInvocation.from_json(
            tool_use_id=event.get("toolUseId", ""),
            name=event.get("toolName", ""),
            raw=event.get("content", "") or "",
        )
        try:
            if self._tools is None:
                result_json = json.dumps(
                    {"error": "no_tools", "message": "no tool router configured"}
                )
            else:
                result_json = (await self._tools.invoke(invocation)).as_nova_content()
        except Exception as exc:  # noqa: BLE001 - silence would hang the model
            result_json = json.dumps({"error": type(exc).__name__, "message": str(exc)})

        await self._send(
            _event(
                {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                        "interactive": False,
                        "type": "TOOL",
                        "role": "TOOL",
                        "toolResultInputConfiguration": {
                            "toolUseId": invocation.tool_use_id,
                            "type": "TEXT",
                            "textInputConfiguration": {"mediaType": "text/plain"},
                        },
                    }
                }
            )
        )
        await self._send(
            _event(
                {
                    "toolResult": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                        "content": result_json,
                    }
                }
            )
        )
        await self._send(
            _event(
                {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": content_name,
                    }
                }
            )
        )

    @override
    async def close(self) -> None:
        """End the session. Idempotent, and tolerant of a half-dead stream."""
        if not self._open:
            return
        self._open = False

        # Let any tool reply finish before the stream goes. A result
        # cancelled mid-send leaves the model waiting for something that
        # will never arrive, which is the one failure the docs warn about.
        if self._tool_tasks:
            pending, self._tool_tasks = self._tool_tasks, []
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=_PUMP_DRAIN_S,
                )
        try:
            if self._audio_content_open:
                await self._send(
                    _event(
                        {
                            "contentEnd": {
                                "promptName": self._prompt_name,
                                "contentName": self._audio_content,
                            }
                        }
                    )
                )
                self._audio_content_open = False
            await self._send(_event({"promptEnd": {"promptName": self._prompt_name}}))
            await self._send(_event({"sessionEnd": {}}))
        except Exception:  # noqa: BLE001 - close must tolerate a dead stream
            pass
        finally:
            try:
                await self._stream.input_stream.close()
            except Exception:  # noqa: BLE001 - as above
                pass
            if self._pump is not None:
                # Let the pump finish on its own first. Cancelling it while
                # the CRT transport is mid-body makes awscrt resolve an
                # already-cancelled future and print a traceback of its own —
                # harmless here, unreadable at a thousand concurrent trials.
                # Closing the input stream is what ends the output stream, so
                # a short wait is usually enough; cancellation is the fallback.
                try:
                    await asyncio.wait_for(self._pump, timeout=_PUMP_DRAIN_S)
                except TimeoutError:
                    self._pump.cancel()
                    try:
                        await self._pump
                    except asyncio.CancelledError:
                        pass
                except Exception:  # noqa: BLE001 - recorded as pump_error
                    pass
            await self._inbox.put(None)

    # -------------------------------------------------------------- readouts

    @property
    def usage(self) -> Usage:
        """What this session consumed, in the units Nova Sonic bills.

        It meters speech tokens separately from text tokens, which is the
        concrete reason harbor's ``UsageInfo`` is wrapped rather than bent:
        that model has three token fields and no notion of a speech token, so
        the speech count had nowhere truthful to go.
        """
        totals = self._usage.get("details", {}).get("total", {})
        speech = sum(side.get("speechTokens", 0) or 0 for side in totals.values())
        text = sum(side.get("textTokens", 0) or 0 for side in totals.values())
        return Usage(
            tokens=UsageInfo(
                prompt_tokens=int(self._usage.get("totalInputTokens", 0) or 0),
                completion_tokens=int(self._usage.get("totalOutputTokens", 0) or 0),
                cache_tokens=0,
                cost_usd=0.0,
            ),
            speech_tokens=speech,
            text_tokens=text,
        )

    @property
    def heard(self) -> list[str]:
        """What Nova Sonic transcribed from our audio — the HEARD side."""
        return [t.text for t in self.transcripts if t.is_heard]

    @property
    def said(self) -> list[str]:
        """What Nova Sonic itself said.

        Speculative text is included, and the name is what misled me.
        AWS's own sample displays assistant text precisely when
        ``generationStage`` is ``SPECULATIVE``, and in every observed run the
        speculative text is the only assistant text and it matches the spoken
        audio word for word. Excluding it — the first thing this code did —
        produced an agent that emitted 2.3 seconds of speech and reported
        having said nothing.
        """
        return [t.text for t in self.transcripts if not t.is_heard]


def _is_speculative(content_start: dict[str, Any]) -> bool:
    """Whether this content block is a draft the model may revise.

    Nova Sonic marks speculative generations in ``additionalModelFields``.
    Counting a draft as final would double-count the agent's words in every
    text metric, so drafts are kept and flagged rather than silently merged.
    """
    raw = content_start.get("additionalModelFields")
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get("generationStage") == "SPECULATIVE")
    except (ValueError, TypeError):
        return False


_NOT_ENABLED_HINTS = ("not authorized", "access denied", "accessdenied")


def _classify(
    exc: Exception, model: str, provider: str = "aws.nova-sonic"
) -> ProviderError:
    """Turn an SDK exception into the failure vocabulary.

    ``NOT_ENABLED`` and ``AUTH`` are genuinely hard to separate here — Bedrock
    raises ``AccessDeniedException`` for both a bad key and a model the
    account has not enabled in this region. When the message names the model
    we call it ``NOT_ENABLED``, because that is the overwhelmingly more common
    cause once a key has worked at all, and the message says which to check.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "throttl" in name or "throttl" in text:
        kind = ProviderFailure.THROTTLED
    elif "timeout" in name or "timeout" in text:
        kind = ProviderFailure.TIMEOUT
    elif "validation" in name:
        kind = ProviderFailure.MALFORMED
    elif any(hint in text for hint in _NOT_ENABLED_HINTS):
        kind = (
            ProviderFailure.NOT_ENABLED
            if model.lower() in text
            else ProviderFailure.AUTH
        )
    elif "inference profile" in text:
        # A config error, not a bad request: this model is only reachable via
        # an inference profile in this region. Every trial would fail the same
        # way, so it fails the job rather than producing N identical zeros.
        kind = ProviderFailure.NOT_ENABLED
    elif "serviceunavailable" in name or "internalserver" in name:
        kind = ProviderFailure.STREAM_BROKEN
    else:
        kind = ProviderFailure.STREAM_BROKEN
    detail = str(exc) or type(exc).__name__
    if kind is ProviderFailure.NOT_ENABLED:
        detail += (
            " — enable this model for your account in this region "
            "(Bedrock console -> Model access), use an inference-profile id "
            "such as 'us.<model>', or check the credentials"
        )
    return ProviderError(kind, detail, provider=provider, model=model)


SPEC = ProviderSpec(
    name="aws.nova-sonic",
    modality=Modality.S2S,
    capabilities=ProviderCapabilities(
        modality=Modality.S2S,
        streams=True,
        emits_partials=False,
        word_timings=False,
        diarizes=False,
    ),
    credential_provider="aws",
    requires_extra=_AWS_EXTRA,
    default_model=MODEL_ID,
)
"""Static description, readable without credentials or the SDK.

Nothing is attested: ``harbor_voice.conformance`` against a live session moves
a claim into ``attested``, and until it runs every number resting on these
reports as ``unattested``.
"""


def build(*, model: str = MODEL_ID, **config: Any) -> NovaSonicTransport:
    """Registry entry point."""
    party = config.pop("party", None) or PartyId("agent")
    return NovaSonicTransport(party=PartyId(str(party)), model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
