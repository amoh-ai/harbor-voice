"""xAI's Grok Voice Agent API as a :class:`~harbor_voice.transport.base.VoiceTransport`.

xAI documents its realtime API as compatible with OpenAI Realtime
(https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech,
"OpenAI Realtime API Compatibility"): the same websocket handshake with a
Bearer key, ``session.created`` / ``session.updated``, base64
``input_audio_buffer.append``, the same server events for audio, transcripts,
VAD and ``response.done``, the same function-call round trip, and
``conversation.item.truncate``. So this is
:class:`~harbor_voice.providers.openai.realtime.OpenAIRealtimeTransport` with
four differences, each an override here:

- the URL (``wss://api.x.ai/v1/realtime``) and the key (``XAI_API_KEY``);
- ``session.update`` in xAI's documented shape (``voice`` and
  ``turn_detection`` at the top of ``session``, ``language_hint`` for
  transcription, no ``tool_choice``, VAD tuning, noise reduction or
  transcription model, none of which xAI documents);
- usage: xAI bills a session by the minute and reports only
  ``input_tokens`` / ``output_tokens`` in ``response.done``, so the meter is
  the seconds the session was open plus those two counts;
- the provider name in the ref and in errors raised here.
"""

from __future__ import annotations

from typing import Any
from typing import ClassVar
from typing import override

from harbor.models.metric import UsageInfo

from harbor_voice.ids import PartyId
from harbor_voice.providers.base import ProviderRef
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.openai.realtime import OpenAIRealtimeTransport
from harbor_voice.providers.openai.realtime import RealtimeSettings
from harbor_voice.providers.openai.realtime import wire_format
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.transport.base import SessionConfig

NAME = "xai.realtime"
KEY_ENV = "XAI_API_KEY"
URL = "wss://api.x.ai/v1/realtime"
DEFAULT_MODEL = "grok-voice-3.0"
"""Pinned rather than ``grok-voice-latest``, which xAI moves to each new model."""

XAI_SETTINGS = RealtimeSettings(
    voice="eve", language="en", noise_reduction=None, tool_choice=None
)
"""``eve``, xAI's default voice; English; μ-law 8 kHz both ways, as the
OpenAI transport runs a phone benchmark."""

UNSENT = (
    "vad_threshold",
    "prefix_padding_ms",
    "silence_duration_ms",
    "transcription_model",
    "noise_reduction",
    "tool_choice",
    "parallel_tool_calls",
)
"""OpenAI settings with no documented xAI field. Changing one is refused,
not silently dropped."""


class XAIRealtimeTransport(OpenAIRealtimeTransport):
    """One Grok Voice Agent session over a websocket."""

    provider_name: ClassVar[str] = NAME
    api_key_env: ClassVar[str] = KEY_ENV
    """Read in place of ``OPENAI_API_KEY``, so an OpenAI key never reaches xAI."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        settings: RealtimeSettings = XAI_SETTINGS,
        url: str = URL,
        **kwargs: Any,
    ) -> None:
        changed = [
            f for f in UNSENT if getattr(settings, f) != getattr(XAI_SETTINGS, f)
        ]
        if changed:
            raise ValueError(f"{NAME} has no setting for {', '.join(changed)}")
        super().__init__(model=model, settings=settings, url=url, **kwargs)
        self._opened_at: float | None = None
        self._closed_at: float | None = None

    @property
    @override
    def ref(self) -> ProviderRef:
        config = {
            k: v for k, v in self._settings.as_config().items() if k not in UNSENT
        }
        return ProviderRef.build(
            provider=NAME,
            model=self._model,
            config={**config, "system_prompt": self._system_prompt},
            capabilities=SPEC.capabilities,
        )

    @property
    def seconds(self) -> float:
        """How long the session has been open: what xAI bills."""
        if self._opened_at is None:
            return 0.0
        end = self._closed_at if self._closed_at is not None else self._clock.now()
        return end - self._opened_at

    @property
    @override
    def usage(self) -> Usage:
        u = self._usage
        return Usage(
            audio_seconds=self.seconds,
            tokens=UsageInfo(
                prompt_tokens=u["input"],
                completion_tokens=u["output"],
                cache_tokens=0,
                cost_usd=0.0,
            ),
        )

    @override
    def usage_record(self, component: str = "agent") -> UsageRecord:
        return UsageRecord(
            component=component,
            model=self._model,
            requests=self._responses,
            input_tokens=self._usage["input"],
            output_tokens=self._usage["output"],
            seconds=self.seconds,
        )

    @override
    def session_update(self) -> dict[str, Any]:
        """``session.update`` as xAI's reference documents it."""
        s = self._settings
        session: dict[str, Any] = {
            "instructions": self._system_prompt,
            "voice": s.voice,
            "turn_detection": {"type": "server_vad"},
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
        if s.reasoning_effort is not None:
            session["reasoning"] = {"effort": s.reasoning_effort}
        audio_input: dict[str, Any] = {"format": wire_format(s.audio_format)}
        if s.language is not None:
            audio_input["transcription"] = {"language_hint": s.language}
        session["audio"] = {
            "input": audio_input,
            "output": {"format": wire_format(s.audio_format)},
        }
        return {"type": "session.update", "session": session}

    @override
    async def open(self, config: SessionConfig) -> None:
        if self.is_open:
            return
        await super().open(config)
        self._opened_at = self._clock.now()

    @override
    async def close(self) -> None:
        if self.is_open:
            self._closed_at = self._clock.now()
        await super().close()


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
    credential_provider="xai",
    requires_extra="voice",
    default_model=DEFAULT_MODEL,
)


def build(*, model: str = DEFAULT_MODEL, **config: Any) -> XAIRealtimeTransport:
    party = config.pop("party", None) or PartyId("agent")
    return XAIRealtimeTransport(party=PartyId(str(party)), model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
