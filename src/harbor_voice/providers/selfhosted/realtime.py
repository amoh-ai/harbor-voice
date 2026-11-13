"""A self-hosted speech-to-speech model over vLLM-Omni's realtime websocket.

vLLM-Omni serves ``/v1/realtime`` as a subset of OpenAI's Realtime API, and
documents it for Qwen3-Omni
(docs.vllm.ai/projects/vllm-omni, "Realtime API"): the same handshake, the
same ``input_audio_buffer`` and ``response`` events, server VAD and function
calls. So this is
:class:`~harbor_voice.providers.openai.realtime.OpenAIRealtimeTransport`
with four differences, each an override here:

- the URL: ``SELFHOSTED_REALTIME_BASE_URL`` (the server, without ``/v1``)
  or ``url=``; ``SELFHOSTED_API_KEY`` as the bearer token when the server
  was started with ``--api-key``;
- ``session.update`` in the subset vLLM-Omni accepts: PCM16 at 24 kHz both
  ways (any other format is refused), server VAD, a voice, no input
  transcription or noise reduction;
- the provider name in the ref and in errors;
- no permessage-deflate: through Modal's web proxy a compressed session is
  closed before ``session.created`` arrives.

The model half of the spec is the served model, e.g.
``selfhosted.realtime/Qwen/Qwen3-Omni-30B-A3B-Instruct``.
"""

from __future__ import annotations

import os
from typing import Any
from typing import ClassVar
from typing import override

from harbor_voice.ids import PartyId
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderRef
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.openai.realtime import OpenAIRealtimeTransport
from harbor_voice.providers.openai.realtime import RealtimeSettings
from harbor_voice.providers.openai.realtime import wire_format
from harbor_voice.providers.selfhosted.speech import sandbox_env
from harbor_voice.transport.base import SessionConfig

NAME = "selfhosted.realtime"
KEY_ENV = "SELFHOSTED_API_KEY"
URL_ENV = "SELFHOSTED_REALTIME_BASE_URL"

SETTINGS = RealtimeSettings(
    voice="ethan",
    audio_format="pcm24k",
    noise_reduction=None,
    tool_choice=None,
)
"""Qwen3-Omni's ``ethan`` voice, as vLLM-Omni's example; PCM16 at 24 kHz, the
only format it takes."""


class SelfHostedRealtimeTransport(OpenAIRealtimeTransport):
    """One session with a speech-to-speech model on your own server."""

    provider_name: ClassVar[str] = NAME
    api_key_env: ClassVar[str] = KEY_ENV
    connect_options: ClassVar[dict[str, Any]] = {"compression": None}

    def __init__(
        self,
        *,
        model: str,
        settings: RealtimeSettings = SETTINGS,
        url: str | None = None,
        **kwargs: Any,
    ) -> None:
        if settings.audio_format != "pcm24k":
            raise ValueError(f"{NAME} takes PCM16 at 24 kHz only")
        super().__init__(model=model, settings=settings, url=url or "", **kwargs)

    @property
    @override
    def ref(self) -> ProviderRef:
        return ProviderRef.build(
            provider=NAME,
            model=self._model,
            config={
                "voice": self._settings.voice,
                "system_prompt": self._system_prompt,
            },
            capabilities=SPEC.capabilities,
        )

    @override
    def session_update(self) -> dict[str, Any]:
        """``session.update`` in the fields vLLM-Omni documents."""
        s = self._settings
        fmt = wire_format(s.audio_format)
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
            "audio": {
                "input": {
                    "format": fmt,
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": s.vad_threshold,
                        "prefix_padding_ms": s.prefix_padding_ms,
                        "silence_duration_ms": s.silence_duration_ms,
                    },
                },
                "output": {"format": fmt, "voice": s.voice},
            },
        }
        return {"type": "session.update", "session": session}

    @override
    async def open(self, config: SessionConfig) -> None:
        if not self._url:
            base = os.environ.get(URL_ENV)
            if not base:
                raise ProviderError(
                    ProviderFailure.AUTH,
                    f"{URL_ENV} is not set and no url was configured",
                    provider=NAME,
                    model=self._model,
                )
            self._url = realtime_url(base)
        # A server started without --api-key ignores the token, so any value does.
        self._api_key = self._api_key or os.environ.get(KEY_ENV) or "none"
        await super().open(config)


_SOCKET_SCHEMES = {"https": "wss", "wss": "wss", "http": "ws", "ws": "ws"}


def realtime_url(base: str) -> str:
    """``https://host[/v1]`` as ``wss://host/v1/realtime``; ``http`` as ``ws``."""
    base = base.rstrip("/").removesuffix("/v1")
    scheme, sep, rest = base.partition("://")
    if not sep or scheme not in _SOCKET_SCHEMES:
        raise ValueError(f"{URL_ENV} needs an http(s) or ws(s) URL, got {base!r}")
    return f"{_SOCKET_SCHEMES[scheme]}://{rest}/v1/realtime"


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
    credential_provider="selfhosted-realtime",
    sandbox_env=sandbox_env,
)


def build(*, model: str, **config: Any) -> SelfHostedRealtimeTransport:
    party = config.pop("party", None) or PartyId("agent")
    return SelfHostedRealtimeTransport(party=PartyId(str(party)), model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
