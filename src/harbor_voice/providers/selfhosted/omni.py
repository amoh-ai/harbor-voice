"""An omni model: hears the caller's audio and writes the reply as text.

Qwen3-Omni and models like it take audio in a chat message (an
``input_audio`` content part) and answer in text, so one call does a
cascade's hearing and reasoning together. A TTS then speaks the reply
(:class:`~harbor_voice.cascade.omni.OmniCascadeTransport`).

Both places that serve one use OpenAI's chat completions route:

``selfhosted.omni``
    Your server (vLLM or vLLM-Omni), at ``SELFHOSTED_OMNI_BASE_URL``
    (without ``/v1``) or ``base_url=``; ``SELFHOSTED_API_KEY`` as the bearer
    token when the server was started with ``--api-key``.
``openrouter.omni``
    OpenRouter, with ``OPENROUTER_API_KEY``.

The model half of the spec is the served model, e.g.
``selfhosted.omni/Qwen/Qwen3-Omni-30B-A3B-Instruct`` or
``openrouter.omni/qwen/qwen3.8-omni-flash``.
"""

from __future__ import annotations

import base64
import io
import os
import time
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from harbor_voice.audio.format import CANONICAL
from harbor_voice.providers.base import HTTPSession
from harbor_voice.providers.base import PooledHTTP
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import http_failure
from harbor_voice.providers.base import post_with_retries
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.selfhosted.speech import API_KEY_ENV
from harbor_voice.providers.selfhosted.speech import sandbox_env

NAME = "selfhosted.omni"
BASE_URL_ENV = "SELFHOSTED_OMNI_BASE_URL"
RETRY_ATTEMPTS = 3

DEFAULT_SYSTEM = (
    "You are a helpful agent on a phone call. You hear the caller's voice. "
    "Answer in one to three short, plain spoken sentences, with no markdown, "
    "lists or emojis, since your words are read aloud."
)


@dataclass(frozen=True, slots=True)
class OmniTurn:
    """One past exchange: what the caller said (as audio) and the reply."""

    audio: bytes
    """PCM16 mono at :data:`~harbor_voice.audio.format.CANONICAL`'s rate."""
    reply: str


@dataclass(frozen=True, slots=True)
class OmniReply:
    text: str
    latency_s: float
    usage: dict[str, Any]


class OmniLLM(PooledHTTP):
    """Audio in, text out, over ``/v1/chat/completions``.

    Args:
        model: The served model.
        base_url: The server, with or without ``/v1``.
        api_key: Bearer token; ``None`` sends none.
        system: The agent's instructions.
        temperature, max_tokens: Sent as given.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        system: str = DEFAULT_SYSTEM,
        temperature: float = 0.2,
        max_tokens: int = 400,
        timeout_s: float = 120.0,
        name: str = NAME,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._url = base_url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions"
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._http = HTTPSession(client, follow_redirects=True)

    async def reply(self, audio: bytes, history: Sequence[OmniTurn] = ()) -> OmniReply:
        """The model's text answer to the caller's latest ``audio``.

        Earlier turns go back as they happened: the caller's audio, then the
        reply. The model never wrote down what it heard, so audio is the only
        faithful record of the caller's side.
        """
        messages: list[dict[str, Any]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        for turn in history:
            messages.append({"role": "user", "content": [_audio_part(turn.audio)]})
            messages.append({"role": "assistant", "content": turn.reply})
        messages.append({"role": "user", "content": [_audio_part(audio)]})
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        started = time.perf_counter()
        response = await post_with_retries(
            self._http.client(self._timeout_s),
            self._url,
            attempts=RETRY_ATTEMPTS,
            provider=self.name,
            model=self.model,
            retry_statuses=(408, 409, 429),
            json=body,
            headers=headers,
        )
        latency = time.perf_counter() - started
        if response.status_code != 200:
            raise ProviderError(
                http_failure(
                    response.status_code, not_found=ProviderFailure.NOT_ENABLED
                ),
                f"HTTP {response.status_code}: {response.text[:300]}",
                provider=self.name,
                model=self.model,
            )
        try:
            data = response.json()
            text = data["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"no choices in the reply: {response.text[:300]}",
                provider=self.name,
                model=self.model,
            ) from exc
        if isinstance(text, list):  # content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        return OmniReply(
            text=str(text).strip(),
            latency_s=latency,
            usage=dict(data.get("usage") or {}),
        )


def wav(pcm: bytes, rate: int = CANONICAL.sample_rate_hz) -> bytes:
    """PCM16 mono as a WAV file."""
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()


def _audio_part(pcm: bytes) -> dict[str, Any]:
    return {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(wav(pcm)).decode(), "format": "wav"},
    }


def build(*, model: str, **config: Any) -> OmniLLM:
    base = config.pop("base_url", None) or os.environ.get(BASE_URL_ENV)
    if not base:
        raise ProviderError(
            ProviderFailure.AUTH,
            f"{BASE_URL_ENV} is not set and no base_url was configured",
            provider=NAME,
            model=model,
        )
    key = config.pop("api_key", None) or os.environ.get(API_KEY_ENV)
    return OmniLLM(model=model, base_url=base, api_key=key, **config)


SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.OMNI,
    capabilities=ProviderCapabilities(modality=Modality.OMNI, streams=False),
    credential_provider="selfhosted-omni",
    sandbox_env=sandbox_env,
)
"""No default model: the server decides what it serves."""

REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
