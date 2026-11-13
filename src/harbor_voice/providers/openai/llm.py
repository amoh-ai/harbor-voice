"""OpenAI chat completions.

Built for the tau-voice caller, whose request is assembled by tau2 itself (the
sidecar's ``/user_context``) and must reach OpenAI unchanged. So the primary
surface is :meth:`OpenAILLM.complete`: a request body in, one reply out, no
rewriting. tau2 calls ``litellm.completion`` non-streaming with
``num_retries=3``; this sends the same body with the same retry budget.

Plain ``httpx``, like the ElevenLabs provider: one endpoint, and harbor already
depends on it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import httpx

from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import HTTPSession
from harbor_voice.providers.base import PooledHTTP
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.tools import ChatRound
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolSpec
from harbor_voice.trajectory.model import UsageRecord

NAME = "openai.llm"
DEFAULT_MODEL = "gpt-4.1-2025-04-14"
"""tau2's ``DEFAULT_LLM_USER``."""

API_BASE = "https://api.openai.com/v1"
RETRY_ATTEMPTS = 4
"""One call plus litellm's ``num_retries=3`` (tau2's ``DEFAULT_MAX_RETRIES``)."""


@dataclass(frozen=True, slots=True)
class ChatToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    """As returned. Kept because a model can emit JSON that does not parse."""


@dataclass(frozen=True, slots=True)
class ChatReply:
    """One completion: text, tool calls, or both."""

    text: str | None
    tool_calls: tuple[ChatToolCall, ...] = ()
    latency_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)


def usage_record(
    replies: Iterable[ChatReply], *, component: str, model: str | None = None
) -> UsageRecord:
    """Chat completions' tokens, summed: prompt (cached included) and completion."""
    requests = prompt = cached = completion = 0
    for reply in replies:
        usage = reply.usage or {}
        requests += 1
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached += int(details.get("cached_tokens") or 0)
    return UsageRecord(
        component=component,
        model=model,
        requests=requests,
        input_tokens=prompt,
        cached_input_tokens=cached,
        output_tokens=completion,
    )


class OpenAILLM(PooledHTTP):
    """Chat completions, sent as given.

    Args:
        model: Used by :meth:`respond`; :meth:`complete` takes the model from
            the request body it is handed.
        api_key: Defaults to ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        api_key: str | None = None,
        api_base: str = API_BASE,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = NAME
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._timeout_s = timeout_s
        self._http = HTTPSession(client)

    async def complete(self, body: dict[str, Any]) -> ChatReply:
        """Send ``body`` to ``/chat/completions`` unchanged and parse the reply."""
        started = time.perf_counter()
        data = await self._post(body)
        latency = time.perf_counter() - started
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"no choices in the reply: {str(data)[:300]}",
                provider=self.name,
                model=body.get("model"),
            ) from exc
        calls = tuple(_tool_call(c) for c in message.get("tool_calls") or ())
        return ChatReply(
            text=message.get("content"),
            tool_calls=calls,
            latency_s=latency,
            usage=dict(data.get("usage") or {}),
            model=data.get("model"),
        )

    async def respond(self, prompt: str, history: Sequence[str]) -> AsyncIterator[str]:
        """The :class:`~harbor_voice.providers.protocols.LLMProvider` shape.

        History alternates user/assistant starting with the user, as
        ``BedrockLLM`` reads it. Yields the whole reply once — the spec says
        ``streams=False``, so nothing reads the first chunk as a first token.
        """
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": h}
            for i, h in enumerate(history)
        ]
        messages.append({"role": "user", "content": prompt})
        reply = await self.complete(
            {"model": self.model, "messages": messages, "temperature": self.temperature}
        )
        if reply.text:
            yield reply.text

    async def chat(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSpec]
    ) -> ChatRound:
        """The :class:`~harbor_voice.providers.tools.ChatLLM` shape: one completion."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = [spec.as_chat() for spec in tools]
        reply = await self.complete(body)
        return ChatRound(
            text=reply.text or "",
            calls=tuple(
                ToolInvocation(tool_use_id=c.id, name=c.name, arguments=c.arguments)
                for c in reply.tool_calls
            ),
        )

    # ----------------------------------------------------------- internals

    def _key_or_raise(self) -> str:
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError(
                ProviderFailure.AUTH,
                "OPENAI_API_KEY is not set",
                provider=self.name,
            )
        return key

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self._key_or_raise()}"}
        model = body.get("model")
        client = self._http.client(self._timeout_s)
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                response = await client.post(url, json=body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == RETRY_ATTEMPTS:
                    raise ProviderError(
                        ProviderFailure.TIMEOUT,
                        f"OpenAI request failed after {attempt} attempts: {exc}",
                        provider=self.name,
                        model=model,
                    ) from exc
                await asyncio.sleep(_backoff(attempt))
                continue
            if response.status_code == 200:
                return response.json()
            if _retryable(response.status_code) and attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(_backoff(attempt))
                continue
            raise _error(response, self.name, model)
        raise ValueError(  # pragma: no cover
            f"RETRY_ATTEMPTS must be at least 1, got {RETRY_ATTEMPTS}"
        )


def _tool_call(raw: dict[str, Any]) -> ChatToolCall:
    function = raw.get("function") or {}
    arguments = function.get("arguments") or ""
    try:
        parsed = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        # tau2 (via litellm) hands unparseable arguments on as an empty dict;
        # keep the raw string so the trajectory shows what really came back.
        parsed = {}
    return ChatToolCall(
        id=raw.get("id", ""),
        name=function.get("name", ""),
        arguments=parsed if isinstance(parsed, dict) else {},
        raw_arguments=arguments,
    )


def _retryable(status: int) -> bool:
    return status in (408, 409, 429) or status >= 500


def _backoff(attempt: int) -> float:
    return min(10.0, 2.0 ** (attempt - 1))


def _error(response: httpx.Response, provider: str, model: str | None) -> ProviderError:
    status = response.status_code
    detail = response.text[:300]
    if status in (401, 403):
        kind, what = ProviderFailure.AUTH, "rejected the key"
    elif status == 404:
        kind, what = ProviderFailure.NOT_ENABLED, "does not know the model"
    elif status == 429:
        kind, what = ProviderFailure.THROTTLED, "rate limit or quota"
    elif status >= 500:
        kind, what = ProviderFailure.STREAM_BROKEN, "server error"
    else:
        kind, what = ProviderFailure.MALFORMED, "rejected the request"
    return ProviderError(
        kind, f"OpenAI {what} ({status}): {detail}", provider=provider, model=model
    )


OPENAI_LLM_SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.LLM,
    capabilities=ProviderCapabilities(modality=Modality.LLM, streams=False),
    credential_provider="openai",
    default_model=DEFAULT_MODEL,
)


def build_llm(*, model: str = DEFAULT_MODEL, **config: Any) -> OpenAILLM:
    return OpenAILLM(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((OPENAI_LLM_SPEC, build_llm),)
