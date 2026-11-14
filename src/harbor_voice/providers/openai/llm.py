"""OpenAI chat completions.

Built for the tau-voice caller, whose request is assembled by tau2 itself (the
sidecar's ``/user_context``) and must reach OpenAI unchanged. So the primary
surface is :meth:`OpenAILLM.complete`: a request body in, one reply out, no
rewriting. tau2 calls ``litellm.completion`` non-streaming with
``num_retries=3``; this sends the same body with the same retry budget.

Plain ``httpx``, like the ElevenLabs provider, and harbor already depends on
it.

``api="responses"`` sends :meth:`OpenAILLM.chat` and :meth:`OpenAILLM.respond`
to ``/responses`` instead. OpenAI's reasoning models refuse function tools on
``/chat/completions`` unless ``reasoning_effort`` is ``"none"``; the Responses
API takes tools at any effort. :meth:`OpenAILLM.complete` always stays on
``/chat/completions``, since its body is in that format.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

import httpx

from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import HTTPSession
from harbor_voice.providers.base import PooledHTTP
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import http_failure
from harbor_voice.providers.base import post_with_retries
from harbor_voice.providers.base import require_key
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

API = Literal["chat", "responses"]
APIS: tuple[str, ...] = ("chat", "responses")


@dataclass(frozen=True, slots=True)
class ChatToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    """As returned. Kept because a model can emit JSON that does not parse."""


@dataclass(frozen=True, slots=True)
class ChatReply:
    """One completion: text, tool calls, or both.

    ``usage`` uses chat completions' keys on both APIs: ``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``, ``prompt_tokens_details.
    cached_tokens`` and ``completion_tokens_details.reasoning_tokens``. A
    Responses reply is renamed into them (see :func:`_responses_usage`).
    """

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
        reasoning_effort: Sent when set. Reasoning models refuse tools on
            ``/chat/completions`` at any effort but ``"none"``
            (``--agent-config '{"llm": {"reasoning_effort": "none"}}'``). To
            reason and call tools, use ``api="responses"``.
        api: ``"chat"`` (default) or ``"responses"``: the endpoint
            :meth:`chat` and :meth:`respond` use. On ``"responses"``,
            ``temperature`` is left out when a reasoning effort other than
            ``"none"`` is set, as reasoning models refuse it.
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
        reasoning_effort: str | None = None,
        api: API = "chat",
    ) -> None:
        if api not in APIS:
            raise ValueError(f"{NAME}: api must be one of {APIS}, not {api!r}")
        self.name = NAME
        self.api = api
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._timeout_s = timeout_s
        self._http = HTTPSession(client)
        # Responses API reasoning items, by the call ids of the round that
        # produced them. See _responses_input.
        self._rounds: dict[tuple[str, ...], list[dict[str, Any]]] = {}

    async def complete(self, body: dict[str, Any]) -> ChatReply:
        """Send ``body`` to ``/chat/completions`` unchanged and parse the reply."""
        started = time.perf_counter()
        data = await self._post("/chat/completions", body)
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
        if self.api == "responses":
            reply = await self._respond(messages, ())
        else:
            reply = await self.complete(self._body(messages))
        if reply.text:
            yield reply.text

    async def chat(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSpec]
    ) -> ChatRound:
        """The :class:`~harbor_voice.providers.tools.ChatLLM` shape: one model call.

        ``messages`` are chat completions messages on either API; on
        ``"responses"`` they are converted to input items and back.
        """
        if self.api == "responses":
            reply = await self._respond(messages, tools)
        else:
            body = self._body([dict(m) for m in messages])
            if tools:
                body["tools"] = [spec.as_chat() for spec in tools]
            reply = await self.complete(body)
        return ChatRound(
            text=reply.text or "",
            calls=tuple(
                ToolInvocation(tool_use_id=c.id, name=c.name, arguments=c.arguments)
                for c in reply.tool_calls
            ),
            usage=dict(reply.usage),
        )

    # ----------------------------------------------------------- internals

    def _body(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    async def _respond(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSpec]
    ) -> ChatReply:
        """One ``/responses`` call over a chat completions conversation."""
        body: dict[str, Any] = {
            "model": self.model,
            "input": self._responses_input(messages),
            # Stateless: nothing kept on OpenAI's side, so no previous_response_id.
            # Reasoning items come back encrypted and go back in the next input.
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        reasoning = self.reasoning_effort not in (None, "none")
        if not reasoning:
            body["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if tools:
            body["tools"] = [_responses_tool(spec) for spec in tools]
        started = time.perf_counter()
        data = await self._post("/responses", body)
        latency = time.perf_counter() - started
        output = data.get("output")
        if data.get("status") == "failed" or not isinstance(output, list):
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"no output in the reply: {str(data)[:300]}",
                provider=self.name,
                model=self.model,
            )
        texts: list[str] = []
        calls: list[ChatToolCall] = []
        for item in output:
            if item.get("type") == "message":
                texts.extend(
                    part.get("text") or ""
                    for part in item.get("content") or ()
                    if part.get("type") == "output_text"
                )
            elif item.get("type") == "function_call":
                calls.append(_responses_call(item))
        if calls:
            self._rounds[tuple(c.id for c in calls)] = [
                item for item in output if item.get("type") == "reasoning"
            ]
        return ChatReply(
            text="".join(texts) or None,
            tool_calls=tuple(calls),
            latency_s=latency,
            usage=_responses_usage(data.get("usage") or {}),
            model=data.get("model"),
        )

    def _responses_input(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Chat completions messages as Responses input items.

        The conversation is kept as chat messages (by the cascade, say), which
        have no place for reasoning. OpenAI asks for a tool round's reasoning
        items back with its results, so the ones this instance received are
        put back in front of the calls they came with, matched by call ids.
        Text-only rounds' reasoning is not kept; OpenAI requires it only
        around tool calls.
        """
        items: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.get("tool_call_id", ""),
                        "output": content
                        if isinstance(content, str)
                        else json.dumps(content),
                    }
                )
                continue
            calls = message.get("tool_calls") or ()
            if role == "assistant" and calls:
                ids = tuple(c.get("id", "") for c in calls)
                items.extend(self._rounds.get(ids, ()))
            if content:
                items.append(
                    {"role": role, "content": _responses_content(role, content)}
                )
            for call in calls:
                function = call.get("function") or {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id", ""),
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments") or "{}",
                    }
                )
        return items

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        key = require_key(
            self._api_key, "OPENAI_API_KEY", provider=self.name, model=None
        )
        model = body.get("model")
        response = await post_with_retries(
            self._http.client(self._timeout_s),
            f"{self._api_base}{path}",
            attempts=RETRY_ATTEMPTS,
            provider=self.name,
            model=model,
            retry_statuses=(408, 409, 429),
            json=body,
            headers={"Authorization": f"Bearer {key}"},
        )
        if response.status_code == 200:
            return response.json()
        raise _error(response, self.name, model)


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


def _responses_tool(spec: ToolSpec) -> dict[str, Any]:
    """A Responses ``tools[]`` entry: the function's fields at the top level."""
    return {"type": "function", **spec.as_chat()["function"]}


def _responses_call(item: Mapping[str, Any]) -> ChatToolCall:
    """A ``function_call`` output item; its ``call_id`` is what results answer."""
    return _tool_call(
        {
            "id": item.get("call_id", ""),
            "function": {
                "name": item.get("name", ""),
                "arguments": item.get("arguments") or "",
            },
        }
    )


def _responses_content(role: Any, content: Any) -> Any:
    """A chat message's content as Responses message content.

    Strings pass as they are. Chat ``text`` parts become ``output_text`` from
    the assistant and ``input_text`` otherwise; other parts pass unchanged.
    """
    if isinstance(content, str):
        return content
    kind = "output_text" if role == "assistant" else "input_text"
    return [
        {"type": kind, "text": part.get("text", "")}
        if isinstance(part, Mapping) and part.get("type") == "text"
        else part
        for part in content
    ]


def _responses_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    """Responses usage under chat completions' keys, so one reader serves both."""
    if not usage:
        return {}
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(usage.get("total_tokens") or prompt + completion),
        "prompt_tokens_details": {"cached_tokens": int(cached)},
        "completion_tokens_details": {"reasoning_tokens": int(reasoning or 0)},
    }


def _error(response: httpx.Response, provider: str, model: str | None) -> ProviderError:
    """A 404 is a model the account cannot use."""
    status = response.status_code
    return ProviderError(
        http_failure(status, not_found=ProviderFailure.NOT_ENABLED),
        f"OpenAI returned {status}: {response.text[:300]}",
        provider=provider,
        model=model,
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
