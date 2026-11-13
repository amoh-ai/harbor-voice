"""openai.llm: the body goes out unchanged; tool calls and failures parse."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.openai import llm as oa

pytestmark = pytest.mark.unit

BODY = {
    "model": "gpt-4.1-2025-04-14",
    "messages": [{"role": "system", "content": "You are a customer."}],
    "temperature": 0.0,
}


def _completion(
    content: str | None = None, tool_calls: list[Any] | None = None
) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "model": "gpt-4.1-2025-04-14",
        "choices": [{"message": message}],
        "usage": {"total_tokens": 9},
    }


def _tool(name: str, arguments: str, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _client(handler) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(record)), seen


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr(oa.asyncio, "sleep", instant)


class TestOpenAILLM:
    async def test_the_body_is_sent_unchanged(self) -> None:
        client, seen = _client(lambda _: httpx.Response(200, json=_completion("Hi.")))
        llm = oa.OpenAILLM(client=client, api_key="k")
        reply = await llm.complete(BODY)

        (request,) = seen
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer k"
        assert json.loads(request.content) == BODY
        assert reply.text == "Hi." and not reply.is_tool_call
        assert reply.usage == {"total_tokens": 9}

    async def test_tool_calls_are_parsed_and_bad_json_is_kept_raw(self) -> None:
        payload = _completion(
            None,
            [
                _tool("toggle_airplane_mode", '{"on": false}'),
                _tool("check", "{not json", "call_2"),
            ],
        )
        client, _ = _client(lambda _: httpx.Response(200, json=payload))
        reply = await oa.OpenAILLM(client=client, api_key="k").complete(BODY)

        first, second = reply.tool_calls
        assert (first.name, first.arguments) == ("toggle_airplane_mode", {"on": False})
        assert second.arguments == {} and second.raw_arguments == "{not json"

    async def test_a_conversation_round_offers_the_tools_and_returns_the_calls(
        self,
    ) -> None:
        from harbor_voice.providers.tools import ToolSpec

        payload = _completion("One moment.", [_tool("find_user", '{"id": "u1"}')])
        client, seen = _client(lambda _: httpx.Response(200, json=payload))
        llm = oa.OpenAILLM(client=client, api_key="k")
        spec = ToolSpec("find_user", "Look a customer up.", {"type": "object"})
        round_ = await llm.chat(BODY["messages"], [spec])

        body = json.loads(seen[0].content)
        assert body["messages"] == BODY["messages"]
        assert body["tools"] == [spec.as_chat()]
        assert round_.text == "One moment."
        (call,) = round_.calls
        assert (call.tool_use_id, call.name, call.arguments) == (
            "call_1",
            "find_user",
            {"id": "u1"},
        )

    async def test_429_is_retried_and_401_is_not(self) -> None:
        replies = iter(
            [httpx.Response(429), httpx.Response(200, json=_completion("ok"))]
        )
        client, seen = _client(lambda _: next(replies))
        await oa.OpenAILLM(client=client, api_key="k").complete(BODY)
        assert len(seen) == 2

        client, seen = _client(lambda _: httpx.Response(401, text="no"))
        with pytest.raises(ProviderError) as info:
            await oa.OpenAILLM(client=client, api_key="k").complete(BODY)
        assert info.value.kind is ProviderFailure.AUTH and len(seen) == 1
