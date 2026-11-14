"""openai.llm: the body goes out unchanged; tool calls and failures parse."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from harbor_voice.providers import base as provider_base
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

    monkeypatch.setattr(provider_base.asyncio, "sleep", instant)


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
        assert round_.usage == payload["usage"]  # the cascade reports it
        (call,) = round_.calls
        assert (call.tool_use_id, call.name, call.arguments) == (
            "call_1",
            "find_user",
            {"id": "u1"},
        )

    @pytest.mark.parametrize("effort", [None, "none"])
    async def test_reasoning_effort_is_sent_only_when_set(self, effort) -> None:
        """Reasoning models refuse tools at their default effort; "none" lets
        a cascade's LLM call them."""
        client, seen = _client(lambda _: httpx.Response(200, json=_completion("Hi")))
        llm = oa.OpenAILLM(client=client, api_key="k", reasoning_effort=effort)
        await llm.chat(BODY["messages"], [])
        body = json.loads(seen[0].content)
        assert body.get("reasoning_effort") == effort

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

    @pytest.mark.parametrize(
        ("status", "kind", "attempts"),
        [
            (402, ProviderFailure.AUTH, 1),
            (404, ProviderFailure.NOT_ENABLED, 1),
            (408, ProviderFailure.TIMEOUT, oa.RETRY_ATTEMPTS),
            (504, ProviderFailure.TIMEOUT, oa.RETRY_ATTEMPTS),
        ],
    )
    async def test_statuses_map_like_every_other_provider(
        self, status: int, kind: ProviderFailure, attempts: int
    ) -> None:
        client, seen = _client(lambda _: httpx.Response(status, text="no"))
        with pytest.raises(ProviderError) as info:
            await oa.OpenAILLM(client=client, api_key="k").complete(BODY)
        assert info.value.kind is kind
        assert len(seen) == attempts

    async def test_a_dead_connection_is_retried_then_a_timeout(self) -> None:
        def refuse(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client, seen = _client(refuse)
        with pytest.raises(ProviderError) as info:
            await oa.OpenAILLM(client=client, api_key="k").complete(BODY)
        assert info.value.kind is ProviderFailure.TIMEOUT
        assert len(seen) == oa.RETRY_ATTEMPTS

    async def test_no_key_is_auth_before_any_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client, seen = _client(lambda _: httpx.Response(200, json=_completion("x")))
        with pytest.raises(ProviderError) as info:
            await oa.OpenAILLM(client=client).complete(BODY)
        assert info.value.kind is ProviderFailure.AUTH
        assert seen == []


SPEC_ARGS = ("find_user", "Look a customer up.", {"type": "object"})

HISTORY: list[dict[str, Any]] = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Find me, I am u1."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool("find_user", '{"id": "u1"}', "call_a")],
    },
    {"role": "tool", "tool_call_id": "call_a", "content": '{"name": "Ana"}'},
]

REASONING = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [],
    "encrypted_content": "opaque",
}


def _response(*output: dict[str, Any], status: str = "completed") -> dict:
    return {
        "model": "gpt-6-luna",
        "status": status,
        "output": list(output),
        "usage": {
            "input_tokens": 50,
            "input_tokens_details": {"cached_tokens": 10},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 12},
            "total_tokens": 70,
        },
    }


def _call_item(call_id: str, name: str, arguments: str) -> dict:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _message_item(text: str) -> dict:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class TestResponsesAPI:
    def _llm(self, client: httpx.AsyncClient, **config: Any) -> oa.OpenAILLM:
        return oa.OpenAILLM(
            client=client,
            api_key="k",
            model="gpt-6-luna",
            api="responses",
            reasoning_effort="low",
            **config,
        )

    async def test_the_request_has_input_items_flat_tools_and_effort(self) -> None:
        from harbor_voice.providers.tools import ToolSpec

        client, seen = _client(
            lambda _: httpx.Response(200, json=_response(_message_item("Hi Ana.")))
        )
        spec = ToolSpec(*SPEC_ARGS)
        await self._llm(client).chat(HISTORY, [spec])

        (request,) = seen
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        assert body["input"] == [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Find me, I am u1."},
            {
                "type": "function_call",
                "call_id": "call_a",
                "name": "find_user",
                "arguments": '{"id": "u1"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_a",
                "output": '{"name": "Ana"}',
            },
        ]
        assert body["tools"] == [
            {
                "type": "function",
                "name": "find_user",
                "description": "Look a customer up.",
                "parameters": {"type": "object"},
            }
        ]
        assert body["reasoning"] == {"effort": "low"}
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
        # Reasoning models refuse temperature on /responses.
        assert "temperature" not in body and "messages" not in body

    @pytest.mark.parametrize("effort", [None, "none"])
    async def test_temperature_is_sent_without_reasoning(self, effort) -> None:
        client, seen = _client(
            lambda _: httpx.Response(200, json=_response(_message_item("Hi")))
        )
        llm = oa.OpenAILLM(
            client=client, api_key="k", api="responses", reasoning_effort=effort
        )
        await llm.chat(HISTORY[:2], [])
        body = json.loads(seen[0].content)
        assert body["temperature"] == 0.0
        assert body.get("reasoning") == (None if effort is None else {"effort": effort})
        assert "tools" not in body

    async def test_a_tool_round_then_a_final_answer_replays_the_reasoning(
        self,
    ) -> None:
        """Reasoning items from a tool round go back in front of its calls."""
        from harbor_voice.cascade.duplex import _assistant
        from harbor_voice.providers.tools import ToolSpec

        replies = iter(
            [
                _response(
                    REASONING,
                    _call_item("call_b", "find_user", '{"id": "u2"}'),
                    _call_item("call_c", "find_user", "{bad"),
                ),
                _response(_message_item("Found "), _message_item("both.")),
            ]
        )
        client, seen = _client(lambda _: httpx.Response(200, json=next(replies)))
        llm = self._llm(client)
        spec = ToolSpec(*SPEC_ARGS)
        messages = list(HISTORY[:2])

        first = await llm.chat(messages, [spec])
        assert first.text == ""
        assert [(c.tool_use_id, c.name, c.arguments) for c in first.calls] == [
            ("call_b", "find_user", {"id": "u2"}),
            ("call_c", "find_user", {}),
        ]

        # The history as the cascade keeps it.
        messages.append(_assistant(first))
        messages += [
            {"role": "tool", "tool_call_id": c.tool_use_id, "content": "ok"}
            for c in first.calls
        ]
        second = await llm.chat(messages, [spec])
        assert second.text == "Found both." and second.calls == ()

        items = json.loads(seen[1].content)["input"]
        assert items[2:] == [
            REASONING,
            {
                "type": "function_call",
                "call_id": "call_b",
                "name": "find_user",
                "arguments": '{"id": "u2"}',
            },
            {
                "type": "function_call",
                "call_id": "call_c",
                "name": "find_user",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_b", "output": "ok"},
            {"type": "function_call_output", "call_id": "call_c", "output": "ok"},
        ]

    async def test_assistant_text_and_content_parts_convert(self) -> None:
        client, seen = _client(
            lambda _: httpx.Response(200, json=_response(_message_item("ok")))
        )
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "user", "content": "Bye"},
        ]
        await self._llm(client).chat(messages, [])
        assert json.loads(seen[0].content)["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
            },
            {"role": "user", "content": "Bye"},
        ]

    async def test_usage_is_renamed_to_chat_completions_keys(self) -> None:
        client, _ = _client(
            lambda _: httpx.Response(200, json=_response(_message_item("Hi.")))
        )
        reply = await self._llm(client)._respond(HISTORY[:2], ())
        assert reply.usage == {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
            "prompt_tokens_details": {"cached_tokens": 10},
            "completion_tokens_details": {"reasoning_tokens": 12},
        }
        record = oa.usage_record([reply], component="agent")
        assert (record.input_tokens, record.cached_input_tokens) == (50, 10)
        assert record.output_tokens == 20

    async def test_respond_uses_responses_and_complete_stays_on_chat(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/responses":
                return httpx.Response(200, json=_response(_message_item("Sure.")))
            return httpx.Response(200, json=_completion("Chat."))

        client, seen = _client(handler)
        llm = self._llm(client)
        chunks = [c async for c in llm.respond("Again?", ["Hi", "Hello"])]
        assert chunks == ["Sure."]
        assert json.loads(seen[0].content)["input"] == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Again?"},
        ]
        assert (await llm.complete(BODY)).text == "Chat."
        assert seen[1].url.path == "/v1/chat/completions"

    @pytest.mark.parametrize(
        ("status", "kind", "attempts"),
        [
            (400, ProviderFailure.MALFORMED, 1),
            (401, ProviderFailure.AUTH, 1),
            (404, ProviderFailure.NOT_ENABLED, 1),
            (429, ProviderFailure.THROTTLED, oa.RETRY_ATTEMPTS),
        ],
    )
    async def test_statuses_map_as_on_chat(
        self, status: int, kind: ProviderFailure, attempts: int
    ) -> None:
        client, seen = _client(lambda _: httpx.Response(status, text="no"))
        with pytest.raises(ProviderError) as info:
            await self._llm(client).chat(HISTORY, [])
        assert info.value.kind is kind
        assert len(seen) == attempts

    async def test_a_failed_response_is_malformed(self) -> None:
        payload = {"status": "failed", "error": {"code": "server_error"}}
        client, _ = _client(lambda _: httpx.Response(200, json=payload))
        with pytest.raises(ProviderError) as info:
            await self._llm(client).chat(HISTORY, [])
        assert info.value.kind is ProviderFailure.MALFORMED

    def test_an_unknown_api_is_refused(self) -> None:
        with pytest.raises(ValueError, match="api must be one of"):
            oa.OpenAILLM(api="completions")  # type: ignore[arg-type]


class TestInstructions:
    """The agent's instructions, sent the ways OpenAI accepts them."""

    async def _sent(self, **config: Any) -> dict:
        api = config.get("api", "chat")
        reply = (
            _response(_message_item("Sure."))
            if api == "responses"
            else _completion("Sure.")
        )
        client, seen = _client(lambda _: httpx.Response(200, json=reply))
        llm = oa.OpenAILLM(client=client, api_key="k", **config)
        _ = [t async for t in llm.respond("Change my flight.", [])]
        return json.loads(seen[0].content)

    async def test_no_system_prompt_sends_none(self) -> None:
        body = await self._sent()
        assert [m["role"] for m in body["messages"]] == ["user"]

    async def test_a_system_prompt_goes_first_as_system(self) -> None:
        body = await self._sent(system="No markdown.")
        assert body["messages"][0] == {"role": "system", "content": "No markdown."}

    async def test_it_can_go_as_a_developer_message(self) -> None:
        body = await self._sent(system="No markdown.", system_role="developer")
        assert body["messages"][0] == {
            "role": "developer",
            "content": "No markdown.",
        }

    async def test_the_responses_api_can_take_it_as_instructions(self) -> None:
        body = await self._sent(
            system="No markdown.", system_role="instructions", api="responses"
        )
        assert body["instructions"] == "No markdown."
        assert [i.get("role") for i in body["input"]] == ["user"]

    def test_instructions_need_the_responses_api(self) -> None:
        with pytest.raises(ValueError, match="Responses API"):
            oa.OpenAILLM(api_key="k", system="x", system_role="instructions")

    async def test_it_follows_a_conversations_own_instructions(self) -> None:
        client, seen = _client(lambda _: httpx.Response(200, json=_completion("Ok.")))
        llm = oa.OpenAILLM(
            client=client, api_key="k", system="No markdown.", system_role="developer"
        )
        await llm.chat(
            [
                {"role": "system", "content": "tau2 policy"},
                {"role": "user", "content": "Hi"},
            ],
            [],
        )
        body = json.loads(seen[0].content)
        assert body["messages"] == [
            {"role": "developer", "content": "tau2 policy"},
            {"role": "developer", "content": "No markdown."},
            {"role": "user", "content": "Hi"},
        ]


@pytest.mark.parametrize("body", ["<html>loading</html>", "[1, 2]"])
async def test_a_200_that_is_not_a_json_object_is_malformed(body: str) -> None:
    """A proxy in front of a self-hosted server can answer 200 with a page."""
    client, _ = _client(lambda _: httpx.Response(200, text=body))
    llm = oa.OpenAILLM(api_key="k", client=client)
    with pytest.raises(ProviderError) as caught:
        await llm.complete(dict(BODY))
    assert caught.value.kind is ProviderFailure.MALFORMED
