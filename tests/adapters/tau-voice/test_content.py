"""tau-voice caller words: OpenAI provider, sidecar wire format, Tau2Content."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from tau_voice import Sidecar
from tau_voice import SidecarError
from tau_voice import Tau2Content
from harbor_voice.providers.openai import llm as oa
from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.reactive import HangUp
from tau_voice.reactive import Heard
from tau_voice.reactive import ReactiveContentSource
from tau_voice.reactive import Say
from tau_voice.reactive import StepRecord
from tau_voice.reactive import UseTools

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


def _content(llm_reply: dict) -> tuple[Tau2Content, list[httpx.Request]]:
    """Tau2Content over a fake sidecar and a fake OpenAI."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/user_context":
            return httpx.Response(200, json=BODY)
        if request.url.path == "/tool":
            sent = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "content": f"ran {sent['name']}",
                    "error": False,
                    "stop": False,
                    "too_many_errors": False,
                },
            )
        return httpx.Response(200, json=llm_reply)

    client, seen = _client(handler)
    sidecar = Sidecar("http://sidecar", client=client)
    llm = oa.OpenAILLM(client=client, api_key="k")
    return Tau2Content(sidecar, llm), seen


STEPS = (
    StepRecord(step=0, t_s=0.0, said_text="Hi, I ", said_speech=True),
    StepRecord(step=1, t_s=0.2, heard=Heard("Sure,", True)),
    StepRecord(
        step=2,
        t_s=0.4,
        said_tools=(CallerToolCall("u1", "check_status", {"x": 1}),),
    ),
    StepRecord(step=3, t_s=0.6, tool_results=(CallerToolResult("u1", "ok"),)),
)


class TestTau2Content:
    def test_it_is_a_reactive_source(self) -> None:
        content, _ = _content(_completion("x"))
        assert isinstance(content, ReactiveContentSource)

    async def test_the_step_record_reaches_the_sidecar_in_its_wire_format(self) -> None:
        content, seen = _content(_completion("My id is 42."))
        await content.respond(STEPS, Heard("Hello", True))

        context = json.loads(seen[0].content)
        assert context["pending"] == {"text": "Hello", "speech": True}
        assert context["steps"][0]["said"] == {
            "text": "Hi, I ",
            "speech": True,
            "tool_calls": [],
        }
        assert context["steps"][1] == {
            "step": 1,
            "said": None,
            "heard": {"text": "Sure,", "speech": True},
            "tool_results": [],
        }
        assert context["steps"][2]["said"]["tool_calls"] == [
            {"id": "u1", "name": "check_status", "arguments": {"x": 1}}
        ]
        assert context["steps"][3]["tool_results"] == [
            {"id": "u1", "content": "ok", "error": False}
        ]
        # ...and the body the sidecar returned goes to OpenAI untouched.
        assert json.loads(seen[1].content) == BODY

    async def test_text_is_a_line_to_say(self) -> None:
        content, _ = _content(_completion("My id is 42."))
        assert await content.respond(STEPS, None) == Say("My id is 42.")
        assert len(content.exchanges) == 1 and content.exchanges[0].step == 3

    async def test_whitespace_is_nothing_to_say(self) -> None:
        content, _ = _content(_completion("  "))
        assert await content.respond(STEPS, None) == Say("")

    @pytest.mark.parametrize(
        "token", ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"]
    )
    async def test_a_stop_token_hangs_up(self, token: str) -> None:
        content, _ = _content(_completion(f"Thanks, bye. {token}"))
        reply = await content.respond(STEPS, None)
        assert isinstance(reply, HangUp) and reply.reason == token

    async def test_tool_calls_win_over_text(self) -> None:
        content, _ = _content(_completion("###STOP###", [_tool("toggle_data", "{}")]))
        reply = await content.respond(STEPS, None)
        assert isinstance(reply, UseTools)
        assert reply.calls[0].name == "toggle_data"

    async def test_caller_tools_run_on_the_sidecar_as_the_user(self) -> None:
        content, seen = _content(_completion("x"))
        results = await content.run_tools(
            [CallerToolCall("u9", "toggle_data", {})], step=7
        )

        sent = json.loads(seen[0].content)
        assert sent == {
            "side": "user",
            "step": 7,
            "id": "u9",
            "name": "toggle_data",
            "arguments": {},
        }
        assert results == (CallerToolResult("u9", "ran toggle_data"),)


async def test_a_sidecar_refusal_is_an_error_not_a_silent_default() -> None:
    client, _ = _client(lambda _: httpx.Response(409, text="no open session"))
    with pytest.raises(SidecarError, match="409"):
        await Sidecar("http://sidecar", client=client).user_context(STEPS, None)


async def test_the_listener_counts_the_decision_models_tokens() -> None:
    from tau_voice.caller import CallerState
    from tau_voice.content import Tau2Listener

    class FakeSidecar:
        async def decide(self, steps, pending):
            return {
                "should_interrupt": False,
                "should_backchannel": True,
                "interrupt_usage": {"prompt_tokens": 900, "completion_tokens": 1},
                "backchannel_usage": {"prompt_tokens": 950, "completion_tokens": 1},
            }

    listener = Tau2Listener(FakeSidecar())  # type: ignore[arg-type]
    state = CallerState(heard_buffer=[Heard("so the fee is", True)])
    reaction = await listener.react(state)
    await listener.react(state)
    assert reaction.backchannel and not reaction.interrupt
    record = listener.usage_record()
    assert (record.component, record.requests) == ("caller.decisions", 4)
    assert (record.input_tokens, record.output_tokens) == (3700, 4)
    assert listener.summary() == {"checks": 2, "interrupt": 0, "backchannel": 2}


async def test_the_listener_names_tau2s_decision_model_and_its_cache() -> None:
    from tau_voice.caller import CallerState
    from tau_voice.content import Tau2Listener

    class FakeSidecar:
        async def decide(self, steps, pending):
            return {
                "should_interrupt": False,
                "should_backchannel": False,
                "interrupt_usage": {
                    "prompt_tokens": 900,
                    "completion_tokens": 1,
                    "cached_tokens": 768,
                },
                "backchannel_usage": None,
                "model": "gpt-4.1",
            }

    listener = Tau2Listener(FakeSidecar())  # type: ignore[arg-type]
    await listener.react(CallerState(heard_buffer=[Heard("one moment", True)]))
    record = listener.usage_record()
    assert (record.model, record.input_tokens, record.cached_input_tokens) == (
        "gpt-4.1",
        900,
        768,
    )
