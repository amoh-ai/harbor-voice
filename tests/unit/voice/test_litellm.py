"""litellm.llm: real deltas, token usage, failures classified — offline.

The happy path runs through litellm itself with ``mock_response``, which
streams without a network call; failures replace ``acompletion``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import litellm
import pytest

from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.litellm import llm as ll
from harbor_voice.providers.registry import ProviderRegistry

pytestmark = pytest.mark.unit

REPLY = "Sure, I'll reschedule that for Tuesday."


async def _reply(chat: ll.LiteLLMChat, **kw: Any) -> list[str]:
    return [c async for c in chat.respond("Can you move it?", ["Hi", "Hello!"], **kw)]


class TestStreaming:
    async def test_deltas_concatenate_to_the_reply_and_usage_is_tokens(self) -> None:
        chat = ll.LiteLLMChat(model="gemini/gemini-2.5-flash")
        original = litellm.acompletion

        async def mocked(**kwargs: Any) -> Any:
            return await original(**kwargs, mock_response=REPLY)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(litellm, "acompletion", mocked)
            chunks = await _reply(chat)

        assert len(chunks) > 1, "a streaming provider yields more than one delta"
        assert "".join(chunks) == REPLY
        tokens = chat.usage.tokens
        assert tokens is not None
        assert tokens.prompt_tokens > 0 and tokens.completion_tokens > 0

    async def test_the_request_carries_history_system_and_stream_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        async def fake(**kwargs: Any) -> Any:
            seen.update(kwargs)

            async def stream() -> Any:
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))],
                    usage=None,
                )

            return stream()

        monkeypatch.setattr(litellm, "acompletion", fake)
        chat = ll.LiteLLMChat(
            model="hosted_vllm/Qwen/Qwen3-8B",
            system="Be brief.",
            api_base="http://gpu:8000/v1",
        )
        assert await _reply(chat) == ["ok"]
        assert seen["model"] == "hosted_vllm/Qwen/Qwen3-8B"
        assert seen["api_base"] == "http://gpu:8000/v1"
        assert seen["stream"] is True
        assert seen["stream_options"] == {"include_usage": True}
        assert [m["role"] for m in seen["messages"]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert seen["messages"][-1]["content"] == "Can you move it?"

    async def test_an_unpriced_model_has_no_cost_rather_than_zero(self) -> None:
        assert ll._cost("hosted_vllm/my-finetune", 10, 5, 0) is None
        assert ll._cost("gemini/gemini-2.5-flash", 1000, 100, 0) > 0


def _raise(exc: Exception):
    async def fake(**_: Any) -> Any:
        raise exc

    return fake


_REQUEST = httpx.Request("POST", "https://example.invalid")

ERRORS = [
    (litellm.AuthenticationError("bad key", "gemini", "m"), ProviderFailure.AUTH),
    (
        litellm.PermissionDeniedError(
            "denied", "bedrock", "m", httpx.Response(403, request=_REQUEST)
        ),
        ProviderFailure.AUTH,
    ),
    (litellm.NotFoundError("no model", "m", "gemini"), ProviderFailure.NOT_ENABLED),
    (litellm.RateLimitError("slow down", "xai", "m"), ProviderFailure.THROTTLED),
    (litellm.Timeout("took too long", "m", "xai"), ProviderFailure.TIMEOUT),
    (
        litellm.APIConnectionError("refused", "hosted_vllm", "m"),
        ProviderFailure.TIMEOUT,
    ),
    (litellm.InternalServerError("boom", "gemini", "m"), ProviderFailure.STREAM_BROKEN),
    (litellm.BadRequestError("bad", "m", "gemini"), ProviderFailure.MALFORMED),
]


class TestFailures:
    @pytest.mark.parametrize(("exc", "kind"), ERRORS, ids=lambda v: type(v).__name__)
    async def test_litellm_errors_map_to_failures(
        self, monkeypatch: pytest.MonkeyPatch, exc: Exception, kind: ProviderFailure
    ) -> None:
        monkeypatch.setattr(litellm, "acompletion", _raise(exc))
        with pytest.raises(ProviderError) as info:
            await _reply(ll.LiteLLMChat(model="gemini/gemini-2.5-flash"))
        assert info.value.kind is kind
        assert info.value.provider == "litellm.llm"

    async def test_an_error_after_the_first_delta_is_a_broken_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake(**_: Any) -> Any:
            async def stream() -> Any:
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="Sure"))],
                    usage=None,
                )
                raise litellm.APIConnectionError("reset", "gemini", "m")

            return stream()

        monkeypatch.setattr(litellm, "acompletion", fake)
        chat = ll.LiteLLMChat(model="gemini/gemini-2.5-flash")
        got: list[str] = []
        with pytest.raises(ProviderError) as info:
            async for chunk in chat.respond("hi", []):
                got.append(chunk)
        assert got == ["Sure"]
        assert info.value.kind is ProviderFailure.STREAM_BROKEN

    def test_an_unroutable_model_fails_at_construction(self) -> None:
        with pytest.raises(ValueError, match="cannot route"):
            ll.LiteLLMChat(model="no-such-model-xyz")


class TestRegistration:
    def test_spec_string_keeps_the_whole_litellm_model(self) -> None:
        from harbor_voice.providers import litellm as package

        registry = ProviderRegistry()
        package.register_all(registry)
        chat = registry.create(
            "litellm.llm/vertex_ai/gemini-2.5-flash", modality=Modality.LLM
        )
        assert isinstance(chat, ll.LiteLLMChat)
        assert chat.model == "vertex_ai/gemini-2.5-flash"
        assert registry.capabilities("litellm.llm/xai/grok-4").streams

    def test_a_bare_spec_names_no_model(self) -> None:
        registry = ProviderRegistry()
        registry.register(ll.LITELLM_LLM_SPEC, ll.build_llm)
        with pytest.raises(ValueError, match="names no model"):
            registry.create("litellm.llm", modality=Modality.LLM)

    @pytest.mark.parametrize(
        ("model", "slug"),
        [
            ("gemini/gemini-2.5-flash", "gemini"),
            ("vertex_ai/gemini-2.5-flash", "vertex_ai"),
            ("xai/grok-4", "xai"),
            ("bedrock/amazon.nova-pro-v1:0", "aws"),
            ("openai/gpt-4.1", "openai"),
            ("openrouter/anthropic/claude-sonnet-4.5", "openrouter"),
            ("hosted_vllm/Qwen/Qwen3-8B", None),
        ],
    )
    def test_credentials_follow_the_model(self, model: str, slug: str | None) -> None:
        assert ll.credential_slug(model) == slug
