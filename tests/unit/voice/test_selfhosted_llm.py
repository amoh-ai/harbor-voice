"""selfhosted.llm: OpenAI chat completions on your own server — offline."""

from __future__ import annotations

import httpx
import pytest

from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.selfhosted import llm as sl

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sl.BASE_URL_ENV, raising=False)
    monkeypatch.delenv(sl.API_KEY_ENV, raising=False)


async def test_it_posts_to_the_servers_chat_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(sl.BASE_URL_ENV, "http://vllm:8000/v1/")
    seen: list[httpx.Request] = []

    def reply(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Hi."}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
    llm = sl.build(model="Qwen/Qwen3-8B", client=client)
    reply_ = await llm.complete({"model": "Qwen/Qwen3-8B", "messages": []})
    assert reply_.text == "Hi."
    assert str(seen[0].url) == "http://vllm:8000/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer none"
    assert llm.name == "selfhosted.llm"


def test_without_a_server_it_says_which_variable() -> None:
    with pytest.raises(ProviderError, match="SELFHOSTED_LLM_BASE_URL"):
        sl.build(model="Qwen/Qwen3-8B")
