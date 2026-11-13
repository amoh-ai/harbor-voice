"""selfhosted.realtime against a fake vLLM-Omni socket. No test opens a real one."""

from __future__ import annotations

import pytest

from harbor_voice.ids import PartyId
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.openai.realtime import RealtimeSettings
from harbor_voice.providers.registry import default_registry
from harbor_voice.providers.selfhosted import realtime as sr
from harbor_voice.providers.selfhosted import register_all
from harbor_voice.transport.base import SessionConfig
from tests.unit.voice.test_xai import Connect
from tests.unit.voice.test_xai import FakeSocket

pytestmark = pytest.mark.unit

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SESSION = SessionConfig(party=PartyId("agent"))


def _transport(**kwargs) -> tuple[sr.SelfHostedRealtimeTransport, Connect]:
    connect = Connect(FakeSocket())
    return sr.build(
        model=MODEL, system_prompt="Be brief.", connect=connect, **kwargs
    ), connect


@pytest.mark.parametrize(
    ("base", "url"),
    [
        ("https://host.example/", "wss://host.example/v1/realtime"),
        ("http://localhost:8000/v1", "ws://localhost:8000/v1/realtime"),
    ],
)
async def test_it_connects_to_the_servers_realtime_route(
    monkeypatch, base, url
) -> None:
    monkeypatch.setenv("SELFHOSTED_REALTIME_BASE_URL", base)
    monkeypatch.setenv("SELFHOSTED_API_KEY", "server-key")
    transport, connect = _transport()
    await transport.open(SESSION)
    await transport.close()
    ((called, headers),) = connect.calls
    assert called == f"{url}?model={MODEL}"
    assert headers == {"Authorization": "Bearer server-key"}


async def test_a_server_without_a_key_needs_none(monkeypatch) -> None:
    monkeypatch.delenv("SELFHOSTED_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    transport, connect = _transport(url="ws://localhost:8000/v1/realtime")
    await transport.open(SESSION)
    await transport.close()
    assert connect.calls[0][1] == {"Authorization": "Bearer none"}


async def test_it_connects_without_compression() -> None:
    """Modal's web proxy closes a compressed session before session.created."""
    transport, connect = _transport(url="ws://localhost:8000/v1/realtime")
    await transport.open(SESSION)
    await transport.close()
    assert connect.options == [{"compression": None}]


async def test_without_a_url_it_fails_before_connecting(monkeypatch) -> None:
    monkeypatch.delenv("SELFHOSTED_REALTIME_BASE_URL", raising=False)
    transport, connect = _transport()
    with pytest.raises(ProviderError) as info:
        await transport.open(SESSION)
    assert info.value.kind is ProviderFailure.AUTH
    assert connect.calls == []


async def test_session_update_is_the_subset_vllm_omni_documents() -> None:
    transport, connect = _transport(url="ws://localhost:8000/v1/realtime")
    await transport.open(SESSION)
    await transport.close()
    (update,) = [e for e in connect.socket.sent if e["type"] == "session.update"]
    audio = update["session"]["audio"]
    pcm = {"type": "audio/pcm", "rate": 24_000}
    assert audio["input"]["format"] == pcm
    assert audio["output"] == {"format": pcm, "voice": "ethan"}
    assert audio["input"]["turn_detection"]["type"] == "server_vad"
    assert "transcription" not in audio["input"]
    assert "noise_reduction" not in audio["input"]
    assert update["session"]["instructions"] == "Be brief."


def test_only_pcm24k_is_accepted() -> None:
    with pytest.raises(ValueError, match="24 kHz"):
        sr.build(model=MODEL, settings=RealtimeSettings(audio_format="pcmu"))


def test_registered_with_the_other_self_hosted_providers() -> None:
    registry = default_registry()
    register_all(registry)
    assert registry.resolve(f"selfhosted.realtime/{MODEL}")[0].spec is sr.SPEC


@pytest.mark.parametrize(
    ("base", "url"),
    [
        ("wss://host.example", "wss://host.example/v1/realtime"),
        ("ws://localhost:8000/v1/", "ws://localhost:8000/v1/realtime"),
    ],
)
def test_socket_schemes_keep_their_encryption(base: str, url: str) -> None:
    assert sr.realtime_url(base) == url


@pytest.mark.parametrize("base", ["localhost:8000", "ftp://host"])
def test_a_base_without_a_web_scheme_is_refused(base: str) -> None:
    with pytest.raises(ValueError, match="SELFHOSTED_REALTIME_BASE_URL"):
        sr.realtime_url(base)
