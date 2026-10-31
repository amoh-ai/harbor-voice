"""The driver reads an OpenAI Live agent's settings into tau2's --live-config."""

from __future__ import annotations

import pytest

from harbor_voice.trial import AgentSpec
from tau_voice.trial import live_config


def _spec(s2s: str, **settings: str) -> AgentSpec:
    return AgentSpec.model_validate(
        {"pipeline": "s2s", "s2s": s2s, "config": {"s2s": settings}}
    )


def test_a_live_agent_gives_tau2_its_backend_and_voice() -> None:
    assert live_config(_spec("openai.live/gpt-live-1", backend_model="gpt-6-luna")) == {
        "backend_model": "gpt-6-luna"
    }
    assert live_config(
        _spec("openai.live/gpt-live-1", backend_model="gpt-6-luna", voice="cedar")
    ) == {"backend_model": "gpt-6-luna", "voice": "cedar"}


def test_any_other_agent_is_not_live() -> None:
    assert live_config(None) is None
    assert live_config(_spec("openai.realtime/gpt-realtime-1.5")) is None


def test_live_without_a_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="backend_model"):
        live_config(_spec("openai.live/gpt-live-1"))
