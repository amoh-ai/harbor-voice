"""Usage: what a call consumed, in tokens and characters — never dollars."""

from __future__ import annotations

import pytest

from harbor_voice.ids import PartyId
from harbor_voice.providers.openai.llm import ChatReply
from harbor_voice.providers.openai.llm import usage_record as chat_usage
from harbor_voice.providers.openai.realtime import OpenAIRealtimeTransport
from harbor_voice.providers.openai.realtime import token_counts
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trajectory.model import merge_usage

pytestmark = pytest.mark.unit

RESPONSE_USAGE = {
    "total_tokens": 1500,
    "input_tokens": 1200,
    "output_tokens": 300,
    "input_token_details": {
        "text_tokens": 400,
        "audio_tokens": 800,
        "cached_tokens": 1000,
        "cached_tokens_details": {"text_tokens": 300, "audio_tokens": 700},
    },
    "output_token_details": {"text_tokens": 60, "audio_tokens": 240},
}


def test_a_realtime_usage_block_flattens() -> None:
    assert token_counts(RESPONSE_USAGE) == {
        "input": 1200,
        "input_cached": 1000,
        "input_text": 400,
        "input_audio": 800,
        "output": 300,
        "output_text": 60,
        "output_audio": 240,
    }


def test_the_realtime_transport_sums_every_response() -> None:
    transport = OpenAIRealtimeTransport(party=PartyId("agent"), model="gpt-realtime-x")
    for _ in range(2):
        transport._dispatch(
            {"type": "response.done", "response": {"usage": RESPONSE_USAGE}}
        )
    record = transport.usage_record()
    assert record == UsageRecord(
        component="agent",
        model="gpt-realtime-x",
        requests=2,
        input_tokens=2400,
        cached_input_tokens=2000,
        output_tokens=600,
        input_audio_tokens=1600,
        output_audio_tokens=480,
    )
    usage = transport.usage
    assert usage.tokens is not None and usage.tokens.cache_tokens == 2000
    assert usage.speech_tokens == 1600 + 480 and usage.text_tokens == 800 + 120


def test_chat_replies_sum_with_their_cache() -> None:
    replies = [
        ChatReply(
            text="hi",
            usage={
                "prompt_tokens": 2000,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 1536},
            },
        ),
        ChatReply(text="ok", usage={"prompt_tokens": 2100, "completion_tokens": 10}),
    ]
    record = chat_usage(replies, component="caller.llm", model="gpt-4.1-mini")
    assert (record.requests, record.input_tokens, record.cached_input_tokens) == (
        2,
        4100,
        1536,
    )
    assert record.output_tokens == 30


def test_attempts_merge_per_component() -> None:
    a = UsageRecord(component="agent", model="m", requests=3, input_tokens=10)
    b = UsageRecord(component="caller.tts", characters=500)
    c = UsageRecord(component="agent", model="m", requests=1, input_tokens=5)
    d = UsageRecord(component="caller.tts", characters=200)
    merged = merge_usage([a, b, c, d])
    assert merged == (
        UsageRecord(component="agent", model="m", requests=4, input_tokens=15),
        UsageRecord(component="caller.tts", characters=700),
    )


def test_atif_carries_token_totals_and_no_cost() -> None:
    trajectory = VoiceTrajectory(
        usage=(
            UsageRecord(
                component="agent",
                input_tokens=1200,
                cached_input_tokens=1000,
                output_tokens=300,
            ),
            UsageRecord(component="caller.llm", input_tokens=4100, output_tokens=30),
            UsageRecord(component="caller.tts", characters=700),
        )
    )
    metrics = trajectory.to_atif().final_metrics
    assert metrics is not None
    assert metrics.total_prompt_tokens == 5300
    assert metrics.total_completion_tokens == 330
    assert metrics.total_cached_tokens == 1000
    assert metrics.total_cost_usd is None
    assert metrics.extra is not None and len(metrics.extra["usage"]) == 3
    # Round trip: the voice payload keeps the records too.
    again = VoiceTrajectory.from_atif(trajectory.to_atif())
    assert again.usage == trajectory.usage


def test_a_call_without_usage_claims_no_count() -> None:
    metrics = VoiceTrajectory().to_atif().final_metrics
    assert metrics is not None
    assert metrics.total_prompt_tokens is None and metrics.extra is None


def test_billed_seconds_sum_across_attempts() -> None:
    """A part billed by duration (OpenAI Live's voice session) keeps seconds."""
    a = UsageRecord(component="agent", model="gpt-live-1", requests=3, seconds=61.5)
    b = UsageRecord(component="agent", model="gpt-live-1", requests=2, seconds=30.0)
    (merged,) = merge_usage([a, b])
    assert (merged.requests, merged.seconds) == (5, 91.5)
    (none,) = merge_usage([UsageRecord(component="caller.tts", characters=1)] * 2)
    assert none.seconds is None
