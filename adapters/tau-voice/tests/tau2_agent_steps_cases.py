"""Reference outputs from tau2's real ``DiscreteTimeOpenAIAdapter`` tick template.

Each scenario is a list of steps; each step a list of provider-neutral events.
They are fed to tau2's adapter through a fake provider, and what tau2 makes of
every tick is recorded, including the ``TickResult`` fields its agent keeps in
``raw_data`` for the interaction metrics. ``test_agent_steps.py`` checks
:mod:`tau_voice.agent_steps` against the result. Regenerate inside the
sidecar image::

    docker run --rm -v "$PWD/adapters/tau-voice/tests:/cases" tau-voice-runtime:<tag> \\
        python /cases/tau2_agent_steps_cases.py /cases/fixtures/tau2_agent_steps.json
"""

import asyncio
import base64
import hashlib
import json
import sys

from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeOpenAIAdapter,
)
from tau2.voice.audio_native.openai.events import AudioDeltaEvent
from tau2.voice.audio_native.openai.events import AudioTranscriptDeltaEvent
from tau2.voice.audio_native.openai.events import SpeechStartedEvent
from tau2.voice.audio_native.openai.events import SpeechStoppedEvent
from tau2.voice.audio_native.openai.events import TimeoutEvent

RAW_DATA_KEYS = (
    "vad_events",
    "proportional_transcript",
    "was_truncated",
    "truncated_audio_bytes",
    "interruption_audio_start_ms",
    "cumulative_user_audio_at_tick_start_ms",
)


def audio(n, byte, item):
    return {"kind": "audio", "pcm_hex": bytes([byte]).hex() * n, "utterance": item}


def text(item, delta):
    return {"kind": "transcript", "utterance": item, "delta": delta}


def speech(started, ms):
    return {"kind": "speech", "started": started, "audio_ms": ms}


SCENARIOS = {
    "long_utterance_buffers_across_steps": [
        [audio(1000, 1, "a"), text("a", "Hello, how can "), audio(1000, 2, "a")],
        [audio(1000, 3, "a"), text("a", "I help you today?")],
        [],
        [],
        [],
    ],
    "barge_in_truncates_and_skips_the_rest": [
        [audio(1600, 1, "a"), text("a", "Let me look that up for you right now.")],
        [audio(1600, 2, "a"), audio(800, 3, "a")],
        [audio(400, 4, "a"), speech(True, 500), audio(1200, 5, "a")],
        [
            audio(1600, 6, "a"),
            speech(False, 900),
            audio(800, 7, "b"),
            text("b", "Sure."),
        ],
        [],
    ],
    "transcript_before_audio_and_two_utterances": [
        [text("a", "One. "), text("b", "Two three.")],
        [audio(600, 1, "a"), audio(600, 2, "b")],
        [audio(600, 3, "b")],
        [],
    ],
    "caller_speech_with_no_agent_audio_truncates_nothing": [
        [speech(True, 100)],
        [speech(False, 400)],
        [audio(800, 9, "a"), text("a", "Yes?")],
        [],
    ],
}


def to_openai(e):
    if e["kind"] == "audio":
        return AudioDeltaEvent(
            type="response.output_audio.delta",
            delta=base64.b64encode(bytes.fromhex(e["pcm_hex"])).decode(),
            item_id=e["utterance"],
        )
    if e["kind"] == "transcript":
        return AudioTranscriptDeltaEvent(
            type="response.output_audio_transcript.delta",
            delta=e["delta"],
            item_id=e["utterance"],
        )
    if e["started"]:
        return SpeechStartedEvent(
            type="input_audio_buffer.speech_started", audio_start_ms=e["audio_ms"]
        )
    return SpeechStoppedEvent(
        type="input_audio_buffer.speech_stopped", audio_end_ms=e["audio_ms"]
    )


class FakeProvider:
    def __init__(self, steps):
        self.steps, self.i, self.truncations = steps, 0, []

    async def send_audio(self, data):
        pass

    async def send_tool_result(self, *a, **k):
        pass

    async def truncate_item(self, item_id, content_index, audio_end_ms):
        self.truncations.append({"utterance": item_id, "audio_end_ms": audio_end_ms})

    async def receive_events_for_duration(self, d):
        evs = (
            [to_openai(e) for e in self.steps[self.i]]
            if self.i < len(self.steps)
            else []
        )
        self.i += 1
        return evs or [TimeoutEvent(type="timeout")]


async def run(steps):
    prov = FakeProvider(steps)
    ad = DiscreteTimeOpenAIAdapter(
        tick_duration_ms=200, send_audio_instant=True, provider=prov
    )
    out = []
    silence = b"\x7f" * ad.bytes_per_tick
    for n in range(len(steps)):
        before = len(prov.truncations)
        r = await ad._async_run_tick(silence, n)
        played = r.get_played_agent_audio()
        dumped = r.model_dump(serialize_as_any=True)
        out.append(
            {
                "played_sha256": hashlib.sha256(played).hexdigest(),
                "played_len": len(played),
                "raw_len": r.agent_audio_bytes,
                "contains_speech": r.agent_audio_bytes > 0,
                "transcript": r.proportional_transcript,
                "was_truncated": r.was_truncated,
                "truncated_audio_bytes": r.truncated_audio_bytes,
                "utterances": r.item_ids,
                "truncations": prov.truncations[before:],
                "raw_data": {key: dumped[key] for key in RAW_DATA_KEYS},
            }
        )
    return out


if __name__ == "__main__":
    result = {
        "source": "tau2-bench@2174a60 DiscreteTimeOpenAIAdapter._async_run_tick",
        "bytes_per_step": 1600,
        "bytes_per_second": 8000,
        "silence_byte": "7f",
        "scenarios": {},
    }
    for name, steps in SCENARIOS.items():
        result["scenarios"][name] = {
            "steps": steps,
            "expected": asyncio.run(run(steps)),
        }
    with open(sys.argv[1], "w") as f:
        json.dump(result, f, indent=1)
