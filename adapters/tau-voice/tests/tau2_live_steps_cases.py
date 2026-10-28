"""Reference ticks from tau2's ``DiscreteTimeOpenAILiveAdapter`` (OpenAI Live).

Scripted provider events — 24 kHz agent audio (speech and WebRTC silence),
transcript deltas placed on the audio timeline, the caller's input transcript
— are fed to tau2's Live adapter through a fake provider, and every tick it
produces is recorded, with the caller audio it would send. ``test_agent_steps``
runs :class:`tau_voice.agent_steps.LiveAgentStepper` on the same script.
Regenerate inside the sidecar image (Live needs tau2's ``[live]`` extra)::

    docker run --rm -v "$PWD/adapters/tau-voice/tests:/cases" tau-voice-runtime:<tag> \\
        bash -c "pip install -q aiortc==1.15.0 && python /cases/tau2_live_steps_cases.py \\
        /cases/fixtures/tau2_live_steps.json"
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from typing import Any

PROVIDER_RATE = 24_000
TICKS = 10


def pcm(ms: int, tone: float, phase: int = 0) -> bytes:
    """24 kHz PCM16: a tone (speech) or zeros (WebRTC silence)."""
    n = PROVIDER_RATE * ms // 1000
    if not tone:
        return bytes(2 * n)
    return b"".join(
        struct.pack(
            "<h", int(8000 * math.sin(2 * math.pi * tone * (i + phase) / PROVIDER_RATE))
        )
        for i in range(n)
    )


def caller_audio(tick: int) -> bytes:
    """One step of caller μ-law (8 kHz): a tone on even ticks, silence on odd."""
    if tick % 2:
        return bytes([0xFF]) * 1600
    import audioop

    lin = b"".join(
        struct.pack(
            "<h", int(6000 * math.sin(2 * math.pi * 300 * (i + tick * 1600) / 8000))
        )
        for i in range(1600)
    )
    return audioop.lin2ulaw(lin, 2)


def script() -> list[list[dict[str, Any]]]:
    """Events per tick; transcript positions are the PCM bytes received before."""
    a = lambda ms, tone=440.0: {"kind": "audio", "ms": ms, "tone": tone}  # noqa: E731
    ticks: list[list[dict[str, Any]]] = [
        [a(200), {"kind": "transcript", "delta": "Hello, "}],
        [a(100), {"kind": "transcript", "delta": "how can "}, a(100)],
        [
            a(400),
            {"kind": "transcript", "delta": "I help "},
            {"kind": "input", "delta": "um"},
        ],
        [{"kind": "transcript", "delta": "you today?"}],
        [a(200, 0)],
        [a(200, 0), {"kind": "input", "delta": "I need"}],
        [],
        [a(300), {"kind": "transcript", "delta": "Sure."}],
        [a(100, 0)],
        [],
    ]
    received = 0
    for events in ticks:
        for event in events:
            if event["kind"] == "audio":
                received += len(pcm(event["ms"], event["tone"]))
            elif event["kind"] == "transcript":
                event["position"] = received
    return ticks


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_tau2() -> dict[str, Any]:
    import asyncio
    import base64

    from tau2.voice.audio_native.openai.discrete_time_adapter import (
        DiscreteTimeOpenAIAdapter,
    )
    from tau2.voice.audio_native.openai.events import AudioDeltaEvent
    from tau2.voice.audio_native.openai.events import TimeoutEvent
    from tau2.voice.audio_native.openai.live_adapter import (
        DiscreteTimeOpenAILiveAdapter,
    )
    from tau2.voice.audio_native.openai.live_config import LiveConfig
    from tau2.voice.audio_native.openai.live_provider import LiveInputTranscriptDelta
    from tau2.voice.audio_native.openai.live_provider import LiveTranscriptDelta

    steps = script()
    sent: list[bytes] = []

    class FakeProvider:
        model = "gpt-live-1"
        _closing = False

        def __init__(self) -> None:
            self.i = 0

        async def send_audio(self, data: bytes) -> None:
            sent.append(data)

        async def send_tool_result(self, *a: Any, **k: Any) -> None:
            pass

        async def receive_events_for_duration(self, duration: float) -> list[Any]:
            events: list[Any] = []
            for e in steps[self.i] if self.i < len(steps) else []:
                if e["kind"] == "audio":
                    events.append(
                        AudioDeltaEvent(
                            type="response.output_audio.delta",
                            delta=base64.b64encode(pcm(e["ms"], e["tone"])).decode(),
                        )
                    )
                elif e["kind"] == "transcript":
                    events.append(
                        LiveTranscriptDelta(
                            type="response.output_audio_transcript.delta",
                            delta=e["delta"],
                            audio_position_bytes=e["position"],
                            start_ms=0,
                            end_ms=0,
                        )
                    )
                else:
                    events.append(
                        LiveInputTranscriptDelta(
                            type="session.input_transcript.delta",
                            delta=e["delta"],
                            start_ms=0,
                            end_ms=0,
                        )
                    )
            self.i += 1
            return events or [TimeoutEvent(type="timeout")]

    loop = asyncio.new_event_loop()

    def run_tick(self: Any, user_audio: bytes, tick_number: int | None = None) -> Any:
        return loop.run_until_complete(
            self._async_run_tick(user_audio, tick_number or 0)
        )

    import os

    os.environ.setdefault("OPENAI_API_KEY", "offline")  # the fake provider replaces it
    # The base run_tick hops to a background loop; run the same coroutine here.
    DiscreteTimeOpenAIAdapter.run_tick = run_tick
    adapter = DiscreteTimeOpenAILiveAdapter(
        tick_duration_ms=200, model="gpt-live-1", config=LiveConfig(backend_model="x")
    )
    adapter._provider = FakeProvider()
    adapter.realtime_pacing = False  # wall-clock scheduling; the ticks are scripted
    out = []
    for n in range(TICKS):
        before = len(sent)
        r = adapter.run_tick(caller_audio(n), n)
        played = r.get_played_agent_audio()
        out.append(
            {
                "sent": digest(b"".join(sent[before:])),
                "sent_len": sum(len(s) for s in sent[before:]),
                "played_sha256": digest(played),
                "played_len": len(played),
                "raw_len": r.agent_audio_bytes,
                "transcript": r.proportional_transcript,
                "contains_speech": r.contains_speech,
                "vad_events": list(r.vad_events),
                "was_truncated": r.was_truncated,
                "truncated_audio_bytes": r.truncated_audio_bytes,
                "utterances": r.item_ids,
            }
        )
    return {"source": "tau2-bench@2174a60 DiscreteTimeOpenAILiveAdapter", "steps": out}


def run_ours() -> list[dict[str, Any]]:
    from harbor_voice.transport.stream import InputSpeech
    from harbor_voice.transport.stream import OutputAudio
    from harbor_voice.transport.stream import OutputTranscript
    from tau_voice.agent_steps import LiveAgentStepper

    stepper = LiveAgentStepper(bytes_per_step=1600, bytes_per_second=8000)
    out = []
    for n, events in enumerate(script() + [[]] * (TICKS - len(script()))):
        stepper.begin()
        audio = caller_audio(n)
        converted = stepper.convert_input(audio)
        for e in events:
            if e["kind"] == "audio":
                stepper.feed(OutputAudio(pcm(e["ms"], e["tone"]), None, 0.0))
            elif e["kind"] == "transcript":
                stepper.feed(OutputTranscript(None, e["delta"], 0.0, e["position"]))
            else:
                stepper.feed(InputSpeech(True, None, 0.0))
        step = stepper.end(user_bytes=len(audio))
        out.append(
            {
                "sent": digest(converted),
                "sent_len": len(converted),
                "played_sha256": digest(step.played),
                "played_len": len(step.played),
                "raw_len": step.raw_bytes,
                "transcript": step.transcript,
                "contains_speech": step.contains_speech,
                "vad_events": list(step.vad),
                "was_truncated": step.was_truncated,
                "truncated_audio_bytes": step.truncated_bytes,
                "utterances": list(step.utterances),
            }
        )
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump(run_tau2(), f, indent=1)
