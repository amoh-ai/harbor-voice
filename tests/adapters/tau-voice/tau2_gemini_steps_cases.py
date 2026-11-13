"""Reference ticks from tau2's ``DiscreteTimeGeminiAdapter`` (Gemini Live).

Scripted provider events (24 kHz agent audio tagged by utterance, transcript
deltas, and Gemini's own interruption signal) are fed to tau2's Gemini
adapter through a fake provider, and every tick it produces is recorded, with
the caller audio it would send. ``test_agent_steps`` runs
:class:`tau_voice.agent_steps.GeminiAgentStepper` on the same script.
Regenerate inside the sidecar image::

    docker run --rm -v "$PWD/tests/adapters/tau-voice:/cases" tau-voice-runtime:<tag> \\
        python /cases/tau2_gemini_steps_cases.py /cases/fixtures/tau2_gemini_steps.json
"""

from __future__ import annotations

import json
import sys
from typing import Any

from tau2_live_steps_cases import caller_audio
from tau2_live_steps_cases import digest
from tau2_live_steps_cases import pcm

TICKS = 10


def script() -> list[list[dict[str, Any]]]:
    """Events per tick. Covers a carried excess, an interruption mid-step with
    audio after it, audio of the cut utterance arriving a step later, and an
    interruption landing on audio carried from the step before."""

    def a(ms: int, item: str, tone: float = 440.0) -> dict[str, Any]:
        return {"kind": "audio", "ms": ms, "item": item, "tone": tone}

    def t(text: str, item: str) -> dict[str, Any]:
        return {"kind": "text", "text": text, "item": item}

    cut = {"kind": "interrupted"}
    return [
        [a(150, "turn-0"), t("Hello, ", "turn-0")],
        [a(300, "turn-0"), t("how can I help you today?", "turn-0")],
        [a(100, "turn-0"), cut, a(100, "turn-0")],
        [a(100, "turn-0")],
        [],
        [a(250, "turn-1", 330.0), t("Sure, one moment.", "turn-1")],
        [a(300, "turn-1", 330.0)],
        [cut],
        [a(100, "turn-2", 550.0), t("Okay.", "turn-2")],
        [],
    ]


def run_tau2() -> dict[str, Any]:
    import asyncio
    import os

    from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
    from tau2.voice.audio_native.gemini.discrete_time_adapter import (
        DiscreteTimeGeminiAdapter,
    )
    from tau2.voice.audio_native.gemini.events import GeminiAudioDeltaEvent
    from tau2.voice.audio_native.gemini.events import GeminiInterruptionEvent
    from tau2.voice.audio_native.gemini.events import GeminiTextDeltaEvent
    from tau2.voice.audio_native.gemini.events import GeminiTimeoutEvent

    steps = script()
    sent: list[bytes] = []

    class FakeProvider:
        needs_reconnection = False
        is_connected = True

        def __init__(self) -> None:
            self.i = 0

        async def send_audio(self, data: bytes) -> None:
            sent.append(data)

        async def send_tool_results(self, *a: Any, **k: Any) -> None:
            pass

        async def receive_events_for_duration(self, duration: float) -> list[Any]:
            events: list[Any] = []
            for e in steps[self.i] if self.i < len(steps) else []:
                if e["kind"] == "audio":
                    events.append(
                        GeminiAudioDeltaEvent(
                            data=pcm(e["ms"], e["tone"]), item_id=e["item"]
                        )
                    )
                elif e["kind"] == "text":
                    events.append(
                        GeminiTextDeltaEvent(text=e["text"], item_id=e["item"])
                    )
                else:
                    events.append(GeminiInterruptionEvent())
            self.i += 1
            return events or [GeminiTimeoutEvent()]

    loop = asyncio.new_event_loop()

    def run_tick(self: Any, user_audio: bytes, tick_number: int | None = None) -> Any:
        return loop.run_until_complete(
            self._async_run_tick(user_audio, tick_number or 0)
        )

    os.environ.setdefault("GEMINI_API_KEY", "offline")  # the fake provider replaces it
    # The base run_tick hops to a background loop; run the same coroutine here.
    DiscreteTimeAdapter.run_tick = run_tick
    DiscreteTimeGeminiAdapter.run_tick = run_tick
    adapter = DiscreteTimeGeminiAdapter(tick_duration_ms=200, model="gemini-live")
    adapter._provider = FakeProvider()
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
    return {"source": "tau2-bench@2174a60 DiscreteTimeGeminiAdapter", "steps": out}


def run_ours() -> list[dict[str, Any]]:
    from harbor_voice.transport.stream import OutputAudio
    from harbor_voice.transport.stream import OutputTranscript
    from harbor_voice.transport.stream import ResponseDone
    from tau_voice.agent_steps import GeminiAgentStepper

    stepper = GeminiAgentStepper(bytes_per_step=1600, bytes_per_second=8000)
    out = []
    for n, events in enumerate(script()):
        stepper.begin()
        audio = caller_audio(n)
        converted = stepper.convert_input(audio)
        for e in events:
            if e["kind"] == "audio":
                stepper.feed(OutputAudio(pcm(e["ms"], e["tone"]), e["item"], 0.0))
            elif e["kind"] == "text":
                stepper.feed(OutputTranscript(e["item"], e["text"], 0.0))
            else:
                stepper.feed(ResponseDone(None, "cancelled", False, "", 0.0))
        step = stepper.end(user_bytes=len(audio))
        out.append(
            {
                "sent": digest(converted),
                "sent_len": len(converted),
                "played_sha256": digest(step.played),
                "played_len": len(step.played),
                "raw_len": step.raw_bytes,
                "transcript": step.transcript,
                # tau2 leaves it unset for Gemini (audio presence decides).
                "contains_speech": step.speech,
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
