"""The offline reference set — deterministic, networkless slot fillers.

Deliberately dumb. They are not a product; they are the control condition, the
nop baseline, and what lets the whole pipeline be tested without
a network or a credential.

Every one of them is reproducible by construction, which is what makes them
usable as the *fixed* half of an ablation: vary the real STT against this LLM,
or the real LLM against this STT, and only one thing moved.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from itertools import takewhile
from typing import Any

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import CANONICAL_FRAME_MS
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.capabilities import ProviderCapability
from harbor_voice.providers.protocols import Transcript
from harbor_voice.providers.tools import ChatRound
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolSpec


@dataclass(slots=True)
class ScriptedSTT:
    """Returns a fixed transcript per turn, ignoring the audio.

    Deterministic by construction, which is what makes it usable as the fixed
    half of an ablation: vary the real STT against this LLM, or vary the LLM
    against this STT, and only one thing moved.
    """

    turns: Sequence[str] = ()
    emit_partials: bool = False
    name: str = "scripted-stt"
    _index: int = field(default=0, init=False)

    async def transcribe(
        self, frames: Sequence[AudioFrame]
    ) -> AsyncIterator[Transcript]:
        duration = sum(f.duration_s for f in frames)
        text = self.turns[self._index] if self._index < len(self.turns) else ""
        self._index += 1

        if self.emit_partials and text:
            words = text.split()
            for cut in range(1, len(words)):
                yield Transcript(
                    text=" ".join(words[:cut]),
                    is_final=False,
                    audio_duration_s=duration,
                )
        yield Transcript(text=text, is_final=True, audio_duration_s=duration)


@dataclass(slots=True)
class EchoLLM:
    """Echoes what it heard, optionally with a prefix.

    The nop baseline: an agent that does nothing useful should
    score near zero. If it scores 0.4, the metric has a free floor and the
    benchmark is measuring something other than capability.
    """

    prefix: str = ""
    chunk_words: int = 3
    name: str = "echo-llm"

    async def respond(self, prompt: str, history: Sequence[str]) -> AsyncIterator[str]:
        text = f"{self.prefix}{prompt}".strip()
        if not text:
            return
        words = text.split()
        for start in range(0, len(words), self.chunk_words):
            piece = " ".join(words[start : start + self.chunk_words])
            # Carry the separator, because fragments concatenate. An offline
            # provider whose output shape differs from a real one is how a
            # space-join bug survives a full test suite.
            yield piece if start == 0 else f" {piece}"

    async def chat(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSpec]
    ) -> ChatRound:
        """The conversation shape: echo the last thing the user said."""
        del tools
        heard = next(
            (m.get("content") or "" for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        return ChatRound(text=f"{self.prefix}{heard}".strip())


@dataclass(slots=True)
class ScriptedLLM:
    """Returns fixed replies in order. The oracle's half of the pair."""

    replies: Sequence[str] = ()
    name: str = "scripted-llm"
    _index: int = field(default=0, init=False)

    async def respond(self, prompt: str, history: Sequence[str]) -> AsyncIterator[str]:
        reply = self.replies[self._index] if self._index < len(self.replies) else ""
        self._index += 1
        if reply:
            yield reply


@dataclass(slots=True)
class ToolCallingEchoLLM:
    """Calls one named tool, then reports what it got back.

    Deterministic by construction, like everything else in this module, and
    it exists so the whole tool path -- declaration, invocation, L6, the
    result round trip, recording -- is exercised offline. A path that can
    only be tested against a live model is a path that is tested rarely.

    Implements :class:`~harbor_voice.providers.tools.ToolCallingLLM`, so
    :class:`~harbor_voice.cascade.transport.CascadeTransport` routes to it
    exactly as it would to a real provider.
    """

    tool: str
    arguments: dict[str, object] = field(default_factory=dict)
    prefix: str = "the tool said: "
    rounds: int = 1
    """How many times to call before answering. >1 exercises the loop cap."""

    name: str = "offline.tool-echo-llm"

    async def respond(self, prompt: str, history: Sequence[str]) -> AsyncIterator[str]:
        """The plain path, for a task that configured no tools."""
        del history
        yield prompt

    async def respond_with_tools(
        self,
        prompt: str,
        history: Sequence[str],
        *,
        router: object,
    ) -> AsyncIterator[str]:
        del prompt, history
        from harbor_voice.providers.tools import ToolRouter

        if not isinstance(router, ToolRouter):  # pragma: no cover - typing guard
            raise TypeError("respond_with_tools needs a ToolRouter")

        last: object = None
        for turn in range(self.rounds):
            result = await router.invoke(
                ToolInvocation(
                    tool_use_id=f"call-{turn}",
                    name=self.tool,
                    arguments=dict(self.arguments),
                )
            )
            last = result.content
        for index, piece in enumerate(f"{self.prefix}{last}".split()):
            yield piece if index == 0 else f" {piece}"

    async def chat(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSpec]
    ) -> ChatRound:
        """The conversation shape: ``rounds`` calls since the user last spoke,
        then what the last one returned."""
        del tools
        since_user = list(takewhile(lambda m: m["role"] != "user", reversed(messages)))
        made = sum(1 for m in since_user if m.get("tool_calls"))
        if made < self.rounds:
            ran = sum(1 for m in messages if m["role"] == "tool")
            return ChatRound(
                calls=(
                    ToolInvocation(
                        tool_use_id=f"call-{ran}",
                        name=self.tool,
                        arguments=dict(self.arguments),
                    ),
                )
            )
        last = next((m["content"] for m in since_user if m["role"] == "tool"), None)
        return ChatRound(text=f"{self.prefix}{last}")


@dataclass(slots=True)
class ToneTTS:
    """Synthesises a tone whose length is proportional to the text.

    Deliberately not silence. A silent reference signal makes every SNR
    measurement downstream undefined, so a "TTS" that emits zeros produces a
    pipeline that looks like it works and measures nothing. The tone carries
    no speech, so nothing should judge its *content* — but it has energy, so
    everything can measure its *channel*.
    """

    ms_per_char: float = 20.0
    audio_format: AudioFormat = CANONICAL
    hz: float = 220.0
    amplitude: float = 0.3
    name: str = "tone-tts"

    async def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        if not text:
            return
        import math

        total_ms = max(CANONICAL_FRAME_MS, round(len(text) * self.ms_per_char))
        n_frames = max(1, round(total_ms / CANONICAL_FRAME_MS))
        rate = self.audio_format.sample_rate_hz
        per_frame = rate * CANONICAL_FRAME_MS // 1000
        phase = 0.0
        step = 2.0 * math.pi * self.hz / rate
        peak = 32767

        for _ in range(n_frames):
            out = bytearray()
            for _sample in range(per_frame):
                value = int(self.amplitude * peak * math.sin(phase))
                out += int(value).to_bytes(2, "little", signed=True)
                phase += step
            yield bytes(out)


# ------------------------------------------------------------- registrations

SCRIPTED_STT_SPEC = ProviderSpec(
    name="offline.scripted-stt",
    modality=Modality.STT,
    capabilities=ProviderCapabilities(
        modality=Modality.STT,
        streams=True,
        emits_partials=True,
        attested=frozenset(
            {ProviderCapability.STREAMS, ProviderCapability.EMITS_PARTIALS}
        ),
    ),
    default_model="fixed",
)
"""Attested outright: there is no network between the claim and the behaviour,
so a conformance run would be checking this file against itself."""

ECHO_LLM_SPEC = ProviderSpec(
    name="offline.echo-llm",
    modality=Modality.LLM,
    capabilities=ProviderCapabilities(
        modality=Modality.LLM,
        streams=True,
        attested=frozenset({ProviderCapability.STREAMS}),
    ),
    default_model="echo",
)

SCRIPTED_LLM_SPEC = ProviderSpec(
    name="offline.scripted-llm",
    modality=Modality.LLM,
    capabilities=ProviderCapabilities(
        modality=Modality.LLM,
        streams=True,
        attested=frozenset({ProviderCapability.STREAMS}),
    ),
    default_model="fixed",
)

TONE_TTS_SPEC = ProviderSpec(
    name="offline.tone-tts",
    modality=Modality.TTS,
    capabilities=ProviderCapabilities(
        modality=Modality.TTS,
        streams=True,
        attested=frozenset({ProviderCapability.STREAMS}),
    ),
    default_model="tone",
)


def _build_scripted_stt(
    *,
    model: str,
    turns: Sequence[str] = (),
    emit_partials: bool = False,
) -> ScriptedSTT:
    """Build from config. The "model" of a scripted STT is its script."""
    del model
    return ScriptedSTT(turns=tuple(turns), emit_partials=emit_partials)


def _build_echo_llm(*, model: str, prefix: str = "") -> EchoLLM:
    del model
    return EchoLLM(prefix=prefix)


TOOL_ECHO_LLM_SPEC = ProviderSpec(
    name="offline.tool-echo-llm",
    modality=Modality.LLM,
    capabilities=ProviderCapabilities(
        modality=Modality.LLM,
        streams=True,
        attested=frozenset({ProviderCapability.STREAMS}),
    ),
    default_model="tool-echo",
)


def _build_tool_echo_llm(
    *,
    model: str,
    tool: str = "noop",
    arguments: dict[str, object] | None = None,
    prefix: str = "the tool said: ",
    rounds: int = 1,
) -> ToolCallingEchoLLM:
    del model
    return ToolCallingEchoLLM(
        tool=tool, arguments=dict(arguments or {}), prefix=prefix, rounds=rounds
    )


def _build_scripted_llm(*, model: str, replies: Sequence[str] = ()) -> ScriptedLLM:
    del model
    return ScriptedLLM(replies=tuple(replies))


def _build_tone_tts(*, model: str, hz: float = 220.0) -> ToneTTS:
    del model
    return ToneTTS(hz=hz)


REGISTRATIONS: tuple[tuple[ProviderSpec, Callable[..., object]], ...] = (
    (SCRIPTED_STT_SPEC, _build_scripted_stt),
    (ECHO_LLM_SPEC, _build_echo_llm),
    (TOOL_ECHO_LLM_SPEC, _build_tool_echo_llm),
    (SCRIPTED_LLM_SPEC, _build_scripted_llm),
    (TONE_TTS_SPEC, _build_tone_tts),
)
"""What :func:`~harbor_voice.providers.registry.default_registry` installs."""
