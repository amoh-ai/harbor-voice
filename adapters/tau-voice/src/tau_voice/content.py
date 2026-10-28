"""``Tau2Content`` — what the tau-voice caller says; ``Tau2Listener`` — how
it reacts while the agent talks.

The words only. *When* the caller speaks is a turn policy's call; *how* it
sounds is the TTS and the perturbation layers.

Each time it is asked, it does what tau2's
``VoiceStreamingUserSimulator._generate_full_duplex_voice_message`` does, in
two halves:

1. **The context** comes from the sidecar's ``/user_context``: tau2 linearises
   the step record, flips roles, wraps it in the caller's system prompt and
   the role reminder, and returns the request body. Verified byte-identical to
   the request tau2's own caller builds.
2. **The call and the reply** happen here, through our OpenAI provider, so the
   timing and the record are ours: tool calls go back to the sidecar's
   environment, a stop token ends the call, text becomes the line.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tau_voice.caller import CallerState
from tau_voice.caller import Reaction
from harbor_voice.trajectory.model import UsageRecord
from tau_voice.client import Sidecar
from harbor_voice.providers.openai.llm import ChatReply
from harbor_voice.providers.openai.llm import OpenAILLM
from harbor_voice.simulator.content import SourceKind
from tau_voice.reactive import CallerReply
from tau_voice.reactive import CallerToolCall
from tau_voice.reactive import CallerToolResult
from tau_voice.reactive import HangUp
from tau_voice.reactive import Pending
from tau_voice.reactive import Say
from tau_voice.reactive import StepRecord
from tau_voice.reactive import UseTools

STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")
"""tau2's ``STOP``, ``TRANSFER`` and ``OUT_OF_SCOPE`` (``user_simulator_base``)."""


@dataclass(frozen=True, slots=True)
class Exchange:
    """One caller-LLM call, for the trajectory."""

    step: int
    messages: int
    reply: ChatReply


class Tau2Content:
    """A :class:`~tau_voice.reactive.ReactiveContentSource`."""

    def __init__(self, sidecar: Sidecar, llm: OpenAILLM) -> None:
        self._sidecar = sidecar
        self._llm = llm
        self.exchanges: list[Exchange] = []

    @property
    def kind(self) -> SourceKind:
        return SourceKind.LLM

    @property
    def is_deterministic(self) -> bool:
        # Temperature 0 is not determinism: providers do not guarantee it.
        return False

    async def respond(
        self, steps: Sequence[StepRecord], pending: Pending
    ) -> CallerReply:
        body = await self._sidecar.user_context(steps, pending)
        reply = await self._llm.complete(body)
        step = steps[-1].step if steps else 0
        self.exchanges.append(
            Exchange(step=step, messages=len(body["messages"]), reply=reply)
        )

        # Same order as tau2: tool calls first, then the stop check, then text.
        if reply.tool_calls:
            return UseTools(
                tuple(
                    CallerToolCall(id=c.id, name=c.name, arguments=c.arguments)
                    for c in reply.tool_calls
                ),
                text=reply.text,
            )
        text = reply.text or ""
        for token in STOP_TOKENS:
            if token in text:
                return HangUp(text=text, reason=token)
        # tau2 synthesises nothing for whitespace-only content.
        return Say(text if text.strip() else "")

    async def run_tools(
        self, calls: Sequence[CallerToolCall], *, step: int
    ) -> tuple[CallerToolResult, ...]:
        results: list[CallerToolResult] = []
        for call in calls:
            outcome = await self._sidecar.tool(
                side="user",
                step=step,
                call_id=call.id,
                name=call.name,
                arguments=call.arguments,
            )
            results.append(
                CallerToolResult(
                    id=call.id, content=outcome.content, error=outcome.error
                )
            )
        return tuple(results)


@dataclass(frozen=True, slots=True)
class Check:
    """One listener-reaction check, for the run summary."""

    step: int
    interrupt: bool
    backchannel: bool
    calls: int
    """Decision-model calls it made (interrupt and backchannel, as enabled)."""
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None = None
    """Of the prompt tokens, cached; ``None`` when the sidecar reports none."""


class Tau2Listener:
    """A :class:`~tau_voice.caller.Listener`: tau2's own decisions.

    The sidecar runs tau2's ``evaluate_listener_reactions`` with the callbacks
    tau2's caller would use (interruption for a persona that interrupts,
    backchannel with the LLM backchannel on), on the state it builds for
    ``/user_context``. The prompts, the decision model and the parsing are all
    tau2's; nothing of them is ported.
    """

    def __init__(self, sidecar: Sidecar) -> None:
        self._sidecar = sidecar
        self.checks: list[Check] = []
        self.model: str | None = None
        """tau2's decision model, as the sidecar names it."""

    async def react(self, state: CallerState) -> Reaction:
        pending = state.heard_buffer[-1] if state.heard_buffer else None
        if pending is not None and not pending.text:
            pending = None
        result = await self._sidecar.decide(state.records, pending)
        reaction = Reaction(
            interrupt=bool(result["should_interrupt"]),
            backchannel=bool(result["should_backchannel"]),
        )
        usages = [
            u
            for u in (result.get("interrupt_usage"), result.get("backchannel_usage"))
            if u
        ]
        self.model = result.get("model") or self.model
        cached = [
            u["cached_tokens"] for u in usages if u.get("cached_tokens") is not None
        ]
        self.checks.append(
            Check(
                step=len(state.records),
                interrupt=reaction.interrupt,
                backchannel=reaction.backchannel,
                calls=len(usages),
                prompt_tokens=sum(int(u.get("prompt_tokens") or 0) for u in usages),
                completion_tokens=sum(
                    int(u.get("completion_tokens") or 0) for u in usages
                ),
                cached_tokens=sum(int(c) for c in cached) if cached else None,
            )
        )
        return reaction

    def summary(self) -> dict[str, Any]:
        return {
            "checks": len(self.checks),
            "interrupt": sum(1 for c in self.checks if c.interrupt),
            "backchannel": sum(1 for c in self.checks if c.backchannel),
        }

    def usage_record(self) -> UsageRecord:
        """The decision model's tokens (tau2's model, gpt-4.1 by default)."""
        cached = [c.cached_tokens for c in self.checks if c.cached_tokens is not None]
        return UsageRecord(
            component="caller.decisions",
            model=self.model,
            requests=sum(c.calls for c in self.checks),
            input_tokens=sum(c.prompt_tokens for c in self.checks),
            cached_input_tokens=sum(cached) if cached else None,
            output_tokens=sum(c.completion_tokens for c in self.checks),
        )
