"""Tool calling — the seam L6 acts on.

An agent that can only talk is not the agent anyone deploys. Every EVA
scenario is tool-calling, and until something here can execute one, L6 is
inert: :meth:`~harbor_voice.perturbation.world.WorldPlan.before_call` has no
calls to intercept.

### One executor, two loops

The two pipelines differ in who drives, and pretending otherwise would mean
fighting one of them:

``cascade``
    Bedrock ``ConverseStream`` stops with ``stopReason == "tool_use"`` and
    waits for a new message carrying the result. The loop is ours.
``s2s``
    Nova Sonic declares tools once at ``promptStart`` and emits ``toolUse``
    in the same stream as the audio, continuing to speak while the tool runs.
    The loop is the provider's; we only answer.

What they share is this module: a :class:`ToolSpec` to declare with, a
:class:`ToolInvocation` to receive, a :class:`ToolResult` to send back, and
one :class:`ToolRouter` in front of the task's executor. L6 and the
trajectory attach to the router, so they attach once rather than per
pipeline.

### Silence is not a failure mode

AWS's documentation is unusually direct about this: *"Nova 2 Sonic expects a
toolResult event after every toolUse event it sends. If your application
fails to respond, even when an error occurs, the model enters a waiting
state, causing unresponsive behavior."*

So an injected L6 failure still returns a result — one carrying the error
— and never nothing. Dropping the reply would not model a broken backend; it
would hang the agent, and every timing number from that run would be about
our injector rather than about the agent. :meth:`ToolRouter.invoke` has no
path that returns ``None``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from harbor_voice.perturbation.world import ToolFailure
from harbor_voice.perturbation.world import WorldPlan


class ToolError(ValueError):
    """A tool was declared or invoked in a way that cannot be honoured."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the agent may call.

    ``input_schema`` is JSON Schema. Both providers take the same shape under
    different keys, so the wire formats are built here rather than in each
    transport — one place to be wrong, and one place to fix.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ToolError("a tool needs a name")
        if not self.description.strip():
            raise ToolError(
                f"tool {self.name!r} has no description; both providers select "
                f"tools by their description, so an empty one makes the tool "
                f"present but effectively uncallable"
            )

    def as_bedrock(self) -> dict[str, Any]:
        """The `toolConfig.tools[]` entry for Converse / ConverseStream."""
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {"json": self.input_schema or {"type": "object"}},
            }
        }

    def as_nova(self) -> dict[str, Any]:
        """The `promptStart.toolConfiguration.tools[]` entry for Nova Sonic.

        ``inputSchema.json`` is a serialized JSON string here, not the
        object Bedrock's Converse takes. AWS's own two pages disagree: the
        tool-configuration guide prints it as an object, while the
        input-events protocol reference types it ``"json": "{}"``. The
        protocol reference is the one the service enforces — sending an
        object gets ``ValidationException: Unable to parse input chunk``,
        with nothing to say which field was wrong.

        Keeping this separate from :meth:`as_bedrock` rather than sharing a
        helper is what made the difference tractable. Two vendors, two wire
        formats, even when they look alike.
        """
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {
                    "json": json.dumps(self.input_schema or {"type": "object"})
                },
            }
        }


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """What the model asked for."""

    tool_use_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, *, tool_use_id: str, name: str, raw: str) -> ToolInvocation:
        """Parse arguments that arrived as a JSON string.

        Both providers send arguments as text — Nova Sonic in ``content``,
        Bedrock as accumulated ``ToolUseBlockDelta.input`` fragments. Malformed
        JSON becomes empty arguments rather than an exception: the model asked
        for something, the call has to be answered, and an answer saying the
        arguments were unreadable is more use than a crashed turn.
        """
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        return cls(tool_use_id=tool_use_id, name=name, arguments=parsed)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What goes back to the model."""

    tool_use_id: str
    name: str
    content: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False
    latency_ms: float = 0.0
    """Total wall time: whatever L6 added, plus the tool's own."""

    injected_delay_ms: float = 0.0
    """The part of :attr:`latency_ms` that L6 added.

    Separate from the total because a trajectory has to be able to tell a
    slow backend from a backend we made slow. Folding them together would
    make every L6 latency experiment unfalsifiable after the fact.
    """

    injected: ToolFailure | None = None
    """Set when L6 produced the *failure*, rather than the task's executor."""

    @property
    def was_injected(self) -> bool:
        """Whether L6 touched this call at all, by delay or by failure."""
        return self.injected is not None or self.injected_delay_ms > 0.0

    def as_bedrock(self) -> dict[str, Any]:
        """The `toolResult` content block for a Converse user message."""
        return {
            "toolResult": {
                "toolUseId": self.tool_use_id,
                "content": [{"json": self.content}],
                "status": "error" if self.is_error else "success",
            }
        }

    def as_text(self) -> str:
        """The result as a string, for providers whose tool output is text.

        A payload of exactly ``{"text": ...}`` is sent as that text, verbatim:
        benchmarks whose tools return strings (tau2's do) must reach the model
        unchanged, not wrapped in a JSON object the original never sent.
        Anything else is JSON.
        """
        if set(self.content) == {"text"} and isinstance(self.content["text"], str):
            return self.content["text"]
        return json.dumps(self.content)

    def as_nova_content(self) -> str:
        """Nova Sonic's `toolResult.content`: a stringified JSON object."""
        return json.dumps(self.content)


@runtime_checkable
class ToolExecutor(Protocol):
    """Runs a tool. Supplied by the task, never by the harness.

    A benchmark's tools are the benchmark's, the same boundary as
    for its metrics. harbor-voice routes, perturbs and records the call; what
    a reservation lookup *does* belongs to whoever ported the benchmark.
    """

    @property
    def specs(self) -> Sequence[ToolSpec]:
        """What the agent may call."""
        ...

    async def execute(self, invocation: ToolInvocation) -> Mapping[str, Any]:
        """Run one tool and return its result payload."""
        ...


@dataclass(frozen=True, slots=True)
class ToolRecord:
    """One completed call, for the trajectory."""

    name: str
    tool_use_id: str
    arguments: dict[str, Any]
    latency_ms: float
    is_error: bool
    injected_delay_ms: float = 0.0
    injected: ToolFailure | None = None

    @property
    def injected_latency_ms(self) -> float | None:
        """Latency L6 added, or ``None`` if it added none.

        ``None`` and ``0.0`` are different answers: the first says L6 was not
        in the path, the second says it was and chose to add nothing.
        """
        return self.injected_delay_ms if self.was_injected else None

    @property
    def was_injected(self) -> bool:
        return self.injected is not None or self.injected_delay_ms > 0.0


class ToolRouter:
    """Everything between the model asking and the model being answered.

    One object so L6, timing and recording attach once rather than per
    pipeline. The transports call :meth:`invoke` and get a
    :class:`ToolResult`; they never see the world plan or the executor.

    Args:
        executor: The task's tools.
        world: L6. Defaults to an empty plan, which injects nothing.
        sleep: Injected-latency sleeper, overridable so tests do not wait.
            A real run sleeps because the agent's silence *is* the
            perturbation -- a text agent shows a spinner, a voice agent says
            nothing.
    """

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        world: WorldPlan | None = None,
        sleep: Any = None,
    ) -> None:
        self._executor = executor
        self._world = world or WorldPlan()
        self._records: list[ToolRecord] = []
        if sleep is None:
            import asyncio

            sleep = asyncio.sleep
        self._sleep = sleep

    @property
    def specs(self) -> Sequence[ToolSpec]:
        return self._executor.specs

    @property
    def records(self) -> tuple[ToolRecord, ...]:
        """Every call made this session, in order."""
        return tuple(self._records)

    def take_records(self) -> tuple[ToolRecord, ...]:
        """Drain the records, for a caller recording one turn at a time."""
        out = tuple(self._records)
        self._records.clear()
        return out

    async def invoke(
        self, invocation: ToolInvocation, *, at_s: float = 0.0
    ) -> ToolResult:
        """Run one call under L6, and always return something.

        There is no path here that returns ``None`` or raises past the
        caller. Nova Sonic waits forever for a result it was promised, and a
        cascade's conversation cannot continue without one either, so an
        unanswered call is a hung agent rather than a failed tool.
        """
        outcome = self._world.before_call(invocation.name, at_s=at_s)

        if outcome.delay_ms:
            await self._sleep(outcome.delay_ms / 1000.0)

        failure = outcome.failure
        if failure is not None:
            return self._record(
                ToolResult(
                    tool_use_id=invocation.tool_use_id,
                    name=invocation.name,
                    content={
                        "error": failure.value,
                        "message": outcome.reason or "injected failure",
                        "retryable": failure.is_retryable,
                    },
                    is_error=True,
                    latency_ms=outcome.delay_ms,
                    injected_delay_ms=outcome.delay_ms,
                    injected=failure,
                ),
                invocation,
            )

        started = time.perf_counter()
        try:
            payload = await self._executor.execute(invocation)
        except Exception as exc:  # noqa: BLE001 - a broken tool is a result
            # The task's tool raised. That is a fact about the task, not a
            # harness error, and the model still has to be told something.
            elapsed = (time.perf_counter() - started) * 1000.0
            return self._record(
                ToolResult(
                    tool_use_id=invocation.tool_use_id,
                    name=invocation.name,
                    content={"error": type(exc).__name__, "message": str(exc)},
                    is_error=True,
                    latency_ms=outcome.delay_ms + elapsed,
                    injected_delay_ms=outcome.delay_ms,
                ),
                invocation,
            )

        elapsed = (time.perf_counter() - started) * 1000.0
        return self._record(
            ToolResult(
                tool_use_id=invocation.tool_use_id,
                name=invocation.name,
                content=dict(payload),
                latency_ms=outcome.delay_ms + elapsed,
                injected_delay_ms=outcome.delay_ms,
            ),
            invocation,
        )

    def _record(self, result: ToolResult, invocation: ToolInvocation) -> ToolResult:
        self._records.append(
            ToolRecord(
                name=result.name,
                tool_use_id=result.tool_use_id,
                arguments=dict(invocation.arguments),
                latency_ms=result.latency_ms,
                is_error=result.is_error,
                injected_delay_ms=result.injected_delay_ms,
                injected=result.injected,
            )
        )
        return result


@dataclass(slots=True)
class MappingExecutor:
    """Tools as plain Python callables, run in-process.

    The shape a benchmark adapter usually wants. EVA's tools are deterministic
    functions over scenario JSON, and running them beside the harness — which
    already drives the call — saves standing up a container to hold them. A
    sidecar earns its place when tools need isolation or non-Python deps, not
    before.

    Handlers may be sync or async; a sync one is run as-is, because a
    dictionary lookup does not need a thread and the latency that matters is
    the one L6 injects.
    """

    specs: Sequence[ToolSpec]
    handlers: Mapping[str, Any]

    def __post_init__(self) -> None:
        declared = {spec.name for spec in self.specs}
        missing = declared - set(self.handlers)
        if missing:
            raise ToolError(
                f"declared but unimplemented: {sorted(missing)}. The model "
                f"would be offered a tool that cannot run, and would be "
                f"marked wrong for calling it"
            )

    async def execute(self, invocation: ToolInvocation) -> Mapping[str, Any]:
        handler = self.handlers.get(invocation.name)
        if handler is None:
            # Not raised: the model invented a tool, which is a fact about
            # the model, and the turn still needs an answer.
            return {
                "error": "unknown_tool",
                "message": f"no tool named {invocation.name!r}",
            }
        result = handler(**invocation.arguments)
        if hasattr(result, "__await__"):
            result = await result
        return result if isinstance(result, Mapping) else {"result": result}


@runtime_checkable
class ToolCallingLLM(Protocol):
    """An LLM slot that can run a tool loop — optional, and declared.

    Separate from :class:`~harbor_voice.providers.protocols.LLMProvider`,
    which keeps its three methods. The rule: a provider that cannot do
    this says so, rather than being discovered to be unable at the first
    tool call, halfway through a benchmark that is entirely tool-calling.

    It yields text fragments exactly as ``respond`` does, having handled the
    tool rounds internally. That puts the cascade and S2S on the same mental
    model — *the provider owns the loop, the harness owns the executor* —
    and keeps :class:`~harbor_voice.cascade.transport.CascadeTransport` from
    growing a second state machine.
    """

    @property
    def name(self) -> str: ...

    def respond_with_tools(
        self,
        prompt: str,
        history: Sequence[str],
        *,
        router: ToolRouter,
    ) -> Any:
        """Yield response fragments, calling tools through ``router``.

        Fragments concatenate, as in
        :meth:`~harbor_voice.providers.protocols.LLMProvider.respond`.
        """
        ...
