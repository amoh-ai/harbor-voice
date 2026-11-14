"""L6 — what the tools do.

The layer that is not about audio at all.

Full-Duplex-Bench v3's `latency_injector.py` is the only prior art, and harbor
already owns the MCP sidecar, so this layer comes nearly free.

Every other layer degrades the *call*. L6 degrades the world the agent is
acting on: the booking system is slow, the lookup fails, the backend is
down for forty seconds. It lands on the tool boundary, not the wire.

### Why this belongs in a voice benchmark specifically

A text agent waiting four seconds for a tool shows a spinner. A voice agent
waiting four seconds is silent, and silence in a conversation is not
neutral — the caller repeats themselves, interrupts, or hangs up. Tool latency
is a conversational perturbation that happens to arrive through an API, and
it is invisible to any harness that measures tools and speech separately.

That also makes the paired metric obvious:
``timing.response_latency_s`` already declares ``sensitive_to=[CHANNEL,
WORLD]``, so a slow backend moves a number that was already being reported.

### Three failures, deliberately distinct

``tool_latency``
    It answers, slowly. The agent's problem is what to say meanwhile.
``tool_error``
    It answers with a failure. The agent's problem is recovery, and whether
    it tells the caller the truth about what went wrong.
``backend_outage``
    It stops answering for a window. Every tool, not one — which is what
    distinguishes an outage from a flaky call and is the case agents handle
    worst, because retrying is exactly the wrong move.

Collapsing these into one "tool fails" knob would make the three recoveries
indistinguishable in the results, and they are the interesting part.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Any
from typing import override

from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import LayerKey
from harbor_voice.perturbation.base import RunningMean
from harbor_voice.perturbation.base import SeedSet

_WORLD_STREAM = "policy"


class WorldError(ValueError):
    """An L6 layer was configured in a way that cannot produce a measurement."""


class ToolFailure(StrEnum):
    """What a failing tool call looks like to the agent.

    The class matters: an agent that retries a ``timeout`` is behaving well
    and an agent that retries a ``not_found`` is looping. A single "error"
    outcome cannot tell those apart.
    """

    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    NOT_FOUND = "not_found"
    INVALID = "invalid"

    @property
    def is_retryable(self) -> bool:
        """Whether retrying is the correct response."""
        return self in (
            ToolFailure.TIMEOUT,
            ToolFailure.UNAVAILABLE,
            ToolFailure.RATE_LIMITED,
        )


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """A tool's response time, as a percentile pair.

    p50 and p99 rather than a mean and a spread, because that is how anyone
    who runs a service talks about it, and because the p99 is the whole
    point: the typical call is fine and the occasional one strands the agent
    mid-sentence. FDB-v3's injector takes the same shape.
    """

    p50_ms: float
    p99_ms: float

    def __post_init__(self) -> None:
        if self.p50_ms < 0:
            raise WorldError(f"p50_ms cannot be negative: {self.p50_ms}")
        if self.p99_ms < self.p50_ms:
            raise WorldError(
                f"p99_ms ({self.p99_ms}) is below p50_ms ({self.p50_ms}); "
                f"a 99th percentile under the median is not a distribution"
            )

    def draw(self, rng: Any) -> float:
        """One latency in milliseconds.

        Log-normal fitted through the two percentiles: a real service's
        response time is right-skewed, and sampling uniformly between p50 and
        p99 would make the tail as likely as the typical case.
        """
        import math

        if self.p99_ms == self.p50_ms:
            return self.p50_ms
        # z(0.99) = 2.326. sigma follows from the ratio of the two quantiles.
        sigma = math.log(self.p99_ms / max(self.p50_ms, 1e-9)) / 2.326
        return float(self.p50_ms * math.exp(sigma * float(rng.normal(0.0, 1.0))))


@dataclass(frozen=True, slots=True)
class Outage:
    """A window in which every tool is down."""

    start_s: float
    duration_s: float
    failure: ToolFailure = ToolFailure.UNAVAILABLE

    def __post_init__(self) -> None:
        if self.start_s < 0:
            raise WorldError(f"start_s cannot be negative: {self.start_s}")
        if self.duration_s <= 0:
            raise WorldError(f"duration_s must be positive: {self.duration_s}")

    def covers(self, t: float) -> bool:
        return self.start_s <= t < self.start_s + self.duration_s


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """What L6 decided should happen to one tool call."""

    tool: str
    delay_ms: float = 0.0
    failure: ToolFailure | None = None
    reason: str = ""

    @property
    def failed(self) -> bool:
        return self.failure is not None

    @property
    def is_clean(self) -> bool:
        """Whether L6 left this call alone entirely."""
        return self.failure is None and self.delay_ms == 0.0


@dataclass(slots=True)
class WorldPlan:
    """The active L6 layers — latency, errors and outages on the tool boundary.

    Beside the signal chain, like L1 and L5: it transforms no audio. The
    trial consults it before dispatching a tool call and applies whatever it
    returns.

    Args:
        latency: Per-tool profiles. ``"*"`` applies to any tool without an
            entry of its own, so a task can slow one lookup without
            enumerating fifty-nine tools.
        error_rate: Chance a call fails, per call.
        failures: Which failure classes to draw from.
        outages: Windows in which everything is down.
    """

    latency: dict[str, LatencyProfile] = field(default_factory=dict)
    error_rate: float = 0.0
    failures: tuple[ToolFailure, ...] = (ToolFailure.UNAVAILABLE,)
    outages: tuple[Outage, ...] = ()
    seeds: SeedSet = field(default_factory=lambda: SeedSet.derive(0))

    _rng: Any = field(default=None, init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)
    _delayed: int = field(default=0, init=False, repr=False)
    _failed: int = field(default=0, init=False, repr=False)
    _outaged: int = field(default=0, init=False, repr=False)
    _observed: RunningMean = field(default_factory=RunningMean, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.error_rate <= 1.0:
            raise WorldError(f"error_rate out of range: {self.error_rate}")
        if self.error_rate > 0 and not self.failures:
            raise WorldError(
                "error_rate is non-zero but no failure classes were given; "
                "there would be nothing to fail with"
            )
        self.reset(self.seeds)

    @property
    def is_empty(self) -> bool:
        return not self.latency and self.error_rate == 0.0 and not self.outages

    @property
    def active_layers(self) -> frozenset[LayerKey]:
        """What a task has switched on. Feeds metric applicability."""
        keys: set[LayerKey] = set()
        if self.latency:
            keys.add(LayerKey(layer=Layer.WORLD, name="tool_latency"))
        if self.error_rate > 0.0:
            keys.add(LayerKey(layer=Layer.WORLD, name="tool_error"))
        if self.outages:
            keys.add(LayerKey(layer=Layer.WORLD, name="backend_outage"))
        return frozenset(keys)

    def reset(self, seeds: SeedSet | None = None) -> None:
        """Re-seed and clear per-run counters."""
        if seeds is not None:
            self.seeds = seeds
        self._rng = self.seeds.rng(_WORLD_STREAM)
        self._calls = 0
        self._delayed = 0
        self._failed = 0
        self._outaged = 0
        self._observed = RunningMean()

    # -------------------------------------------------------------- running

    def before_call(self, tool: str, *, at_s: float = 0.0) -> ToolOutcome:
        """Decide what happens to one tool call.

        An outage wins over everything: during one, a tool does not answer
        slowly, it does not answer. Returning a latency as well would model a
        backend that is simultaneously down and responsive.
        """
        self._calls += 1

        for outage in self.outages:
            if outage.covers(at_s):
                self._outaged += 1
                self._failed += 1
                return ToolOutcome(
                    tool=tool,
                    failure=outage.failure,
                    reason=(
                        f"backend outage from {outage.start_s:.1f}s for "
                        f"{outage.duration_s:.1f}s"
                    ),
                )

        if self.error_rate and float(self._rng.random()) < self.error_rate:
            self._failed += 1
            pick = self.failures[int(self._rng.integers(len(self.failures)))]
            return ToolOutcome(tool=tool, failure=pick, reason=f"injected {pick.value}")

        profile = self.latency.get(tool) or self.latency.get("*")
        if profile is None:
            return ToolOutcome(tool=tool)
        delay = profile.draw(self._rng)
        self._delayed += 1
        self._observed.add(delay)
        return ToolOutcome(tool=tool, delay_ms=delay, reason="injected latency")

    # --------------------------------------------------------------- record

    def params(self) -> dict[str, Any]:
        return {
            "latency": {
                tool: {"p50_ms": p.p50_ms, "p99_ms": p.p99_ms}
                for tool, p in sorted(self.latency.items())
            },
            "error_rate": self.error_rate,
            "failures": [f.value for f in self.failures],
            "outages": [
                {
                    "start_s": o.start_s,
                    "duration_s": o.duration_s,
                    "failure": o.failure.value,
                }
                for o in self.outages
            ],
        }

    def achieved(self) -> dict[str, float]:
        """What the agent actually met.

        A 5% error rate over eleven tool calls may inject nothing at all. The
        run would then be a clean run wearing a degraded profile's name, and
        the observed rate is the only thing that says so.
        """
        if not self._calls:
            return {}
        record = {
            "calls": float(self._calls),
            "delayed": float(self._delayed),
            "failed": float(self._failed),
            "outaged": float(self._outaged),
            "observed_error_rate": self._failed / self._calls,
        }
        record.update(self._observed.as_record("latency_ms"))
        return record

    def as_record(self) -> dict[str, dict[str, Any]]:
        """The L6 half of the written plan."""
        if self.is_empty:
            return {}
        params = self.params()
        achieved = self.achieved()
        out: dict[str, dict[str, Any]] = {}
        for key in sorted(self.active_layers, key=str):
            out[str(key)] = {"params": params, "achieved": achieved}
        return out

    @override
    def __repr__(self) -> str:
        axes = ", ".join(sorted(str(k) for k in self.active_layers)) or "none"
        return f"WorldPlan({axes})"
