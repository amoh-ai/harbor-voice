"""The conformance framework.

The suite is **main** code, not test code: *two verified providers* is a
stronger claim than *thirty unverified ones*.

It exists to make attestation mean something: a capability in
:data:`~harbor_voice.capabilities._ATTESTABLE` is only legitimately listed in
``Capabilities.attested`` once a check here has verified it, and an
unattested claim cannot gate a published number.

Usage from a third party's test suite::

    from harbor_voice.conformance import run_conformance, silence_stimulus

    async def test_my_adapter_conforms():
        report = await run_conformance(
            lambda: MyTransport(...),
            SessionConfig(party=PartyId("agent")),
            stimulus=silence_stimulus(seconds=1.0, party=PartyId("caller")),
        )
        assert report.ok, report.summary()

Checks declare what they *attest* and what they *need*. A check whose needs are
unmet is **skipped**, never failed — a transport that does not claim
``send_dtmf`` is not broken for lacking it. The distinction matters because a
skipped check leaves the capability unattested rather than refuted.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import override

from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import OptionalOp
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import VoiceTransport

DEFAULT_TIMEOUT_S = 10.0


class Outcome(StrEnum):
    """What a single check concluded."""

    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"
    ERROR = "error"
    """The check itself blew up — distinct from the transport failing it."""


class ConformanceFailure(AssertionError):
    """A transport did not satisfy a check. Raised by check bodies."""


class CheckSkipped(Exception):
    """A check's preconditions are unmet. Not a failure."""


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Everything a check body is given."""

    transport: VoiceTransport
    config: SessionConfig
    clock: SessionClock
    stimulus: tuple[AudioFrame, ...] = ()
    timeout_s: float = DEFAULT_TIMEOUT_S

    def require_stimulus(self) -> tuple[AudioFrame, ...]:
        """Return the stimulus, or skip.

        Stream-level behaviour cannot be verified provider-agnostically without
        audio the provider will actually respond to, so the caller supplies it.
        Pretending otherwise would produce a green suite that proves nothing.
        """
        if not self.stimulus:
            raise CheckSkipped("no stimulus supplied; stream checks need audio")
        return self.stimulus

    def require_op(self, op: OptionalOp) -> None:
        """Skip unless the transport declares ``op``."""
        if not self.transport.capabilities.supports(op):
            raise CheckSkipped(f"transport does not declare {op.value}")


CheckBody = Callable[[CheckContext], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Check:
    """One verifiable property, and the capability claims it attests."""

    name: str
    body: CheckBody
    attests: frozenset[str] = frozenset()
    needs_session: bool = False
    description: str = ""


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one check."""

    name: str
    outcome: Outcome
    detail: str = ""
    attests: frozenset[str] = frozenset()

    @property
    def attested(self) -> frozenset[str]:
        """Claims this result actually establishes. Empty unless it passed."""
        return self.attests if self.outcome is Outcome.PASS else frozenset()

    @override
    def __repr__(self) -> str:
        tail = f" — {self.detail}" if self.detail else ""
        return f"{self.outcome.value.upper():8} {self.name}{tail}"


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    """What a transport was shown to do."""

    transport: str
    results: tuple[CheckResult, ...] = field(default=())

    def _of(self, outcome: Outcome) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.outcome is outcome)

    @property
    def passed(self) -> tuple[CheckResult, ...]:
        return self._of(Outcome.PASS)

    @property
    def failed(self) -> tuple[CheckResult, ...]:
        return self._of(Outcome.FAIL)

    @property
    def errored(self) -> tuple[CheckResult, ...]:
        return self._of(Outcome.ERROR)

    @property
    def skipped(self) -> tuple[CheckResult, ...]:
        return self._of(Outcome.SKIPPED)

    @property
    def ok(self) -> bool:
        """Whether nothing failed or errored. Skips are fine."""
        return not self.failed and not self.errored

    @property
    def attested(self) -> frozenset[str]:
        """The capability set this transport may legitimately declare.

        Feed this into ``Capabilities.attested``. Anything absent stays
        unattested and cannot gate a published number.
        """
        return (
            frozenset().union(*(r.attested for r in self.results))
            if self.results
            else frozenset()
        )

    def summary(self) -> str:
        """A readable report, suitable for an assertion message."""
        lines = [
            f"conformance: {self.transport}",
            f"  {len(self.passed)} passed, {len(self.failed)} failed, "
            f"{len(self.errored)} errored, {len(self.skipped)} skipped",
        ]
        lines.extend(f"  {r!r}" for r in self.results)
        attested = sorted(self.attested)
        lines.append(f"  attested: {attested or 'nothing'}")
        return "\n".join(lines)


# ----------------------------------------------------------------- registry

_REGISTRY: list[Check] = []


def conformance_check(
    *,
    attests: Sequence[str] = (),
    needs_session: bool = False,
    name: str | None = None,
) -> Callable[[CheckBody], CheckBody]:
    """Register a check.

    Args:
        attests: Capability names this check verifies. Validated against the
            attestable set, so a typo cannot silently attest nothing.
        needs_session: Whether the runner should open a session first.
        name: Defaults to the function name with underscores as spaces.
    """
    from harbor_voice.capabilities import _ATTESTABLE

    unknown = set(attests) - _ATTESTABLE
    if unknown:
        raise ValueError(
            f"check attests unknown capabilities {sorted(unknown)}; "
            f"attestable: {sorted(_ATTESTABLE)}"
        )

    def decorate(body: CheckBody) -> CheckBody:
        _REGISTRY.append(
            Check(
                name=name or _derive_name(body),
                body=body,
                attests=frozenset(attests),
                needs_session=needs_session,
                description=inspect.getdoc(body) or "",
            )
        )
        return body

    return decorate


def _derive_name(body: CheckBody) -> str:
    """Readable name from the function, for checks that do not pass one."""
    raw = getattr(body, "__name__", "")
    return raw.removeprefix("check_").replace("_", " ") if raw else "unnamed check"


def all_checks() -> tuple[Check, ...]:
    """Every registered check, in registration order."""
    # Imported for its registration side effects; the module body appends
    # every check to _REGISTRY at import time.
    from harbor_voice.conformance import checks as _checks  # noqa: F401  # registration side effect

    return tuple(_REGISTRY)


# ------------------------------------------------------------------- runner


async def run_conformance(
    make_transport: Callable[[], VoiceTransport],
    config: SessionConfig,
    *,
    stimulus: Sequence[AudioFrame] = (),
    timeout_s: float = DEFAULT_TIMEOUT_S,
    checks: Sequence[Check] | None = None,
) -> ConformanceReport:
    """Run the suite against a transport and report what it was shown to do.

    Args:
        make_transport: A factory, not an instance. Several checks need a
            fresh, unopened transport — testing ``open`` twice on the same
            object proves nothing about a cold start.
        config: Session config to open with.
        stimulus: Audio the provider will respond to. Without it, stream-level
            checks skip rather than pass vacuously.
        timeout_s: Per-check ceiling.
        checks: Defaults to every registered check.

    Returns:
        A report whose ``attested`` set is what the transport may declare.
    """
    selected = tuple(checks) if checks is not None else all_checks()
    probe = make_transport()
    name = type(probe).__name__
    results: list[CheckResult] = []

    for check in selected:
        transport = make_transport()
        ctx = CheckContext(
            transport=transport,
            config=config,
            clock=SessionClock(),
            stimulus=tuple(stimulus),
            timeout_s=timeout_s,
        )
        results.append(await _run_one(check, ctx))

    return ConformanceReport(transport=name, results=tuple(results))


async def _run_one(check: Check, ctx: CheckContext) -> CheckResult:
    """Run one check, converting every outcome into a result rather than a raise."""
    opened = False
    try:
        if check.needs_session:
            await ctx.transport.open(ctx.config)
            opened = True
        await check.body(ctx)
    except CheckSkipped as skip:
        return CheckResult(check.name, Outcome.SKIPPED, str(skip), check.attests)
    except ConformanceFailure as fail:
        return CheckResult(check.name, Outcome.FAIL, str(fail), check.attests)
    except Exception as exc:  # noqa: BLE001 - a broken check is not a failed transport
        return CheckResult(
            check.name,
            Outcome.ERROR,
            f"{type(exc).__name__}: {exc}",
            check.attests,
        )
    else:
        return CheckResult(check.name, Outcome.PASS, "", check.attests)
    finally:
        if opened:
            try:
                await ctx.transport.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the result
                pass
