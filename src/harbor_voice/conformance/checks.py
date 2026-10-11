"""The checks themselves.

Grouped by what they need:

* **static** — no session, no audio. Declaration consistency and refusal
  behaviour. These always run.
* **lifecycle** — a session, no audio. Open, close, idempotency, use-after-close.
* **stream** — a session and a stimulus. Format conformance, clock
  monotonicity, drain. Skipped without audio the provider will answer, because
  a green stream check over silence proves nothing.

Every check either returns (pass), raises :class:`ConformanceFailure`, or
raises :class:`CheckSkipped`. Nothing here asserts on provider-specific
behaviour, so the suite is portable across adapters by construction.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Callable

from harbor_voice.audio.clock import Clock
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import OptionalOp
from harbor_voice.conformance.suite import CheckContext
from harbor_voice.conformance.suite import CheckSkipped
from harbor_voice.conformance.suite import ConformanceFailure
from harbor_voice.conformance.suite import conformance_check
from harbor_voice.transport.base import TransportError
from harbor_voice.transport.drain import drain_until_idle

TAIL_SILENCE_S = 0.2
"""Quiet period after which a reply is considered finished."""

_POLL_S = 0.01

# ------------------------------------------------------------------- static


@conformance_check(name="declaration matches implementation")
async def check_contract_consistency(ctx: CheckContext) -> None:
    """Declared optional ops are implemented, and vice versa.

    This catches a transport whose declared capability matrix its code does
    not honour.
    """
    problems = ctx.transport.check_contract()
    if problems:
        raise ConformanceFailure("; ".join(problems))


@conformance_check(name="undeclared optional ops refuse")
async def check_undeclared_ops_refuse(ctx: CheckContext) -> None:
    """An op the transport does not declare must raise, not no-op.

    A silent no-op is worse than a refusal: the caller believes the DTMF was
    sent, and the failure surfaces as an unexplained wrong answer much later.
    """
    caps = ctx.transport.capabilities
    failures: list[str] = []

    async def _expect_refusal(op: OptionalOp, call: Callable[[], object]) -> None:
        if caps.supports(op):
            return
        try:
            result = call()
            if inspect.isawaitable(result):
                await result
        except TransportError:
            return
        except Exception as exc:  # noqa: BLE001 - any wrong exception type is itself the finding
            failures.append(
                f"{op.value} raised {type(exc).__name__}, want TransportError"
            )
            return
        failures.append(f"{op.value} is undeclared but did not refuse")

    await _expect_refusal(OptionalOp.SEND_TEXT, lambda: ctx.transport.send_text("x"))
    await _expect_refusal(OptionalOp.SEND_DTMF, lambda: ctx.transport.send_dtmf("1"))
    await _expect_refusal(OptionalOp.CONTROL_STREAM, lambda: ctx.transport.control())

    if failures:
        raise ConformanceFailure("; ".join(failures))


@conformance_check(name="rejects use before open")
async def check_tx_before_open_refuses(ctx: CheckContext) -> None:
    """``tx`` on an unopened transport raises rather than silently dropping."""
    if ctx.transport.is_open:
        raise ConformanceFailure("transport reports is_open before open() was called")
    frame = AudioFrame(pcm=b"\x00" * 960, seq=0, party=ctx.config.party, t_produced=0.0)
    try:
        await ctx.transport.tx(frame)
    except TransportError:
        return
    raise ConformanceFailure("tx() before open() did not raise")


# ---------------------------------------------------------------- lifecycle


@conformance_check(needs_session=True, name="open reports ready")
async def check_open_reports_ready(ctx: CheckContext) -> None:
    """``open`` must not return until the session is usable.

    An ``open()`` that succeeds by assigning a sentinel ``object()`` defers the
    failure to first audio, and a caller cannot distinguish that from a working
    transport until minutes later.
    """
    if not ctx.transport.is_open:
        raise ConformanceFailure("open() returned but is_open is False")


@conformance_check(name="close is idempotent")
async def check_close_is_idempotent(ctx: CheckContext) -> None:
    """``close`` twice is legal; the context manager depends on it."""
    await ctx.transport.open(ctx.config)
    await ctx.transport.close()
    try:
        await ctx.transport.close()
    except Exception as exc:  # noqa: BLE001 - close() must tolerate anything
        raise ConformanceFailure(
            f"second close() raised {type(exc).__name__}: {exc}"
        ) from exc
    if ctx.transport.is_open:
        raise ConformanceFailure("is_open is still True after close()")


@conformance_check(name="refuses use after close")
async def check_use_after_close_refuses(ctx: CheckContext) -> None:
    """A closed session cannot be written to."""
    await ctx.transport.open(ctx.config)
    await ctx.transport.close()
    frame = AudioFrame(pcm=b"\x00" * 960, seq=0, party=ctx.config.party, t_produced=0.0)
    try:
        await ctx.transport.tx(frame)
    except TransportError:
        return
    raise ConformanceFailure("tx() after close() did not raise")


@conformance_check(name="session context manager closes")
async def check_session_closes(ctx: CheckContext) -> None:
    """``session()`` closes on the way out, including when the body raises."""
    async with ctx.transport.session(ctx.config):
        if not ctx.transport.is_open:
            raise ConformanceFailure("session() body entered with a closed transport")
    if ctx.transport.is_open:
        raise ConformanceFailure("session() exited without closing")

    class _Boom(Exception):
        pass

    try:
        async with ctx.transport.session(ctx.config):
            raise _Boom
    except _Boom:
        pass
    if ctx.transport.is_open:
        raise ConformanceFailure("session() left the transport open after an error")


# ------------------------------------------------------------------- stream


@conformance_check(
    needs_session=True,
    attests=("audio_out",),
    name="inbound frames match the declared format",
)
async def check_inbound_format(ctx: CheckContext) -> None:
    """Every received frame is in the format the transport declared.

    Attests ``audio_out``. A provider that reports one rate and emits another
    pitch-shifts silently.
    """
    frames = await _exchange(ctx)
    declared = ctx.transport.capabilities.audio_out
    for frame in frames:
        if frame.fmt != declared:
            raise ConformanceFailure(
                f"frame seq={frame.seq} is {frame.fmt}, declared {declared}"
            )


@conformance_check(needs_session=True, name="inbound clocks are monotonic")
async def check_inbound_clock_monotonic(ctx: CheckContext) -> None:
    """Sequence numbers and receive timestamps both advance.

    A frame that arrives out of order is a transport bug; one that arrives with
    a timestamp before its predecessor makes every latency built from it wrong.
    """
    frames = await _exchange(ctx)
    last_seq = -1
    last_rx: float | None = None
    for frame in frames:
        if frame.seq <= last_seq:
            raise ConformanceFailure(
                f"seq went {last_seq} -> {frame.seq}; frames must advance"
            )
        last_seq = frame.seq
        rx = frame.at(Clock.RX)
        if rx is not None:
            if last_rx is not None and rx < last_rx:
                raise ConformanceFailure(
                    f"rx clock went backwards at seq={frame.seq}: {last_rx} -> {rx}"
                )
            last_rx = rx


@conformance_check(needs_session=True, name="rx terminates after close")
async def check_rx_drains(ctx: CheckContext) -> None:
    """The inbound iterator ends rather than hanging once the session closes.

    An ``rx()`` that never completes deadlocks every caller that gathers it
    alongside ``tx``.
    """
    ctx.require_stimulus()
    stream = ctx.transport.rx()
    await ctx.transport.close()
    try:
        async with asyncio.timeout(ctx.timeout_s):
            async for _ in stream:
                pass
    except TimeoutError as exc:
        raise ConformanceFailure(
            f"rx() did not terminate within {ctx.timeout_s}s of close()"
        ) from exc


@conformance_check(
    needs_session=True,
    attests=("exposes_asr", "exposes_partials", "can_be_interrupted", "turn_control"),
    name="emitted stages are declared",
)
async def check_stages_are_declared(ctx: CheckContext) -> None:
    """Every stage mark the transport emits appears in ``exposes_stages``.

    Attests the stage-backed capability claims. Requires the control stream;
    without it there is nothing to observe and the claims stay unattested.
    """
    ctx.require_op(OptionalOp.CONTROL_STREAM)
    ctx.require_stimulus()
    raise CheckSkipped(
        "stage observation needs the recorder; the claims stay unattested"
    )


# ------------------------------------------------------------------ helpers


async def _exchange(ctx: CheckContext) -> list[AudioFrame]:
    """Send the stimulus and drain the reply to tail silence.

    Delegates to :func:`~harbor_voice.transport.drain.drain_until_idle` so the
    suite and the adapter share one definition of "the turn ended".
    """
    stimulus = ctx.require_stimulus()
    stream = ctx.transport.rx()

    async def _send() -> None:
        for frame in stimulus:
            await ctx.transport.tx(frame)

    send_task = asyncio.create_task(_send())
    try:
        result = await drain_until_idle(
            stream,
            idle_s=min(TAIL_SILENCE_S, ctx.timeout_s / 4),
            timeout_s=ctx.timeout_s,
        )
    finally:
        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await send_task

    if not result.frames:
        raise CheckSkipped(
            "provider returned no audio for this stimulus; nothing to verify"
        )
    return list(result.frames)
