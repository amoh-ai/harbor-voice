"""Unit tests for the conformance suite.

The suite's own correctness matters more than most code here: if it passes a
broken transport, every capability it attests is false, and attestation stops
meaning anything. So each test below breaks exactly one property and asserts
the suite notices.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import override

import pytest

from harbor_voice.audio.clock import Clock
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import OptionalOp
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.conformance import ConformanceReport
from harbor_voice.conformance import Outcome
from harbor_voice.conformance import all_checks
from harbor_voice.conformance import run_conformance
from harbor_voice.conformance import silence_stimulus
from harbor_voice.ids import PartyId
from harbor_voice.transport import SessionConfig
from harbor_voice.transport import VoiceTransport

pytestmark = pytest.mark.unit

AGENT = PartyId("agent")
CFG = SessionConfig(party=AGENT)


def _caps(**over: object) -> Capabilities:
    base: dict[str, object] = {
        "duplex": frozenset({Duplex.FULL}),
        "pipeline": Pipeline.S2S,
        "turn_control": TurnControl.NATIVE_VAD,
        "exposes_stages": frozenset({Stage.RESPONSE_START, Stage.RESPONSE_DONE}),
    }
    return Capabilities(**(base | over))  # type: ignore[arg-type]


class EchoTransport(VoiceTransport):
    """A well-behaved transport that echoes what it is sent.

    Backed by a queue rather than a list so ``rx()`` genuinely waits for
    frames and terminates on close. A list-backed fake returns immediately
    when the buffer is momentarily empty, which races the sender.
    """

    def __init__(self, capabilities: Capabilities | None = None) -> None:
        self._caps = capabilities or _caps()
        self._open = False
        self._closed_once = False
        self._q: asyncio.Queue[AudioFrame | None] = asyncio.Queue()

    @property
    @override
    def capabilities(self) -> Capabilities:
        return self._caps

    @property
    @override
    def is_open(self) -> bool:
        return self._open

    @override
    async def open(self, config: SessionConfig) -> None:
        self._open = True
        self._q = asyncio.Queue()

    @override
    async def tx(self, frame: AudioFrame) -> None:
        self.require_open()
        echoed = frame.stamp(Clock.TX, frame.t_produced + 0.001)
        await self._q.put(echoed.stamp(Clock.RX, frame.t_produced + 0.003))

    @override
    def rx(self) -> AsyncIterator[AudioFrame]:
        async def _gen() -> AsyncIterator[AudioFrame]:
            while True:
                frame = await self._q.get()
                if frame is None:
                    return
                yield frame

        return _gen()

    @override
    async def close(self) -> None:
        self._open = False
        self._closed_once = True
        await self._q.put(None)


def _stimulus() -> tuple[AudioFrame, ...]:
    return silence_stimulus(seconds=0.1, party=AGENT)


async def _run(make: object, **kw: object) -> ConformanceReport:
    return await run_conformance(make, CFG, **kw)  # type: ignore[arg-type]


# -------------------------------------------------------------- happy path


class TestConformingTransport:
    async def test_nothing_fails(self) -> None:
        report = await _run(EchoTransport, stimulus=_stimulus(), timeout_s=0.4)
        assert report.ok, report.summary()
        assert not report.failed and not report.errored

    async def test_every_registered_check_runs(self) -> None:
        report = await _run(EchoTransport, stimulus=_stimulus(), timeout_s=0.4)
        assert len(report.results) == len(all_checks())

    async def test_passing_a_check_attests_its_claims(self) -> None:
        report = await _run(EchoTransport, stimulus=_stimulus(), timeout_s=0.4)
        assert "audio_out" in report.attested

    async def test_a_fresh_transport_is_used_per_check(self) -> None:
        """Testing open() twice on one object proves nothing about a cold start."""
        made: list[EchoTransport] = []

        def factory() -> EchoTransport:
            t = EchoTransport()
            made.append(t)
            return t

        report = await _run(factory, stimulus=_stimulus(), timeout_s=0.4)
        # one probe for the name, plus one per check
        assert len(made) == len(report.results) + 1

    async def test_summary_is_readable(self) -> None:
        report = await _run(EchoTransport, stimulus=_stimulus(), timeout_s=0.4)
        text = report.summary()
        assert "EchoTransport" in text
        assert "attested:" in text


# ------------------------------------------------- it catches broken things


class TestCatchesViolations:
    async def test_contract_violation_is_caught(self) -> None:
        """Declares send_text, never overrides it — the stub pattern."""

        def factory() -> EchoTransport:
            return EchoTransport(_caps(optional_ops=frozenset({OptionalOp.SEND_TEXT})))

        report = await _run(factory, stimulus=_stimulus(), timeout_s=0.4)
        assert not report.ok
        failed = {r.name for r in report.failed}
        assert "declaration matches implementation" in failed

    async def test_silently_noop_optional_is_caught(self) -> None:
        """An undeclared op that quietly succeeds is worse than one that raises."""

        class Sloppy(EchoTransport):
            @override
            async def send_dtmf(self, digits: str) -> None:
                return None

        report = await _run(Sloppy, stimulus=_stimulus(), timeout_s=0.4)
        assert not report.ok
        names = {r.name for r in report.failed}
        assert "undeclared optional ops refuse" in names

    async def test_tx_before_open_that_does_not_raise_is_caught(self) -> None:
        class Loose(EchoTransport):
            @override
            async def tx(self, frame: AudioFrame) -> None:
                await self._q.put(frame)

        report = await _run(Loose, stimulus=_stimulus(), timeout_s=0.4)
        assert "rejects use before open" in {r.name for r in report.failed}

    async def test_non_idempotent_close_is_caught(self) -> None:
        class Brittle(EchoTransport):
            @override
            async def close(self) -> None:
                if self._closed_once:
                    raise RuntimeError("already closed")
                await super().close()

        report = await _run(Brittle, stimulus=_stimulus(), timeout_s=0.4)
        assert "close is idempotent" in {r.name for r in report.failed}

    async def test_open_that_lies_about_readiness_is_caught(self) -> None:
        """A stub that succeeds by assigning a sentinel and defers failure."""

        class Liar(EchoTransport):
            @override
            async def open(self, config: SessionConfig) -> None:
                self._buffer = []  # never sets _open

        report = await _run(Liar, stimulus=_stimulus(), timeout_s=0.4)
        assert "open reports ready" in {r.name for r in report.failed}

    async def test_wrong_inbound_format_is_caught_and_not_attested(self) -> None:
        class Shifty(EchoTransport):
            @override
            def rx(self) -> AsyncIterator[AudioFrame]:
                async def _gen() -> AsyncIterator[AudioFrame]:
                    yield AudioFrame(
                        pcm=b"\x00" * 160,
                        seq=0,
                        party=AGENT,
                        t_produced=0.0,
                        fmt=AudioFormat(sample_rate_hz=8_000, encoding="mulaw"),
                    )

                return _gen()

        report = await _run(Shifty, stimulus=_stimulus(), timeout_s=0.4)
        assert "inbound frames match the declared format" in {
            r.name for r in report.failed
        }
        assert "audio_out" not in report.attested, "a failed check must not attest"

    async def test_out_of_order_frames_are_caught(self) -> None:
        class Jumbled(EchoTransport):
            @override
            def rx(self) -> AsyncIterator[AudioFrame]:
                async def _gen() -> AsyncIterator[AudioFrame]:
                    for seq in (3, 1):
                        yield AudioFrame(
                            pcm=b"\x00" * 960, seq=seq, party=AGENT, t_produced=0.0
                        )

                return _gen()

        report = await _run(Jumbled, stimulus=_stimulus(), timeout_s=0.4)
        assert "inbound clocks are monotonic" in {r.name for r in report.failed}

    async def test_a_broken_check_errors_rather_than_failing(self) -> None:
        """A transport that raises something unexpected is ERROR, not FAIL."""

        class Exploding(EchoTransport):
            @override
            async def open(self, config: SessionConfig) -> None:
                raise ZeroDivisionError("boom")

        report = await _run(Exploding, stimulus=_stimulus(), timeout_s=0.4)
        assert report.errored
        assert not report.ok
        assert "ZeroDivisionError" in report.errored[0].detail


# ------------------------------------------------------------------- skips


class TestSkipping:
    async def test_stream_checks_skip_without_stimulus(self) -> None:
        """A green stream check over no audio would prove nothing."""
        report = await _run(EchoTransport, timeout_s=0.4)
        skipped = {r.name for r in report.skipped}
        assert "inbound frames match the declared format" in skipped
        assert "rx terminates after close" in skipped

    async def test_skips_do_not_make_the_report_fail(self) -> None:
        report = await _run(EchoTransport, timeout_s=0.4)
        assert report.ok

    async def test_a_skipped_check_attests_nothing(self) -> None:
        """Unattested, not refuted — the distinction attestation depends on."""
        report = await _run(EchoTransport, timeout_s=0.4)
        assert "audio_out" not in report.attested

    async def test_stage_check_skips_without_a_control_stream(self) -> None:
        report = await _run(EchoTransport, stimulus=_stimulus(), timeout_s=0.4)
        stage = next(
            r for r in report.results if r.name == "emitted stages are declared"
        )
        assert stage.outcome is Outcome.SKIPPED
        assert "exposes_asr" not in report.attested


# ------------------------------------------------------------- report shape


class TestReport:
    def test_empty_report_is_ok_and_attests_nothing(self) -> None:
        r = ConformanceReport(transport="X")
        assert r.ok
        assert r.attested == frozenset()

    def test_counts_partition_the_results(self) -> None:
        async def _check() -> ConformanceReport:
            return await _run(EchoTransport, stimulus=_stimulus(), timeout_s=0.4)

        report = asyncio.run(_check())
        total = (
            len(report.passed)
            + len(report.failed)
            + len(report.errored)
            + len(report.skipped)
        )
        assert total == len(report.results)


class TestStimulusHelper:
    def test_frame_count_and_sizing(self) -> None:
        frames = silence_stimulus(seconds=0.1, party=AGENT)
        assert len(frames) == 5
        assert all(len(f.pcm) == 960 for f in frames)
        assert [f.seq for f in frames] == [0, 1, 2, 3, 4]

    def test_batch_synthesis_stamps_one_instant(self) -> None:
        """Default. The whole utterance came into existence at once, so a
        frame is never claimed to exist in the future — which would make the
        adapter's tx stamp precede it and trip the monotonic guard."""
        frames = silence_stimulus(seconds=0.1, party=AGENT, t0=1.5)
        assert {f.t_produced for f in frames} == {1.5}

    def test_paced_synthesis_advances_by_the_frame_duration(self) -> None:
        """Opt-in, for a caller that genuinely emits in real time."""
        frames = silence_stimulus(seconds=0.1, party=AGENT, paced=True)
        assert frames[1].t_produced == pytest.approx(0.020)
        assert frames[4].t_produced == pytest.approx(0.080)

    def test_respects_a_non_canonical_format(self) -> None:
        frames = silence_stimulus(
            seconds=0.04,
            party=AGENT,
            fmt=AudioFormat(sample_rate_hz=8_000, encoding="mulaw"),
        )
        assert all(len(f.pcm) == 160 for f in frames)

    def test_rejects_nonpositive_duration(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            silence_stimulus(seconds=0, party=AGENT)


class TestRegistration:
    def test_attesting_an_unknown_capability_is_refused(self) -> None:
        from harbor_voice.conformance import conformance_check

        with pytest.raises(ValueError, match="unknown capabilities"):

            @conformance_check(attests=("teleportation",))
            async def _bad(ctx: object) -> None: ...

    def test_only_attestable_names_are_claimed(self) -> None:
        from harbor_voice.capabilities import _ATTESTABLE

        for check in all_checks():
            assert check.attests <= _ATTESTABLE
