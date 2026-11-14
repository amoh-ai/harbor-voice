"""Unit tests for the reference cascade."""

from __future__ import annotations

import pytest

from harbor_voice.audio.stages import CASCADE_ONLY_STAGES
from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Pipeline
from harbor_voice.cascade import CascadeAdapter
from harbor_voice.cascade import CascadeTransport
from harbor_voice.providers import EchoLLM
from harbor_voice.providers import ScriptedLLM
from harbor_voice.providers import ScriptedSTT
from harbor_voice.providers import ToneTTS
from harbor_voice.conformance import run_conformance
from harbor_voice.conformance import silence_stimulus
from harbor_voice.ids import PartyId
from harbor_voice.transport import SessionConfig

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
AGENT = PartyId("agent")
CFG = SessionConfig(party=AGENT)


def _transport(
    *,
    heard: list[str] | None = None,
    llm: object | None = None,
    partials: bool = False,
) -> CascadeTransport:
    return CascadeTransport(
        stt=ScriptedSTT(turns=heard or ["hello there"], emit_partials=partials),
        llm=llm or EchoLLM(prefix="you said: "),  # type: ignore[arg-type]
        tts=ToneTTS(),
        party=AGENT,
        input_idle_s=0.05,
        emit_partials=partials,
    )


async def _adapter(transport: CascadeTransport) -> CascadeAdapter:
    a = CascadeAdapter(transport, party=AGENT, idle_s=0.1, turn_timeout_s=3.0)
    await a.open(CFG)
    return a


def _input(n_s: float = 0.1) -> tuple:
    return silence_stimulus(seconds=n_s, party=CALLER)


# ------------------------------------------------------------- the pipeline


class TestPipeline:
    async def test_a_turn_goes_all_the_way_through(self) -> None:
        t = _transport()
        a = await _adapter(t)
        turn = await a.call(_input())
        assert turn.received, "the cascade produced audio"
        assert turn.is_complete
        await a.close()

    async def test_the_heard_side_is_recorded(self) -> None:
        """Said-vs-heard needs the HEARD side kept."""
        t = _transport(heard=["I need to reschedule"])
        a = await _adapter(t)
        await a.call(_input())
        assert t.transcripts[0].text == "I need to reschedule"
        assert t.transcripts[0].is_final
        await a.close()

    async def test_the_llm_sees_what_the_stt_heard(self) -> None:
        t = _transport(heard=["book me in"], llm=EchoLLM(prefix="ok: "))
        a = await _adapter(t)
        turn = await a.call(_input())
        said = turn.stage_at(Stage.LLM_DONE)
        assert said is not None
        done = next(m for m in turn.stages if m.stage is Stage.LLM_DONE)
        assert done.meta["text"] == "ok: book me in"
        await a.close()

    async def test_consecutive_turns_advance_the_script(self) -> None:
        t = _transport(heard=["first", "second"])
        a = await _adapter(t)
        await a.call(_input())
        await a.call(_input())
        assert [tr.text for tr in t.transcripts] == ["first", "second"]
        await a.close()

    async def test_an_empty_reply_produces_no_audio(self) -> None:
        t = _transport(heard=[""], llm=ScriptedLLM(replies=[""]))
        a = CascadeAdapter(t, party=AGENT, idle_s=0.1, turn_timeout_s=0.6)
        await a.open(CFG)
        turn = await a.call(_input())
        assert turn.received == ()
        assert turn.response_latency is None, "no reply means no latency, not zero"
        await a.close()


# ------------------------------------------------------- the seven seams


class TestStages:
    async def test_every_cascade_seam_is_marked(self) -> None:
        """A cascade exposes seven seams; an S2S exposes two."""
        t = _transport()
        a = await _adapter(t)
        turn = await a.call(_input())
        emitted = {m.stage for m in turn.stages}
        expected = {
            Stage.VAD_SPEECH_START,
            Stage.VAD_SPEECH_END,
            Stage.ASR_FINAL,
            Stage.LLM_FIRST_TOKEN,
            Stage.LLM_DONE,
            Stage.TTS_FIRST_BYTE,
            Stage.TTS_DONE,
            Stage.RESPONSE_START,
            Stage.RESPONSE_DONE,
        }
        assert expected <= emitted
        await a.close()

    async def test_it_sees_seams_an_s2s_provider_cannot(self) -> None:
        t = _transport()
        a = await _adapter(t)
        turn = await a.call(_input())
        emitted = {m.stage for m in turn.stages}
        assert emitted & CASCADE_ONLY_STAGES, "the whole point of the cascade"
        await a.close()

    async def test_seams_are_marked_in_physical_order(self) -> None:
        t = _transport()
        a = await _adapter(t)
        turn = await a.call(_input())
        order = [m.stage for m in turn.stages]
        for earlier, later in (
            (Stage.VAD_SPEECH_END, Stage.ASR_FINAL),
            (Stage.ASR_FINAL, Stage.LLM_FIRST_TOKEN),
            (Stage.LLM_FIRST_TOKEN, Stage.LLM_DONE),
            (Stage.LLM_DONE, Stage.TTS_FIRST_BYTE),
            (Stage.TTS_FIRST_BYTE, Stage.TTS_DONE),
        ):
            assert order.index(earlier) < order.index(later), (
                f"{earlier} before {later}"
            )
        await a.close()

    async def test_stage_timestamps_are_non_decreasing(self) -> None:
        t = _transport()
        a = await _adapter(t)
        turn = await a.call(_input())
        times = [m.t for m in turn.stages]
        assert times == sorted(times)
        await a.close()

    async def test_partials_are_off_unless_declared(self) -> None:
        t = _transport(heard=["one two three"], partials=False)
        a = await _adapter(t)
        turn = await a.call(_input())
        assert Stage.ASR_PARTIAL not in {m.stage for m in turn.stages}
        assert not t.capabilities.exposes_partials
        await a.close()

    async def test_partials_appear_when_declared(self) -> None:
        t = _transport(heard=["one two three"], partials=True)
        a = await _adapter(t)
        turn = await a.call(_input())
        assert Stage.ASR_PARTIAL in {m.stage for m in turn.stages}
        assert t.capabilities.exposes_partials
        await a.close()

    async def test_stages_do_not_leak_between_turns(self) -> None:
        t = _transport(heard=["a", "b"])
        a = await _adapter(t)
        first = await a.call(_input())
        second = await a.call(_input())
        assert len(second.stages) <= len(first.stages) + 1
        assert all(m.t >= max(x.t for x in first.stages) for m in second.stages)
        await a.close()


# ------------------------------------------------------------ capabilities


class TestCapabilities:
    def test_declares_cascade(self) -> None:
        assert _transport().capabilities.pipeline is Pipeline.CASCADE

    def test_declares_asr_because_it_owns_the_seam(self) -> None:
        caps = _transport().capabilities
        assert caps.exposes_asr
        assert Stage.ASR_FINAL in caps.exposes_stages

    def test_partial_declaration_tracks_the_flag(self) -> None:
        assert not _transport(partials=False).capabilities.exposes_partials
        assert _transport(partials=True).capabilities.exposes_partials

    def test_adapter_pipeline_matches_transport(self) -> None:
        a = CascadeAdapter(_transport(), party=AGENT)
        assert a.pipeline is a.capabilities.pipeline

    async def test_declared_stages_are_the_ones_actually_emitted(self) -> None:
        """The claim the conformance suite exists to check."""
        t = _transport(partials=True)
        a = await _adapter(t)
        turn = await a.call(_input())
        emitted = {m.stage for m in turn.stages}
        assert emitted <= t.capabilities.exposes_stages, (
            "emitted a stage it never declared"
        )
        await a.close()


# -------------------------------------------------------------- ablation


class TestAblation:
    async def test_swapping_the_stt_changes_what_the_llm_reasons_over(self) -> None:
        """The experiment no speech-to-speech provider permits.

        Same LLM, same TTS, same audio — only recognition differs. Whatever
        moves in the output is attributable to mishearing, not to reasoning.
        """
        outputs: list[str] = []
        for heard in ("amoxicillin", "amlodipine"):
            t = _transport(heard=[heard], llm=EchoLLM(prefix="prescribing "))
            a = await _adapter(t)
            turn = await a.call(_input())
            done = next(m for m in turn.stages if m.stage is Stage.LLM_DONE)
            outputs.append(str(done.meta["text"]))
            await a.close()

        assert outputs == ["prescribing amoxicillin", "prescribing amlodipine"]

    async def test_holding_the_stt_isolates_the_llm(self) -> None:
        """The other direction: same recognition, different reasoning."""
        outputs: list[str] = []
        for llm in (EchoLLM(prefix="A: "), EchoLLM(prefix="B: ")):
            t = _transport(heard=["same input"], llm=llm)
            a = await _adapter(t)
            turn = await a.call(_input())
            done = next(m for m in turn.stages if m.stage is Stage.LLM_DONE)
            outputs.append(str(done.meta["text"]))
            await a.close()

        assert outputs == ["A: same input", "B: same input"]


# ------------------------------------------------------------ conformance


class TestConformance:
    async def test_the_cascade_passes_its_own_conformance_suite(self) -> None:
        report = await run_conformance(
            _transport,
            CFG,
            stimulus=silence_stimulus(seconds=0.1, party=CALLER),
            timeout_s=1.0,
        )
        assert report.ok, report.summary()

    async def test_it_attests_its_audio_format(self) -> None:
        report = await run_conformance(
            _transport,
            CFG,
            stimulus=silence_stimulus(seconds=0.1, party=CALLER),
            timeout_s=1.0,
        )
        assert "audio_out" in report.attested


class TestASlotFailureEndsTheCall:
    """A failing slot must stop the call, not leave every later turn waiting."""

    async def test_the_tts_error_comes_out_of_rx(self) -> None:
        import asyncio

        from harbor_voice.providers.base import ProviderError
        from harbor_voice.providers.base import ProviderFailure

        class BrokenTTS(ToneTTS):
            async def synthesize(self, text, *, party):  # noqa: ANN001, ANN202
                raise ProviderError(
                    ProviderFailure.STREAM_BROKEN, "cut off", provider="t", model="m"
                )
                yield b""  # pragma: no cover - makes this a generator

        t = CascadeTransport(
            stt=ScriptedSTT(turns=["hello"]),
            llm=EchoLLM(),
            tts=BrokenTTS(),
            party=AGENT,
            input_idle_s=0.05,
        )
        await t.open(CFG)
        for frame in _input():
            await t.tx(frame)

        async def read() -> None:
            async for _ in t.rx():
                pass

        with pytest.raises(ProviderError, match="cut off"):
            await asyncio.wait_for(read(), 2.0)
        await t.close()
