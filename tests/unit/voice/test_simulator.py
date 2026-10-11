"""Unit tests for the universal simulator."""

from __future__ import annotations

import pytest

from harbor_voice.audio.format import CANONICAL
from harbor_voice.capabilities import Layer
from harbor_voice.providers import EchoLLM
from harbor_voice.providers import ToneTTS
from harbor_voice.ids import PartyId
from harbor_voice.ids import PlanId
from harbor_voice.perturbation import BackgroundBed
from harbor_voice.perturbation import GilbertElliottLoss
from harbor_voice.perturbation import MuLawCodec
from harbor_voice.perturbation import PerturbationPipeline
from harbor_voice.perturbation import SeedSet
from harbor_voice.perturbation import synthetic_bed
from harbor_voice.simulator import CorpusClip
from harbor_voice.simulator import CorpusContent
from harbor_voice.simulator import PersonaContent
from harbor_voice.simulator import ScriptContent
from harbor_voice.simulator import SimulatorError
from harbor_voice.simulator import SourceKind
from harbor_voice.simulator import UniversalSimulator

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
TURNS = ["I need to reschedule", "next Tuesday please"]


def _pipeline(seed: int = 1) -> PerturbationPipeline:
    return PerturbationPipeline(
        [
            BackgroundBed(
                source=synthetic_bed(seconds=1.0, seed=1),
                target_snr_db=15.0,
                snr_drift_db=2.0,
                variation_speed=20.0,
            ),
            MuLawCodec(),
            GilbertElliottLoss(rate=0.3),
        ],
        seeds=SeedSet.derive(seed),
        plan_id=PlanId("test"),
    )


def _sim(**over: object) -> UniversalSimulator:
    kwargs: dict[str, object] = {
        "party": CALLER,
        "content": ScriptContent(turns=TURNS),
        "tts": ToneTTS(),
        "seeds": SeedSet.derive(1),
    }
    return UniversalSimulator(**(kwargs | over))  # type: ignore[arg-type]


async def _drain(sim: UniversalSimulator) -> list:
    out = []
    while (u := await sim.next_utterance([])) is not None:
        out.append(u)
    return out


# --------------------------------------------------------------- content


class TestContentSources:
    async def test_script_yields_turns_in_order_then_stops(self) -> None:
        source = ScriptContent(turns=TURNS)
        assert (await source.next([])).text == TURNS[0]  # type: ignore[union-attr]
        assert (await source.next([])).text == TURNS[1]  # type: ignore[union-attr]
        assert await source.next([]) is None

    async def test_script_is_deterministic_and_rewindable(self) -> None:
        source = ScriptContent(turns=TURNS)
        await source.next([])
        source.reset()
        assert (await source.next([])).text == TURNS[0]  # type: ignore[union-attr]
        assert source.is_deterministic
        assert source.kind is SourceKind.SCRIPT

    async def test_script_carries_intents(self) -> None:
        source = ScriptContent(turns=["hi"], intents=["greeting"])
        assert (await source.next([])).intent == "greeting"  # type: ignore[union-attr]

    async def test_corpus_yields_prerecorded_audio(self) -> None:
        """A synthesised 'um' is not a hesitation."""
        clip = CorpusClip(
            text="uh, I think so",
            audio=bytes(960),
            metadata={"disfluency": "filler"},
        )
        content = await CorpusContent(clips=[clip]).next([])
        assert content is not None
        assert content.is_prerecorded
        assert content.metadata["disfluency"] == "filler"

    async def test_corpus_text_is_still_the_said_side(self) -> None:
        """No TTS produced it, but it is still the WER reference."""
        clip = CorpusClip(text="annotated words", audio=bytes(960))
        content = await CorpusContent(clips=[clip]).next([])
        assert content is not None and content.text == "annotated words"

    async def test_persona_is_marked_non_deterministic(self) -> None:
        """So anything needing replay can refuse it up front."""
        source = PersonaContent(brief="a patient", llm=EchoLLM())
        assert not source.is_deterministic
        assert source.kind is SourceKind.LLM

    async def test_persona_stops_at_its_turn_cap(self) -> None:
        source = PersonaContent(brief="x", llm=EchoLLM(), max_turns=2)
        assert await source.next(["a"]) is not None
        assert await source.next(["b"]) is not None
        assert await source.next(["c"]) is None


# ------------------------------------------------------------- synthesis


class TestSynthesis:
    async def test_it_produces_framed_audio(self) -> None:
        utterances = await _drain(_sim())
        assert len(utterances) == 2
        for u in utterances:
            assert u.frames
            assert all(len(f.pcm) == CANONICAL.frame_bytes() for f in u.frames)

    async def test_the_said_side_is_kept(self) -> None:
        """The reference half of said-vs-heard."""
        utterances = await _drain(_sim())
        assert [u.said for u in utterances] == TURNS

    async def test_frames_share_one_produced_instant(self) -> None:
        """Batch synthesis: a frame cannot exist before it is made."""
        u = await _sim().next_utterance([])
        assert u is not None
        assert len({f.t_produced for f in u.frames}) == 1

    async def test_sequence_numbers_advance_across_utterances(self) -> None:
        utterances = await _drain(_sim())
        seqs = [f.seq for u in utterances for f in u.frames]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)

    async def test_text_without_a_tts_is_refused(self) -> None:
        sim = _sim(tts=None)
        with pytest.raises(SimulatorError, match="no TTS is configured"):
            await sim.next_utterance([])

    async def test_a_corpus_needs_no_tts(self) -> None:
        sim = _sim(
            content=CorpusContent(clips=[CorpusClip(text="hi", audio=bytes(1920))]),
            tts=None,
        )
        u = await sim.next_utterance([])
        assert u is not None and u.prerecorded
        assert len(u.frames) == 2

    async def test_misaligned_corpus_audio_is_rejected(self) -> None:
        sim = _sim(
            content=CorpusContent(clips=[CorpusClip(text="hi", audio=bytes(961))]),
            tts=None,
        )
        with pytest.raises(Exception, match="whole number of samples"):
            await sim.next_utterance([])


# ----------------------------------------------------------- degradation


class TestDegradation:
    async def test_the_clean_profile_leaves_audio_alone(self) -> None:
        sim = _sim()
        u = await sim.next_utterance([])
        assert u is not None and not u.is_degraded

    async def test_perturbation_marks_every_frame(self) -> None:
        sim = _sim(pipeline=_pipeline())
        u = await sim.next_utterance([])
        assert u is not None and u.is_degraded
        assert all(f.clean_ref is not None for f in u.frames)
        assert all(f.plan_ref == PlanId("test") for f in u.frames)

    async def test_the_clean_twin_is_retrievable(self) -> None:
        """Without this there is no SNR and no said-vs-heard WER."""
        pipe = _pipeline()
        sim = _sim(pipeline=pipe)
        u = await sim.next_utterance([])
        assert u is not None and u.clean_digest is not None
        clean = pipe.clean_store.get(u.clean_digest)
        assert len(clean) == sum(len(f.pcm) for f in u.frames)

    async def test_every_frame_passes_the_invariant(self) -> None:
        pipe = _pipeline()
        sim = _sim(pipeline=pipe)
        for u in await _drain(sim):
            for frame in u.frames:
                pipe.verify(frame)

    async def test_achieved_snr_is_recorded_and_near_target(self) -> None:
        """A tone, not silence — a silent reference makes SNR undefined."""
        pipe = _pipeline()
        await _drain(_sim(pipeline=pipe))
        bed = pipe.as_record().layers["L3.background_bed"]["achieved"]
        assert bed["achieved_snr_db_mean"] == pytest.approx(15.0, abs=0.5)


# ------------------------------------------------------- reproducibility


class TestReproducibility:
    async def test_the_same_seed_reproduces_the_audio(self) -> None:
        """The experiment the benchmark exists to run."""
        first = await _drain(_sim(pipeline=_pipeline(seed=5)))
        second = await _drain(_sim(pipeline=_pipeline(seed=5)))
        assert [f.pcm for u in first for f in u.frames] == [
            f.pcm for u in second for f in u.frames
        ]

    async def test_a_different_channel_seed_changes_only_the_channel(self) -> None:
        """Content held fixed, channel varied — the ablation."""
        a = await _drain(_sim(pipeline=_pipeline(seed=5)))
        b = await _drain(_sim(pipeline=_pipeline(seed=9)))
        assert [u.said for u in a] == [u.said for u in b], "words unchanged"
        assert [f.pcm for u in a for f in u.frames] != [
            f.pcm for u in b for f in u.frames
        ], "line degraded differently"

    async def test_reset_rewinds_everything(self) -> None:
        sim = _sim(pipeline=_pipeline())
        first = await _drain(sim)
        sim.reset()
        second = await _drain(sim)
        assert [u.said for u in first] == [u.said for u in second]
        assert [f.pcm for u in first for f in u.frames] == [
            f.pcm for u in second for f in u.frames
        ]


# ---------------------------------------------------------- capabilities


class TestCapabilities:
    def test_it_declares_the_layers_it_can_serve(self) -> None:
        sim = _sim(pipeline=_pipeline())
        layers = sim.capabilities.layers
        assert Layer.ENVIRONMENT in layers
        assert Layer.CHANNEL in layers
        assert Layer.PERSONA in layers, "it has a TTS, so it can pick a voice"

    def test_a_clean_simulator_declares_no_degradation_layers(self) -> None:
        layers = _sim().capabilities.layers
        assert Layer.ENVIRONMENT not in layers
        assert Layer.CHANNEL not in layers

    def test_a_corpus_simulator_has_no_persona_layer(self) -> None:
        sim = _sim(content=CorpusContent(clips=[]), tts=None)
        assert Layer.PERSONA not in sim.capabilities.layers


# -------------------------------------------------------------- fidelity


class TestFidelity:
    async def test_in_persona_is_absent_until_measured(self) -> None:
        """A fabricated 1.0 would hide exactly what needs reporting."""
        sim = _sim()
        await _drain(sim)
        fidelity = sim.fidelity()
        assert fidelity.in_persona is None
        assert not fidelity.is_measured
        assert any("not been measured" in n for n in fidelity.notes)

    async def test_a_measured_value_is_carried(self) -> None:
        sim = _sim()
        await _drain(sim)
        assert sim.fidelity(in_persona=0.93).in_persona == 0.93

    async def test_a_non_deterministic_source_is_flagged(self) -> None:
        sim = _sim(content=PersonaContent(brief="x", llm=EchoLLM(), max_turns=1))
        await _drain(sim)
        fidelity = sim.fidelity()
        assert not fidelity.deterministic_source
        assert any("not replayable" in n for n in fidelity.notes)

    async def test_prerecorded_fraction_is_reported(self) -> None:
        sim = _sim(
            content=CorpusContent(clips=[CorpusClip(text="a", audio=bytes(960))]),
            tts=None,
        )
        await _drain(sim)
        assert sim.fidelity().prerecorded_fraction == 1.0

    async def test_counts_are_reported(self) -> None:
        sim = _sim()
        await _drain(sim)
        assert sim.fidelity().utterances == 2
        assert sim.fidelity().source_kind is SourceKind.SCRIPT


class TestRepr:
    def test_repr_names_the_source_and_layers(self) -> None:
        r = repr(_sim(pipeline=_pipeline()))
        assert "source=script" in r and "layers=3" in r
