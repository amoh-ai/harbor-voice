"""L1 persona and L2 speech.

The question running through this file: when a run asks for a condition it
cannot actually produce, does it say so, or does it quietly report the
condition's name over a default?
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from harbor_voice.audio.format import CANONICAL
from harbor_voice.capabilities import Layer
from harbor_voice.ids import PartyId
from harbor_voice.perturbation import PerturbationPipeline
from harbor_voice.perturbation import SeedSet
from harbor_voice.perturbation import get_profile
from harbor_voice.perturbation.base import ApplyContext
from harbor_voice.perturbation.persona import Accent
from harbor_voice.perturbation.persona import Behaviour
from harbor_voice.perturbation.persona import Gender
from harbor_voice.perturbation.persona import PersonaError
from harbor_voice.perturbation.persona import PersonaPlan
from harbor_voice.perturbation.persona import VoiceSelectable
from harbor_voice.perturbation.persona import VoiceSelection
from harbor_voice.perturbation.speech import DynamicMuffling
from harbor_voice.perturbation.speech import Loudness
from harbor_voice.perturbation.speech import SpeakingRate
from harbor_voice.perturbation.speech import SpeechError
from harbor_voice.perturbation.speech import SpeechPlan
from harbor_voice.perturbation.speech import VocalTics
from harbor_voice.providers import ToneTTS
from harbor_voice.simulator import ScriptContent
from harbor_voice.simulator import UniversalSimulator

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")


class AccentedTTS:
    """A TTS that can do French and nothing else. Declares it properly."""

    audio_format = CANONICAL
    name = "accented-tts"

    def __init__(self) -> None:
        self.selection: VoiceSelection | None = None

    def select_voice(self, selection: VoiceSelection) -> bool:
        self.selection = selection
        return selection.accent in (None, Accent.FRENCH)

    async def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        async for chunk in ToneTTS().synthesize(text, party=party):
            yield chunk


# ------------------------------------------------------------------ L1 voice


class TestVoiceSelectionIsDeclaredNotDiscovered:
    def test_a_tts_that_can_select_is_recognised(self) -> None:
        assert isinstance(AccentedTTS(), VoiceSelectable)

    def test_one_that_cannot_is_not(self) -> None:
        """ToneTTS has no voices. That is a fact about it, not a failure."""
        assert not isinstance(ToneTTS(), VoiceSelectable)

    def test_a_default_selection_is_honoured_by_anything(self) -> None:
        """Asking for nothing cannot fail to be given."""
        plan = PersonaPlan()
        assert plan.apply_to_tts(ToneTTS()) is True
        assert plan.achieved() == {"voice_honoured": 1.0}

    def test_an_accent_a_provider_cannot_do_is_recorded_not_raised(self) -> None:
        """On the persona axis: requested is not achieved."""
        plan = PersonaPlan(voice=VoiceSelection(accent=Accent.INDIAN))
        assert plan.apply_to_tts(AccentedTTS()) is False
        assert plan.achieved() == {"voice_honoured": 0.0}

    def test_an_accent_it_can_do_is_honoured(self) -> None:
        tts = AccentedTTS()
        plan = PersonaPlan(voice=VoiceSelection(accent=Accent.FRENCH))
        assert plan.apply_to_tts(tts) is True
        assert tts.selection is not None
        assert tts.selection.accent is Accent.FRENCH

    def test_a_provider_with_no_voices_at_all_fails_the_request(self) -> None:
        """The failure that would otherwise be invisible: an accent profile
        running on a TTS that has never heard of accents."""
        plan = PersonaPlan(voice=VoiceSelection(accent=Accent.FRENCH))
        assert plan.apply_to_tts(ToneTTS()) is False
        assert plan.achieved() == {"voice_honoured": 0.0}

    def test_unasked_is_not_the_same_as_refused(self) -> None:
        plan = PersonaPlan(voice=VoiceSelection(accent=Accent.FRENCH))
        assert plan.voice_honoured is None
        assert plan.achieved() == {}


class TestWhichLayersASelectionSwitchesOn:
    def test_an_empty_selection_switches_on_nothing(self) -> None:
        assert PersonaPlan().active_layers == frozenset()
        assert PersonaPlan().is_empty

    def test_each_axis_has_its_own_key(self) -> None:
        plan = PersonaPlan(
            voice=VoiceSelection(
                accent=Accent.SPANISH, gender=Gender.FEMALE, language="es"
            )
        )
        assert {str(k) for k in plan.active_layers} == {
            "L1.accent",
            "L1.gender",
            "L1.language",
        }

    def test_the_record_reports_each_axis_separately(self) -> None:
        """One verdict for the whole layer would hide a half-honoured run."""
        plan = PersonaPlan(
            voice=VoiceSelection(accent=Accent.INDIAN),
            behaviour=Behaviour(name="elderly", brief="You speak slowly."),
        )
        plan.apply_to_tts(AccentedTTS())
        record = plan.as_record()
        assert set(record) == {"L1.accent", "L1.behaviour"}
        assert record["L1.accent"]["achieved"] == {"voice_honoured": 0.0}
        assert record["L1.behaviour"]["params"]["name"] == "elderly"


class TestBehaviourComposesRatherThanReplaces:
    def test_it_appends_to_the_task_persona(self) -> None:
        """ "Rebooking a cancelled flight" and "elderly" are both true."""
        plan = PersonaPlan(
            behaviour=Behaviour(name="elderly", brief="You speak slowly.")
        )
        composed = plan.brief_for("You are rebooking a cancelled flight.")
        assert "rebooking" in composed
        assert "speak slowly" in composed

    def test_with_no_base_it_is_just_the_behaviour(self) -> None:
        plan = PersonaPlan(behaviour=Behaviour(name="b", brief="Be terse."))
        assert plan.brief_for("") == "Be terse."

    def test_with_no_behaviour_the_base_is_untouched(self) -> None:
        assert PersonaPlan().brief_for("base") == "base"

    def test_an_empty_brief_is_refused(self) -> None:
        """It would declare L1.behaviour active while changing nothing."""
        with pytest.raises(PersonaError, match="empty brief"):
            Behaviour(name="hollow", brief="   ")


# ------------------------------------------------------------- L2 text side


class TestVocalTicsRewriteTheTextNotTheWaveform:
    def _tics(self, **over: object) -> VocalTics:
        kwargs: dict[str, object] = {
            "phrases": ("um",),
            "probability": 1.0,
            "min_words": 3,
            "seeds": SeedSet.derive(5),
        }
        return VocalTics(**(kwargs | over))  # type: ignore[arg-type]

    def test_it_inserts_a_tic(self) -> None:
        out = self._tics().apply("I need to reschedule my appointment")
        assert "um" in out.split()
        assert len(out.split()) == 7

    def test_a_short_utterance_is_left_alone(self) -> None:
        """A two-word answer with an "um" bolted on is corrupted, not disfluent."""
        tics = self._tics(min_words=6)
        assert tics.apply("yes please") == "yes please"

    def test_an_ineligible_utterance_does_not_dilute_the_achieved_rate(self) -> None:
        tics = self._tics(min_words=6)
        tics.apply("yes please")
        assert tics.achieved() == {}

    def test_it_never_opens_the_turn(self) -> None:
        """Turn-initial hesitation is a different phenomenon."""
        for _ in range(20):
            out = self._tics().apply("one two three four five six")
            assert out.split()[0] == "one"

    def test_the_same_seed_rewrites_the_same_way(self) -> None:
        """An unreproducible perturbation is not a measurement."""
        text = "I would like to reschedule my appointment please"
        a = VocalTics(probability=0.5, seeds=SeedSet.derive(11))
        b = VocalTics(probability=0.5, seeds=SeedSet.derive(11))
        assert [a.apply(text) for _ in range(6)] == [b.apply(text) for _ in range(6)]

    def test_the_achieved_rate_exposes_a_layer_that_never_fired(self) -> None:
        """A p=0 run under a disfluent profile's name produced a clean call."""
        tics = self._tics(probability=0.0)
        for _ in range(4):
            tics.apply("one two three four five")
        assert tics.achieved() == {"utterances": 4.0, "modified": 0.0, "rate": 0.0}

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("phrases", ()),
            ("probability", 1.5),
            ("min_words", 0),
            ("max_per_utterance", 0),
        ],
    )
    def test_nonsense_is_refused(self, field: str, value: object) -> None:
        with pytest.raises(SpeechError):
            VocalTics(**{field: value})  # type: ignore[arg-type]


class TestSpeechPlan:
    def test_one_layer_one_transform(self) -> None:
        with pytest.raises(SpeechError, match="one layer, one transform"):
            SpeechPlan([VocalTics(), VocalTics()])

    def test_an_empty_plan_passes_text_through(self) -> None:
        assert SpeechPlan().apply("unchanged") == "unchanged"
        assert SpeechPlan().is_empty


# ----------------------------------------------------------- L2 signal side


def _ctx() -> ApplyContext:
    return ApplyContext(fmt=CANONICAL)


def _tone(n_frames: int = 3) -> bytes:
    import math

    rate = CANONICAL.sample_rate_hz
    n = CANONICAL.frame_bytes() // 2 * n_frames
    return b"".join(
        int(0.4 * 32767 * math.sin(2 * math.pi * 220 * i / rate)).to_bytes(
            2, "little", signed=True
        )
        for i in range(n)
    )


class TestSignalSideKeepsTheFrameLength:
    """Clean and transmitted must stay alignable, or nothing compares."""

    @pytest.mark.parametrize(
        "layer",
        [
            Loudness(gain_db=-6.0),
            Loudness(gain_db=12.0),
            SpeakingRate(factor=1.3),
            SpeakingRate(factor=0.7),
            DynamicMuffling(probability=1.0),
        ],
        ids=["quiet", "loud", "faster", "slower", "muffled"],
    )
    def test_length_is_preserved(self, layer: object) -> None:
        pcm = _tone()
        layer.reset(SeedSet.derive(3))  # type: ignore[attr-defined]
        out = layer.apply(pcm, _ctx())  # type: ignore[attr-defined]
        assert len(out) == len(pcm)

    def test_loudness_actually_changes_the_level(self) -> None:
        from harbor_voice.perturbation import dsp

        pcm = _tone()
        quiet = Loudness(gain_db=-20.0)
        quiet.reset(SeedSet.derive(1))
        out = quiet.apply(pcm, _ctx())
        assert dsp.rms(dsp.to_array(out)) < dsp.rms(dsp.to_array(pcm)) / 5

    def test_clipping_is_counted_not_prevented(self) -> None:
        """A caller who shouts into a phone clips. Limiting it invents a
        compressor nobody said was there."""
        loud = Loudness(gain_db=24.0)
        loud.reset(SeedSet.derive(1))
        loud.apply(_tone(), _ctx())
        assert loud.achieved()["clipped_fraction"] > 0.0

    def test_speaking_rate_admits_it_shifts_pitch(self) -> None:
        """Shipping resampling as prosody control would be the plausible
        wrong number this design exists to make unavailable."""
        assert SpeakingRate(factor=1.3).params()["pitch_preserved"] is False

    def test_a_rate_of_one_is_a_no_op(self) -> None:
        pcm = _tone()
        rate = SpeakingRate(factor=1.0)
        rate.reset(SeedSet.derive(1))
        assert rate.apply(pcm, _ctx()) == pcm

    def test_muffling_reports_how_much_of_the_call_it_touched(self) -> None:
        muffle = DynamicMuffling(probability=1.0, segment_s=10.0)
        muffle.reset(SeedSet.derive(1))
        for _ in range(5):
            muffle.apply(_tone(1), _ctx())
        assert muffle.achieved()["muffled_fraction"] == pytest.approx(1.0)

    def test_muffling_that_never_fires_says_so(self) -> None:
        muffle = DynamicMuffling(probability=0.0)
        muffle.reset(SeedSet.derive(1))
        for _ in range(5):
            muffle.apply(_tone(1), _ctx())
        assert muffle.achieved()["muffled_fraction"] == 0.0

    @pytest.mark.parametrize(
        ("cls", "field", "value"),
        [
            (SpeakingRate, "factor", 0.0),
            (DynamicMuffling, "probability", 2.0),
            (DynamicMuffling, "cutoff_hz", 0.0),
            (DynamicMuffling, "segment_s", 0.0),
            (DynamicMuffling, "transition_s", -1.0),
        ],
    )
    def test_nonsense_is_refused(self, cls: type, field: str, value: float) -> None:
        with pytest.raises(SpeechError):
            cls(**{field: value})


# --------------------------------------------------------------- composition


class TestTheProfilesThatNeededThem:
    @pytest.mark.parametrize("name", ["eva_a", "eva_a_plus_b", "tau_voice_default"])
    def test_they_build_now(self, name: str) -> None:
        pipeline = get_profile(name).pipeline(SeedSet.derive(1))
        assert not pipeline.is_unperturbed

    def test_eva_a_pins_accent_and_not_behaviour(self) -> None:
        """EVA makes them mutually exclusive — one agent ID per call."""
        pipeline = get_profile("eva_a").pipeline(SeedSet.derive(1))
        assert {str(k) for k in pipeline.active_layers} == {"L1.accent"}

    def test_eva_a_degrades_no_signal(self) -> None:
        """An accented caller on a clean line is still a clean line."""
        assert get_profile("eva_a").pipeline(SeedSet.derive(1)).is_clean

    def test_tau_combines_a_text_layer_with_three_signal_ones(self) -> None:
        pipeline = get_profile("tau_voice_default").pipeline(SeedSet.derive(1))
        assert {str(k) for k in pipeline.active_layers} == {
            "L2.vocal_tics",
            "L3.background_bed",
            "L4.codec",
            "L4.frame_loss",
        }
        assert not pipeline.speech.is_empty
        assert not pipeline.is_clean

    def test_the_written_plan_carries_every_insertion_point(self) -> None:
        """One record, whether the layer touched audio or not."""
        pipeline = get_profile("eva_a_plus_b").pipeline(SeedSet.derive(1))
        layers = pipeline.as_record().as_dict()["layers"]
        assert set(layers) == {"L1.accent", "L3.background_bed"}


class TestThroughTheSimulator:
    async def test_said_is_the_text_after_the_tic_went_in(self) -> None:
        """Scoring against the pre-tic sentence would charge the
        agent for hearing correctly."""
        pipeline = PerturbationPipeline(
            seeds=SeedSet.derive(2),
            speech=SpeechPlan(
                [VocalTics(phrases=("um",), probability=1.0, min_words=3)],
                seeds=SeedSet.derive(2),
            ),
        )
        sim = UniversalSimulator(
            party=CALLER,
            content=ScriptContent(turns=["I need to reschedule my appointment"]),
            tts=ToneTTS(),
            pipeline=pipeline,
            seeds=SeedSet.derive(2),
        )
        utterance = await sim.next_utterance()
        assert utterance is not None
        assert "um" in utterance.said.split()

    async def test_the_simulator_declares_the_persona_layer(self) -> None:
        pipeline = PerturbationPipeline(
            seeds=SeedSet.derive(2),
            persona=PersonaPlan(voice=VoiceSelection(accent=Accent.FRENCH)),
        )
        sim = UniversalSimulator(
            party=CALLER,
            content=ScriptContent(turns=["hello"]),
            tts=AccentedTTS(),
            pipeline=pipeline,
            seeds=SeedSet.derive(2),
        )
        assert Layer.PERSONA in sim.capabilities.layers

    async def test_a_run_that_could_not_get_its_accent_records_that(self) -> None:
        """The whole point: the trajectory must not claim the condition."""
        pipeline = PerturbationPipeline(
            seeds=SeedSet.derive(2),
            persona=PersonaPlan(voice=VoiceSelection(accent=Accent.INDIAN)),
        )
        UniversalSimulator(
            party=CALLER,
            content=ScriptContent(turns=["hello"]),
            tts=AccentedTTS(),
            pipeline=pipeline,
            seeds=SeedSet.derive(2),
        )
        record = pipeline.as_record().as_dict()["layers"]
        assert record["L1.accent"]["achieved"] == {"voice_honoured": 0.0}
