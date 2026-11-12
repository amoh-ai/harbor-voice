"""L1 persona and L2 speech.

The question running through this file: when a run asks for a condition it
cannot actually produce, does it say so, or does it quietly report the
condition's name over a default?
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np
import pytest

from harbor_voice.audio.format import CANONICAL
from harbor_voice.capabilities import Layer
from harbor_voice.ids import PartyId
from harbor_voice.perturbation import dsp
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
            DynamicMuffling(probability=1.0),
        ],
        ids=["quiet", "loud", "muffled"],
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
            (DynamicMuffling, "taps", 4),
        ],
    )
    def test_nonsense_is_refused(self, cls: type, field: str, value: float) -> None:
        with pytest.raises(SpeechError):
            cls(**{field: value})


def _signal(seconds: float, *freqs: float) -> np.ndarray:
    """A sum of tones, as float samples at the canonical rate."""
    rate = CANONICAL.sample_rate_hz
    t = np.arange(int(seconds * rate)) / rate
    return sum(0.2 * np.sin(2 * np.pi * f * t) for f in freqs) * np.ones_like(t)


def _framed(layer: DynamicMuffling, x: np.ndarray) -> np.ndarray:
    """Run float samples through a frame layer one canonical frame at a time."""
    size = CANONICAL.frame_bytes()
    pcm = dsp.to_pcm(x)
    out = b"".join(
        layer.apply(pcm[i : i + size], _ctx()) for i in range(0, len(pcm), size)
    )
    return dsp.to_array(out)


class TestSpeakingRateIsAppliedToTheUtterance:
    """A faster caller finishes sooner, at the same pitch."""

    def test_faster_is_shorter_and_keeps_its_pitch(self) -> None:
        rate = CANONICAL.sample_rate_hz
        x = _signal(2.0, 300.0)
        y = dsp.to_array(SpeakingRate(factor=1.3).apply(dsp.to_pcm(x), CANONICAL))
        assert y.size == pytest.approx(x.size / 1.3, abs=2)
        peak = np.fft.rfftfreq(y.size, 1 / rate)[np.argmax(np.abs(np.fft.rfft(y)))]
        assert peak == pytest.approx(300.0, abs=3.0)

    def test_slower_is_longer(self) -> None:
        x = _signal(1.0, 300.0)
        y = dsp.to_array(SpeakingRate(factor=0.8).apply(dsp.to_pcm(x), CANONICAL))
        assert y.size == pytest.approx(x.size / 0.8, abs=2)

    def test_the_stretch_leaves_no_gaps(self) -> None:
        """No silent gaps between frames."""
        x = _signal(1.0, 220.0, 330.0)
        y = dsp.to_array(SpeakingRate(factor=1.3).apply(dsp.to_pcm(x), CANONICAL))
        edge = CANONICAL.sample_rate_hz // 20
        quiet = np.abs(y[edge:-edge]) < 1e-4
        longest = max(
            (len(run) for run in "".join("1" if q else "0" for q in quiet).split("0")),
            default=0,
        )
        assert longest < 20

    def test_a_rate_of_one_is_a_no_op(self) -> None:
        pcm = _tone()
        assert SpeakingRate(factor=1.0).apply(pcm, CANONICAL) == pcm

    def test_it_says_pitch_is_kept(self) -> None:
        assert SpeakingRate(factor=1.3).params()["pitch_preserved"] is True

    def test_it_is_refused_as_a_frame_layer(self) -> None:
        with pytest.raises(SpeechError, match="SpeechPlan"):
            PerturbationPipeline([SpeakingRate(factor=1.3)])  # type: ignore[list-item]

    def test_the_plan_declares_and_records_it(self) -> None:
        seeds = SeedSet.derive(1)
        rate = SpeakingRate(factor=1.3)
        pipeline = PerturbationPipeline(
            seeds=seeds, speech=SpeechPlan(rate=rate, seeds=seeds)
        )
        rate.apply(dsp.to_pcm(_signal(1.0, 300.0)), CANONICAL)
        assert {str(k) for k in pipeline.active_layers} == {"L2.speaking_rate"}
        record = pipeline.as_record().as_dict()["layers"]["L2.speaking_rate"]
        assert record["params"]["factor"] == 1.3
        assert record["achieved"]["factor_mean"] == pytest.approx(1.3, abs=0.01)

    def test_a_rate_of_one_is_not_an_active_layer(self) -> None:
        assert SpeechPlan(rate=SpeakingRate(factor=1.0)).is_empty


class TestMufflingIsOneFilterOverTheStream:
    """Continuous across frames, and strong enough to hear."""

    def test_frame_by_frame_equals_one_pass(self) -> None:
        """No step at frame boundaries."""
        x = _signal(1.0, 220.0, 2500.0)
        muffle = DynamicMuffling(probability=1.0, segment_s=10.0, transition_s=0.0)
        muffle.reset(SeedSet.derive(1))
        framed = _framed(muffle, x)
        kernel = dsp.low_pass_kernel(cutoff_hz=1500.0, rate_hz=CANONICAL.sample_rate_hz)
        one_pass = np.convolve(
            np.concatenate([np.zeros(kernel.size - 1), x]), kernel, mode="valid"
        )
        assert np.max(np.abs(framed - one_pass)) < 1e-3

    def test_unmuffled_audio_is_only_delayed(self) -> None:
        x = _signal(0.5, 220.0)
        muffle = DynamicMuffling(probability=0.0)
        muffle.reset(SeedSet.derive(1))
        out = _framed(muffle, x)
        delay = muffle.latency_samples
        assert np.max(np.abs(out[delay:] - x[: x.size - delay])) < 1e-3

    def test_the_band_above_the_corner_is_gone(self) -> None:
        muffle = DynamicMuffling(probability=1.0, segment_s=10.0)
        muffle.reset(SeedSet.derive(1))
        high = _framed(muffle, _signal(1.0, 3000.0))[CANONICAL.sample_rate_hz // 2 :]
        muffle.reset(SeedSet.derive(1))
        low = _framed(muffle, _signal(1.0, 300.0))[CANONICAL.sample_rate_hz // 2 :]
        assert dsp.db(dsp.rms(high) / dsp.rms(_signal(0.5, 3000.0))) < -40.0
        assert dsp.db(dsp.rms(low) / dsp.rms(_signal(0.5, 300.0))) > -1.0

    def test_the_fade_runs_across_frames(self) -> None:
        """The fade-in completes once over transition_s, not once per frame."""
        rate = CANONICAL.sample_rate_hz
        muffle = DynamicMuffling(probability=1.0, segment_s=10.0, transition_s=0.1)
        muffle.reset(SeedSet.derive(1))
        out = _framed(muffle, _signal(0.3, 3000.0))
        frame = CANONICAL.frame_bytes() // 2
        level = [
            dsp.rms(out[i : i + frame])
            for i in range(muffle.latency_samples, int(0.12 * rate), frame)
        ]
        assert all(
            b <= a + 1e-6 for a, b in zip(level, level[1:])
        )  # falls, never jumps back
        assert level[-1] < level[0] / 20

    def test_attenuation_takes_level_away(self) -> None:
        x = _signal(1.0, 300.0)
        plain = DynamicMuffling(probability=1.0, segment_s=10.0)
        quieter = DynamicMuffling(probability=1.0, segment_s=10.0, attenuation_db=-6.0)
        plain.reset(SeedSet.derive(1))
        quieter.reset(SeedSet.derive(1))
        half = CANONICAL.sample_rate_hz // 2
        drop = dsp.db(
            dsp.rms(_framed(quieter, x)[half:]) / dsp.rms(_framed(plain, x)[half:])
        )
        assert drop == pytest.approx(-6.0, abs=0.2)


# --------------------------------------------------------------- composition


class TestTheProfilesThatNeededThem:
    @pytest.mark.parametrize("name", ["accented", "accented_noisy"])
    def test_they_build_now(self, name: str) -> None:
        pipeline = get_profile(name).pipeline(SeedSet.derive(1))
        assert not pipeline.is_unperturbed

    def test_accented_pins_accent_and_not_behaviour(self) -> None:
        """Accent only, so a difference is the accent's."""
        pipeline = get_profile("accented").pipeline(SeedSet.derive(1))
        assert {str(k) for k in pipeline.active_layers} == {"L1.accent"}

    def test_accented_degrades_no_signal(self) -> None:
        """An accented caller on a clean line is still a clean line."""
        assert get_profile("accented").pipeline(SeedSet.derive(1)).is_clean

    def test_a_text_layer_combines_with_three_signal_ones(self) -> None:
        seeds = SeedSet.derive(1)
        noisy_line = get_profile("noisy_line").pipeline(seeds)
        pipeline = PerturbationPipeline(
            noisy_line.perturbations,
            seeds=seeds,
            speech=SpeechPlan([VocalTics()], seeds=seeds),
        )
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
        pipeline = get_profile("accented_noisy").pipeline(SeedSet.derive(1))
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

    async def test_a_faster_caller_finishes_sooner(self) -> None:
        """The rate lands on the synthesised line, before the clean copy."""

        async def spoken_bytes(rate: SpeakingRate | None) -> int:
            seeds = SeedSet.derive(2)
            pipeline = PerturbationPipeline(
                seeds=seeds, speech=SpeechPlan(rate=rate, seeds=seeds)
            )
            sim = UniversalSimulator(
                party=CALLER,
                content=ScriptContent(turns=["I need to reschedule my appointment"]),
                tts=ToneTTS(),
                pipeline=pipeline,
                seeds=seeds,
            )
            line = await sim.prepare("I need to reschedule my appointment")
            return len(line.pcm)

        normal = await spoken_bytes(None)
        faster = await spoken_bytes(SpeakingRate(factor=1.3))
        assert faster == pytest.approx(normal / 1.3, rel=0.02)

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
