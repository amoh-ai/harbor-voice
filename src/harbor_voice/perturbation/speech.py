"""L2 — how they speak.

The layer that does not fit in one place, because speech does not. Half of L2
changes **what the text says** before anything synthesises it; the other half
changes **the signal** after something has.

=====================  ==============  ==============================  ========
Perturbation           Applied to      What it models                  Prior art
=====================  ==============  ==============================  ========
``vocal_tics``         text            "um", "you know", mid-sentence  tau
``disfluency``         text / corpus   real hesitation                 FDB-v3
``speaking_rate``      signal          faster or slower delivery       —
``loudness``           signal          a quiet or shouting caller      —
``dynamic_muffling``   signal          a hand over the mic, movement   tau
=====================  ==============  ==============================  ========

Putting a tic in the text rather than in the waveform is not a shortcut, it is
the only version that works: an "um" spliced into a finished waveform has no
coarticulation with its neighbours and a listener hears an edit. Synthesised
from text it is at least a plausible utterance.

**The honest limit:** TTS-generated "um" is not
what a hesitation sounds like, which is why FDB-v3 uses *recorded humans
annotated by disfluency type*. ``vocal_tics`` here models tau's version — a
phrase list inserted into text — and nothing here claims to be FDB's. Real
disfluency arrives as a corpus through
:class:`~harbor_voice.simulator.content.CorpusContent`, not from this module.

### Speaking rate changes pitch, and says so

Resampling is time-stretch **and** pitch-shift together; separating them needs
a phase vocoder, which is a different and much larger piece of work. A caller
sped up 1.3x here sounds slightly chipmunked, and that is recorded in the plan
rather than described as "speaking rate". Shipping a silent approximation of
prosody and calling it speaking rate would be exactly the kind of plausible
wrong number this design exists to make unavailable.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import override

from harbor_voice._deps import numpy as _numpy
from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import LayerKey
from harbor_voice.perturbation import dsp
from harbor_voice.perturbation.base import ApplyContext
from harbor_voice.perturbation.base import Perturbation
from harbor_voice.perturbation.base import RunningMean
from harbor_voice.perturbation.base import SeedSet

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

_PROSODY_STREAM = "prosody"


class SpeechError(ValueError):
    """An L2 layer was configured in a way that cannot produce a measurement."""


# ------------------------------------------------------------------ text side


class TextPerturbation(ABC):
    """An L2 layer that rewrites the caller's words before synthesis.

    Deliberately not a :class:`~harbor_voice.perturbation.base.Perturbation`:
    that contract is ``apply(pcm, ctx) -> bytes`` with an enforced equal
    length, and a tic inserted into a sentence changes its length by design.
    Same reasoning as L5 — a layer that does not transform a frame does not
    belong in the frame chain, however convenient the base class would be.

    The **said** side of said-vs-heard is the text *after* this runs.
    A caller who said "I need to, um, reschedule" said that, and scoring the
    agent against the pre-tic sentence would charge it for hearing correctly.
    """

    def __init__(self, *, name: str) -> None:
        self._name = name
        self._rng: Any = None
        self._applied = 0
        self._considered = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def layer(self) -> Layer:
        return Layer.SPEECH

    @property
    def key(self) -> LayerKey:
        return LayerKey(layer=self._layer_name(), name=self._name)

    def _layer_name(self) -> Layer:
        return Layer.SPEECH

    @abstractmethod
    def apply(self, text: str) -> str:
        """Rewrite one utterance."""

    def reset(self, seeds: SeedSet) -> None:
        """Re-seed and clear per-run counters."""
        self._rng = seeds.rng(_PROSODY_STREAM)
        self._applied = 0
        self._considered = 0

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """The requested settings, as written into the plan."""

    def achieved(self) -> dict[str, float]:
        """How often it actually fired.

        A ``p=0.3`` tic layer that happened to fire on none of four utterances
        produced a clean run under a disfluent profile's name. The rate is the
        only way anyone finds that out.
        """
        if not self._considered:
            return {}
        return {
            "utterances": float(self._considered),
            "modified": float(self._applied),
            "rate": self._applied / self._considered,
        }

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.key})"


class VocalTics(TextPerturbation):
    """Insert filler phrases into the caller's own sentences — tau's version.

    Args:
        phrases: What to insert.
        probability: Chance an eligible utterance gets one.
        min_words: Utterances shorter than this are left alone. A two-word
            answer with an "um" bolted on is not a disfluent caller, it is a
            corrupted one, and tau sets this for the same reason.
        max_per_utterance: Upper bound per turn.
    """

    def __init__(
        self,
        *,
        phrases: Sequence[str] = ("um", "uh", "you know", "like"),
        probability: float = 0.5,
        min_words: int = 6,
        max_per_utterance: int = 1,
        seeds: SeedSet | None = None,
    ) -> None:
        super().__init__(name="vocal_tics")
        if not phrases:
            raise SpeechError("vocal_tics needs at least one phrase")
        if not 0.0 <= probability <= 1.0:
            raise SpeechError(f"probability out of range: {probability}")
        if min_words < 1:
            raise SpeechError(f"min_words must be >= 1: {min_words}")
        if max_per_utterance < 1:
            raise SpeechError(f"max_per_utterance must be >= 1: {max_per_utterance}")
        self._phrases = tuple(phrases)
        self._probability = probability
        self._min_words = min_words
        self._max = max_per_utterance
        self.reset(seeds or SeedSet.derive(0))

    @override
    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) < self._min_words:
            # Not counted as considered: it was never eligible, and folding
            # it in would make the achieved rate a function of how terse the
            # script is rather than of how often the layer fired.
            return text
        self._considered += 1
        if float(self._rng.random()) > self._probability:
            return text

        count = 1 + int(self._rng.integers(self._max))
        for _ in range(count):
            phrase = self._phrases[int(self._rng.integers(len(self._phrases)))]
            # Never at position 0: a turn that opens with "um" is a different
            # phenomenon (turn-initial hesitation) and tau does not model it.
            at = 1 + int(self._rng.integers(max(1, len(words) - 1)))
            words.insert(at, phrase)
        self._applied += 1
        return " ".join(words)

    @override
    def params(self) -> dict[str, Any]:
        return {
            "phrases": list(self._phrases),
            "probability": self._probability,
            "min_words": self._min_words,
            "max_per_utterance": self._max,
        }


# ---------------------------------------------------------------- signal side


class Loudness(Perturbation):
    """Scale the caller's level — a quiet, distant, or shouting speaker.

    Args:
        gain_db: Positive is louder. Clipping is **counted, not prevented**:
            a caller who shouts into a phone clips, and silently limiting it
            would model a compressor nobody said was there.
    """

    def __init__(self, *, gain_db: float = -6.0) -> None:
        super().__init__(name="loudness", layer=Layer.SPEECH)
        self._gain_db = gain_db
        self._clipped = 0
        self._samples = 0
        self._level = RunningMean()

    @override
    def apply(self, pcm: bytes, ctx: ApplyContext) -> bytes:
        samples = dsp.to_array(pcm)
        scaled = samples * (10.0 ** (self._gain_db / 20.0))
        self._clipped += int(dsp.clipped_fraction(scaled) * scaled.size)
        self._samples += int(scaled.size)
        if scaled.size:
            self._level.add(dsp.rms(scaled))
        self._record(
            clipped_fraction=(self._clipped / self._samples if self._samples else 0.0),
            **self._level.as_record("rms"),
        )
        return dsp.to_pcm(scaled)

    @override
    def reset(self, seeds: SeedSet) -> None:
        del seeds  # deterministic: a gain has nothing to randomise
        self._clipped = 0
        self._samples = 0
        self._level = RunningMean()

    @override
    def params(self) -> dict[str, Any]:
        return {"gain_db": self._gain_db}


class SpeakingRate(Perturbation):
    """Speed the caller up or down, **pitch included**.

    Args:
        factor: Above 1.0 is faster. 1.3 makes a caller noticeably quicker and
            noticeably higher — this is resampling, not a phase vocoder, and
            the pitch shift is real. ``pitch_preserved: false`` goes into the
            plan so nobody reads this as prosody control.

    The frame length is held constant, which is what the clean reference requires of
    anything in the signal chain: a faster caller delivers *more* speech per
    frame, so the tail of a sped-up utterance is padded with the silence the
    caller would really have left, and a slowed one is truncated at the frame
    boundary and continues in the next.
    """

    def __init__(self, *, factor: float = 1.0) -> None:
        super().__init__(name="speaking_rate", layer=Layer.SPEECH)
        if factor <= 0:
            raise SpeechError(f"factor must be positive: {factor}")
        self._factor = factor
        self._carry: np.ndarray | None = None

    @override
    def apply(self, pcm: bytes, ctx: ApplyContext) -> bytes:
        if self._factor == 1.0:
            return pcm
        np = _numpy()
        samples = dsp.to_array(pcm)
        want = samples.size

        rate = ctx.fmt.sample_rate_hz
        resampled = dsp.resample_clean(samples, rate, round(rate / self._factor))
        if self._carry is not None and self._carry.size:
            resampled = np.concatenate([self._carry, resampled])

        if resampled.size >= want:
            out = resampled[:want]
            self._carry = resampled[want:]
        else:
            out = np.zeros(want, dtype=resampled.dtype)
            out[: resampled.size] = resampled
            self._carry = None

        self._record(
            factor=self._factor,
            carried_samples=float(0 if self._carry is None else self._carry.size),
        )
        return dsp.to_pcm(out)

    @override
    def reset(self, seeds: SeedSet) -> None:
        del seeds  # deterministic
        self._carry = None

    @override
    def params(self) -> dict[str, Any]:
        return {"factor": self._factor, "pitch_preserved": False}


class DynamicMuffling(Perturbation):
    """Intermittent low-pass, as if the caller turned away or covered the mic.

    tau's version, and the reason it is *dynamic*: a constant low-pass is a
    stationary channel that an agent's front end adapts to once and then
    ignores. Muffling that comes and goes is the hard case, and it is the one
    that happens in a real call.

    Args:
        probability: Chance of entering a muffled stretch at each decision.
        cutoff_hz: Low-pass corner while muffled.
        segment_s: Mean length of a muffled stretch.
        transition_s: Fade in and out, so the onset is not a click that a VAD
            reads as an event of its own.
    """

    def __init__(
        self,
        *,
        probability: float = 0.15,
        cutoff_hz: float = 1200.0,
        segment_s: float = 1.5,
        transition_s: float = 0.1,
    ) -> None:
        super().__init__(name="dynamic_muffling", layer=Layer.SPEECH)
        if not 0.0 <= probability <= 1.0:
            raise SpeechError(f"probability out of range: {probability}")
        if cutoff_hz <= 0:
            raise SpeechError(f"cutoff_hz must be positive: {cutoff_hz}")
        if segment_s <= 0:
            raise SpeechError(f"segment_s must be positive: {segment_s}")
        if transition_s < 0:
            raise SpeechError(f"transition_s cannot be negative: {transition_s}")
        self._probability = probability
        self._cutoff_hz = cutoff_hz
        self._segment_s = segment_s
        self._transition_s = transition_s
        self._rng: Any = None
        self._remaining_s = 0.0
        self._muffled_s = 0.0
        self._total_s = 0.0

    @override
    def apply(self, pcm: bytes, ctx: ApplyContext) -> bytes:
        duration = ctx.fmt.duration_s(len(pcm))
        self._total_s += duration

        if self._remaining_s <= 0.0:
            if float(self._rng.random()) < self._probability:
                self._remaining_s = self._segment_s
            else:
                self._record(muffled_fraction=self._muffled_s / self._total_s)
                return pcm

        self._remaining_s -= duration
        self._muffled_s += duration

        samples = dsp.to_array(pcm)
        muffled = dsp.low_pass(
            samples, cutoff_hz=self._cutoff_hz, rate_hz=ctx.fmt.sample_rate_hz
        )
        blended = self._blend(samples, muffled, ctx)
        self._record(muffled_fraction=self._muffled_s / self._total_s)
        return dsp.to_pcm(blended)

    def _blend(self, dry: np.ndarray, wet: np.ndarray, ctx: ApplyContext) -> np.ndarray:
        """Fade between dry and muffled across the transition window."""
        if self._transition_s <= 0:
            return wet
        np = _numpy()
        n = dry.size
        fade = max(1, int(self._transition_s * ctx.fmt.sample_rate_hz))
        if fade >= n:
            ramp = np.linspace(0.0, 1.0, n, dtype=dry.dtype)
        else:
            ramp = np.ones(n, dtype=dry.dtype)
            ramp[:fade] = np.linspace(0.0, 1.0, fade, dtype=dry.dtype)
        return dry * (1.0 - ramp) + wet * ramp

    @override
    def reset(self, seeds: SeedSet) -> None:
        self._rng = seeds.rng(_PROSODY_STREAM)
        self._remaining_s = 0.0
        self._muffled_s = 0.0
        self._total_s = 0.0

    @override
    def params(self) -> dict[str, Any]:
        return {
            "probability": self._probability,
            "cutoff_hz": self._cutoff_hz,
            "segment_s": self._segment_s,
            "transition_s": self._transition_s,
        }


# --------------------------------------------------------------- the text plan


class SpeechPlan:
    """The L2 text-side layers, beside the signal chain.

    The signal-side L2 perturbations are ordinary
    :class:`~harbor_voice.perturbation.base.Perturbation` objects and go in
    the pipeline's chain like any other. Only the text rewriters need a home
    of their own, because they run before there is audio.
    """

    def __init__(
        self,
        transforms: Sequence[TextPerturbation] = (),
        *,
        seeds: SeedSet | None = None,
    ) -> None:
        names = [t.name for t in transforms]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise SpeechError(
                f"two transforms named {sorted(duplicates)}; one layer, one "
                f"transform, or the plan record cannot say which produced what"
            )
        self._transforms = tuple(transforms)
        self._seeds = seeds or SeedSet.derive(0)
        self.reset(self._seeds)

    @property
    def transforms(self) -> tuple[TextPerturbation, ...]:
        return self._transforms

    @property
    def is_empty(self) -> bool:
        return not self._transforms

    @property
    def active_layers(self) -> frozenset[LayerKey]:
        return frozenset(t.key for t in self._transforms)

    def reset(self, seeds: SeedSet | None = None) -> None:
        if seeds is not None:
            self._seeds = seeds
        for transform in self._transforms:
            transform.reset(self._seeds)

    def apply(self, text: str) -> str:
        """Run every transform, in declaration order."""
        for transform in self._transforms:
            text = transform.apply(text)
        return text

    def as_record(self) -> dict[str, dict[str, Any]]:
        return {
            str(t.key): {"params": t.params(), "achieved": t.achieved()}
            for t in self._transforms
        }

    @override
    def __repr__(self) -> str:
        layers = ", ".join(str(t.key) for t in self._transforms) or "none"
        return f"SpeechPlan({layers})"
