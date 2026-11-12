"""L2 — how they speak.

The layer that does not fit in one place, because speech does not. Part of L2
changes what the text says before anything synthesises it, part changes the
whole synthesised utterance, and the rest changes the signal frame by frame.

=====================  ==============  ==============================  ========
Perturbation           Applied to      What it models                  Prior art
=====================  ==============  ==============================  ========
``vocal_tics``         text            "um", "you know", mid-sentence  tau
``disfluency``         text / corpus   real hesitation                 FDB-v3
``speaking_rate``      utterance       faster or slower delivery       —
``loudness``           signal          a quiet or shouting caller      —
``dynamic_muffling``   signal          a hand over the mic, movement   tau
=====================  ==============  ==============================  ========

Putting a tic in the text rather than in the waveform is not a shortcut, it is
the only version that works: an "um" spliced into a finished waveform has no
coarticulation with its neighbours and a listener hears an edit. Synthesised
from text it is at least a plausible utterance.

The honest limit: TTS-generated "um" is not
what a hesitation sounds like, which is why FDB-v3 uses *recorded humans
annotated by disfluency type*. ``vocal_tics`` here models tau's version — a
phrase list inserted into text — and nothing here claims to be FDB's. Real
disfluency arrives as a corpus through
:class:`~harbor_voice.simulator.content.CorpusContent`, not from this module.

### Speaking rate applies to the whole utterance

A faster caller finishes sooner, which a frame-at-a-time stage cannot produce.
So the rate stretches each synthesised utterance, pitch kept, before the clean
reference is kept.
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

    from harbor_voice.audio.format import AudioFormat

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

    The said side of said-vs-heard is the text *after* this runs.
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
        gain_db: Positive is louder. Clipping is counted, not prevented:
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


class SpeakingRate:
    """How fast the caller talks, applied to each synthesised utterance.

    Args:
        factor: Above 1.0 is faster; the utterance comes out about
            ``1 / factor`` as long, at the same pitch.

    Pass it as ``SpeechPlan(rate=...)``, not as a frame perturbation.
    Pre-recorded audio is left as it is.
    """

    def __init__(self, *, factor: float = 1.0) -> None:
        if factor <= 0:
            raise SpeechError(f"factor must be positive: {factor}")
        self._factor = factor
        self._ratio = RunningMean()

    @property
    def name(self) -> str:
        return "speaking_rate"

    @property
    def layer(self) -> Layer:
        return Layer.SPEECH

    @property
    def key(self) -> LayerKey:
        return LayerKey(layer=Layer.SPEECH, name=self.name)

    @property
    def factor(self) -> float:
        return self._factor

    def apply(self, pcm: bytes, fmt: AudioFormat) -> bytes:
        """Stretch one whole utterance; a rate of one returns it untouched."""
        if self._factor == 1.0 or not pcm:
            return pcm
        if fmt.channels != 1:
            raise SpeechError(
                f"speaking rate needs mono audio, got {fmt.channels} channels"
            )
        samples = dsp.to_array(pcm)
        stretched = dsp.time_stretch(samples, self._factor, fmt.sample_rate_hz)
        self._ratio.add(samples.size / max(stretched.size, 1))
        return dsp.to_pcm(stretched)

    def reset(self, seeds: SeedSet) -> None:
        del seeds  # deterministic
        self._ratio = RunningMean()

    def params(self) -> dict[str, Any]:
        return {"factor": self._factor, "pitch_preserved": True, "method": "wsola"}

    def achieved(self) -> dict[str, float]:
        return self._ratio.as_record("factor")

    @override
    def __repr__(self) -> str:
        return f"SpeakingRate(factor={self._factor})"


class DynamicMuffling(Perturbation):
    """Intermittent low-pass, as if the caller turned away or covered the mic.

    tau's version. Muffling that comes and goes is harder for an agent than
    a constant low-pass, which its front end adapts to once.

    Args:
        probability: Chance of starting a stretch, decided each unmuffled
            frame (0.02 at 20 ms frames is about once a second).
        cutoff_hz: Low-pass corner while muffled (tau: 1.5 kHz).
        segment_s: Length of a stretch (tau: 0.5 s).
        transition_s: Fade in and out (tau: 0.1 s).
        attenuation_db: Level change while muffled (tau: 0).
        taps: Filter length, odd.

    The filter and the fade carry across frames, so frame boundaries add no
    clicks. The filter delays every frame by ``latency_samples`` (about 5 ms
    at 24 kHz).
    """

    def __init__(
        self,
        *,
        probability: float = 0.02,
        cutoff_hz: float = 1500.0,
        segment_s: float = 0.5,
        transition_s: float = 0.1,
        attenuation_db: float = 0.0,
        taps: int = 255,
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
        if taps < 3 or taps % 2 == 0:
            raise SpeechError(f"taps must be odd and at least 3: {taps}")
        self._probability = probability
        self._cutoff_hz = cutoff_hz
        self._segment_s = segment_s
        self._transition_s = transition_s
        self._attenuation_db = attenuation_db
        self._taps = taps
        self._rng: Any = None
        self._remaining_s = 0.0
        self._muffled_s = 0.0
        self._total_s = 0.0
        self._mix = 0.0
        self._history: np.ndarray | None = None
        self._kernel: np.ndarray | None = None
        self._kernel_rate = 0

    @property
    def latency_samples(self) -> int:
        """The constant delay the stream filter adds."""
        return self._taps // 2

    @override
    def apply(self, pcm: bytes, ctx: ApplyContext) -> bytes:
        np = _numpy()
        samples = dsp.to_array(pcm)
        n = samples.size
        if n == 0:
            return pcm
        rate = ctx.fmt.sample_rate_hz
        duration = ctx.fmt.duration_s(len(pcm))
        self._total_s += duration

        kernel = self._kernel_for(rate)
        if self._history is None:
            self._history = np.zeros(self._taps - 1)
        stream = np.concatenate([self._history, samples])
        self._history = stream[-(self._taps - 1) :]
        dry = stream[self.latency_samples : self.latency_samples + n]

        if self._remaining_s <= 0.0 and float(self._rng.random()) < self._probability:
            self._remaining_s = self._segment_s
        target = 1.0 if self._remaining_s > 0.0 else 0.0
        if target:
            self._remaining_s -= duration
            self._muffled_s += duration
        self._record(muffled_fraction=self._muffled_s / self._total_s)

        if target == 0.0 and self._mix == 0.0:
            return dsp.to_pcm(dry)
        envelope = self._envelope(target, n, rate)
        wet = np.convolve(stream, kernel, mode="valid") * 10.0 ** (
            self._attenuation_db / 20.0
        )
        return dsp.to_pcm(dry * (1.0 - envelope) + wet * envelope)

    def _envelope(self, target: float, n: int, rate: int) -> np.ndarray:
        """Move the mix toward ``target`` at the transition's pace, from where it is."""
        np = _numpy()
        if self._transition_s <= 0:
            envelope = np.full(n, target)
        else:
            step = 1.0 / max(1, int(self._transition_s * rate))
            direction = 1.0 if target > self._mix else -1.0
            ramp = self._mix + direction * step * np.arange(1, n + 1)
            low, high = sorted((self._mix, target))
            envelope = np.clip(ramp, low, high)
        self._mix = float(envelope[-1])
        return envelope

    def _kernel_for(self, rate: int) -> np.ndarray:
        if self._kernel is None or self._kernel_rate != rate:
            self._kernel = dsp.low_pass_kernel(
                cutoff_hz=self._cutoff_hz, rate_hz=rate, taps=self._taps
            )
            self._kernel_rate = rate
        return self._kernel

    @override
    def reset(self, seeds: SeedSet) -> None:
        self._rng = seeds.rng(_PROSODY_STREAM)
        self._remaining_s = 0.0
        self._muffled_s = 0.0
        self._total_s = 0.0
        self._mix = 0.0
        self._history = None

    @override
    def params(self) -> dict[str, Any]:
        return {
            "probability": self._probability,
            "cutoff_hz": self._cutoff_hz,
            "segment_s": self._segment_s,
            "transition_s": self._transition_s,
            "attenuation_db": self._attenuation_db,
            "taps": self._taps,
            "latency_samples": self.latency_samples,
        }


# --------------------------------------------------------------- the text plan


class SpeechPlan:
    """The L2 layers that run before the signal chain: text and utterance.

    The signal-side L2 perturbations are ordinary
    :class:`~harbor_voice.perturbation.base.Perturbation` objects and go in
    the pipeline's chain like any other. The text rewriters run before there
    is audio, and the speaking rate runs on each whole utterance once there
    is, so both live here.
    """

    def __init__(
        self,
        transforms: Sequence[TextPerturbation] = (),
        *,
        rate: SpeakingRate | None = None,
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
        self._rate = rate
        self._seeds = seeds or SeedSet.derive(0)
        self.reset(self._seeds)

    @property
    def transforms(self) -> tuple[TextPerturbation, ...]:
        return self._transforms

    @property
    def rate(self) -> SpeakingRate | None:
        return self._rate

    @property
    def _rate_active(self) -> bool:
        return self._rate is not None and self._rate.factor != 1.0

    @property
    def is_empty(self) -> bool:
        return not self._transforms and not self._rate_active

    @property
    def active_layers(self) -> frozenset[LayerKey]:
        keys = {t.key for t in self._transforms}
        if self._rate is not None and self._rate_active:
            keys.add(self._rate.key)
        return frozenset(keys)

    def reset(self, seeds: SeedSet | None = None) -> None:
        if seeds is not None:
            self._seeds = seeds
        for transform in self._transforms:
            transform.reset(self._seeds)
        if self._rate is not None:
            self._rate.reset(self._seeds)

    def apply(self, text: str) -> str:
        """Run every transform, in declaration order."""
        for transform in self._transforms:
            text = transform.apply(text)
        return text

    def stretch(self, pcm: bytes, fmt: AudioFormat) -> bytes:
        """Apply the speaking rate to one synthesised utterance."""
        return self._rate.apply(pcm, fmt) if self._rate is not None else pcm

    def as_record(self) -> dict[str, dict[str, Any]]:
        record = {
            str(t.key): {"params": t.params(), "achieved": t.achieved()}
            for t in self._transforms
        }
        if self._rate is not None and self._rate_active:
            record[str(self._rate.key)] = {
                "params": self._rate.params(),
                "achieved": self._rate.achieved(),
            }
        return record

    @override
    def __repr__(self) -> str:
        layers = (
            ", ".join(str(k) for k in sorted(self.active_layers, key=str)) or "none"
        )
        return f"SpeechPlan({layers})"
