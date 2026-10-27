"""Wire encoders — the last stage of the signal chain, where format changes.

Every perturbation keeps a frame's length and format, which is what lets the
clean twin stay comparable sample for sample. A phone line breaks
that: it resamples to 8 kHz and encodes μ-law, so the transmitted frame is a
different size and a different format from the clean one. Forcing that into
a perturbation would mean decoding back to the pipeline format to keep the
length — a frame the agent never receives.

So it gets its own slot. A :class:`WireEncoder` runs once, after every
per-frame layer:

- it may change rate and encoding, never duration — clean and transmitted
  align **by time** across it;
- it is stateful across frames — a resampler and a filter carry state from
  one frame into the next, exactly as on a real line, and silence goes through
  it too;
- it is part of the plan record, under L4, like any other channel layer.

Core ships :class:`G711Encoder`. A benchmark with its own channel recipe
(tau-voice's band-pass phone line, say) brings its own encoder.
"""

from __future__ import annotations

import audioop
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import override

from harbor_voice.audio import g711
from harbor_voice.audio.format import AudioFormat
from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import LayerKey
from harbor_voice.perturbation.base import SeedSet

TELEPHONY = AudioFormat(sample_rate_hz=8_000, channels=1, encoding="mulaw")


class WireEncoderError(ValueError):
    """An encoder was given audio it cannot encode."""


class WireEncoder(ABC):
    """Converts the pipeline's PCM into what goes on the wire, frame by frame."""

    def __init__(self, *, name: str) -> None:
        self._name = name
        self._frames = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def layer(self) -> Layer:
        return Layer.CHANNEL

    @property
    def key(self) -> LayerKey:
        return LayerKey(layer=Layer.CHANNEL, name=self._name)

    @property
    @abstractmethod
    def output_format(self) -> AudioFormat:
        """What :meth:`encode` returns."""

    def accepts(self, fmt: AudioFormat) -> bool:
        """Whether this encoder can take frames in ``fmt``. Mono PCM16 by default."""
        return fmt.encoding == "pcm16" and fmt.channels == 1

    def encode(self, pcm: bytes, fmt: AudioFormat) -> bytes:
        """Encode one frame. Stateful: call for every frame, in order."""
        if not self.accepts(fmt):
            raise WireEncoderError(f"{self._name} cannot encode {fmt}")
        self._frames += 1
        return self._encode(pcm, fmt)

    @abstractmethod
    def _encode(self, pcm: bytes, fmt: AudioFormat) -> bytes: ...

    def reset(self, seeds: SeedSet) -> None:
        """Clear carried state for a new run."""
        del seeds
        self._frames = 0

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """The requested settings, as written into the plan."""

    def achieved(self) -> dict[str, float]:
        return {"frames": float(self._frames)}


class G711Encoder(WireEncoder):
    """Resample to 8 kHz and encode real G.711 μ-law.

    The resampler is ``audioop.ratecv``, which carries its state across
    frames — the property a streamed encoder needs, and the same resampler
    tau2 uses. It interpolates linearly and does not band-limit first, so
    energy above 4 kHz folds back; a benchmark that defines its own phone
    line (with a band-pass, say) supplies its own encoder.
    """

    def __init__(self) -> None:
        super().__init__(name="g711")
        self._state: Any = None

    @property
    @override
    def output_format(self) -> AudioFormat:
        return TELEPHONY

    @override
    def _encode(self, pcm: bytes, fmt: AudioFormat) -> bytes:
        if fmt.sample_rate_hz != TELEPHONY.sample_rate_hz:
            pcm, self._state = audioop.ratecv(
                pcm, 2, 1, fmt.sample_rate_hz, TELEPHONY.sample_rate_hz, self._state
            )
        return g711.encode(pcm)

    @override
    def reset(self, seeds: SeedSet) -> None:
        super().reset(seeds)
        self._state = None

    @override
    def params(self) -> dict[str, Any]:
        return {"encoding": "mulaw", "sample_rate_hz": TELEPHONY.sample_rate_hz}
