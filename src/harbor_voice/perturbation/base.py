"""Perturbation contract and seeding.

Two rules this module exists to make unavoidable:

Everything is seeded. A perturbation that builds an unseeded
``np.random.default_rng()`` inside each apply cannot be reproduced, and an
unreproducible perturbation is not a measurement.

Seeds are layered. Deriving both the dialogue text and the interrupt
coin-flip from one key makes it impossible to hold the conversation fixed
while varying the channel — which is the experiment the
whole benchmark exists to run. :class:`SeedSet` derives independent streams
from one root so each axis moves on its own.
"""

from __future__ import annotations

import hashlib
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import override

from harbor_voice._deps import numpy as _numpy
from harbor_voice.audio.format import AudioFormat
from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import LayerKey

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

_STREAMS = ("content", "prosody", "channel", "policy")


@dataclass(frozen=True, slots=True)
class SeedSet:
    """Independent seed streams derived from one root.

    Deriving rather than storing four unrelated integers means a run is
    reproducible from a single number, while the axes stay separable: fix
    ``content`` and vary ``channel`` and only the line degraded.
    """

    root: int
    content: int
    prosody: int
    channel: int
    policy: int

    @classmethod
    def derive(cls, root: int) -> SeedSet:
        """Derive every stream from ``root``, stably across processes.

        Uses SHA-256 rather than :func:`hash`, which is randomised per process
        and would make a "reproducible" run reproduce differently tomorrow.
        """
        values: dict[str, int] = {}
        for stream in _STREAMS:
            digest = hashlib.sha256(f"{root}:{stream}".encode()).digest()
            values[stream] = int.from_bytes(digest[:8], "big")
        return cls(root=root, **values)

    def rng(self, stream: str) -> np.random.Generator:
        """A generator for one named stream."""
        if stream not in _STREAMS:
            raise ValueError(f"unknown seed stream {stream!r}; expected {_STREAMS}")
        np = _numpy()
        return np.random.default_rng(getattr(self, stream))

    def as_record(self) -> dict[str, int]:
        """The form written into the trajectory."""
        return {"root": self.root, **{s: getattr(self, s) for s in _STREAMS}}


@dataclass(slots=True)
class ApplyContext:
    """State a perturbation may read while processing one frame.

    ``elapsed_s`` is what makes a bed continuous: it advances across
    the whole call, including silence, so the layer is a function of position
    in the call rather than position in an utterance.
    """

    fmt: AudioFormat
    elapsed_s: float = 0.0
    frame_index: int = 0
    is_speech: bool = True

    def advance(self, duration_s: float) -> None:
        self.elapsed_s += duration_s
        self.frame_index += 1


class Perturbation(ABC):
    """One named effect at one layer.

    Subclasses are stateful by design — a bed tracks its position, a bursty
    channel tracks its state — and must be reset between runs via
    :meth:`reset`, which also re-seeds.
    """

    def __init__(self, *, name: str, layer: Layer) -> None:
        self._name = name
        self._layer = layer
        self._achieved: dict[str, float] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def layer(self) -> Layer:
        return self._layer

    @property
    def key(self) -> LayerKey:
        """The ``L3.background_bed``-style key a task activates it by."""
        return LayerKey(layer=self._layer, name=self._name)

    @abstractmethod
    def apply(self, pcm: bytes, ctx: ApplyContext) -> bytes:
        """Transform one frame."""

    @abstractmethod
    def reset(self, seeds: SeedSet) -> None:
        """Re-seed and clear state for a new run."""

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """The requested settings, as written into the plan."""

    def achieved(self) -> dict[str, float]:
        """What was actually delivered, measured rather than assumed.

        The difference between this and :meth:`params` is the whole point: a
        run states the SNR it *got*, not the one it asked for.
        """
        return dict(self._achieved)

    def _record(self, **values: float) -> None:
        self._achieved.update(values)

    def _accumulate(self, key: str, value: float) -> None:
        self._achieved[key] = self._achieved.get(key, 0.0) + value

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.key})"


@dataclass(slots=True)
class RunningMean:
    """Streaming mean and spread, for achieved values measured per frame."""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = field(default=float("inf"))
    maximum: float = field(default=float("-inf"))

    def add(self, value: float) -> None:
        if value != value or value in (float("inf"), float("-inf")):
            return
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        variance = self.total_sq / self.count - self.mean**2
        return variance**0.5 if variance > 0 else 0.0

    def as_record(self, prefix: str) -> dict[str, float]:
        if not self.count:
            return {}
        return {
            f"{prefix}_mean": self.mean,
            f"{prefix}_std": self.std,
            f"{prefix}_min": self.minimum,
            f"{prefix}_max": self.maximum,
        }
