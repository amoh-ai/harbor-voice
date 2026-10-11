"""The unit on the wire: a timestamped frame.

A transport of ``send_audio(whole_utterance)`` -> ``recv_audio()`` would make
jitter architecturally impossible (there is no inter-packet interval to
perturb), force every full-duplex provider into a bespoke background loop, and
leave barge-in unimplementable. Frames avoid all three.

A frame also carries :attr:`AudioFrame.clean_ref`, the digest of its
pre-perturbation twin. Without a reference signal there is no measurable SNR,
no degradation ladder and no said-vs-heard WER — which is exactly the hole a
one-way destructive effect chain leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import override

from harbor_voice.audio.clock import Clock
from harbor_voice.audio.clock import ClockError
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.ids import Digest
from harbor_voice.ids import PartyId
from harbor_voice.ids import PlanId

# ``t_produced`` is when the content came into EXISTENCE, not its offset in the
# media timeline. Batch synthesis therefore stamps every frame of an utterance
# with one instant; streaming synthesis stamps each as it is produced. A
# synthetic schedule (t0 + i * frame_ms) is neither, and claims frames exist
# before they do — the monotonic guard rejects transmitting one, which is how
# this was found. Media position is implicit in frame order and duration.

_CLOCK_FIELD: dict[Clock, str] = {
    Clock.PRODUCED: "t_produced",
    Clock.TX: "t_tx",
    Clock.RX: "t_rx",
    Clock.VOICED: "t_voiced",
}


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """One frame of audio, with every clock it has passed so far.

    Frames are immutable. Stamping a later clock returns a new frame via
    :meth:`stamp`, so a frame handed to a recorder cannot be mutated underneath
    it by the transport.
    """

    pcm: bytes
    seq: int
    party: PartyId
    t_produced: float
    fmt: AudioFormat = CANONICAL
    t_tx: float | None = None
    t_rx: float | None = None
    t_voiced: float | None = None
    clean_ref: Digest | None = None
    plan_ref: PlanId | None = None

    def __post_init__(self) -> None:
        self.fmt.validate_buffer(len(self.pcm))
        if self.seq < 0:
            raise ValueError(f"seq must be non-negative, got {self.seq}")
        self._check_monotonic()

    def _check_monotonic(self) -> None:
        """Clocks a frame has passed must be non-decreasing in physical order."""
        ordered = [
            (Clock.PRODUCED, self.t_produced),
            (Clock.TX, self.t_tx),
            (Clock.RX, self.t_rx),
            (Clock.VOICED, self.t_voiced),
        ]
        seen = [(c, t) for c, t in ordered if t is not None]
        for (prev_clock, prev_t), (clock, t) in zip(seen, seen[1:], strict=False):
            if t < prev_t:
                raise ClockError(
                    f"frame seq={self.seq} party={self.party}: "
                    f"{clock.value}={t} precedes {prev_clock.value}={prev_t}"
                )

    @property
    def duration_s(self) -> float:
        """Audio duration this frame carries."""
        return self.fmt.duration_s(len(self.pcm))

    @property
    def is_clean(self) -> bool:
        """Whether this frame is its own clean reference (unperturbed)."""
        return self.clean_ref is None and self.plan_ref is None

    def at(self, clock: Clock) -> float | None:
        """Timestamp in ``clock``'s domain, or ``None`` if not yet stamped."""
        return getattr(self, _CLOCK_FIELD[clock])

    def stamp(
        self,
        clock: Clock,
        t: float,
        *,
        overwrite: bool = False,
    ) -> AudioFrame:
        """Return a copy with ``clock`` set to ``t``.

        Args:
            clock: Which domain to stamp. ``PRODUCED`` is set at construction
                and cannot be stamped again.
            t: Timestamp, in the session clock's base.
            overwrite: Allow replacing an already-set value. Off by default so
                a double-stamp is a bug report rather than a silent overwrite.

        Raises:
            ClockError: if already stamped and ``overwrite`` is false, if
                stamping ``PRODUCED``, or if the result would be non-monotonic.
        """
        if clock is Clock.PRODUCED:
            raise ClockError(
                "t_produced is set at construction and cannot be re-stamped"
            )
        field = _CLOCK_FIELD[clock]
        if not overwrite and getattr(self, field) is not None:
            raise ClockError(
                f"frame seq={self.seq} already has {field}="
                f"{getattr(self, field)}; pass overwrite=True to replace"
            )
        return replace(self, **{field: t})

    def with_clean_ref(self, digest: Digest, plan: PlanId) -> AudioFrame:
        """Return a copy marked as the perturbed twin of ``digest``.

        Called by the perturbation pipeline at the point of degradation, so the
        clean signal is addressable for the lifetime of the trajectory.
        """
        return replace(self, clean_ref=digest, plan_ref=plan)

    @override
    def __repr__(self) -> str:
        stamped = ",".join(c.value for c in Clock if self.at(c) is not None)
        return (
            f"AudioFrame(seq={self.seq}, party={self.party!r}, "
            f"{len(self.pcm)}B, {self.duration_s * 1000:.0f}ms, "
            f"clocks=[{stamped}])"
        )
