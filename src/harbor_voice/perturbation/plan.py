"""The perturbation plan and pipeline.

Three invariants live here.

**The clean reference is never discarded.** A chain written as
``for effect in effects: audio = effect(audio)`` overwrites the clean PCM in a
local. The consequence is not subtle: with no reference signal there is no
measurable SNR, no degradation ladder, and no said-vs-heard WER. Pairing the
*clean transcript* with the *degraded audio* in one message also leaves a
text judge is structurally blind to the perturbation under test. Here every
perturbed frame carries its clean twin's digest and the twin itself is kept.

**The plan is written, not just declared.** A plan type that is
defined but never constructed records nothing.
:meth:`PerturbationPipeline.as_record` materialises every active layer, its
parameters, its seeds, and the values actually **achieved** — so a run can
state its own conditions and two runs can be compared.

**Order is a correctness property.** Layers apply in physical order:
persona, speech, environment, then channel. Noise enters the room before the
room goes down the wire; reversing them produces audio no telephone carried.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import override

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import LayerKey
from harbor_voice.ids import Digest
from harbor_voice.ids import PlanId
from harbor_voice.ids import sha256_digest
from harbor_voice.perturbation.base import ApplyContext
from harbor_voice.perturbation.base import Perturbation
from harbor_voice.perturbation.base import SeedSet
from harbor_voice.perturbation.conversation import ConversationPlan
from harbor_voice.perturbation.encoder import WireEncoder
from harbor_voice.perturbation.jitter import FramePerturbation
from harbor_voice.perturbation.persona import PersonaPlan
from harbor_voice.perturbation.speech import SpeechPlan
from harbor_voice.perturbation.world import WorldPlan

_LAYER_ORDER: dict[Layer, int] = {
    Layer.PERSONA: 0,
    Layer.SPEECH: 1,
    Layer.ENVIRONMENT: 2,
    Layer.CHANNEL: 3,
    Layer.CONVERSATION: 4,
    Layer.WORLD: 5,
}


class CleanReferenceError(RuntimeError):
    """The clean/transmitted invariant was broken."""


@dataclass(slots=True)
class CleanStore:
    """Holds clean audio by digest, so a perturbed frame stays comparable.

    In-memory for now. A content-addressed blob store behind the artifact
    tiers can replace it — the interface is the same, which is
    why it is a seam rather than a dict inline in the pipeline.
    """

    _blobs: dict[Digest, bytes] = field(default_factory=dict)

    def put(self, pcm: bytes) -> Digest:
        digest = sha256_digest(pcm)
        self._blobs[digest] = pcm
        return digest

    def get(self, digest: Digest) -> bytes:
        try:
            return self._blobs[digest]
        except KeyError as exc:
            raise CleanReferenceError(
                f"no clean reference for {digest}; the degraded frame cannot "
                f"be compared to anything"
            ) from exc

    def __len__(self) -> int:
        return len(self._blobs)

    @property
    def total_bytes(self) -> int:
        return sum(len(b) for b in self._blobs.values())


@dataclass(frozen=True, slots=True)
class PlanRecord:
    """The plan as it lands in the trajectory."""

    plan_id: PlanId
    seeds: dict[str, int]
    layers: dict[str, dict[str, Any]]
    frames_processed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": str(self.plan_id),
            "seeds": dict(self.seeds),
            "layers": {k: dict(v) for k, v in self.layers.items()},
            "frames_processed": self.frames_processed,
        }


class PerturbationPipeline:
    """Applies layers in physical order, keeping the clean reference.

    Args:
        perturbations: Active layers, in any order; they are sorted.
        seeds: Seed set for the run. Every perturbation is reset with it.
        plan_id: Identifier recorded on each frame.
        fmt: Format the frames are in.
        clean_store: Where clean twins are kept. One is created if omitted.
        conversation: L5 layers. They sit *beside* the signal chain, never in
            it — L5 decides when the caller speaks and transforms no audio, so
            putting it in the apply loop with an identity transform would put
            a layer in the plan record that never touched a sample.
        persona: L1 — who speaks. Lands at synthesis and at the content
            source, both of which happen before there is audio.
        speech: L2's **text** half. Its signal half (loudness, speaking rate,
            muffling) are ordinary perturbations and go in ``perturbations``.
        encoder: The wire format, applied last — see
            :mod:`~harbor_voice.perturbation.encoder`. The only stage allowed to
            change rate and encoding; clean and transmitted then align by time.

    Four insertion points, because the six layers genuinely have four:
    content and voice (L1, L2 text), the signal chain (L2 signal, L3, L4),
    and the floor (L5). Collapsing them into one list would need three
    identity transforms and a plan record that lies about what touched the
    audio.
    """

    def __init__(
        self,
        perturbations: Iterable[Perturbation | FramePerturbation] = (),
        *,
        seeds: SeedSet | None = None,
        plan_id: PlanId = PlanId("plan"),
        fmt: AudioFormat = CANONICAL,
        clean_store: CleanStore | None = None,
        conversation: ConversationPlan | None = None,
        persona: PersonaPlan | None = None,
        speech: SpeechPlan | None = None,
        world: WorldPlan | None = None,
        encoder: WireEncoder | None = None,
    ) -> None:
        ordered = sorted(perturbations, key=lambda p: (_LAYER_ORDER[p.layer], p.name))
        # Split by shape rather than asking the caller to. A frame-stream
        # layer runs after every per-frame one: a codec runs in the handset
        # and the network delays what it produced, never the other way round.
        self._perturbations = tuple(
            p for p in ordered if not isinstance(p, FramePerturbation)
        )
        self._stream = tuple(p for p in ordered if isinstance(p, FramePerturbation))
        self._seeds = seeds or SeedSet.derive(0)
        self._plan_id = plan_id
        self._fmt = fmt
        self._clean = clean_store or CleanStore()
        self._conversation = conversation or ConversationPlan()
        self._persona = persona or PersonaPlan()
        self._speech = speech or SpeechPlan()
        self._world = world or WorldPlan()
        self._encoder = encoder
        if encoder is not None and not encoder.accepts(fmt):
            raise ValueError(f"encoder {encoder.name} cannot take {fmt} frames")
        self._ctx = ApplyContext(fmt=fmt)
        self._frames = 0
        self.reset()

    # ------------------------------------------------------------ identity

    @property
    def perturbations(self) -> tuple[Perturbation, ...]:
        """Active layers, in the order they are applied."""
        return self._perturbations

    @property
    def conversation(self) -> ConversationPlan:
        """The L5 layers. Empty unless a profile switched them on."""
        return self._conversation

    @property
    def persona(self) -> PersonaPlan:
        """The L1 axes — voice at synthesis, behaviour at the content source."""
        return self._persona

    @property
    def speech(self) -> SpeechPlan:
        """L2's text half. The signal half lives in :attr:`perturbations`."""
        return self._speech

    @property
    def world(self) -> WorldPlan:
        """L6 — latency, errors and outages on the tool boundary."""
        return self._world

    @property
    def stream(self) -> tuple[FramePerturbation, ...]:
        """L4 layers that transform the frame stream, not a frame.

        Separate because they are not definable on one frame: delay
        *variation* needs a sequence, so :meth:`process` cannot apply them
        and :meth:`process_all` must.
        """
        return self._stream

    @property
    def active_layers(self) -> frozenset[LayerKey]:
        """What a task has switched on. Feeds metric applicability."""
        return (
            frozenset(p.key for p in self._perturbations)
            | frozenset(p.key for p in self._stream)
            | self._world.active_layers
            | self._conversation.active_layers
            | self._persona.active_layers
            | self._speech.active_layers
        )

    @property
    def is_clean(self) -> bool:
        """Whether anything here degrades the *signal*.

        L5 deliberately does not count. A conversation-only profile such as
        ``overlap`` transmits the caller's audio untouched, so there is no
        clean/transmitted distinction to store and ``clean_tx_parity`` has
        nothing to check. Folding L5 in here would make every barge-in run
        store a redundant copy of every frame and then verify it against
        itself.
        """
        return not self._perturbations and not self._stream and self._encoder is None

    @property
    def is_unperturbed(self) -> bool:
        """Whether *no* layer of any kind is active — the ``clean`` profile."""
        return not self.active_layers

    @property
    def encoder(self) -> WireEncoder | None:
        return self._encoder

    @property
    def output_format(self) -> AudioFormat:
        """What :meth:`process` returns: the encoder's format, if there is one."""
        return self._encoder.output_format if self._encoder else self._fmt

    @property
    def can_stream(self) -> bool:
        """Whether frames can be processed one at a time.

        A frame-stream layer (jitter) schedules delays across a whole stream
        and reorders by arrival; streaming it needs a playout-buffer model
        that does not exist yet. Until then a pipeline with one refuses to
        stream rather than silently skipping the stage.
        """
        return not self._stream

    @property
    def clean_store(self) -> CleanStore:
        return self._clean

    @property
    def seeds(self) -> SeedSet:
        return self._seeds

    # ------------------------------------------------------------- running

    def reset(self) -> None:
        """Re-seed every layer and clear per-run state."""
        self._ctx = ApplyContext(fmt=self._fmt)
        self._frames = 0
        for perturbation in (*self._perturbations, *self._stream):
            perturbation.reset(self._seeds)
        if self._encoder is not None:
            self._encoder.reset(self._seeds)
        self._conversation.reset(self._seeds)
        self._world.reset(self._seeds)
        self._persona.reset()
        self._speech.reset(self._seeds)

    def process(self, frame: AudioFrame, *, is_speech: bool = True) -> AudioFrame:
        """Degrade one frame, retaining its clean twin.

        The clean reference is stored **before** anything touches the audio,
        which is the one ordering that cannot be got wrong later.
        """
        if self.is_clean:
            # Nothing degraded it, so there is no clean/transmitted distinction
            # to record. Marking the frame anyway would make `is_degraded` lie
            # and would store a redundant copy of every frame in the run.
            self._ctx.is_speech = is_speech
            self._ctx.advance(frame.duration_s)
            self._frames += 1
            return frame

        clean_digest = self._clean.put(frame.pcm)
        self._ctx.is_speech = is_speech

        pcm = frame.pcm
        for perturbation in self._perturbations:
            pcm = perturbation.apply(pcm, self._ctx)
            if len(pcm) != len(frame.pcm):
                raise CleanReferenceError(
                    f"{perturbation.key} changed frame length "
                    f"{len(frame.pcm)} -> {len(pcm)}; clean and transmitted "
                    f"must stay alignable"
                )

        self._ctx.advance(frame.duration_s)
        self._frames += 1
        fmt = frame.fmt
        if self._encoder is not None:
            pcm = self._encoder.encode(pcm, frame.fmt)
            fmt = self._encoder.output_format
        degraded = AudioFrame(
            pcm=pcm,
            seq=frame.seq,
            party=frame.party,
            t_produced=frame.t_produced,
            fmt=fmt,
            t_tx=frame.t_tx,
            t_rx=frame.t_rx,
            t_voiced=frame.t_voiced,
        )
        return degraded.with_clean_ref(clean_digest, self._plan_id)

    def process_all(
        self, frames: Sequence[AudioFrame], *, is_speech: bool = True
    ) -> tuple[AudioFrame, ...]:
        """Degrade a whole utterance, per frame and then as a stream.

        The stream stage can **return fewer frames than it was given**: a
        frame that misses its playout deadline is discarded, which is loss
        caused by delay. That is why jitter cannot live in ``process`` — a
        per-frame contract has nowhere to express "this one never arrived".
        """
        out = tuple(self.process(f, is_speech=is_speech) for f in frames)
        for stream in self._stream:
            out = stream.apply(out)
        return out

    # -------------------------------------------------------------- record

    def as_record(self) -> PlanRecord:
        """Materialise the plan."""
        return PlanRecord(
            plan_id=self._plan_id,
            seeds=self._seeds.as_record(),
            layers={
                **{
                    str(p.key): {"params": p.params(), "achieved": p.achieved()}
                    for p in self._perturbations
                },
                **self._conversation.as_record(),
                **self._persona.as_record(),
                **self._speech.as_record(),
                **self._world.as_record(),
                **{
                    str(p.key): {"params": p.params(), "achieved": p.achieved()}
                    for p in self._stream
                },
                **(
                    {
                        str(self._encoder.key): {
                            "params": self._encoder.params(),
                            "achieved": self._encoder.achieved(),
                        }
                    }
                    if self._encoder is not None
                    else {}
                ),
            },
            frames_processed=self._frames,
        )

    def verify(self, frame: AudioFrame) -> None:
        """Raise unless ``frame`` can still be compared to its clean twin.

        An enforced invariant: clean and
        transmitted must be equal length and share one transport end, and a
        turn missing either view makes the call ``measurement_invalid``.
        """
        if self.is_clean:
            return
        if frame.clean_ref is None:
            raise CleanReferenceError(
                f"frame seq={frame.seq} was perturbed but carries no clean_ref"
            )
        clean = self._clean.get(frame.clean_ref)
        if self._encoder is None:
            if len(clean) != len(frame.pcm):
                raise CleanReferenceError(
                    f"frame seq={frame.seq}: clean is {len(clean)} bytes, "
                    f"transmitted is {len(frame.pcm)}"
                )
            return
        # Across an encoder the formats differ, so alignment is by time. A
        # streamed resampler may land a sample early or late on any one frame
        # and settles across the stream, hence one output sample of slack.
        clean_s = self._fmt.duration_s(len(clean))
        slack = 1.0 / frame.fmt.sample_rate_hz
        if abs(frame.duration_s - clean_s) > slack + 1e-9:
            raise CleanReferenceError(
                f"frame seq={frame.seq}: clean is {clean_s * 1000:.3f} ms, "
                f"transmitted is {frame.duration_s * 1000:.3f} ms"
            )

    @override
    def __repr__(self) -> str:
        signal = [str(p.key) for p in (*self._perturbations, *self._stream)]
        beside = self.active_layers - {
            p.key for p in (*self._perturbations, *self._stream)
        }
        layers = ", ".join(signal + sorted(str(k) for k in beside)) or "clean"
        return f"PerturbationPipeline({layers}, seed={self._seeds.root})"
