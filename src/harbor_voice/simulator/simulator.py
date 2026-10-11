"""The universal simulator.

::

    L5 POLICY        when to speak
         |
    CONTENT SOURCE   llm | script | corpus
         |
    SYNTHESIS        L1 persona + L2 speech   -> CLEAN PCM, hashed and kept
         |
    DEGRADATION      L3 environment + L4 channel  -> TX frames, achieved SNR
         |
    transport.tx()

Three duties beyond "produce audio":

**Reproducible.** Seeded per axis, so content can be held fixed while the
channel varies — the experiment the whole benchmark exists to run.

**Gradeable.** EVA's quiet innovation is that two of its three ``validation/``
metrics grade the *simulator*, not the agent. A run whose caller drifted out of
persona is not evidence about the agent; it is ``measurement_invalid``. The
:class:`SimulatorFidelity` record carries what a grader needs to decide that.

**Self-declaring.** It states what it can do, so a task activating a layer it
cannot serve is a load error rather than a silent no-op.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import override

from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.providers import TTSProvider
from harbor_voice.ids import Digest
from harbor_voice.ids import PartyId
from harbor_voice.perturbation.base import SeedSet
from harbor_voice.perturbation.conversation import Probe
from harbor_voice.perturbation.conversation import ProbeKind
from harbor_voice.perturbation.plan import PerturbationPipeline
from harbor_voice.simulator.content import Content
from harbor_voice.simulator.content import ContentSource
from harbor_voice.simulator.content import SourceKind


class SimulatorError(RuntimeError):
    """The simulator was asked for something it cannot serve."""


@dataclass(frozen=True, slots=True)
class Utterance:
    """One thing the caller said, clean and as transmitted.

    Both sides are kept because a measurement needs both: ``said`` is the
    reference for WER, ``frames`` is what the agent actually heard, and
    ``clean_digest`` addresses the undegraded audio.
    """

    index: int
    party: PartyId
    said: str
    """The SAID side of said-vs-heard."""

    frames: tuple[AudioFrame, ...]
    clean_digest: Digest | None = None
    intent: str | None = None
    source_kind: SourceKind = SourceKind.SCRIPT
    prerecorded: bool = False
    probe: ProbeKind | None = None
    """Set when this is an L5 interjection rather than a turn of content.

    Explicit rather than a key in :attr:`metadata`, because three metrics and
    the floor's bid decision all branch on it, and a stringly-typed dict entry
    is one typo away from a selectivity score that silently reads zero.
    """

    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def audio_duration_s(self) -> float:
        return sum(f.duration_s for f in self.frames)

    @property
    def is_degraded(self) -> bool:
        """Whether a perturbation plan touched this utterance."""
        return any(f.clean_ref is not None for f in self.frames)

    @property
    def is_probe(self) -> bool:
        return self.probe is not None


@dataclass(frozen=True, slots=True)
class SimulatorFidelity:
    """What a grader needs to decide whether the caller behaved.

    Deliberately **not** a score. The simulator reports facts; a judge decides
    whether they amount to a run worth keeping. ``in_persona`` stays ``None``
    until something measures it, rather than defaulting to a confident 1.0 —
    an unmeasured judgment has no agreement to report, and a fabricated
    default would hide exactly that.
    """

    utterances: int
    deterministic_source: bool
    source_kind: SourceKind
    prerecorded_fraction: float
    in_persona: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_measured(self) -> bool:
        return self.in_persona is not None


class UniversalSimulator:
    """Produces the caller's audio: content, synthesis, then degradation.

    Args:
        party: Who this simulator speaks as.
        content: Where the words come from.
        tts: Synthesis slot. Unused for pre-recorded corpus audio.
        pipeline: Perturbation layers. Defaults to the ``clean`` profile.
        seeds: Layered seeds.
        fmt: Format of produced audio.
        clock: Session clock, for ``t_produced``.
    """

    def __init__(
        self,
        *,
        party: PartyId,
        content: ContentSource,
        tts: TTSProvider | None = None,
        pipeline: PerturbationPipeline | None = None,
        seeds: SeedSet | None = None,
        fmt: AudioFormat = CANONICAL,
        clock: SessionClock | None = None,
    ) -> None:
        self._party = party
        self._content = content
        self._tts = tts
        self._seeds = seeds or SeedSet.derive(0)
        self._pipeline = pipeline or PerturbationPipeline(seeds=self._seeds)
        self._fmt = fmt
        self._clock = clock or SessionClock()

        self._index = 0
        self._seq = 0
        self._utterances: list[Utterance] = []

    # ------------------------------------------------------------- declared

    @property
    def party(self) -> PartyId:
        return self._party

    @property
    def pipeline(self) -> PerturbationPipeline:
        return self._pipeline

    @property
    def utterances(self) -> tuple[Utterance, ...]:
        return tuple(self._utterances)

    @property
    def capabilities(self) -> Capabilities:
        """What this simulator can serve. Enforced at load."""
        layers = {p.layer for p in self._pipeline.perturbations}
        if not self._pipeline.conversation.is_empty:
            layers.add(Layer.CONVERSATION)
        if self._tts is not None:
            layers.add(Layer.PERSONA)
        return Capabilities(
            duplex=frozenset({Duplex.HALF, Duplex.FULL}),
            pipeline=Pipeline.CASCADE,
            turn_control=TurnControl.HARNESS_VAD,
            layers=frozenset(layers),
            audio_in=self._fmt,
            audio_out=self._fmt,
        )

    # -------------------------------------------------------------- running

    def reset(self) -> None:
        """Rewind content, perturbations and counters for a new run."""
        self._content.reset()
        self._pipeline.reset()
        self._index = 0
        self._seq = 0
        self._utterances.clear()

    async def next_utterance(self, heard: Sequence[str] = ()) -> Utterance | None:
        """Produce the caller's next turn, or ``None`` when the script ends."""
        content = await self._content.next(heard)
        if content is None:
            return None

        clean_pcm = await self._audio_for(content)
        clean_digest = self._pipeline.clean_store.put(clean_pcm) if clean_pcm else None
        frames = self._frame_up(clean_pcm)
        degraded = self._pipeline.process_all(frames)

        utterance = Utterance(
            index=self._index,
            party=self._party,
            said=content.text,
            frames=degraded,
            clean_digest=clean_digest,
            intent=content.intent,
            source_kind=self._content.kind,
            prerecorded=content.is_prerecorded,
            metadata=dict(content.metadata),
        )
        self._index += 1
        self._utterances.append(utterance)
        return utterance

    async def say(
        self,
        text: str,
        *,
        probe: ProbeKind | None = None,
        intent: str | None = None,
    ) -> Utterance:
        """Produce an utterance for text the content source did not supply.

        An L5 probe is the caller speaking, so it goes through the same
        synthesis and the same degradation as everything else: a backchannel
        that arrived on a clean line while the turn it interrupts arrived over
        mu-law would be a different caller, and the agent would be right to
        treat it differently.
        """
        clean_pcm = await self._synthesize(text)
        clean_digest = self._pipeline.clean_store.put(clean_pcm) if clean_pcm else None
        degraded = self._pipeline.process_all(self._frame_up(clean_pcm))

        utterance = Utterance(
            index=self._index,
            party=self._party,
            said=text,
            frames=degraded,
            clean_digest=clean_digest,
            intent=intent,
            source_kind=self._content.kind,
            probe=probe,
        )
        self._index += 1
        self._utterances.append(utterance)
        return utterance

    async def probe_utterance(self, probe: Probe) -> Utterance:
        """Synthesise one scheduled L5 interjection."""
        return await self.say(probe.text, probe=probe.kind, intent=probe.kind.value)

    async def _audio_for(self, content: Content) -> bytes:
        """Pre-recorded audio wins; otherwise synthesise."""
        if content.audio is not None:
            self._fmt.validate_buffer(len(content.audio))
            return content.audio
        return await self._synthesize(content.text)

    async def _synthesize(self, text: str) -> bytes:
        if self._tts is None:
            raise SimulatorError(
                f"content source {self._content.kind.value!r} yielded text but "
                f"no TTS is configured; supply one or use a corpus source"
            )
        chunks = [
            chunk async for chunk in self._tts.synthesize(text, party=self._party)
        ]
        return b"".join(chunks)

    def _frame_up(self, pcm: bytes) -> tuple[AudioFrame, ...]:
        """Split a buffer into canonical frames, stamped at synthesis time.

        One ``t_produced`` for the whole utterance, because batch synthesis
        brought it into existence at one instant. A per-frame schedule would
        claim frames exist before they do, and the monotonic guard in
        :class:`~harbor_voice.audio.frame.AudioFrame` rejects transmitting one.
        """
        if not pcm:
            return ()
        size = self._fmt.frame_bytes()
        produced_at = self._clock.now()
        frames: list[AudioFrame] = []
        for start in range(0, len(pcm), size):
            chunk = pcm[start : start + size]
            if len(chunk) % self._fmt.bytes_per_frame_slot:
                chunk = chunk[
                    : len(chunk) - (len(chunk) % self._fmt.bytes_per_frame_slot)
                ]
            if not chunk:
                break
            frames.append(
                AudioFrame(
                    pcm=chunk,
                    seq=self._seq,
                    party=self._party,
                    t_produced=produced_at,
                    fmt=self._fmt,
                )
            )
            self._seq += 1
        return tuple(frames)

    # ------------------------------------------------------------- fidelity

    def fidelity(self, in_persona: float | None = None) -> SimulatorFidelity:
        """Facts a grader uses to decide whether the run is evidence."""
        total = len(self._utterances)
        prerecorded = sum(1 for u in self._utterances if u.prerecorded)
        notes: list[str] = []
        if not self._content.is_deterministic:
            notes.append(
                "content source is non-deterministic; this run is not replayable "
                "from its seed alone"
            )
        if in_persona is None:
            notes.append("in_persona has not been measured for this run")
        return SimulatorFidelity(
            utterances=total,
            deterministic_source=self._content.is_deterministic,
            source_kind=self._content.kind,
            prerecorded_fraction=(prerecorded / total) if total else 0.0,
            in_persona=in_persona,
            notes=tuple(notes),
        )

    @override
    def __repr__(self) -> str:
        return (
            f"UniversalSimulator(party={self._party!r}, "
            f"source={self._content.kind.value}, "
            f"layers={len(self._pipeline.perturbations)}, "
            f"utterances={self._index})"
        )
