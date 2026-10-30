"""What a party can do, declared and **enforced**.

An adapter can declare ``streaming_transcripts=True``, ``native_vad=True`` and
``pcm16/48000``, have a capability gate trust all three, and then hang forever
polling an attribute nothing ever sets. **An advisory capability flag is not a
capability model.**

Three rules:

1. A task activating something a party cannot serve is a **load error**, not a
   silent no-op. The rule: *a key that
   loads and changes nothing is worse than a key that is refused*.
2. A metric whose preconditions are unmet reports **not_applicable**, never
   ``0``. A half-duplex run has no barge-in score; it has no barge-in
   *measurement*.
3. A capability the conformance suite has not verified is **unattested** and
   cannot gate a published number.

Internal consistency is checked at construction, which is where that failure
is caught: a speech-to-speech party **cannot** claim
``asr_final``, because by definition it does not expose one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated
from typing import Literal
from typing import Self
from typing import override

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.stages import CASCADE_STAGES
from harbor_voice.audio.stages import S2S_STAGES
from harbor_voice.audio.stages import Stage

# --------------------------------------------------------------------- axes


class Duplex(StrEnum):
    """Whether both parties may have audio on the wire at once.

    A property of the *floor policy*, not of the transport — the transport
    is always bidirectional. A party declares which floors it can honour.
    """

    HALF = "half"
    FULL = "full"


class Pipeline(StrEnum):
    """How a party turns audio into audio. Orthogonal to :class:`Duplex`."""

    CASCADE = "cascade"
    """STT -> LLM -> TTS, composed by harbor-voice, so every seam is observable."""

    S2S = "s2s"
    """Native audio in, native audio out. Two observable seams, not seven."""


class TurnControl(StrEnum):
    """Who decides a turn has ended."""

    HARNESS_VAD = "harness_vad"
    NATIVE_VAD = "native_vad"
    PROVIDER_ENDPOINTING = "provider_endpointing"


class OptionalOp(StrEnum):
    """Transport operations beyond the four every transport must implement.

    Found by ``getattr`` at call time, the real contract would be invisible to
    anyone writing a subclass. Declaring them
    here makes the surface part of the capability matrix, which means a
    caller can ask before calling and the conformance suite can verify the
    answer.
    """

    SEND_TEXT = "send_text"
    SEND_DTMF = "send_dtmf"
    CONTROL_STREAM = "control_stream"
    RECONNECT = "reconnect"
    EVENT_STREAM = "event_stream"
    """Agent audio, its own transcript and provider VAD as one ordered stream."""
    TRUNCATE = "truncate"
    """Tell the provider how much of an utterance was heard before a barge-in."""


class Layer(StrEnum):
    """Perturbation layers, applied in physical order."""

    PERSONA = "L1"
    SPEECH = "L2"
    ENVIRONMENT = "L3"
    CHANNEL = "L4"
    CONVERSATION = "L5"
    WORLD = "L6"


_LAYER_ALIASES: dict[str, Layer] = {
    **{layer.value.lower(): layer for layer in Layer},
    **{layer.name.lower(): layer for layer in Layer},
}


class CapabilityError(ValueError):
    """A capability declaration is internally inconsistent."""


class RequirementError(ValueError):
    """A task demands something no party can serve. Rule 1 — a load error."""


# ----------------------------------------------------------------- layer key


@dataclass(frozen=True, slots=True)
class LayerKey:
    """A specific perturbation within a layer, e.g. ``L5.interruption``."""

    layer: Layer
    name: str

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Parse ``L5.interruption`` or ``conversation.interruption``.

        Raises:
            ValueError: if the layer is unknown or the form is wrong.
        """
        head, _, tail = raw.partition(".")
        if not tail:
            raise ValueError(
                f"layer key {raw!r} must be '<layer>.<name>', e.g. 'L3.background_bed'"
            )
        layer = _LAYER_ALIASES.get(head.strip().lower())
        if layer is None:
            raise ValueError(
                f"unknown layer {head!r} in {raw!r}; "
                f"expected one of {sorted(_LAYER_ALIASES)}"
            )
        return cls(layer=layer, name=tail.strip())

    @override
    def __str__(self) -> str:
        return f"{self.layer.value}.{self.name}"


# --------------------------------------------------------------- capabilities


class Capabilities(BaseModel):
    """What one party can do. Frozen, declared in TOML, stored in the trajectory.

    Note:
        Consistency failures raise :class:`CapabilityError` inside the validator,
        which pydantic re-raises wrapped in ``pydantic.ValidationError``. Catch
        ``ValidationError``; the :class:`CapabilityError` message is preserved
        inside it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    duplex: frozenset[Duplex] = Field(
        description="Floor policies this party can honour. At least one."
    )
    pipeline: Pipeline
    turn_control: TurnControl

    exposes_asr: bool = Field(
        default=False,
        description="Exposes its own recognised text. Gates said-vs-heard WER.",
    )
    exposes_partials: bool = Field(
        default=False,
        description="Emits incremental transcripts. Gates after-words interruption.",
    )
    exposes_stages: frozenset[Stage] = Field(
        default=frozenset(),
        description="Stage marks this party actually emits.",
    )

    can_interrupt: bool = False
    can_be_interrupted: bool = False

    layers: frozenset[Layer] = Field(
        default=frozenset(),
        description="Perturbation layers this party can apply (simulators only).",
    )
    exclusive_groups: tuple[frozenset[str], ...] = Field(
        default=(),
        description=(
            "Sets of perturbations that cannot be combined. EVA's accent / "
            "behaviour / language each claim the same provider voice slot; that "
            "is a provider limitation and belongs here as data, not in code."
        ),
    )

    optional_ops: frozenset[OptionalOp] = Field(
        default=frozenset(),
        description=(
            "Transport operations this party supports beyond open/tx/rx/close. "
            "Declared, not duck-typed."
        ),
    )

    audio_in: AudioFormat = CANONICAL
    audio_out: AudioFormat = CANONICAL
    max_session_s: Annotated[float, Field(gt=0)] | None = None

    attested: frozenset[str] = Field(
        default=frozenset(),
        description=(
            "Capability names the conformance suite has verified. A claim "
            "absent here is usable but cannot gate a published number — rule 3."
        ),
    )

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> Self:
        if not self.duplex:
            raise CapabilityError("a party must declare at least one duplex mode")

        allowed = CASCADE_STAGES if self.pipeline is Pipeline.CASCADE else S2S_STAGES
        impossible = self.exposes_stages - allowed
        if impossible:
            raise CapabilityError(
                f"pipeline={self.pipeline.value} cannot expose "
                f"{sorted(s.value for s in impossible)}; a speech-to-speech party "
                f"composes no such seam. Allowed: {sorted(s.value for s in allowed)}"
            )

        if self.exposes_partials and not self.exposes_asr:
            raise CapabilityError(
                "exposes_partials=True requires exposes_asr=True — "
                "partial transcripts are ASR output"
            )
        if self.exposes_partials and Stage.ASR_PARTIAL not in self.exposes_stages:
            raise CapabilityError(
                "exposes_partials=True requires Stage.ASR_PARTIAL in exposes_stages; "
                "a transport cannot claim partials it never emits"
            )
        if self.exposes_asr and Stage.ASR_FINAL not in self.exposes_stages:
            raise CapabilityError(
                "exposes_asr=True requires Stage.ASR_FINAL in exposes_stages"
            )

        unknown = self.attested - set(_ATTESTABLE)
        if unknown:
            raise CapabilityError(
                f"cannot attest unknown capabilities {sorted(unknown)}; "
                f"attestable: {sorted(_ATTESTABLE)}"
            )
        return self

    def is_attested(self, capability: str) -> bool:
        """Whether the conformance suite has verified ``capability`` (rule 3)."""
        return capability in self.attested

    def supports_duplex(self, mode: Duplex) -> bool:
        return mode in self.duplex

    def supports(self, op: OptionalOp) -> bool:
        """Whether this party implements ``op`` (declared, not probed)."""
        return op in self.optional_ops


_ATTESTABLE: frozenset[str] = frozenset(
    {
        "exposes_asr",
        "exposes_partials",
        "can_interrupt",
        "can_be_interrupted",
        "audio_in",
        "audio_out",
        "turn_control",
        "send_text",
        "send_dtmf",
        "control_stream",
        "reconnect",
    }
)
"""Claims the conformance suite knows how to verify."""


# --------------------------------------------------------------- requirements


class Requirement(BaseModel):
    """What a task demands of the party under test — ``voice.toml [requires]``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    duplex: Duplex | None = None
    pipeline: Pipeline | Literal["any"] = "any"
    can_interrupt: bool | None = None
    can_be_interrupted: bool | None = None
    exposes_asr: bool | None = None
    exposes_partials: bool | None = None
    min_session_s: Annotated[float, Field(gt=0)] | None = None


def check_requirements(
    requirement: Requirement,
    capabilities: Capabilities,
    *,
    active_layers: frozenset[LayerKey] = frozenset(),
    party: str = "party",
) -> list[str]:
    """Rule 1. Return every violation; empty means the task may load.

    Returns a list rather than raising on the first problem so ``doctor``
    can print the whole picture in one pass instead of one error per run.
    """
    problems: list[str] = []

    if requirement.duplex is not None and not capabilities.supports_duplex(
        requirement.duplex
    ):
        problems.append(
            f"{party}: task requires duplex={requirement.duplex.value}, "
            f"party declares {sorted(d.value for d in capabilities.duplex)}"
        )

    if (
        requirement.pipeline != "any"
        and capabilities.pipeline is not requirement.pipeline
    ):
        problems.append(
            f"{party}: task requires pipeline={requirement.pipeline}, "
            f"party is {capabilities.pipeline.value}"
        )

    for flag in (
        "can_interrupt",
        "can_be_interrupted",
        "exposes_asr",
        "exposes_partials",
    ):
        wanted = getattr(requirement, flag)
        if wanted is not None and getattr(capabilities, flag) != wanted:
            problems.append(
                f"{party}: task requires {flag}={wanted}, "
                f"party declares {getattr(capabilities, flag)}"
            )

    if (
        requirement.min_session_s is not None
        and capabilities.max_session_s is not None
        and capabilities.max_session_s < requirement.min_session_s
    ):
        problems.append(
            f"{party}: task needs {requirement.min_session_s}s sessions, "
            f"party caps at {capabilities.max_session_s}s"
        )

    problems.extend(check_layers(active_layers, capabilities, party=party))
    return problems


def check_layers(
    active_layers: frozenset[LayerKey],
    capabilities: Capabilities,
    *,
    party: str = "party",
) -> list[str]:
    """Rule 1, for perturbations: undeclared layer or exclusivity breach."""
    problems: list[str] = []

    for key in sorted(active_layers, key=str):
        if key.layer not in capabilities.layers:
            problems.append(
                f"{party}: task activates {key}, but party serves only "
                f"{sorted(layer.value for layer in capabilities.layers)}"
            )

    active_names = {key.name for key in active_layers}
    for group in capabilities.exclusive_groups:
        clash = sorted(active_names & group)
        if len(clash) > 1:
            problems.append(
                f"{party}: {clash} are mutually exclusive for this party "
                f"(declared exclusive group {sorted(group)})"
            )
    return problems


# -------------------------------------------------------------- applicability


class Verdict(StrEnum):
    """Outcome of evaluating a metric's preconditions — rules 2 and 3."""

    SATISFIED = "satisfied"
    NOT_APPLICABLE = "not_applicable"
    UNATTESTED = "unattested"


@dataclass(frozen=True, slots=True)
class Applicability:
    """Whether a metric may report, and why not when it may not."""

    verdict: Verdict
    reasons: tuple[str, ...] = ()

    @property
    def may_report(self) -> bool:
        """Whether the metric should produce a value at all."""
        return self.verdict is not Verdict.NOT_APPLICABLE

    @property
    def may_publish(self) -> bool:
        """Whether the value may gate a published number (rule 3)."""
        return self.verdict is Verdict.SATISFIED


PreconditionKind = Literal["duplex", "pipeline", "cap", "layer", "stage"]
_PRECONDITION_KINDS: tuple[PreconditionKind, ...] = (
    "duplex",
    "pipeline",
    "cap",
    "layer",
    "stage",
)


@dataclass(frozen=True, slots=True)
class Precondition:
    """One condition a metric needs, e.g. ``duplex:full`` or ``layer:L5.interruption``."""

    kind: PreconditionKind
    value: str

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Parse ``<kind>:<value>``.

        Iterating the typed tuple narrows ``kind`` to :data:`PreconditionKind`,
        so this needs no type suppression — see the suppression gate in
        ``AGENTS.md``.
        """
        head, _, value = raw.partition(":")
        head = head.strip().lower()
        value = value.strip()
        if value:
            for kind in _PRECONDITION_KINDS:
                if head == kind:
                    return cls(kind=kind, value=value)
        raise ValueError(
            f"precondition {raw!r} must be '<kind>:<value>' with kind in "
            f"{list(_PRECONDITION_KINDS)}"
        )

    @override
    def __str__(self) -> str:
        return f"{self.kind}:{self.value}"


def evaluate(
    preconditions: tuple[Precondition | str, ...],
    capabilities: Capabilities,
    *,
    active_layers: frozenset[LayerKey] = frozenset(),
    floor: Duplex | None = None,
) -> Applicability:
    """Rules 2 and 3. Decide whether a metric may report, and whether it may publish.

    Args:
        preconditions: What the metric needs; strings are parsed.
        capabilities: The party being measured.
        active_layers: Perturbations actually switched on for this run. A
            selectivity metric exists only because its perturbation does,
            so an inactive layer makes it not applicable.
        floor: The floor policy in force, when it differs from what the party
            could support. A party that *can* do full duplex is still not
            measurable for barge-in under a half-duplex floor.

    Returns:
        ``SATISFIED`` when every condition holds and every capability relied on
        is attested; ``UNATTESTED`` when conditions hold but a relied-on claim
        is unverified; ``NOT_APPLICABLE`` with reasons otherwise.
    """
    parsed = tuple(
        p if isinstance(p, Precondition) else Precondition.parse(p)
        for p in preconditions
    )
    unmet: list[str] = []
    unattested: list[str] = []

    for pre in parsed:
        match pre.kind:
            case "duplex":
                wanted = Duplex(pre.value)
                effective = {floor} if floor is not None else capabilities.duplex
                if wanted not in effective:
                    where = "floor in force" if floor is not None else "party"
                    unmet.append(f"needs duplex={wanted.value}, {where} is not")
            case "pipeline":
                if capabilities.pipeline is not Pipeline(pre.value):
                    unmet.append(
                        f"needs pipeline={pre.value}, "
                        f"party is {capabilities.pipeline.value}"
                    )
            case "cap":
                if not getattr(capabilities, pre.value, False):
                    unmet.append(f"needs {pre.value}, party does not declare it")
                elif pre.value in _ATTESTABLE and not capabilities.is_attested(
                    pre.value
                ):
                    unattested.append(pre.value)
            case "stage":
                if Stage(pre.value) not in capabilities.exposes_stages:
                    unmet.append(f"needs stage {pre.value}, party does not emit it")
            case "layer":
                if LayerKey.parse(pre.value) not in active_layers:
                    unmet.append(f"needs layer {pre.value} active, it is not")

    if unmet:
        return Applicability(Verdict.NOT_APPLICABLE, tuple(unmet))
    if unattested:
        return Applicability(
            Verdict.UNATTESTED,
            tuple(
                f"{name} is declared but not verified by conformance"
                for name in unattested
            ),
        )
    return Applicability(Verdict.SATISFIED)


def combine(*verdicts: Applicability) -> Applicability:
    """Fold several applicability verdicts into one.

    A metric can depend on the **party** (does this agent expose ASR?) and on
    the **provider** (does this TTS stream?) at once. The two are evaluated
    separately, by different owners, and have to meet somewhere.

    The fold is ordered by severity, not by argument position:
    ``NOT_APPLICABLE`` dominates ``UNATTESTED`` dominates ``SATISFIED``. That
    ordering is the point — a metric that is inapplicable for one reason and
    merely unverified for another is **inapplicable**, and reporting it as
    unverified would invite someone to publish it with a caveat.

    Reasons accumulate across every verdict at the winning severity, because
    fixing the first blocker only to meet the second is how a configuration
    takes three runs to debug.
    """
    if not verdicts:
        return Applicability(Verdict.SATISFIED)

    for level in (Verdict.NOT_APPLICABLE, Verdict.UNATTESTED):
        matching = [v for v in verdicts if v.verdict is level]
        if matching:
            return Applicability(
                level, tuple(reason for v in matching for reason in v.reasons)
            )
    return Applicability(Verdict.SATISFIED)
