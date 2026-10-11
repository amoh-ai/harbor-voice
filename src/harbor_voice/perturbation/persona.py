"""L1 — who speaks.

The second layer that transforms no samples. L3 puts a room around a voice and
L4 puts it down a wire; L1 decides **whose voice it is in the first place**, and
that happens before there is any audio to transform.

It reaches the call at two points, and they are different points:

====================  ==========================  ==========================
Axis                  Where it lands              What it changes
====================  ==========================  ==========================
``accent`` ``voice``  synthesis                   which voice renders the text
``gender`` ``language``
``behaviour``         the content source          what the caller says, and how
====================  ==========================  ==========================

### Requested is not achieved

A TTS that cannot do accents will happily synthesise an "Indian-accented"
caller in its default voice, and nothing downstream would know. That is the
same failure as on the L3 side — asking for 15 dB SNR and
reporting 15 dB whatever came out — so it gets the same treatment: the plan
records what was **asked for** and what was **honoured**, and a run under
``eva_a`` whose TTS ignored the accent says so rather than claiming the
condition.

This is enforced through an optional protocol rather than a required one.
:class:`~harbor_voice.providers.protocols.TTSProvider` stays a three-method
interface; a provider that can be asked to speak as someone specific declares
:class:`VoiceSelectable` as well, exactly like every other optional
capability — declared, not discovered by calling it and seeing what happens.

### A note on EVA

EVA's ``PerturbationConfig`` makes accent and behaviour **mutually exclusive**,
because it maps each variant onto a pre-provisioned ElevenLabs agent ID and a
call only gets one ID. So EVA cannot produce an elderly caller with an Indian
accent.

That is a constraint of their provisioning, not of the idea, and splitting the
axes by where they land — voice at synthesis, behaviour at the content source —
makes them independent here for free. The ``eva_a`` profile stays faithful to
whatever EVA actually ran, for parity; the layer does not inherit the
restriction.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Any
from typing import Protocol
from typing import override
from typing import runtime_checkable

from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import LayerKey


class PersonaError(ValueError):
    """An L1 layer was configured in a way that cannot produce a measurement."""


class Accent(StrEnum):
    """Accent variants. EVA's four, which is what parity needs."""

    FRENCH = "french"
    INDIAN = "indian"
    SPANISH = "spanish"
    CHINESE = "chinese"


class Gender(StrEnum):
    """Perceived voice gender, as providers enumerate it.

    ``UNSPECIFIED`` is the default and is not the same as a missing value: it
    means the run did not ask, so nothing may be concluded about gender from
    it. A benchmark comparing male and female callers needs both arms to have
    asked.
    """

    FEMALE = "female"
    MALE = "male"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True, slots=True)
class VoiceSelection:
    """Which voice should render the caller's words.

    ``language`` is a plain BCP-47-ish string rather than an enum, on purpose.
    Every provider supports a different set, the sets change, and a closed
    enum here would mean a provider that gained Welsh could not be used until
    harbor-voice shipped a release.
    """

    accent: Accent | None = None
    gender: Gender = Gender.UNSPECIFIED
    language: str = "en"
    voice_id: str | None = None
    """A provider-specific identifier, when a task pins an exact voice."""

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise PersonaError("language cannot be blank; use 'en' for the default")

    @property
    def is_default(self) -> bool:
        """Whether this asks for anything at all."""
        return (
            self.accent is None
            and self.gender is Gender.UNSPECIFIED
            and self.language == "en"
            and self.voice_id is None
        )

    @property
    def layers(self) -> frozenset[LayerKey]:
        """Which L1 keys this selection switches on."""
        keys: set[LayerKey] = set()
        if self.accent is not None:
            keys.add(LayerKey(layer=Layer.PERSONA, name="accent"))
        if self.gender is not Gender.UNSPECIFIED:
            keys.add(LayerKey(layer=Layer.PERSONA, name="gender"))
        if self.language != "en":
            keys.add(LayerKey(layer=Layer.PERSONA, name="language"))
        if self.voice_id is not None:
            keys.add(LayerKey(layer=Layer.PERSONA, name="voice"))
        return frozenset(keys)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accent": self.accent.value if self.accent else None,
            "gender": self.gender.value,
            "language": self.language,
            "voice_id": self.voice_id,
        }


@dataclass(frozen=True, slots=True)
class Behaviour:
    """How the caller behaves, as a brief the content source speaks from.

    Text, not parameters. EVA's ``user_behaviors.yaml`` is three paragraphs of
    prose per variant, and that is the right shape: "speaks slowly with
    frequent deliberate pauses and asks for repetition" is not expressible as
    a float, and trying would produce a caller that sounds like a slider.

    harbor-voice ships none of the benchmark's own briefs. A benchmark's
    personas belong to its adapter, the same boundary as for its
    metrics — otherwise the core accumulates one paper's prompt text and
    quietly becomes a port of it.
    """

    name: str
    brief: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise PersonaError("a behaviour needs a name; it goes in the plan record")
        if not self.brief.strip():
            raise PersonaError(
                f"behaviour {self.name!r} has an empty brief, so it would change "
                f"nothing while declaring L1.behaviour active"
            )

    @property
    def key(self) -> LayerKey:
        return LayerKey(layer=Layer.PERSONA, name="behaviour")


@runtime_checkable
class VoiceSelectable(Protocol):
    """A TTS that can be asked to speak as someone specific — L1.

    Optional, and declared rather than discovered. A provider that
    does not implement this is not broken; it just cannot serve L1's voice
    axis, and a task that activates one gets told so at load time instead of
    getting a default voice wearing an accent's name.
    """

    def select_voice(self, selection: VoiceSelection) -> bool:
        """Adopt ``selection`` if possible.

        Returns:
            Whether it was honoured in full. ``False`` is a legitimate answer
            and is recorded — a provider with French but no Hindi should say
            so rather than raise, because the run is still worth having as
            long as it does not claim the condition.
        """
        ...


@dataclass(slots=True)
class PersonaPlan:
    """The active L1 axes, and which of them the providers actually honoured.

    Sits beside the signal chain in
    :class:`~harbor_voice.perturbation.plan.PerturbationPipeline`, like
    :class:`~harbor_voice.perturbation.conversation.ConversationPlan`: it
    contributes to ``active_layers`` and to the written plan, never to
    ``apply``.
    """

    voice: VoiceSelection = field(default_factory=VoiceSelection)
    behaviour: Behaviour | None = None

    _voice_honoured: bool | None = field(default=None, init=False, repr=False)

    @property
    def is_empty(self) -> bool:
        return self.voice.is_default and self.behaviour is None

    @property
    def active_layers(self) -> frozenset[LayerKey]:
        """What a task has switched on. Feeds metric applicability."""
        keys = set(self.voice.layers)
        if self.behaviour is not None:
            keys.add(self.behaviour.key)
        return frozenset(keys)

    @property
    def voice_honoured(self) -> bool | None:
        """``None`` until a provider has been asked — not the same as ``False``."""
        return self._voice_honoured

    def reset(self) -> None:
        """Clear what was achieved, for a new run."""
        self._voice_honoured = None

    # ------------------------------------------------------------- applying

    def apply_to_tts(self, tts: object) -> bool:
        """Ask a TTS to adopt the selected voice, and record the answer.

        A default selection is honoured trivially — there is nothing to ask
        for, so a provider that cannot select voices has not failed to do
        anything.
        """
        if self.voice.is_default:
            self._voice_honoured = True
            return True
        if not isinstance(tts, VoiceSelectable):
            self._voice_honoured = False
            return False
        self._voice_honoured = bool(tts.select_voice(self.voice))
        return self._voice_honoured

    def brief_for(self, base: str = "") -> str:
        """Compose the persona brief the content source speaks from.

        The behaviour is appended rather than substituted, so a task's own
        persona survives: "you are rebooking a cancelled flight" and "you are
        elderly and speak slowly" are both true of the same caller.
        """
        if self.behaviour is None:
            return base
        if not base.strip():
            return self.behaviour.brief
        return f"{base.rstrip()}\n\n{self.behaviour.brief}"

    # --------------------------------------------------------------- record

    def params(self) -> dict[str, Any]:
        record: dict[str, Any] = {"voice": self.voice.as_dict()}
        if self.behaviour is not None:
            record["behaviour"] = {
                "name": self.behaviour.name,
                "brief_chars": len(self.behaviour.brief),
            }
        return record

    def achieved(self) -> dict[str, float]:
        """Whether the request survived contact with the provider."""
        if self._voice_honoured is None:
            return {}
        return {"voice_honoured": float(self._voice_honoured)}

    def as_record(self) -> dict[str, dict[str, Any]]:
        """The L1 half of the written plan.

        Keyed per active axis rather than once for the layer, so a reader can
        see that ``L1.accent`` was requested and not honoured while
        ``L1.behaviour`` was applied — one verdict for the whole layer would
        hide exactly that.
        """
        if self.is_empty:
            return {}
        params = self.params()
        achieved = self.achieved()
        out: dict[str, dict[str, Any]] = {}
        for key in sorted(self.active_layers, key=str):
            if key.name == "behaviour":
                out[str(key)] = {"params": params["behaviour"], "achieved": {}}
            else:
                out[str(key)] = {"params": params["voice"], "achieved": achieved}
        return out

    @override
    def __repr__(self) -> str:
        axes = ", ".join(sorted(str(k) for k in self.active_layers)) or "none"
        return f"PersonaPlan({axes})"
