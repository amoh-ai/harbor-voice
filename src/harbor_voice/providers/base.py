"""Provider identity, metering and failure.

> A provider is a **metered, fallible, capability-declaring resource**, not a
> function.

Three types carry that, and each exists because the obvious shortcut is wrong:

``ProviderSpec``
    Static description, readable without credentials, so ``doctor`` can
    preflight.
``ProviderRef``
    What lands in a published record. Has **no field a secret could occupy**.
``Usage``
    The meter. Audio is not billed in tokens, so harbor's ``UsageInfo`` is
    *wrapped*, not bent.

And :class:`ProviderError`, whose whole point is that a provider failure is not
the agent scoring zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import override

from harbor_voice.ids import Digest
from harbor_voice.ids import sha256_digest
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor.models.metric import UsageInfo

    from harbor_voice.audio.clock import Interval


# ------------------------------------------------------------------- failures


class ProviderFailure(StrEnum):
    """Why a provider call did not produce a usable result.

    A vocabulary, not a boolean. Harbor's verifier collapses a missing or
    unparseable result to ``0``, which is indistinguishable from an agent that
    genuinely failed; having named that as its weakness, reproducing it here
    would be indefensible.
    """

    AUTH = "auth"
    """Credentials absent, malformed or rejected."""

    NOT_ENABLED = "not_enabled"
    """The model exists but is not enabled for this account in this region.

    Its own member because it is the overwhelmingly likely first AWS failure:
    Bedrock requires per-account, per-region model enablement and raises an
    ``AccessDeniedException`` that reads exactly like bad credentials.
    Distinguishing the two is the difference between a five-minute fix and an
    afternoon.
    """

    REGION = "region"
    """The model is not offered in the configured region at all."""

    THROTTLED = "throttled"
    """Rate limited. Transient by definition."""

    TIMEOUT = "timeout"
    """No response within the configured budget."""

    STREAM_BROKEN = "stream_broken"
    """The stream died mid-call, leaving a partial trajectory."""

    MALFORMED = "malformed"
    """The provider violated its own protocol."""


class Disposition(StrEnum):
    """What a failure does to the run."""

    FAIL_JOB = "fail_job"
    """Stop everything. Every trial would fail identically."""

    RETRY = "retry"
    """Transient; retry per policy, and mark the trial invalid if exhausted."""

    INVALID = "invalid"
    """This trial produced no measurement. Not a zero — an absence."""


_DISPOSITIONS: dict[ProviderFailure, Disposition] = {
    ProviderFailure.AUTH: Disposition.FAIL_JOB,
    ProviderFailure.NOT_ENABLED: Disposition.FAIL_JOB,
    ProviderFailure.REGION: Disposition.FAIL_JOB,
    ProviderFailure.THROTTLED: Disposition.RETRY,
    ProviderFailure.TIMEOUT: Disposition.RETRY,
    ProviderFailure.STREAM_BROKEN: Disposition.INVALID,
    ProviderFailure.MALFORMED: Disposition.INVALID,
}
"""Routing. Configuration failures fail the **job**, not the trial.

Reporting 200 identical zeros because one env var is misspelled hides the typo
behind a plausible result table. The first trial already knows.
"""


class ProviderError(RuntimeError):
    """A provider call failed, with enough structure to route the failure."""

    def __init__(
        self,
        kind: ProviderFailure,
        message: str,
        *,
        provider: str,
        model: str | None = None,
    ) -> None:
        self.kind = kind
        self.provider = provider
        self.model = model
        where = f"{provider}/{model}" if model else provider
        super().__init__(f"[{kind.value}] {where}: {message}")

    @property
    def disposition(self) -> Disposition:
        """What this failure does to the run."""
        return _DISPOSITIONS[self.kind]

    @property
    def is_retryable(self) -> bool:
        return self.disposition is Disposition.RETRY

    @property
    def invalidates_trial(self) -> bool:
        """Whether the trial yields no measurement — as opposed to a zero."""
        return self.disposition in (Disposition.INVALID, Disposition.RETRY)


# -------------------------------------------------------------------- the meter


@dataclass(frozen=True, slots=True)
class Usage:
    """What one provider call consumed.

    ``None`` means *this provider does not bill in this unit*, which is a
    different statement from ``0`` — that it does, and the count was zero.
    Transcribe bills per audio-second, Polly per character, Bedrock per token;
    forcing audio into ``prompt_tokens`` would make both the cost column and
    the type wrong.
    """

    audio_seconds: float | None = None
    characters: int | None = None
    tokens: UsageInfo | None = None
    """Harbor's own model, unmodified, for the LLM slot."""

    speech_tokens: int | None = None
    """Tokens billed as audio rather than text.

    Nova Sonic meters these separately from text tokens and prices them
    differently, so folding them together would make the cost column wrong.
    This is the concrete case for a separate meter: harbor's ``UsageInfo``
    has three token fields and no notion of a speech token, so the count had
    nowhere truthful to go.
    """

    text_tokens: int | None = None
    """Tokens billed as text, where a provider separates them from speech."""

    cost_usd: float | None = None
    latency: Interval | None = None
    """Clock-paired — a duration cannot be recorded without naming
    both of its endpoints."""

    cache_hit: bool = False
    """Served from the simulator's :class:`~harbor_voice.providers.cache.BlobCache`."""

    replayed: bool = False
    """Served from a :class:`~harbor_voice.providers.cache.Cassette`.

    A run with any replayed call **cannot be published as a measurement**.
    Separate from ``cache_hit`` because the two have opposite
    legitimacy and merging them would make the distinction unenforceable.
    """

    def __post_init__(self) -> None:
        for name in (
            "audio_seconds",
            "characters",
            "cost_usd",
            "speech_tokens",
            "text_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative: {value}")

    @property
    def is_free(self) -> bool:
        """Whether this call cost nothing because nothing was sent upstream."""
        return self.cache_hit or self.replayed

    def as_dict(self) -> dict[str, Any]:
        """Record form. Omits units this provider does not bill in."""
        out: dict[str, Any] = {}
        if self.audio_seconds is not None:
            out["audio_seconds"] = self.audio_seconds
        if self.characters is not None:
            out["characters"] = self.characters
        if self.tokens is not None:
            out["tokens"] = self.tokens.model_dump()
        if self.speech_tokens is not None:
            out["speech_tokens"] = self.speech_tokens
        if self.text_tokens is not None:
            out["text_tokens"] = self.text_tokens
        if self.cost_usd is not None:
            out["cost_usd"] = self.cost_usd
        if self.latency is not None:
            out[self.latency.named("latency")] = self.latency.seconds
        if self.cache_hit:
            out["cache_hit"] = True
        if self.replayed:
            out["replayed"] = True
        return out


# -------------------------------------------------------- identity and record


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Static description of a provider — no credentials, no network.

    Readable from the class so ``doctor``, ``caps`` and ``profiles`` can report
    on providers this machine cannot even construct.
    """

    name: str
    """Registry name, e.g. ``aws.polly``. The left side of a spec string."""

    modality: Modality
    capabilities: ProviderCapabilities

    credential_provider: str | None = None
    """Slug in harbor's ``PROVIDERS`` table (or our overlay), e.g.
    ``amazon-bedrock``. ``None`` for providers needing no credentials."""

    requires_extra: str | None = None
    """pip extra that must be installed to construct this, e.g. ``aws``."""

    default_model: str | None = None

    def __post_init__(self) -> None:
        if self.capabilities.modality is not self.modality:
            raise ValueError(
                f"spec {self.name!r} declares modality={self.modality.value} but "
                f"its capabilities say {self.capabilities.modality.value}"
            )
        if "/" in self.name:
            raise ValueError(
                f"provider name {self.name!r} cannot contain '/': the slash "
                f"separates provider from model in a spec string"
            )

    @property
    def is_offline(self) -> bool:
        """Whether this provider runs with no credential and no network."""
        return self.credential_provider is None and self.requires_extra is None


@dataclass(frozen=True, slots=True)
class ProviderRef:
    """What a published artifact says about a provider.

    There is deliberately **no field a credential could land in**. A key in
    ``measurement.json`` or in an ATIF trajectory is not a bug you get to fix
    later, so the type makes it unrepresentable rather than relying on everyone
    remembering to redact.
    """

    provider: str
    model: str
    region: str | None = None
    config_digest: Digest | None = None
    """Digest of the resolved **non-secret** configuration."""

    capabilities: ProviderCapabilities | None = None

    @classmethod
    def build(
        cls,
        *,
        provider: str,
        model: str,
        region: str | None = None,
        config: dict[str, Any] | None = None,
        capabilities: ProviderCapabilities | None = None,
    ) -> ProviderRef:
        """Build a ref, digesting ``config`` rather than storing it.

        Args:
            config: Non-secret settings that change the output — voice,
                sampling rate, temperature. The caller is responsible for not
                putting a credential in here; the digest would hide it rather
                than remove it.
        """
        digest: Digest | None = None
        if config:
            import json

            payload = json.dumps(config, sort_keys=True, default=str)
            digest = sha256_digest(payload.encode("utf-8"))
        return cls(
            provider=provider,
            model=model,
            region=region,
            config_digest=digest,
            capabilities=capabilities,
        )

    @property
    def spec_string(self) -> str:
        """``provider/model``, the form the CLI accepts."""
        return f"{self.provider}/{self.model}"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"provider": self.provider, "model": self.model}
        if self.region is not None:
            out["region"] = self.region
        if self.config_digest is not None:
            out["config_digest"] = self.config_digest
        return out

    @override
    def __str__(self) -> str:
        return self.spec_string


@dataclass(frozen=True, slots=True)
class ProviderCall:
    """One provider interaction, as the trajectory records it."""

    ref: ProviderRef
    usage: Usage = field(default_factory=Usage)
    failure: ProviderFailure | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None

    def as_dict(self) -> dict[str, Any]:
        out = self.ref.as_dict()
        usage = self.usage.as_dict()
        if usage:
            out["usage"] = usage
        if self.failure is not None:
            out["failure"] = self.failure.value
            out["disposition"] = _DISPOSITIONS[self.failure].value
        return out
