"""Provider identity, metering and failure.

> A provider is a metered, fallible, capability-declaring resource, not a
> function.

Three types carry that, and each exists because the obvious shortcut is wrong:

``ProviderSpec``
    Static description, readable without credentials, so ``doctor`` can
    preflight.
``ProviderRef``
    What lands in a published record. Has no field a secret could occupy.
``Usage``
    The meter. Audio is not billed in tokens, so harbor's ``UsageInfo`` is
    *wrapped*, not bent.

And :class:`ProviderError`, whose whole point is that a provider failure is not
the agent scoring zero.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import override

import httpx

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
"""Routing. Configuration failures fail the job, not the trial.

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
    This is the concrete case for wrapping it rather than bending it: harbor's ``UsageInfo``
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

    A run with any replayed call cannot be published as a measurement.
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
    """Registry name, e.g. ``aws.nova-sonic``. The left side of a spec string."""

    modality: Modality
    capabilities: ProviderCapabilities

    credential_provider: str | None = None
    """Slug in harbor's ``PROVIDERS`` table (or our overlay), e.g.
    ``amazon-bedrock``. ``None`` for providers needing no credentials."""

    requires_extra: str | None = None
    """pip extra that must be installed to construct this, e.g. ``aws``."""

    default_model: str | None = None

    credential_slug_for: Callable[[str], str | None] | None = None
    """For a provider whose credentials depend on the model (``litellm.llm``):
    the slug for a model, in place of :attr:`credential_provider`."""

    sandbox_env: Callable[[dict[str, str]], dict[str, str]] | None = None
    """Adjusts this provider's variables before they go into a trial's
    sandbox, for settings that only work on this machine."""

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

    There is deliberately no field a credential could land in. A key in
    ``measurement.json`` or in an ATIF trajectory is not a bug you get to fix
    later, so the type makes it unrepresentable rather than relying on everyone
    remembering to redact.
    """

    provider: str
    model: str
    region: str | None = None
    config_digest: Digest | None = None
    """Digest of the resolved non-secret configuration."""

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


def require_key(
    explicit: str | None,
    *env_names: str,
    provider: str,
    model: str | None,
    hint: str | None = None,
) -> str:
    """An API key from the argument or the first of ``env_names`` that is set.

    Never logged. Missing, it is an AUTH failure naming the variables, plus
    ``hint`` when there is another way to sign in.
    """
    key = explicit or next(filter(None, map(os.environ.get, env_names)), None)
    if not key:
        message = f"{' or '.join(env_names)} is not set"
        raise ProviderError(
            ProviderFailure.AUTH,
            f"{message} ({hint})" if hint else message,
            provider=provider,
            model=model,
        )
    return key


def http_failure(
    status: int, *, not_found: ProviderFailure = ProviderFailure.MALFORMED
) -> ProviderFailure:
    """An HTTP status (or a websocket error event's) as a failure kind.

    402 is no credit left: every call fails the same way, so it fails the job
    like a bad key. What a 404 means differs by vendor (an unknown voice, a
    region without the model), so the caller says.
    """
    if status in (401, 402, 403):
        return ProviderFailure.AUTH
    if status == 404:
        return not_found
    if status in (408, 504):
        return ProviderFailure.TIMEOUT
    if status == 429:
        return ProviderFailure.THROTTLED
    if status >= 500:
        return ProviderFailure.STREAM_BROKEN
    return ProviderFailure.MALFORMED


def socket_failure(
    exc: Exception,
    *,
    provider: str,
    model: str | None,
    close_codes: Mapping[int, ProviderFailure] | None = None,
) -> ProviderError:
    """A websocket exception as a :class:`ProviderError`.

    Checked in this order:

    - a refused handshake carries an HTTP status: :func:`http_failure`;
    - a close frame whose code is in ``close_codes``, the vendor's documented
      codes: that kind;
    - ``TimeoutError``: TIMEOUT, so it is retried;
    - JSON that does not parse: MALFORMED;
    - anything else: STREAM_BROKEN.

    ``OSError`` falls in the last group. A refused or reset connection is not
    a timeout: the server said no, or dropped a call already under way, and
    retrying the same call at once rarely helps. STREAM_BROKEN marks the trial
    invalid (no measurement) without a retry. ``TimeoutError`` is an
    ``OSError`` too, which is why it is checked first.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    code = getattr(getattr(exc, "rcvd", None), "code", None)
    if isinstance(status, int):
        kind = http_failure(status)
    elif close_codes and code in close_codes:
        kind = close_codes[code]
    elif isinstance(exc, TimeoutError):
        kind = ProviderFailure.TIMEOUT
    elif isinstance(exc, json.JSONDecodeError):
        kind = ProviderFailure.MALFORMED
    else:
        kind = ProviderFailure.STREAM_BROKEN
    return ProviderError(
        kind,
        f"{type(exc).__name__}: {str(exc)[:300]}",
        provider=provider,
        model=model,
    )


RETRY_MIN_WAIT_S = 1.0
RETRY_MAX_WAIT_S = 10.0


def retry_wait(attempt: int) -> float:
    """Seconds to wait after failed ``attempt`` (from 1): 1, 2, 4, up to 10.

    tenacity's ``wait_exponential(multiplier=1, min=1, max=10)``, as tau2
    retries.
    """
    return min(RETRY_MAX_WAIT_S, max(RETRY_MIN_WAIT_S, 2.0 ** (attempt - 1)))


async def post_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    attempts: int,
    provider: str,
    model: str | None,
    retry_statuses: Collection[int] = (409, 429),
    **request: Any,
) -> httpx.Response:
    """POST, retrying failed connections, 5xx and ``retry_statuses``.

    Returns the last response whatever its status; the caller judges it.
    If no attempt gets a response, raises TIMEOUT.
    """
    for attempt in range(1, attempts + 1):
        try:
            response = await client.post(url, **request)
        except httpx.TransportError as exc:  # timeouts included
            if attempt == attempts:
                raise ProviderError(
                    ProviderFailure.TIMEOUT,
                    f"POST {url} failed after {attempt} attempts: {exc}",
                    provider=provider,
                    model=model,
                ) from exc
            await asyncio.sleep(retry_wait(attempt))
            continue
        status = response.status_code
        if (status in retry_statuses or status >= 500) and attempt < attempts:
            await asyncio.sleep(retry_wait(attempt))
            continue
        return response
    raise ValueError(f"attempts must be at least 1, got {attempts}")


class HTTPSession:
    """One HTTP client for a provider's whole life, not one per request.

    A new client per turn is a new TCP and TLS handshake per turn, which
    lands in the measured response time as harness overhead. The client is
    made on first use; :meth:`aclose` closes it unless the caller passed
    its own.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owned = client is None

    def client(self, timeout_s: float) -> httpx.AsyncClient:
        if self._client is None:
            # Redirects are followed: a server behind Modal's web proxy that
            # takes longer than ~150 s (a model still loading) answers 303 to
            # a URL where the result will appear. httpx turns the POST into a
            # GET for a 303, as the proxy expects, and drops the auth header
            # if a redirect leaves the original host.
            self._client = httpx.AsyncClient(timeout=timeout_s, follow_redirects=True)
        return self._client

    async def aclose(self) -> None:
        if self._owned and self._client is not None:
            await self._client.aclose()
        self._client = None


class PooledHTTP:
    """For a provider holding an :class:`HTTPSession` as ``_http``."""

    _http: HTTPSession

    async def aclose(self) -> None:
        """Close the provider's connections. Safe to call more than once."""
        await self._http.aclose()
