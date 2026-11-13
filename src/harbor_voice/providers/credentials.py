"""Which environment variables a provider needs.

Harbor already ships a ~95-provider declarative table (``PROVIDERS``) of the
environment variables each provider reads. We consult that table rather than
rebuild it, and add voice-only entries in a local overlay consulted first.

Harbor has moved and reshaped that table between releases
(``harbor.agents.model_connection`` with ``api_key_envs`` tuples, then
``harbor.agents.agent_auth_resolution`` with ``api_key_candidate_names``
lists). We read either into our own :class:`CredentialAccess`, so this module
works across those versions and nothing else here depends on harbor's shape. The overlay rather than an in-place edit
because harbor-voice is a fork that may take upstream merges: a table we have
rewritten conflicts on every upstream change to it, and an overlay conflicts
never.

What this module does *not* do is resolve connections the way harbor's
``resolve_model_connection`` does. That function's job is rewriting env-var
names for CLI agents running inside containers, which is a different problem
from ours — we need to know whether the credentials are present so
``doctor`` can say so before a container starts, and we need to never print
them. So we reuse harbor's *table*, and the policy here is ours.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True, slots=True)
class CredentialAccess:
    """The environment variables one provider reads."""

    api_key_envs: tuple[str, ...]
    """Alternatives: exactly one must be set."""
    extra_envs: tuple[str, ...] = ()
    base_url_envs: tuple[str, ...] = ()

    @classmethod
    def from_harbor(cls, access: Any) -> CredentialAccess:
        """Read a harbor ``ProviderAccess`` of either shape."""

        def names(new: str, old: str) -> tuple[str, ...]:
            return tuple(getattr(access, new, None) or getattr(access, old, ()) or ())

        return cls(
            api_key_envs=names("api_key_candidate_names", "api_key_envs"),
            extra_envs=names("extra_env_names", "extra_envs"),
            base_url_envs=names("base_url_candidate_names", "base_url_envs"),
        )


def _harbor_providers() -> dict[str, CredentialAccess]:
    for module in (
        "harbor.agents.agent_auth_resolution",  # harbor >= 0.25
        "harbor.agents.model_connection",  # earlier
    ):
        try:
            table = importlib.import_module(module).PROVIDERS
        except (ImportError, AttributeError):
            continue
        return {
            slug: CredentialAccess.from_harbor(access) for slug, access in table.items()
        }
    return {}


HARBOR_PROVIDERS: dict[str, CredentialAccess] = _harbor_providers()
"""Harbor's provider table, in our shape."""

_AWS_ACCESS = CredentialAccess(
    ("AWS_ACCESS_KEY_ID",),
    extra_envs=("AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION"),
)
"""Transcribe, Polly and Nova Sonic all authenticate the same way.

One entry rather than three, because three identical entries would invite them
to drift apart for no reason.
"""

VOICE_PROVIDER_ACCESS: dict[str, CredentialAccess] = {
    "aws": _AWS_ACCESS,
    "azure-speech": CredentialAccess(
        ("AZURE_SPEECH_KEY",), extra_envs=("AZURE_SPEECH_ENDPOINT",)
    ),
    "elevenlabs": CredentialAccess(("ELEVENLABS_API_KEY",)),
    "cartesia": CredentialAccess(("CARTESIA_API_KEY",)),
    "deepgram": CredentialAccess(("DEEPGRAM_API_KEY",)),
    "fishaudio": CredentialAccess(("FISH_API_KEY",)),
    "sarvam": CredentialAccess(("SARVAM_API_KEY",)),
    # One entry per slot, each needing its own server's address: a model
    # server that is not configured is "missing", not "ready".
    **{
        f"selfhosted-{slot}": CredentialAccess(
            (), extra_envs=(f"SELFHOSTED_{slot.upper()}_BASE_URL", "SELFHOSTED_API_KEY")
        )
        for slot in ("stt", "tts", "llm", "realtime")
    },
}
"""Voice-only providers, consulted before harbor's table."""


def access_for(slug: str | None) -> CredentialAccess | None:
    """Find a provider's credential requirements. Overlay first, then harbor."""
    if slug is None:
        return None
    return VOICE_PROVIDER_ACCESS.get(slug) or HARBOR_PROVIDERS.get(slug)


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    """Whether a provider's credentials are available, without revealing them.

    ``doctor`` prints this. Nothing here is a secret: the names of the
    variables, and whether each is set.
    """

    slug: str | None
    present: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    optional_missing: tuple[str, ...] = ()
    """Set but not required — ``AWS_SESSION_TOKEN`` is absent for a long-lived
    key pair and that is normal, so it must not read as a failure."""

    _values: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    @property
    def ready(self) -> bool:
        """Whether every required variable is set."""
        return not self.missing and self.slug is not None

    @property
    def needs_nothing(self) -> bool:
        """Whether this provider needs no credentials at all."""
        return self.slug is None

    def container_env(self) -> dict[str, str]:
        """The variables to pass into a trial container.

        Kept off ``repr`` and out of ``__eq__`` so a status can be logged,
        compared or asserted on without a key reaching a log line.
        """
        return dict(self._values)

    def explain(self) -> str:
        """One line for ``doctor``, naming what is missing and nothing else."""
        if self.needs_nothing:
            return "no credentials required"
        if self.ready:
            extra = (
                f" ({len(self.optional_missing)} optional unset)"
                if self.optional_missing
                else ""
            )
            return f"{self.slug}: {len(self.present)} set{extra}"
        return f"{self.slug}: missing {', '.join(self.missing)}"


def resolve(
    slug: str | None,
    *,
    getenv: Callable[[str], str | None] = os.environ.get,
) -> CredentialStatus:
    """Report which of ``slug``'s variables are set.

    Args:
        slug: Credential-provider slug, e.g. ``aws``. ``None`` for a provider
            that needs no credentials.
        getenv: Lookup, injectable so this is testable without touching the
            real environment.

    Returns:
        A status that is safe to print. A required variable is one of the
        provider's ``api_key_envs`` — exactly one of them must be set, since
        they are alternatives — plus every ``extra_env`` that is not optional.
    """
    access = access_for(slug)
    if access is None:
        return CredentialStatus(slug=None)

    present: list[str] = []
    missing: list[str] = []
    optional_missing: list[str] = []
    values: dict[str, str] = {}

    alternatives = [name for name in access.api_key_envs if getenv(name)]
    if alternatives:
        for name in alternatives:
            present.append(name)
            value = getenv(name)
            if value:
                values[name] = value
    elif access.api_key_envs:
        missing.append(" or ".join(access.api_key_envs))

    for name in access.extra_envs:
        value = getenv(name)
        if value:
            present.append(name)
            values[name] = value
        elif name in _OPTIONAL_ENVS:
            optional_missing.append(name)
        else:
            missing.append(name)

    for name in access.base_url_envs:
        value = getenv(name)
        if value:
            present.append(name)
            values[name] = value

    return CredentialStatus(
        slug=slug,
        present=tuple(present),
        missing=tuple(missing),
        optional_missing=tuple(optional_missing),
        _values=values,
    )


_OPTIONAL_ENVS: frozenset[str] = frozenset(
    {
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_API_VERSION",
        "SELFHOSTED_API_KEY",
    }
)
"""Variables whose absence is normal rather than a misconfiguration.

``AWS_SESSION_TOKEN`` is the one that matters here: it is set for temporary
credentials and unset for a long-lived key pair, so reporting it as missing
would make a correct setup look broken.
"""
