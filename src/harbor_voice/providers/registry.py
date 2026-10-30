"""Resolving a spec string to a provider.

Two operations, deliberately separate:

``lookup``
    What is this provider, what can it do, what does it need? Answered from
    the **class**, with no credentials, no network and no SDK installed, so
    ``doctor`` and ``caps`` can report on providers this machine cannot
    construct.
``create``
    Actually build one. This is where a missing extra or a missing credential
    becomes an error.

The split is what lets ``doctor`` check whether a configuration measures what
the task claims: that can't be done if describing a provider requires
constructing it.

Spec strings are ``provider/model``, matching harbor and litellm. **Modality
comes from the slot being filled, not from the string**: the CLI flag
(``--stt``, ``--tts``, ``--llm``) already knows it, so encoding it twice would
only create a way for the two to disagree.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.credentials import CredentialStatus
from harbor_voice.providers.credentials import resolve as resolve_credentials

Builder = Callable[..., Any]
"""Constructs a provider. Called with ``model=`` plus provider-specific config.

Deliberately untyped in its return: the four modalities have four unrelated
interfaces, and a union of them would be satisfied by none of their
call sites. The slot protocols are what type the result at the point of use.
"""


class UnknownProvider(LookupError):
    """No provider is registered under that name."""


class ModalityMismatch(ValueError):
    """A provider was asked to fill a slot it cannot fill."""


def parse_spec(raw: str, *, default_model: str | None = None) -> tuple[str, str]:
    """Split ``provider/model``.

    Args:
        raw: The spec string. A bare ``provider`` is allowed when the
            registration supplies a ``default_model``.
        default_model: Used when ``raw`` names no model.

    Returns:
        ``(provider, model)``.

    Raises:
        ValueError: If no model is given and none is defaulted. A provider
            without a model is not a usable configuration, and guessing one
            would make two runs that read identically use different models.
    """
    text = raw.strip()
    if not text:
        raise ValueError("provider spec is empty")
    provider, slash, model = text.partition("/")
    provider = provider.strip()
    model = model.strip()
    if not provider:
        raise ValueError(f"provider spec {raw!r} has no provider before '/'")
    if slash and not model:
        raise ValueError(f"provider spec {raw!r} has a '/' but no model after it")
    if not model:
        if default_model is None:
            raise ValueError(
                f"provider spec {raw!r} names no model and {provider!r} has no "
                f"default; write '{provider}/<model>'"
            )
        model = default_model
    return provider, model


@dataclass(frozen=True, slots=True)
class Registration:
    """A provider the registry knows about, built or not."""

    spec: ProviderSpec
    build: Builder | None = None
    """``None`` means the name is reserved and describable but not implemented.

    Such a provider still answers ``caps`` and still appears in ``doctor``'s
    output — it simply refuses to be constructed. This is the honest version of
    "no stubs": an adapter whose ``connect()`` *succeeds* by assigning a
    sentinel object defers failure to first audio while its capability matrix
    claims otherwise.
    """

    @property
    def is_implemented(self) -> bool:
        return self.build is not None


class ProviderRegistry:
    """Names to providers. Describable without being constructable."""

    def __init__(self) -> None:
        self._entries: dict[str, Registration] = {}

    def register(self, spec: ProviderSpec, build: Builder | None = None) -> None:
        """Add a provider.

        Raises:
            ValueError: If the name is already taken. Silent replacement would
                make the provider you get depend on import order.
        """
        if spec.name in self._entries:
            raise ValueError(f"provider {spec.name!r} is already registered")
        self._entries[spec.name] = Registration(spec=spec, build=build)

    def lookup(self, name: str) -> Registration:
        """Describe a provider without constructing it."""
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(sorted(self._entries)) or "(none)"
            raise UnknownProvider(
                f"unknown provider {name!r}; registered: {known}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[Registration]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def for_modality(self, modality: Modality) -> list[Registration]:
        """Everything that can fill one slot, in name order."""
        return sorted(
            (e for e in self._entries.values() if e.spec.modality is modality),
            key=lambda e: e.spec.name,
        )

    def resolve(self, spec_string: str) -> tuple[Registration, str]:
        """Find the registration and the model a spec string names.

        The provider is read before the model because the registration is what
        supplies the default model — ``offline.tone-tts`` is a complete spec
        and ``aws.polly`` is not, and only the registry knows which.
        """
        name = spec_string.partition("/")[0].strip()
        if not name:
            raise ValueError(f"provider spec {spec_string!r} has no provider")
        entry = self.lookup(name)
        _, model = parse_spec(spec_string, default_model=entry.spec.default_model)
        return entry, model

    def capabilities(self, spec_string: str) -> ProviderCapabilities:
        """What a provider can do, from its name alone."""
        entry, _ = self.resolve(spec_string)
        return entry.spec.capabilities

    def credentials(self, spec_string: str, **resolve_kwargs: Any) -> CredentialStatus:
        """Whether this provider's credentials are available. Never prints them."""
        entry, _ = self.resolve(spec_string)
        return resolve_credentials(entry.spec.credential_provider, **resolve_kwargs)

    def create(self, spec_string: str, *, modality: Modality, **config: Any) -> Any:
        """Build a provider for one slot.

        Raises:
            UnknownProvider: The name is not registered.
            ModalityMismatch: The provider cannot fill this slot. Caught here
                rather than at first use, because an STT in a TTS slot fails
                somewhere far from the configuration that caused it.
            NotImplementedError: The name is reserved but unimplemented.
        """
        entry, model = self.resolve(spec_string)
        spec = entry.spec

        if spec.modality is not modality:
            raise ModalityMismatch(
                f"{spec.name!r} is a {spec.modality.value} provider and cannot "
                f"fill the {modality.value} slot"
            )
        if entry.build is None:
            raise NotImplementedError(
                f"provider {spec.name!r} is registered but not implemented; "
                f"its capabilities are declared so 'doctor' and 'caps' can "
                f"report on it, but it cannot run"
            )
        return entry.build(model=model, **config)


def default_registry() -> ProviderRegistry:
    """A registry holding the offline reference set.

    Offline only. Cloud providers register through their own modules, which are
    imported lazily so that the 380-test offline suite keeps running with no
    SDK and no credential installed.
    """
    from harbor_voice.providers import offline

    registry = ProviderRegistry()
    for spec, build in offline.REGISTRATIONS:
        registry.register(spec, build)
    return registry
