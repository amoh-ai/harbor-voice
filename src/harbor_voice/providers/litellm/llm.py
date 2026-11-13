"""Any chat model litellm can reach, in the LLM slot.

One provider for every vendor litellm routes to. The model half of the spec
string is litellm's own model string, passed through unchanged:

=================================================  ==========================
Spec                                               Reaches
=================================================  ==========================
``litellm.llm/gemini/gemini-2.5-flash``            Gemini, AI Studio key
``litellm.llm/vertex_ai/gemini-2.5-flash``         Gemini on Vertex AI
``litellm.llm/xai/grok-4``                         xAI
``litellm.llm/bedrock/amazon.nova-pro-v1:0``       Bedrock
``litellm.llm/hosted_vllm/Qwen/Qwen3-8B``          vLLM / SGLang / any
                                                   OpenAI-compatible server
=================================================  ==========================

The registry splits a spec on its first ``/``, so everything after
``litellm.llm/`` is the model, slashes included.

litellm reads each vendor's credentials from that vendor's own environment
variables (``GEMINI_API_KEY``, ``XAI_API_KEY``, ``AWS_*``,
``HOSTED_VLLM_API_BASE``...), so nothing here takes a key. Which variables a
model needs is answered by :func:`credential_slug`, from harbor's table.

litellm is imported on first use, not at import time: it takes seconds to
load, and every CLI command builds the provider registry.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities

NAME = "litellm.llm"

CREDENTIALS = "litellm"
"""Not a slug in any table: the variables depend on the model, so
:func:`credential_slug` resolves them per spec string."""

NUM_RETRIES = 3
"""litellm's own retry budget, as tau2 sets it (``DEFAULT_MAX_RETRIES``)."""


def credential_slug(model: str) -> str | None:
    """The credential table slug for a litellm model string.

    Uses harbor's own resolution: litellm names the provider, harbor's alias
    table maps it to a ``PROVIDERS`` slug. ``None`` means nothing is needed
    (a self-hosted ``hosted_vllm/...`` server) or nothing is known.
    """
    from harbor.agents.agent_auth_resolution import parse_model_name
    from harbor.agents.agent_auth_resolution import provider_aliases

    from harbor_voice.providers.credentials import access_for

    provider, _ = parse_model_name(model)
    if provider is None:
        return None
    if provider == "bedrock":
        # Harbor's ``amazon-bedrock`` requires a bearer token and every AWS
        # variable; litellm signs with the usual key pair, which is ours.
        return "aws"
    slug = provider_aliases().get(provider, provider)
    return slug if access_for(slug) is not None else None


@dataclass(slots=True)
class LiteLLMChat:
    """Streamed chat completions through ``litellm.acompletion``.

    Args:
        model: A litellm model string, e.g. ``gemini/gemini-2.5-flash``.
        system: Optional system prompt.
        api_base: Server URL, for an OpenAI-compatible server addressed as
            ``openai/<model>``. ``hosted_vllm/`` reads
            ``HOSTED_VLLM_API_BASE`` instead.
    """

    model: str
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    api_base: str | None = None
    timeout_s: float = 120.0
    name: str = NAME
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        import litellm

        try:
            litellm.get_llm_provider(model=self.model)
        except Exception as exc:  # noqa: BLE001 - litellm raises several types
            # A setup mistake, not a provider failure: nothing was sent.
            raise ValueError(
                f"{self.name}: litellm cannot route {self.model!r}; prefix it "
                f"with a provider, e.g. 'gemini/{self.model}'"
            ) from exc

    async def respond(self, prompt: str, history: Sequence[str]) -> AsyncIterator[str]:
        """Yield litellm's text deltas as they arrive.

        History alternates user/assistant starting with the user, as
        ``BedrockLLM`` reads it.
        """
        import litellm
        from litellm.exceptions import LITELLM_EXCEPTION_TYPES

        messages = [{"role": "system", "content": self.system}] if self.system else []
        messages += [
            {"role": "user" if i % 2 == 0 else "assistant", "content": h}
            for i, h in enumerate(history)
        ]
        messages.append({"role": "user", "content": prompt})

        started = False
        try:
            stream: Any = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_base=self.api_base,
                timeout=self.timeout_s,
                num_retries=NUM_RETRIES,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if chunk.choices and (text := chunk.choices[0].delta.content):
                    started = True
                    yield text
                if usage := getattr(chunk, "usage", None):
                    self.usage = self._usage(usage)
        except tuple(LITELLM_EXCEPTION_TYPES) as exc:
            raise _classify(exc, self.name, self.model, started=started) from exc

    def _usage(self, usage: Any) -> Usage:
        from harbor.models.metric import UsageInfo

        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        cost = _cost(self.model, prompt, completion, cached)
        return Usage(
            tokens=UsageInfo(
                prompt_tokens=prompt,
                completion_tokens=completion,
                cache_tokens=cached,
                cost_usd=cost or 0.0,
            ),
            cost_usd=cost,
        )


def _cost(model: str, prompt: int, completion: int, cached: int) -> float | None:
    """litellm's price for these tokens; ``None`` for an unpriced model.

    A self-hosted model has no price, and ``0.0`` would claim it was free.
    """
    import litellm

    try:
        prompt_usd, completion_usd = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cache_read_input_tokens=cached,
        )
    except Exception:  # noqa: BLE001 - "model not mapped" has no single type
        return None
    return float(prompt_usd + completion_usd)


def _classify(
    exc: Exception, provider: str, model: str, *, started: bool
) -> ProviderError:
    from litellm import exceptions as e

    if started:
        kind = ProviderFailure.STREAM_BROKEN
    elif isinstance(exc, e.AuthenticationError | e.PermissionDeniedError):
        kind = ProviderFailure.AUTH
    elif isinstance(exc, e.NotFoundError):
        kind = ProviderFailure.NOT_ENABLED
    elif isinstance(exc, e.RateLimitError):
        kind = ProviderFailure.THROTTLED
    elif isinstance(exc, e.Timeout | e.APIConnectionError):
        # Two checks: litellm's Timeout subclasses openai's connection error,
        # not litellm's.
        kind = ProviderFailure.TIMEOUT
    elif isinstance(
        exc, e.InternalServerError | e.ServiceUnavailableError | e.BadGatewayError
    ):
        kind = ProviderFailure.STREAM_BROKEN
    else:
        kind = ProviderFailure.MALFORMED
    detail = str(exc)[:300]
    return ProviderError(
        kind, f"{type(exc).__name__}: {detail}", provider=provider, model=model
    )


LITELLM_LLM_SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.LLM,
    capabilities=ProviderCapabilities(modality=Modality.LLM, streams=True),
    credential_provider=CREDENTIALS,
    credential_slug_for=credential_slug,
)
"""No default model: which vendor to call is the whole choice."""


def build_llm(*, model: str, **config: Any) -> LiteLLMChat:
    return LiteLLMChat(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((LITELLM_LLM_SPEC, build_llm),)
