"""Fake judges for the sidecar goldens: EVA's LiteLLM router, replaced.

Every judge EVA runs reaches its model through one call,
``eva.utils.router.get().acompletion(**kwargs)`` (``eva/utils/llm_client.py``).
This puts a recording fake in the router's place, so a metric run makes no
API call and every judge request is captured: the model, its parameters and
each message part. Text and audio parts are kept as SHA-256 and length, so
two captures compare byte for byte without storing the prompts.

The fake answers each judge in the shape its parser reads, so the run goes
on as with real judges: the top rating of the metric's scale, and one entry
per turn the prompt lists. Validation therefore passes, ``/grade`` runs, and
no judge retries (EVA retries an audio judge that rates no turn).

Two uses, inside EVA's image:

- imported by ``sidecar_goldens.py`` around EVA's own runner (the native
  side of the judge-prompt golden);
- run as the sidecar's entrypoint, ``python sidecar_fake_judges.py``: it
  installs the fake, then serves the sidecar's app with the captured
  requests at ``GET /_fake_judges`` (the sidecar side).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from types import SimpleNamespace
from typing import Any

TURN = re.compile(r"(?m)^\s*Turn (\d+)\b")
"""A turn line in a per-turn judge prompt: ``Turn 3: ...`` (intended turns)
or ``Turn 3 - [Assistant speaks]`` (the redacted trace)."""


def _sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _part(part: Any) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "text", "sha256": _sha(part), "chars": len(part)}
    kind = part.get("type", "?")
    if kind == "text":
        return {
            "type": "text",
            "sha256": _sha(part["text"]),
            "chars": len(part["text"]),
        }
    payload = json.dumps(part, sort_keys=True, ensure_ascii=False)
    return {"type": kind, "sha256": _sha(payload), "chars": len(payload)}


def digest(metric: str | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    """One judge request, as compared across the two sides."""
    messages = []
    for message in kwargs.get("messages", []):
        content = message.get("content")
        parts = [content] if isinstance(content, str) else list(content or [])
        messages.append(
            {"role": message.get("role"), "parts": [_part(p) for p in parts]}
        )
    params = {k: v for k, v in kwargs.items() if k not in ("messages", "model")}
    return {
        "metric": metric,
        "model": kwargs.get("model"),
        "params": json.loads(json.dumps(params, sort_keys=True, default=str)),
        "messages": messages,
    }


def _calling_metric() -> Any:
    """The EVA metric whose judge call this is: the nearest ``self`` on the
    stack that is a ``BaseMetric`` (the await chain keeps the frames linked)."""
    from eva.metrics.base import BaseMetric

    frame = sys._getframe(1)
    while frame is not None:
        candidate = frame.f_locals.get("self")
        if isinstance(candidate, BaseMetric):
            return candidate
        frame = frame.f_back
    return None


def _text(kwargs: dict[str, Any]) -> str:
    texts = []
    for message in kwargs.get("messages", []):
        content = message.get("content")
        for part in [content] if isinstance(content, str) else list(content or []):
            if isinstance(part, str):
                texts.append(part)
            elif part.get("type") == "text":
                texts.append(part["text"])
    return "\n".join(texts)


def answer(metric: Any, kwargs: dict[str, Any]) -> str:
    """A judge response every EVA judge parses: the scale's top rating."""
    _, high = getattr(metric, "rating_scale", None) or (1, 3)
    turns = sorted({int(t) for t in TURN.findall(_text(kwargs))})
    return json.dumps(
        {
            "rating": high,
            "explanation": "fake judge",
            "turns": [
                {
                    "turn_id": t,
                    "rating": high,
                    "explanation": "fake judge",
                    "transcript": "",
                    "language": "English",
                    "failure_modes": [],
                    "has_entities": True,
                }
                for t in turns
            ],
        }
    )


class FakeRouter:
    """``litellm.Router`` as EVA uses it: ``acompletion`` only."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def acompletion(self, **kwargs: Any) -> Any:
        metric = _calling_metric()
        self.requests.append(digest(getattr(metric, "name", None), kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=answer(metric, kwargs)))
            ],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
            model=kwargs.get("model"),
        )


def install() -> FakeRouter:
    """Put the fake in EVA's router slot; ``router.init`` no longer replaces it."""
    from eva.utils import router

    fake = FakeRouter()
    router._router = fake  # type: ignore[assignment]
    router.init = lambda model_list: None  # type: ignore[assignment]
    return fake


def serve() -> None:
    """The sidecar, with fake judges and ``GET /_fake_judges``."""
    sys.path.insert(0, "/opt/eva-sidecar")
    import server  # the sidecar's server.py

    fake = install()

    @server.app.get("/_fake_judges")
    def fake_judges() -> list[dict[str, Any]]:
        return fake.requests

    server.main()


if __name__ == "__main__":
    serve()
