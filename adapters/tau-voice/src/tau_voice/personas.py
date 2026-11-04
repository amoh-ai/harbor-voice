"""tau2's caller voices on ElevenLabs: ``tau-voice voices create`` and ``list``.

Every tau-voice task speaks as one of tau2's seven personas, each an
ElevenLabs voice the task names by ``TAU2_VOICE_ID_<PERSONA>``. tau2 makes
them with its ``setup_voices`` script (``src/tau2/voice/scripts/
setup_voices.py``), which needs a tau2 checkout; this makes the same voices
without one, with the same requests:

- ``POST /v1/text-to-voice/design`` with the persona's prompt and tau2's
  Voice Design settings (model, loudness, guidance scale, seed, sample text),
  keeping the first preview;
- ``POST /v1/text-to-voice`` to save it as ``tau2_<persona>``, described by
  the persona's short description.

The prompts and settings are tau2's, copied verbatim into
``tau_voice/elevenlabs/personas.json``. tau2 says the same seed and
prompt give the same voice.

Voices already on the account (``GET /v1/voices``, named ``tau2_<persona>``)
are kept unless ``--replace`` (``DELETE`` first, as tau2's "replace"). Only
``create --yes`` writes to the account; voice IDs never go in the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

SPEC_PATH = Path(__file__).resolve().parent / "elevenlabs" / "personas.json"
"""``tau_voice/elevenlabs/personas.json``."""
API_BASE = "https://api.elevenlabs.io"
TIMEOUT_S = 120.0
"""Voice Design renders previews before it answers."""
COMPLEXITIES = ("all", "control", "regular")


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def personas(
    complexity: str = "all", spec: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """tau2's personas in tau2's order, or one complexity's."""
    spec = load_spec() if spec is None else spec
    return [
        p
        for p in spec["personas"]
        if complexity == "all" or p["complexity"] == complexity
    ]


def env_var(persona: str) -> str:
    """Where a task reads the persona's voice ID (``TAU2_VOICE_ID_<PERSONA>``)."""
    return f"TAU2_VOICE_ID_{persona.upper()}"


def voice_name(persona: str, spec: dict[str, Any] | None = None) -> str:
    spec = load_spec() if spec is None else spec
    return f"{spec['voice_name_prefix']}_{persona}"


def design_body(persona: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """``POST /v1/text-to-voice/design``, as tau2's SDK call sends it."""
    return {"voice_description": persona["prompt"], **spec["design"]}


def create_body(
    persona: dict[str, Any], spec: dict[str, Any], generated_voice_id: str
) -> dict[str, Any]:
    """``POST /v1/text-to-voice``: the preview saved under tau2's name."""
    return {
        "voice_name": voice_name(persona["name"], spec),
        "voice_description": persona["short_description"],
        "generated_voice_id": generated_voice_id,
    }


class VoicesAPI:
    """The four ElevenLabs requests ``setup_voices`` makes, over ``httpx``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str = API_BASE,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            raise SystemExit("ELEVENLABS_API_KEY is not set")
        self._client = httpx.Client(
            base_url=api_base.rstrip("/"),
            headers={"xi-api-key": key},
            timeout=TIMEOUT_S,
            transport=transport,
        )

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        response = self._client.request(method, path, json=body)
        if response.status_code >= 300:
            raise RuntimeError(
                f"{method} {path}: HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.json() if response.content else None

    def tau2_voices(self, prefix: str) -> dict[str, list[str]]:
        """Voice name -> IDs, for the account's voices named ``<prefix>_…``."""
        found: dict[str, list[str]] = {}
        for voice in self._call("GET", "/v1/voices").get("voices", []):
            name = voice.get("name") or ""
            if name.startswith(f"{prefix}_"):
                found.setdefault(name, []).append(voice["voice_id"])
        return found

    def design(self, body: dict[str, Any]) -> str:
        previews = self._call("POST", "/v1/text-to-voice/design", body).get("previews")
        if not previews:
            raise RuntimeError("Voice Design returned no previews")
        return previews[0]["generated_voice_id"]

    def save(self, body: dict[str, Any]) -> str:
        return self._call("POST", "/v1/text-to-voice", body)["voice_id"]

    def delete(self, voice_id: str) -> None:
        self._call("DELETE", f"/v1/voices/{voice_id}")


def create(
    api: VoicesAPI,
    complexity: str = "all",
    *,
    replace: bool = False,
    spec: dict[str, Any] | None = None,
    pause_s: float = 1.0,
) -> tuple[dict[str, str], list[str]]:
    """Each persona's voice ID, made where missing, and what failed.

    A voice already on the account is kept (its newest ID if there are
    several) unless ``replace``, which deletes it and makes a new one.
    """
    spec = load_spec() if spec is None else spec
    existing = api.tau2_voices(spec["voice_name_prefix"])
    ids: dict[str, str] = {}
    failed: list[str] = []
    for n, persona in enumerate(personas(complexity, spec)):
        name = voice_name(persona["name"], spec)
        if name in existing and not replace:
            ids[persona["name"]] = existing[name][-1]
            print(f"kept    {name}: {ids[persona['name']]}")
            continue
        if n:
            time.sleep(pause_s)  # tau2 spaces its requests by a second
        try:
            for old in existing.get(name, []):
                api.delete(old)
                print(f"deleted {name}: {old}")
            generated = api.design(design_body(persona, spec))
            ids[persona["name"]] = api.save(create_body(persona, spec, generated))
            print(f"created {name}: {ids[persona['name']]}")
        except (RuntimeError, httpx.HTTPError) as exc:
            failed.append(persona["name"])
            print(f"failed  {name}: {exc}", file=sys.stderr)
    return ids, failed


def env_lines(ids: dict[str, str]) -> str:
    return "".join(
        f"{env_var(persona)}={voice_id}\n" for persona, voice_id in ids.items()
    )


def main_voices(argv: Sequence[str] | None = None) -> int:
    """``tau-voice voices``: tau2's caller voices on your ElevenLabs account."""
    parser = argparse.ArgumentParser(
        prog="tau-voice voices", description=main_voices.__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser(
        "create", help="make the personas' voices on the account (writes!)"
    )
    make.add_argument("--complexity", choices=COMPLEXITIES, default="all")
    make.add_argument(
        "--replace",
        action="store_true",
        help="delete and remake voices that already exist (default: keep them)",
    )
    make.add_argument("--yes", action="store_true", help="really create them")
    show = sub.add_parser("list", help="the account's tau2 voices, as env lines")
    show.add_argument("--complexity", choices=COMPLEXITIES, default="all")
    args = parser.parse_args(argv)
    spec = load_spec()
    chosen = personas(args.complexity, spec)

    if args.command == "create" and not args.yes:
        for persona in chosen:
            print(
                json.dumps(
                    {voice_name(persona["name"], spec): design_body(persona, spec)},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        print(
            f"not created: this designs and saves {len(chosen)} voice(s) on the "
            "ElevenLabs account; add --yes",
            file=sys.stderr,
        )
        return 2

    api = VoicesAPI()
    if args.command == "list":
        existing = api.tau2_voices(spec["voice_name_prefix"])
        ids = {
            p["name"]: existing[voice_name(p["name"], spec)][-1]
            for p in chosen
            if voice_name(p["name"], spec) in existing
        }
        missing = [p["name"] for p in chosen if p["name"] not in ids]
        print(env_lines(ids), end="")
        if missing:
            print(f"missing: {', '.join(missing)}", file=sys.stderr)
        return 1 if missing else 0

    ids, failed = create(api, args.complexity, replace=args.replace, spec=spec)
    print("\nAdd to your environment (.env):\n")
    print(env_lines(ids), end="")
    return 1 if failed else 0
