"""EVA's caller agents on ElevenLabs: create them, read them, check them.

EVA's paper ran its callers as four ElevenLabs agents set up by hand in the
UI from a table (``docs/experiment_setup.md:133-188``). Nothing in EVA's repo
pins those settings, and ElevenLabs fills everything the table leaves out
with defaults that can change. So the table lives here as data —
``eva_adapter/elevenlabs/agents.json`` — and every run can show which caller
produced it:

- :func:`payload` — one agent's ``POST /v1/convai/agents/create`` body: the
  table's settings and nothing else, as EVA's "Blank Agent" instructions say;
- :func:`create` / :func:`read` — the API calls (``read`` is ``GET``);
- :func:`snapshot` — each agent's live config written beside a run's results,
  account fields removed;
- :func:`diff` — every place a live agent differs from EVA's table
  (``mismatch``), from the defaults read back when ours were made
  (``drift``), or where the table has no API field we could confirm
  (``unconfirmed``).

Agent IDs are never in the repo. EVA reads them from ``EVA_{LANG}_USER_{F|M}``
(``elevenlabs.py:146-163``); so does :func:`agent_id_from_env`.

Nothing here runs on import or in tests against the real API; ``create``
writes to the ElevenLabs account and needs an explicit ``--yes`` on the CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor_voice.providers.elevenlabs.agents import AgentsAPI

SPEC_PATH = Path(__file__).resolve().parent / "elevenlabs" / "agents.json"
"""``eva_adapter/elevenlabs/agents.json``."""

PERSONA_GENDER: dict[int, str] = {1: "F", 2: "M"}
"""EVA's persona → agent gender (``elevenlabs.py:31``)."""

ACCOUNT_FIELDS = (
    "access_info",
    "access_permissions",
    "phone_numbers",
    "whatsapp_accounts",
)
"""Top-level fields of a ``GET`` agent that describe the account (creator name
and email, sharing, numbers), not the agent. A snapshot drops them."""

SNAPSHOT_PREFIX = "elevenlabs_agent_"


# -------------------------------------------------------------- which agent


def gender_of(persona: int | str) -> str:
    """``F`` or ``M`` from EVA's ``user_persona_id`` (1, 2) or a letter."""
    if isinstance(persona, int):
        return PERSONA_GENDER[persona]
    letter = persona.strip().upper()[:1]
    if letter not in ("F", "M"):
        raise ValueError(f"not a caller gender: {persona!r}")
    return letter


def agent_env_var(
    language: str = "en",
    gender: int | str = "F",
    *,
    accent: str | None = None,
    behavior: str | None = None,
) -> str:
    """The env var EVA reads the agent ID from (``elevenlabs.py:149-157``).

    Accent first, then behaviour, then language, as EVA chooses.
    """
    g = gender_of(gender)
    if accent:
        return f"EVA_{accent.upper()}_ACCENT_USER_{g}"
    if behavior:
        return f"EVA_{behavior.upper()}_USER_{g}"
    lang = (language or "en").upper().replace("-", "_")
    return f"EVA_{lang}_USER_{g}"


def agent_id_from_env(
    language: str = "en",
    gender: int | str = "F",
    *,
    environ: Mapping[str, str] | None = None,
    **variant: str | None,
) -> str:
    """The agent ID, as EVA finds it; a missing variable is EVA's ``ValueError``."""
    env = os.environ if environ is None else environ
    name = agent_env_var(language, gender, **variant)
    agent_id = env.get(name)
    if not agent_id:
        raise ValueError(f"Missing ElevenLabs agent ID environment variable: {name}")
    return agent_id


def label_for(language: str, gender: int | str) -> str:
    """The ``agents.json`` key for a language and gender, e.g. ``en_f``."""
    return f"{language.lower()}_{gender_of(gender).lower()}"


# ------------------------------------------------------------------- the spec


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def payload(label: str, spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One agent's create body: ``name`` + the table's ``conversation_config``.

    No ``platform_settings``: EVA's table sets none, so ElevenLabs' defaults
    apply, as they did for EVA's agents.
    """
    spec = load_spec() if spec is None else spec
    try:
        agent = spec["agents"][label]
    except KeyError:
        known = ", ".join(sorted(spec["agents"]))
        raise KeyError(f"no agent {label!r} in agents.json; known: {known}") from None
    body = _merge(
        dict(spec["common"]), {"conversation_config": agent["conversation_config"]}
    )
    return {"name": agent["name"], **body}


def _leaves(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping) and value:
            out.update(_leaves(value, path))
        else:
            out[path] = value
    return out


_MISSING = object()


def _at(tree: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(tree, Mapping) or part not in tree:
            return _MISSING
        tree = tree[part]
    return tree


@dataclass(frozen=True, slots=True)
class Finding:
    """One place a live agent is not what the spec says."""

    kind: str
    """``mismatch`` (EVA's table), ``drift`` (an observed default moved) or
    ``unconfirmed`` (no confirmed API field; the live value is shown)."""
    path: str
    expected: Any
    live: Any
    note: str = ""


def diff(
    label: str, live: Mapping[str, Any], spec: Mapping[str, Any] | None = None
) -> list[Finding]:
    """Compare a live agent (a ``GET`` body) with agent ``label`` of the spec.

    Every leaf of the create body except ``name`` must match (``mismatch``);
    every observed default should (``drift``); each ``unconfirmed`` entry is
    reported with its live value. Numbers compare by value (15 == 15.0).
    """
    spec = load_spec() if spec is None else spec
    body = payload(label, spec)
    findings: list[Finding] = []
    for path, expected in _leaves(
        {k: v for k, v in body.items() if k != "name"}
    ).items():
        actual = _at(live, path)
        if actual is _MISSING or actual != expected:
            findings.append(
                Finding(
                    "mismatch",
                    path,
                    expected,
                    None if actual is _MISSING else actual,
                    "missing" if actual is _MISSING else "",
                )
            )
    for path, expected in (spec.get("observed_defaults") or {}).items():
        actual = _at(live, path)
        if actual is _MISSING or actual != expected:
            findings.append(
                Finding(
                    "drift",
                    path,
                    expected,
                    None if actual is _MISSING else actual,
                    "missing" if actual is _MISSING else "",
                )
            )
    for item in spec.get("unconfirmed") or ():
        actual = _at(live, item["path"])
        findings.append(
            Finding(
                "unconfirmed",
                item["path"],
                item.get("eva"),
                None if actual is _MISSING else actual,
                item.get("note", ""),
            )
        )
    return findings


def strip_account_fields(agent: Mapping[str, Any]) -> dict[str, Any]:
    """A ``GET`` agent without the account's fields (:data:`ACCOUNT_FIELDS`)."""
    return {k: v for k, v in agent.items() if k not in ACCOUNT_FIELDS}


# -------------------------------------------------------------------- the API


async def create(
    label: str, api: AgentsAPI, spec: Mapping[str, Any] | None = None
) -> str:
    """``POST /v1/convai/agents/create`` for one agent; returns its new ID.

    Writes to the ElevenLabs account. A placeholder agent is refused.
    """
    spec = load_spec() if spec is None else spec
    status = spec["agents"].get(label, {}).get("status")
    if status != "active":
        raise ValueError(f"agent {label!r} is a {status or 'unknown'}, not created")
    return await api.create_agent(payload(label, spec))


async def read(agent_id: str, api: AgentsAPI) -> dict[str, Any]:
    """``GET /v1/convai/agents/{id}``: the agent's whole live configuration."""
    return await api.agent(agent_id)


async def snapshot(
    agent_ids: Mapping[str, str],
    out_dir: Path,
    api: AgentsAPI,
    spec: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Store each agent's live config in ``out_dir``, with its findings.

    ``agent_ids`` maps a label (``en_f``) to an agent ID. Writes
    ``elevenlabs_agent_<label>.json`` (the ``GET`` body without account
    fields) per agent, and ``elevenlabs_agents.json``: per label, the agent
    and version IDs and :func:`diff`'s findings when the label is in the spec.
    """
    spec = load_spec() if spec is None else spec
    out_dir.mkdir(parents=True, exist_ok=True)
    configs: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {}
    for label, agent_id in agent_ids.items():
        live = strip_account_fields(await read(agent_id, api))
        configs[label] = live
        (out_dir / f"{SNAPSHOT_PREFIX}{label}.json").write_text(
            json.dumps(live, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        entry: dict[str, Any] = {
            "agent_id": agent_id,
            "version_id": live.get("version_id"),
        }
        if label in spec["agents"]:
            entry["findings"] = [asdict(f) for f in diff(label, live, spec)]
        summary[label] = entry
    (out_dir / "elevenlabs_agents.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return configs


# -------------------------------------------------------------------- the CLI


def _active(spec: Mapping[str, Any]) -> list[str]:
    return [k for k, v in spec["agents"].items() if v.get("status") == "active"]


def _ids(
    labels: Sequence[str], spec: Mapping[str, Any], given: str | None
) -> dict[str, str]:
    if given is not None:
        if len(labels) != 1:
            raise SystemExit("--id needs exactly one --agent")
        return {labels[0]: given}
    out: dict[str, str] = {}
    for label in labels:
        env = spec["agents"][label]["env"]
        agent_id = os.environ.get(env)
        if not agent_id:
            raise SystemExit(f"{env} is not set (the {label} agent's ID)")
        out[label] = agent_id
    return out


def _print_findings(label: str, findings: Sequence[Finding]) -> None:
    if not findings:
        print(f"{label}: matches EVA's setup and the observed defaults")
        return
    for f in findings:
        note = f" ({f.note})" if f.note else ""
        print(
            f"{label}: {f.kind:<11} {f.path}: expected {f.expected!r}, live {f.live!r}{note}"
        )


def main_agents(
    argv: Sequence[str] | None = None,
    *,
    api_factory: Callable[[], AgentsAPI] = AgentsAPI,
) -> int:
    """``eva-adapter agents …``: show, create, read, snapshot, diff.

    Agent IDs come from EVA's env vars (``EVA_EN_USER_F`` …) unless ``--id``
    names one. Only ``create`` writes, and only with ``--yes``.
    """
    parser = argparse.ArgumentParser(
        prog="eva-adapter agents", description=main_agents.__doc__
    )
    parser.add_argument(
        "--spec", type=Path, default=SPEC_PATH, help="agents.json to use"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show", help="print create bodies (offline)")
    show.add_argument(
        "--agent", action="append", help="label, e.g. en_f (default: active)"
    )
    make = sub.add_parser("create", help="create an agent on the account (writes!)")
    make.add_argument("--agent", required=True)
    make.add_argument("--yes", action="store_true", help="really create it")
    get = sub.add_parser("read", help="GET one agent's live config")
    get.add_argument("--agent", required=True)
    get.add_argument("--id", help="agent ID (default: its EVA env var)")
    get.add_argument(
        "--out", type=Path, help="write the JSON here (account fields removed)"
    )
    snap = sub.add_parser(
        "snapshot", help="store live configs + findings in a directory"
    )
    snap.add_argument("--out", type=Path, required=True)
    snap.add_argument("--agent", action="append", help="label (default: active)")
    check = sub.add_parser("diff", help="compare live agents with EVA's setup")
    check.add_argument("--agent", action="append", help="label (default: active)")
    check.add_argument("--id", help="agent ID, with one --agent")
    check.add_argument(
        "--live", type=Path, help="compare this saved JSON instead of GET"
    )
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)

    if args.command == "show":
        for label in args.agent or _active(spec):
            print(
                json.dumps({label: payload(label, spec)}, indent=2, ensure_ascii=False)
            )
        return 0

    if args.command == "diff" and args.live is not None:
        labels = args.agent or []
        if len(labels) != 1:
            raise SystemExit("--live needs exactly one --agent")
        findings = diff(
            labels[0], json.loads(args.live.read_text(encoding="utf-8")), spec
        )
        _print_findings(labels[0], findings)
        return 1 if any(f.kind == "mismatch" for f in findings) else 0

    if args.command == "create" and not args.yes:
        print(json.dumps(payload(args.agent, spec), indent=2, ensure_ascii=False))
        print(
            "not created: this writes to the ElevenLabs account; add --yes",
            file=sys.stderr,
        )
        return 2

    api = api_factory()
    api.require_key()

    if args.command == "create":
        agent_id = asyncio.run(create(args.agent, api, spec))
        print(
            f"created {args.agent}: {agent_id}  (set {spec['agents'][args.agent]['env']})"
        )
        return 0

    if args.command == "read":
        ids = _ids([args.agent], spec, args.id)
        live = strip_account_fields(asyncio.run(read(ids[args.agent], api)))
        text = json.dumps(live, indent=2, ensure_ascii=False)
        if args.out is not None:
            args.out.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    if args.command == "snapshot":
        ids = _ids(args.agent or _active(spec), spec, None)
        asyncio.run(snapshot(ids, args.out, api, spec))
        print(f"wrote {len(ids)} agent configs to {args.out}")
        return 0

    labels = args.agent or _active(spec)
    ids = _ids(labels, spec, args.id)

    async def fetch_all() -> dict[str, dict[str, Any]]:
        return {label: await read(agent_id, api) for label, agent_id in ids.items()}

    worst = 0
    for label, live in asyncio.run(fetch_all()).items():
        findings = diff(label, live, spec)
        _print_findings(label, findings)
        if any(f.kind == "mismatch" for f in findings):
            worst = 1
    return worst
