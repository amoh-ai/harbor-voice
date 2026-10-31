"""The parity samples: which EVA records both sides of a comparison run.

::

    eva-adapter sample                     # parity_records.json: 10 per domain
    eva-adapter sample --per-domain 20 -o records.json
    eva-adapter sample --all --domains airline -o all-airline.json

A fixed seed, stratified within each domain by EVA's record families
(:func:`family`), ids in EVA's dataset order. ``convert --records`` and EVA's
``EVA_RECORD_IDS`` take the ids as they are.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence
from fractions import Fraction
from math import floor
from pathlib import Path
from typing import Any

from eva_adapter import EVA_COMMIT
from eva_adapter.main import DOMAINS
from eva_adapter.main import load_records

ADAPTER = Path(__file__).resolve().parents[2]
PARITY_RECORDS = ADAPTER / "parity_records.json"
"""The parity sample both sides run (``sample``)."""

SAMPLE_SEED = 300
"""Fixed: the sample is the same list every time it is drawn."""
PER_DOMAIN = 10
"""Parity: 10 records per domain, so every domain row has the same weight."""

ITSM_FAMILIES = (
    (1, 33, "single_intent"),
    (35, 54, "double_intent"),
    (56, 74, "triple_intent"),
    (75, 91, "quadruple_intent"),
    (95, 103, "adversarial_intent"),
)
"""ITSM's ids are plain numbers; EVA's ``scenario_context.category`` at the
commit is a band of them (single 29, double 14, triple 14, quadruple 14,
adversarial 9 records)."""
MEDICAL_FAMILIES = {
    "": "single_intent",
    "A": "adversarial",
    "D": "double_intent",
    "T": "triple_intent",
}
"""medical_hr's id prefix is EVA's ``scenario_context.category`` (24, 9, 29, 21)."""
ITSM_ESCALATION = frozenset(
    {"20", "21", "25", "37", "42", "54", "58", "65", "78", "85", "95"}
)
"""ITSM records with EVA's ``scenario_context.escalate_to_live_agent`` at the
commit: 11 of 80, in every category."""
REQUIRED = {"itsm": {"escalate_to_live_agent": ITSM_ESCALATION}}
"""Labels that cut across a domain's families: a sample of the domain holds at
least one record of each (:func:`_require`)."""


def family(domain: str, record_id: str) -> tuple[str, str]:
    """A record's family and scenario, from its EVA id.

    - airline ``<family>.<scenario>.<variant>`` (EVA has no category there;
      the id's first number groups the scenarios, 7 families; family 7 is
      EVA's adversarial callers: false claims, no confirmation code, another
      passenger's booking);
    - medical_hr ``<N>.<v>``, ``A<N>``, ``D<N>.<v>``, ``T<N>.<v>``;
    - itsm: :data:`ITSM_FAMILIES`, each record its own scenario.
    """
    if domain == "airline":
        parts = record_id.split(".")
        return parts[0], ".".join(parts[:2])
    if domain == "medical_hr":
        prefix = record_id[0] if record_id[0].isalpha() else ""
        return MEDICAL_FAMILIES[prefix], record_id.split(".")[0]
    if domain == "itsm":
        number = int(record_id)
        for low, high, name in ITSM_FAMILIES:
            if low <= number <= high:
                return name, record_id
    raise ValueError(f"no family for {domain} record {record_id!r}")


def allocate(
    count: int, sizes: Mapping[str, int], rng: random.Random
) -> dict[str, int]:
    """How many records each family gets: one each when there are enough,
    the rest in proportion to what remains (largest remainder; exact ties
    broken by the seeded ``rng``)."""
    if count >= sum(sizes.values()):
        return dict(sizes)
    base = dict.fromkeys(sizes, 1 if count >= len(sizes) else 0)
    left = count - sum(base.values())
    spare = {f: sizes[f] - base[f] for f in sizes}
    total = sum(spare.values())
    quotas = {f: Fraction(left * spare[f], total) for f in sizes}
    out = {f: base[f] + floor(quotas[f]) for f in sizes}
    tiebreak = {f: rng.random() for f in sizes}
    order = sorted(sizes, key=lambda f: (floor(quotas[f]) - quotas[f], tiebreak[f]))
    for f in order[: count - sum(out.values())]:
        out[f] += 1
    return out


def _spread(domain: str, ids: list[str], count: int, rng: random.Random) -> list[str]:
    """``count`` of ``ids``, one scenario at a time, so siblings come last."""
    pools: dict[str, list[str]] = defaultdict(list)
    for record_id in ids:
        pools[family(domain, record_id)[1]].append(record_id)
    order = list(pools)
    rng.shuffle(order)
    for name in order:
        rng.shuffle(pools[name])
    chosen: list[str] = []
    while len(chosen) < count:
        for name in order:
            if pools[name] and len(chosen) < count:
                chosen.append(pools[name].pop())
    return chosen


def _require(
    domain: str, picked: set[str], by_family: Mapping[str, list[str]], seed: int
) -> None:
    """Give ``picked`` a record of each :data:`REQUIRED` label it lacks, by
    swapping one record for another of the same family, so family counts hold.

    Its own seeded ``rng``: the other domains' draws stay as they were."""
    rng = random.Random(f"{seed}-{domain}-required")
    for label, members in REQUIRED.get(domain, {}).items():
        if picked & members:
            continue
        swaps = [
            name
            for name, ids in by_family.items()
            if set(ids) & members and set(ids) & (picked - members)
        ]
        if not swaps:
            raise ValueError(f"cannot give the {domain} sample a {label} record")
        name = rng.choice(swaps)
        picked.remove(rng.choice(sorted(set(by_family[name]) & picked)))
        picked.add(rng.choice(sorted(set(by_family[name]) & members)))


def sample(
    *,
    per_domain: int | None = PER_DOMAIN,
    domains: Sequence[str] = DOMAINS,
    seed: int = SAMPLE_SEED,
) -> dict[str, Any]:
    """A fixed-seed sample of EVA's records, stratified within each domain.

    Every domain gets ``per_domain`` records (equal weight per domain row),
    spread over EVA's record families
    (:func:`family`) in proportion to their size, each family at least once,
    and within a family over distinct scenarios first. Labels that cut across
    families (:data:`REQUIRED`) are then each present at least once. ``None``
    takes every record. The ids come back in EVA's dataset order.
    """
    rng = random.Random(seed)
    every = load_records()
    out: dict[str, Any] = {
        "source": f"ServiceNow/eva@{EVA_COMMIT} (template/records.json)",
        "seed": seed,
        "per_domain": per_domain,
        "families": {},
        "records": {},
    }
    for domain in domains:
        ids = every[domain]
        by_family: dict[str, list[str]] = defaultdict(list)
        for record_id in ids:
            by_family[family(domain, record_id)[0]].append(record_id)
        sizes = {f: len(members) for f, members in by_family.items()}
        counts = allocate(len(ids) if per_domain is None else per_domain, sizes, rng)
        picked: set[str] = set()
        for name, members in by_family.items():
            picked.update(_spread(domain, members, counts[name], rng))
        _require(domain, picked, by_family, seed)
        out["families"][domain] = {
            name: {"records": sizes[name], "sampled": counts[name]} for name in sizes
        }
        if domain in REQUIRED:
            out.setdefault("required", {})[domain] = {
                label: [r for r in ids if r in picked and r in members]
                for label, members in REQUIRED[domain].items()
            }
        out["records"][domain] = [r for r in ids if r in picked]
    return out


def read_records(path: Path) -> dict[str, list[str]]:
    """A sample file's records per domain, checked against EVA's ids."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", data)
    every = load_records()
    out: dict[str, list[str]] = {}
    for domain, ids in records.items():
        if domain not in every:
            raise SystemExit(f"{path}: {domain!r} is not an EVA domain")
        unknown = sorted(set(ids) - set(every[domain]))
        if unknown:
            raise SystemExit(f"{path}: not EVA {domain} records: {', '.join(unknown)}")
        if ids:
            out[domain] = list(ids)
    if not out:
        raise SystemExit(f"{path} names no records")
    return out


# ------------------------------------------------------------------ CLI


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """``sample``'s options (the parity runner's ``sample`` takes the same)."""
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--per-domain", type=int, default=PER_DOMAIN)
    size.add_argument("--all", action="store_true", help="every record (213)")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("-o", "--out", type=Path)


def write_sample(args: argparse.Namespace) -> Path:
    """Draw the sample the options name and write it (by default the
    checked-in :data:`PARITY_RECORDS`)."""
    chosen = sample(
        per_domain=None if args.all else args.per_domain,
        domains=args.domains,
        seed=args.seed,
    )
    target = args.out or PARITY_RECORDS
    target.write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    counts = {d: len(ids) for d, ids in chosen["records"].items()}
    print(f"wrote {target}: {counts}")
    return target


def main_sample(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="eva-adapter sample",
        description="write the records both sides of a parity comparison run",
    )
    add_arguments(parser)
    write_sample(parser.parse_args(list(argv)))
    return 0
