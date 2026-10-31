"""The parity samples: fixed, stratified by EVA's record families, in EVA's order."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from eva_adapter import sample as samples
from eva_adapter.main import load_records
from eva_adapter.main import main

pytestmark = pytest.mark.unit


def test_the_sample_is_fixed_stratified_and_in_evas_order() -> None:
    every = load_records()
    first, again = samples.sample(), samples.sample()
    assert first == again  # fixed seed
    assert samples.sample(seed=1)["records"] != first["records"]
    for domain, ids in first["records"].items():
        assert len(ids) == 10 and len(set(ids)) == 10
        assert ids == [r for r in every[domain] if r in ids]  # EVA's order
        families = first["families"][domain]
        # Every family is sampled, in proportion, and the counts add up.
        assert all(f["sampled"] >= 1 for f in families.values())
        assert sum(f["sampled"] for f in families.values()) == 10
        assert {samples.family(domain, r)[0] for r in ids} == set(families)
    # EVA's categories at the commit (scenario_context.category).
    assert {k: v["records"] for k, v in first["families"]["itsm"].items()} == {
        "single_intent": 29,
        "double_intent": 14,
        "triple_intent": 14,
        "quadruple_intent": 14,
        "adversarial_intent": 9,
    }
    assert {k: v["records"] for k, v in first["families"]["medical_hr"].items()} == {
        "single_intent": 24,
        "adversarial": 9,
        "double_intent": 29,
        "triple_intent": 21,
    }
    # Within a family, distinct scenarios before siblings.
    picked = first["records"]["medical_hr"]
    assert len({samples.family("medical_hr", r)[1] for r in picked}) == 10
    # A label across families is drawn too: an ITSM escalation to a live agent.
    escalations = set(first["records"]["itsm"]) & samples.ITSM_ESCALATION
    assert escalations and first["required"]["itsm"] == {
        "escalate_to_live_agent": sorted(escalations, key=int)
    }


@pytest.mark.parametrize("seed", range(40))
def test_every_seed_draws_each_required_label_and_keeps_family_counts(
    seed: int,
) -> None:
    chosen = samples.sample(seed=seed)
    for domain, ids in chosen["records"].items():
        assert len(set(ids)) == 10
        counts = Counter(samples.family(domain, r)[0] for r in ids)
        sampled = {k: v["sampled"] for k, v in chosen["families"][domain].items()}
        assert counts == sampled
        for members in samples.REQUIRED.get(domain, {}).values():
            assert set(ids) & members


def test_all_is_every_record() -> None:
    every = samples.sample(per_domain=None)["records"]
    assert sum(len(ids) for ids in every.values()) == 213


def test_allocation_gives_every_family_one_then_the_rest_by_size() -> None:
    rng = random.Random(0)
    sizes = {"a": 9, "b": 10, "c": 3, "d": 7, "e": 8, "f": 4, "g": 9}  # airline
    counts = samples.allocate(10, sizes, rng)
    assert sum(counts.values()) == 10 and min(counts.values()) == 1
    assert samples.allocate(2, {"x": 30, "y": 1}, rng) == {"x": 1, "y": 1}
    assert samples.allocate(1, {"x": 30, "y": 1}, rng) == {"x": 1, "y": 0}
    assert samples.allocate(4, {"x": 30, "y": 1, "z": 1}, rng) == {
        "x": 2,
        "y": 1,
        "z": 1,
    }
    assert samples.allocate(99, {"x": 3}, rng) == {"x": 3}


def test_a_sample_file_is_checked_against_evas_ids(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"records": {"airline": ["1.2.2"], "itsm": []}}))
    assert samples.read_records(path) == {"airline": ["1.2.2"]}
    path.write_text(json.dumps({"records": {"airline": ["9.9.9"]}}))
    with pytest.raises(SystemExit, match="9.9.9"):
        samples.read_records(path)


def test_the_cli_writes_the_sample(tmp_path: Path) -> None:
    target = tmp_path / "airline.json"
    with pytest.raises(SystemExit) as done:
        main(
            ["sample", "--per-domain", "20", "--domains", "airline", "-o", f"{target}"]
        )
    assert done.value.code == 0
    assert json.loads(target.read_text()) == samples.sample(
        per_domain=20, domains=["airline"]
    )


def test_the_checked_in_sample_is_what_sample_draws() -> None:
    parity_file = json.loads(samples.PARITY_RECORDS.read_text())
    assert parity_file == samples.sample()
