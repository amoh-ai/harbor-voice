"""The parity sample: fixed, proportional, and the checked-in file is what it draws."""

from __future__ import annotations

import json
from pathlib import Path

from tau_voice.main import PARITY_TASKS
from tau_voice.sample import SAMPLE
from tau_voice.sample import main
from tau_voice.sample import sample

TEST_SPLIT = Path(__file__).parent / "fixtures" / "tau2_test_split.json"
"""tau2's ``test`` split at the pinned commit, its task ids only."""


def _tau2(root: Path, splits: dict[str, list[str]]) -> Path:
    for domain, ids in splits.items():
        folder = root / "data" / "tau2" / "domains" / domain
        folder.mkdir(parents=True)
        (folder / "split_tasks.json").write_text(json.dumps({"test": ids}))
    return root


def test_the_sample_is_fixed_proportional_and_in_split_order(tmp_path: Path) -> None:
    splits = {d: [f"{d}-{i}" for i in range(count * 3)] for d, count in SAMPLE.items()}
    root = _tau2(tmp_path, splits)
    first, again = sample(root), sample(root)
    assert first == again  # fixed seed
    for domain, count in SAMPLE.items():
        ids = first[domain]
        assert len(ids) == count and len(set(ids)) == count
        assert ids == sorted(ids, key=lambda i: int(i.rsplit("-", 1)[1]))
    assert sample(root, seed=1) != first


def test_the_checked_in_sample_is_what_sample_draws(tmp_path: Path) -> None:
    split = json.loads(TEST_SPLIT.read_text(encoding="utf-8"))["test"]
    assert {d: len(ids) for d, ids in split.items()} == {
        "airline": 20,
        "retail": 40,
        "telecom": 40,
    }
    root = _tau2(tmp_path / "tau2", split)
    checked_in = json.loads(PARITY_TASKS.read_text(encoding="utf-8"))
    assert sample(root) == checked_in

    out = tmp_path / "parity_tasks.json"
    main(["--tau2-root", str(root), "-o", str(out)])
    assert out.read_text(encoding="utf-8") == PARITY_TASKS.read_text(encoding="utf-8")
