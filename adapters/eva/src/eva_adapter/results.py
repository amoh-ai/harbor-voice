"""Reading EVA's results, as a harbor trial and EVA's own CLI leave them.

A harbor trial of an EVA task keeps EVA's outputs under ``agent/call/eva/``
(:func:`eva_dir`): the driver's ``summary.json``, the counted record's
``metrics.json`` and the trial's EVA run directory ``run/``; the verifier's
``verifier/result.json`` sits beside them. An EVA run directory (EVA's CLI,
or a trial's ``run/``) holds ``config.json``, ``records/<id>/trial_<k>/``,
each failed attempt archived as ``<id>_failed_attempt_<n>``, and
``evaluation_summary.json``, which EVA writes last.

Standard library only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ERROR_CHARS = 300

AUDIO_METRICS = ("agent_speech_fidelity", "user_speech_fidelity")
"""EVA's audio judges: ``user_speech_fidelity`` is a validation metric (it
runs on every attempt), ``agent_speech_fidelity`` a scored one."""
EVA_AUDIO_JUDGE: dict[str, Any] = {
    "model": "gemini-3-flash-preview",
    "params": {"temperature": 0.0, "max_tokens": 40000, "reasoning_effort": "minimal"},
}
"""EVA's own (``AudioJudgeMetric.default_model`` / ``default_params`` @039efa0a)."""

_ARCHIVE = re.compile(r"^(?P<name>.+)_failed_attempt_(?P<n>\d+)$")


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def metric_errors(metrics: Any) -> dict[str, str]:
    """A ``metrics.json``'s metrics with an error (EVA's ``error is not None``),
    each with the start of its message."""
    scores: Any = metrics.get("metrics") if isinstance(metrics, dict) else None
    if not isinstance(scores, dict):
        return {}
    out: dict[str, str] = {}
    for name, score in sorted(scores.items()):
        error = score.get("error") if isinstance(score, dict) else None
        if error is not None:
            out[str(name)] = str(error)[:ERROR_CHARS]
    return out


def archives(run_dir: Path, output_id: str) -> list[Path]:
    """An output id's archived attempts (``<id>_failed_attempt_<n>``), oldest first."""
    record = run_dir / "records" / output_id
    if not record.parent.is_dir():
        return []
    found = []
    for path in record.parent.iterdir():
        match = _ARCHIVE.match(path.name)
        if path.is_dir() and match and match["name"] == record.name:
            found.append((int(match["n"]), path))
    return [path for _, path in sorted(found)]


def attempt_errors(run_dir: Path, output_id: str) -> list[list[str]]:
    """Each archived attempt's errored metrics, oldest first. An archived
    attempt's ``metrics.json`` holds what validation computed, so an error
    there is a validation judge's, and it failed the attempt."""
    return [
        sorted(metric_errors(load(path / "metrics.json")))
        for path in archives(run_dir, output_id)
    ]


def eva_dir(trial_dir: Path) -> Path:
    """Where the driver writes a trial's EVA outputs."""
    return trial_dir / "agent" / "call" / "eva"


def harbor_outcome(trial_dir: Path) -> dict[str, Any]:
    """What a harbor trial came to, from the driver's and the verifier's
    files, never from harbor's exception (``RewardFileNotFoundError`` is how
    harbor reports a trial without a reward, which is a result here):

    - ``errored``: the driver left no summary (the trial broke, infrastructure);
    - ``not_validated``: no attempt passed EVA's validation (EVA scores no
      such record); ``attempt_errors`` says which validation judges errored;
    - ``ungraded``: validated, but no metrics came back;
    - ``judge_errored``: graded, but a metric has an error (re-gradable);
    - ``graded``.

    Also: the calls made (attempts) and the reward.
    """
    eva = eva_dir(trial_dir)
    summary = load(eva / "summary.json")
    verifier = load(trial_dir / "verifier" / "result.json")
    out: dict[str, Any] = {
        "status": "errored",
        "output_id": None,
        "calls": len(list(eva.glob("attempt-*/attempt.json"))),
        "reward": None,
        "errors": {},
        "attempt_errors": [],
    }
    if not isinstance(summary, Mapping):
        return out
    output_id = str(
        summary.get("output_id")
        or f"{summary.get('record_id')}/trial_{summary.get('trial', 0)}"
    )
    out["output_id"] = output_id
    out["calls"] = int(summary.get("attempts_used") or out["calls"])
    out["attempt_errors"] = attempt_errors(eva / "run", output_id)
    if not summary.get("validated"):
        out["status"] = "not_validated"
        return out
    metrics = load(eva / "run" / "records" / output_id / "metrics.json")
    if not isinstance(metrics, Mapping):
        metrics = load(eva / "metrics.json")
    if not isinstance(metrics, Mapping):
        out["status"] = "ungraded"
        return out
    out["errors"] = metric_errors(metrics)
    out["status"] = "judge_errored" if out["errors"] else "graded"
    if isinstance(verifier, Mapping) and verifier.get("graded"):
        out["reward"] = verifier.get("reward")
    else:
        out["reward"] = summary.get("reward")
    return out


def audio_judges(metric_configs: Any) -> str:
    """The audio judges an EVA run's ``metric_configs`` asks for, as a label
    (``<model>@<reasoning effort>``): ``gemini-3-flash-preview@minimal``,
    EVA's own, when it overrides nothing."""
    configs = metric_configs if isinstance(metric_configs, Mapping) else {}
    found = set()
    for metric in AUDIO_METRICS:
        config = configs.get(metric) or {}
        model = config.get("audio_judge_model", EVA_AUDIO_JUDGE["model"])
        params = {**EVA_AUDIO_JUDGE["params"], **(config.get("judge_params") or {})}
        found.add(f"{model}@{params.get('reasoning_effort')}")
    return " / ".join(sorted(found))
