"""What a running harbor job is doing — ``harbor-voice status <job> [--watch]``.

Reads only the job folder harbor writes, so it works for any backend and
while the job runs: harbor's own counts from the job's ``result.json``, and
per trial its phase (the last line of ``trial.log``, where ``VoiceCallAgent``
writes install, call and per-attempt lines), its reward, and its error.
Tasks that declare ``[metadata] domain`` are grouped by it.
"""

from __future__ import annotations

import json
import statistics
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrialState:
    name: str
    domain: str
    done: bool
    reward: float | None
    error: str | None
    attempts: int | None
    elapsed_s: float
    phase: str


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _domain(trial_config: dict[str, Any] | None) -> str:
    task = Path(str(((trial_config or {}).get("task") or {}).get("path") or ""))
    try:
        meta = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "all"
    return str((meta.get("metadata") or {}).get("domain") or "all")


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def _last_line(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return next((line.strip() for line in reversed(lines) if line.strip()), "")


def trial_state(trial: Path, now: float) -> TrialState:
    config = _json(trial / "config.json")
    result = _json(trial / "result.json")
    outcome = _json(trial / "agent" / "call" / "outcome.json")
    attempts = len(((outcome or {}).get("summary") or {}).get("attempts") or []) or None
    if result is None:
        return TrialState(
            name=trial.name,
            domain=_domain(config),
            done=False,
            reward=None,
            error=None,
            attempts=attempts,
            elapsed_s=now - (trial / "config.json").stat().st_mtime,
            phase=_last_line(trial / "trial.log"),
        )
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    exception = result.get("exception_info") or {}
    error = None
    if exception:
        message = str(exception.get("exception_message") or "").splitlines()
        error = f"{exception.get('exception_type')}: {message[0] if message else ''}"
    return TrialState(
        name=trial.name,
        domain=_domain(config),
        done=True,
        reward=None if reward is None else float(reward),
        error=error,
        attempts=attempts,
        elapsed_s=_seconds(result.get("started_at"), result.get("finished_at")) or 0.0,
        phase="done",
    )


def _clock(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def render(job: Path, *, now: float | None = None, limit: int = 20) -> str:
    """A job's status as text."""
    now = time.time() if now is None else now
    summary = _json(job / "result.json") or {}
    stats = summary.get("stats") or {}
    concurrency = (_json(job / "config.json") or {}).get("n_concurrent_trials") or 1
    trials = [trial_state(path.parent, now) for path in sorted(job.glob("*/trial.log"))]
    done = [t for t in trials if t.done]
    running = [t for t in trials if not t.done]
    total = summary.get("n_total_trials") or len(trials)
    started = summary.get("started_at")
    finished = summary.get("finished_at")
    elapsed = (
        _seconds(started, finished)
        if finished
        else now - datetime.fromisoformat(started).timestamp()
        if started
        else None
    )

    durations = [t.elapsed_s for t in done if t.elapsed_s]
    remaining = max(0, total - len(done))
    eta = statistics.fmean(durations) * remaining / concurrency if durations else None
    errored = [t for t in done if t.error or t.reward is None]
    head = [
        f"{job.name}: done {len(done)}/{total} · running {len(running)} · "
        f"errored {len(errored)} · retries {stats.get('n_retries', 0)}"
        + (f" · elapsed {_clock(elapsed)}" if elapsed is not None else "")
        + (f" · ETA ~{_clock(eta)}" if eta is not None and remaining else ""),
    ]

    by_domain: dict[str, list[TrialState]] = defaultdict(list)
    for trial in done:
        by_domain[trial.domain].append(trial)
    for domain in sorted(by_domain):
        rows = by_domain[domain]
        rewards = [t.reward for t in rows if t.reward is not None]
        reran = [t for t in rows if (t.attempts or 1) > 1]
        head.append(
            f"  {domain:<10} reward "
            + (f"{statistics.fmean(rewards):.3f}" if rewards else "n/a")
            + f" ({len(rewards)} graded)"
            + (f" · reran {len(reran)}/{len(rows)}" if rows else "")
        )

    if running:
        head.append("running:")
        for trial in sorted(running, key=lambda t: -t.elapsed_s)[:limit]:
            head.append(
                f"  {trial.name:<34} {_clock(trial.elapsed_s):>6}  {trial.phase[:90]}"
            )
        if len(running) > limit:
            head.append(f"  … and {len(running) - limit} more")
    if errored:
        head.append("errored:")
        for trial in errored[:limit]:
            head.append(f"  {trial.name:<34} {(trial.error or 'no grade')[:100]}")
    return "\n".join(head)


def watch(job: Path, every_s: float) -> None:
    while True:
        print("\x1b[2J\x1b[H" + render(job), flush=True)
        summary = _json(job / "result.json") or {}
        if summary.get("finished_at"):
            return
        time.sleep(every_s)
