"""Parity: the harbor side against EVA's own run, in harbor's format.

::

    eva-adapter compare --harbor <job> [<job> ...] --original <EVA run> [...]
                        [--records <sample>] [--trials 3] [<condition flags>]
                        [-o parity_experiment.json]

Reads harbor job folders (``harbor-voice jobs start``; each trial's
``agent/call/eva/``) and EVA run folders (EVA itself: ``config.json``,
``records/<id>/trial_<k>/``, ``evaluation_summary.json``), and reports, per
domain and pooled (``overall``), for each side:

- EVA's record composites, ``EVA-A_pass``, ``EVA-X_pass`` and
  ``EVA-overall_pass``: the mean over each run's graded trials (run k is
  EVA's trial k), then mean ± sample SEM over the runs with the per-run
  values, as harbor's ``parity_experiment.json`` states them (its rule: the
  two sides match when their run ranges overlap); and pass@k / pass^k by
  EVA's estimator (``pass_at_k.py``, over records with every trial scored);
- the six components of the composites, mean ± SEM over runs;
- diagnostics that explain a gap without being a parity claim:
  validation (graded, judge errored, not validated, errored, missing),
  attempts and reruns, the calls made against the trials planned, end
  reasons (the counted attempt's, and every attempt's), call length, the
  caller's turns and words per turn, the agent's response speed;
- tokens, every attempt included: the agent's (``pipecat_metrics.jsonl``,
  which both sides write the same way), the caller's LLM (each ElevenLabs
  conversation record) and the judges' (``judge_token_usage.json``). Counts,
  never dollars.

One judge at a time. The audio judges every trial's EVA ``config.json``
recorded (its ``metric_configs``: EVA's ``gemini-3-flash-preview`` unless it
names another, :func:`~eva_adapter.results.audio_judges`) must be one and
the same, so trials judged differently are never pooled, and the report
names them.

One condition at a time. The EVA perturbation condition compared (clean
unless named) is checked against what every trial recorded it ran (harbor's
``summary.json``, EVA's ``config.json``), trials of another condition are
never pooled, and a perturbed comparison says so in
``parity_experiment.json``'s notes: EVA's noise is unseeded, so the sides
are compared statistically, never trial by trial.

Nothing is dropped silently: a trial expected on a side and absent is
``missing``; a harbor trial whose driver left no summary (it broke) is
``errored``; a trial no attempt validated is ``not_validated`` (EVA scores no
such record), also when harbor reports it as an error only because the
verifier wrote no reward for it; a validated one whose metrics are absent is
``ungraded``. None of these enters a mean; all are counted. A graded trial
with a metric whose judge failed is ``judge_errored``: its other scores count
as EVA counts them, and the composites that need the failed metric are unset
until the metric is re-graded from the recording (EVA's own rerun of failed
metrics); a trial whose folders record a re-grade says so (``regraded``).
``eva-adapter compare`` writes ``parity_experiment.json`` (harbor's format)
and ``.report.json`` (everything above, and every trial).

:func:`compare_folders` takes further sides and an ``extend`` hook, for an
analysis that adds to the report (more statistics, more tables) without this
module carrying it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from pathlib import Path
from typing import Any

from eva_adapter.conditions import CLEAN
from eva_adapter.conditions import ConditionError
from eva_adapter.conditions import PerturbationConfig
from eva_adapter.conditions import as_record
from eva_adapter.conditions import from_record
from eva_adapter.conditions import label_of
from eva_adapter.main import add_condition_args
from eva_adapter.main import condition_from_args
from eva_adapter.main import load_records
from eva_adapter.main import task_dir_name
from eva_adapter.results import audio_judges
from eva_adapter.results import harbor_outcome
from eva_adapter.results import metric_errors
from eva_adapter.trial import NUM_TRIALS

GRADED = "graded"
JUDGE_ERRORED = "judge_errored"
"""Graded, but a metric has an error (a judge failed): re-gradable."""
UNGRADED = "ungraded"
NOT_VALIDATED = "not_validated"
ERRORED = "errored"
MISSING = "missing"
STATUSES = (GRADED, JUDGE_ERRORED, UNGRADED, NOT_VALIDATED, ERRORED, MISSING)
SCORED = frozenset({GRADED, JUDGE_ERRORED})
"""Trials whose scores count: EVA's means take every value a record has."""
_PREFERENCE = {status: rank for rank, status in enumerate(STATUSES)}

COMPOSITES = ("EVA-A_pass", "EVA-X_pass", "EVA-overall_pass")
COMPONENTS = (
    "task_completion",
    "faithfulness",
    "agent_speech_fidelity",
    "turn_taking",
    "conciseness",
    "conversation_progression",
)
"""The components of EVA-A (the first three) and EVA-X (``aggregation.py``)."""
SIDES = ("harbor", "original")
PAIRS = (("harbor", "original"),)
ADAPTED_SIZE = 213
ORIGINAL_REPO = "https://github.com/ServiceNow/eva"

_ARCHIVE = re.compile(r"^(?P<oid>.+)_failed_attempt_(?P<n>\d+)$")


@dataclass(slots=True)
class Trial:
    """One trial (one record, one EVA trial index) on one side."""

    side: str
    domain: str
    record_id: str
    trial: int
    status: str
    composites: dict[str, float | None] = field(default_factory=dict)
    components: dict[str, float | None] = field(default_factory=dict)
    attempts: int | None = None
    end_reason: str | None = None
    """The counted attempt's (the last attempt's when none validated)."""
    end_reasons: list[str] = field(default_factory=list)
    """Every attempt's, in order."""
    call_s: float | None = None
    calls_s: float = 0.0
    """Every attempt's call, summed."""
    caller_turns: int | None = None
    caller_words: int | None = None
    response_speeds: list[float] = field(default_factory=list)
    time_limit_accepted: bool = False
    error: str | None = None
    tokens: dict[str, float] = field(default_factory=dict)
    condition: str | None = None
    """The perturbation condition the trial's own record says it ran
    (``None``: it left no record of it)."""
    judges: str | None = None
    """The audio judges its EVA run's ``config.json`` names
    (:func:`eva_adapter.results.audio_judges`; ``None``: no config)."""
    calls: int = 0
    """The calls it made: every attempt (a superseded duplicate's too)."""
    metric_errors: list[str] = field(default_factory=list)
    """The counted record's metrics with an error (``judge_errored``)."""
    attempt_errors: list[list[str]] = field(default_factory=list)
    """Each failed attempt's errored validation metrics, oldest first: a
    judge error there failed the attempt, and EVA called again."""
    regraded: dict[str, Any] | None = None
    """A re-grade's record, when its folders keep one (:data:`TrialRegrade`):
    the metrics, whether they are fixed."""
    harbor_exception: str | None = None
    """What harbor raised for a trial that is not ``errored`` here (a trial
    without a reward is ``RewardFileNotFoundError`` to harbor)."""

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.domain, self.record_id, self.trial)


TrialRegrade = Callable[[Path], dict[str, Any] | None]
"""A harbor trial's re-grade record, from its folder: ``metrics``, ``fixed``
(and anything else to keep), or ``None``. EVA leaves none of its own."""
RunRegrades = Callable[[Path], Mapping[str, dict[str, Any]]]
"""An EVA run's re-grade records, from its directory, per output id."""
Extension = Callable[[dict[str, Any], list[Trial]], str]
"""Adds to a report from its trials; returns tables to print (``""``: none)."""


def no_regrade(trial_dir: Path) -> dict[str, Any] | None:
    return None


def no_regrades(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def _lines(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            yield entry


# ------------------------------------------------------------------ records


def score(metric: Any) -> float | None:
    """A metric's value as EVA reads it (``RecordMetrics.get_score``)."""
    if not isinstance(metric, Mapping) or metric.get("error"):
        return None
    value = metric.get("normalized_score")
    return _float(metric.get("score") if value is None else value)


def metric_values(
    metrics: Mapping[str, Any],
) -> tuple[dict[str, float | None], dict[str, float | None], list[float]]:
    """A ``metrics.json``: the composites, the components, per-turn response speeds."""
    scores = metrics.get("metrics") or {}
    aggregates = metrics.get("aggregate_metrics") or {}
    speed = (scores.get("response_speed") or {}).get("details") or {}
    return (
        {name: _float(aggregates.get(name)) for name in COMPOSITES},
        {name: score(scores.get(name)) for name in COMPONENTS},
        [float(v) for v in speed.get("per_turn_speeds") or []],
    )


def record_facts(record_dir: Path) -> dict[str, Any]:
    """One attempt's record directory (EVA's files; both sides write them)."""
    result = _json(record_dir / "result.json") or {}
    events = record_dir / "user_simulator_events.jsonl"
    turns = words = 0
    for event in _lines(events):
        data = event.get("data") or {}
        if event.get("type") == "user_speech" and data.get("is_final", True):
            text = str(data.get("text") or "")
            if text.strip():
                turns += 1
                words += len(text.split())
    return {
        "end_reason": result.get("conversation_ended_reason"),
        "call_s": _float(result.get("duration_seconds")),
        "time_limit_accepted": bool(result.get("time_limit_accepted")),
        "caller_turns": turns if events.exists() else None,
        "caller_words": words if events.exists() else None,
    }


def attempt_tokens(record_dir: Path) -> dict[str, float]:
    """One attempt's tokens: the agent's, and the caller's from ElevenLabs."""
    from harbor_voice.providers.elevenlabs.agents import usage_records

    tokens: dict[str, float] = defaultdict(float)
    for entry in _lines(record_dir / "pipecat_metrics.jsonl"):
        if entry.get("type") == "LLMTokenUsageMetricsData":
            value = entry.get("value") or {}
            tokens["agent_input"] += int(value.get("prompt_tokens") or 0)
            tokens["agent_output"] += int(value.get("completion_tokens") or 0)
    details = _json(record_dir / "elevenlabs_conversation_details.json")
    if isinstance(details, Mapping):
        for usage in usage_records(details):
            if usage.component == "caller":
                tokens["caller_seconds"] += usage.seconds or 0
                tokens["caller_calls"] += 1
            elif usage.component in ("caller.llm", "caller.llm.initiated"):
                # Generations that went through, and every one started
                # (speculative turns too): ElevenLabs' two views, kept apart.
                view = (
                    "caller" if usage.component == "caller.llm" else "caller_initiated"
                )
                tokens[f"{view}_input"] += usage.input_tokens or 0
                tokens[f"{view}_output"] += usage.output_tokens or 0
    return dict(tokens)


def judge_tokens(run_dir: Path) -> dict[str, float]:
    """An EVA run's judges: ``judge_token_usage.json``, every metric."""
    data = _json(run_dir / "judge_token_usage" / "judge_token_usage.json") or {}
    tokens: dict[str, float] = defaultdict(float)
    for entry in data.values():
        if isinstance(entry, Mapping):
            tokens["judge_input"] += int(entry.get("total_input_tokens") or 0)
            tokens["judge_output"] += int(entry.get("total_output_tokens") or 0)
            tokens["judge_calls"] += int(entry.get("num_calls") or 0)
    return dict(tokens)


def output_dirs(records: Path) -> dict[str, dict[str, Any]]:
    """EVA's ``records/``: per output id, the counted directory and the
    archived attempts (``<id>_failed_attempt_<n>``), oldest first.

    Output ids are ``<record>/trial_<k>``, or ``<record>`` when EVA ran one
    trial per record.
    """
    found: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"counted": None, "archives": []}
    )
    if not records.is_dir():
        return {}
    for d in sorted(records.iterdir()):
        if not d.is_dir():
            continue
        archived = _ARCHIVE.match(d.name)
        if archived:
            found[archived["oid"]]["archives"].append((int(archived["n"]), d))
            continue
        trials = [
            s for s in sorted(d.iterdir()) if s.is_dir() and s.name.startswith("trial_")
        ]
        if not trials:
            found[d.name]["counted"] = d
        for s in trials:
            archived = _ARCHIVE.match(s.name)
            if archived:
                found[f"{d.name}/{archived['oid']}"]["archives"].append(
                    (int(archived["n"]), s)
                )
            else:
                found[f"{d.name}/{s.name}"]["counted"] = s
    for entry in found.values():
        entry["archives"] = [path for _, path in sorted(entry["archives"])]
    return dict(found)


def _split(output_id: str) -> tuple[str, int]:
    record_id, sep, k = output_id.rpartition("/trial_")
    return (record_id, int(k)) if sep else (output_id, 0)


def _attempts(trial: Trial, entry: Mapping[str, Any]) -> None:
    """Fill a trial's per-attempt facts from its EVA output directories."""
    dirs = [*entry["archives"], *([entry["counted"]] if entry["counted"] else [])]
    facts = [record_facts(d) for d in dirs]
    # The driver's own count when it left one (harbor), else EVA's folders.
    trial.attempts = trial.attempts or len(dirs) or None
    trial.calls = trial.attempts or 0
    trial.attempt_errors = [
        sorted(metric_errors(_json(d / "metrics.json"))) for d in entry["archives"]
    ]
    trial.end_reasons = [str(f["end_reason"]) for f in facts]
    trial.calls_s = sum(f["call_s"] or 0.0 for f in facts)
    for d in dirs:
        for name, n in attempt_tokens(d).items():
            trial.tokens[name] = trial.tokens.get(name, 0) + n
    if facts:
        last = facts[-1]
        trial.end_reason = last["end_reason"]
        trial.call_s = last["call_s"]
        trial.time_limit_accepted = last["time_limit_accepted"]
        trial.caller_turns = last["caller_turns"]
        trial.caller_words = last["caller_words"]
    counted = entry["counted"]
    metrics = _json(counted / "metrics.json") if counted else None
    if isinstance(metrics, Mapping):
        trial.composites, trial.components, trial.response_speeds = metric_values(
            metrics
        )
        trial.metric_errors = sorted(metric_errors(metrics))


def _condition(record: Any, where: Path) -> str:
    """The condition a perturbation record names (EVA's or a summary's)."""
    try:
        return label_of(from_record(record if isinstance(record, Mapping) else None))
    except ConditionError as exc:
        raise SystemExit(f"{where}: {exc}") from exc


def _judges(run_dir: Path) -> str | None:
    """The audio judges an EVA run's ``config.json`` names, if it has one."""
    config = _json(run_dir / "config.json")
    if not isinstance(config, Mapping):
        return None
    return audio_judges(config.get("metric_configs"))


def eva_trials(
    run_dir: Path, side: str, *, regrades: RunRegrades = no_regrades
) -> list[Trial]:
    """An EVA run's trials: EVA's own record of what it ran."""
    config = _json(run_dir / "config.json") or {}
    domain = str(config.get("domain") or "unknown")
    condition = _condition(config.get("perturbation"), run_dir / "config.json")
    judges = _judges(run_dir)
    summary = _json(run_dir / "evaluation_summary.json")
    sim = (summary.get("simulation") or summary) if isinstance(summary, dict) else {}
    passed = set(sim.get("successful_record_ids") or [])
    failed = set(sim.get("failed_record_ids") or [])
    entries = output_dirs(run_dir / "records")
    regraded = regrades(run_dir)
    trials = []
    for output_id in sorted(set(entries) | passed | failed):
        entry = entries.get(output_id, {"counted": None, "archives": []})
        record_id, k = _split(output_id)
        counted = entry["counted"]
        if counted is not None and (counted / "metrics.json").exists():
            errors = metric_errors(_json(counted / "metrics.json"))
            status = JUDGE_ERRORED if errors else GRADED
        elif output_id in failed:
            status = NOT_VALIDATED
        elif output_id in passed:
            status = UNGRADED
        else:
            status = MISSING
        trial = Trial(
            side, domain, record_id, k, status, condition=condition, judges=judges
        )
        if status == MISSING:
            trial.error = (
                "EVA's run has not settled this trial (no evaluation_summary entry)"
            )
        _attempts(trial, entry)
        trial.regraded = regraded.get(output_id)
        trials.append(trial)
    return trials


def task_index(
    perturbation: PerturbationConfig | None = None,
) -> dict[str, tuple[str, str]]:
    """Task folder name → (domain, record id), for trials that left no summary
    (a perturbed task's folder ends with its condition)."""
    return {
        task_dir_name(domain, record_id, perturbation): (domain, record_id)
        for domain, ids in load_records().items()
        for record_id in ids
    }


def _error(result: Mapping[str, Any]) -> str | None:
    exception = result.get("exception_info") or {}
    if not exception:
        return None
    lines = [
        line.strip(" │|")
        for line in str(exception.get("exception_message") or "").splitlines()
    ]
    meaningful = [line for line in lines if len(line) > 3]
    return f"{exception.get('exception_type')}: {(meaningful or [''])[-1][:300]}"


def harbor_trials(
    job: Path,
    index: Mapping[str, tuple[str, str]] | None = None,
    *,
    regrade: TrialRegrade = no_regrade,
) -> list[Trial]:
    """A harbor job's trials; one still running is not read (it is ``missing``).

    Each is classified from the driver's and the verifier's files
    (:func:`eva_adapter.results.harbor_outcome`), not from harbor's exception:
    harbor raises ``RewardFileNotFoundError`` for a trial EVA never validated,
    or graded with a judge error, which are results here."""
    index = task_index() if index is None else index
    trials = []
    for trial_dir in sorted(p.parent for p in job.glob("*/trial.log")):
        result = _json(trial_dir / "result.json")
        if not isinstance(result, Mapping):
            continue
        config = _json(trial_dir / "config.json") or {}
        seed = ((config.get("agent") or {}).get("kwargs") or {}).get("seed")
        eva = trial_dir / "agent" / "call" / "eva"
        summary = _json(eva / "summary.json")
        outcome = harbor_outcome(trial_dir)
        if not isinstance(summary, Mapping):
            task = str(
                (config.get("task") or {}).get("path") or result.get("task_name") or ""
            )
            name = task.replace("\\", "/").rpartition("/")[2]
            domain, record_id = index.get(name, ("unknown", name))
            trial = Trial("harbor", domain, record_id, int(seed or 0), ERRORED)
            trial.error = _error(result) or "the trial left no EVA summary"
            trial.calls = outcome["calls"]
            trials.append(trial)
            continue
        metrics = _json(eva / "metrics.json")
        status = {
            "graded": GRADED,
            "judge_errored": JUDGE_ERRORED,
            "not_validated": NOT_VALIDATED,
            "ungraded": UNGRADED,
        }[outcome["status"]]
        trial = Trial(
            "harbor",
            str(summary["domain"]),
            str(summary["record_id"]),
            int(summary.get("trial", seed or 0)),
            status,
            attempts=summary.get("attempts_used"),
            condition=_condition(summary.get("perturbation"), eva / "summary.json"),
            judges=_judges(eva / "run"),
            metric_errors=sorted(outcome["errors"]),
            regraded=regrade(trial_dir),
            harbor_exception=_error(result),
        )
        output_id = str(
            summary.get("output_id") or f"{trial.record_id}/trial_{trial.trial}"
        )
        entry = output_dirs(eva / "run" / "records").get(output_id)
        if entry is not None:
            _attempts(trial, entry)
        else:  # EVA's run directory never came back; the driver's own record
            attempts = summary.get("attempts") or []
            trial.end_reasons = [str(a.get("end_reason")) for a in attempts]
            calls = [(a.get("timings") or {}).get("call_s") for a in attempts]
            trial.calls_s = sum(float(c or 0.0) for c in calls)
            if attempts:
                trial.end_reason = attempts[-1].get("end_reason")
                trial.call_s = _float(calls[-1])
        trial.calls = outcome["calls"]
        trial.attempt_errors = outcome["attempt_errors"]
        if isinstance(metrics, Mapping):
            trial.composites, trial.components, trial.response_speeds = metric_values(
                metrics
            )
        trials.append(trial)
    return trials


def harbor_model(jobs: Iterable[Path]) -> str | None:
    """The agent under test, as the harbor jobs recorded it (``config.json``)."""
    models = {
        str(agent["model_name"])
        for job in jobs
        for agent in (_json(job / "config.json") or {}).get("agents") or []
        if agent.get("model_name")
    }
    if len(models) > 1:
        raise SystemExit(
            f"the harbor jobs ran different agents: {', '.join(sorted(models))}"
        )
    return next(iter(models), None)


def eva_models(runs: Iterable[Path]) -> set[str]:
    """The agent each EVA run recorded: its framework and ``model.s2s``."""
    found = set()
    for run_dir in runs:
        config = _json(run_dir / "config.json") or {}
        s2s = (config.get("model") or {}).get("s2s")
        if s2s:
            found.add(f"{config.get('framework')}/{s2s}")
    return found


# ------------------------------------------------------------------ statistics


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sem(values: Sequence[float]) -> float | None:
    return (
        statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    )


Value = Callable[[Trial], float | None]


def _composite(name: str) -> Value:
    return lambda trial: trial.composites.get(name)


def _component(name: str) -> Value:
    return lambda trial: trial.components.get(name)


def run_means(trials: Iterable[Trial], value: Value) -> dict[int, float]:
    """Each run's mean of ``value`` over its graded trials that have one."""
    by_run: dict[int, list[float]] = defaultdict(list)
    for trial in trials:
        v = value(trial) if trial.status in SCORED else None
        if v is not None:
            by_run[trial.trial].append(v)
    return {k: statistics.fmean(vs) for k, vs in sorted(by_run.items())}


def describe(
    trials: Sequence[Trial], value: Value, *, scale: float = 1.0
) -> dict[str, Any]:
    """Mean ± sample SEM over runs, with the per-run values (harbor's format)."""
    means = run_means(trials, value)
    runs = [means[k] * scale for k in sorted(means)]
    return {
        "runs": [round(r, 4) for r in runs],
        "run_index": sorted(means),
        "mean": _mean(runs),
        "sem": _sem(runs),
        "values": sum(1 for t in trials if t.status in SCORED and value(t) is not None),
    }


def pass_k(trials: Iterable[Trial], name: str, k_max: int) -> dict[str, Any]:
    """pass@k and pass^k, EVA's estimator: per record C(c,k)/C(n,k) (pass^k)
    and 1 − C(n−c,k)/C(n,k) (pass@k), over records whose ``k_max`` trials
    were all scored (``aggregation._compute_aggregate_pass_k``)."""
    by_record: dict[tuple[str, str], list[float]] = defaultdict(list)
    for trial in trials:
        v = trial.composites.get(name) if trial.status in SCORED else None
        if v is not None:
            by_record[(trial.domain, trial.record_id)].append(v)
    rows = [
        (len(vs), sum(1 for v in vs if math.isclose(v, 1.0, abs_tol=1e-9)))
        for vs in by_record.values()
        if len(vs) >= k_max
    ]
    ks = range(1, k_max + 1)
    return {
        "records_all_scored": len(rows),
        "pass_hat_k": {
            k: _mean([math.comb(c, k) / math.comb(n, k) for n, c in rows]) for k in ks
        },
        "pass_at_k": {
            k: _mean([1 - math.comb(n - c, k) / math.comb(n, k) for n, c in rows])
            for k in ks
        },
    }


def overlap(a: Sequence[float], b: Sequence[float]) -> bool | None:
    """harbor's matching criterion: the two sides' run ranges overlap."""
    if not a or not b:
        return None
    return max(a) >= min(b) and max(b) >= min(a)


def side_summary(trials: Sequence[Trial], k_max: int) -> dict[str, Any]:
    """Everything one side shows, for one scope."""
    statuses = Counter(t.status for t in trials)
    validated = statuses[GRADED] + statuses[JUDGE_ERRORED] + statuses[UNGRADED]
    finished = [
        t
        for t in trials
        if t.status in (GRADED, JUDGE_ERRORED, UNGRADED, NOT_VALIDATED)
    ]
    attempts = [t.attempts for t in finished if t.attempts]
    graded = [t for t in trials if t.status in SCORED]
    calls = [t.call_s for t in graded if t.call_s is not None]
    turns = [t for t in graded if t.caller_turns is not None]
    spoken = sum(t.caller_turns or 0 for t in turns)
    words = sum(t.caller_words or 0 for t in turns)
    speeds = [s for t in graded for s in t.response_speeds]
    return {
        "trials": len(trials),
        "status": {s: statuses[s] for s in STATUSES},
        "calls": {"planned": len(trials), "made": sum(t.calls for t in trials)},
        "validation_rate": validated / len(finished) if finished else None,
        "attempts": dict(sorted(Counter(attempts).items())),
        "mean_attempts": _mean(attempts),
        "rerun_rate": sum(1 for n in attempts if n > 1) / len(attempts)
        if attempts
        else None,
        "time_limit_accepted": sum(1 for t in graded if t.time_limit_accepted),
        "end_reasons": dict(Counter(str(t.end_reason) for t in finished).most_common()),
        "end_reasons_all_attempts": dict(
            Counter(r for t in finished for r in t.end_reasons).most_common()
        ),
        "call_s": {
            "median": statistics.median(calls) if calls else None,
            "mean": _mean(calls),
            "all_attempts_total": round(sum(t.calls_s for t in trials), 1),
        },
        "caller": {
            "turns_per_call": _mean([float(t.caller_turns or 0) for t in turns]),
            "words_per_turn": words / spoken if spoken else None,
        },
        "response_speed_s": {
            "mean": _mean(speeds),
            "median": statistics.median(speeds) if speeds else None,
            "turns": len(speeds),
        },
        "composites": {
            name: {
                **describe(trials, _composite(name), scale=100),
                **pass_k(trials, name, k_max),
            }
            for name in COMPOSITES
        },
        "components": {name: describe(trials, _component(name)) for name in COMPONENTS},
    }


# ------------------------------------------------------------------ the report


def add_missing(
    trials: list[Trial], expected: Mapping[str, set[tuple[str, str, int]]] | None = None
) -> list[Trial]:
    """Keep one trial per side and key (the best outcome, should a pooled
    input hold two) and add a ``missing`` one for every expected key absent.

    Without ``expected``, every side is expected to have what any side has.
    """
    best: dict[tuple[str, tuple[str, str, int]], Trial] = {}
    for trial in trials:
        slot = (trial.side, trial.key)
        kept = best.get(slot)
        if kept is None:
            best[slot] = trial
            continue
        # Calls made by either count, whichever outcome is kept.
        if _PREFERENCE[trial.status] < _PREFERENCE[kept.status]:
            trial.calls += kept.calls
            best[slot] = trial
        else:
            kept.calls += trial.calls
    sides = sorted({t.side for t in trials} | set(expected or {}))
    if expected is None:
        union = {t.key for t in trials}
        expected = {side: union for side in sides}
    out = list(best.values())
    for side in sides:
        for key in sorted(expected.get(side, set())):
            if (side, key) not in best:
                out.append(
                    Trial(side, key[0], key[1], key[2], MISSING, error="no trial found")
                )
    return sorted(
        out, key=lambda t: (SIDES.index(t.side) if t.side in SIDES else 9, t.key)
    )


def tokens(
    trials: Iterable[Trial], runs: Mapping[str, Iterable[Path]]
) -> dict[str, dict[str, float]]:
    """Per side: every attempt's agent and caller tokens, and the judges'."""
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts = [(t.side, t.tokens) for t in trials]
    counts += [(side, judge_tokens(d)) for side, dirs in runs.items() for d in dirs]
    for side, found in counts:
        for name, n in found.items():
            out[side][name] += n
    return {
        side: {k: round(v, 1) for k, v in sorted(c.items())} for side, c in out.items()
    }


def build_report(
    trials: Sequence[Trial],
    *,
    k_max: int,
    judge_runs: Mapping[str, Iterable[Path]] | None = None,
    eva_summaries: Mapping[str, Any] | None = None,
    perturbation: PerturbationConfig | None = None,
    sides: Sequence[str] = SIDES,
) -> dict[str, Any]:
    """Per scope (each domain, then ``overall``) and side; harbor against EVA.

    ``notes`` starts empty: an ``extend`` hook (:func:`compare_folders`) may
    add to it, and it leads ``parity_experiment.json``'s notes.
    """
    present = [s for s in sides if any(t.side == s for t in trials)]
    domains = sorted({t.domain for t in trials})
    scopes: dict[str, dict[str, Any]] = {}
    pairs: dict[str, dict[str, Any]] = {}
    for scope in [*domains, "overall"]:
        picked = [t for t in trials if scope == "overall" or t.domain == scope]
        scopes[scope] = {
            s: side_summary([t for t in picked if t.side == s], k_max) for s in present
        }
        for a, b in PAIRS:
            if a not in present or b not in present:
                continue
            entry = pairs.setdefault(f"{a}-vs-{b}", {}).setdefault(scope, {})
            for name in COMPOSITES:
                runs_a = scopes[scope][a]["composites"][name]["runs"]
                runs_b = scopes[scope][b]["composites"][name]["runs"]
                mean_a = scopes[scope][a]["composites"][name]["mean"]
                mean_b = scopes[scope][b]["composites"][name]["mean"]
                entry[name] = {
                    "gap_pts": None
                    if mean_a is None or mean_b is None
                    else abs(mean_a - mean_b),
                    "ranges_overlap": overlap(runs_a, runs_b),
                }
    return {
        "condition": label_of(perturbation),
        "perturbation": as_record(perturbation),
        "sides": present,
        "trials_per_record": k_max,
        "notes": [],
        "scopes": scopes,
        "pairs": pairs,
        "tokens": tokens(trials, judge_runs or {}),
        "eva_summaries": dict(eva_summaries or {}),
        "trials": [asdict(t) for t in trials],
    }


def _pct(entry: Mapping[str, Any]) -> str:
    if entry["mean"] is None:
        return "n/a"
    sem = entry["sem"]
    return (
        f"{entry['mean']:.2f}% +/- {sem:.2f}%"
        if sem is not None
        else f"{entry['mean']:.2f}%"
    )


def parity_experiment(
    report: Mapping[str, Any],
    *,
    model: str,
    agent: str,
    records: int,
    notes: str = "",
) -> list[dict[str, Any]]:
    """harbor's ``parity_experiment.json`` entry: the three composites per scope."""
    metrics = []
    for scope, sides in report["scopes"].items():
        if "harbor" not in sides or "original" not in sides:
            continue
        for name in COMPOSITES:
            original = sides["original"]["composites"][name]
            harbor = sides["harbor"]["composites"][name]
            metrics.append(
                {
                    "benchmark_name": f"eva/{scope}",
                    "metric": name,
                    "original": _pct(original),
                    "harbor": _pct(harbor),
                    "original_runs": original["runs"],
                    "harbor_runs": harbor["runs"],
                }
            )
    return [
        {
            "adapter_name": "eva",
            "agent": agent,
            "model": model,
            "date": date.today().isoformat(),
            "adapted_benchmark_size": ADAPTED_SIZE,
            "parity_benchmark_size": records,
            "number_of_runs": report["trials_per_record"],
            "notes": notes,
            "original_parity_repo": ORIGINAL_REPO,
            "metrics": metrics,
        }
    ]


# ------------------------------------------------------------------ tables


def _cell(value: float | None, fmt: str = ".3f") -> str:
    return "n/a" if value is None else format(value, fmt)


def tables(report: Mapping[str, Any]) -> str:
    k_max = report["trials_per_record"]
    out = [
        f"Condition: {report.get('condition', CLEAN)}",
        *(
            [f"Audio judges: {report['judges']['audio']}"]
            if report.get("judges")
            else []
        ),
        "",
        "| Scope | Side | EVA-A pass | EVA-X pass | EVA-overall pass | A runs | A pass^"
        f"{k_max} | Graded | Judge errored | Not validated | Errored | Missing"
        " | Rerun rate | Median call (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for scope, sides in report["scopes"].items():
        for side, s in sides.items():
            c = s["composites"]
            out.append(
                f"| {scope} | {side} | {_pct(c['EVA-A_pass'])} | {_pct(c['EVA-X_pass'])} | "
                f"{_pct(c['EVA-overall_pass'])} | {c['EVA-A_pass']['runs']} | "
                f"{_cell(c['EVA-A_pass']['pass_hat_k'].get(k_max))} | "
                f"{s['status'][GRADED]} | {s['status'][JUDGE_ERRORED]} | "
                f"{s['status'][NOT_VALIDATED]} | "
                f"{s['status'][ERRORED]} | {s['status'][MISSING]} | "
                f"{_cell(s['rerun_rate'], '.0%')} | {_cell(s['call_s']['median'], '.0f')} |"
            )
    out += ["", "| Scope | Side | " + " | ".join(COMPONENTS) + " |"]
    out.append("|---|---|" + "---|" * len(COMPONENTS))
    for scope, sides in report["scopes"].items():
        for side, s in sides.items():
            cells = [
                "n/a"
                if s["components"][n]["mean"] is None
                else f"{s['components'][n]['mean']:.3f}"
                + (
                    f" +/- {s['components'][n]['sem']:.3f}"
                    if s["components"][n]["sem"] is not None
                    else ""
                )
                for n in COMPONENTS
            ]
            out.append(f"| {scope} | {side} | " + " | ".join(cells) + " |")
    if report["pairs"]:
        out += [
            "",
            "| Pair | Scope | Metric | Gap (pts) | Ranges overlap |",
            "|---|---|---|---|---|",
        ]
        for pair, scopes in report["pairs"].items():
            for scope, metrics in scopes.items():
                for name in COMPOSITES:
                    m = metrics[name]
                    out.append(
                        f"| {pair} | {scope} | {name} | {_cell(m['gap_pts'], '.2f')} | "
                        f"{m['ranges_overlap']} |"
                    )
    out += [
        "",
        "| Scope | Side | Calls made / planned | Validation rate | Mean attempts"
        " | End reasons (counted) | Caller turns/call | Words/turn | Response speed (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for scope, sides in report["scopes"].items():
        for side, s in sides.items():
            reasons = ", ".join(
                f"{k} {v}" for k, v in list(s["end_reasons"].items())[:4]
            )
            out.append(
                f"| {scope} | {side} | {s['calls']['made']} / {s['calls']['planned']} | "
                f"{_cell(s['validation_rate'], '.0%')} | "
                f"{_cell(s['mean_attempts'], '.2f')} | {reasons or 'n/a'} | "
                f"{_cell(s['caller']['turns_per_call'], '.1f')} | "
                f"{_cell(s['caller']['words_per_turn'], '.1f')} | "
                f"{_cell(s['response_speed_s']['mean'], '.2f')} |"
            )
    names = sorted({k for side in report["tokens"].values() for k in side})
    if names:
        out += [
            "",
            "| Side | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for side, counts in report["tokens"].items():
            out.append(
                f"| {side} | "
                + " | ".join(f"{counts.get(n, 0):,.0f}" for n in names)
                + " |"
            )
    out.extend(f"\nNote: {note}" for note in report.get("notes") or [])
    return "\n".join(out)


# ------------------------------------------------------------------ entry


def compare_folders(
    runs: Mapping[str, Sequence[Path]],
    *,
    output: Path,
    expected: Mapping[str, set[tuple[str, str, int]]] | None = None,
    trials: int | None = None,
    perturbation: PerturbationConfig | None = None,
    judge_label: str | None = None,
    judges: Mapping[str, Any] | None = None,
    eva_summaries: Mapping[str, Any] | None = None,
    model: str | None = None,
    agent: str = "harbor-voice@0.1.0",
    notes: str = "",
    run_notes: Sequence[str] = (),
    harbor_regrade: TrialRegrade = no_regrade,
    eva_regrades: RunRegrades = no_regrades,
    sides: Sequence[str] = SIDES,
    extend: Extension | None = None,
) -> dict[str, Any]:
    """Compare the sides from their folders; writes ``output``
    (``parity_experiment.json``) and its ``.report.json``.

    ``runs`` holds each side's folders: harbor job folders under ``harbor``,
    EVA run directories under ``original``, each pooled; ``sides`` names the
    sides allowed, so a caller can add more EVA runs under names of its own.
    ``expected`` is what each side should have, as (domain, record id, trial
    index); without it, every side is expected to have what any side has.
    ``trials`` is EVA's k (default: as many as the trial indices found).
    ``extend`` gets the report and the trials before anything is printed or
    written: it may add to the report (its ``notes`` included) and returns
    tables to print after these.

    Every trial must have run ``perturbation`` (clean when ``None``) and been
    judged by one set of audio judges: ``judge_label``
    (:func:`~eva_adapter.results.audio_judges`) or, without it, whichever the
    trials share; otherwise nothing is written. ``judges`` is kept in the
    report as given, and ``run_notes`` lead ``parity_experiment.json``'s
    notes after the condition's. ``harbor_regrade`` and ``eva_regrades``
    read a re-grade's records from the folders (none by default).
    """
    strange = sorted(set(runs) - set(sides))
    if strange:
        raise SystemExit(f"sides must be among {tuple(sides)}; got {strange}")
    jobs = list(runs.get("harbor") or [])
    eva_runs = {side: list(dirs) for side, dirs in runs.items() if side != "harbor"}
    condition = label_of(perturbation)
    index = task_index(perturbation)
    found: list[Trial] = []
    for job in jobs:
        found.extend(harbor_trials(job, index, regrade=harbor_regrade))
    for side, run_dirs in eva_runs.items():
        for run_dir in run_dirs:
            found.extend(eva_trials(run_dir, side, regrades=eva_regrades))
    strays = [t for t in found if t.condition not in (None, condition)]
    if strays:
        raise SystemExit(
            f"{len(strays)} trial(s) ran another condition than the run's {condition}: "
            + ", ".join(
                f"{t.side} {t.domain}/{t.record_id}/trial_{t.trial} ({t.condition})"
                for t in strays[:8]
            )
        )
    if judge_label is None:
        labels = sorted({t.judges for t in found if t.judges is not None})
        if len(labels) > 1:
            raise SystemExit(
                f"the trials were judged by different audio judges ({', '.join(labels)})"
                "; compare trials of one judge at a time"
            )
        judge_label = labels[0] if labels else audio_judges(None)
    misjudged = [t for t in found if t.judges not in (None, judge_label)]
    if misjudged:
        raise SystemExit(
            f"{len(misjudged)} trial(s) were judged by other audio judges than the "
            f"run's {judge_label}: "
            + ", ".join(
                f"{t.side} {t.domain}/{t.record_id}/trial_{t.trial} ({t.judges})"
                for t in misjudged[:8]
            )
        )
    k_max = trials or max((t.trial + 1 for t in found), default=1)
    found = add_missing(found, expected)
    judge_runs: dict[str, list[Path]] = {
        side: list(dirs) for side, dirs in eva_runs.items()
    }
    judge_runs["harbor"] = sorted(
        p.parent for job in jobs for p in job.glob("*/agent/call/eva/run/config.json")
    )
    report = build_report(
        found,
        k_max=k_max,
        judge_runs=judge_runs,
        eva_summaries=eva_summaries,
        perturbation=perturbation,
        sides=sides,
    )
    report["judges"] = {"audio": judge_label, "run": judges}
    extra = extend(report, found) if extend is not None else ""
    print(tables(report))
    if extra:
        print(extra)

    ran = harbor_model(jobs)
    model = model or ran
    if model is None:
        raise SystemExit("no harbor job records its agent; name it with --model")
    eva = eva_models(run_dir for dirs in eva_runs.values() for run_dir in dirs)
    if any(m.rpartition("/")[2] != model.rpartition("/")[2] for m in eva):
        raise SystemExit(
            f"the sides ran different agents: harbor {model}, EVA {sorted(eva)}"
        )
    keys = (
        {key for keys in expected.values() for key in keys}
        if expected is not None
        else {t.key for t in found}
    )
    records = len({(d, r) for d, r, _ in keys})
    if report["notes"]:
        notes = " ".join([*report["notes"], *([notes] if notes else [])])
    lead = " ".join(run_notes)
    if lead:
        notes = lead + (f" {notes}" if notes else "")
    if perturbation is not None:
        notes = (
            f"EVA perturbation condition {condition} on both sides "
            f"({perturbation.describe()}, EVA's own AudioPerturbator). EVA's noise "
            "is unseeded, so the sides compare statistically, not trial by trial."
            + (f" {notes}" if notes else "")
        )
    entry = parity_experiment(
        report, model=model, agent=agent, records=records, notes=notes
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote {output} and {output.with_suffix('.report.json').name}")
    return report


def main_compare(argv: Sequence[str]) -> int:
    from eva_adapter.sample import read_records

    parser = argparse.ArgumentParser(
        prog="eva-adapter compare",
        description="harbor's side against EVA's own run -> parity_experiment.json",
    )
    parser.add_argument(
        "--harbor", type=Path, nargs="+", default=[], help="harbor job folders"
    )
    parser.add_argument(
        "--original",
        type=Path,
        nargs="+",
        required=True,
        help="EVA run directories (EVA_OUTPUT_DIR/EVA_RUN_ID), one per domain",
    )
    parser.add_argument(
        "--records",
        type=Path,
        help="the sample both sides ran (eva-adapter sample); a trial of it "
        "absent on a side is missing (default: whatever any side has)",
    )
    parser.add_argument(
        "--trials", type=int, default=NUM_TRIALS, help="EVA's k (default %(default)s)"
    )
    parser.add_argument(
        "--model", help="the agent under test (default: what harbor ran)"
    )
    parser.add_argument("--agent", default="harbor-voice@0.1.0")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("parity_experiment.json")
    )
    add_condition_args(parser)
    args = parser.parse_args(list(argv))
    perturbation = condition_from_args(parser, args)
    runs = {"harbor": args.harbor, "original": args.original}
    expected = None
    if args.records:
        keys = {
            (domain, record_id, k)
            for domain, ids in read_records(args.records).items()
            for record_id in ids
            for k in range(args.trials)
        }
        expected = {side: set(keys) for side in runs}
    compare_folders(
        runs,
        output=args.output,
        expected=expected,
        trials=args.trials,
        perturbation=perturbation,
        model=args.model,
        agent=args.agent,
        notes=args.notes,
    )
    return 0
