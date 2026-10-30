"""tau2's interaction metrics for both sides of a parity run, by tau2's code.

The panel of the tau-voice paper — L_R and L_Y (latency), R_R and R_Y
(responsiveness), I_A (agent interruptions), S_BC, S_VT and S_ND
(selectivity) — per domain and averaged over domains, computed with
``tau2.metrics.voice_interaction_metrics`` on tau2's own simulations and on
the harbor side's (each trial's final attempt, as the sidecar's ``/close``
recorded it). It needs tau2, so it runs in the sidecar image::

    python interaction_metrics.py --complexity regular \\
        --harbor <harbor job dir>... --original <tau2 save dir>... -o metrics.json

Both sides go through the same function on the same kind of record; only the
loading differs (tau2's ``Results.load``, or the harbor trial folders).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tau2.data_model.simulation import Results
from tau2.data_model.simulation import SimulationRun
from tau2.metrics.voice_interaction_metrics import INTERACTION_METRICS_VERSION
from tau2.metrics.voice_interaction_metrics import InteractionMetricsConfig
from tau2.metrics.voice_interaction_metrics import NoVoiceTicksError
from tau2.metrics.voice_interaction_metrics import aggregate_domain_metrics
from tau2.metrics.voice_interaction_metrics import (
    compute_interaction_metrics_for_experiment,
)

Simulations = dict[str, list[SimulationRun]]


def _complexity(sim: SimulationRun) -> str | None:
    env = sim.speech_environment
    return env.complexity if env is not None else None


def harbor_simulations(job_dirs: list[Path], complexity: str | None) -> Simulations:
    """Each harbor trial's final attempt, by domain."""
    by_domain: Simulations = defaultdict(list)
    for job in job_dirs:
        for trial in sorted(job.glob("*/agent/call/tau_voice/trial.json")):
            record = json.loads(trial.read_text(encoding="utf-8"))
            attempt = trial.parent / f"attempt-{record['final_attempt']}"
            state = attempt / "tau2_state.json"
            if not state.exists():
                continue
            data = json.loads(state.read_text(encoding="utf-8")).get("simulation")
            if not data:
                continue
            sim = SimulationRun.model_validate(data)
            if complexity is None or _complexity(sim) == complexity:
                by_domain[str(record["domain"])].append(sim)
    return by_domain


def original_simulations(save_dirs: list[Path], complexity: str | None) -> Simulations:
    """tau2's saved runs (``results.json`` folders), by domain."""
    by_domain: Simulations = defaultdict(list)
    found = {
        p.parent
        for path in save_dirs
        for p in (*path.glob("results.json"), *path.glob("*/results.json"))
    }
    for folder in sorted(found):
        results = Results.load(folder)
        domain = results.info.environment_info.domain_name
        by_domain[domain].extend(
            s
            for s in results.simulations
            if complexity is None or _complexity(s) == complexity
        )
    return by_domain


def panel(by_domain: Simulations, config: InteractionMetricsConfig) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    for domain, sims in sorted(by_domain.items()):
        if not sims:
            continue
        try:
            domains[domain] = compute_interaction_metrics_for_experiment(
                SimpleNamespace(simulations=sims), config
            )
        except NoVoiceTicksError:
            continue
    return {
        "domains": domains,
        "overall": aggregate_domain_metrics(domains) if domains else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--harbor", type=Path, nargs="*", default=[])
    parser.add_argument("--original", type=Path, nargs="*", default=[])
    parser.add_argument("--complexity", help="only simulations of this preset")
    parser.add_argument("--tick-s", type=float, default=0.2)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    config = InteractionMetricsConfig(tick_duration_sec=args.tick_s)
    report = {
        "version": INTERACTION_METRICS_VERSION,
        "config": config.as_dict(),
        "complexity": args.complexity,
        "harbor": panel(harbor_simulations(args.harbor, args.complexity), config),
        "original": panel(original_simulations(args.original, args.complexity), config),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
