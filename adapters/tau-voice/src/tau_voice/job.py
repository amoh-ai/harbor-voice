"""A harbor job config that runs tau-voice the way tau2 runs its trials.

Harbor's ``n_attempts`` repeats a trial identically; tau2 gives trial k its
own seed. So the job lists the call agent once per tau2 seed (``kwargs.seed``)
with one attempt each — harbor pools them into one metric because metrics key
on agent name and model, not kwargs. Infrastructure failures retry three
times (tau2's ``max_retries``) and are left out of ``tau2_mean``, as tau2
leaves them out of its average::

    python -m tau_voice.main job-config --dataset datasets/tau-voice \\
        --model openai.realtime/gpt-realtime-1.5 --trials 3 -n 4 -o job.yaml
    harbor-voice jobs start -c job.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harbor_voice.harbor_agent import harbor_parses_model_names

from tau_voice.main import TAU2_RUN_SEED
from tau_voice.main import TEMPLATE
from tau_voice.main import trial_seeds

AGENT = "harbor_voice.harbor_agent:VoiceCallAgent"
METRIC = TEMPLATE / "metrics" / "tau2_mean.py"


def job_config(
    *,
    dataset: Path,
    model: str,
    trials: int = 1,
    concurrent: int = 1,
    run_seed: int = TAU2_RUN_SEED,
    job_name: str | None = None,
    task_names: list[str] | None = None,
    environment: str | None = None,
) -> dict[str, Any]:
    """The job's own parts. ``environment`` names harbor's backend (``docker``,
    ``daytona``, ``ec2``, ...); account settings (an EC2 AMI, key pair, ...)
    are the user's and go in ``environment.kwargs`` of their copy."""
    dataset_entry: dict[str, Any] = {"path": str(dataset.resolve())}
    if task_names:
        dataset_entry["task_names"] = task_names
    config: dict[str, Any] = {
        "n_attempts": 1,
        "n_concurrent_trials": concurrent,
        "retry": {
            "max_retries": 3,
            "include_exceptions": ["VoiceInfrastructureError"],
        },
        "metrics": [
            {"type": "uv-script", "kwargs": {"script_path": str(METRIC.resolve())}}
        ],
        "agents": [
            {
                "import_path": AGENT,
                "model_name": model,
                "kwargs": {"seed": seed},
                **({"skip_model_parsing": True} if harbor_parses_model_names() else {}),
            }
            for seed in trial_seeds(run_seed, trials)
        ],
        "datasets": [dataset_entry],
    }
    if job_name:
        config["job_name"] = job_name
    if environment:
        config["environment"] = {"type": environment}
    return config
