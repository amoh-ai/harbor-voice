"""A harbor job config that runs tau-voice the way tau2 runs its trials.

Harbor's ``n_attempts`` repeats a trial identically; tau2 gives trial k its
own seed. So the job lists the call agent once per tau2 seed (``kwargs.seed``)
with one attempt each — harbor pools them into one metric because metrics key
on agent name and model, not kwargs. Infrastructure failures (``RETRY_ON``:
the call's, and the trial environment's) retry three times (tau2's
``max_retries``) and are left out of ``tau2_mean``, as tau2 leaves them out of
its average::

    python -m tau_voice.main job-config --dataset datasets/tau-voice \\
        --model openai.realtime/gpt-realtime-1.5 --trials 3 -n 4 -o job.yaml
    harbor-voice jobs start -c job.yaml

``--hub-dataset tau-voice-control@0.1.1`` runs a published dataset instead of
a local folder: harbor fetches its tasks at the commit ``registry.json``
pins (:mod:`harbor_voice.datasets`).

``--cascade STT LLM TTS --allow-cascade`` runs a cascade agent instead of a
speech-to-speech model; without ``--allow-cascade`` it is refused
(:mod:`tau_voice.cascade`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harbor_voice import datasets
from harbor_voice.capabilities import Pipeline
from harbor_voice.harbor_agent import harbor_parses_model_names
from harbor_voice.trial import AgentSpec

from tau_voice.cascade import ALLOW_CASCADE
from tau_voice.main import TAU2_RUN_SEED
from tau_voice.main import trial_seeds

AGENT = "harbor_voice.harbor_agent:VoiceCallAgent"
RETRY_ON = (
    # The call: provider, sidecar, transport (VoiceCallAgent's own).
    "VoiceInfrastructureError",
    # The trial's environment, before any call (harbor's). A plain
    # RuntimeError is how harbor reports "docker compose build failed", e.g.
    # a network drop while the tau2 image clones tau2-bench.
    "RuntimeError",
    "EnvironmentStartTimeoutError",
    "HealthcheckError",
    "SandboxBuildFailedError",
)
METRIC = Path(__file__).resolve().parent / "metrics" / "tau2_mean.py"
REGISTRY = datasets.checkout_registry()
"""The checkout's ``registry.json``; ``None`` when installed from an index."""


def hub_dataset(spec: str, registry: Path | None = None) -> dict[str, Any]:
    """Harbor's dataset entry for ``name@version`` (or ``name``, the latest):
    :func:`harbor_voice.datasets.hub_dataset`, against ``registry`` or this
    checkout's."""
    return datasets.hub_dataset(spec, REGISTRY if registry is None else registry)


def job_config(
    *,
    dataset: Path | None = None,
    hub: str | None = None,
    model: str | None = None,
    cascade: Sequence[str] | None = None,
    allow_cascade: bool = False,
    trials: int = 1,
    concurrent: int = 1,
    run_seed: int = TAU2_RUN_SEED,
    job_name: str | None = None,
    task_names: list[str] | None = None,
    environment: str | None = None,
    agent_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The job's own parts. The tasks are a local ``dataset`` folder or a
    published ``hub`` dataset (:func:`hub_dataset`). ``environment`` names
    harbor's backend (``docker``, ``daytona``, ``ec2``, ...); account settings
    (an EC2 AMI, key pair, ...) are the user's and go in
    ``environment.kwargs`` of their copy.

    The agent is ``model``, a speech-to-speech provider, or ``cascade``, its
    three slots (STT, LLM, TTS); a cascade needs ``allow_cascade``, which the
    driver reads from the agent's ``call_config``. ``agent_config`` is the
    agent's settings, sent as its ``agent_json``: s2s settings for ``model``
    (OpenAI Live's ``backend_model``), per-slot settings for a cascade
    (``{"llm": {...}}``). ``model`` (or ``stt+llm+tts``) still names it."""
    if (model is None) == (cascade is None):
        raise ValueError("name one agent: a model, or a cascade's three slots")
    spec: AgentSpec | None = None
    if cascade is not None:
        if not allow_cascade:
            raise ValueError(
                "tau-voice runs speech-to-speech agents; a cascade is off by "
                "default. Add --allow-cascade to run one"
            )
        stt, llm, tts = cascade
        model = f"{stt}+{llm}+{tts}"
        spec = AgentSpec(
            pipeline=Pipeline.CASCADE,
            stt=stt,
            llm=llm,
            tts=tts,
            config=agent_config or {},
        )
    elif agent_config:
        spec = AgentSpec(pipeline=Pipeline.S2S, s2s=model, config={"s2s": agent_config})
    dataset_entry: dict[str, Any]
    if hub is not None and dataset is None:
        dataset_entry = hub_dataset(hub)
    elif dataset is not None and hub is None:
        dataset_entry = {"path": str(dataset.resolve())}
    else:
        raise ValueError("name one of: a dataset folder, a hub dataset")
    extra: dict[str, Any] = {}
    if spec is not None:
        extra["agent_json"] = spec.model_dump_json()
    if allow_cascade:
        extra["call_config"] = {ALLOW_CASCADE: True}
    if task_names:
        dataset_entry["task_names"] = task_names
    config: dict[str, Any] = {
        "n_attempts": 1,
        "n_concurrent_trials": concurrent,
        "retry": {
            "max_retries": 3,
            "include_exceptions": list(RETRY_ON),
        },
        "metrics": [
            {"type": "uv-script", "kwargs": {"script_path": str(METRIC.resolve())}}
        ],
        "agents": [
            {
                "import_path": AGENT,
                "model_name": model,
                "kwargs": {"seed": seed, **extra},
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
