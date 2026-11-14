"""The ``eva-adapter`` command: EVA records as harbor-voice tasks, and their images.

::

    eva-adapter convert --out datasets/eva [--domains ...] [--records ...]
                        [--language en] [--reward EVA-A_pass]
                        [--prebuilt | --registry <prefix>]
                        [--background-noise <noise> [--snr-db 15]]
                        [--connection-degradation] [--sidecar-env NAME ...]
    eva-adapter build-image [--registry <prefix> --push] [--from-source] [--force]
    eva-adapter job-config (--dataset datasets/eva | --hub-dataset eva@0.1.1) \\
                           --model openai.realtime/gpt-realtime-1.5 \\
                           [--trials 3] [-n 4] [--env daytona]
                           [--override-storage-mb <MB>] -o job.yaml
    eva-adapter judges [--gemini api-key|vertex|vertex-json]
                                  EVA's judge deployments, for EVA_MODEL_LIST
    eva-adapter agents ...        EVA's ElevenLabs caller agents (agents.py)
    eva-adapter sample ...        the records a parity comparison runs (sample.py)
    eva-adapter compare --harbor <job> ... --original <EVA run> ...
                                  harbor's side against EVA's own (compare.py)

One task per EVA record (airline 50, itsm 80, medical_hr 83; the ids are
EVA's, ``template/records.json``)::

    <domain>-<slug>-<hash>/
    ├── task.toml                   harbor's task config, for harbor's runner
    ├── instruction.md              carries voice.toml for the call runner
    ├── voice.toml                  [call] driver, [environment], [verifier]
    ├── environment/
    │   ├── Dockerfile              main: where harbor-voice runs the call
    │   ├── docker-compose.yaml     adds the eva service next to main
    │   └── sidecar/                EVA's image + our server (unless prebuilt)
    └── tests/                      the verifier: test.sh → grade.py → /grade

``convert`` also writes ``job.yaml``: the reference system
(``openai.realtime/gpt-realtime-1.5``) over EVA's k = 3 trials, locally;
``job-config`` writes any other.

With EVA's perturbation flags, every task is that one condition
(:mod:`eva_adapter.conditions`): ``<domain>-<slug>-<hash>-<condition>/``,
``[call.config.perturbation]``, and ``main`` with EVA's numpy and noise WAVs
(``images.MAIN_PERTURBED_DOCKERFILE``). Convert each condition into its own
folder.

The agent under test is not in the task: the run picks it
(``-m openai.realtime/gpt-realtime-1.5``). No secret is ever written: the
environment names variables as ``${VAR}`` templates, resolved from the
orchestrator's environment at trial start. ``--sidecar-env`` names more of
them for the ``eva`` service (optional, empty when unset): what a judge
deployment of your own ``EVA_MODEL_LIST`` reads, say.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harbor_voice import datasets

from eva_adapter import EVA_COMMIT
from eva_adapter import images
from eva_adapter.agents import agent_env_var
from eva_adapter.call import GUARD_TIMEOUT_S
from eva_adapter.client import JUDGE_TIMEOUT_S
from eva_adapter.conditions import ACCENTS
from eva_adapter.conditions import BACKGROUND_NOISES
from eva_adapter.conditions import BEHAVIORS
from eva_adapter.conditions import DEFAULT_SNR_DB
from eva_adapter.conditions import ConditionError
from eva_adapter.conditions import PerturbationConfig
from eva_adapter.conditions import label_of
from eva_adapter.trial import DEFAULT_REWARD
from eva_adapter.trial import NUM_TRIALS
from eva_adapter.trial import REWARDS

DOMAINS = ("airline", "itsm", "medical_hr")
TEMPLATE = Path(__file__).resolve().parents[2] / "template"
RECORDS = TEMPLATE / "records.json"
JUDGES = Path(__file__).resolve().parent / "judges.json"
"""EVA's judge deployments, the value of ``EVA_MODEL_LIST`` (keys by
``os.environ/`` reference only)."""
JUDGE_MODELS = (
    "gpt-5.2",
    "gpt-5.2-medium",
    "gemini-3-flash-preview",
    "us.anthropic.claude-opus-4-6-v1",
)
"""Every ``default_model`` of EVA's judge metrics @039efa0a."""
REFERENCE_AGENT = "openai.realtime/gpt-realtime-1.5"
"""EVA's reference S2S system (published EVA-A 0.467)."""

MAX_ATTEMPTS = 3
"""EVA's ``max_rerun_attempts`` default: attempts per trial."""
ATTEMPT_BUDGET_S = GUARD_TIMEOUT_S + 60 + JUDGE_TIMEOUT_S
"""One attempt at worst: EVA's worker guard (the call, the ``end_call``
check included), the agent's shutdown and ``/record``, then ``/validate``."""
TRIAL_TIMEOUT_S = int(MAX_ATTEMPTS * ATTEMPT_BUDGET_S + JUDGE_TIMEOUT_S + 300)
"""Every attempt used, each to its limit, then ``/grade`` and ``/files``."""
VERIFIER_TIMEOUT_S = int(JUDGE_TIMEOUT_S + 60)
"""``/grade`` again: a re-read, unless the driver never got to grade."""

# Names only, as ${VAR} templates resolved from the host at run time; values
# are never written into a task directory.
SIDECAR_ENV = ("OPENAI_API_KEY", "EVA_MODEL_LIST")
"""The eva service's, required: the text judges (gpt-5.2) and EVA's LiteLLM
deployments (``EVA_MODEL_LIST``), which validation needs on every attempt.
Without them no attempt can pass, so a trial refuses to start."""
SIDECAR_OPTIONAL_ENV = (
    # Gemini, for the audio judges: an API key, or Vertex AI with
    # VERTEXAI_PROJECT and either ADC (GOOGLE_APPLICATION_CREDENTIALS naming
    # a file inside the container) or the credentials JSON itself in
    # VERTEXAI_CREDENTIALS. See GEMINI_ROUTES. VERTEXAI_LOCATION is only for
    # a model list of your own: the generated one pins "global".
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "VERTEXAI_PROJECT",
    "VERTEXAI_LOCATION",
    "VERTEXAI_CREDENTIALS",
    # Bedrock, for the faithfulness judge (Claude Opus).
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    # EVA's judge override (an empty value is EVA's default).
    "JUDGE_MODEL",
    # EVA's metric_configs for the sidecar to hand to EVA's RunConfig (JSON,
    # EVA_METRIC_CONFIGS' format; empty is EVA's defaults): which judges
    # grade, the audio judges' model, say.
    "EVA_SIDECAR_METRIC_CONFIGS",
)
"""The eva service's, optional (empty when unset): which are needed depends
on the deployments in ``EVA_MODEL_LIST``. EVA's own ``EVA_*`` settings are
not passed: an empty one would fail EVA's config, and a set one changes EVA's
protocol. ``EVA_SIDECAR_METRIC_CONFIGS`` is not EVA's (EVA ignores it); the
sidecar reads it itself. ``convert --sidecar-env`` adds names to these."""

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def sidecar_env_names(extra: Sequence[str] = ()) -> tuple[str, ...]:
    """Every name the eva service gets: :data:`SIDECAR_ENV`,
    :data:`SIDECAR_OPTIONAL_ENV`, then ``extra`` (``--sidecar-env``), each once."""
    bad = [name for name in extra if not _ENV_NAME.fullmatch(name)]
    if bad:
        raise ValueError(f"not environment variable names: {', '.join(bad)}")
    return tuple(dict.fromkeys((*SIDECAR_ENV, *SIDECAR_OPTIONAL_ENV, *extra)))


def main_env(language: str = "en") -> tuple[str, ...]:
    """main's: the agent under test (the reference is OpenAI Realtime), the
    caller's ElevenLabs key, and EVA's caller agent IDs for the language."""
    return (
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        agent_env_var(language, "F"),
        agent_env_var(language, "M"),
    )


def task_env(language: str = "en", sidecar_env: Sequence[str] = ()) -> str:
    """``[environment].env``: every variable main or the compose file uses.

    On a remote backend harbor runs ``docker compose`` with the task's env
    and its own infra vars only, so a variable the compose file passes to the
    sidecar must be declared here to reach it. ``sidecar_env``'s names are
    optional, like :data:`SIDECAR_OPTIONAL_ENV`'s.
    """
    required = dict.fromkeys((*main_env(language), *SIDECAR_ENV))
    optional = [n for n in sidecar_env_names(sidecar_env) if n not in required]
    pairs = [f'{name} = "${{{name}}}"' for name in required]
    pairs += [f'{name} = "${{{name}:-}}"' for name in optional]
    return "{ " + ", ".join(pairs) + " }"


COMPOSE = """\
# Generated by eva_adapter.main. Harbor supplies main; this adds the eva
# service: EVA's own code for prompts, tools, validation and grading.
services:
  main:
    depends_on:
      eva:
        condition: service_healthy
    # Not every backend passes the task's env to main (harbor's EC2 does
    # not), so main names what it needs, as the eva service does.
    environment:
{main_env}
    volumes:
      # Wheels' dependencies survive across trials.
      - ${{HARBOR_VOICE_CACHE_DIR:-${{HOME}}/.cache/harbor-voice}}/uv:/root/.cache/uv
  eva:
{source}
    environment:
{env}
      - AWS_REGION_NAME=${{AWS_REGION:-}}
"""

RESOURCES = """\
cpus = 2
memory_mb = 4096"""
"""One trial: the call runs in real time in main while the sidecar idles.
CPU and memory are the reservations in EVA's ``compose.yaml``. EVA sets no
disk, and neither does the task: the backend's default. A trial holds the
sidecar image (~3.6 GB) and each attempt's record (three 24 kHz tracks,
~90 MB for a 10-minute call); a job can set the disk with
``job-config --override-storage-mb``."""


def load_records(path: Path = RECORDS) -> dict[str, list[str]]:
    """EVA's record ids per domain, in EVA's order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {domain: list(ids) for domain, ids in data["records"].items()}


def task_dir_name(
    domain: str, record_id: str, perturbation: PerturbationConfig | None = None
) -> str:
    """Readable and collision-free; a perturbed task's ends with its condition."""
    slug = re.sub(r"[^a-z0-9]+", "-", record_id.lower()).strip("-")[:48].strip("-")
    digest = hashlib.sha1(f"{domain}/{record_id}".encode()).hexdigest()[:8]
    name = f"{domain}-{slug}-{digest}"
    return name if perturbation is None else f"{name}-{perturbation.label}"


def task_name(
    domain: str, record_id: str, perturbation: PerturbationConfig | None = None
) -> str:
    """``voice.toml``'s ``task``: ``eva/<domain>/<record>``, ``@<condition>`` if perturbed."""
    name = f"eva/{domain}/{record_id}"
    return name if perturbation is None else f"{name}@{perturbation.label}"


def _environment(
    prebuilt: bool,
    registry: str | None,
    language: str,
    perturbed: bool = False,
    sidecar_env: Sequence[str] = (),
) -> str:
    image = images.pinned_main(registry, perturbed=perturbed)
    main_image = f"docker_image = {json.dumps(image)}\n" if prebuilt else ""
    return f"""{main_image}build_timeout_sec = 3600
{RESOURCES}
env = {task_env(language, sidecar_env)}"""


def _perturbation_table(perturbation: PerturbationConfig | None) -> str:
    """``[call.config.perturbation]``: EVA's audio settings, by EVA's names."""
    if perturbation is None:
        return ""
    return f"""
[call.config.perturbation]
# EVA's PerturbationConfig (EVA_PERTURBATION__*): EVA's own AudioPerturbator on
# the caller's audio. Condition {perturbation.label}.
{perturbation.toml()}
"""


def voice_toml(
    domain: str,
    record_id: str,
    *,
    language: str = "en",
    reward: str = DEFAULT_REWARD,
    num_trials: int = NUM_TRIALS,
    prebuilt: bool = False,
    registry: str | None = None,
    perturbation: PerturbationConfig | None = None,
    sidecar_env: Sequence[str] = (),
) -> str:
    perturbed = perturbation is not None
    return f"""\
# Generated by eva_adapter.main from ServiceNow/eva@{EVA_COMMIT}.
# The seed is EVA's trial index k (records/<id>/trial_<k>); a job sets it.
task = {json.dumps(task_name(domain, record_id, perturbation))}
seed = 0

[call]
driver = "eva_adapter.trial:drive"
timeout_sec = {TRIAL_TIMEOUT_S}

[call.config]
domain = {json.dumps(domain)}
record_id = {json.dumps(record_id)}
language = {json.dumps(language)}
reward = {json.dumps(reward)}
num_trials = {num_trials}
{_perturbation_table(perturbation)}
[environment]
{_environment(prebuilt, registry, language, perturbed, sidecar_env)}

[verifier]
timeout_sec = {VERIFIER_TIMEOUT_S}
"""


def _metadata_condition(perturbation: PerturbationConfig | None) -> str:
    if perturbation is None:
        return ""
    return f"""condition = {json.dumps(perturbation.label)}

[metadata.perturbation]
{perturbation.toml()}
"""


def task_toml(
    name: str,
    domain: str,
    record_id: str,
    *,
    language: str = "en",
    prebuilt: bool = False,
    registry: str | None = None,
    perturbation: PerturbationConfig | None = None,
    sidecar_env: Sequence[str] = (),
) -> str:
    """Harbor's ``task.toml``: the same task, for harbor's job runner."""
    perturbed = perturbation is not None
    description = f"EVA {domain} record {record_id}: a phone call" + (
        f" ({perturbation.label})" if perturbation is not None else ""
    )
    return f"""\
# Generated by eva_adapter.main from ServiceNow/eva@{EVA_COMMIT}. Run with
# harbor's job runner: harbor-voice trials start -p <task> -m <s2s spec>
[task]
name = {json.dumps(f"eva/{name}")}
description = {json.dumps(description)}

[metadata]
benchmark = "eva"
domain = {json.dumps(domain)}
record_id = {json.dumps(record_id)}
language = {json.dumps(language)}
source = "ServiceNow/eva@{EVA_COMMIT}"
{_metadata_condition(perturbation)}
[agent]
timeout_sec = {TRIAL_TIMEOUT_S}

[verifier]
timeout_sec = {VERIFIER_TIMEOUT_S}

[environment]
{_environment(prebuilt, registry, language, perturbed, sidecar_env)}
"""


def instruction_md(
    domain: str,
    record_id: str,
    spec: str,
    perturbation: PerturbationConfig | None = None,
) -> str:
    """What harbor hands the agent: the call spec, for the call runner."""
    heard = (
        ""
        if perturbation is None
        else f"\nThe agent hears the caller through {perturbation.describe()}\n"
        f"(EVA's perturbation condition {perturbation.label}).\n"
    )
    return f"""\
# EVA {domain} record {record_id}

A customer-service phone call: EVA's simulated caller (an ElevenLabs agent)
calls the agent under test, which answers by voice and uses the domain's
tools. Validated and graded by EVA's own code.
{heard}
This task is run by harbor-voice's call runner
(`harbor_voice.harbor_agent:VoiceCallAgent`), which reads the spec below.

```toml
{spec.rstrip()}
```
"""


def write_task(
    out: Path,
    domain: str,
    record_id: str,
    *,
    language: str = "en",
    reward: str = DEFAULT_REWARD,
    num_trials: int = NUM_TRIALS,
    prebuilt: bool = False,
    registry: str | None = None,
    perturbation: PerturbationConfig | None = None,
    sidecar_env: Sequence[str] = (),
) -> Path:
    """One task folder. ``registry`` implies ``prebuilt``: images pulled from it.

    ``perturbation`` is the task's condition (:mod:`eva_adapter.conditions`):
    in its name, its metadata and ``[call.config.perturbation]``, and ``main``
    built with EVA's numpy and WAVs. ``None`` writes the clean task.
    ``sidecar_env`` names more variables for the eva service, each optional
    (:func:`sidecar_env_names`).
    """
    if reward not in REWARDS:
        raise ValueError(f"reward {reward!r} is not one of {REWARDS}")
    names = sidecar_env_names(sidecar_env)
    prebuilt = prebuilt or registry is not None
    perturbed = perturbation is not None
    task = out / task_dir_name(domain, record_id, perturbation)
    (task / "environment").mkdir(parents=True, exist_ok=True)
    (task / "tests").mkdir(exist_ok=True)

    def write(path: Path, text: str) -> None:
        # LF everywhere: these run under bash and Python in Linux containers,
        # and a CRLF checkout (Windows) must not leak into them.
        path.write_text(text, encoding="utf-8", newline="\n")

    spec = voice_toml(
        domain,
        record_id,
        language=language,
        reward=reward,
        num_trials=num_trials,
        prebuilt=prebuilt,
        registry=registry,
        perturbation=perturbation,
        sidecar_env=sidecar_env,
    )
    write(task / "voice.toml", spec)
    write(
        task / "task.toml",
        task_toml(
            task.name,
            domain,
            record_id,
            language=language,
            prebuilt=prebuilt,
            registry=registry,
            perturbation=perturbation,
            sidecar_env=sidecar_env,
        ),
    )
    write(
        task / "instruction.md", instruction_md(domain, record_id, spec, perturbation)
    )
    main_image = images.pinned_main(registry, perturbed=perturbed)
    write(
        task / "environment" / "Dockerfile",
        images.from_image(main_image)
        if prebuilt and "@" in main_image
        else images.main_dockerfile(perturbed),
    )
    if prebuilt:
        source = f"    image: {images.pinned_sidecar(registry)}"
    else:
        # Built where it runs, from EVA's public repo; every task's build
        # lands on one tag.
        source = (
            "    build:\n"
            "      context: ./sidecar\n"
            "      args:\n"
            f"        EVA_BASE: {images.FROM_SOURCE}\n"
            f"    image: {images.sidecar_image(from_source=True)}"
        )
        sidecar = task / "environment" / "sidecar"
        sidecar.mkdir(exist_ok=True)
        for name in images.SIDECAR_FILES:
            write(
                sidecar / name,
                (images.SIDECAR / name).read_text(encoding="utf-8"),
            )
    env = "\n".join(f"      - {name}" for name in names)
    main_names = "\n".join(f"      - {name}" for name in main_env(language))
    write(
        task / "environment" / "docker-compose.yaml",
        COMPOSE.format(env=env, main_env=main_names, source=source),
    )
    write(
        task / "tests" / "test.sh",
        (TEMPLATE / "tests" / "test.sh")
        .read_text(encoding="utf-8")
        .replace("__REWARD__", reward),
    )
    write(
        task / "tests" / "grade.py",
        (TEMPLATE / "tests" / "grade.py").read_text(encoding="utf-8"),
    )
    return task


# --------------------------------------------------------------------- job

AGENT = "harbor_voice.harbor_agent:VoiceCallAgent"
RETRY_ON = (
    # The call: provider, sidecar, transport (VoiceCallAgent's own).
    "VoiceInfrastructureError",
    # The trial's environment, before any call (harbor's): a failed compose
    # build is a RuntimeError, e.g. a network drop while EVA is fetched.
    "RuntimeError",
    "EnvironmentStartTimeoutError",
    "HealthcheckError",
    "SandboxBuildFailedError",
)
"""Infrastructure only. A record EVA never validated is a result (unscored),
not an error, and is never retried here: EVA's own reruns already ran."""
METRIC = Path(__file__).resolve().parent / "metrics" / "eva_mean.py"
REGISTRY = datasets.checkout_registry()
"""The checkout's ``registry.json``; ``None`` when installed from an index."""


def hub_dataset(spec: str, registry: Path | None = None) -> dict[str, Any]:
    """Harbor's dataset entry for ``name@version`` (or ``name``, the latest):
    :func:`harbor_voice.datasets.hub_dataset`, against ``registry`` or this
    checkout's. Harbor fetches the tasks at the commit ``registry.json``
    pins."""
    return datasets.hub_dataset(spec, REGISTRY if registry is None else registry)


def trial_seeds(trials: int = NUM_TRIALS) -> list[int]:
    """EVA's trial indices: trial k is ``records/<id>/trial_<k>``."""
    return list(range(trials))


def job_config(
    *,
    dataset: Path | None = None,
    hub: str | None = None,
    model: str = REFERENCE_AGENT,
    trials: int = NUM_TRIALS,
    concurrent: int = 1,
    job_name: str | None = None,
    task_names: list[str] | None = None,
    environment: str | None = None,
    agent_config: dict[str, Any] | None = None,
    storage_mb: int | None = None,
) -> dict[str, Any]:
    """A harbor job that runs EVA's k trials of every task.

    Harbor's ``n_attempts`` repeats a trial identically; EVA's trials are
    distinct (``trial_0`` … ``trial_{k-1}``). So the call agent is listed once
    per trial index (``kwargs.seed``) with one attempt each; harbor pools
    them into one metric, which keys on agent name and model, not kwargs.

    The tasks are a local ``dataset`` folder or a published ``hub`` dataset
    (:func:`hub_dataset`). ``storage_mb`` sets each trial's disk (harbor's
    ``override_storage_mb``; the tasks ask for none, so the backend's default
    applies without it).
    """
    from harbor_voice.capabilities import Pipeline
    from harbor_voice.harbor_agent import harbor_parses_model_names
    from harbor_voice.trial import AgentSpec

    dataset_entry: dict[str, Any]
    if hub is not None and dataset is None:
        dataset_entry = hub_dataset(hub)
    elif dataset is not None and hub is None:
        dataset_entry = {"path": str(dataset.resolve())}
    else:
        raise ValueError("name one of: a dataset folder, a hub dataset")
    extra: dict[str, Any] = {}
    if agent_config:
        extra["agent_json"] = AgentSpec(
            pipeline=Pipeline.S2S, s2s=model, config={"s2s": agent_config}
        ).model_dump_json()
    if task_names:
        dataset_entry["task_names"] = task_names
    config: dict[str, Any] = {
        "n_attempts": 1,
        "n_concurrent_trials": concurrent,
        "retry": {"max_retries": 3, "include_exceptions": list(RETRY_ON)},
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
            for seed in trial_seeds(trials)
        ],
        "datasets": [dataset_entry],
    }
    if job_name:
        config["job_name"] = job_name
    if environment:
        config["environment"] = {"type": environment}
    if storage_mb is not None:
        config.setdefault("environment", {})["override_storage_mb"] = storage_mb
    return config


def _dump_yaml(config: dict[str, Any], path: Path) -> None:
    import yaml

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


# --------------------------------------------------------------------- CLI


def _first_line(doc: str | None) -> str:
    return doc.strip().splitlines()[0] if doc else ""


def add_condition_args(parser: argparse.ArgumentParser) -> None:
    """EVA's perturbation knobs (``EVA_PERTURBATION__*``), under EVA's names.

    Accents and behaviours are EVA's too, and accepted only to be refused by
    name (:func:`condition_from_args`) rather than as unknown flags.
    """
    group = parser.add_argument_group(
        "perturbation (EVA_PERTURBATION__*; one condition per dataset or launch)"
    )
    group.add_argument(
        "--background-noise",
        choices=BACKGROUND_NOISES,
        help="EVA's background noise, mixed into the caller's speech and its "
        "silences (bad_connection_static is synthetic; the rest are EVA's WAVs)",
    )
    group.add_argument(
        "--snr-db",
        type=float,
        help=f"the noise's signal-to-noise ratio in dB (EVA's default "
        f"{DEFAULT_SNR_DB:g}; higher is cleaner); needs --background-noise",
    )
    group.add_argument(
        "--connection-degradation",
        action="store_true",
        help="EVA's VoIP degradation: mu-law codec, 3%% packet loss, random gain",
    )
    group.add_argument("--accent", choices=ACCENTS, help="not supported yet")
    group.add_argument("--behavior", choices=BEHAVIORS, help="not supported yet")


def condition_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> PerturbationConfig | None:
    """The condition the flags name, ``None`` for clean; bad ones are usage errors."""
    try:
        return PerturbationConfig.from_options(
            args.background_noise,
            args.snr_db,
            args.connection_degradation,
            accent=args.accent,
            behavior=args.behavior,
        )
    except ConditionError as exc:
        parser.error(str(exc))


def conditions_in(out: Path) -> set[str]:
    """The conditions of the EVA tasks already in ``out`` (harbor's job metric
    pools a dataset folder, so a folder holds one)."""
    import tomllib

    found = set()
    for spec in out.glob("*/voice.toml") if out.is_dir() else ():
        call = tomllib.loads(spec.read_text(encoding="utf-8")).get("call") or {}
        if call.get("driver") != "eva_adapter.trial:drive":
            continue
        table = (call.get("config") or {}).get("perturbation")
        try:
            found.add(
                label_of(
                    None if table is None else PerturbationConfig.from_mapping(table)
                )
            )
        except ConditionError:
            found.add(f"unreadable ({spec.parent.name})")
    return found


def convert(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="eva-adapter convert", description="EVA records → harbor-voice tasks"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--records", nargs="+", help="only these record ids (with one domain)"
    )
    which.add_argument(
        "--records-from",
        type=Path,
        metavar="SAMPLE",
        help="only a sample file's records (`eva-adapter sample -o`), in each "
        "of --domains it names",
    )
    parser.add_argument("--limit", type=int, help="first N records per domain")
    parser.add_argument(
        "--language", default="en", help="EVA's language (en; others need agents)"
    )
    parser.add_argument("--reward", default=DEFAULT_REWARD, choices=REWARDS)
    parser.add_argument(
        "--trials", type=int, default=NUM_TRIALS, help="EVA's num_trials (k)"
    )
    parser.add_argument(
        "--prebuilt",
        action="store_true",
        help="name the images from `eva-adapter build-image` instead of building them",
    )
    parser.add_argument(
        "--registry",
        help="prebuilt images pulled from this registry prefix (implies --prebuilt)",
    )
    parser.add_argument(
        "--sidecar-env",
        nargs="+",
        default=[],
        metavar="NAME",
        help="more variables for the eva service, by name (optional, empty "
        "when unset), e.g. what your EVA_MODEL_LIST's deployments read",
    )
    add_condition_args(parser)
    args = parser.parse_args(argv)
    perturbation = condition_from_args(parser, args)
    try:
        sidecar_env_names(args.sidecar_env)
    except ValueError as exc:
        parser.error(str(exc))
    try:  # before any task is written
        images.pinned_sidecar(args.registry)
        images.pinned_main(args.registry, perturbed=perturbation is not None)
    except images.UnpublishedImage as exc:
        parser.error(str(exc))
    others = conditions_in(args.out) - {label_of(perturbation)}
    if others:
        parser.error(
            f"{args.out} holds tasks of condition {', '.join(sorted(others))}, not "
            f"{label_of(perturbation)}: convert each condition into its own folder, "
            "so a job's metric never pools two"
        )
    every = load_records()
    if args.records and len(args.domains) != 1:
        parser.error("--records needs exactly one --domains")
    chosen = dict(every)
    if args.records_from:
        from eva_adapter.sample import read_records

        chosen = read_records(args.records_from)  # checked against EVA's ids
        if not set(chosen) & set(args.domains):
            parser.error(f"{args.records_from} names none of {' '.join(args.domains)}")
    written = 0
    for domain in (d for d in args.domains if d in chosen):
        ids = list(args.records or chosen[domain])
        unknown = sorted(set(ids) - set(every[domain]))
        if unknown:
            parser.error(f"not EVA {domain} records: {', '.join(unknown)}")
        if args.limit:
            ids = ids[: args.limit]
        for record_id in ids:
            write_task(
                args.out,
                domain,
                record_id,
                language=args.language,
                reward=args.reward,
                num_trials=args.trials,
                prebuilt=args.prebuilt,
                registry=args.registry,
                perturbation=perturbation,
                sidecar_env=args.sidecar_env,
            )
            written += 1
    job = args.out / "job.yaml"
    _dump_yaml(job_config(dataset=args.out, trials=args.trials), job)
    condition = "" if perturbation is None else f" ({perturbation.label})"
    print(f"wrote {written} task(s){condition} and {job.name} to {args.out}")
    return 0


def build_image(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="eva-adapter build-image", description=_first_line(images.__doc__)
    )
    parser.add_argument("--force", action="store_true", help="rebuild them all")
    parser.add_argument(
        "--registry", help="name them under this registry prefix, e.g. ghcr.io/<org>"
    )
    parser.add_argument("--push", action="store_true", help="push to --registry")
    parser.add_argument(
        "--from-source",
        action="store_true",
        help="build EVA in the sidecar's own stages, not on eva-base",
    )
    args = parser.parse_args(argv)
    for tag in images.build(
        force=args.force,
        registry=args.registry,
        push=args.push,
        from_source=args.from_source,
    ):
        print(tag)
    return 0


def write_job_config(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="eva-adapter job-config", description=_first_line(job_config.__doc__)
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="a folder of converted tasks")
    source.add_argument(
        "--hub-dataset",
        metavar="NAME[@VERSION]",
        help="a published dataset, as this checkout's registry.json pins it",
    )
    parser.add_argument("--model", default=REFERENCE_AGENT, help="the S2S agent")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS, help="EVA's k")
    parser.add_argument("-n", "--concurrent", type=int, default=1)
    parser.add_argument("--job-name")
    parser.add_argument("--env", help="harbor backend: docker, daytona, ec2, ...")
    parser.add_argument("--task-names", nargs="+", help="only these task folders")
    parser.add_argument(
        "--agent-config", type=json.loads, help="the agent's s2s settings as JSON"
    )
    parser.add_argument(
        "--override-storage-mb",
        type=int,
        help="each trial's disk in MB (the tasks set none: the backend's default)",
    )
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = job_config(
            dataset=args.dataset,
            hub=args.hub_dataset,
            model=args.model,
            trials=args.trials,
            concurrent=args.concurrent,
            job_name=args.job_name,
            environment=args.env,
            task_names=args.task_names,
            agent_config=args.agent_config,
            storage_mb=args.override_storage_mb,
        )
    except ValueError as exc:
        parser.error(str(exc))
    _dump_yaml(config, args.out)
    print(args.out)
    return 0


GEMINI_ROUTES = ("api-key", "vertex", "vertex-json")
"""How the Gemini judge is reached. ``api-key``: the Gemini API with
``GEMINI_API_KEY`` (EVA's default). ``vertex``: Vertex AI in
``VERTEXAI_PROJECT`` with Application Default Credentials (``gcloud auth
application-default login`` on this machine, ``GOOGLE_APPLICATION_CREDENTIALS``
naming a file, or a GCP workload identity). ``vertex-json``: Vertex AI with the
credentials JSON itself in ``VERTEXAI_CREDENTIALS`` (a service account key or
an ADC file's contents), for a container that has no ADC of its own."""

VERTEX_GEMINI_LOCATION = "global"
"""Gemini 3 previews are served from Vertex's global endpoint only."""


def judge_list(gemini: str = "api-key") -> list[dict[str, Any]]:
    """EVA's judge deployments, with the Gemini judge on ``gemini``'s route.

    The deployment keeps its name (``gemini-3-flash-preview``), so EVA's
    metrics find it whichever route serves it.
    """
    if gemini not in GEMINI_ROUTES:
        raise ValueError(f"gemini route must be one of {GEMINI_ROUTES}, not {gemini!r}")
    deployments = json.loads(JUDGES.read_text(encoding="utf-8"))
    if gemini == "api-key":
        return deployments
    for deployment in deployments:
        params = deployment["litellm_params"]
        if not params["model"].startswith("gemini/"):
            continue
        # Swap the route and its auth; keep any other settings.
        params.pop("api_key", None)
        params["model"] = "vertex_ai/" + params["model"].removeprefix("gemini/")
        params["vertex_project"] = "os.environ/VERTEXAI_PROJECT"
        params["vertex_location"] = VERTEX_GEMINI_LOCATION
        if gemini == "vertex-json":
            params["vertex_credentials"] = "os.environ/VERTEXAI_CREDENTIALS"
    return deployments


def print_judges(argv: Sequence[str]) -> int:
    """Print EVA's judge deployments, the value of ``EVA_MODEL_LIST``."""
    parser = argparse.ArgumentParser(
        prog="eva-adapter judges", description=_first_line(print_judges.__doc__)
    )
    parser.add_argument(
        "--gemini",
        choices=GEMINI_ROUTES,
        default="api-key",
        help="how the Gemini audio judge is reached: the Gemini API key "
        "(default), Vertex AI with Application Default Credentials, or Vertex "
        "AI with a credentials JSON in VERTEXAI_CREDENTIALS",
    )
    args = parser.parse_args(argv)
    if args.gemini == "api-key":
        sys.stdout.write(JUDGES.read_text(encoding="utf-8"))
    else:
        sys.stdout.write(json.dumps(judge_list(args.gemini), indent=2) + "\n")
    return 0


COMMANDS = {
    "convert": convert,
    "build-image": build_image,
    "job-config": write_job_config,
    "judges": print_judges,
}


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else None
    if command == "agents":
        from eva_adapter.agents import main_agents

        sys.exit(main_agents(argv[1:]))
    if command == "sample":
        from eva_adapter.sample import main_sample

        sys.exit(main_sample(argv[1:]))
    if command == "compare":
        from eva_adapter.compare import main_compare

        sys.exit(main_compare(argv[1:]))
    if command in COMMANDS:
        sys.exit(COMMANDS[command](argv[1:]))
    if command in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    print(__doc__, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
