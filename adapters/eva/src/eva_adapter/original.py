"""The original side of a parity comparison: EVA's own CLI, unchanged.

EVA runs one domain per invocation (``RunConfig.domain``), as one long
process that runs, validates, reruns and grades every record itself
(``BenchmarkRunner.run``). That is not harbor-shaped, so the original side
is EVA's CLI on the records a harbor job runs, in EVA's image (the one the
tasks' ``eva`` sidecar is built on: EVA's Dockerfile at the pinned commit),
through the wrapper EVA's own container runs (``/usr/local/bin/eva``)::

    cd /app && EVA_FRAMEWORK=openai_realtime EVA_MODEL__S2S=gpt-realtime-1.5 \\
        EVA_DOMAIN=airline EVA_RECORD_IDS=1.1.2,2.1.1 EVA_NUM_TRIALS=3 \\
        EVA_MAX_CONCURRENT_CONVERSATIONS=3 EVA_MAX_RERUN_ATTEMPTS=3 \\
        EVA_USER_SIMULATOR__PROVIDER=elevenlabs EVA_LANGUAGE=en \\
        EVA_RUN_ID=<label>-original-airline EVA_OUTPUT_DIR=/work/output \\
        EVA_MODEL__S2S_PARAMS="$(python3 -c '<model + $OPENAI_API_KEY as JSON>')" eva

Every setting is in EVA's own variable names (``.env.example``), so the
command is the one EVA documents; everything it leaves out is EVA's default.
The caller is EVA's ElevenLabs simulator on the agents named by
``EVA_EN_USER_F`` / ``EVA_EN_USER_M``, and the judges are ``EVA_MODEL_LIST``
(``eva_adapter/judges.json``, the same file the harbor side's sidecars
get), both read from the environment the command runs in, as are the
judges' keys.

Keys. None is in the command. EVA wants the agent's key inside
``EVA_MODEL__S2S_PARAMS``, so that one value is assembled at run time, by
the shell that runs the command, from ``$OPENAI_API_KEY``; the command holds
only the name. EVA's ``config.json`` redacts it (``***``).

Re-entry. The run id is fixed (``EVA_RUN_ID``), so the same command run
again on the same output directory is EVA's own re-entry on an existing run
(``run_benchmark.py``): it validates what is there and reruns what failed,
with a fresh attempt budget. That changes the attempt count for those
records, so it is a deliberate step, never a retry.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from eva_adapter.conditions import PerturbationConfig
from eva_adapter.main import MAX_ATTEMPTS
from eva_adapter.main import REFERENCE_AGENT

EVA_APP = "/app"
"""EVA's repo root in its image; EVA reads ``data/`` and ``configs/`` from it."""
EVA_PYTHON = "/opt/venv/bin/python"
"""EVA's venv, in its image."""
EVA_CLI = "eva"
"""EVA's container wrapper (``scripts/docker_eva_wrapper.sh``), first on PATH."""
FRAMEWORK = "openai_realtime"
"""EVA's framework for the reference system (``gpt-realtime-1.5``)."""
SIMULATOR = "elevenlabs"
"""EVA's caller provider: its ElevenLabs Agents simulator."""
REALTIME = "openai.realtime"
"""The harbor provider whose EVA counterpart is :data:`FRAMEWORK`."""

_MODEL_NAME = re.compile(r"[A-Za-z0-9._-]+")


def s2s_model(model: str = REFERENCE_AGENT) -> str:
    """EVA's ``model.s2s`` for a harbor agent spec (``openai.realtime/<model>``).

    EVA's ``openai_realtime`` framework is the only one both sides share, so
    any other provider is refused here rather than run on mismatched agents.
    """
    provider, _, name = model.partition("/")
    if provider != REALTIME or not _MODEL_NAME.fullmatch(name):
        raise SystemExit(
            f"the original side runs EVA's {FRAMEWORK} framework, so the agent "
            f"must be {REALTIME}/<model>; got {model!r}"
        )
    return name


def eva_settings(
    domain: str,
    record_ids: Sequence[str],
    *,
    model: str,
    trials: int,
    concurrency: int,
    run_id: str,
    output_dir: str,
    perturbation: PerturbationConfig | None = None,
    metric_configs: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """EVA's settings for one run, in EVA's own variable names (no key).

    A perturbation is EVA's own ``EVA_PERTURBATION__*``; none is EVA's clean
    run, and adds nothing. ``metric_configs`` is EVA's ``EVA_METRIC_CONFIGS``
    (which judges grade; the harbor side's sidecars take the same JSON as
    ``EVA_SIDECAR_METRIC_CONFIGS``); none keeps EVA's judges.
    """
    judges = (
        {
            "EVA_METRIC_CONFIGS": json.dumps(
                metric_configs, separators=(",", ":"), sort_keys=True
            )
        }
        if metric_configs
        else {}
    )
    return {
        "EVA_FRAMEWORK": FRAMEWORK,
        "EVA_MODEL__S2S": s2s_model(model),
        "EVA_DOMAIN": domain,
        # Explicit ids, never EVA_DEBUG or a count: both sides run this list.
        "EVA_RECORD_IDS": ",".join(record_ids),
        "EVA_NUM_TRIALS": str(trials),
        "EVA_MAX_CONCURRENT_CONVERSATIONS": str(concurrency),
        # EVA's default, as the harbor side's attempt loop (main.MAX_ATTEMPTS).
        "EVA_MAX_RERUN_ATTEMPTS": str(MAX_ATTEMPTS),
        "EVA_USER_SIMULATOR__PROVIDER": SIMULATOR,
        "EVA_LANGUAGE": "en",
        "EVA_RUN_ID": run_id,
        "EVA_OUTPUT_DIR": output_dir,
        **(perturbation.eva_env() if perturbation is not None else {}),
        **judges,
    }


def s2s_params(model: str) -> str:
    """``EVA_MODEL__S2S_PARAMS``, assembled at run time from the key's name.

    EVA's Realtime server reads its key from ``s2s_params.api_key`` only
    (``openai_realtime_server.py:234``). The JSON is built when the command
    runs, so the command holds ``OPENAI_API_KEY``, never its value.
    """
    code = (
        "import json, os; print(json.dumps({"
        f'"model": {json.dumps(s2s_model(model))}, '
        '"api_key": os.environ["OPENAI_API_KEY"]}))'
    )
    return f'EVA_MODEL__S2S_PARAMS="$(python3 -c {shlex.quote(code)})"'


def eva_command(
    domain: str,
    record_ids: Sequence[str],
    *,
    model: str,
    trials: int,
    concurrency: int,
    run_id: str,
    output_dir: str,
    perturbation: PerturbationConfig | None = None,
    metric_configs: Mapping[str, Any] | None = None,
) -> str:
    """EVA's own run of ``record_ids`` in ``domain``, as one shell command."""
    settings = eva_settings(
        domain,
        record_ids,
        model=model,
        trials=trials,
        concurrency=concurrency,
        run_id=run_id,
        output_dir=output_dir,
        perturbation=perturbation,
        metric_configs=metric_configs,
    )
    assignments = " ".join(f"{k}={shlex.quote(v)}" for k, v in settings.items())
    return f"cd {EVA_APP} && {assignments} {s2s_params(model)} {EVA_CLI}"


PERTURBATION_CHECK = """\
import json, sys
from eva.models.config import RunConfig
from eva.user_simulator.perturbation import _ASSETS_DIR, AudioPerturbator
found = RunConfig(_env_file=None).perturbation
want = json.loads(sys.argv[1])
got = None if found is None else found.model_dump(mode="json")
if got != want:
    sys.exit(f"EVA reads the perturbation as {got}, not {want}")
AudioPerturbator(found)
print(f"EVA's perturbation: {got}; noise files in {_ASSETS_DIR}")
"""
"""Run in EVA's image before a perturbed run starts: EVA's own settings
parse the condition from its ``EVA_PERTURBATION__*`` (``EVA_AGGREGATE_ONLY``
skips the checks a run needs, nothing else), and EVA's own perturbator reads
its noise file from where EVA looks (``Path(perturbation.py).parents[3] /
"assets"``: ``/app/assets/noise``, while the image's ``PYTHONPATH`` is
``/app/src``)."""


def perturbation_check(perturbation: PerturbationConfig) -> str:
    """The command for :data:`PERTURBATION_CHECK`, with the run's own settings."""
    settings = {**perturbation.eva_env(), "EVA_AGGREGATE_ONLY": "true"}
    assignments = " ".join(f"{k}={shlex.quote(v)}" for k, v in settings.items())
    want = {**perturbation.as_dict(), "accent": None, "behavior": None}
    return (
        f"cd {EVA_APP} && {assignments} {EVA_PYTHON} -c "
        f"{shlex.quote(PERTURBATION_CHECK)} {shlex.quote(json.dumps(want))}"
    )
