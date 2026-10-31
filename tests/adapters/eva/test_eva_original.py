"""The original side: EVA's own command, with the agent's key by name only."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys

import pytest

from eva_adapter.conditions import PerturbationConfig
from eva_adapter.original import eva_command
from eva_adapter.original import eva_settings
from eva_adapter.original import perturbation_check
from eva_adapter.original import s2s_model
from eva_adapter.original import s2s_params

pytestmark = pytest.mark.unit

SECRET = "sk-test-NEVER-WRITTEN"
COMMON = {
    "model": "openai.realtime/gpt-realtime-1.5",
    "trials": 3,
    "concurrency": 2,
    "run_id": "p-original-airline",
    "output_dir": "/work/output",
}


def test_the_command_is_evas_cli_with_evas_own_variable_names() -> None:
    command = eva_command(
        "airline",
        ["1.2.2", "2.2.5"],
        model="openai.realtime/gpt-realtime-1.5",
        trials=3,
        concurrency=4,
        run_id="p-original-airline",
        output_dir="/work/output",
    )
    assert command.startswith("cd /app && ")
    assert command.endswith(" eva")  # EVA's container wrapper, unchanged
    for setting in (
        "EVA_FRAMEWORK=openai_realtime",
        "EVA_MODEL__S2S=gpt-realtime-1.5",
        "EVA_DOMAIN=airline",
        "EVA_RECORD_IDS=1.2.2,2.2.5",
        "EVA_NUM_TRIALS=3",
        "EVA_MAX_CONCURRENT_CONVERSATIONS=4",
        "EVA_MAX_RERUN_ATTEMPTS=3",
        "EVA_USER_SIMULATOR__PROVIDER=elevenlabs",
        "EVA_LANGUAGE=en",
        "EVA_RUN_ID=p-original-airline",
        "EVA_OUTPUT_DIR=/work/output",
    ):
        assert setting in command
    # The caller agents and judges come from the environment it runs in.
    assert "EVA_EN_USER" not in command and "EVA_MODEL_LIST" not in command
    assert 'os.environ["OPENAI_API_KEY"]' in command


def test_the_agent_key_is_assembled_at_run_time_never_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    snippet = s2s_params("openai.realtime/gpt-realtime-1.5")
    assert SECRET not in snippet
    # EVA_MODEL__S2S_PARAMS="$(python3 -c '<code>')": the shell parses the
    # command substitution on its own; run its code with a key in the env.
    assert snippet.startswith('EVA_MODEL__S2S_PARAMS="$(') and snippet.endswith(')"')
    argv = shlex.split(snippet[snippet.index("$(") + 2 : -2])
    assert argv[:2] == ["python3", "-c"]
    done = subprocess.run(
        [sys.executable, "-c", argv[2]], capture_output=True, text=True, check=True
    )
    assert json.loads(done.stdout) == {"model": "gpt-realtime-1.5", "api_key": SECRET}


def test_only_evas_realtime_framework_has_a_counterpart() -> None:
    assert s2s_model("openai.realtime/gpt-realtime-1.5") == "gpt-realtime-1.5"
    for bad in ("openai.live/gpt-live-1", "gpt-realtime-1.5", "openai.realtime/x;rm"):
        with pytest.raises(SystemExit, match="openai_realtime"):
            s2s_model(bad)
    settings = eva_settings(
        "itsm", ["1"], model="openai.realtime/m", trials=1, concurrency=1,
        run_id="r", output_dir="/o",
    )  # fmt: skip
    assert settings["EVA_MODEL__S2S"] == "m"


def test_the_judges_config_is_evas_own_setting() -> None:
    configs = {
        "agent_speech_fidelity": {
            "audio_judge_model": "gemini-3.8-flash",
            "judge_params": {"reasoning_effort": "low"},
        }
    }
    clean = eva_settings("airline", ["1.2.2"], **COMMON)
    judged = eva_settings("airline", ["1.2.2"], metric_configs=configs, **COMMON)
    compact = json.dumps(configs, separators=(",", ":"), sort_keys=True)
    assert {k: v for k, v in judged.items() if k not in clean} == {
        "EVA_METRIC_CONFIGS": compact
    }
    command = eva_command("airline", ["1.2.2"], metric_configs=configs, **COMMON)
    assert f"EVA_METRIC_CONFIGS={shlex.quote(compact)}" in command
    assert "EVA_METRIC_CONFIGS" not in eva_command("airline", ["1.2.2"], **COMMON)


def test_a_condition_is_evas_own_settings_and_nothing_else() -> None:
    condition = PerturbationConfig("airport_gate", 20.0, True)
    clean = eva_settings("airline", ["1.2.2"], **COMMON)
    noisy = eva_settings("airline", ["1.2.2"], perturbation=condition, **COMMON)
    assert {k: v for k, v in noisy.items() if k not in clean} == {
        "EVA_PERTURBATION__BACKGROUND_NOISE": "airport_gate",
        "EVA_PERTURBATION__SNR_DB": "20.0",
        "EVA_PERTURBATION__CONNECTION_DEGRADATION": "true",
    }
    assert {k: noisy[k] for k in clean} == clean
    command = eva_command("airline", ["1.2.2"], perturbation=condition, **COMMON)
    assert command.startswith("cd /app && ") and command.endswith(" eva")
    check = perturbation_check(condition)
    assert check.startswith(
        "cd /app && EVA_PERTURBATION__BACKGROUND_NOISE=airport_gate "
    )
    assert "EVA_AGGREGATE_ONLY=true /opt/venv/bin/python -c " in check
    assert "OPENAI_API_KEY" not in check
