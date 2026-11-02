"""The converter writes harbor tasks that harbor-voice and harbor accept as-is."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from harbor_voice.trial import VoiceTaskConfig

from eva_adapter import images
from eva_adapter.conditions import PerturbationConfig
from eva_adapter.main import JUDGE_MODELS
from eva_adapter.main import JUDGES
from eva_adapter.main import METRIC
from eva_adapter.main import RETRY_ON
from eva_adapter.main import SIDECAR_ENV
from eva_adapter.main import SIDECAR_OPTIONAL_ENV
from eva_adapter.main import TEMPLATE
from eva_adapter.main import TRIAL_TIMEOUT_S
from eva_adapter.main import job_config
from eva_adapter.main import load_records
from eva_adapter.main import main
from eva_adapter.main import write_task

pytestmark = pytest.mark.unit

EVA_CHECKOUT = Path(__file__).resolve().parents[4] / "eva"


def _compose(task: Path) -> dict[str, Any]:
    return yaml.safe_load(
        (task / "environment" / "docker-compose.yaml").read_text(encoding="utf-8")
    )


def test_a_task_names_the_driver_and_the_record(tmp_path: Path) -> None:
    task = write_task(tmp_path, "airline", "1.1.2")

    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    assert config.agent is None  # the run picks it
    assert config.task == "eva/airline/1.1.2"
    assert config.seed == 0  # EVA's trial index; a job sets 0, 1, 2
    assert config.call is not None
    assert config.call.driver == "eva_adapter.trial:drive"
    assert config.call.config == {
        "domain": "airline",
        "record_id": "1.1.2",
        "language": "en",
        "reward": "EVA-A_pass",
        "num_trials": 3,
    }
    # Every attempt to EVA's limits, then the grade.
    assert config.call.timeout_sec == TRIAL_TIMEOUT_S >= 3 * (620 + 1800)
    assert config.verifier is not None and config.verifier.timeout_sec >= 1800


def test_the_environment_holds_names_never_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-written")
    monkeypatch.setenv("EVA_MODEL_LIST", "[should-never-be-written]")
    task = write_task(tmp_path, "itsm", "12")

    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    assert config.environment is not None
    env = config.environment.env
    for name in (
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "EVA_EN_USER_F",
        "EVA_EN_USER_M",
    ):
        assert env[name] == f"${{{name}}}"
    for name in SIDECAR_ENV:
        assert env[name] == f"${{{name}}}"
    for name in SIDECAR_OPTIONAL_ENV:
        assert env[name] == f"${{{name}:-}}"
    for path in task.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "should-never-be-written" not in text, path


def test_main_and_the_eva_service(tmp_path: Path) -> None:
    task = write_task(tmp_path, "medical_hr", "T7.1")

    compose = _compose(task)
    main_service, eva = compose["services"]["main"], compose["services"]["eva"]
    assert main_service["depends_on"]["eva"]["condition"] == "service_healthy"
    # main names its own env: harbor's EC2 does not pass the task's to main.
    assert main_service["environment"] == [
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "EVA_EN_USER_F",
        "EVA_EN_USER_M",
    ]
    # Built where it runs, from EVA's public repo; one tag for every task.
    assert eva["build"] == {
        "context": "./sidecar",
        "args": {"EVA_BASE": images.FROM_SOURCE},
    }
    assert eva["image"] == images.sidecar_image(from_source=True)
    assert eva["image"].endswith("-src")
    assert eva["environment"][: len(SIDECAR_ENV)] == list(SIDECAR_ENV)
    assert "AWS_REGION_NAME=${AWS_REGION:-}" in eva["environment"]
    for name in images.SIDECAR_FILES:
        copied = (task / "environment" / "sidecar" / name).read_bytes()
        assert b"\r\n" not in copied
        assert copied == (images.SIDECAR / name).read_bytes().replace(b"\r\n", b"\n")
    assert (task / "environment" / "Dockerfile").read_text() == images.MAIN_DOCKERFILE


def test_the_verifier_regrades_through_the_sidecar(tmp_path: Path) -> None:
    task = write_task(tmp_path, "airline", "1.1.2", reward="EVA-overall_pass")

    test_sh = (task / "tests" / "test.sh").read_bytes()
    assert b"\r\n" not in test_sh
    assert b'--reward "EVA-overall_pass"' in test_sh
    assert b"--sidecar http://eva:8000" in test_sh
    assert (task / "tests" / "grade.py").read_text(encoding="utf-8") == (
        TEMPLATE / "tests" / "grade.py"
    ).read_text(encoding="utf-8")
    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    assert (
        config.call is not None and config.call.config["reward"] == "EVA-overall_pass"
    )
    with pytest.raises(ValueError, match="not one of"):
        write_task(tmp_path, "airline", "1.1.2", reward="EVA-A_mean")


def test_a_task_is_also_a_harbor_task(tmp_path: Path) -> None:
    from harbor.models.task.config import TaskConfig

    from harbor_voice.harbor_agent import call_spec

    task = write_task(tmp_path, "airline", "1.1.2")
    harbor = TaskConfig.model_validate_toml(
        (task / "task.toml").read_text(encoding="utf-8")
    )
    assert harbor.task is not None and harbor.task.name == f"eva/{task.name}"
    assert harbor.agent.timeout_sec == TRIAL_TIMEOUT_S
    assert harbor.environment.env["EVA_MODEL_LIST"] == "${EVA_MODEL_LIST}"
    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    assert (
        call_spec(instruction)
        == (task / "voice.toml").read_text(encoding="utf-8").rstrip()
    )


def test_prebuilt_and_registry_tasks_name_their_images(tmp_path: Path) -> None:
    task = write_task(tmp_path / "local", "airline", "1.1.2", prebuilt=True)
    eva = _compose(task)["services"]["eva"]
    assert "build" not in eva and eva["image"] == images.sidecar_image()
    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    assert config.environment is not None
    assert config.environment.docker_image == images.main_image()
    assert not (task / "environment" / "sidecar").exists()

    pushed = write_task(tmp_path / "remote", "airline", "1.1.2", registry="ghcr.io/o/")
    eva = _compose(pushed)["services"]["eva"]
    assert eva["image"] == f"ghcr.io/o/{images.sidecar_image()}"
    assert eva["image"].startswith("ghcr.io/o/eva-sidecar:039efa0a-")


def test_image_tags_name_the_commit_and_our_files() -> None:
    tag = images.sidecar_image().split(":")[1]
    assert tag.startswith("039efa0a-") and len(tag) == len("039efa0a-") + 10
    assert images.sidecar_image(from_source=True) == images.sidecar_image() + "-src"
    assert images.main_image().startswith("eva-main:")
    assert images.EVA_COMMIT_FULL.startswith("039efa0a")


def test_every_eva_record_is_a_task() -> None:
    records = load_records()
    assert {d: len(ids) for d, ids in records.items()} == {
        "airline": 50,
        "itsm": 80,
        "medical_hr": 83,
    }
    if not (EVA_CHECKOUT / "data" / "airline_dataset.json").exists():
        pytest.skip("no EVA checkout beside this repo to compare with")
    for domain, ids in records.items():
        data = json.loads(
            (EVA_CHECKOUT / "data" / f"{domain}_dataset.json").read_text(
                encoding="utf-8"
            )
        )
        assert ids == [record["id"] for record in data]


def test_convert_writes_tasks_and_a_k3_job(tmp_path: Path) -> None:
    from harbor.models.job.config import JobConfig

    with pytest.raises(SystemExit) as done:
        main(["convert", "--out", str(tmp_path), "--domains", "itsm", "--limit", "2"])
    assert done.value.code == 0
    tasks = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert len(tasks) == 2 and all(name.startswith("itsm-") for name in tasks)
    job = JobConfig.model_validate(
        yaml.safe_load((tmp_path / "job.yaml").read_text(encoding="utf-8"))
    )
    assert [agent.kwargs["seed"] for agent in job.agents] == [0, 1, 2]
    assert {agent.model_name for agent in job.agents} == {
        "openai.realtime/gpt-realtime-1.5"
    }

    with pytest.raises(SystemExit):
        main(
            [
                "convert",
                "--out",
                str(tmp_path),
                "--domains",
                "itsm",
                "--records",
                "nope",
            ]
        )


def test_convert_takes_a_sample_files_records(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps({"records": {"itsm": ["13", "16"], "medical_hr": ["A2"]}}),
        encoding="utf-8",
    )
    out = tmp_path / "tasks"

    with pytest.raises(SystemExit) as done:
        main(["convert", "--out", str(out), "--records-from", str(sample)])
    assert done.value.code == 0
    tasks = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert [name.split("-")[0] for name in tasks] == ["itsm", "itsm", "medical_hr"]

    # --domains narrows the file; a domain it lacks is skipped, none is an error.
    narrowed = tmp_path / "narrowed"
    with pytest.raises(SystemExit) as done:
        main(["convert", "--out", str(narrowed), "--records-from", str(sample),
              "--domains", "medical_hr", "airline"])  # fmt: skip
    assert done.value.code == 0
    assert len([p for p in narrowed.iterdir() if p.is_dir()]) == 1
    with pytest.raises(SystemExit) as done:
        main(["convert", "--out", str(tmp_path / "none"), "--records-from",
              str(sample), "--domains", "airline"])  # fmt: skip
    assert done.value.code == 2

    sample.write_text(json.dumps({"records": {"itsm": ["nope"]}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="nope"):
        main(["convert", "--out", str(tmp_path / "bad"), "--records-from", str(sample)])


def test_the_job_lists_one_agent_per_eva_trial(tmp_path: Path) -> None:
    from harbor.models.job.config import JobConfig

    raw = job_config(dataset=tmp_path, trials=3, environment="daytona", concurrent=4)
    config = JobConfig.model_validate(raw)
    assert config.n_attempts == 1 and config.n_concurrent_trials == 4
    assert [agent.kwargs["seed"] for agent in config.agents] == [0, 1, 2]
    assert config.retry.include_exceptions == set(RETRY_ON)
    assert raw["environment"] == {"type": "daytona"}
    assert raw["metrics"][0]["kwargs"]["script_path"] == str(METRIC.resolve())


def test_the_judges_list_serves_every_eva_judge_and_holds_no_key() -> None:
    deployments = json.loads(JUDGES.read_text(encoding="utf-8"))
    assert sorted(d["model_name"] for d in deployments) == sorted(JUDGE_MODELS)
    # Credentials only by reference: the file is committed and goes to containers.
    for d in deployments:
        for key, value in d["litellm_params"].items():
            if "key" in key or "secret" in key or "credentials" in key:
                assert str(value).startswith("os.environ/"), (d["model_name"], key)
    # Each referenced variable reaches the eva service.
    referenced = {
        str(value).removeprefix("os.environ/")
        for d in deployments
        for value in d["litellm_params"].values()
        if str(value).startswith("os.environ/")
    }
    assert referenced <= {*SIDECAR_ENV, *SIDECAR_OPTIONAL_ENV}


def test_more_variables_reach_the_sidecar_by_name(tmp_path: Path) -> None:
    extra = ("MY_JUDGE_URL", "MY_JUDGE_KEY", "OPENAI_API_KEY", "MY_JUDGE_URL")
    task = write_task(tmp_path, "airline", "1.1.2", sidecar_env=extra)
    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    assert config.environment is not None
    env = config.environment.env
    # Optional, like the rest of the sidecar's; a required name stays required.
    assert env["MY_JUDGE_URL"] == "${MY_JUDGE_URL:-}"
    assert env["MY_JUDGE_KEY"] == "${MY_JUDGE_KEY:-}"
    assert env["OPENAI_API_KEY"] == "${OPENAI_API_KEY}"
    eva = _compose(task)["services"]["eva"]["environment"]
    assert eva[: len(SIDECAR_ENV)] == list(SIDECAR_ENV)
    assert eva.count("MY_JUDGE_URL") == 1 and "MY_JUDGE_KEY" in eva
    assert eva.count("OPENAI_API_KEY") == 1
    # task.toml says the same, for harbor's own runner.
    from harbor.models.task.config import TaskConfig

    harbor = TaskConfig.model_validate_toml(
        (task / "task.toml").read_text(encoding="utf-8")
    )
    assert harbor.environment.env["MY_JUDGE_KEY"] == "${MY_JUDGE_KEY:-}"
    with pytest.raises(ValueError, match="not environment variable names: A-B"):
        write_task(tmp_path, "airline", "1.1.2", sidecar_env=("A-B",))


def test_the_cli_forwards_more_variables_and_refuses_bad_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["convert", "--out", str(tmp_path), "--domains", "itsm", "--limit", "1"]
    with pytest.raises(SystemExit) as done:
        main([*base, "--sidecar-env", "MY_JUDGE_URL", "MY_JUDGE_KEY"])
    assert done.value.code == 0
    (task,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    eva = _compose(task)["services"]["eva"]["environment"]
    assert {"MY_JUDGE_URL", "MY_JUDGE_KEY"} <= set(eva)
    with pytest.raises(SystemExit) as refused:
        main(["convert", "--out", str(tmp_path / "x"), "--sidecar-env", "1BAD"])
    assert refused.value.code == 2 and "1BAD" in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


def test_the_task_asks_for_no_disk_and_a_job_can_set_one(tmp_path: Path) -> None:
    from harbor.models.task.config import TaskConfig

    task = write_task(tmp_path, "airline", "1.1.2")
    harbor = TaskConfig.model_validate_toml(
        (task / "task.toml").read_text(encoding="utf-8")
    )
    assert harbor.environment.storage_mb is None  # the backend's default
    assert (harbor.environment.cpus, harbor.environment.memory_mb) == (2, 4096)
    config = job_config(dataset=tmp_path, environment="daytona", storage_mb=10240)
    assert config["environment"] == {"type": "daytona", "override_storage_mb": 10240}
    assert "environment" not in job_config(dataset=tmp_path)


def test_the_mean_leaves_unscored_trials_out(tmp_path: Path) -> None:
    rewards = tmp_path / "rewards.jsonl"
    rewards.write_text(
        '{"reward": 1.0, "EVA-A_pass": 1.0, "EVA-X_pass": 0.0}\n'
        "null\n"
        '{"reward": 0.0, "EVA-A_pass": 0.0, "EVA-X_pass": 1.0}\n'
        '{"reward": 1.0, "EVA-A_pass": 1.0}\n'
    )
    out = tmp_path / "metric.json"
    subprocess.run(
        [sys.executable, str(METRIC), "-i", str(rewards), "-o", str(out)],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    # Each composite's mean covers the scored trials that have it.
    assert json.loads(out.read_text()) == {
        "eva_mean": 2 / 3,
        "scored": 3,
        "unscored": 1,
        "EVA-A_pass": 2 / 3,
        "EVA-X_pass": 0.5,
    }


class _Grade(BaseHTTPRequestHandler):
    reply: tuple[int, dict[str, Any]] = (200, {})

    def do_POST(self) -> None:  # noqa: N802 - http.server's name
        status, body = self.reply
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        return None


@pytest.mark.parametrize(
    ("status", "body", "reward"),
    [
        (
            200,
            {
                "record_id": "1.1.2/trial_0",
                "metrics": {"task_completion": {"score": 1.0, "details": {}}},
                "aggregates": {
                    "EVA-A_pass": 1.0,
                    "EVA-X_pass": 0.0,
                    "EVA-overall_pass": None,
                },
            },
            # A composite EVA left unset is left out, not written as null.
            {"reward": 1.0, "EVA-A_pass": 1.0, "EVA-X_pass": 0.0},
        ),
        (409, {"detail": "nothing to grade: no attempt passed validation"}, None),
    ],
)
def test_grade_py_writes_harbors_files(
    tmp_path: Path,
    status: int,
    body: dict[str, Any],
    reward: dict[str, float] | None,
) -> None:
    handler = type("Handler", (_Grade,), {"reply": (status, body)})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        subprocess.run(
            [
                sys.executable,
                str(TEMPLATE / "tests" / "grade.py"),
                "--reward",
                "EVA-A_pass",
                "--sidecar",
                f"http://127.0.0.1:{server.server_port}",
                "--out",
                str(tmp_path),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
        )
    finally:
        server.shutdown()
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    if reward is None:
        assert result["graded"] is False and "no attempt passed" in result["detail"]
        assert not (tmp_path / "reward.json").exists()
    else:
        assert result["metrics"] == {
            "task_completion": {
                "score": 1.0,
                "normalized_score": None,
                "skipped": None,
                "error": None,
            }
        }
        assert json.loads((tmp_path / "reward.json").read_text()) == reward
        assert result["rewards"] == reward


# ------------------------------------------------------------- conditions

CLEAN_BYTES = {
    # sha256 of airline 1.1.2's files as convert wrote them before EVA's
    # perturbations existed: a clean task must not change by one byte. Since
    # then the eva service's optional env names grew by
    # EVA_SIDECAR_METRIC_CONFIGS, and [environment] lost its storage_mb (the
    # backend's default; a job sets one with --override-storage-mb). Each
    # time the earlier digests were checked to be what the old text gives.
    "": {
        "voice.toml": "7f5778d3e485597c11635a43f7c196d0123198d350fde3fe8252d2227c738ca4",
        "task.toml": "9bd31900e434ea5636739fdb1c021f79066db7d28292fc697ddaf3b3713ccf87",
        "instruction.md": "eb9ba595162b736ffe3f2d9a78153dbd2c622b23236eb5895ee01e7a5f01f66c",
        "environment/Dockerfile": "645b00ea31f2abf86fcb42f8a1aab1a9e07b2a3dbe2330359dd5baac4b8ff0ae",
    },
    "ghcr.io/x": {
        "voice.toml": "29f9a60d340c65fa8ce82173e5b4c417f4ee4af0d9c9069fd2974282a6447804",
        "task.toml": "a158caf3e52625dcd6bd6dd346927623cf68d3da6cccea458859a8f99a22590d",
        "instruction.md": "62c63d182e5cb7f23fe08ac55eeef193908d30af90470a346afc25d394696082",
        "environment/Dockerfile": "645b00ea31f2abf86fcb42f8a1aab1a9e07b2a3dbe2330359dd5baac4b8ff0ae",
    },
}


@pytest.mark.parametrize("registry", sorted(CLEAN_BYTES))
def test_without_a_condition_the_task_is_todays_byte_for_byte(
    tmp_path: Path, registry: str
) -> None:
    import hashlib

    task = write_task(tmp_path, "airline", "1.1.2", registry=registry or None)
    assert task.name == "airline-1-1-2-17dc1179"
    found = {
        name: hashlib.sha256((task / name).read_bytes()).hexdigest()
        for name in CLEAN_BYTES[registry]
    }
    assert found == CLEAN_BYTES[registry]


def test_a_condition_is_in_the_tasks_config_name_and_metadata(tmp_path: Path) -> None:
    from harbor.models.task.config import TaskConfig

    from harbor_voice.harbor_agent import call_spec

    condition = PerturbationConfig("coffee_shop", 10.0, True)
    task = write_task(tmp_path, "airline", "1.1.2", perturbation=condition)
    label = "coffee_shop-10db-connection_degradation"
    assert task.name == f"airline-1-1-2-17dc1179-{label}"

    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    assert config.task == f"eva/airline/1.1.2@{label}"
    assert config.call is not None
    assert config.call.config["perturbation"] == {
        "background_noise": "coffee_shop",
        "snr_db": 10.0,
        "connection_degradation": True,
    }
    # What the driver reads back is the same condition.
    assert (
        PerturbationConfig.from_mapping(config.call.config["perturbation"]) == condition
    )

    harbor = TaskConfig.model_validate_toml(
        (task / "task.toml").read_text(encoding="utf-8")
    )
    assert harbor.task is not None and harbor.task.name == f"eva/{task.name}"
    assert harbor.metadata["condition"] == label
    assert harbor.metadata["perturbation"]["background_noise"] == "coffee_shop"
    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    assert "coffee_shop noise at 10 dB SNR" in instruction
    assert call_spec(instruction) == (task / "voice.toml").read_text().rstrip()
    # main brings EVA's numpy and WAVs; the sidecar is the clean task's.
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert dockerfile == images.MAIN_PERTURBED_DOCKERFILE
    assert f"numpy=={images.EVA_NUMPY}" in dockerfile
    assert _compose(task)["services"]["eva"]["image"] == images.sidecar_image(
        from_source=True
    )


def test_every_condition_uses_one_perturbed_main(tmp_path: Path) -> None:
    for condition in (
        PerturbationConfig("bad_connection_static"),
        PerturbationConfig(connection_degradation=True),
        PerturbationConfig("road_noise", 0.0),
    ):
        task = write_task(
            tmp_path / condition.label,
            "itsm",
            "16",
            perturbation=condition,
            registry="ghcr.io/o",
        )
        config = VoiceTaskConfig.from_toml(task / "voice.toml")
        assert config.environment is not None
        assert config.environment.docker_image == images.main_image(
            "ghcr.io/o", perturbed=True
        )
    assert images.main_image(perturbed=True).startswith("eva-main:perturbed-")
    assert images.main_image(perturbed=True) != images.main_image()


def test_the_cli_takes_evas_knobs_and_refuses_the_rest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as done:
        main(
            [
                "convert",
                "--out",
                str(tmp_path),
                "--domains",
                "itsm",
                "--limit",
                "1",
                "--background-noise",
                "bad_connection_static",
                "--snr-db",
                "5",
            ]
        )
    assert done.value.code == 0
    (task,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert task.name.endswith("-bad_connection_static-5db")
    assert "(bad_connection_static-5db)" in capsys.readouterr().out

    for flags, message in (
        (["--accent", "french"], "accent perturbations are not supported yet"),
        (
            ["--behavior", "elderly_slow"],
            "behavior perturbations are not supported yet",
        ),
        (["--snr-db", "10"], "needs --background-noise"),
        (["--background-noise", "jungle"], "invalid choice"),
    ):
        with pytest.raises(SystemExit) as refused:
            main(["convert", "--out", str(tmp_path / "x"), *flags])
        assert refused.value.code == 2
        assert message in capsys.readouterr().err
        assert not (tmp_path / "x").exists()


def test_a_folder_holds_one_condition(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """harbor's job metric pools a dataset folder: two conditions never share one."""
    base = ["convert", "--out", str(tmp_path), "--domains", "itsm", "--limit", "1"]
    with pytest.raises(SystemExit) as done:
        main([*base, "--background-noise", "coffee_shop"])
    assert done.value.code == 0
    with pytest.raises(SystemExit) as again:  # the same condition again is fine
        main([*base, "--background-noise", "coffee_shop", "--snr-db", "15"])
    assert again.value.code == 0
    capsys.readouterr()
    for flags in ([], ["--background-noise", "coffee_shop", "--snr-db", "10"]):
        with pytest.raises(SystemExit) as refused:
            main([*base, *flags])
        assert refused.value.code == 2
        assert "holds tasks of condition coffee_shop-15db" in capsys.readouterr().err
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 1


def test_a_published_registry_names_images_by_digest(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(images, "RECORD", tmp_path / "images.json")
    hub = "ghcr.io/amoh-ai"
    sidecar, clean, perturbed = images.refs(hub)
    digest = {ref: "sha256:" + f"{n}" * 64 for n, ref in enumerate((sidecar, clean), 1)}
    record = {ref.removeprefix(hub + "/"): d for ref, d in digest.items()}
    images.RECORD.write_text(json.dumps({hub: record}), encoding="utf-8")

    task = write_task(tmp_path, "airline", "1.1.2", registry=hub)
    assert _compose(task)["services"]["eva"]["image"] == f"{sidecar}@{digest[sidecar]}"
    config = VoiceTaskConfig.from_toml(task / "voice.toml")
    assert config.environment is not None
    assert config.environment.docker_image == f"{clean}@{digest[clean]}"
    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert f"FROM {clean}@{digest[clean]}" in dockerfile.splitlines()
    # The perturbed main isn't recorded: converting a condition fails first.
    with pytest.raises(SystemExit):
        main(["convert", "--out", str(tmp_path / "noisy"), "--domains", "airline",
              "--limit", "1", "--registry", hub, "--background-noise",
              "coffee_shop"])  # fmt: skip
    assert not (tmp_path / "noisy").exists()


def test_job_config_runs_a_hub_dataset(tmp_path: Path, monkeypatch) -> None:
    from eva_adapter import main as eva_main

    registry = tmp_path / "registry.json"
    entry = {"name": "eva", "version": "0.1.0", "description": "", "tasks": []}
    registry.write_text(json.dumps([entry]), encoding="utf-8")
    monkeypatch.setattr(eva_main, "REGISTRY", registry)
    out = tmp_path / "job.yaml"
    with pytest.raises(SystemExit):  # one of --dataset, --hub-dataset
        main(["job-config", "-o", str(out)])
    with pytest.raises(SystemExit):
        main(["job-config", "--hub-dataset", "eva@9", "-o", str(out)])
    with pytest.raises(SystemExit) as done:
        main(["job-config", "--hub-dataset", "eva@0.1.0", "-o", str(out)])
    assert done.value.code == 0
    raw = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert raw["datasets"] == [
        {"name": "eva", "registry_path": str(registry), "version": "0.1.0"}
    ]
