"""The original side runs tau2's own command in the mode the harbor side reproduces."""

from __future__ import annotations

import shlex

from tau_voice.main import REVIEW_MODEL
from tau_voice.main import trial_seeds
from tau_voice.original import tau2_command


def _flags(command: str) -> dict[str, str | None]:
    parts = shlex.split(command)
    assert parts[:2] == ["tau2", "run"]
    flags: dict[str, str | None] = {}
    rest = parts[2:]
    for i, part in enumerate(rest):
        if part.startswith("--"):
            value = rest[i + 1] if i + 1 < len(rest) else None
            flags[part] = None if value is None or value.startswith("--") else value
    return flags


def test_the_original_run_matches_the_harbor_side() -> None:
    flags = _flags(
        tau2_command(
            "airline",
            model="gpt-realtime-1.5",
            split="test",
            trials=3,
            seed=300,
            concurrency=4,
            review_model=REVIEW_MODEL,
            save_to="parity-airline",
        )
    )
    assert flags["--audio-native"] is None
    assert flags["--audio-native-provider"] == "openai"
    assert flags["--audio-native-model"] == "gpt-realtime-1.5"
    assert flags["--speech-complexity"] == "control"  # the mode we reproduce
    assert flags["--task-split-name"] == "test"
    assert flags["--num-trials"] == "3"
    # Run seed 300 is what gives the harbor side its per-trial seeds.
    assert flags["--seed"] == "300" and trial_seeds(300, 3)[0] == 626729
    assert flags["--hallucination-retries"] == "3"
    assert flags["--review-model"] == REVIEW_MODEL
    assert "--task-ids" not in flags


def test_a_smoke_run_names_its_tasks() -> None:
    command = tau2_command(
        "retail",
        model="m",
        split="test",
        trials=1,
        seed=300,
        concurrency=1,
        review_model="r",
        save_to="s",
        task_ids=["2", "6"],
    )
    parts = shlex.split(command)
    at = parts.index("--task-ids")
    assert parts[at + 1 : at + 3] == ["2", "6"]


def test_results_unpack_on_windows_despite_telecom_task_ids(tmp_path) -> None:  # noqa: ANN001
    import io
    import tarfile

    from tau_voice.original import unpack

    name = "smoke-telecom/artifacts/task_[mobile]x|y[PERSONA:Easy]/audio.wav"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, body in ((name, b"wav"), ("smoke-telecom/results.json", b"{}")):
            info = tarfile.TarInfo(path)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    unpack(buffer.getvalue(), tmp_path)
    assert (tmp_path / "smoke-telecom" / "results.json").read_bytes() == b"{}"
    (wav,) = (tmp_path / "smoke-telecom" / "artifacts").glob("*/audio.wav")
    assert "|" not in wav.parent.name and wav.read_bytes() == b"wav"


def test_the_preset_is_passed_through() -> None:
    command = tau2_command(
        "airline",
        model="m",
        split="test",
        trials=1,
        seed=300,
        concurrency=1,
        review_model="r",
        save_to="s",
        complexity="control_audio",
        user_llm="gpt-4.1-mini-2025-04-14",
    )
    assert _flags(command)["--speech-complexity"] == "control_audio"
    assert _flags(command)["--user-llm"] == "gpt-4.1-mini-2025-04-14"


def test_openai_live_runs_as_tau2s_live_provider_without_realtime_generation() -> None:
    command = tau2_command(
        "airline",
        model="gpt-live-1",
        split="test",
        trials=1,
        seed=300,
        concurrency=1,
        review_model="r",
        save_to="s",
        live_config={"backend_model": "gpt-6-luna"},
    )
    flags = _flags(command)
    assert flags["--audio-native-provider"] == "openai_live"
    assert flags["--audio-native-model"] == "gpt-live-1"
    assert flags["--live-config"] == '{"backend_model": "gpt-6-luna"}'
    # tau2 turns real-time user generation on for Live; our caller does not.
    assert "--no-realtime-generation" in flags
