"""The call golden: harbor's call loop reproduces EVA's, event for event.

EVA's side is checked in: ``fixtures/call/g5`` is the record EVA's own
bridge, Realtime server and ElevenLabs simulator wrote for the scripted call
in ``scripts/goldens/call_scenario.py``, in ``eva-base:039efa0a``. This runs
the same script through harbor's port (``RealtimeCall`` with the real
``openai.realtime`` and ``elevenlabs.agents`` transports and the real
``ElevenLabsCaller``, each on a fake socket) and compares the two records.

Both sides run on the same virtual clock (``call_vtime``). On the wall clock
EVA does not reproduce itself: its 250 ms input-stream tick and its
non-catching-up pacer move timestamps by up to ~250 ms from run to run
(measured, EVA against EVA). On the virtual clock EVA's two runs agree to the
millisecond, so the comparison is of logic, not scheduling.

Regenerate the fixture (in the repo root)::

    OUT=$(mktemp -d)
    MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD/adapters/eva/scripts/goldens:/goldens" \\
        -v "$OUT:/out" --entrypoint /opt/venv/bin/python eva-base:039efa0a \\
        /goldens/call_eva_golden.py /out/eva
    uv run python adapters/eva/scripts/goldens/call_compare.py fixture \\
        "$OUT/eva" adapters/eva/tests/fixtures/call/g5
    MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD/adapters/eva/scripts/goldens:/goldens" \\
        -v "$OUT:/records" --entrypoint /opt/venv/bin/python eva-base:039efa0a \\
        /goldens/call_processor.py /records/eva \\
        > adapters/eva/tests/fixtures/call/g5/processor.json
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).parent
GOLDENS = HERE.parent / "scripts" / "goldens"
FIXTURE = HERE / "fixtures" / "call" / "g5"
EVA_IMAGE = "eva-base:039efa0a"
sys.path.insert(0, str(GOLDENS))

from call_compare import compare  # noqa: E402
from call_compare import compare_processor  # noqa: E402
from call_harbor import run_harbor_virtual  # noqa: E402

TIMESTAMP_MS = 20.0
"""The brief's tolerance; on the virtual clock the measured worst is ~1 ms
(EVA's stamps are truncated to the millisecond)."""
LATENCY_MS = 30.0
STREAMS = ("events", "audit", "framework", "metrics")
TRACES = ("agent_frames", "caller_heard", "pads")


@pytest.fixture(scope="module")
def harbor_record(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("g5") / "harbor"
    # EVA's start instant: without it, which near-simultaneous stamps share a
    # millisecond (and so where EVA puts "[user interrupts]") varied by run.
    eva_t0 = json.loads((FIXTURE / "g5_meta.json").read_text(encoding="utf-8"))["t0"]
    meta = run_harbor_virtual(out, epoch0=eva_t0)
    assert meta["error"] is None, meta["error"]
    assert meta["agent_error"] is None, meta["agent_error"]
    return out


@pytest.fixture(scope="module")
def report(harbor_record: Path) -> dict[str, Any]:
    return compare(FIXTURE, harbor_record)


def test_the_call_ends_as_in_eva(report: dict[str, Any]) -> None:
    assert report["end_reason"] == ["goodbye", "goodbye"]
    eva, harbor = report["stats"]
    for key in ("num_turns", "num_tool_calls", "tools_called"):
        assert harbor[key] == eva[key]


@pytest.mark.parametrize("stream", STREAMS)
def test_record_stream_matches_eva(report: dict[str, Any], stream: str) -> None:
    result = report["streams"][stream]
    assert result["same_sequence"], result.get("first_mismatch")
    assert result["max_abs_dt_ms"] <= TIMESTAMP_MS, result["worst"]


@pytest.mark.parametrize("trace", TRACES)
def test_audio_path_matches_eva(report: dict[str, Any], trace: str) -> None:
    """What the agent received, what the caller heard, how the tracks were aligned."""
    result = report["traces"][trace]
    assert result["same_sequence"], result.get("first_mismatch")
    assert result["max_abs_dt_ms"] <= TIMESTAMP_MS, result["worst"]


def test_caller_hears_eva_bytes_up_to_soxr_dither(report: dict[str, Any]) -> None:
    assert report["caller_heard_voiced_max_diff"] <= 4


def test_interrupted_partial_matches_eva(report: dict[str, Any]) -> None:
    eva, harbor = report["partials"]
    assert eva and harbor == eva


def test_bridge_latencies_match_eva(report: dict[str, Any]) -> None:
    deltas = report["response_latencies_ms"]
    assert deltas is not None and len(deltas) == 4
    assert max(abs(d) for d in deltas) <= TIMESTAMP_MS


def test_wav_lengths_match_eva(report: dict[str, Any]) -> None:
    for name, (eva, harbor) in report["wav_frames"].items():
        assert eva is not None and harbor is not None, name
        assert harbor[1] == eva[1], name
        one_frame = eva[1] // 50  # 20 ms
        assert abs(harbor[0] - eva[0]) <= one_frame, (name, eva, harbor)


def _eva_image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", EVA_IMAGE],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


@pytest.mark.skipif(not _eva_image_available(), reason=f"needs Docker and {EVA_IMAGE}")
def test_eva_processor_reads_harbor_record_as_eva_record(harbor_record: Path) -> None:
    """EVA's ``MetricsContextProcessor`` on harbor's record: EVA's turns and latencies."""
    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{GOLDENS}:/goldens",
            "-v",
            f"{harbor_record.parent}:/records",
            "--entrypoint",
            "/opt/venv/bin/python",
            EVA_IMAGE,
            "/goldens/call_processor.py",
            f"/records/{harbor_record.name}",
        ],
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert run.returncode == 0, run.stderr.decode(errors="replace")[-2000:]
    harbor = json.loads(run.stdout)[harbor_record.name]
    eva = json.loads((FIXTURE / "processor.json").read_text())["eva"]
    result = compare_processor(eva, harbor)
    assert all(result["equal"].values()), result["equal"]
    assert result["latency_ms"]
    assert max(abs(v) for v in result["latency_ms"].values()) <= LATENCY_MS
    for role, turns in result["segments_ms"].items():
        for turn, segments in turns.items():
            for start, end in segments:
                assert abs(start) <= TIMESTAMP_MS and abs(end) <= TIMESTAMP_MS, (
                    role,
                    turn,
                )
