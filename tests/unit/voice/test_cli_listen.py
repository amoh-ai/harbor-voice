"""harbor-voice listen: the recording a trajectory names, found by its hash."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harbor_voice import cli
from harbor_voice.trajectory.stepped import Tick
from harbor_voice.trajectory.stepped import from_steps

pytestmark = pytest.mark.unit


def _wav(path: Path, fill: bytes = b"\x10") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(1)
        wav.setframerate(8000)
        wav.writeframes(fill * 2 * 8000)
    return path


def _trial(tmp_path: Path) -> tuple[Path, Path]:
    """A trial folder as a driver leaves it: the trajectory, and the
    recording it names, further down."""
    audio = _wav(tmp_path / "agent" / "call" / "attempt-0" / "both.wav")
    voice = from_steps(
        [Tick(0, "caller", "Hi.", True), Tick(2, "agent", "Hello.", True)],
        tick_s=0.2,
        agent="agent",
        caller="caller",
        n_steps=5,
        audio=audio,
    )
    trajectory = tmp_path / "agent" / "trajectory.json"
    trajectory.write_text(voice.to_atif_json(), encoding="utf-8")
    return trajectory, audio


def _listen(*args: str):
    return CliRunner().invoke(cli.voice_app, ["listen", *args])


def test_it_writes_the_recording_found_beside_the_trajectory(tmp_path: Path) -> None:
    trajectory, audio = _trial(tmp_path)
    # Same size, other samples: found by hash, not by name or size.
    _wav(tmp_path / "agent" / "call" / "attempt-0" / "decoy.wav", fill=b"\x20")
    out = tmp_path / "out" / "call.wav"

    result = _listen(str(trajectory), "--out", str(out))

    assert result.exit_code == 0, result.output
    assert out.read_bytes() == audio.read_bytes()
    assert "both.wav" in result.output


def test_it_looks_where_audio_dir_says(tmp_path: Path) -> None:
    trajectory, audio = _trial(tmp_path)
    elsewhere = tmp_path / "moved" / "both.wav"
    elsewhere.parent.mkdir()
    audio.replace(elsewhere)
    out = tmp_path / "call.wav"

    assert _listen(str(trajectory), "--out", str(out)).exit_code == 1
    result = _listen(
        str(trajectory), "--out", str(out), "--audio-dir", str(elsewhere.parent)
    )

    assert result.exit_code == 0, result.output
    assert out.read_bytes() == elsewhere.read_bytes()


def test_a_missing_recording_fails_and_writes_nothing(tmp_path: Path) -> None:
    trajectory, audio = _trial(tmp_path)
    audio.unlink()
    out = tmp_path / "call.wav"

    result = _listen(str(trajectory), "--out", str(out))

    assert result.exit_code == 1
    assert "--audio-dir" in result.output
    assert not out.exists()


def test_a_trajectory_without_a_recording_says_so(tmp_path: Path) -> None:
    voice = from_steps(
        [Tick(0, "caller", "Hi.", True)],
        tick_s=0.2,
        agent="agent",
        caller="caller",
        n_steps=2,
    )
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(voice.to_atif_json(), encoding="utf-8")

    result = _listen(str(trajectory), "--out", str(tmp_path / "call.wav"))

    assert result.exit_code == 2
    assert "names no recording" in result.output
