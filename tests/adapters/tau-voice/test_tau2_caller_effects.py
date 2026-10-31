"""Our caller with tau2's effects sends tau2's bytes, step for step.

The golden is tau2's own caller run on a scripted call
(``tau2_caller_effects_cases.py --tau2``, in the sidecar image); this runs
the same script through :class:`~tau_voice.voice.SimulatorVoice` with
:class:`~tau_voice.effects.Tau2CallerEffects`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from tau2_caller_effects_cases import run_ours  # noqa: E402

pytestmark = pytest.mark.unit

GOLDEN = json.loads(
    (HERE / "fixtures" / "tau2_caller_effects.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
async def ours(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return await run_ours(GOLDEN, tmp_path_factory.mktemp("noise"))


async def test_lines_get_tau2_tics_and_muffling(ours: dict) -> None:
    assert ours["lines"] == GOLDEN["result"]["lines"]


async def test_timeline_matches(ours: dict) -> None:
    assert ours["timeline"] == GOLDEN["result"]["timeline"]


async def test_step_text_matches(ours: dict) -> None:
    assert ours["texts"] == GOLDEN["result"]["texts"]


async def test_each_step_carries_tau2s_effect_results(ours: dict) -> None:
    """What tau2's chunks carry: the line's speech effects, the step's source
    effects, the turn's channel effects plus frame drops."""
    assert ours["step_effects_carried"] == GOLDEN["result"]["step_effects_carried"]
    assert ours["step_effects"] == GOLDEN["result"]["step_effects"]


async def test_wire_audio_matches_step_for_step(ours: dict) -> None:
    expected = GOLDEN["result"]["step_digests"]
    assert len(ours["step_digests"]) == len(expected)
    first = next(
        (i for i, (a, b) in enumerate(zip(ours["step_digests"], expected)) if a != b),
        None,
    )
    assert first is None, f"first differing step: {first}"
    assert ours["wire"] == GOLDEN["result"]["wire"]


def test_the_case_exercises_every_effect() -> None:
    counts = GOLDEN["result"]["event_counts"]
    assert {"burst_noise", "frame_drop", "out_of_turn_speech"} <= set(counts)
    lines = GOLDEN["result"]["lines"]
    assert any(muffled for _, muffled in lines)
    assert any("[" in said for said, _ in lines)


async def test_run_fetches_tau2_noise_once(tmp_path: Path) -> None:
    import httpx

    from tau_voice.client import Sidecar
    from tau_voice.client import SidecarError
    from tau_voice.run import tau2_effects

    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched.append(request.url.path)
        return httpx.Response(200, content=b"RIFF" + request.url.path.encode())

    sidecar = Sidecar(
        "http://sidecar",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    user = {
        "voice_settings": GOLDEN["voice_settings"],
        "tau2_burst_files": GOLDEN["tau2_burst_files"],
    }
    effects = await tau2_effects(sidecar, user, step_s=0.2, folder=tmp_path)
    env = GOLDEN["voice_settings"]["speech_environment"]
    assert fetched == [f"/asset/continuous/{env['background_noise_file']}"] + [
        f"/asset/bursts/{name}" for name in env["burst_noise_files"]
    ]
    bed = effects.assets[env["background_noise_file"]]
    assert bed == tmp_path / "continuous" / env["background_noise_file"]
    assert bed.read_bytes().startswith(b"RIFF")

    await tau2_effects(sidecar, user, step_s=0.2, folder=tmp_path)
    assert len(fetched) == 1 + len(env["burst_noise_files"])  # cached

    bad = json.loads(json.dumps(user))
    bad["voice_settings"]["speech_environment"]["background_noise_file"] = "../x.wav"
    with pytest.raises(SidecarError):
        await tau2_effects(sidecar, bad, step_s=0.2, folder=tmp_path)
