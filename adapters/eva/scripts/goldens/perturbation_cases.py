"""EVA's audio perturbations, case by case, run on EVA's ``AudioPerturbator`` or on ours.

Run inside EVA's image to (re)generate the goldens::

    MSYS_NO_PATHCONV=1 docker run --rm \\
        -v "$PWD/adapters/eva/scripts/goldens:/goldens" \\
        --entrypoint /opt/venv/bin/python -w /app eva-base:039efa0a \\
        /goldens/perturbation_cases.py --eva \\
        > tests/adapters/eva/fixtures/call/eva_perturbation.json

EVA's side is ``eva.user_simulator.perturbation.AudioPerturbator`` with EVA's
``PerturbationConfig``, reading the WAVs in its image (``/app/assets/noise``,
where EVA looks from its repo root). Ours is the vendored copy
(``eva_adapter.vendored.perturbation``) with ``eva_adapter.conditions``'
config, reading the same WAVs from wherever ``--noise-dir`` (or the test)
points it. ``tests/adapters/eva/test_eva_perturbation.py`` runs the cases on ours and
compares.

Seeded from outside. EVA draws the static, the packet loss and the gain
from numpy's global generator and never seeds it; each case here seeds it
(``np.random.seed``) before building the perturbator, on both sides, so the
two can be compared byte for byte. Inputs are built with integer arithmetic
only (``call_audio_cases.speech_like``). Each operation's output is compared
as its length and SHA-256: any one-sample difference fails.

Each case is a perturbator's life in one conversation, as the bridge drives
it: ``apply`` on the caller's chunks as ElevenLabs delivers them (sizes from
a few ms to seconds), interleaved with ``get_ambient_chunk`` for the 20 ms
frames of silence between turns, on one shared noise cursor.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from call_audio_cases import speech_like  # noqa: E402

SEED = 20261029
FRAME = 640
"""The bridge's 20 ms frame at 16 kHz (``SEND_CHUNK_SIZE_PCM``): its ambient chunk."""

# The script of one conversation: ("apply", bytes) or ("ambient", bytes).
TURNS: list[tuple[str, int]] = [
    ("ambient", FRAME),
    ("ambient", FRAME),
    ("ambient", FRAME),
    ("apply", 3200),  # 100 ms
    ("apply", 6400),
    ("apply", 640),
    ("apply", 1000),  # not a whole number of 20 ms segments
    ("apply", 12800),
    ("apply", 2),  # one sample
    ("apply", 0),  # nothing: returned as is
    ("ambient", FRAME),
    ("ambient", FRAME),
    ("apply", 32000),  # 1 s
    ("ambient", 2),
    ("ambient", FRAME),
]


def _cases() -> list[tuple[str, dict[str, Any], list[tuple[str, int]], str]]:
    """``(name, config, script, input)``; input is ``speech``, ``silence`` or ``loud``."""
    noises = (
        "airport_gate",
        "baby_crying",
        "background_music",
        "coffee_shop",
        "loud_construction",
        "nyc_street",
        "road_noise",
    )
    cases: list[tuple[str, dict[str, Any], list[tuple[str, int]], str]] = []
    for noise in noises:  # every WAV at EVA's default SNR
        cases.append(
            (f"noise/{noise}/15", {"background_noise": noise}, TURNS, "speech")
        )
    for snr in (0.0, 5.0, 7.5, 30.0, -5.0):
        cases.append(
            (
                f"noise/coffee_shop/{snr:g}",
                {"background_noise": "coffee_shop", "snr_db": snr},
                TURNS,
                "speech",
            )
        )
    for snr in (15.0, 5.0):
        cases.append(
            (
                f"static/{snr:g}",
                {"background_noise": "bad_connection_static", "snr_db": snr},
                TURNS,
                "speech",
            )
        )
    cases.append(("degradation", {"connection_degradation": True}, TURNS, "speech"))
    cases.append(
        (
            "combo/coffee_shop/10+degradation",
            {
                "background_noise": "coffee_shop",
                "snr_db": 10.0,
                "connection_degradation": True,
            },
            TURNS,
            "speech",
        )
    )
    cases.append(
        (
            "combo/static/15+degradation",
            {
                "background_noise": "bad_connection_static",
                "connection_degradation": True,
            },
            TURNS,
            "speech",
        )
    )
    # The bridge's ambient stream alone: 4 s of 20 ms frames between turns.
    for noise in ("coffee_shop", "bad_connection_static"):
        cases.append(
            (
                f"ambient/{noise}",
                {"background_noise": noise},
                [("ambient", FRAME)] * 200,
                "speech",
            )
        )
    # The shortest WAV (31 s) wraps inside one chunk, then the cursor goes on.
    cases.append(
        (
            "wrap/background_music",
            {"background_noise": "background_music"},
            [("apply", 1_200_000), ("ambient", FRAME), ("apply", 6400)],
            "speech",
        )
    )
    # Silence in (EVA's rms < 1e-9 branches) and speech near full scale (clipping).
    for name, config in (
        ("coffee_shop", {"background_noise": "coffee_shop"}),
        ("static", {"background_noise": "bad_connection_static"}),
        ("degradation", {"connection_degradation": True}),
    ):
        cases.append((f"silence/{name}", config, TURNS, "silence"))
        cases.append((f"loud/{name}", config, TURNS, "loud"))
    # A chunk of an odd number of bytes: both sides must fail alike.
    for name, config in (
        ("coffee_shop", {"background_noise": "coffee_shop"}),
        ("degradation", {"connection_degradation": True}),
    ):
        cases.append((f"odd/{name}", config, [("apply", 641)], "speech"))
    return cases


def _inputs() -> dict[str, bytes]:
    n = sum(size for op, size in TURNS if op == "apply")
    n = max(n, 1_200_000 + 6400)
    return {
        "speech": speech_like(n // 2, 16000, seed=11),
        "silence": b"\x00" * n,
        "loud": speech_like(n // 2, 16000, seed=12, level=30000),
    }


def digest(data: bytes) -> dict[str, Any]:
    return {"len": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def modules(use_eva: bool) -> tuple[Callable[..., Any], Callable[..., Any], Any]:
    """``(AudioPerturbator, config factory, its module)``, from EVA or from our copy."""
    if use_eva:
        from eva.models.config import PerturbationConfig
        from eva.user_simulator import perturbation

        return perturbation.AudioPerturbator, PerturbationConfig, perturbation
    from eva_adapter.conditions import PerturbationConfig
    from eva_adapter.vendored import perturbation

    return perturbation.AudioPerturbator, PerturbationConfig, perturbation


def run_case(
    use_eva: bool, config: dict[str, Any], script: list[tuple[str, int]], pcm: bytes
) -> dict[str, Any]:
    import numpy as np

    perturbator_class, config_class, _ = modules(use_eva)
    np.random.seed(SEED)
    out: dict[str, Any] = {"config": config, "ops": []}
    try:
        p = perturbator_class(config_class(**config))
    except Exception as exc:  # noqa: BLE001 - both sides must fail alike
        out["error"] = type(exc).__name__
        return out
    out["has_ambient_noise"] = p.has_ambient_noise
    whole = hashlib.sha256()
    pos = 0
    for op, size in script:
        try:
            if op == "apply":
                result = p.apply(pcm[pos : pos + size])
                pos += size
            else:
                result = p.get_ambient_chunk(size)
        except Exception as exc:  # noqa: BLE001 - both sides must fail alike
            out["ops"].append({"op": op, "n": size, "error": type(exc).__name__})
            continue
        whole.update(result)
        out["ops"].append({"op": op, "n": size, **digest(result)})
    out["cursor"] = int(p._noise_cursor)
    out["sha256"] = whole.hexdigest()
    return out


def knobs(use_eva: bool) -> dict[str, Any]:
    """The condition's vocabulary: EVA's config at the commit, or ours."""
    if use_eva:
        from eva.models.config import AccentType
        from eva.models.config import BackgroundNoiseType
        from eva.models.config import BehaviorType
        from eva.models.config import PerturbationConfig
        from eva.models.config import RunConfig
        from eva.user_simulator import perturbation

        fields = PerturbationConfig.model_fields
        settings = RunConfig.model_config
        return {
            "background_noise": [e.value for e in BackgroundNoiseType],
            "file_noise": sorted(perturbation._FILE_NOISE_TYPES),
            "accent": [e.value for e in AccentType],
            "behavior": [e.value for e in BehaviorType],
            "snr_db_default": fields["snr_db"].default,
            "connection_degradation_default": fields["connection_degradation"].default,
            "env_prefix": f"{settings['env_prefix']}PERTURBATION"
            f"{settings['env_nested_delimiter']}",
        }
    from eva_adapter import conditions

    return {
        "background_noise": list(conditions.BACKGROUND_NOISES),
        "file_noise": sorted(conditions.NOISE_FILES),
        "accent": list(conditions.ACCENTS),
        "behavior": list(conditions.BEHAVIORS),
        "snr_db_default": conditions.DEFAULT_SNR_DB,
        "connection_degradation_default": False,
        "env_prefix": conditions.ENV_PREFIX,
    }


def run_cases(use_eva: bool, noise_dir: Path | None = None) -> dict[str, Any]:
    """Every case on one side; ``noise_dir`` points ours at EVA's WAVs."""
    import numpy as np

    _, _, module = modules(use_eva)
    if noise_dir is not None:
        module._ASSETS_DIR = Path(noise_dir)
    inputs = _inputs()
    cases: dict[str, Any] = {
        "numpy": np.__version__,
        "knobs": knobs(use_eva),
        "inputs": {name: digest(data) for name, data in inputs.items()},
    }
    for name, config, script, source in _cases():
        cases[f"case/{name}"] = run_case(use_eva, config, script, inputs[source])
    return cases


def main() -> None:
    use_eva = "--eva" in sys.argv[1:]
    noise_dir = None
    if "--noise-dir" in sys.argv[1:]:
        noise_dir = Path(sys.argv[sys.argv.index("--noise-dir") + 1])
    json.dump(run_cases(use_eva, noise_dir), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
