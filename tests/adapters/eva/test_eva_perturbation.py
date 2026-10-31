"""EVA's perturbations, harbor side: EVA's own ``AudioPerturbator``, where EVA's bridge applies it.

- The code is EVA's. ``eva_adapter.vendored.perturbation`` is compared
  with the file in ``eva-base:039efa0a``: the same but for ``_ASSETS_DIR``.
- Its bytes are EVA's (goldens). ``fixtures/call/eva_perturbation.json``
  is EVA's perturbator run in EVA's image on seeded inputs
  (``adapters/eva/scripts/goldens/perturbation_cases.py --eva``): every noise at EVA's
  default SNR, other SNRs, static, degradation, combinations, the ambient
  stream, a wrap of the noise file, silence, clipping. Our copy must give
  the same bytes, every case, inside a perturbed task's ``main`` (EVA's
  numpy). In the test environment the same holds for every chunk numpy
  rounds alike: all of them on EVA's numpy 2.2; on numpy >= 2.3, chunks of
  up to 8192 samples (numpy 2.3 changed how longer float32 sums round, and
  EVA takes each chunk's RMS that way: :data:`eva_adapter.images.EVA_NUMPY`).
- The bridge puts the perturbed copy on the wire and the clean one in the
  recording, and streams ambient noise whenever the caller is quiet, as
  EVA's does.
- The vocabulary is EVA's: the noises, the default SNR and the settings'
  names, from EVA's config at the commit (in the golden); accents and
  behaviours are refused by name.
- The original side: EVA's own settings parse our ``EVA_PERTURBATION__*``
  and EVA's perturbator finds its WAVs in EVA's image.

The Docker tests skip without Docker or ``eva-base:039efa0a``; the perturbed
``main`` is built on first use (it fetches EVA's WAVs and numpy).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy
import pytest

from eva_adapter import images
from eva_adapter import original
from eva_adapter.audio import SEND_CHUNK_SIZE_PCM
from eva_adapter.audio import pcm16k_to_mulaw8k
from eva_adapter.bridge import Bridge
from eva_adapter.conditions import ACCENTS
from eva_adapter.conditions import BEHAVIORS
from eva_adapter.conditions import NOISE_FILES
from eva_adapter.conditions import ConditionError
from eva_adapter.conditions import PerturbationConfig
from eva_adapter.conditions import from_record
from eva_adapter.conditions import label_of
from eva_adapter.conditions import perturbator
from eva_adapter.main import JUDGES
from eva_adapter.vendored import perturbation as vendored

HERE = Path(__file__).parent
ADAPTER = HERE.parents[2] / "adapters" / "eva"
GOLDENS_DIR = ADAPTER / "scripts" / "goldens"
sys.path.insert(0, str(GOLDENS_DIR))

import perturbation_cases  # noqa: E402

GOLDEN = json.loads(
    (HERE / "fixtures" / "call" / "eva_perturbation.json").read_text(encoding="utf-8")
)
CASES = {
    f"case/{name}": (config, script, source)
    for name, config, script, source in perturbation_cases._cases()
}
NEEDS_WAV = sorted(
    case
    for case, (config, _, _) in CASES.items()
    if config.get("background_noise") in NOISE_FILES
)
SYNTHETIC = sorted(set(CASES) - set(NEEDS_WAV))
EVA_IMAGE = images.EVA_BASE
EVA_FILE = "/app/src/eva/user_simulator/perturbation.py"
ROUNDING_LIMIT = 8192
"""Samples in a chunk up to which numpy >= 2.3 sums float32 as EVA's 2.2 does."""


def _same_numpy_as_eva() -> bool:
    return numpy.__version__.split(".")[:2] == images.EVA_NUMPY.split(".")[:2]


# ------------------------------------------------------------------ docker


def _docker(
    *args: str, check: bool = True, timeout: float = 600, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        input=stdin,
        stdin=None if stdin is not None else subprocess.DEVNULL,
        check=check,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def eva_image() -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    try:
        found = _docker("image", "inspect", EVA_IMAGE, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"docker unavailable: {exc}")
    if found.returncode != 0:
        pytest.skip(f"{EVA_IMAGE} not built (EVA's Dockerfile at 039efa0a)")
    return EVA_IMAGE


@pytest.fixture(scope="module")
def noise_dir(eva_image: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """EVA's WAVs, copied out of EVA's own image (``/app/assets/noise``)."""
    out = tmp_path_factory.mktemp("eva-noise")
    container = _docker("create", eva_image).stdout.decode().strip()
    try:
        _docker("cp", f"{container}:/app/assets/noise", str(out))
    finally:
        _docker("rm", container, check=False)
    return out / "noise"


@pytest.fixture(scope="module")
def perturbed_main(eva_image: str) -> str:
    """A perturbed task's ``main``, built as a task builds it if it is missing."""
    tag = images.main_image(perturbed=True)
    if _docker("image", "inspect", tag, check=False, timeout=60).returncode != 0:
        _docker(
            "build",
            "-q",
            "-t",
            tag,
            "-",
            stdin=images.MAIN_PERTURBED_DOCKERFILE.encode(),
            timeout=1200,
        )
    return tag


# ------------------------------------------------------------ the code


def test_the_vendored_file_is_evas_but_for_where_the_wavs_are(eva_image: str) -> None:
    theirs = _docker("run", "--rm", "--entrypoint", "cat", eva_image, EVA_FILE).stdout
    ours = Path(vendored.__file__).read_bytes().replace(b"\r\n", b"\n")
    header, _, body = ours.partition(b'"""Audio perturbations')
    assert header.startswith(b"# Vendored from ServiceNow/eva @ 039efa0a")
    body = b'"""Audio perturbations' + body
    eva_line = b'_ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets" / "noise"\n'
    ours_lines = (
        b"# harbor-voice: EVA's WAVs are where eva_adapter.images puts them in main, not\n"
        b"# beside an EVA checkout (this file is not in one).\n"
        b'_ASSETS_DIR = Path("/opt/eva/assets/noise")\n'
    )
    assert body.count(ours_lines) == 1
    assert body.replace(ours_lines, eva_line) == theirs


def test_the_copy_reads_the_wavs_where_main_keeps_them() -> None:
    assert vendored._ASSETS_DIR == Path(images.NOISE_DIR)
    assert vendored._FILE_NOISE_TYPES == set(NOISE_FILES)


def test_evas_image_holds_the_wavs_main_fetches(noise_dir: Path) -> None:
    found = {
        p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in noise_dir.glob("*.wav")
    }
    assert found == NOISE_FILES
    layer = images.PERTURBATION_LAYER
    assert images.EVA_COMMIT_FULL in layer and "sha256sum -c -" in layer
    assert all(f"{digest} {name}.wav" in layer for name, digest in NOISE_FILES.items())


def test_the_vocabulary_is_evas_config() -> None:
    assert perturbation_cases.knobs(use_eva=False) == GOLDEN["knobs"]


# ----------------------------------------------------------------- goldens


@pytest.fixture(scope="module")
def inputs() -> dict[str, bytes]:
    made = perturbation_cases._inputs()
    assert {k: perturbation_cases.digest(v) for k, v in made.items()} == GOLDEN[
        "inputs"
    ]
    return made


def _compare(case: str, ours: dict[str, Any]) -> None:
    """Every operation's bytes. A noise chunk numpy >= 2.3 rounds differently
    (its RMS is a float32 sum) is left to the perturbed ``main`` test, and
    only on such a numpy; degradation alone sums nothing."""
    theirs = GOLDEN[case]
    exact = _same_numpy_as_eva() or CASES[case][0].get("background_noise") is None
    assert ours.get("error") == theirs.get("error")
    assert ours.get("has_ambient_noise") == theirs.get("has_ambient_noise")
    assert ours.get("cursor") == theirs.get("cursor")
    assert len(ours["ops"]) == len(theirs["ops"])
    skipped = 0
    for i, (a, b) in enumerate(zip(ours["ops"], theirs["ops"], strict=True)):
        if not exact and b["op"] == "apply" and b["n"] > 2 * ROUNDING_LIMIT:
            assert a.get("len") == b.get("len"), f"{case} op {i}"
            skipped += 1
            continue
        assert a == b, f"{case} op {i}"
    if not skipped:
        assert ours.get("sha256") == theirs.get("sha256")


@pytest.mark.parametrize("case", SYNTHETIC)
def test_synthetic_conditions_give_evas_bytes(
    case: str, inputs: dict[str, bytes]
) -> None:
    config, script, source = CASES[case]
    _compare(case, perturbation_cases.run_case(False, config, script, inputs[source]))


@pytest.mark.parametrize("case", NEEDS_WAV)
def test_noise_conditions_give_evas_bytes(
    case: str,
    inputs: dict[str, bytes],
    noise_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vendored, "_ASSETS_DIR", noise_dir)
    config, script, source = CASES[case]
    _compare(case, perturbation_cases.run_case(False, config, script, inputs[source]))


def test_in_perturbed_main_every_case_is_evas_byte_for_byte(
    perturbed_main: str,
) -> None:
    """The harbor side as it runs: Python 3.13, EVA's numpy, the WAVs main fetched."""
    out = _docker(
        "run",
        "--rm",
        "-v",
        f"{ADAPTER / 'src'}:/src:ro",
        "-v",
        f"{GOLDENS_DIR}:/goldens:ro",
        "-e",
        "PYTHONPATH=/src",
        perturbed_main,
        "python",
        "/goldens/perturbation_cases.py",
    )
    ours = json.loads(out.stdout)
    assert ours["numpy"] == images.EVA_NUMPY == GOLDEN["numpy"]
    assert ours.keys() == GOLDEN.keys()
    assert [k for k in GOLDEN if k != "numpy" and ours[k] != GOLDEN[k]] == []


def test_eva_seeds_nothing_so_two_conversations_differ() -> None:
    """EVA's static draws from numpy's global generator, unseeded: the same
    caller audio is perturbed differently each time (compare runs
    statistically, never sample for sample)."""
    config = PerturbationConfig(background_noise="bad_connection_static")
    speech = bytes(range(256)) * 25
    first = perturbator(config)
    second = perturbator(config)
    assert first is not None and second is not None
    assert first.apply(speech) != second.apply(speech)


# ------------------------------------------------------------------ bridge


class Wire:
    def __init__(self) -> None:
        self.is_open = True
        self.media: list[bytes] = []

    def send_media(self, mulaw: bytes) -> None:
        self.media.append(mulaw)

    def send_user_speech_start(self, timestamp_ms: str) -> None:
        pass

    def send_user_speech_stop(self, timestamp_ms: str) -> None:
        pass

    def send_stop(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False


class Spy:
    """EVA's perturbator surface, marking what it touched."""

    AMBIENT = (1000).to_bytes(2, "little", signed=True)

    def __init__(self, *, ambient: bool) -> None:
        self.has_ambient_noise = ambient
        self.applied: list[bytes] = []
        self.ambient_asks: list[int] = []

    def apply(self, pcm_bytes: bytes) -> bytes:
        self.applied.append(pcm_bytes)
        return bytes(b ^ 0x55 for b in pcm_bytes)

    def get_ambient_chunk(self, n_bytes: int) -> bytes:
        self.ambient_asks.append(n_bytes)
        return self.AMBIENT * (n_bytes // 2)


async def test_the_wire_gets_the_perturbed_audio_and_the_recording_the_clean() -> None:
    wire = Wire()
    spy = Spy(ambient=False)
    recorded: list[tuple[str, bytes]] = []
    bridge = Bridge(
        wire,
        "c",
        record_callback=lambda source, pcm: recorded.append((source, pcm)),
        perturbator=spy,
    )
    await bridge.start_async()
    clean = bytes(range(256)) * 10  # 2560 bytes: four 20 ms frames
    bridge.output(clean)
    for _ in range(100):
        if len(wire.media) >= 4:
            break
        await asyncio.sleep(0.01)
    await bridge.stop_async()

    assert spy.applied == [clean]  # each caller chunk as it arrives, once
    assert recorded == [("user_clean", clean)]
    perturbed = bytes(b ^ 0x55 for b in clean)
    frame = SEND_CHUNK_SIZE_PCM
    assert wire.media[:4] == [
        pcm16k_to_mulaw8k(perturbed[i : i + frame]) for i in range(0, len(clean), frame)
    ]
    assert spy.ambient_asks == []  # no ambient noise: silence stays EVA's


async def test_ambient_noise_fills_the_callers_silence_always_open_mic() -> None:
    """EVA streams ambient noise whenever the caller is quiet, from the
    start and through the agent's speech; the clean track records silence."""
    wire = Wire()
    spy = Spy(ambient=True)
    recorded: list[tuple[str, bytes]] = []
    bridge = Bridge(
        wire,
        "c",
        record_callback=lambda source, pcm: recorded.append((source, pcm)),
        perturbator=spy,
    )
    await bridge.start_async()
    for _ in range(100):
        if len(wire.media) >= 3:
            break
        await asyncio.sleep(0.01)
    before_agent = len(wire.media)
    bridge.receive_media(b"\xff" * 160)  # the agent starts speaking
    assert bridge.is_assistant_playing()
    for _ in range(100):
        if len(wire.media) >= before_agent + 3:
            break
        await asyncio.sleep(0.01)
    await bridge.stop_async()

    ambient = pcm16k_to_mulaw8k(Spy.AMBIENT * (SEND_CHUNK_SIZE_PCM // 2))
    assert before_agent >= 3 and len(wire.media) >= before_agent + 3
    assert set(wire.media) == {ambient}
    assert set(spy.ambient_asks) == {SEND_CHUNK_SIZE_PCM}
    assert spy.applied == []
    assert recorded and set(recorded) == {("user_clean", b"\x00" * SEND_CHUNK_SIZE_PCM)}
    assert len(recorded) == len(wire.media)


async def test_without_ambient_noise_nothing_is_sent_before_the_agent_speaks() -> None:
    wire = Wire()
    spy = Spy(ambient=False)  # connection degradation alone
    bridge = Bridge(wire, "c", perturbator=spy)
    await bridge.start_async()
    await asyncio.sleep(0.15)
    await bridge.stop_async()
    assert wire.media == [] and spy.ambient_asks == []


async def test_evas_own_static_reaches_the_wire_while_the_caller_is_quiet() -> None:
    wire = Wire()
    bridge = Bridge(
        wire,
        "c",
        perturbator=perturbator(
            PerturbationConfig(background_noise="bad_connection_static")
        ),
    )
    await bridge.start_async()
    for _ in range(100):
        if len(wire.media) >= 3:
            break
        await asyncio.sleep(0.01)
    await bridge.stop_async()
    silence = pcm16k_to_mulaw8k(b"\x00" * SEND_CHUNK_SIZE_PCM)
    assert len(wire.media) >= 3 and all(m != silence for m in wire.media)


# -------------------------------------------------------------- conditions


def test_a_condition_is_named_once_by_evas_settings() -> None:
    noise = PerturbationConfig.from_options("coffee_shop")
    assert noise == PerturbationConfig("coffee_shop", 15.0, False)
    assert noise.label == "coffee_shop-15db"
    assert PerturbationConfig("coffee_shop", 7.5, True).label == (
        "coffee_shop-7p5db-connection_degradation"
    )
    assert PerturbationConfig("road_noise", -5.0).label == "road_noise-m5db"
    assert PerturbationConfig(connection_degradation=True).label == (
        "connection_degradation"
    )
    assert PerturbationConfig.from_options() is None
    assert label_of(None) == "clean"
    assert noise.eva_env() == {
        "EVA_PERTURBATION__BACKGROUND_NOISE": "coffee_shop",
        "EVA_PERTURBATION__SNR_DB": "15.0",
    }
    assert PerturbationConfig(connection_degradation=True).eva_env() == {
        "EVA_PERTURBATION__CONNECTION_DEGRADATION": "true"
    }
    assert noise.convert_args() == [
        "--background-noise",
        "coffee_shop",
        "--snr-db",
        "15.0",
    ]
    # A record (a trial's summary, EVA's config.json) reads back the same.
    assert from_record(noise.as_dict()) == noise
    assert (
        from_record(
            {
                "background_noise": "coffee_shop",
                "snr_db": 15.0,
                "accent": None,
                "behavior": None,
                "connection_degradation": False,
            }
        )
        == noise
    )
    assert (
        from_record(
            {
                "background_noise": None,
                "snr_db": 15.0,
                "accent": None,
                "behavior": None,
                "connection_degradation": False,
            }
        )
        is None
    )
    assert from_record(None) is None
    # EVA ignores an SNR without noise, so its record reads as what EVA ran.
    assert from_record(
        {"background_noise": None, "snr_db": 20.0, "connection_degradation": True}
    ) == PerturbationConfig(connection_degradation=True)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"accent": ACCENTS[0]}, "accent perturbations are not supported yet"),
        ({"behavior": BEHAVIORS[0]}, "behavior perturbations are not supported yet"),
        ({"snr_db": 10.0}, "needs --background-noise"),
    ],
)
def test_what_harbor_does_not_run_is_refused_by_name(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConditionError, match=message):
        PerturbationConfig.from_options(**options)


def test_a_bad_table_is_refused() -> None:
    with pytest.raises(ConditionError, match="not one of EVA's"):
        PerturbationConfig.from_mapping({"background_noise": "jungle"})
    with pytest.raises(ConditionError, match="unknown perturbation setting"):
        PerturbationConfig.from_mapping({"background_noise": "coffee_shop", "snr": 3})
    with pytest.raises(ConditionError, match="accent perturbations"):
        PerturbationConfig.from_mapping({"accent": "french"})
    with pytest.raises(ConditionError, match="finite"):
        PerturbationConfig("coffee_shop", float("nan"))
    with pytest.raises(ConditionError, match="leave it out"):
        PerturbationConfig.from_mapping({"connection_degradation": False})
    assert (
        PerturbationConfig.from_mapping(
            {"background_noise": "coffee_shop", "snr_db": 15}
        ).snr_db
        == 15.0
    )


def test_a_perturbator_exactly_when_eva_builds_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``base.py:63-67``: noise or degradation; a missing WAV fails at once."""
    assert perturbator(None) is None
    built = perturbator(PerturbationConfig(connection_degradation=True))
    assert built is not None and not built.has_ambient_noise
    static = perturbator(PerturbationConfig("bad_connection_static"))
    assert static is not None and static.has_ambient_noise
    monkeypatch.setattr(vendored, "_ASSETS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="coffee_shop.wav"):
        perturbator(PerturbationConfig("coffee_shop"))


# ---------------------------------------------------------- original side


@pytest.mark.parametrize(
    "condition",
    [
        PerturbationConfig("coffee_shop"),
        PerturbationConfig("bad_connection_static", 7.5, True),
        PerturbationConfig(connection_degradation=True),
    ],
    ids=lambda c: c.label,
)
def test_evas_own_settings_read_the_condition_and_find_the_wavs(
    eva_image: str, condition: PerturbationConfig
) -> None:
    """The check before a perturbed EVA run, in EVA's image."""
    judges = json.dumps(json.loads(JUDGES.read_text(encoding="utf-8")))
    out = _docker(
        "run",
        "--rm",
        "-e",
        f"EVA_MODEL_LIST={judges}",
        "--entrypoint",
        "/bin/sh",
        eva_image,
        "-c",
        original.perturbation_check(condition),
        check=False,
    )
    assert out.returncode == 0, out.stderr.decode()[-1500:]
    assert b"noise files in /app/assets/noise" in out.stdout
