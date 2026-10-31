"""EVA's audio perturbations as conditions: one per dataset, one per parity comparison.

EVA perturbs the caller's audio with its ``AudioPerturbator``
(``eva/user_simulator/perturbation.py`` @039efa0a), configured by
``PerturbationConfig`` (``eva/models/config.py:428-465``), which EVA's runs
set as ``EVA_PERTURBATION__*`` (``.env.example:461-483``). harbor-voice runs
the same class, vendored (:mod:`eva_adapter.vendored.perturbation`), at the
same two places in the call as EVA's bridge: each chunk of the caller's
16 kHz audio as it arrives, the perturbed copy to the agent and the clean one
to the recording (:meth:`~eva_adapter.bridge.Bridge.output`); and in place
of the silence the bridge streams while the caller is quiet. A condition is
EVA's three audio settings, under EVA's names:

- ``background_noise``: one of EVA's eight (:data:`BACKGROUND_NOISES`).
  Seven mix a WAV from EVA's repo (:data:`NOISE_FILES`);
  ``bad_connection_static`` is Gaussian noise. Mixed into the caller's
  speech at ``snr_db``, and streamed whenever the caller is quiet, the
  agent's turns included.
- ``snr_db``: EVA's default 15.0; EVA checks nothing else (its settings UI
  offers 0–40 in 1 dB steps).
- ``connection_degradation``: μ-law quantisation, 3 % packet loss per 20 ms
  and a random gain per chunk, on top of any noise.

EVA's other two axes are not supported yet: an accent needs ElevenLabs
agents that EVA's repo does not define, and a behaviour changes the caller's
prompt and agent. Both are refused by name.

One condition per dataset. ``eva-adapter convert`` writes it into every
task: ``[call.config.perturbation]`` in ``voice.toml``, the task's
``[metadata]``, and its name (:attr:`PerturbationConfig.label`), so trials of
two conditions never share a task, and a job over one dataset is one
condition.

The randomness is EVA's. The static, the packet loss and the gain draw
from numpy's global generator, unseeded, as in EVA, and nothing here seeds
it. Two runs of a condition, on either side, never hear the same noise
sample for sample: they compare statistically (against the noise floor of
two EVA runs), never trial by trial. The goldens seed it from outside, only
to show the code is EVA's (``tests/test_eva_perturbation.py``).

So is the arithmetic. EVA's perturbator scales noise by each chunk's
float32 RMS, and numpy 2.3 changed how a float32 sum of more than 8192
elements rounds. A perturbed task's ``main`` therefore installs EVA's numpy
(:data:`eva_adapter.images.EVA_NUMPY`), and a trial records the numpy it ran.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from eva_adapter import EVA_COMMIT

if TYPE_CHECKING:
    from eva_adapter.vendored.perturbation import AudioPerturbator

BACKGROUND_NOISES = (
    "airport_gate",
    "baby_crying",
    "background_music",
    "bad_connection_static",
    "coffee_shop",
    "loud_construction",
    "nyc_street",
    "road_noise",
)
"""EVA's ``BackgroundNoiseType`` (``config.py:374-384``), in EVA's order."""
STATIC = "bad_connection_static"
"""The one noise with no WAV: Gaussian, at the SNR, drawn per chunk."""
NOISE_FILES = {
    "airport_gate": "9bbc27e15a0c3cee05298816826cd0280cb2ab851d93e3ceae0c01e53b49b2f8",
    "baby_crying": "05595be7bb1087a2b5921e6394899520f09784db7e0895df56a8c77a76d24a06",
    "background_music": "ceb99f8a04c88153e4f1be7ff663953f4dfbb3118bf2f1a7772bf5ec7db0f779",
    "coffee_shop": "1dbc8e8f827950044b93962fed5b36416b0e4643023fda7fe7185bbeafb1b990",
    "loud_construction": "6ac497196e90ba59d9b94f6bae2ca68dd7010473a8eebaa9a9ad4c1addf3dda1",
    "nyc_street": "fdbcbfb64b98cb4bd2189a0a4946048109e9a545c50c959415b5f14163b83aa4",
    "road_noise": "6222fe6fb4cf0de4c82a042374e74e3d8f9c9f11d5b3eb6d637e475f422b0c33",
}
"""The WAVs EVA mixes (``perturbation.py:34-42``), ``assets/noise/<name>.wav``
in EVA's repo at the commit, by SHA-256 (16 kHz mono PCM16, 55 MB in all).
They are not redistributed here: ``main`` fetches them when it is built and
checks these (:data:`eva_adapter.images.PERTURBATION_LAYER`)."""
DEFAULT_SNR_DB = 15.0
"""EVA's default (``config.py:447-450``)."""
ACCENTS = ("french", "indian", "spanish", "chinese")
"""EVA's ``AccentType`` (``config.py:387-393``): not supported yet."""
BEHAVIORS = ("aggressive_impatient", "elderly_slow", "forgetful_disorganized")
"""EVA's ``BehaviorType`` (``config.py:396-401``): not supported yet."""
ENV_PREFIX = "EVA_PERTURBATION__"
"""EVA's settings for its ``perturbation`` field (``env_prefix`` ``EVA_``,
nested by ``__``)."""
CLEAN = "clean"
"""The label of no perturbation: today's tasks, unchanged."""
FIELDS = ("background_noise", "snr_db", "connection_degradation")


class ConditionError(ValueError):
    """Not one of EVA's conditions, or not one harbor-voice runs yet."""


def not_supported(axis: str) -> str:
    return (
        f"EVA's {axis} perturbations are not supported yet: harbor-voice runs "
        "EVA's audio perturbations only (background noise, SNR, connection "
        "degradation); see 'Perturbations' in adapters/eva/README.md"
    )


def _db(snr_db: float) -> str:
    """The SNR as it appears in a name: ``15``, ``7p5``, ``m5`` (no dots or signs)."""
    return f"{snr_db:g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True, slots=True)
class PerturbationConfig:
    """EVA's ``PerturbationConfig``, its audio fields: what ``AudioPerturbator`` reads.

    Clean is no config at all (``None``), so every condition has one
    spelling: a config always sets noise or degradation, and ``snr_db`` only
    with noise, where it means something.
    """

    background_noise: str | None = None
    snr_db: float = DEFAULT_SNR_DB
    connection_degradation: bool = False

    def __post_init__(self) -> None:
        if (
            self.background_noise is not None
            and self.background_noise not in BACKGROUND_NOISES
        ):
            raise ConditionError(
                f"background_noise {self.background_noise!r} is not one of EVA's: "
                + ", ".join(BACKGROUND_NOISES)
            )
        if isinstance(self.snr_db, bool) or not isinstance(self.snr_db, int | float):
            raise ConditionError(f"snr_db must be a number of dB, not {self.snr_db!r}")
        if not math.isfinite(self.snr_db):
            raise ConditionError(f"snr_db must be finite, not {self.snr_db!r}")
        if not isinstance(self.connection_degradation, bool):
            raise ConditionError("connection_degradation must be true or false")
        if self.background_noise is None and self.snr_db != DEFAULT_SNR_DB:
            raise ConditionError(
                "snr_db sets the background noise's level; it needs background_noise"
            )
        if self.background_noise is None and not self.connection_degradation:
            raise ConditionError(
                "a perturbation sets background_noise or connection_degradation "
                "(no perturbation is the clean condition: leave it out)"
            )

    @classmethod
    def from_options(
        cls,
        background_noise: str | None = None,
        snr_db: float | None = None,
        connection_degradation: bool = False,
        *,
        accent: str | None = None,
        behavior: str | None = None,
    ) -> PerturbationConfig | None:
        """A condition from EVA's knobs (CLI flags), ``None`` for clean."""
        if accent is not None:
            raise ConditionError(not_supported("accent"))
        if behavior is not None:
            raise ConditionError(not_supported("behavior"))
        if background_noise is None and snr_db is not None:
            raise ConditionError(
                "--snr-db sets the background noise's level; it needs --background-noise"
            )
        if background_noise is None and not connection_degradation:
            return None
        return cls(
            background_noise=background_noise,
            snr_db=DEFAULT_SNR_DB if snr_db is None else float(snr_db),
            connection_degradation=connection_degradation,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PerturbationConfig:
        """``[call.config.perturbation]``, or a JSON record of one."""
        for axis in ("accent", "behavior"):
            if data.get(axis) is not None:
                raise ConditionError(not_supported(axis))
        unknown = sorted(set(data) - {*FIELDS, "accent", "behavior"})
        if unknown:
            raise ConditionError(
                f"unknown perturbation setting(s) {', '.join(unknown)}; "
                f"EVA's audio settings are {', '.join(FIELDS)}"
            )
        snr = data.get("snr_db", DEFAULT_SNR_DB)
        if isinstance(snr, int) and not isinstance(snr, bool):
            snr = float(snr)  # TOML's `snr_db = 15`
        return cls(
            background_noise=data.get("background_noise"),
            snr_db=snr,
            connection_degradation=data.get("connection_degradation", False),
        )

    @property
    def label(self) -> str:
        """The condition's name, safe in task, job and folder names.

        ``coffee_shop-15db``, ``bad_connection_static-5db``,
        ``coffee_shop-15db-connection_degradation``,
        ``connection_degradation``.
        """
        parts = []
        if self.background_noise is not None:
            parts.append(f"{self.background_noise}-{_db(self.snr_db)}db")
        if self.connection_degradation:
            parts.append("connection_degradation")
        return "-".join(parts)

    def describe(self) -> str:
        """In words, for a task's instruction."""
        parts = []
        if self.background_noise == STATIC:
            parts.append(f"EVA's synthetic static at {self.snr_db:g} dB SNR")
        elif self.background_noise is not None:
            parts.append(
                f"EVA's {self.background_noise} noise at {self.snr_db:g} dB SNR"
            )
        if self.connection_degradation:
            parts.append("EVA's connection degradation")
        return " and ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        """EVA's audio fields, all three, under EVA's names (JSON records)."""
        return {
            "background_noise": self.background_noise,
            "snr_db": self.snr_db,
            "connection_degradation": self.connection_degradation,
        }

    def toml(self) -> str:
        """The body of ``[call.config.perturbation]`` (TOML has no null)."""
        lines = []
        if self.background_noise is not None:
            lines.append(f'background_noise = "{self.background_noise}"')
            lines.append(f"snr_db = {float(self.snr_db)!r}")
        lines.append(
            f"connection_degradation = {str(self.connection_degradation).lower()}"
        )
        return "\n".join(lines)

    def eva_env(self) -> dict[str, str]:
        """The same condition as EVA's own settings, for EVA's CLI.

        Only what differs from EVA's defaults, plus the SNR whenever there is
        noise, so the command says it.
        """
        env = {}
        if self.background_noise is not None:
            env[f"{ENV_PREFIX}BACKGROUND_NOISE"] = self.background_noise
            env[f"{ENV_PREFIX}SNR_DB"] = repr(float(self.snr_db))
        if self.connection_degradation:
            env[f"{ENV_PREFIX}CONNECTION_DEGRADATION"] = "true"
        return env

    def convert_args(self) -> list[str]:
        """The same condition as ``eva-adapter convert`` flags."""
        args = []
        if self.background_noise is not None:
            args += ["--background-noise", self.background_noise]
            args += ["--snr-db", repr(float(self.snr_db))]
        if self.connection_degradation:
            args.append("--connection-degradation")
        return args

    def plan(self) -> dict[str, Any]:
        """The trajectory's ``perturbation_plan``: EVA's layers, by EVA's names.

        ``plan_id`` is the trajectory's ``config_matrix.profile``, so a
        perturbed call never compares as a clean one. EVA seeds nothing, so
        there are no seeds to record.
        """
        layers: dict[str, dict[str, Any]] = {}
        if self.background_noise is not None:
            layers["eva.background_noise"] = {
                "type": self.background_noise,
                "snr_db": self.snr_db,
            }
        if self.connection_degradation:
            layers["eva.connection_degradation"] = {
                "codec": "mulaw",
                "packet_loss_prob": 0.03,
                "gain": [0.5, 1.2],
            }
        return {
            "plan_id": f"eva/{self.label}",
            "seeds": {},
            "layers": layers,
            "source": f"ServiceNow/eva@{EVA_COMMIT} "
            "src/eva/user_simulator/perturbation.py",
            "seeded": False,
        }


def label_of(config: PerturbationConfig | None) -> str:
    """The condition's name; :data:`CLEAN` for none."""
    return CLEAN if config is None else config.label


def as_record(config: PerturbationConfig | None) -> dict[str, Any] | None:
    """What a trial's summary (and a run's record) keeps: EVA's fields, or null."""
    return None if config is None else config.as_dict()


def from_record(data: Mapping[str, Any] | None) -> PerturbationConfig | None:
    """The inverse of :func:`as_record`; EVA's ``config.json`` reads the same.

    EVA records ``perturbation`` as its whole ``PerturbationConfig`` (accent
    and behaviour included, null when unset), or null; one that sets neither
    noise nor degradation is EVA's clean run (``base.py:63-67``).
    """
    if not data:
        return None
    if data.get("background_noise") is None:
        # EVA ignores an SNR without noise (it scales nothing).
        data = {k: v for k, v in data.items() if k != "snr_db"}
    if data.get("background_noise") is None and not data.get("connection_degradation"):
        if any(data.get(axis) is not None for axis in ("accent", "behavior")):
            return PerturbationConfig.from_mapping(data)  # refused by name
        return None
    return PerturbationConfig.from_mapping(data)


def perturbator(config: PerturbationConfig | None) -> AudioPerturbator | None:
    """EVA's perturbator for one conversation, or none (``base.py:63-67``).

    EVA builds one per conversation, so the noise file is read again and its
    cursor starts at 0 for every attempt; so does this.
    """
    if config is None or (
        config.background_noise is None and not config.connection_degradation
    ):
        return None
    from eva_adapter.vendored.perturbation import AudioPerturbator

    return AudioPerturbator(config)
