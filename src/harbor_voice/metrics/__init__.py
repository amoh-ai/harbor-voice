"""Metrics.

Importing this package registers every family into :data:`REGISTRY`, so a
caller that only wants to score a trajectory needs one import.
"""

from __future__ import annotations

from harbor_voice.metrics import families as _families  # noqa: F401 - registration
from harbor_voice.metrics.base import REGISTRY
from harbor_voice.metrics.base import Direction
from harbor_voice.metrics.base import Family
from harbor_voice.metrics.base import MetricContext
from harbor_voice.metrics.base import MetricDef
from harbor_voice.metrics.base import MetricRegistry
from harbor_voice.metrics.base import MetricResult
from harbor_voice.metrics.base import NotApplicable
from harbor_voice.metrics.base import Outcome
from harbor_voice.metrics.base import Value
from harbor_voice.metrics.base import evaluate_metric
from harbor_voice.metrics.base import metric
from harbor_voice.metrics.outputs import Outputs
from harbor_voice.metrics.outputs import build_measurement
from harbor_voice.metrics.outputs import build_outputs
from harbor_voice.metrics.outputs import build_reward
from harbor_voice.metrics.outputs import build_validity
from harbor_voice.metrics.rollup import Bootstrap
from harbor_voice.metrics.rollup import Rollup
from harbor_voice.metrics.rollup import TrialResults
from harbor_voice.metrics.rollup import bootstrap_mean
from harbor_voice.metrics.rollup import harbor_metric_payload
from harbor_voice.metrics.rollup import load_job
from harbor_voice.metrics.rollup import load_trial_dir
from harbor_voice.metrics.rollup import paired_delta
from harbor_voice.metrics.rollup import roll_up
from harbor_voice.metrics.rollup import roll_up_all
from harbor_voice.metrics.task import TASK_PREFIX
from harbor_voice.metrics.task import SelectionError
from harbor_voice.metrics.task import SelectionProblem
from harbor_voice.metrics.task import TaskNamespaceError
from harbor_voice.metrics.task import check_selection
from harbor_voice.metrics.task import require_valid_selection
from harbor_voice.metrics.task import task_metric
from harbor_voice.metrics.text import edit_counts
from harbor_voice.metrics.text import word_error_rate

__all__ = [
    "REGISTRY",
    "TASK_PREFIX",
    "SelectionError",
    "SelectionProblem",
    "TaskNamespaceError",
    "check_selection",
    "require_valid_selection",
    "task_metric",
    "Bootstrap",
    "Direction",
    "Family",
    "MetricContext",
    "MetricDef",
    "MetricRegistry",
    "MetricResult",
    "NotApplicable",
    "Outcome",
    "Outputs",
    "Rollup",
    "TrialResults",
    "Value",
    "bootstrap_mean",
    "harbor_metric_payload",
    "load_job",
    "load_trial_dir",
    "build_measurement",
    "build_outputs",
    "build_reward",
    "build_validity",
    "edit_counts",
    "evaluate_metric",
    "metric",
    "paired_delta",
    "roll_up",
    "roll_up_all",
    "word_error_rate",
]
