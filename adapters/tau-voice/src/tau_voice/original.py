"""The original side of parity: tau2's own harness, its command.

harbor runs the harbor side, one environment per trial. The original side is
tau2's own ``tau2 run``, one long process per domain and speech complexity,
which is not harbor-shaped. It runs in the tau2 image the tasks' ``tau2``
service is built from (``adapters/tau-voice/sidecar``), so both sides use the
same tau2 commit, models, voices, seeds and reviewer::

    cd /opt/tau2-bench && tau2 run --domain airline --audio-native \\
        --audio-native-provider openai --audio-native-model gpt-realtime-1.5 \\
        --speech-complexity control --task-split-name test --num-trials 3 \\
        --seed 300 --max-concurrency 4 --hallucination-retries 3 \\
        --review-model bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0 \\
        --auto-resume --save-to parity-control-airline --task-ids 2 18 ...

The image pins ``PYTHONHASHSEED=0``, as each harbor trial's sidecar does, so
tau2's effect seed (``hash(task.id)``) is the same on both sides. Keys come
from the environment the command runs in: ``OPENAI_API_KEY``,
``ELEVENLABS_API_KEY``, the seven ``TAU2_VOICE_ID_*`` and the reviewer's
``AWS_*`` (with ``AWS_REGION_NAME`` set to ``AWS_REGION``). tau2 saves its
results under ``data/simulations/<save_to>``.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from typing import Any


def tau2_command(
    domain: str,
    *,
    model: str,
    split: str,
    trials: int,
    seed: int,
    concurrency: int,
    review_model: str,
    save_to: str,
    hallucination_retries: int = 3,
    task_ids: list[str] | None = None,
    complexity: str = "control",
    user_llm: str | None = None,
    live_config: Mapping[str, Any] | None = None,
) -> str:
    """tau2's own run, in the mode the harbor side reproduces.

    ``live_config`` runs OpenAI Live (``openai_live``) with tau2's real-time
    user generation off, as harbor-voice's caller runs.
    """
    provider = (
        [
            "--audio-native-provider",
            "openai_live",
            "--live-config",
            json.dumps(dict(live_config), sort_keys=True),
            "--no-realtime-generation",
        ]
        if live_config is not None
        else ["--audio-native-provider", "openai"]
    )
    return shlex.join(
        [
            "tau2", "run",
            "--domain", domain,
            "--audio-native",
            *provider,
            "--audio-native-model", model,
            "--speech-complexity", complexity,
            "--task-split-name", split,
            "--num-trials", str(trials),
            "--seed", str(seed),
            "--max-concurrency", str(concurrency),
            "--hallucination-retries", str(hallucination_retries),
            "--review-model", review_model,
            "--auto-resume",
            "--save-to", save_to,
            # Explicit ids, never --num-tasks: tau2 orders tasks by its task
            # file, the converter by the split file, so "the first N" differ.
            *(["--task-ids", *task_ids] if task_ids else []),
            *(["--user-llm", user_llm] if user_llm else []),
        ]
    )  # fmt: skip
