"""The tau-voice call driver — what a converted task's ``voice.toml`` names.

``[call] driver = "tau_voice.trial:drive"``. It runs inside the task's
``main`` container (harbor-voice installs it there at trial start), next to
the ``tau2`` sidecar service on the environment's own network. It runs tau2's
trial: the call, a grade of every attempt, and tau2's hallucination retry.

Each attempt is graded by the sidecar's ``/grade`` — the same code as the
task's ``tests/test.sh`` — because ``main`` cannot exec into a sibling
container. The orchestrator's ``tests/test.sh`` run afterwards remains the
official grade.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harbor_voice.driver import DriverContext
from harbor_voice.driver import DriverOutcome
from harbor.models.trajectories.trajectory import Trajectory
from harbor_voice.driver import VerifierResult
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.trajectory.model import VoiceTrajectory
from harbor_voice.trial import AgentSpec
from harbor_voice.transport.base import VoiceTransport

from tau_voice.client import Sidecar
from tau_voice.main import HALLUCINATION_RETRIES
from tau_voice.run import run_trial

SIDECAR_URL = "http://tau2:8000"
"""The sidecar by compose service name, inside the environment's network."""

CACHE_ENV = "HARBOR_VOICE_TTS_CACHE"


LIVE = "openai.live"


def live_config(agent: AgentSpec | None) -> dict[str, Any] | None:
    """tau2's ``--live-config`` for an OpenAI Live agent, from its settings.

    The backend model is the agent's (``[agent.config.s2s] backend_model``);
    tau2 renders both prompts from it and the system prompt.
    """
    if agent is None or not agent.s2s or agent.s2s.partition("/")[0] != LIVE:
        return None
    settings = agent.config.get("s2s", {})
    if not settings.get("backend_model"):
        raise ValueError(f"{LIVE} needs backend_model in the agent's s2s config")
    config = {"backend_model": settings["backend_model"]}
    if settings.get("voice"):
        config["voice"] = settings["voice"]
    return config


def synthesis_cache() -> Path:
    """Where caller audio is cached across trials; a rerun spends no credits."""
    configured = os.environ.get(CACHE_ENV)
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "harbor-voice" / "tts"


async def drive(ctx: DriverContext) -> DriverOutcome:
    config = ctx.driver_config
    url = str(config.get("sidecar_url", SIDECAR_URL))

    async def grade(attempt_dir: Path) -> Mapping[str, Any]:
        sidecar = Sidecar(url, timeout_s=600.0)
        try:
            result = await sidecar.grade()
        finally:
            await sidecar.aclose()
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    def agent(
        router: ToolRouter, system_prompt: str, **overrides: Any
    ) -> VoiceTransport:
        return ctx.build_agent(tools=router, system_prompt=system_prompt, **overrides)

    live = live_config(ctx.agent)

    trial = await run_trial(
        grade=grade,
        out_dir=ctx.out_dir / "tau_voice",
        seed=int(config.get("seed", ctx.config.seed)),
        hallucination_retries=int(
            config.get("hallucination_retries", HALLUCINATION_RETRIES)
        ),
        sidecar_url=url,
        domain=str(config["domain"]),
        task_id=str(config["task_id"]),
        speech_complexity=str(config.get("speech_complexity", "control")),
        # tau2's --user-llm; absent, tau2's default caller model.
        user_llm=config.get("user_llm"),
        agent_factory=agent,
        live_config=live,
        # Live's frontend model, for tau2's own record of the session.
        agent_model=(ctx.agent.s2s or "").partition("/")[2] or None
        if live is not None and ctx.agent is not None
        else None,
        synthesis_cache=synthesis_cache(),
        on_progress=ctx.progress,
    )
    final = ctx.out_dir / "tau_voice" / f"attempt-{trial['final_attempt']}"
    graded = final / "result.json"
    result = json.loads(graded.read_text(encoding="utf-8")) if graded.exists() else {}
    atif = final / "trajectory.json"
    return DriverOutcome(
        verified=VerifierResult(reward=trial["reward"], result=result),
        summary=trial,
        # The attempt that counts; every attempt keeps its own beside it.
        trajectory=(
            VoiceTrajectory.from_atif(
                Trajectory.model_validate_json(atif.read_text(encoding="utf-8"))
            )
            if atif.exists()
            else None
        ),
    )
