"""The tau-voice call driver — what a converted task's ``voice.toml`` names.

``[call] driver = "tau_voice.trial:drive"``. Core has already brought the
task's sidecar up and hands over its URL, a builder for the agent under test
(from the task's ``[agent]``), and the verifier. This runs tau2's trial: the
call, the grade, and tau2's hallucination retry.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harbor_voice.driver import DriverContext
from harbor_voice.driver import DriverError
from harbor_voice.driver import DriverOutcome
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.services import VerifierResult
from harbor_voice.transport.base import VoiceTransport

from tau_voice.run import HALLUCINATION_RETRIES
from tau_voice.run import run_trial

SERVICE = "tau2"
"""The sidecar's service name in the task's compose file."""

CACHE_ENV = "HARBOR_VOICE_TTS_CACHE"


def synthesis_cache() -> Path:
    """Where caller audio is cached across trials; a rerun spends no credits."""
    configured = os.environ.get(CACHE_ENV)
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "harbor-voice" / "tts"


async def drive(ctx: DriverContext) -> DriverOutcome:
    config = ctx.driver_config
    url = ctx.services.get(SERVICE)
    if url is None:
        raise DriverError(
            f"tau-voice needs the {SERVICE!r} service; the task did not declare it"
        )
    if ctx.verify is None:
        raise DriverError(
            "tau-voice grades every attempt; the task declares no [verifier]"
        )
    verify = ctx.verify
    logs = ctx.out_dir / "logs"

    async def grade(attempt_dir: Path) -> Mapping[str, Any]:
        verified = await verify()
        # The verifier writes into the shared logs mount; keep each attempt's.
        for source in (
            logs / "agent" / "tau_voice_state.json",
            logs / "verifier" / "result.json",
            logs / "verifier" / "reward.txt",
        ):
            if source.exists():
                shutil.copy2(source, attempt_dir / source.name)
        return {**verified.result, "reward": verified.reward}

    def agent(router: ToolRouter, system_prompt: str) -> VoiceTransport:
        return ctx.build_agent(tools=router, system_prompt=system_prompt)

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
        agent_factory=agent,
        synthesis_cache=synthesis_cache(),
    )
    final = (
        ctx.out_dir / "tau_voice" / f"attempt-{trial['final_attempt']}" / "result.json"
    )
    result = json.loads(final.read_text(encoding="utf-8")) if final.exists() else {}
    return DriverOutcome(
        verified=VerifierResult(reward=trial["reward"], result=result),
        summary=trial,
    )
