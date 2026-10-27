"""The original side of parity: tau2's own harness, run on Daytona.

harbor runs the harbor side, one environment per trial. The original side is
tau2's own ``tau2 run``, one long process per domain, which is not
harbor-shaped. So it runs in one Daytona sandbox built from the **same
Dockerfile** as the tasks' tau2 service (``adapters/tau-voice/sidecar``), so
both sides use the same tau2 commit, models, voices, seeds and reviewer, on
the same provider's network::

    tau-voice original --split test --trials 3 --out parity/original

Keys come from this machine's environment and are passed to the sandbox as
environment variables, as harbor passes a task's; they are never written to
a file. The sandbox is deleted afterwards unless ``--keep``.
"""

from __future__ import annotations

import argparse
import io
import re
import json
import os
import shlex
import tarfile
import time
from pathlib import Path
from typing import Any

from tau_voice.images import SIDECAR
from tau_voice.main import DOMAINS
from tau_voice.main import REVIEW_MODEL
from tau_voice.main import TAU2_RUN_SEED

ENV_NAMES = (
    "OPENAI_API_KEY",  # the agent under test, the caller's LLM
    "ELEVENLABS_API_KEY",  # the caller's voice
    "TAU2_VOICE_ID_MATT_DELANEY",
    "TAU2_VOICE_ID_LISA_BRENNER",
    "AWS_ACCESS_KEY_ID",  # the reviewer, on Bedrock
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
)
SIMULATIONS = "/opt/tau2-bench/data/simulations"


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
) -> str:
    """tau2's own run, in the mode the harbor side reproduces."""
    return shlex.join(
        [
            "tau2", "run",
            "--domain", domain,
            "--audio-native",
            "--audio-native-provider", "openai",
            "--audio-native-model", model,
            "--speech-complexity", "control",
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
        ]
    )  # fmt: skip


PROGRESS_SCRIPT = r"""
import json, os, statistics, sys
for s in json.loads(sys.argv[1]):
    splits = f"/opt/tau2-bench/data/tau2/domains/{s['domain']}/split_tasks.json"
    tasks = json.load(open(splits))[s["split"]]
    expected = len(s["task_ids"] or tasks) * s["trials"]
    try:
        index = json.load(open(os.path.join(s["dir"], "results.json"))).get("simulation_index") or []
    except (OSError, ValueError):
        index = []
    rewards = [e["reward"] for e in index if e.get("reward") is not None]
    discarded = os.path.join(s["dir"], "hallucination_discarded")
    print(json.dumps({
        "domain": s["domain"], "done": len(index), "expected": expected,
        "reward": statistics.fmean(rewards) if rewards else None,
        "ungraded": len(index) - len(rewards),
        "reruns": len(os.listdir(discarded)) if os.path.isdir(discarded) else 0,
        "agent_cost": sum(e.get("agent_cost") or 0 for e in index),
    }))
"""
"""Run in the sandbox: each domain's progress from tau2's own results index,
which tau2 rewrites as every simulation finishes."""


def _progress(sandbox: Any, specs: list[dict[str, Any]]) -> list[str]:
    command = (
        f"python -c {shlex.quote(PROGRESS_SCRIPT)} {shlex.quote(json.dumps(specs))}"
    )
    try:
        response = sandbox.process.exec(command, timeout=120)
    except Exception as exc:  # noqa: BLE001 - progress is best effort
        return [f"(progress unavailable: {exc})"]
    lines = []
    for raw in (response.result or "").splitlines():
        try:
            s = json.loads(raw)
        except ValueError:
            continue
        reward = "n/a" if s["reward"] is None else f"{s['reward']:.3f}"
        lines.append(
            f"{s['domain']:<8} {s['done']}/{s['expected']} sims · reward {reward}"
            f" · reruns {s['reruns']} · ungraded {s['ungraded']}"
            f" · agent cost ${s['agent_cost']:.2f}"
        )
    return lines


_UNPORTABLE = re.compile(r'[<>:"|?*]')


def unpack(data: bytes, out: Path) -> None:
    """Unpack tau2's results, renaming what Windows cannot hold.

    tau2 names artifact folders after task ids, and telecom's contain ``|``
    (``...roaming_enabled_off[PERSONA:Easy]``), which no Windows path may.
    """
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            member.name = _UNPORTABLE.sub("_", member.name)
        archive.extractall(out, members=members, filter="data")


def _env() -> dict[str, str]:
    missing = [name for name in ENV_NAMES if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"set these first: {', '.join(missing)}")
    env = {name: os.environ[name] for name in ENV_NAMES}
    env["AWS_REGION_NAME"] = env["AWS_REGION"]  # litellm's Bedrock region
    return env


def run(
    *,
    domains: list[str],
    model: str,
    split: str,
    trials: int,
    seed: int,
    concurrency: int,
    review_model: str,
    label: str,
    out: Path,
    keep: bool = False,
    poll_s: float = 60.0,
    task_ids: list[str] | None = None,
) -> None:
    from daytona import CreateSandboxFromImageParams
    from daytona import Daytona
    from daytona import Image
    from daytona import Resources
    from daytona import SessionExecuteRequest

    out.mkdir(parents=True, exist_ok=True)
    daytona = Daytona()
    print("building the tau2 image and creating the sandbox ...", flush=True)
    sandbox = daytona.create(
        CreateSandboxFromImageParams(
            image=Image.from_dockerfile(SIDECAR / "Dockerfile"),
            resources=Resources(cpu=4, memory=8, disk=10),
            env_vars=_env(),
            labels={"purpose": "tau-voice-parity-original", "label": label},
            auto_stop_interval=0,  # a run is hours of activity Daytona may not see
        ),
        timeout=1800,
        on_snapshot_create_logs=lambda line: None,
    )
    try:
        running: dict[str, str] = {}
        for domain in domains:
            session = f"tau2-{domain}"
            sandbox.process.create_session(session)
            command = tau2_command(
                domain,
                model=model,
                split=split,
                trials=trials,
                seed=seed,
                concurrency=concurrency,
                review_model=review_model,
                save_to=f"{label}-{domain}",
                task_ids=task_ids,
            )
            response = sandbox.process.execute_session_command(
                session,
                SessionExecuteRequest(
                    command=f"cd /opt/tau2-bench && {command}", run_async=True
                ),
            )
            running[domain] = str(response.cmd_id)
            print(f"started {domain}: {command}", flush=True)

        specs = [
            {
                "domain": domain,
                "dir": f"{SIMULATIONS}/{label}-{domain}",
                "split": split,
                "task_ids": task_ids,
                "trials": trials,
            }
            for domain in running
        ]
        started = time.monotonic()
        exit_codes: dict[str, int] = {}
        while len(exit_codes) < len(running):
            time.sleep(poll_s)
            for domain, cmd_id in running.items():
                if domain in exit_codes:
                    continue
                status = sandbox.process.get_session_command(f"tau2-{domain}", cmd_id)
                if status.exit_code is not None:
                    exit_codes[domain] = int(status.exit_code)
                    print(f"{domain} finished: exit {status.exit_code}", flush=True)
            minutes = (time.monotonic() - started) / 60
            print(f"--- {minutes:.0f} min", flush=True)
            for line in _progress(sandbox, specs):
                print(f"  {line}", flush=True)

        for domain, cmd_id in running.items():
            logs = sandbox.process.get_session_command_logs(f"tau2-{domain}", cmd_id)
            (out / f"{domain}.log").write_text(
                f"{logs.stdout or ''}\n{logs.stderr or ''}", encoding="utf-8"
            )
            name = f"{label}-{domain}"
            sandbox.process.exec(
                f"tar czf /tmp/{name}.tgz -C {SIMULATIONS} {name}", timeout=600
            )
            data = sandbox.fs.download_file(f"/tmp/{name}.tgz")
            if not data:
                print(f"{domain}: no results to download", flush=True)
                continue
            # Kept as downloaded first: unpacking can fail, the sandbox is gone.
            (out / f"{name}.tgz").write_bytes(data)
            unpack(data, out)
            print(f"{domain}: results in {out / name}", flush=True)
        failed = {d: c for d, c in exit_codes.items() if c != 0}
        if failed:
            raise SystemExit(
                f"tau2 run failed for {failed}; see the .log files in {out}"
            )
    finally:
        if keep:
            print(f"kept sandbox {sandbox.id}", flush=True)
        else:
            daytona.delete(sandbox)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="tau-voice original", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    parser.add_argument("--model", default="gpt-realtime-1.5")
    parser.add_argument("--split", default="test")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=TAU2_RUN_SEED)
    parser.add_argument("--concurrency", type=int, default=4, help="per domain")
    parser.add_argument("--review-model", default=REVIEW_MODEL)
    parser.add_argument("--label", default="parity")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", action="store_true", help="keep the sandbox")
    parser.add_argument(
        "--task-ids", nargs="+", help="only these tasks (one domain; smoke runs)"
    )
    args = parser.parse_args(argv)
    run(
        domains=args.domains,
        model=args.model,
        split=args.split,
        trials=args.trials,
        seed=args.seed,
        concurrency=args.concurrency,
        review_model=args.review_model,
        label=args.label,
        out=args.out,
        keep=args.keep,
        task_ids=args.task_ids,
    )
