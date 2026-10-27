"""The voice commands.

**Extend harbor's CLI; do not invent a parallel one.** This is a Typer
sub-app, so harbor's ``main.py`` mounts it with one line::

    from harbor_voice.cli import voice_app
    app.add_typer(voice_app, name="voice")

and everything harbor already does — telemetry, auth, job management — keeps
working. Until that line lands upstream it also runs standalone via
``python -m harbor_voice.cli``, which is how it is tested.

The commands here are the ones that need no job to exist yet: inspect a
configuration, list what is available, render something you already have.
``run`` is deliberately absent — it belongs to harbor's job machinery, not to
a parallel implementation of it.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Annotated

import typer

voice_app = typer.Typer(
    name="voice",
    help="Voice evaluation: preflight, profiles, capabilities, conformance.",
    no_args_is_help=True,
)


def _registry():  # noqa: ANN202 - typer-facing helper
    """A provider registry with the AWS providers if their SDK is present.

    Registration is attempted and failure is *reported*, not raised: ``caps``
    and ``profiles`` are useful on a laptop with no cloud SDK, and a CLI that
    refuses to start because an optional extra is missing is a CLI nobody
    runs during authoring.
    """
    from harbor_voice.providers.registry import default_registry

    registry = default_registry()
    try:
        from harbor_voice.providers.aws import register_all

        register_all(registry)
    except Exception as exc:  # noqa: BLE001 - reported below, never fatal
        typer.secho(f"note: AWS providers unavailable ({exc})", fg="yellow", err=True)
    for vendor in ("elevenlabs", "openai"):
        try:
            module = importlib.import_module(f"harbor_voice.providers.{vendor}")
            module.register_all(registry)
        except Exception as exc:  # noqa: BLE001 - reported below, never fatal
            typer.secho(
                f"note: {vendor} providers unavailable ({exc})", fg="yellow", err=True
            )
    return registry


# ----------------------------------------------------------------- doctor


@voice_app.command()
def run(
    task_dir: Annotated[Path, typer.Argument(help="Task directory with voice.toml.")],
    out: Annotated[Path, typer.Option(help="Where this trial's files go.")],
    trial_id: Annotated[
        str | None, typer.Option(help="Trial id; also names its services.")
    ] = None,
) -> None:
    """Run one trial of a task directory and print its reward."""
    import asyncio

    from harbor_voice.trial import VoiceTaskConfig
    from harbor_voice.trial import VoiceTrial

    config = VoiceTaskConfig.from_toml(task_dir / "voice.toml")
    if trial_id is not None:
        config = config.model_copy(update={"trial_id": trial_id})
    trial = VoiceTrial(config, registry=_registry(), out_dir=out, task_dir=task_dir)
    outcome = asyncio.run(trial.run())
    typer.echo(json.dumps({"reward": outcome.reward, "out": str(out)}))


@voice_app.command()
def doctor(
    profile: Annotated[
        str | None, typer.Option(help="Named condition to preflight.")
    ] = None,
    agent: Annotated[str | None, typer.Option(help="Agent/provider name.")] = None,
    stt: Annotated[str | None, typer.Option(help="STT provider spec.")] = None,
    llm: Annotated[str | None, typer.Option(help="LLM provider spec.")] = None,
    tts: Annotated[str | None, typer.Option(help="TTS provider spec.")] = None,
    metric: Annotated[
        list[str] | None, typer.Option(help="Metric to report; repeatable.")
    ] = None,
    trajectory: Annotated[
        Path | None,
        typer.Option(help="Take the capability matrix from a recorded run."),
    ] = None,
) -> None:
    """Preflight a configuration: what will report, abstain, or block.

    The most valuable command and the cheapest to build, because every
    piece it needs already exists: capabilities readable without credentials,
    applicability with reasons, profile-to-metric pairing, and credential
    status that names variables without printing them.
    """
    from harbor_voice.doctor import diagnose

    capabilities = None
    if trajectory is not None:
        from harbor_voice.verify import load_trajectory

        traj = load_trajectory(trajectory)
        for party in traj.parties:
            if party.capabilities is not None:
                capabilities = party.capabilities
                break

    providers = {
        slot: spec
        for slot, spec in (("stt", stt), ("llm", llm), ("tts", tts), ("agent", agent))
        if spec
    }
    report = diagnose(
        capabilities=capabilities,
        agent=agent,
        profile=profile,
        metrics=metric or None,
        providers=providers or None,
        provider_registry=_registry(),
    )
    typer.echo(report.render())
    raise typer.Exit(0 if report.ok else 1)


# --------------------------------------------------------------- profiles


@voice_app.command()
def profiles(
    buildable: Annotated[
        bool, typer.Option(help="Only conditions that can run here today.")
    ] = False,
) -> None:
    """List named conditions, their layers, and which metrics read them."""
    from harbor_voice.perturbation.profiles import PROFILES
    from harbor_voice.perturbation.profiles import summarise

    for name in sorted(PROFILES):
        profile = PROFILES[name]
        if buildable and not profile.is_buildable:
            continue
        state = "ready" if profile.is_buildable else "not yet"
        typer.echo(f"{name:<20} [{state:<7}] {profile.reproduces}")
        if profile.note:
            typer.echo(f"{'':<20}  note: {profile.note}")
        for layer, readers in summarise(profile).paired_metrics.items():
            shown = ", ".join(readers) if readers else "NO METRIC READS THIS"
            typer.echo(f"{'':<20}  {layer:<22} {shown}")


# ------------------------------------------------------------------- caps


@voice_app.command()
def caps(
    provider: Annotated[
        str | None, typer.Option(help="Provider name; omit to list all.")
    ] = None,
) -> None:
    """Print capability matrices and their attestation status."""
    registry = _registry()
    entries = (
        [registry.lookup(provider)]
        if provider
        else sorted(registry, key=lambda e: e.spec.name)
    )
    for entry in entries:
        spec = entry.spec
        state = "implemented" if entry.is_implemented else "declared only"
        typer.echo(f"{spec.name:<22} {spec.modality.value:<4} [{state}]")
        claims = spec.capabilities
        for field_name in ("streams", "emits_partials", "word_timings", "diarizes"):
            value = getattr(claims, field_name)
            if not value:
                continue
            from harbor_voice.providers.capabilities import ProviderCapability

            attested = claims.is_attested(ProviderCapability(field_name))
            mark = "attested" if attested else "UNATTESTED"
            typer.echo(f"{'':<22}   {field_name:<16} {mark}")
        if spec.credential_provider:
            status = registry.credentials(f"{spec.name}/{spec.default_model or '_'}")
            typer.echo(f"{'':<22}   credentials      {status.explain()}")


# -------------------------------------------------------------- conformance


@voice_app.command()
def conformance(
    provider: Annotated[str, typer.Argument(help="Provider to verify.")],
) -> None:
    """Run the portable conformance suite.

    This is what moves a claim from ``UNATTESTED`` to ``attested``. Until it
    runs, every number resting on that claim is usable for debugging and not
    for a published result.
    """
    typer.secho(
        f"conformance for {provider!r} needs a live session and is not wired "
        f"to the CLI yet; the suite itself is in harbor_voice.conformance",
        fg="yellow",
    )
    raise typer.Exit(2)


# ---------------------------------------------------------------- metrics


@voice_app.command()
def metrics() -> None:
    """List registered metrics, what they need, and which layers move them."""
    from harbor_voice.metrics import REGISTRY

    for name in REGISTRY.names():
        defn = REGISTRY.get(name)
        needs = ", ".join(defn.requires) or "-"
        moved = ", ".join(layer.value for layer in defn.sensitive_to) or "-"
        owner = f" [{defn.benchmark}]" if defn.benchmark else ""
        typer.echo(f"{name:<34}{defn.direction.value:<15}{owner}")
        typer.echo(f"{'':<34}requires: {needs}")
        typer.echo(f"{'':<34}moved by: {moved}")


# ----------------------------------------------------------------- report


@voice_app.command()
def report(
    job: Annotated[Path, typer.Argument(help="Directory of scored trials.")],
    baseline: Annotated[
        str, typer.Option(help="Profile every other one is compared against.")
    ] = "clean",
) -> None:
    """Roll a job up, and diff each profile against the baseline.

    Reads ``measurement.json`` per trial, never ``reward.json``: the reward
    file is harbor's flat surface and has already lost the verdicts, the
    sample sizes and the reasons.
    """
    from harbor_voice.metrics.rollup import load_job
    from harbor_voice.metrics.rollup import robustness

    trials = load_job(job)
    if not trials:
        typer.secho(f"no scored trials under {job}", fg="yellow", err=True)
        raise typer.Exit(2)

    valid = sum(1 for t in trials if t.measurement_valid)
    typer.echo(f"{len(trials)} trials, {valid} valid, {len(trials) - valid} excluded")

    from harbor_voice.metrics.rollup import roll_up_all

    for name, rolled in sorted(roll_up_all(trials).items()):
        if rolled.mean is None:
            continue
        spread = ""
        if rolled.bootstrap is not None:
            spread = f"  +/-{rolled.bootstrap.std:.3f} over a single draw"
        typer.echo(
            f"  {name:<34}{rolled.mean:>8.3f}  median={rolled.median:.3f}  "
            f"n={rolled.n_reported}{spread}"
        )

    result = robustness(trials, baseline=baseline)
    typer.echo("")
    typer.echo(f"robustness vs {baseline!r}")
    if result.is_empty:
        typer.secho("  nothing paired", fg="yellow")
    for profile in result.profiles:
        typer.echo(f"  {profile}  ({result.pairs.get(profile, 0)} pairs)")
        for name, delta in sorted(result.deltas[profile].items()):
            if delta.mean is not None:
                typer.echo(f"    {name:<32}{delta.mean:>+8.3f}")

    # Loud, because a table built from four pairs out of fifty trials looks
    # exactly like one built from fifty.
    for entry in result.unpaired:
        typer.secho(
            f"  UNPAIRED {entry.trial_id or '?'} ({entry.profile}): {entry.reason}",
            fg="yellow",
        )


# ----------------------------------------------------------------- listen


@voice_app.command()
def listen(
    trajectory: Annotated[Path, typer.Argument(help="Recorded ATIF-voice run.")],
    out: Annotated[Path, typer.Option(help="Where to write the WAV.")] = Path(
        "timeline.wav"
    ),
) -> None:
    """Render a recorded call's mixed timeline to a WAV you can play.

    The render **mixes**, it does not concatenate. If two parties
    talked at once the file says so — a recording that quietly reorders an
    interruption into an orderly exchange is worse than no recording, because
    someone will use it to settle an argument.
    """
    from harbor_voice.verify import load_trajectory

    traj = load_trajectory(trajectory)
    mixed = traj.artifacts.mixed_timeline
    if mixed is None:
        typer.secho(
            "this trajectory stores no mixed timeline (tier "
            f"{traj.artifacts.tier}); re-run with --artifacts tier1 or higher",
            fg="yellow",
            err=True,
        )
        raise typer.Exit(2)
    typer.echo(f"{mixed.duration_s:.2f}s, {mixed.bytes} bytes, digest {mixed.digest}")
    typer.secho(
        f"audio is stored by reference; fetch {mixed.digest} from the "
        f"artifact store to write {out}",
        fg="yellow",
    )


# ----------------------------------------------------------------- inspect


@voice_app.command("show")
def show(
    trajectory: Annotated[Path, typer.Argument(help="Recorded ATIF-voice run.")],
) -> None:
    """Summarise a recorded run: matrix, turns, validity, comparison keys."""
    from harbor_voice.verify import load_trajectory

    traj = load_trajectory(trajectory)
    typer.echo(f"trial      : {traj.trial_id}  task={traj.task}  agent={traj.agent}")
    typer.echo(f"schema     : {traj.schema_version}")
    if traj.config_matrix is not None:
        typer.echo(f"matrix     : {traj.config_matrix.model_dump(mode='json')}")
    typer.echo(f"turns      : {len(traj.turns)}  timeline={len(traj.timeline)} events")
    typer.echo(f"degraded   : {traj.is_degraded}")
    typer.echo(f"valid      : {traj.validity.measurement_valid}")
    for check in traj.validity.checks:
        mark = "ok " if check.ok else "FAIL"
        typer.echo(f"  {mark} {check.name}  {check.detail or ''}")
    typer.echo(f"comparable : {json.dumps(traj.comparable_on, sort_keys=True)}")


def main() -> None:  # pragma: no cover - entry point
    voice_app()


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
