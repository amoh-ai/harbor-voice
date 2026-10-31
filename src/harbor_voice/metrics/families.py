"""The metric families of

Each one declares what it needs and abstains outside it. Importing this module
registers all of them into :data:`harbor_voice.metrics.base.REGISTRY`.

Two families are registered here but will report ``not_applicable`` until the
perturbations that make them meaningful exist:

- selectivity needs L5. Registering it now is not premature —
  it is the perturbation–metric pairing working: *the perturbation is the probe that makes the
  measurement possible*, so the metric must be inert without it, and a test
  that proves it is inert is worth having before the layer lands.
- robustness needs a paired clean run, which is a job-level arrangement
  rather than something a single trajectory can supply.
"""

from __future__ import annotations

from harbor_voice.capabilities import Layer
from harbor_voice.metrics.base import Direction
from harbor_voice.metrics.base import Family
from harbor_voice.metrics.base import MetricContext
from harbor_voice.metrics.base import NotApplicable
from harbor_voice.metrics.base import Outcome
from harbor_voice.metrics.base import Value
from harbor_voice.metrics.base import metric
from harbor_voice.metrics.text import edit_counts
from harbor_voice.trajectory.model import BargeInOutcome
from harbor_voice.trajectory.model import EventType

# ------------------------------------------------------------------ divergence


@metric(
    name="divergence.wer_said_heard",
    sensitive_to=[Layer.SPEECH, Layer.ENVIRONMENT, Layer.CHANNEL],
    family=Family.DIVERGENCE,
    direction=Direction.LOWER_BETTER,
    requires=["cap:exposes_asr"],
    unit="rate",
)
def wer_said_heard(context: MetricContext) -> Outcome:
    """Word error rate between what was said and what was heard.

    Pooled over turns rather than averaged per turn: a per-turn mean weights a
    three-word turn the same as a forty-word one, so one misheard "yes"
    dominates a whole call.
    """
    pairs = [t for t in context.trajectory.turns if t.has_both_sides]
    if not pairs:
        return NotApplicable.because("no turn carried both said and heard")

    errors = reference = 0
    subs = dels = ins = 0
    for turn in pairs:
        counts = edit_counts(turn.said or "", turn.heard or "")
        errors += counts.errors
        reference += counts.reference_words
        subs += counts.substitutions
        dels += counts.deletions
        ins += counts.insertions

    if reference == 0:
        return NotApplicable.because("every reference turn was empty")
    return Value(
        value=errors / reference,
        unit="rate",
        n=len(pairs),
        detail={
            "substitutions": subs,
            "deletions": dels,
            "insertions": ins,
            "reference_words": reference,
        },
    )


# ---------------------------------------------------------------------- timing


@metric(
    name="timing.response_latency_s",
    sensitive_to=[Layer.CHANNEL, Layer.WORLD],
    family=Family.TIMING,
    direction=Direction.LOWER_BETTER,
    unit="s",
)
def response_latency(context: MetricContext) -> Outcome:
    """Median time from the caller's audio ending to the agent's beginning.

    Median, not mean: a single stalled turn drags a mean far enough
    to misrepresent the typical experience, and the stall is better reported
    as a failure rate than smeared across the average.

    Both endpoints name their clock domain — ``tx`` is when the
    caller's audio left, ``rx`` is when the agent's arrived. A latency whose
    endpoints come from different clocks without saying so is the single most
    common way voice numbers become incomparable between harnesses.
    """
    deltas = [
        turn.clocks.rx - turn.clocks.tx
        for turn in context.trajectory.turns
        if turn.clocks.rx is not None and turn.clocks.tx is not None
    ]
    if not deltas:
        return NotApplicable.because("no turn recorded both a tx and an rx clock")
    return Value(
        value=_median(deltas),
        unit="s",
        n=len(deltas),
        detail={
            "clock_pair": "tx_to_rx",
            "mean": sum(deltas) / len(deltas),
            "max": max(deltas),
        },
    )


@metric(
    name="timing.tts_first_byte_s",
    sensitive_to=[Layer.CHANNEL],
    family=Family.TIMING,
    direction=Direction.LOWER_BETTER,
    requires=["stage:tts_first_byte"],
    unit="s",
)
def tts_first_byte(context: MetricContext) -> Outcome:
    """Median time from turn start to the first synthesised byte.

    Gated on the stage being *emitted*, not on the pipeline being a cascade: a
    speech-to-speech party has no such seam, and asking it for one would
    produce a number equal to its total response time.
    """
    marks = _stage_offsets(context, "tts_first_byte")
    if not marks:
        return NotApplicable.because("no turn emitted tts_first_byte")
    return Value(value=_median(marks), unit="s", n=len(marks))


@metric(
    name="timing.asr_final_s",
    sensitive_to=[Layer.SPEECH, Layer.ENVIRONMENT, Layer.CHANNEL],
    family=Family.TIMING,
    direction=Direction.LOWER_BETTER,
    requires=["stage:asr_final"],
    unit="s",
)
def asr_final(context: MetricContext) -> Outcome:
    """Median time from turn start to a final transcript."""
    marks = _stage_offsets(context, "asr_final")
    if not marks:
        return NotApplicable.because("no turn emitted asr_final")
    return Value(value=_median(marks), unit="s", n=len(marks))


# ------------------------------------------------------------------ turn-taking


@metric(
    name="turns.overlap_s",
    sensitive_to=[Layer.CONVERSATION],
    family=Family.TURNS,
    direction=Direction.NEUTRAL,
    requires=["duplex:full"],
    unit="s",
)
def overlap_seconds(context: MetricContext) -> Outcome:
    """Total time more than one party was speaking.

    ``NEUTRAL`` on purpose. Overlap is rudeness in an intake call and
    attentiveness in a crisis line; a global direction would rank half the use
    cases backwards. That is true of turn-overlap ratio and it applies
    identically here.
    """
    events = context.trajectory.events_of(EventType.OVERLAP)
    if not events:
        return Value(value=0.0, unit="s", n=0)
    total = sum(e.duration_s or 0.0 for e in events)
    return Value(value=total, unit="s", n=len(events))


@metric(
    name="turns.overlap_ratio",
    sensitive_to=[Layer.CONVERSATION],
    family=Family.TURNS,
    direction=Direction.NEUTRAL,
    requires=["duplex:full"],
    unit="ratio",
)
def overlap_ratio(context: MetricContext) -> Outcome:
    """Overlapping time as a fraction of the call."""
    duration = context.trajectory.duration_s
    if duration <= 0:
        return NotApplicable.because("call has no measurable duration")
    total = sum(
        e.duration_s or 0.0 for e in context.trajectory.events_of(EventType.OVERLAP)
    )
    return Value(value=total / duration, unit="ratio", n=1)


@metric(
    name="turns.barge_in_success_rate",
    sensitive_to=[Layer.CONVERSATION],
    family=Family.TURNS,
    direction=Direction.HIGHER_BETTER,
    requires=["duplex:full", "cap:can_be_interrupted"],
    unit="rate",
)
def barge_in_success_rate(context: MetricContext) -> Outcome:
    """Fraction of interruptions the agent yielded to in time.

    Three outcomes, not two, matching :class:`~harbor_voice.floor.YieldRecord`:
    yielded in time, yielded late, and never yielded at all. Late and
    never are different failures — one agent listens slowly, the other does
    not listen — and collapsing them loses the distinction that says which to
    fix. ``yielded_in_time`` is ``None`` on the record when the holder never
    stopped, so an absent key here means *never*, not *unknown*.

    Separately, an interruption that never fired is not an agent
    failure. Events carrying a :class:`~harbor_voice.trajectory.model.BargeInOutcome`
    that did not fire are excluded from the denominator rather than counted
    as misses.
    """
    events = [
        e
        for e in context.trajectory.events_of(EventType.BARGE_IN)
        if e.meta.get("outcome") not in _BARGE_IN_DID_NOT_FIRE
    ]
    if not events:
        return NotApplicable.because("no barge-in fired in this call")

    in_time = late = never = 0
    for event in events:
        match event.meta.get("yielded_in_time"):
            case True:
                in_time += 1
            case False:
                late += 1
            case _:
                never += 1

    return Value(
        value=in_time / len(events),
        unit="rate",
        n=len(events),
        detail={"yielded_in_time": in_time, "yielded_late": late, "never": never},
    )


_BARGE_IN_DID_NOT_FIRE = frozenset(
    {BargeInOutcome.PENDING_DONE.value, BargeInOutcome.NO_ADAPTER.value}
)
"""Outcomes meaning the interruption never reached the agent."""


# ---------------------------------------------------------------------- speech


@metric(
    name="speech.holes_per_min",
    sensitive_to=[Layer.CHANNEL],
    family=Family.SPEECH,
    direction=Direction.LOWER_BETTER,
    unit="per_min",
)
def holes_per_min(context: MetricContext) -> Outcome:
    """Transport gaps per minute of call.

    A property of the *channel*, reported alongside agent metrics because a
    run with many holes is one whose other numbers deserve suspicion.
    """
    duration = context.trajectory.duration_s
    if duration <= 0:
        return NotApplicable.because("call has no measurable duration")
    gaps = context.trajectory.events_of(EventType.TRANSPORT_GAP)
    return Value(value=len(gaps) * 60.0 / duration, unit="per_min", n=len(gaps))


@metric(
    name="speech.rate_wpm",
    sensitive_to=[Layer.PERSONA, Layer.SPEECH],
    family=Family.SPEECH,
    direction=Direction.NEUTRAL,
    unit="wpm",
)
def speech_rate(context: MetricContext) -> Outcome:
    """Agent words per minute of agent audio.

    Needs both the words and the audio they were spoken over, so it abstains
    on a turn that recorded one without the other rather than dividing by a
    duration it had to guess.
    """
    words = 0
    seconds = 0.0
    counted = 0
    for turn in context.trajectory.turns:
        text = turn.said
        audio = turn.audio_transmitted
        if not text or audio is None or audio.duration_s <= 0:
            continue
        words += len(text.split())
        seconds += audio.duration_s
        counted += 1
    if seconds <= 0:
        return NotApplicable.because("no turn had both text and timed audio")
    return Value(value=words * 60.0 / seconds, unit="wpm", n=counted)


# ----------------------------------------------------------------- selectivity


def _selectivity(context: MetricContext, probe: str) -> Outcome:
    """Fraction of injected probes the agent correctly ignored.

    The probes arrive as timeline events carrying ``meta["probe"]`` and
    ``meta["ignored"]``, which is the contract
    :class:`~harbor_voice.duplex.DuplexDriver` fills.

    The same rule in its selectivity form: ``ignored`` is ``None`` for a probe
    that landed where no conclusion could be drawn — after the reply had
    ended, or with no agent audio following it. Those leave the denominator
    rather than counting as the agent being derailed, because a probe nobody
    could have reacted to is a fact about the call, not about the agent.
    """
    injected = [
        e
        for e in context.trajectory.events_of(EventType.SPEECH_START)
        if e.meta.get("probe") == probe
    ]
    if not injected:
        return NotApplicable.because(f"no {probe} probe was injected")
    events = [e for e in injected if isinstance(e.meta.get("ignored"), bool)]
    if not events:
        return NotApplicable.because(
            f"{len(injected)} {probe} probe(s) were injected but none landed "
            f"where the agent's response to them could be observed"
        )
    ignored = sum(1 for e in events if e.meta["ignored"] is True)
    return Value(
        value=ignored / len(events),
        unit="rate",
        n=len(events),
        detail={"injected": len(injected), "unmeasurable": len(injected) - len(events)},
    )


@metric(
    name="selectivity.backchannel",
    family=Family.SELECTIVITY,
    direction=Direction.HIGHER_BETTER,
    requires=["layer:L5.backchannel"],
    unit="rate",
)
def selectivity_backchannel(context: MetricContext) -> Outcome:
    """S_BC — the agent kept talking through an "mm-hm"."""
    return _selectivity(context, "backchannel")


@metric(
    name="selectivity.vocal_tic",
    family=Family.SELECTIVITY,
    direction=Direction.HIGHER_BETTER,
    requires=["layer:L5.vocal_tic"],
    unit="rate",
)
def selectivity_vocal_tic(context: MetricContext) -> Outcome:
    """S_VT — the agent was not derailed by a filled pause."""
    return _selectivity(context, "vocal_tic")


@metric(
    name="selectivity.non_directed",
    family=Family.SELECTIVITY,
    direction=Direction.HIGHER_BETTER,
    requires=["layer:L5.non_directed"],
    unit="rate",
)
def selectivity_non_directed(context: MetricContext) -> Outcome:
    """S_ND — the agent ignored speech that was not addressed to it."""
    return _selectivity(context, "non_directed")


# -------------------------------------------------------------------- validity


@metric(
    name="validity.simulator_in_persona",
    family=Family.VALIDITY,
    direction=Direction.HIGHER_BETTER,
    unit="score",
)
def simulator_in_persona(context: MetricContext) -> Outcome:
    """How well the simulated caller stayed in character.

    ``None`` until measured, never a default of 1.0 — an unmeasured fidelity
    that reads as perfect is how an invalid run passes its own validity gate.
    """
    score = context.trajectory.simulator_fidelity.get("in_persona")
    if score is None:
        return NotApplicable.because("simulator fidelity was not measured")
    return Value(value=float(score), unit="score", n=1)


@metric(
    name="validity.checks_passed",
    family=Family.VALIDITY,
    direction=Direction.HIGHER_BETTER,
    unit="rate",
)
def validity_checks_passed(context: MetricContext) -> Outcome:
    """Fraction of named run-validity checks that held."""
    checks = context.trajectory.validity.checks
    if not checks:
        return NotApplicable.because("no validity checks were recorded")
    passed = sum(1 for c in checks if c.ok)
    return Value(
        value=passed / len(checks),
        unit="rate",
        n=len(checks),
        detail={"failed": [c.name for c in context.trajectory.validity.failed]},
    )


# ------------------------------------------------------------------- utilities


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _stage_offsets(context: MetricContext, stage: str) -> list[float]:
    """Seconds from each turn's start to a named stage mark."""
    offsets: list[float] = []
    for turn in context.trajectory.turns:
        marks = [s for s in turn.stages if s.stage == stage]
        if not marks:
            continue
        start = min(s.t for s in turn.stages)
        offsets.append(min(m.t for m in marks) - start)
    return offsets


@metric(
    name="timing.yield_latency_s",
    sensitive_to=[Layer.CONVERSATION],
    family=Family.TIMING,
    direction=Direction.LOWER_BETTER,
    requires=["duplex:full", "cap:can_be_interrupted"],
    unit="s",
)
def yield_latency(context: MetricContext) -> Outcome:
    """Median time from an interruption to the agent falling silent.

    Computed only over interruptions the agent did yield to. Including
    the ones it ignored would need a stand-in duration for "never", and every
    available choice is wrong: the call length flatters a slow agent that
    eventually stopped, and a cap invents data. The never-yielded count is
    reported by ``turns.barge_in_success_rate``, which is where it belongs.
    """
    latencies = [
        float(e.meta["yield_latency_s"])
        for e in context.trajectory.events_of(EventType.BARGE_IN)
        if e.meta.get("yield_latency_s") is not None
    ]
    if not latencies:
        return NotApplicable.because("no interruption was yielded to")
    return Value(
        value=_median(latencies),
        unit="s",
        n=len(latencies),
        detail={"max": max(latencies)},
    )
