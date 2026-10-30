"""Unit tests for the capability model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Applicability
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import CapabilityError
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Layer
from harbor_voice.capabilities import LayerKey
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import Precondition
from harbor_voice.capabilities import Requirement
from harbor_voice.capabilities import TurnControl
from harbor_voice.capabilities import Verdict
from harbor_voice.capabilities import check_layers
from harbor_voice.capabilities import check_requirements
from harbor_voice.capabilities import evaluate

pytestmark = pytest.mark.unit


def s2s(**over: object) -> Capabilities:
    """A plausible speech-to-speech party, e.g. OpenAI Realtime."""
    base: dict[str, object] = {
        "duplex": frozenset({Duplex.HALF, Duplex.FULL}),
        "pipeline": Pipeline.S2S,
        "turn_control": TurnControl.NATIVE_VAD,
        "exposes_stages": frozenset({Stage.RESPONSE_START, Stage.RESPONSE_DONE}),
        "can_be_interrupted": True,
    }
    return Capabilities(**(base | over))  # type: ignore[arg-type]


def cascade(**over: object) -> Capabilities:
    """A plausible cascade party — harbor-voice owns the seams, so all of them."""
    base: dict[str, object] = {
        "duplex": frozenset({Duplex.HALF, Duplex.FULL}),
        "pipeline": Pipeline.CASCADE,
        "turn_control": TurnControl.HARNESS_VAD,
        "exposes_asr": True,
        "exposes_partials": True,
        "exposes_stages": frozenset(Stage),
        "can_interrupt": True,
        "can_be_interrupted": True,
    }
    return Capabilities(**(base | over))  # type: ignore[arg-type]


# ------------------------------------------------------- internal consistency


class TestCapabilityConsistency:
    def test_capability_error_is_a_value_error(self) -> None:
        """Raised inside the validator; pydantic re-raises it wrapped."""
        assert issubclass(CapabilityError, ValueError)

    def test_plausible_parties_construct(self) -> None:
        assert s2s().pipeline is Pipeline.S2S
        assert cascade().exposes_asr

    def test_s2s_cannot_claim_a_seam_it_never_composed(self) -> None:
        """A stub claiming a seam the gate then trusts.

        The seams an S2S party cannot have are the *composed* ones — there is
        no LLM-first-token inside a model that never separated hearing from
        reasoning.
        """
        with pytest.raises(ValidationError, match="cannot expose"):
            s2s(
                exposes_stages=frozenset(
                    {Stage.RESPONSE_START, Stage.RESPONSE_DONE, Stage.LLM_FIRST_TOKEN}
                )
            )

    def test_s2s_may_claim_a_transcript_it_actually_emits(self) -> None:
        """Nova Sonic emits ``textOutput`` with ``role: USER`` — what it heard.

        Emitting a transcript is not the same as exposing a swappable
        recogniser: said-vs-heard works here, the reference-cascade ablation still does
        not. The capability model now tells those apart.
        """
        caps = s2s(
            exposes_asr=True,
            exposes_stages=frozenset(
                {Stage.RESPONSE_START, Stage.RESPONSE_DONE, Stage.ASR_FINAL}
            ),
        )
        assert caps.exposes_asr
        assert not caps.exposes_partials

    def test_s2s_may_claim_vad_stages(self) -> None:
        """Optional for S2S: only when the provider reports its own endpointing."""
        caps = s2s(
            exposes_stages=frozenset(
                {Stage.RESPONSE_START, Stage.RESPONSE_DONE, Stage.VAD_SPEECH_START}
            )
        )
        assert Stage.VAD_SPEECH_START in caps.exposes_stages

    def test_partials_require_asr(self) -> None:
        with pytest.raises(ValidationError, match="requires exposes_asr"):
            cascade(exposes_asr=False, exposes_partials=True)

    def test_partials_require_the_partial_stage(self) -> None:
        with pytest.raises(ValidationError, match="ASR_PARTIAL"):
            cascade(exposes_stages=frozenset(Stage) - {Stage.ASR_PARTIAL})

    def test_asr_requires_the_final_stage(self) -> None:
        with pytest.raises(ValidationError, match="ASR_FINAL"):
            cascade(
                exposes_partials=False,
                exposes_stages=frozenset(Stage) - {Stage.ASR_FINAL, Stage.ASR_PARTIAL},
            )

    def test_at_least_one_duplex_mode_required(self) -> None:
        with pytest.raises(ValidationError, match="at least one duplex"):
            s2s(duplex=frozenset())

    def test_unknown_attestation_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown capabilities"):
            s2s(attested=frozenset({"teleportation"}))

    def test_capabilities_are_frozen(self) -> None:
        caps = s2s()
        with pytest.raises(ValidationError):
            caps.exposes_asr = True  # type: ignore[misc]

    def test_extra_fields_are_rejected(self) -> None:
        """A typo'd capability key must not load silently (UNREAD discipline)."""
        with pytest.raises(ValidationError, match="can_barge_in"):
            s2s(can_barge_in=True)


# ------------------------------------------------------------------ layer key


class TestLayerKey:
    @pytest.mark.parametrize(
        ("raw", "layer", "name"),
        [
            ("L5.interruption", Layer.CONVERSATION, "interruption"),
            ("conversation.interruption", Layer.CONVERSATION, "interruption"),
            ("L3.background_bed", Layer.ENVIRONMENT, "background_bed"),
            ("ENVIRONMENT.background_bed", Layer.ENVIRONMENT, "background_bed"),
        ],
    )
    def test_parses_both_spellings(self, raw: str, layer: Layer, name: str) -> None:
        key = LayerKey.parse(raw)
        assert key.layer is layer
        assert key.name == name

    def test_round_trips_to_canonical_form(self) -> None:
        assert str(LayerKey.parse("conversation.interruption")) == "L5.interruption"

    def test_missing_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be"):
            LayerKey.parse("L5")

    def test_unknown_layer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown layer"):
            LayerKey.parse("L9.teleport")


# ----------------------------------------------------- rule 1: load-time check


class TestRequirements:
    def test_satisfied_requirement_yields_no_problems(self) -> None:
        req = Requirement(duplex=Duplex.FULL, can_be_interrupted=True)
        assert check_requirements(req, s2s()) == []

    def test_unsupported_duplex_is_a_violation(self) -> None:
        caps = s2s(duplex=frozenset({Duplex.HALF}))
        problems = check_requirements(Requirement(duplex=Duplex.FULL), caps)
        assert len(problems) == 1
        assert "requires duplex=full" in problems[0]

    def test_pipeline_any_matches_anything(self) -> None:
        assert check_requirements(Requirement(pipeline="any"), s2s()) == []

    def test_wrong_pipeline_is_a_violation(self) -> None:
        problems = check_requirements(Requirement(pipeline=Pipeline.CASCADE), s2s())
        assert "requires pipeline=cascade" in problems[0]

    def test_all_violations_are_reported_not_just_the_first(self) -> None:
        """`doctor` prints the whole picture in one pass, not one error per run."""
        caps = s2s(duplex=frozenset({Duplex.HALF}))
        req = Requirement(
            duplex=Duplex.FULL, pipeline=Pipeline.CASCADE, exposes_asr=True
        )
        assert len(check_requirements(req, caps)) == 3

    def test_session_length_ceiling(self) -> None:
        caps = s2s(max_session_s=120.0)
        problems = check_requirements(Requirement(min_session_s=600.0), caps)
        assert "caps at 120.0s" in problems[0]

    def test_activating_an_undeclared_layer_is_a_violation(self) -> None:
        """Rule 1: a key that loads and changes nothing must be refused."""
        caps = cascade(layers=frozenset({Layer.ENVIRONMENT}))
        problems = check_layers(frozenset({LayerKey.parse("L4.jitter")}), caps)
        assert len(problems) == 1
        assert "activates L4.jitter" in problems[0]

    def test_declared_layer_passes(self) -> None:
        caps = cascade(layers=frozenset({Layer.ENVIRONMENT, Layer.CHANNEL}))
        assert check_layers(frozenset({LayerKey.parse("L4.jitter")}), caps) == []

    def test_exclusive_group_breach_is_caught(self) -> None:
        """EVA's accent / behaviour / language share one provider voice slot."""
        caps = cascade(
            layers=frozenset({Layer.PERSONA}),
            exclusive_groups=(frozenset({"accent", "behaviour", "language"}),),
        )
        problems = check_layers(
            frozenset({LayerKey.parse("L1.accent"), LayerKey.parse("L1.behaviour")}),
            caps,
        )
        assert len(problems) == 1
        assert "mutually exclusive" in problems[0]

    def test_one_member_of_an_exclusive_group_is_fine(self) -> None:
        caps = cascade(
            layers=frozenset({Layer.PERSONA}),
            exclusive_groups=(frozenset({"accent", "behaviour", "language"}),),
        )
        assert check_layers(frozenset({LayerKey.parse("L1.accent")}), caps) == []


# -------------------------------------------- rules 2 and 3: metric reporting


class TestApplicability:
    def test_all_conditions_met_is_satisfied(self) -> None:
        caps = cascade(attested=frozenset({"can_be_interrupted"}))
        got = evaluate(("duplex:full", "cap:can_be_interrupted"), caps)
        assert got.verdict is Verdict.SATISFIED
        assert got.may_report and got.may_publish

    def test_unmet_condition_is_not_applicable_never_zero(self) -> None:
        """Rule 2: a half-duplex run has no barge-in measurement, not a zero."""
        caps = s2s(duplex=frozenset({Duplex.HALF}))
        got = evaluate(("duplex:full",), caps)
        assert got.verdict is Verdict.NOT_APPLICABLE
        assert not got.may_report
        assert "needs duplex=full" in got.reasons[0]

    def test_every_unmet_reason_is_reported(self) -> None:
        caps = s2s(duplex=frozenset({Duplex.HALF}))
        got = evaluate(("duplex:full", "cap:exposes_asr", "stage:asr_final"), caps)
        assert len(got.reasons) == 3

    def test_floor_overrides_party_capability(self) -> None:
        """A full-duplex-capable party under a half-duplex floor is still N/A."""
        caps = cascade(attested=frozenset({"can_be_interrupted"}))
        got = evaluate(("duplex:full",), caps, floor=Duplex.HALF)
        assert got.verdict is Verdict.NOT_APPLICABLE
        assert "floor in force" in got.reasons[0]

    def test_inactive_layer_makes_a_selectivity_metric_not_applicable(self) -> None:
        """The perturbation is the probe that makes the metric possible."""
        caps = cascade()
        got = evaluate(("layer:L5.backchannel",), caps, active_layers=frozenset())
        assert got.verdict is Verdict.NOT_APPLICABLE
        assert "needs layer L5.backchannel active" in got.reasons[0]

    def test_active_layer_satisfies(self) -> None:
        got = evaluate(
            ("layer:L5.backchannel",),
            cascade(),
            active_layers=frozenset({LayerKey.parse("L5.backchannel")}),
        )
        assert got.verdict is Verdict.SATISFIED

    def test_missing_stage_is_not_applicable(self) -> None:
        got = evaluate(("stage:llm_first_token",), s2s())
        assert got.verdict is Verdict.NOT_APPLICABLE

    def test_unattested_claim_may_report_but_not_publish(self) -> None:
        """Rule 3: usable, but cannot gate a published number."""
        caps = cascade(attested=frozenset())
        got = evaluate(("cap:can_be_interrupted",), caps)
        assert got.verdict is Verdict.UNATTESTED
        assert got.may_report
        assert not got.may_publish
        assert "not verified by conformance" in got.reasons[0]

    def test_unmet_beats_unattested(self) -> None:
        """A missing capability is reported as missing, not as unverified."""
        caps = s2s(duplex=frozenset({Duplex.HALF}), attested=frozenset())
        got = evaluate(("duplex:full", "cap:can_be_interrupted"), caps)
        assert got.verdict is Verdict.NOT_APPLICABLE

    def test_non_attestable_capability_does_not_need_attestation(self) -> None:
        got = evaluate(("pipeline:cascade",), cascade())
        assert got.verdict is Verdict.SATISFIED

    def test_no_preconditions_is_satisfied(self) -> None:
        assert evaluate((), s2s()).verdict is Verdict.SATISFIED

    def test_accepts_parsed_preconditions_too(self) -> None:
        got = evaluate((Precondition.parse("pipeline:s2s"),), s2s())
        assert got.verdict is Verdict.SATISFIED


class TestPrecondition:
    @pytest.mark.parametrize(
        "raw",
        [
            "duplex:full",
            "pipeline:cascade",
            "cap:exposes_asr",
            "layer:L5.x",
            "stage:asr_final",
        ],
    )
    def test_round_trip(self, raw: str) -> None:
        assert str(Precondition.parse(raw)) == raw

    @pytest.mark.parametrize("bad", ["duplex", "nonsense:x", "", ":full"])
    def test_malformed_is_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="must be"):
            Precondition.parse(bad)


class TestApplicabilityShape:
    def test_satisfied_has_no_reasons(self) -> None:
        assert Applicability(Verdict.SATISFIED).reasons == ()

    def test_not_applicable_blocks_reporting_entirely(self) -> None:
        a = Applicability(Verdict.NOT_APPLICABLE, ("because",))
        assert not a.may_report and not a.may_publish
