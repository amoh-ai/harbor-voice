"""Nova Sonic — offline tests over the protocol as it actually behaves.

No network. Every fact asserted here was observed against the live service
during bring-up and is pinned so a refactor cannot quietly undo it.
"""

from __future__ import annotations

import json

import pytest

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.stages import CASCADE_ONLY_STAGES
from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.ids import PartyId
from harbor_voice.providers.aws import nova_sonic as ns
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.capabilities import Modality

pytestmark = pytest.mark.unit

AGENT = PartyId("agent")


def transport(**over: object) -> ns.NovaSonicTransport:
    return ns.NovaSonicTransport(party=AGENT, region="us-east-1", **over)


class TestDeclaration:
    def test_it_is_a_transport_not_a_slot(self) -> None:
        assert ns.SPEC.modality is Modality.S2S
        assert not ns.SPEC.modality.is_slot

    def test_it_declares_the_transcript_it_actually_emits(self) -> None:
        caps = transport().capabilities
        assert caps.pipeline is Pipeline.S2S
        assert caps.exposes_asr
        assert Stage.ASR_FINAL in caps.exposes_stages

    def test_it_does_not_claim_partials(self) -> None:
        """Nothing verified says it streams them; claiming them would hand the
        applicability gate a capability it then trusts."""
        assert not transport().capabilities.exposes_partials
        assert not ns.SPEC.capabilities.emits_partials

    def test_it_cannot_claim_a_composed_seam(self) -> None:
        for stage in (Stage.LLM_FIRST_TOKEN, Stage.TTS_FIRST_BYTE):
            assert stage in CASCADE_ONLY_STAGES
            assert stage not in transport().capabilities.exposes_stages

    def test_nothing_is_attested_until_conformance_runs(self) -> None:
        assert transport().capabilities.attested == frozenset()
        assert ns.SPEC.capabilities.attested == frozenset()

    def test_the_session_cap_is_declared(self) -> None:
        """8 minutes, so a longer task fails at load not mid-measurement."""
        assert transport().capabilities.max_session_s == ns.MAX_SESSION_S == 480.0

    def test_native_vad_because_endpointing_drives_turn_taking(self) -> None:
        assert transport().capabilities.turn_control is TurnControl.NATIVE_VAD

    def test_both_floors_are_supported(self) -> None:
        assert transport().capabilities.duplex == frozenset({Duplex.HALF, Duplex.FULL})


class TestTheRateAsymmetryIsReal:
    def test_input_is_sixteen_k_and_output_is_twenty_four(self) -> None:
        """The provider's asymmetry, not ours."""
        assert ns.NOVA_INPUT_FORMAT.sample_rate_hz == 16_000
        assert ns.NOVA_OUTPUT_FORMAT.sample_rate_hz == 24_000

    def test_output_needs_no_conversion_at_all(self) -> None:
        """Which is why 24 kHz stays canonical."""
        assert ns.NOVA_OUTPUT_FORMAT == CANONICAL


class TestEventProtocol:
    def test_init_events_are_in_protocol_order(self) -> None:
        keys = [next(iter(json.loads(e)["event"])) for e in transport()._init_events()]
        assert keys == [
            "sessionStart",
            "promptStart",
            "contentStart",
            "textInput",
            "contentEnd",
            "contentStart",
        ]

    def test_the_output_config_asks_for_our_canonical_rate(self) -> None:
        prompt = json.loads(transport()._init_events()[1])["event"]["promptStart"]
        audio = prompt["audioOutputConfiguration"]
        assert audio["sampleRateHertz"] == CANONICAL.sample_rate_hz
        assert audio["encoding"] == "base64"

    def test_the_input_config_declares_sixteen_k(self) -> None:
        start = json.loads(transport()._init_events()[5])["event"]["contentStart"]
        assert start["type"] == "AUDIO"
        assert start["role"] == "USER"
        assert start["audioInputConfiguration"]["sampleRateHertz"] == 16_000


class TestSpeculativeTextIsNotADraft:
    """Observed: it is the only assistant text, and it matches the audio."""

    def test_speculative_is_detected(self) -> None:
        assert ns._is_speculative(
            {"additionalModelFields": json.dumps({"generationStage": "SPECULATIVE"})}
        )

    def test_absent_or_malformed_fields_are_not_speculative(self) -> None:
        assert not ns._is_speculative({})
        assert not ns._is_speculative({"additionalModelFields": "not json"})

    def test_speculative_assistant_text_still_counts_as_said(self) -> None:
        """Excluding it produced an agent that spoke and reported silence."""
        t = transport()
        t.transcripts.append(
            ns.NovaSonicTranscript(
                text="I can hear you clearly.", role="ASSISTANT", speculative=True
            )
        )
        assert t.said == ["I can hear you clearly."]
        assert t.heard == []

    def test_user_role_text_is_the_heard_side(self) -> None:
        t = transport()
        t.transcripts.append(ns.NovaSonicTranscript(text="hello there", role="USER"))
        assert t.heard == ["hello there"]
        assert t.said == []


class TestFailureClassification:
    def test_an_unenabled_model_is_distinguished_from_bad_credentials(self) -> None:
        """Bedrock raises AccessDeniedException for both."""
        named = ns._classify(
            RuntimeError(f"User is not authorized to invoke {ns.MODEL_ID}"),
            ns.MODEL_ID,
        )
        assert named.kind is ProviderFailure.NOT_ENABLED
        assert "Model access" in str(named)

        anonymous = ns._classify(RuntimeError("access denied"), ns.MODEL_ID)
        assert anonymous.kind is ProviderFailure.AUTH

    def test_throttling_and_timeouts_are_retryable(self) -> None:
        assert ns._classify(RuntimeError("ThrottlingException"), "m").is_retryable
        assert ns._classify(RuntimeError("timeout"), "m").is_retryable

    def test_an_unknown_failure_invalidates_rather_than_scoring_zero(self) -> None:
        error = ns._classify(RuntimeError("something odd"), "m")
        assert error.kind is ProviderFailure.STREAM_BROKEN
        assert error.invalidates_trial


class TestRecordKeeping:
    def test_the_ref_carries_no_secret(self) -> None:
        ref = transport(voice="joanna").ref.as_dict()
        assert ref["provider"] == "aws.nova-sonic"
        assert ref["region"] == "us-east-1"
        assert "joanna" not in str(ref)
        assert ref["config_digest"].startswith("sha256:")

    def test_usage_separates_speech_tokens_from_text_tokens(self) -> None:
        """The concrete case for wrapping harbor's ``UsageInfo``."""
        t = transport()
        t._usage = {
            "totalInputTokens": 509,
            "totalOutputTokens": 76,
            "details": {
                "total": {
                    "input": {"speechTokens": 206, "textTokens": 303},
                    "output": {"speechTokens": 0, "textTokens": 76},
                }
            },
        }
        usage = t.usage.as_dict()
        assert usage["speech_tokens"] == 206
        assert usage["text_tokens"] == 379
        assert usage["tokens"]["prompt_tokens"] == 509

    def test_an_empty_usage_reports_zeros_not_a_crash(self) -> None:
        assert transport().usage.speech_tokens == 0


class TestSdkIsOptional:
    def test_the_spec_is_readable_without_the_sdk(self) -> None:
        """Doctor runs before a container, without credentials."""
        assert ns.SPEC.requires_extra == "aws"
        assert ns.SPEC.credential_provider == "aws"
        assert ns.SPEC.default_model == ns.MODEL_ID
