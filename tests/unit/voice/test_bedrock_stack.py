"""The all-Bedrock stack — offline. Facts pinned from live bring-up.

No network. Every behaviour asserted here was observed against the live
service, so a refactor cannot quietly undo it.
"""

from __future__ import annotations

import pytest

from harbor_voice.providers.aws import bedrock
from harbor_voice.providers.aws import nova_sonic as ns
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapability
from harbor_voice.providers.registry import default_registry
from harbor_voice.capabilities import Verdict

pytestmark = pytest.mark.unit


class TestTheStackFillsEverySlot:
    """Bedrock alone, without Polly or Transcribe."""

    def test_all_four_modalities_are_registered(self) -> None:
        from harbor_voice.providers.aws import register_all

        registry = default_registry()
        register_all(registry)
        for modality, name in (
            (Modality.STT, "aws.voxtral"),
            (Modality.LLM, "aws.bedrock-llm"),
            (Modality.TTS, "aws.nova-tts"),
            (Modality.S2S, "aws.nova-sonic"),
        ):
            assert name in [e.spec.name for e in registry.for_modality(modality)]

    def test_registering_is_explicit_not_an_import_side_effect(self) -> None:
        """Import order must not decide which providers exist."""
        registry = default_registry()
        assert "aws.voxtral" not in registry

    def test_every_aws_provider_shares_one_credential_slug(self) -> None:
        for spec in (
            bedrock.VOXTRAL_SPEC,
            bedrock.BEDROCK_LLM_SPEC,
            bedrock.NOVA_TTS_SPEC,
            ns.SPEC,
        ):
            assert spec.credential_provider == "aws"
            assert spec.requires_extra == "aws"


class TestCapabilitiesMatchReality:
    def test_voxtral_does_not_claim_streaming(self) -> None:
        """Converse is one-shot: audio in, whole transcript out."""
        caps = bedrock.VOXTRAL_SPEC.capabilities
        assert not caps.streams
        assert not caps.emits_partials

    def test_a_one_shot_stt_abstains_from_first_byte_metrics(self) -> None:
        """The abstention is the point of declaring it."""
        verdict = bedrock.VOXTRAL_SPEC.capabilities.evaluate(ProviderCapability.STREAMS)
        assert verdict.verdict is Verdict.NOT_APPLICABLE

    def test_the_llm_does_claim_streaming(self) -> None:
        """ConverseStream is genuinely incremental, so first-token is real."""
        assert bedrock.BEDROCK_LLM_SPEC.capabilities.streams

    def test_the_tts_does_not_claim_streaming(self) -> None:
        """It withholds audio until the verbatim check passes, so a
        first-byte mark would be taken after a whole round trip."""
        assert not bedrock.NOVA_TTS_SPEC.capabilities.streams

    def test_nothing_is_attested(self) -> None:
        for spec in (
            bedrock.VOXTRAL_SPEC,
            bedrock.BEDROCK_LLM_SPEC,
            bedrock.NOVA_TTS_SPEC,
        ):
            assert spec.capabilities.attested == frozenset()


class TestTheDefaultLlmIsActuallyInvokable:
    def test_it_is_an_on_demand_model(self) -> None:
        """Claude's bare id raises 'needs an inference profile'; Nova does not."""
        assert bedrock.DEFAULT_LLM == "amazon.nova-pro-v1:0"

    def test_an_inference_profile_error_fails_the_job(self) -> None:
        """Every trial would fail identically, so it is a config error."""
        error = ns._classify(
            RuntimeError(
                "Invocation of model ID anthropic.claude-sonnet-4 with "
                "on-demand throughput isn't supported. Retry your request "
                "with the ID or ARN of an inference profile"
            ),
            "anthropic.claude-sonnet-4",
        )
        assert error.kind is ProviderFailure.NOT_ENABLED
        assert not error.is_retryable
        assert "inference-profile" in str(error)


class TestErrorsNameTheProviderThatRaisedThem:
    def test_classify_defaults_to_nova_sonic(self) -> None:
        assert ns._classify(RuntimeError("x"), "m").provider == "aws.nova-sonic"

    def test_but_the_bedrock_stack_names_itself(self) -> None:
        """A Bedrock LLM failure reported as 'aws.nova-sonic' sends whoever
        reads the log to the wrong provider."""
        error = ns._classify(RuntimeError("x"), "m", "aws.bedrock-llm")
        assert error.provider == "aws.bedrock-llm"
        assert "aws.bedrock-llm" in str(error)


class TestTheVerbatimGuard:
    """An S2S model coerced into reading text can paraphrase.

    A paraphrase corrupts ``said`` — the reference side of said-vs-heard —
    and would make every WER in the run wrong in a direction nothing reveals.
    So it is checked, not trusted.
    """

    def test_an_exact_match_is_verbatim(self) -> None:
        spoken = bedrock.SpokenAudio(
            pcm=b"", spoken_text="Next Tuesday.", requested_text="Next Tuesday."
        )
        assert spoken.is_verbatim

    def test_punctuation_and_case_do_not_count_as_paraphrase(self) -> None:
        spoken = bedrock.SpokenAudio(
            pcm=b"", spoken_text="next tuesday", requested_text="Next Tuesday."
        )
        assert spoken.is_verbatim

    def test_a_dropped_word_is_caught(self) -> None:
        spoken = bedrock.SpokenAudio(
            pcm=b"",
            spoken_text="I need to reschedule my appointment for Tuesday.",
            requested_text="I need to reschedule my appointment for next Tuesday.",
        )
        assert not spoken.is_verbatim

    def test_a_conversational_reply_is_caught(self) -> None:
        """The failure mode: the model answers instead of reading."""
        spoken = bedrock.SpokenAudio(
            pcm=b"",
            spoken_text="Sure, I can help you reschedule.",
            requested_text="I need to reschedule my appointment.",
        )
        assert not spoken.is_verbatim

    async def test_synthesize_refuses_to_emit_paraphrased_audio(self) -> None:
        from harbor_voice.ids import PartyId
        from harbor_voice.providers.base import ProviderError

        tts = bedrock.NovaSonicTTS(region="us-east-1")

        async def fake(text: str) -> bedrock.SpokenAudio:
            return bedrock.SpokenAudio(
                pcm=b"\x00" * 960,
                spoken_text="something else entirely",
                requested_text=text,
            )

        tts.synthesize_checked = fake  # type: ignore[method-assign]
        with pytest.raises(ProviderError, match="silently corrupts"):
            async for _ in tts.synthesize("read this", party=PartyId("caller")):
                pass

    async def test_empty_text_yields_nothing_without_a_round_trip(self) -> None:
        from harbor_voice.ids import PartyId

        tts = bedrock.NovaSonicTTS(region="us-east-1")
        chunks = [c async for c in tts.synthesize("", party=PartyId("caller"))]
        assert chunks == []


class TestStreamedFragmentsConcatenate:
    """Found by a real provider, invisible to the whole offline suite.

    A streaming LLM yields arbitrary text deltas. The cascade space-joined
    them, which was correct for ``EchoLLM``'s whole words and wrong for
    everything else — and 618 tests passed either way.
    """

    async def test_fragments_rebuild_the_reply_exactly(self) -> None:
        from harbor_voice.providers import EchoLLM

        llm = EchoLLM(chunk_words=2)
        chunks = [c async for c in llm.respond("one two three four five", [])]
        assert len(chunks) > 1
        assert "".join(chunks) == "one two three four five"

    async def test_a_prefix_survives_concatenation(self) -> None:
        from harbor_voice.providers import EchoLLM

        llm = EchoLLM(prefix="understood: ", chunk_words=2)
        chunks = [c async for c in llm.respond("book it now", [])]
        assert "".join(chunks) == "understood: book it now"

    async def test_token_shaped_deltas_do_not_gain_spaces(self) -> None:
        """The shape a real ConverseStream produces."""
        from collections.abc import AsyncIterator
        from dataclasses import dataclass

        @dataclass
        class TokenLLM:
            name: str = "token-llm"

            async def respond(self, prompt: str, history: object) -> AsyncIterator[str]:
                for piece in ("Sure", ", I'll", " reschedule", " it."):
                    yield piece

        chunks = [c async for c in TokenLLM().respond("x", [])]
        assert "".join(chunks) == "Sure, I'll reschedule it."
        assert " ".join(chunks) != "Sure, I'll reschedule it."


class TestTheVerbatimGuardDoesNotCryWolf:
    def test_whitespace_differences_are_not_paraphrase(self) -> None:
        """The guard's first live firing was this, and it was wrong."""
        spoken = bedrock.SpokenAudio(
            pcm=b"",
            spoken_text="Sure ,  I'll reschedule your appointment for next Tuesday.",
            requested_text="Sure , I'll reschedule  your appointment for next Tuesday.",
        )
        assert spoken.is_verbatim

    def test_a_real_word_change_still_fails(self) -> None:
        spoken = bedrock.SpokenAudio(
            pcm=b"",
            spoken_text="I'll reschedule it for Wednesday.",
            requested_text="I'll reschedule it for Tuesday.",
        )
        assert not spoken.is_verbatim


# ----------------------------------------------- the second recogniser


class TestTwoVendorsInOneAccount:
    """The reference stack needs an independent recogniser. We have two."""

    def test_nova_stt_is_registered_as_its_own_provider(self) -> None:
        from harbor_voice.providers.aws import register_all

        registry = default_registry()
        register_all(registry)
        entry, _ = registry.resolve("aws.nova-stt")
        assert entry.spec.modality is Modality.STT
        assert entry.is_implemented

    def test_a_cascade_agent_can_be_measured_by_the_other_vendor(self) -> None:
        from harbor_voice.providers.measurement import ReferenceStack

        ReferenceStack(reference_stt="aws.nova-stt").check_independence(
            agent_stt="aws.voxtral"
        )

    def test_an_s2s_agent_has_no_stt_to_collide_with(self) -> None:
        from harbor_voice.providers.measurement import ReferenceStack

        ReferenceStack(reference_stt="aws.voxtral").check_independence(agent_stt=None)

    def test_one_vendor_on_both_sides_is_still_refused(self) -> None:
        """The rule is what makes the second recogniser worth having."""
        from harbor_voice.providers.measurement import MeasurementIndependenceError
        from harbor_voice.providers.measurement import ReferenceStack

        with pytest.raises(MeasurementIndependenceError, match="against itself"):
            ReferenceStack(reference_stt="aws.nova-stt").check_independence(
                agent_stt="aws.nova-stt"
            )

    def test_it_declares_no_streaming(self) -> None:
        """A reference transcriber runs offline over stored audio, where a
        first-byte mark would be a fiction."""
        assert bedrock.NOVA_STT_SPEC.capabilities.streams is False


# ------------------------------------------------------------- L1 voice


class TestNovaSonicVoiceSelection:
    def test_it_declares_the_optional_protocol(self) -> None:
        from harbor_voice.perturbation import VoiceSelectable

        assert isinstance(bedrock.NovaSonicTTS(), VoiceSelectable)

    def test_gender_is_servable(self) -> None:
        from harbor_voice.perturbation import Gender
        from harbor_voice.perturbation import VoiceSelection

        tts = bedrock.NovaSonicTTS()
        assert tts.select_voice(VoiceSelection(gender=Gender.FEMALE)) is True
        assert tts.voice == "tiffany"

    def test_an_accent_is_refused_rather_than_faked(self) -> None:
        """EVA condition A varies accented English. A voice catalogue indexed
        by language cannot express that, and pretending otherwise would put a
        condition's name on a default voice."""
        from harbor_voice.perturbation import Accent
        from harbor_voice.perturbation import VoiceSelection

        tts = bedrock.NovaSonicTTS()
        assert tts.select_voice(VoiceSelection(accent=Accent.INDIAN)) is False

    def test_an_unconfirmed_language_is_refused_not_guessed(self) -> None:
        """Naming a voice the service lacks fails at session open, mid-run."""
        from harbor_voice.perturbation import VoiceSelection

        assert (
            bedrock.NovaSonicTTS().select_voice(VoiceSelection(language="ja")) is False
        )

    def test_a_pinned_voice_id_is_taken_at_face_value(self) -> None:
        from harbor_voice.perturbation import VoiceSelection

        tts = bedrock.NovaSonicTTS()
        assert tts.select_voice(VoiceSelection(voice_id="amy")) is True
        assert tts.voice == "amy"

    def test_an_accent_through_this_tts_records_that_it_missed(self) -> None:
        """The honest outcome: the run happens and does not claim the accent."""
        from harbor_voice.perturbation import SeedSet
        from harbor_voice.perturbation import get_profile

        pipeline = get_profile("accented").pipeline(SeedSet.derive(1))
        pipeline.persona.apply_to_tts(bedrock.NovaSonicTTS())
        record = pipeline.as_record().as_dict()["layers"]
        assert record["L1.accent"]["achieved"] == {"voice_honoured": 0.0}
