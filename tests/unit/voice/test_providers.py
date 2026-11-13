"""Unit tests for the provider abstraction."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from harbor_voice.audio.clock import Clock
from harbor_voice.audio.clock import Interval
from harbor_voice.audio.format import AudioFormat
from harbor_voice.capabilities import Applicability
from harbor_voice.capabilities import combine
from harbor_voice.capabilities import Verdict
from harbor_voice.providers.measurement import MeasurementIndependenceError
from harbor_voice.providers.measurement import ReferenceStack
from harbor_voice.providers.cache import BlobCache
from harbor_voice.providers.cache import Cassette
from harbor_voice.providers.cache import CassetteMode
from harbor_voice.providers.cache import ReplayedRunError
from harbor_voice.providers.cache import require_live
from harbor_voice.providers.cache import SynthesisKey
from harbor_voice.providers import Disposition
from harbor_voice.providers import EchoLLM
from harbor_voice.providers import LLMProvider
from harbor_voice.providers import Modality
from harbor_voice.providers import ProviderCall
from harbor_voice.providers import ProviderCapabilities
from harbor_voice.providers import ProviderCapability
from harbor_voice.providers import ProviderCapabilityError
from harbor_voice.providers import ProviderError
from harbor_voice.providers import ProviderFailure
from harbor_voice.providers import ProviderRef
from harbor_voice.providers import ProviderSpec
from harbor_voice.providers import ScriptedSTT
from harbor_voice.providers import STTProvider
from harbor_voice.providers import ToneTTS
from harbor_voice.providers import TTSProvider
from harbor_voice.providers import Usage
from harbor_voice.providers import default_registry
from harbor_voice.providers import ModalityMismatch
from harbor_voice.providers import parse_spec
from harbor_voice.providers import ProviderRegistry
from harbor_voice.providers import UnknownProvider
from harbor_voice.providers.credentials import access_for
from harbor_voice.providers.credentials import resolve as resolve_credentials
from harbor_voice.providers.credentials import VOICE_PROVIDER_ACCESS
from tests.unit.voice.fakes import Closed
from tests.unit.voice.fakes import Rejected

pytestmark = pytest.mark.unit


def caps(modality: Modality = Modality.STT, **over: object) -> ProviderCapabilities:
    base: dict[str, object] = {"modality": modality}
    base.update(over)
    return ProviderCapabilities(**base)


# ------------------------------------------------------- the move itself


class TestProvidersAreTopLevel:
    """The protocols serve four consumers, only one of which is the cascade."""

    def test_the_offline_set_imports_without_touching_cascade(self) -> None:
        """A clean interpreter, because clearing ``sys.modules`` in-process
        leaves this module holding the *old* classes while everything built
        afterwards uses the new ones — enum identity then fails in tests far
        from here."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import harbor_voice.providers, sys; "
                "print(any(m.startswith('harbor_voice.cascade') for m in sys.modules))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"

    def test_the_offline_set_still_satisfies_its_protocols(self) -> None:
        assert isinstance(ScriptedSTT(), STTProvider)
        assert isinstance(EchoLLM(), LLMProvider)
        assert isinstance(ToneTTS(), TTSProvider)


# ------------------------------------------------------------- modality


class TestModality:
    def test_s2s_is_not_a_slot(self) -> None:
        """It is a transport; forcing it into a slot would invent a shape."""
        assert not Modality.S2S.is_slot
        assert all(m.is_slot for m in (Modality.STT, Modality.LLM, Modality.TTS))

    def test_audio_direction(self) -> None:
        assert Modality.STT.consumes_audio and not Modality.STT.produces_audio
        assert Modality.TTS.produces_audio and not Modality.TTS.consumes_audio
        assert Modality.S2S.consumes_audio and Modality.S2S.produces_audio
        assert not Modality.LLM.consumes_audio and not Modality.LLM.produces_audio


# --------------------------------------------------------- capabilities


class TestCapabilitiesRejectImpossibleClaims:
    def test_a_tts_cannot_emit_partial_transcripts(self) -> None:
        with pytest.raises(ValidationError) as exc:
            caps(Modality.TTS, streams=True, emits_partials=True)
        assert "does not consume audio" in str(exc.value)

    def test_partials_require_streaming(self) -> None:
        with pytest.raises(ValidationError) as exc:
            caps(Modality.STT, streams=False, emits_partials=True)
        assert "requires streams=True" in str(exc.value)

    def test_an_llm_cannot_report_word_timings(self) -> None:
        with pytest.raises(ValidationError) as exc:
            caps(Modality.LLM, word_timings=True)
        assert "produces no audio" in str(exc.value)

    def test_a_tts_can_report_word_timings(self) -> None:
        """Polly's speech marks are exactly this, and they are useful."""
        assert caps(Modality.TTS, word_timings=True).word_timings

    def test_cannot_attest_what_was_not_claimed(self) -> None:
        with pytest.raises(ValidationError) as exc:
            caps(
                Modality.STT,
                streams=False,
                attested=frozenset({ProviderCapability.STREAMS}),
            )
        assert "cannot attest" in str(exc.value)

    def test_the_error_is_a_provider_capability_error_inside(self) -> None:
        with pytest.raises(ValidationError) as exc:
            caps(Modality.LLM, word_timings=True)
        assert any(
            isinstance(e.get("ctx", {}).get("error"), ProviderCapabilityError)
            for e in exc.value.errors()
        )


class TestApplicabilityOneLevelDown:
    """A missing capability abstains; it does not report zero."""

    def test_a_one_shot_tts_makes_first_byte_not_applicable(self) -> None:
        verdict = caps(Modality.TTS, streams=False).evaluate(ProviderCapability.STREAMS)
        assert verdict.verdict is Verdict.NOT_APPLICABLE
        assert not verdict.may_report
        assert verdict.reasons == ("provider does not support streams",)

    def test_a_declared_but_unverified_capability_cannot_publish(self) -> None:
        verdict = caps(Modality.TTS, streams=True).evaluate(ProviderCapability.STREAMS)
        assert verdict.verdict is Verdict.UNATTESTED
        assert verdict.may_report
        assert not verdict.may_publish

    def test_an_attested_capability_may_publish(self) -> None:
        verdict = caps(
            Modality.TTS,
            streams=True,
            attested=frozenset({ProviderCapability.STREAMS}),
        ).evaluate(ProviderCapability.STREAMS)
        assert verdict.verdict is Verdict.SATISFIED
        assert verdict.may_publish

    def test_every_missing_capability_is_reported_not_just_the_first(self) -> None:
        verdict = caps(Modality.STT).evaluate(
            ProviderCapability.STREAMS,
            ProviderCapability.WORD_TIMINGS,
            ProviderCapability.DIARIZES,
        )
        assert len(verdict.reasons) == 3

    def test_no_requirements_is_satisfied(self) -> None:
        assert caps().evaluate().verdict is Verdict.SATISFIED

    def test_has_covers_every_member(self) -> None:
        """A new capability must not silently fall through ``has``."""
        full = caps(
            Modality.STT,
            streams=True,
            emits_partials=True,
            word_timings=True,
            diarizes=True,
        )
        assert all(full.has(c) for c in ProviderCapability)


# ------------------------------------------------------------------ spec


class TestProviderSpec:
    def test_capabilities_must_agree_with_the_declared_modality(self) -> None:
        with pytest.raises(ValueError, match="declares modality=tts"):
            ProviderSpec(
                name="aws.polly",
                modality=Modality.TTS,
                capabilities=caps(Modality.STT),
            )

    def test_a_name_cannot_contain_a_slash(self) -> None:
        with pytest.raises(ValueError, match="cannot contain"):
            ProviderSpec(
                name="aws/polly",
                modality=Modality.TTS,
                capabilities=caps(Modality.TTS),
            )

    def test_the_offline_set_is_offline(self) -> None:
        spec = ProviderSpec(
            name="offline.tone-tts",
            modality=Modality.TTS,
            capabilities=caps(Modality.TTS, streams=True),
        )
        assert spec.is_offline

    def test_a_cloud_provider_is_not(self) -> None:
        spec = ProviderSpec(
            name="aws.polly",
            modality=Modality.TTS,
            capabilities=caps(Modality.TTS, streams=True),
            credential_provider="aws-polly",
            requires_extra="aws",
        )
        assert not spec.is_offline

    def test_capabilities_are_readable_without_constructing_anything(self) -> None:
        """``doctor`` runs before a container starts, without creds."""
        spec = ProviderSpec(
            name="aws.transcribe",
            modality=Modality.STT,
            capabilities=caps(Modality.STT, streams=True, emits_partials=True),
            credential_provider="aws-transcribe",
            requires_extra="aws",
        )
        assert spec.capabilities.emits_partials


# ------------------------------------------------------------------- ref


class TestProviderRefHoldsNoSecret:
    def test_there_is_no_field_a_credential_could_occupy(self) -> None:
        fields = set(ProviderRef.__dataclass_fields__)
        assert fields == {
            "provider",
            "model",
            "region",
            "config_digest",
            "capabilities",
        }
        assert not any(
            word in f for f in fields for word in ("key", "secret", "token", "cred")
        )

    def test_config_is_digested_not_stored(self) -> None:
        ref = ProviderRef.build(
            provider="aws.polly",
            model="Joanna",
            region="us-east-1",
            config={"engine": "neural", "rate": "95%"},
        )
        assert ref.config_digest is not None
        assert ref.config_digest.startswith("sha256:")
        assert "neural" not in str(ref.as_dict())

    def test_the_digest_is_order_independent(self) -> None:
        a = ProviderRef.build(provider="p", model="m", config={"x": 1, "y": 2})
        b = ProviderRef.build(provider="p", model="m", config={"y": 2, "x": 1})
        assert a.config_digest == b.config_digest

    def test_different_config_gives_a_different_digest(self) -> None:
        a = ProviderRef.build(provider="p", model="m", config={"engine": "neural"})
        b = ProviderRef.build(provider="p", model="m", config={"engine": "standard"})
        assert a.config_digest != b.config_digest

    def test_no_config_means_no_digest_rather_than_a_digest_of_nothing(self) -> None:
        assert ProviderRef.build(provider="p", model="m").config_digest is None

    def test_spec_string_round_trips_the_cli_form(self) -> None:
        ref = ProviderRef(provider="aws.polly", model="Joanna")
        assert ref.spec_string == "aws.polly/Joanna"
        assert str(ref) == "aws.polly/Joanna"

    def test_region_is_recorded_because_it_changes_the_result(self) -> None:
        ref = ProviderRef(provider="bedrock", model="claude", region="us-east-1")
        assert ref.as_dict()["region"] == "us-east-1"


# ----------------------------------------------------------------- usage


class TestUsageIsVoiceShaped:
    def test_none_and_zero_say_different_things(self) -> None:
        """``None`` = not billed in this unit. ``0`` = billed, and it was zero."""
        polly = Usage(characters=120)
        assert polly.audio_seconds is None
        assert "audio_seconds" not in polly.as_dict()

        empty = Usage(characters=0)
        assert empty.as_dict()["characters"] == 0

    def test_harbor_usage_info_is_wrapped_not_bent(self) -> None:
        from harbor.models.metric import UsageInfo

        usage = Usage(
            tokens=UsageInfo(
                prompt_tokens=10, completion_tokens=5, cache_tokens=0, cost_usd=0.01
            )
        )
        assert usage.as_dict()["tokens"]["prompt_tokens"] == 10

    def test_latency_carries_its_clock_pair(self) -> None:
        """Clock pairing holds one level down too."""
        usage = Usage(latency=Interval(0.0, 0.25, Clock.TX, Clock.RX))
        assert usage.as_dict()["latency_tx_to_rx"] == pytest.approx(0.25)

    def test_negative_consumption_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            Usage(audio_seconds=-1.0)
        with pytest.raises(ValueError, match="cannot be negative"):
            Usage(cost_usd=-0.01)

    def test_cache_hit_and_replay_are_separate_facts(self) -> None:
        """They have opposite legitimacy, so they cannot be one flag."""
        assert Usage(cache_hit=True).is_free
        assert Usage(replayed=True).is_free
        assert "replayed" not in Usage(cache_hit=True).as_dict()
        assert "cache_hit" not in Usage(replayed=True).as_dict()

    def test_an_empty_usage_records_nothing(self) -> None:
        assert Usage().as_dict() == {}


# --------------------------------------------------------------- failure


class TestFailureIsAVocabulary:
    def test_configuration_failures_fail_the_job_not_the_trial(self) -> None:
        """200 identical zeros would hide one typo."""
        for kind in (
            ProviderFailure.AUTH,
            ProviderFailure.NOT_ENABLED,
            ProviderFailure.REGION,
        ):
            error = ProviderError(kind, "nope", provider="bedrock")
            assert error.disposition is Disposition.FAIL_JOB
            assert not error.is_retryable

    def test_transient_failures_retry(self) -> None:
        error = ProviderError(ProviderFailure.THROTTLED, "slow down", provider="p")
        assert error.is_retryable
        assert error.invalidates_trial

    def test_a_broken_stream_invalidates_rather_than_scoring_zero(self) -> None:
        error = ProviderError(ProviderFailure.STREAM_BROKEN, "eof", provider="p")
        assert error.disposition is Disposition.INVALID
        assert error.invalidates_trial

    def test_not_enabled_is_distinct_from_auth(self) -> None:
        """Bedrock raises AccessDeniedException for both; we must not."""
        assert ProviderFailure.NOT_ENABLED is not ProviderFailure.AUTH

    def test_every_failure_has_a_disposition(self) -> None:
        for kind in ProviderFailure:
            assert ProviderError(kind, "x", provider="p").disposition

    def test_the_message_names_the_provider_and_model(self) -> None:
        error = ProviderError(
            ProviderFailure.NOT_ENABLED,
            "enable it in the console",
            provider="bedrock",
            model="anthropic.claude-sonnet-4-5",
        )
        assert "[not_enabled]" in str(error)
        assert "bedrock/anthropic.claude-sonnet-4-5" in str(error)


class TestProviderCall:
    def test_a_successful_call_records_no_failure(self) -> None:
        call = ProviderCall(
            ref=ProviderRef(provider="aws.polly", model="Joanna"),
            usage=Usage(characters=42),
        )
        assert call.succeeded
        assert call.as_dict() == {
            "provider": "aws.polly",
            "model": "Joanna",
            "usage": {"characters": 42},
        }

    def test_a_failed_call_records_both_kind_and_routing(self) -> None:
        call = ProviderCall(
            ref=ProviderRef(provider="bedrock", model="claude"),
            failure=ProviderFailure.THROTTLED,
        )
        assert not call.succeeded
        assert call.as_dict()["failure"] == "throttled"
        assert call.as_dict()["disposition"] == "retry"


# -------------------------------------------------------------- registry


def fake_env(**values: str) -> Callable[[str], str | None]:
    """An environment lookup that never touches the real one."""
    return values.get


class TestSpecStrings:
    def test_provider_and_model_split_on_the_first_slash(self) -> None:
        assert parse_spec("aws.polly/Joanna") == ("aws.polly", "Joanna")

    def test_a_model_may_itself_contain_slashes(self) -> None:
        """Bedrock model ids do."""
        provider, model = parse_spec("bedrock/anthropic/claude-sonnet-4-5")
        assert provider == "bedrock"
        assert model == "anthropic/claude-sonnet-4-5"

    def test_a_bare_provider_needs_a_default(self) -> None:
        assert parse_spec("offline.tone-tts", default_model="tone") == (
            "offline.tone-tts",
            "tone",
        )

    def test_a_bare_provider_without_a_default_is_refused(self) -> None:
        """Guessing would make two identical-looking runs use different models."""
        with pytest.raises(ValueError, match="names no model"):
            parse_spec("aws.polly")

    def test_a_trailing_slash_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no model after it"):
            parse_spec("aws.polly/")

    def test_an_empty_spec_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_spec("   ")


class TestRegistryDescribesWithoutConstructing:
    """``doctor`` runs before any container, without credentials."""

    def test_the_offline_set_is_registered(self) -> None:
        registry = default_registry()
        assert {e.spec.name for e in registry} == {
            "offline.scripted-stt",
            "offline.echo-llm",
            "offline.scripted-llm",
            "offline.tool-echo-llm",
            "offline.tone-tts",
        }

    def test_capabilities_come_from_the_name_alone(self) -> None:
        registry = default_registry()
        assert registry.capabilities("offline.scripted-stt").emits_partials

    def test_an_unimplemented_provider_still_describes_itself(self) -> None:
        registry = ProviderRegistry()
        spec = ProviderSpec(
            name="aws.nova-sonic",
            modality=Modality.S2S,
            capabilities=caps(Modality.S2S, streams=True),
            credential_provider="aws",
            requires_extra="aws",
            default_model="amazon.nova-sonic-v1:0",
        )
        registry.register(spec)

        assert not registry.lookup("aws.nova-sonic").is_implemented
        assert registry.capabilities("aws.nova-sonic").streams
        with pytest.raises(NotImplementedError, match="not implemented"):
            registry.create("aws.nova-sonic", modality=Modality.S2S)

    def test_an_unknown_provider_lists_what_is_known(self) -> None:
        with pytest.raises(UnknownProvider, match="offline.tone-tts"):
            default_registry().lookup("aws.polly")

    def test_registering_a_name_twice_is_refused(self) -> None:
        """Silent replacement makes the result depend on import order."""
        registry = default_registry()
        with pytest.raises(ValueError, match="already registered"):
            registry.register(
                ProviderSpec(
                    name="offline.tone-tts",
                    modality=Modality.TTS,
                    capabilities=caps(Modality.TTS),
                )
            )

    def test_for_modality_filters_the_slot(self) -> None:
        registry = default_registry()
        assert [e.spec.name for e in registry.for_modality(Modality.LLM)] == [
            "offline.echo-llm",
            "offline.scripted-llm",
            "offline.tool-echo-llm",
        ]
        assert registry.for_modality(Modality.S2S) == []


class TestRegistryCreate:
    def test_config_reaches_the_constructor(self) -> None:
        tts = default_registry().create(
            "offline.tone-tts", modality=Modality.TTS, hz=440.0
        )
        assert isinstance(tts, ToneTTS)
        assert tts.hz == 440.0

    def test_the_wrong_slot_fails_at_configuration_not_at_first_audio(self) -> None:
        with pytest.raises(ModalityMismatch, match="cannot fill the stt slot"):
            default_registry().create("offline.tone-tts", modality=Modality.STT)

    def test_the_default_model_is_supplied_by_the_registration(self) -> None:
        entry, model = default_registry().resolve("offline.scripted-stt")
        assert model == "fixed"
        assert entry.spec.name == "offline.scripted-stt"


# ----------------------------------------------------------- credentials


class TestCredentialOverlay:
    def test_harbors_table_is_reused_not_rebuilt(self) -> None:
        access = access_for("amazon-bedrock")
        assert access is not None
        assert "AWS_SECRET_ACCESS_KEY" in access.extra_envs

    def test_the_voice_overlay_is_consulted_first(self) -> None:
        assert "aws" in VOICE_PROVIDER_ACCESS
        assert access_for("aws") is VOICE_PROVIDER_ACCESS["aws"]

    def test_an_unknown_slug_resolves_to_nothing_rather_than_guessing(self) -> None:
        assert access_for("not-a-provider") is None
        assert resolve_credentials("not-a-provider").needs_nothing

    def test_a_complete_aws_environment_is_ready(self) -> None:
        status = resolve_credentials(
            "aws",
            getenv=fake_env(
                AWS_ACCESS_KEY_ID="AKIA...",
                AWS_SECRET_ACCESS_KEY="secret",
                AWS_REGION="us-east-1",
            ),
        )
        assert status.ready
        assert "AWS_ACCESS_KEY_ID" in status.present

    def test_a_missing_session_token_is_normal_not_broken(self) -> None:
        """Unset for a long-lived key pair; reporting it would cry wolf."""
        status = resolve_credentials(
            "aws",
            getenv=fake_env(
                AWS_ACCESS_KEY_ID="AKIA...",
                AWS_SECRET_ACCESS_KEY="secret",
                AWS_REGION="us-east-1",
            ),
        )
        assert status.optional_missing == ("AWS_SESSION_TOKEN",)
        assert status.ready

    def test_a_missing_region_is_reported(self) -> None:
        status = resolve_credentials(
            "aws",
            getenv=fake_env(AWS_ACCESS_KEY_ID="AKIA...", AWS_SECRET_ACCESS_KEY="s"),
        )
        assert not status.ready
        assert "AWS_REGION" in status.missing
        assert "missing AWS_REGION" in status.explain()

    def test_an_empty_environment_reports_every_requirement(self) -> None:
        status = resolve_credentials("aws", getenv=fake_env())
        assert not status.ready
        assert "AWS_ACCESS_KEY_ID" in status.missing[0]
        assert "AWS_SECRET_ACCESS_KEY" in status.missing

    def test_a_secret_never_reaches_repr_or_equality(self) -> None:
        """A status is printed by ``doctor``."""
        status = resolve_credentials(
            "aws",
            getenv=fake_env(
                AWS_ACCESS_KEY_ID="AKIA-SECRET-VALUE",
                AWS_SECRET_ACCESS_KEY="SHHH",
                AWS_REGION="us-east-1",
            ),
        )
        assert "AKIA-SECRET-VALUE" not in repr(status)
        assert "SHHH" not in repr(status)
        assert "SHHH" not in status.explain()
        assert status.container_env()["AWS_SECRET_ACCESS_KEY"] == "SHHH"

    def test_offline_providers_need_nothing(self) -> None:
        status = default_registry().credentials("offline.tone-tts")
        assert status.needs_nothing
        assert status.explain() == "no credentials required"


# ----------------------------------------------------------------- cache


class TestSynthesisKey:
    """The TTS cache key: what changes the synthesised bytes."""

    def test_the_text_changes_the_key(self) -> None:
        a = SynthesisKey(provider="aws.polly", model="Joanna", text="hello")
        b = SynthesisKey(provider="aws.polly", model="Joanna", text="goodbye")
        assert a.digest != b.digest

    def test_the_voice_changes_the_key(self) -> None:
        a = SynthesisKey(provider="aws.polly", model="Joanna", text="hello")
        b = SynthesisKey(provider="aws.polly", model="Matthew", text="hello")
        assert a.digest != b.digest

    def test_the_audio_format_changes_the_key(self) -> None:
        a = SynthesisKey(provider="p", model="v", text="hi")
        b = SynthesisKey(
            provider="p",
            model="v",
            text="hi",
            audio_format=AudioFormat(sample_rate_hz=8_000),
        )
        assert a.digest != b.digest

    def test_the_speaking_rate_changes_the_key(self) -> None:
        a = SynthesisKey(provider="p", model="v", text="hi", speaking_rate=1.0)
        b = SynthesisKey(provider="p", model="v", text="hi", speaking_rate=0.8)
        assert a.digest != b.digest

    def test_a_schema_bump_invalidates_everything(self) -> None:
        a = SynthesisKey(provider="p", model="v", text="hi")
        b = SynthesisKey(provider="p", model="v", text="hi", schema="voice-blob.2")
        assert a.digest != b.digest

    def test_identical_requests_share_a_key(self) -> None:
        a = SynthesisKey(provider="p", model="v", text="hi")
        b = SynthesisKey(provider="p", model="v", text="hi")
        assert a.digest == b.digest

    def test_a_key_without_a_provider_version_is_marked_incomplete(self) -> None:
        """AWS can change a Polly voice under a stable name."""
        assert not SynthesisKey(provider="p", model="v", text="hi").is_complete
        assert SynthesisKey(
            provider="p", model="v", text="hi", provider_version="2026-03"
        ).is_complete


class TestBlobCacheIsLegitimate:
    """It caches an *input*: the caller's voice."""

    def test_a_miss_then_a_hit(self) -> None:
        cache = BlobCache()
        key = SynthesisKey(provider="p", model="v", text="hello")
        assert cache.get(key) is None
        cache.put(key, b"\x01\x02")
        assert cache.get(key) == b"\x01\x02"
        assert (cache.stats.hits, cache.stats.misses) == (1, 1)

    def test_hit_rate_is_none_rather_than_zero_when_unused(self) -> None:
        """0.0 would read as a failing cache rather than an unasked one."""
        assert BlobCache().stats.hit_rate is None

    def test_membership_does_not_count_as_a_lookup(self) -> None:
        cache = BlobCache()
        key = SynthesisKey(provider="p", model="v", text="hello")
        assert key not in cache
        assert cache.stats.lookups == 0

    def test_it_survives_across_instances_on_disk(self, tmp_path: Path) -> None:
        key = SynthesisKey(provider="p", model="v", text="hello")
        BlobCache(tmp_path).put(key, b"audio")
        assert BlobCache(tmp_path).get(key) == b"audio"

    def test_incomplete_keys_are_counted_for_doctor(self) -> None:
        cache = BlobCache()
        cache.put(SynthesisKey(provider="p", model="v", text="a"), b"x")
        cache.put(
            SynthesisKey(provider="p", model="v", text="b", provider_version="1"),
            b"y",
        )
        assert cache.stats.incomplete_keys == 1


class TestCassetteIsNotAMeasurement:
    def test_off_never_replays(self) -> None:
        cassette = Cassette(mode=CassetteMode.OFF)
        cassette.record("k", [b"a"])
        assert cassette.replay("k") is None
        assert not cassette.used_replay

    def test_replay_serves_and_stamps(self) -> None:
        cassette = Cassette(mode=CassetteMode.REPLAY)
        cassette.record("k", [b"a", b"b"])
        assert cassette.replay("k") == [b"a", b"b"]
        assert cassette.used_replay

    def test_a_miss_is_none_not_an_empty_response(self) -> None:
        """An empty provider response is a real and different event."""
        cassette = Cassette(mode=CassetteMode.REPLAY)
        assert cassette.replay("absent") is None
        assert not cassette.used_replay

    def test_record_returns_the_stream_so_it_can_be_yielded(self) -> None:
        cassette = Cassette(mode=CassetteMode.RECORD)
        assert cassette.record("k", iter([b"a", b"b"])) == [b"a", b"b"]


class TestReplayedRunsCannotBePublished:
    """Enforced in code, because convention is not a control."""

    def test_a_live_run_publishes(self) -> None:
        require_live([Usage(characters=10), Usage(audio_seconds=1.0)])

    def test_a_replayed_call_is_refused(self) -> None:
        with pytest.raises(ReplayedRunError, match="not a measurement"):
            require_live([Usage(characters=10), Usage(replayed=True)])

    def test_a_simulator_cache_hit_is_fine(self) -> None:
        """It replays an input, not the behaviour under measurement."""
        require_live([Usage(cache_hit=True)])

    def test_a_used_cassette_is_refused_even_with_clean_usages(self) -> None:
        cassette = Cassette(mode=CassetteMode.REPLAY)
        cassette.record("k", [b"a"])
        cassette.replay("k")
        with pytest.raises(ReplayedRunError):
            require_live([Usage(characters=1)], cassette=cassette)

    def test_an_unused_cassette_does_not_block(self) -> None:
        require_live([Usage()], cassette=Cassette(mode=CassetteMode.REPLAY))


# ----------------------------------------------------- combining verdicts


class TestCombine:
    def test_nothing_combines_to_satisfied(self) -> None:
        assert combine().verdict is Verdict.SATISFIED

    def test_not_applicable_beats_unattested(self) -> None:
        """A metric blocked for one reason and unverified for another is
        blocked; reporting 'unverified' would invite publishing it."""
        result = combine(
            Applicability(Verdict.UNATTESTED, ("a",)),
            Applicability(Verdict.NOT_APPLICABLE, ("b",)),
        )
        assert result.verdict is Verdict.NOT_APPLICABLE
        assert result.reasons == ("b",)

    def test_unattested_beats_satisfied(self) -> None:
        result = combine(
            Applicability(Verdict.SATISFIED),
            Applicability(Verdict.UNATTESTED, ("unverified",)),
        )
        assert result.verdict is Verdict.UNATTESTED
        assert not result.may_publish

    def test_reasons_accumulate_across_blockers(self) -> None:
        """Fixing one only to meet the next is a three-run debug cycle."""
        result = combine(
            Applicability(Verdict.NOT_APPLICABLE, ("party lacks asr",)),
            Applicability(Verdict.NOT_APPLICABLE, ("provider cannot stream",)),
        )
        assert result.reasons == ("party lacks asr", "provider cannot stream")

    def test_party_and_provider_levels_meet(self) -> None:
        party = Applicability(Verdict.SATISFIED)
        provider = caps(Modality.TTS, streams=False).evaluate(
            ProviderCapability.STREAMS
        )
        assert combine(party, provider).verdict is Verdict.NOT_APPLICABLE


# ------------------------------------------------- measuring devices


def registry_with_two_stts() -> ProviderRegistry:
    registry = default_registry()
    for name, vendor in (("aws.transcribe", "aws"), ("deepgram.nova", "deepgram")):
        registry.register(
            ProviderSpec(
                name=name,
                modality=Modality.STT,
                capabilities=caps(
                    Modality.STT,
                    streams=True,
                    emits_partials=True,
                    word_timings=name == "deepgram.nova",
                ),
                credential_provider=vendor,
                default_model="en-US",
            )
        )
    return registry


class TestAMeasuringDeviceIsNotTheSystemUnderTest:
    def test_the_same_vendor_on_both_sides_is_refused(self) -> None:
        """The divergence would be near-zero regardless of the audio."""
        stack = ReferenceStack(reference_stt="aws.transcribe/en-US")
        with pytest.raises(MeasurementIndependenceError, match="against itself"):
            stack.check_independence(agent_stt="aws.transcribe/en-GB")

    def test_independence_is_at_provider_granularity_not_model(self) -> None:
        """Two models from one vendor share a frontend and its failure modes."""
        stack = ReferenceStack(reference_stt="aws.transcribe/en-US")
        stack.check_independence(agent_stt="deepgram.nova/en-US")

    def test_no_agent_stt_is_not_a_conflict(self) -> None:
        """An S2S agent has no recogniser to collide with."""
        ReferenceStack(reference_stt="aws.transcribe/en-US").check_independence(
            agent_stt=None
        )

    def test_no_reference_is_not_a_conflict(self) -> None:
        ReferenceStack().check_independence(agent_stt="aws.transcribe/en-US")


class TestReferenceStackApplicability:
    def test_an_unconfigured_reference_blocks_divergence(self) -> None:
        verdict = ReferenceStack().applicability(registry_with_two_stts())
        assert verdict.verdict is Verdict.NOT_APPLICABLE
        assert "--metrics-stt" in verdict.reasons[0]

    def test_a_tts_cannot_be_the_reference(self) -> None:
        stack = ReferenceStack(reference_stt="offline.tone-tts/tone")
        verdict = stack.applicability(registry_with_two_stts())
        assert verdict.verdict is Verdict.NOT_APPLICABLE
        assert "is a tts provider" in verdict.reasons[0]

    def test_a_configured_stt_satisfies(self) -> None:
        stack = ReferenceStack(reference_stt="aws.transcribe/en-US")
        assert stack.applicability(registry_with_two_stts()).may_report

    def test_word_timings_abstain_when_the_reference_lacks_them(self) -> None:
        stack = ReferenceStack(reference_stt="aws.transcribe/en-US")
        verdict = stack.word_timing_applicability(registry_with_two_stts())
        assert verdict.verdict is Verdict.NOT_APPLICABLE

    def test_word_timings_are_unattested_when_declared_but_unverified(self) -> None:
        stack = ReferenceStack(reference_stt="deepgram.nova/en-US")
        verdict = stack.word_timing_applicability(registry_with_two_stts())
        assert verdict.verdict is Verdict.UNATTESTED


@pytest.mark.parametrize(
    ("status", "not_found", "kind"),
    [
        (401, None, "auth"),
        (402, None, "auth"),
        (404, None, "malformed"),
        (404, "region", "region"),
        (408, None, "timeout"),
        (429, None, "throttled"),
        (503, None, "stream_broken"),
        (422, None, "malformed"),
    ],
)
def test_http_failure_maps_statuses_alike_for_every_provider(
    status: int, not_found: str | None, kind: str
) -> None:
    from harbor_voice.providers.base import ProviderFailure
    from harbor_voice.providers.base import http_failure

    extra = {"not_found": ProviderFailure(not_found)} if not_found else {}
    assert http_failure(status, **extra) is ProviderFailure(kind)


def test_require_key_takes_the_first_variable_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor_voice.providers.base import ProviderError
    from harbor_voice.providers.base import require_key

    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.setenv("B_KEY", "b")
    assert require_key(None, "A_KEY", "B_KEY", provider="p", model=None) == "b"
    assert require_key("given", "A_KEY", provider="p", model=None) == "given"
    monkeypatch.delenv("B_KEY")
    with pytest.raises(ProviderError, match="A_KEY or B_KEY is not set"):
        require_key(None, "A_KEY", "B_KEY", provider="p", model=None)
    with pytest.raises(ProviderError, match=r"A_KEY is not set \(or sign in\)"):
        require_key(None, "A_KEY", provider="p", model=None, hint="or sign in")


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (Rejected(403), "auth"),
        (Rejected(503), "stream_broken"),
        (Closed(1013), "throttled"),
        (Closed(1000), "stream_broken"),
        (TimeoutError(), "timeout"),
        (ConnectionRefusedError(), "stream_broken"),
        (ConnectionResetError(), "stream_broken"),
        (json.JSONDecodeError("bad", "x", 0), "malformed"),
        (RuntimeError("odd"), "stream_broken"),
    ],
)
def test_socket_failure_maps_one_way_for_every_provider(
    exc: Exception, kind: str
) -> None:
    """A refused or reset connection is a broken stream, not a timeout."""
    from harbor_voice.providers.base import socket_failure

    close_codes = {1013: ProviderFailure.THROTTLED}
    error = socket_failure(exc, provider="p", model="m", close_codes=close_codes)
    assert error.kind is ProviderFailure(kind)
    assert (error.provider, error.model) == ("p", "m")


def test_retry_waits_double_from_one_second_up_to_ten() -> None:
    from harbor_voice.providers.base import retry_wait

    assert [retry_wait(n) for n in range(1, 7)] == [1, 2, 4, 8, 10, 10]


async def test_an_http_session_keeps_one_client_until_closed() -> None:
    from harbor_voice.providers.base import HTTPSession

    session = HTTPSession()
    first = session.client(5.0)
    assert session.client(5.0) is first  # one connection pool per provider
    await session.aclose()
    assert first.is_closed
    assert session.client(5.0) is not first  # reopens if used again


async def test_an_http_session_leaves_a_callers_client_open() -> None:
    import httpx

    from harbor_voice.providers.base import HTTPSession

    given = httpx.AsyncClient()
    session = HTTPSession(given)
    assert session.client(5.0) is given
    await session.aclose()
    assert not given.is_closed
    await given.aclose()
