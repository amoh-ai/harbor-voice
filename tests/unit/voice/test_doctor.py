"""Preflight for meaning."""

from __future__ import annotations

import pytest

from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import Requirement
from harbor_voice.capabilities import TurnControl
from harbor_voice.doctor import Severity
from harbor_voice.doctor import diagnose
from harbor_voice.capabilities import LayerKey
from harbor_voice.perturbation import SeedSet
from harbor_voice.perturbation import profiles
from harbor_voice.perturbation.profiles import PROFILES
from harbor_voice.perturbation.profiles import DuplicateProfile
from harbor_voice.perturbation.profiles import Profile
from harbor_voice.perturbation.profiles import ProfileNotBuildable
from harbor_voice.perturbation.profiles import UnknownProfile
from harbor_voice.perturbation.profiles import get_profile
from harbor_voice.perturbation.profiles import profile_names
from harbor_voice.perturbation.profiles import register_profile
from harbor_voice.perturbation.profiles import summarise

pytestmark = pytest.mark.unit


def s2s(**over: object) -> Capabilities:
    base: dict[str, object] = {
        "duplex": frozenset({Duplex.HALF}),
        "pipeline": Pipeline.S2S,
        "turn_control": TurnControl.NATIVE_VAD,
        "exposes_asr": True,
        "exposes_stages": frozenset(
            {Stage.RESPONSE_START, Stage.RESPONSE_DONE, Stage.ASR_FINAL}
        ),
    }
    base.update(over)
    return Capabilities(**base)


@pytest.fixture
def declared(monkeypatch: pytest.MonkeyPatch) -> Profile:
    """A condition an adapter declares before it can build it."""
    profile = Profile(
        name="needs_corpus",
        layers=(LayerKey.parse("L2.disfluency"),),
        description="disfluent speech from a recorded corpus",
        note="needs a recorded disfluency corpus",
    )
    monkeypatch.setitem(PROFILES, profile.name, profile)
    return profile


class TestProfilesAreNamedConditions:
    def test_unbuilt_layers_are_declared_not_hidden(self, declared: Profile) -> None:
        """'Not supported' and 'not yet' are different claims."""
        assert declared.name in profile_names()
        assert declared.name not in profile_names(buildable_only=True)

    def test_an_unbuildable_profile_says_which_layer_is_missing(
        self, declared: Profile
    ) -> None:
        with pytest.raises(ProfileNotBuildable, match=r"corpus"):
            get_profile(declared.name).pipeline(SeedSet.derive(1))

    def test_core_profiles_are_generic(self) -> None:
        """A benchmark's own conditions belong to its adapter."""
        for name in PROFILES:
            assert not name.startswith(("eva", "fdb", "tau")), name

    def test_an_adapter_registers_its_own(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(profiles, "PROFILES", dict(PROFILES))
        mine = Profile(
            name="bench_b",
            layers=get_profile("background_noise").layers,
            description="a benchmark's condition B",
            build=get_profile("background_noise").build,
        )
        assert register_profile(mine) is mine
        assert get_profile("bench_b") is mine
        with pytest.raises(DuplicateProfile):
            register_profile(mine)
        assert register_profile(mine, replace=True) is mine

    def test_l5_profiles_build_now_that_the_layer_exists(self) -> None:
        """Adding L5 moved these from declared to buildable; nothing else did."""
        for name in ("overlap", "selectivity_probes"):
            pipeline = get_profile(name).pipeline(SeedSet.derive(1))
            assert not pipeline.conversation.is_empty
            assert pipeline.is_clean, "L5 degrades no signal"
            assert not pipeline.is_unperturbed, "but it is not the clean profile"

    def test_clean_is_the_baseline_with_no_layers(self) -> None:
        clean = get_profile("clean")
        assert clean.is_clean
        assert clean.pipeline(SeedSet.derive(1)).as_record().as_dict()["layers"] == {}

    def test_a_buildable_profile_activates_its_declared_layers(self) -> None:
        pipeline = get_profile("telephony").pipeline(SeedSet.derive(3))
        recorded = set(pipeline.as_record().as_dict()["layers"])
        declared = {str(k) for k in get_profile("telephony").layers}
        assert recorded == declared

    def test_an_unknown_profile_lists_what_exists(self) -> None:
        with pytest.raises(UnknownProfile, match="telephony"):
            get_profile("telephny")

    def test_every_layer_of_every_profile_has_a_reader(self) -> None:
        """Perturbation–metric pairing, checked over the published conditions."""
        for name, profile in PROFILES.items():
            unpaired = summarise(profile).unpaired
            assert not unpaired, f"{name} activates {unpaired} with no metric"


class TestDoctorPredictsTheRun:
    def test_it_names_every_requirement_violation_not_just_the_first(self) -> None:
        report = diagnose(
            capabilities=s2s(),
            agent="nova",
            requirement=Requirement(duplex=Duplex.FULL, can_be_interrupted=True),
        )
        errors = report.of(Severity.ERROR)
        assert len(errors) == 2
        assert not report.ok

    def test_an_unbuildable_profile_blocks(self, declared: Profile) -> None:
        report = diagnose(capabilities=s2s(), profile=declared.name)
        blocked = report.of(Severity.BLOCKED)
        assert len(blocked) == 1
        assert "corpus" in blocked[0].detail
        assert not report.ok

    def test_it_predicts_which_metrics_abstain_and_why(self) -> None:
        report = diagnose(capabilities=s2s(), profile="clean")
        assert "turns.overlap_s" in report.will_abstain
        reason = next(
            f
            for f in report.of(Severity.NOT_APPLICABLE)
            if f.subject == "turns.overlap_s"
        )
        assert "duplex=full" in reason.detail

    def test_it_separates_unattested_from_reporting(self) -> None:
        """A number that cannot gate a published claim is still a number."""
        report = diagnose(capabilities=s2s(), profile="clean")
        assert "divergence.wer_said_heard" in report.will_report
        assert "divergence.wer_said_heard" in report.unpublishable

    def test_an_attested_capability_is_publishable(self) -> None:
        report = diagnose(
            capabilities=s2s(attested=frozenset({"exposes_asr"})), profile="clean"
        )
        assert "divergence.wer_said_heard" in report.will_report
        assert "divergence.wer_said_heard" not in report.unpublishable

    def test_an_unknown_metric_is_an_error_not_a_missing_column(self) -> None:
        report = diagnose(capabilities=s2s(), metrics=["divergence.wer_said_heardd"])
        assert any(f.severity is Severity.ERROR for f in report.findings)

    def test_an_unknown_profile_is_an_error(self) -> None:
        report = diagnose(capabilities=s2s(), profile="nope")
        assert not report.ok


# ------------------------------------------------------------------- CLI


class TestTheCli:
    """Through the Typer app, because that is what a person runs."""

    def _run(self, *args: str):  # noqa: ANN202
        from typer.testing import CliRunner

        from harbor_voice.cli import voice_app

        return CliRunner().invoke(voice_app, list(args))

    def test_profiles_lists_conditions_and_their_readers(
        self, declared: Profile
    ) -> None:
        result = self._run("profiles")
        assert result.exit_code == 0
        assert "telephony" in result.stdout
        assert "overlap" in result.stdout
        assert "not yet" in result.stdout

    def test_profiles_can_hide_what_cannot_run_here(self, declared: Profile) -> None:
        result = self._run("profiles", "--buildable")
        assert "telephony" in result.stdout
        assert declared.name not in result.stdout

    def test_caps_separates_attested_claims_from_unverified_ones(self) -> None:
        """The offline set is attested outright — there is no network between
        the claim and the behaviour. A cloud provider is not, until its
        conformance run exists."""
        result = self._run("caps")
        assert result.exit_code == 0

        sections = {}
        current = None
        for line in result.stdout.splitlines():
            if line and not line.startswith(" "):
                current = line.split()[0]
                sections[current] = []
            elif current:
                sections[current].append(line)

        assert "attested" in " ".join(sections["offline.tone-tts"])
        assert "UNATTESTED" in " ".join(sections["aws.nova-sonic"])

    def test_caps_never_prints_a_credential_value(self) -> None:
        """It names variables, it does not read them."""
        result = self._run("caps")
        assert "AWS_ACCESS_KEY_ID" in result.stdout
        assert "AKIA" not in result.stdout

    def test_metrics_lists_what_moves_each_one(self) -> None:
        result = self._run("metrics")
        assert result.exit_code == 0
        assert "divergence.wer_said_heard" in result.stdout
        assert "moved by:" in result.stdout

    def test_doctor_exits_nonzero_on_a_broken_configuration(
        self, declared: Profile
    ) -> None:
        result = self._run("doctor", "--profile", declared.name)
        assert result.exit_code == 1
        assert "BLOCKED" in result.stdout

    def test_doctor_exits_zero_on_a_sound_one(self) -> None:
        result = self._run("doctor", "--profile", "clean")
        assert result.exit_code == 0
        assert "OK" in result.stdout

    def test_doctor_reports_an_unknown_provider_rather_than_crashing(self) -> None:
        result = self._run("doctor", "--profile", "clean", "--stt", "nope/x")
        assert result.exit_code == 1
        assert "ERROR" in result.stdout

    def test_the_app_is_mountable_into_harbors_cli(self) -> None:
        """A sub-app, not a parallel CLI."""
        import typer

        from harbor_voice.cli import voice_app

        assert isinstance(voice_app, typer.Typer)
        host = typer.Typer()
        host.add_typer(voice_app, name="voice")


class TestTheCliWorksWithoutCloudSdks:
    def test_the_registry_helper_degrades_rather_than_raising(self) -> None:
        """Authoring on a laptop with no AWS SDK must still work."""
        from harbor_voice.cli import _registry

        registry = _registry()
        assert "offline.tone-tts" in registry
