"""Running one voice trial.

Standalone: nothing here goes through harbor's `Trial`, and `src/harbor` is
untouched. The question each test asks is whether a task declared in a file
turns into a trajectory and three documents without anyone wiring it by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import Pipeline
from harbor_voice.trial import AgentSpec
from harbor_voice.trial import CallerSpec
from harbor_voice.trial import VoiceTaskConfig
from harbor_voice.trial import VoiceTrial

pytestmark = pytest.mark.unit

TURNS = (
    "I need to reschedule my appointment for next Tuesday",
    "yes that works, thank you",
)

TOML = """
profile = "telephony"
seed = 7
duplex = "half"
max_turns = 3
task = "intake"
trial_id = "t-1"

[agent]
pipeline = "cascade"
stt = "offline.scripted-stt"
llm = "offline.echo-llm"
tts = "offline.tone-tts"

[agent.config.stt]
turns = ["one two three four five", "six seven eight"]

[agent.config.llm]
prefix = "understood: "

[caller]
source = "script"
turns = ["one two three four five", "six seven eight"]
"""


LONG = (
    "I need to reschedule the appointment I booked for next Tuesday "
    "afternoon because something has come up at work and I cannot make it"
)
"""Long enough that the reply outlasts a probe scheduled a second in.

ToneTTS runs at 20 ms a character, so a short script produces a reply that
ends before `fdb_overlap`'s interruption fires at 1.0 s -- the probe is then
correctly recorded as never having landed, and a test asserting the
barge-in metric would be asserting against a run that had no barge-in.
"""


def _duplex_config() -> VoiceTaskConfig:
    """Full duplex, with replies long enough for both probes to land."""
    return _config(
        duplex=Duplex.FULL,
        profile="fdb_overlap",
        idle_s=0.2,
        agent=AgentSpec(
            pipeline=Pipeline.CASCADE,
            stt="offline.scripted-stt",
            llm="offline.echo-llm",
            tts="offline.tone-tts",
            config={"stt": {"turns": [LONG, LONG]}},
        ),
        caller=CallerSpec(source="script", turns=(LONG, LONG)),
    )


def _config(**over: object) -> VoiceTaskConfig:
    base: dict[str, object] = {
        "agent": AgentSpec(
            pipeline=Pipeline.CASCADE,
            stt="offline.scripted-stt",
            llm="offline.echo-llm",
            tts="offline.tone-tts",
            config={"stt": {"turns": list(TURNS)}},
        ),
        "caller": CallerSpec(source="script", turns=TURNS),
        "profile": "telephony",
        "seed": 7,
        "max_turns": 3,
        "idle_s": 0.05,
        "turn_timeout_s": 5.0,
    }
    return VoiceTaskConfig(**(base | over))  # type: ignore[arg-type]


# ------------------------------------------------------------------- config


class TestATaskDeclaresOneShape:
    def test_a_cascade_needs_all_three_slots(self) -> None:
        """A default would attribute someone else's component to the agent."""
        with pytest.raises(ValidationError, match=r"\['tts'\]"):
            AgentSpec(
                pipeline=Pipeline.CASCADE,
                stt="offline.scripted-stt",
                llm="offline.echo-llm",
            )

    def test_an_s2s_agent_may_not_also_name_cascade_slots(self) -> None:
        """They would be silently ignored: an S2S model exposes no seam."""
        with pytest.raises(ValidationError, match="no seam"):
            AgentSpec(
                pipeline=Pipeline.S2S, s2s="aws.nova-sonic", stt="offline.scripted-stt"
            )

    def test_an_s2s_agent_needs_a_provider(self) -> None:
        with pytest.raises(ValidationError, match="no s2s provider"):
            AgentSpec(pipeline=Pipeline.S2S)

    def test_a_script_caller_needs_turns(self) -> None:
        with pytest.raises(ValidationError, match="needs turns"):
            CallerSpec(source="script")

    def test_a_persona_caller_needs_a_brief_and_an_llm(self) -> None:
        with pytest.raises(ValidationError, match="brief and an llm"):
            CallerSpec(source="persona", brief="be terse")

    def test_a_corpus_caller_says_it_is_not_wired_yet(self) -> None:
        """Corpus stays first-class; the loader is not written."""
        with pytest.raises(ValidationError, match="not wired yet"):
            CallerSpec(source="corpus")

    def test_an_unknown_key_is_refused(self) -> None:
        """A typo in voice.toml should not read as a default."""
        with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
            VoiceTaskConfig.model_validate(
                {
                    "agent": {"pipeline": "s2s", "s2s": "aws.nova-sonic"},
                    "caller": {"source": "script", "turns": ["hello there"]},
                    "prophile": "telephony",
                }
            )

    def test_it_loads_from_toml(self, tmp_path: Path) -> None:
        path = tmp_path / "voice.toml"
        path.write_text(TOML, encoding="utf-8")
        config = VoiceTaskConfig.from_toml(path)
        assert config.profile == "telephony"
        assert config.agent.pipeline is Pipeline.CASCADE
        assert config.agent.config["llm"]["prefix"] == "understood: "


# --------------------------------------------------------------- half duplex


class TestAHalfDuplexRun:
    async def test_it_produces_a_trajectory(self) -> None:
        result = await VoiceTrial(_config()).run()
        assert len(result.turns) == 2
        assert result.measurement_valid

    async def test_the_matrix_records_what_actually_ran(self) -> None:
        """Pinned comparison dimensions come from the run, not the ask."""
        result = await VoiceTrial(_config()).run()
        matrix = result.trajectory.config_matrix
        assert matrix is not None
        assert matrix.duplex is Duplex.HALF
        assert matrix.floor_policy == "HalfDuplexFloor"
        assert matrix.profile == "telephony"
        assert matrix.seed_root == 7

    async def test_it_writes_the_three_documents(self, tmp_path: Path) -> None:
        """The layering that keeps harbor's waist intact."""
        result = await VoiceTrial(_config(), out_dir=tmp_path).run()
        assert {p.name for p in result.written} == {
            "measurement.json",
            "reward.json",
            "validity.json",
        }

    async def test_nothing_is_written_without_an_out_dir(self) -> None:
        """A trial that only wants the trajectory should not litter."""
        result = await VoiceTrial(_config()).run()
        assert result.written == ()
        assert result.measurement

    async def test_the_agent_transcript_reaches_the_divergence_metric(self) -> None:
        """A cascade is the only pipeline that can serve said-vs-heard from
        the agent itself; one that hid its transcript would make it
        unavailable."""
        result = await VoiceTrial(_config()).run()
        assert result.reward is not None
        assert "divergence.wer_said_heard" in result.reward

    async def test_provider_config_reaches_the_provider(self) -> None:
        """Without it the scripted STT has no script and the agent is mute."""
        result = await VoiceTrial(_config()).run()
        assert result.turns[0].received, "the agent produced no audio"

    async def test_the_budget_is_honoured(self) -> None:
        result = await VoiceTrial(_config(max_turns=1)).run()
        assert len(result.turns) == 1

    async def test_the_same_seed_runs_the_same_way(self) -> None:
        """Seeding at trial level."""
        first = await VoiceTrial(_config()).run()
        second = await VoiceTrial(_config()).run()
        assert [t.said for t in first.trajectory.turns] == [
            t.said for t in second.trajectory.turns
        ]


# --------------------------------------------------------------- full duplex


class TestAFullDuplexRun:
    async def test_it_gets_a_barge_in_floor(self) -> None:
        """A floor that permits overlap and cannot say whether the agent
        stopped has only made the audio harder."""
        result = await VoiceTrial(_duplex_config()).run()
        matrix = result.trajectory.config_matrix
        assert matrix is not None
        assert matrix.floor_policy == "BargeInFloor"

    async def test_the_duplex_metrics_report(self) -> None:
        result = await VoiceTrial(_duplex_config()).run()
        assert result.reward is not None
        for name in (
            "turns.barge_in_success_rate",
            "timing.yield_latency_s",
            "selectivity.backchannel",
            "turns.overlap_s",
        ):
            assert name in result.reward, name

    async def test_a_cascade_keeps_its_own_seams_under_the_duplex_driver(
        self,
    ) -> None:
        """Inferring two marks from frame arrivals would throw five away and
        lose the observability that makes a cascade the control condition."""
        result = await VoiceTrial(_duplex_config()).run()
        assert result.reward is not None
        assert "timing.asr_final_s" in result.reward
        assert "timing.tts_first_byte_s" in result.reward

    async def test_the_overlap_is_real(self) -> None:
        """The floor is on call time, so pushing bytes fast does not
        report a caller who held the floor for 800 microseconds."""
        result = await VoiceTrial(_duplex_config()).run()
        assert result.reward is not None
        assert result.reward["turns.overlap_s"] > 0.0


class TestWhatTheProfileDecides:
    async def test_a_clean_profile_leaves_the_signal_alone(self) -> None:
        result = await VoiceTrial(_config(profile="clean")).run()
        plan = result.trajectory.perturbation_plan
        assert plan["layers"] == {}

    async def test_a_degraded_one_is_written_into_the_trajectory(self) -> None:
        """The plan is constructed and recorded, not just declared."""
        result = await VoiceTrial(_config(profile="telephony")).run()
        layers = result.trajectory.perturbation_plan["layers"]
        assert set(layers) == {"L4.codec", "L4.frame_loss"}
        assert layers["L4.frame_loss"]["achieved"]

    async def test_an_unknown_profile_is_refused(self) -> None:
        from harbor_voice.perturbation import UnknownProfile

        with pytest.raises(UnknownProfile):
            await VoiceTrial(_config(profile="nope")).run()
