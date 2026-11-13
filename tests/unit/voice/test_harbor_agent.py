"""VoiceCallAgent: a voice call as a harbor agent, for harbor's job runner."""

from __future__ import annotations

import contextlib
import json
import shlex
import logging
import types
from pathlib import Path

import pytest

from harbor_voice import harbor_agent
from harbor_voice import install
from harbor_voice.harbor_agent import VoiceCallAgent
from harbor_voice.harbor_agent import VoiceInfrastructureError
from harbor_voice.harbor_agent import call_spec
from harbor_voice.trial import AgentSpec
from harbor_voice.trial import VoiceTaskConfig

pytestmark = pytest.mark.unit

SPEC = """\
task = "fake/1"
seed = 7

[call]
driver = "fake_bench:drive"

[call.config]
domain = "x"
seed = 7
"""

INSTRUCTION = f"# fake task\n\nSome prose.\n\n```toml\n{SPEC}```\n"


class _Exec:
    def __init__(
        self, return_code: int = 0, stderr: str = "", stdout: str = ""
    ) -> None:
        self.return_code, self.stdout, self.stderr = return_code, stdout, stderr


DRIVE_OUTPUT = (
    "harbor-voice: attempt 0: started, seed 7\n"
    "some other line\n"
    "harbor-voice: attempt 0: reward 1.0 · hallucination no · agent_stop · 120 s audio\n"
    '{"reward": 1.0}\n'
)


class _Environment:
    def __init__(self, drive_code: int = 0, streams: bool = False) -> None:
        self.calls: list[tuple] = []
        self.uploaded: dict[str, str] = {}
        self._drive_code = drive_code
        self._streams = streams
        self._callback = None

    @contextlib.contextmanager
    def scoped_output_callback(self, callback):  # noqa: ANN001, ANN201 - harbor's hook
        self._callback = callback
        try:
            yield
        finally:
            self._callback = None

    async def exec(self, command: str, **kw: object) -> _Exec:
        self.calls.append(("exec", command))
        if command.startswith("harbor-voice drive"):
            if self._streams and self._callback is not None:
                # Docker/EC2 stream output in chunks that split lines anywhere.
                for i in range(0, len(DRIVE_OUTPUT), 7):
                    await self._callback(DRIVE_OUTPUT[i : i + 7], "stdout")
            return _Exec(self._drive_code, stderr="provider down", stdout=DRIVE_OUTPUT)
        return _Exec()

    async def upload_file(self, source: Path, target: str) -> None:
        self.calls.append(("upload", target))
        if not target.endswith(".whl"):
            self.uploaded[target] = Path(source).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _no_wheel_builds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def package_for(module: str, out_dir: Path) -> install.Wheel:
        name = "harbor" if module == "harbor_voice" else module
        return install.Wheel(name, "0.0", tmp_path / f"{name}-0.0.whl")

    monkeypatch.setattr(install, "package_for", package_for)
    monkeypatch.setattr(install, "distribution_of", lambda module: "harbor")
    monkeypatch.setattr(harbor_agent, "_packages", harbor_agent._Packages())


def _keys(environment: _Environment) -> dict[str, str] | None:
    """The credentials the call was given, from the uploaded key file."""
    text = environment.uploaded.get(harbor_agent.KEYS_IN_ENV)
    if text is None:
        return None
    return {
        k: shlex.split(v)[0]
        for k, _, v in (line.partition("=") for line in text.splitlines())
    }


def _agent(tmp_path: Path, **kwargs: object) -> VoiceCallAgent:
    return VoiceCallAgent(logs_dir=tmp_path / "agent", **kwargs)  # type: ignore[arg-type]


def test_the_instruction_carries_the_call_spec() -> None:
    assert call_spec(INSTRUCTION) == SPEC.rstrip("\n")
    with pytest.raises(VoiceInfrastructureError, match="toml"):
        call_spec("just words")


def test_the_model_is_the_agent_under_test(tmp_path: Path) -> None:
    agent = _agent(tmp_path, model_name="openai.realtime/gpt-realtime-1.5")
    assert json.loads(agent.agent_spec_json())["s2s"] == (
        "openai.realtime/gpt-realtime-1.5"
    )

    cascade = AgentSpec.model_validate({"pipeline": "s2s", "s2s": "offline.echo-s2s"})
    agent = _agent(tmp_path, agent_json=cascade.model_dump_json())
    assert json.loads(agent.agent_spec_json())["s2s"] == "offline.echo-s2s"

    with pytest.raises(ValueError, match="no agent under test"):
        _agent(tmp_path).agent_spec_json()


@pytest.mark.parametrize(
    ("declared", "with_extra"),
    [
        (["voice"], True),  # voice dependencies in an extra
        ([], False),  # voice dependencies in the core, no extra
    ],
)
async def test_install_puts_harbor_voice_into_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declared: list[str],
    with_extra: bool,
) -> None:
    metadata = types.SimpleNamespace(get_all=lambda key: declared)
    monkeypatch.setattr(install.importlib.metadata, "metadata", lambda dist: metadata)
    environment = _Environment()
    await _agent(tmp_path, model_name="offline.echo-s2s").install(environment)  # type: ignore[arg-type]
    (command,) = [
        c[1] for c in environment.calls if c[0] == "exec" and "uv pip" in c[1]
    ]
    assert "harbor-0.0.whl" in command
    assert ("harbor-0.0.whl[voice]" in command) is with_extra


@pytest.mark.parametrize(
    ("declared", "suffix"),
    [
        (["live", "voice"], "[voice,live]"),
        (["live"], "[live]"),  # voice dependencies in the core
        ([], ""),  # an older harbor-voice without the extra
    ],
)
async def test_install_brings_the_extra_the_agent_needs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declared: list[str],
    suffix: str,
) -> None:
    """OpenAI Live's WebRTC is the ``live`` extra; ``main`` gets it with Live."""
    metadata = types.SimpleNamespace(get_all=lambda key: declared)
    monkeypatch.setattr(install.importlib.metadata, "metadata", lambda dist: metadata)
    environment = _Environment()
    agent = _agent(tmp_path, model_name="openai.live/gpt-live-1")
    assert agent.agent_extras() == ["live"]
    await agent.install(environment)  # type: ignore[arg-type]
    (command,) = [
        c[1] for c in environment.calls if c[0] == "exec" and "uv pip" in c[1]
    ]
    assert command.rstrip("'").endswith(f"harbor-0.0.whl{suffix}")
    assert _agent(tmp_path, model_name="offline.echo-s2s").agent_extras() == []


async def test_run_installs_the_driver_and_drives_the_call(tmp_path: Path) -> None:
    environment = _Environment()
    agent = _agent(tmp_path, model_name="offline.echo-s2s", seed=626729)
    await agent.run(INSTRUCTION, environment, context=None)  # type: ignore[arg-type]

    execs = [c[1] for c in environment.calls if c[0] == "exec"]
    assert any("fake_bench-0.0.whl" in c and "[voice]" not in c for c in execs)
    drive = next(c for c in execs if c.startswith("harbor-voice drive"))
    assert "--seed 626729" in drive and "offline.echo-s2s" in drive
    assert environment.uploaded["/task/voice.toml"] == SPEC.rstrip("\n")


@pytest.mark.parametrize(
    ("model", "key"),
    [
        ("google.live/gemini-live-2.5-flash", "GEMINI_API_KEY"),
        ("xai.realtime/grok-voice", "XAI_API_KEY"),
    ],
    ids=["gemini", "grok"],
)
async def test_the_call_gets_the_agents_own_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, key: str
) -> None:
    """No task names Gemini's or xAI's key, so the agent passes it in."""
    for name in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(key, "test-key")
    environment = _Environment()
    await _agent(tmp_path, model_name=model).run(INSTRUCTION, environment, context=None)  # type: ignore[arg-type]
    assert _keys(environment) == {key: "test-key"}


async def test_gemini_uses_its_key_in_the_container_not_vertex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vertex signs in with a local credentials file no sandbox has."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    environment = _Environment()
    agent = _agent(tmp_path, model_name="google.live/gemini-live-2.5-flash")
    await agent.run(INSTRUCTION, environment, context=None)  # type: ignore[arg-type]
    env = _keys(environment)
    assert env is not None
    assert env["GEMINI_API_KEY"] == "test-key"
    assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "false"


async def test_a_cascade_gets_each_providers_key_including_litellms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """litellm's key depends on the model it routes to (OpenRouter here)."""
    for key in ("CARTESIA_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, f"{key.lower()}-test")
    spec = AgentSpec(
        pipeline="cascade",
        stt="cartesia.stt",
        llm="litellm.llm/openrouter/openai/gpt-5.6-luna",
        tts="openrouter.tts/elevenlabs/eleven-flash-v2.5",
    )
    environment = _Environment()
    agent = _agent(tmp_path, agent_json=spec.model_dump_json())
    await agent.run(INSTRUCTION, environment, context=None)  # type: ignore[arg-type]
    env = _keys(environment)
    assert env is not None
    assert env["CARTESIA_API_KEY"] == "cartesia_api_key-test"
    assert env["OPENROUTER_API_KEY"] == "openrouter_api_key-test"


async def test_settings_naming_this_machine_stay_out_of_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile name or a file path here means nothing in a container."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_PROFILE", "ml")
    environment = _Environment()
    await _agent(tmp_path, model_name="aws.nova-sonic").run(
        INSTRUCTION, environment, context=None
    )  # type: ignore[arg-type]
    env = _keys(environment)
    assert env is not None
    assert env["AWS_ACCESS_KEY_ID"] == "id"
    assert "AWS_PROFILE" not in env


async def test_any_gemini_key_switches_a_vertex_machine_to_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    environment = _Environment()
    agent = _agent(tmp_path, model_name="google.live/gemini-live-2.5-flash")
    await agent.run(INSTRUCTION, environment, context=None)  # type: ignore[arg-type]
    assert (_keys(environment) or {})["GOOGLE_GENAI_USE_VERTEXAI"] == "false"


async def test_keys_reach_the_call_as_a_file_never_a_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``docker exec -e`` would show the key in the host's process list."""
    monkeypatch.setenv("XAI_API_KEY", "secret-xai")
    environment = _Environment()
    agent = _agent(tmp_path, model_name="xai.realtime/grok-voice")
    await agent.run(INSTRUCTION, environment, context=None)  # type: ignore[arg-type]
    commands = [c[1] for c in environment.calls if c[0] == "exec"]
    assert not any("secret-xai" in c for c in commands)
    drive = next(c for c in commands if "harbor-voice drive" in c)
    assert f"rm -f {harbor_agent.KEYS_IN_ENV}" in drive
    assert _keys(environment) == {"XAI_API_KEY": "secret-xai"}


async def test_an_agent_needing_no_key_passes_no_env(tmp_path: Path) -> None:
    environment = _Environment()
    await _agent(tmp_path, model_name="offline.echo-s2s").run(
        INSTRUCTION, environment, context=None
    )  # type: ignore[arg-type]
    assert _keys(environment) is None


async def test_a_call_that_did_not_run_is_retryable_infra(tmp_path: Path) -> None:
    agent = _agent(tmp_path, model_name="offline.echo-s2s")
    with pytest.raises(VoiceInfrastructureError, match="provider down"):
        await agent.run(INSTRUCTION, _Environment(drive_code=1), context=None)  # type: ignore[arg-type]


def test_the_outcome_lands_in_the_trial_context(tmp_path: Path) -> None:
    agent = _agent(tmp_path, model_name="offline.echo-s2s")
    call = tmp_path / "agent" / "call"
    call.mkdir(parents=True)
    (call / "outcome.json").write_text(json.dumps({"summary": {"reward": 1.0}}))
    context = types.SimpleNamespace(metadata=None)
    agent.populate_context_post_run(context)  # type: ignore[arg-type]
    assert context.metadata == {"call": {"reward": 1.0}}


async def test_call_settings_reach_the_driver_through_the_drive_command(
    tmp_path: Path,
) -> None:
    import tomllib

    environment = _Environment()
    agent = _agent(
        tmp_path, model_name="offline.echo-s2s", call_config={"allow_cascade": True}
    )
    await agent.run(INSTRUCTION, environment, context=None)  # type: ignore[arg-type]

    drive = next(
        c[1] for c in environment.calls if c[1].startswith("harbor-voice drive")
    )
    argv = shlex.split(drive)
    overrides = json.loads(argv[argv.index("--call-config") + 1])
    config = VoiceTaskConfig.model_validate(tomllib.loads(SPEC))
    applied = config.with_call_config(overrides)
    assert applied.call is not None
    assert applied.call.config == {"domain": "x", "seed": 7, "allow_cascade": True}


def test_with_seed_moves_the_task_and_its_driver() -> None:
    import tomllib

    config = VoiceTaskConfig.model_validate(tomllib.loads(SPEC)).with_seed(9)
    assert config.seed == 9
    assert config.call is not None and config.call.config["seed"] == 9


@pytest.mark.parametrize("streams", [True, False], ids=["docker-ec2", "daytona"])
async def test_driver_progress_reaches_the_trial_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, streams: bool
) -> None:
    caplog.set_level(logging.INFO)
    agent = _agent(tmp_path, model_name="offline.echo-s2s")
    await agent.run(INSTRUCTION, _Environment(streams=streams), context=None)  # type: ignore[arg-type]
    lines = [r.getMessage() for r in caplog.records]
    assert "attempt 0: started, seed 7" in lines
    assert any(line.startswith("attempt 0: reward 1.0") for line in lines)
    assert not any("some other line" in line for line in lines)
    assert sum(line.startswith("attempt 0: started") for line in lines) == 1
    assert any(line.startswith("call: fake/1 · seed 7") for line in lines)
