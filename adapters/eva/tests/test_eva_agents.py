"""EVA's caller agents as API payloads: the bodies, the diff, the snapshot.

Offline: REST goes to an ``httpx.MockTransport``; nothing reaches ElevenLabs.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from eva_adapter import agents as a
from harbor_voice.providers.elevenlabs.agents import AgentsAPI

pytestmark = pytest.mark.unit

EVA_TABLE = {
    # docs/experiment_setup.md:142-154, as API fields.
    "conversation_config.tts.model_id": "eleven_v3_conversational",
    "conversation_config.tts.expressive_mode": True,
    "conversation_config.tts.suggested_audio_tags": [],
    "conversation_config.agent.prompt.llm": "gpt-5.1",
    "conversation_config.agent.prompt.prompt": "{{prompt}}",
    "conversation_config.agent.prompt.ignore_default_personality": True,
    "conversation_config.agent.first_message": "",
    # "Interruptible: Disabled": the toggle under the first message.
    "conversation_config.agent.disable_first_message_interruptions": True,
    "conversation_config.asr.user_input_audio_format": "ulaw_8000",
    "conversation_config.turn.turn_eagerness": "eager",
    "conversation_config.turn.turn_timeout": 15,
    "conversation_config.conversation.max_duration_seconds": 600,
    "conversation_config.agent.prompt.built_in_tools.end_call.name": "end_call",
    "conversation_config.agent.prompt.built_in_tools.end_call.type": "system",
}


def _set(tree: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        tree = tree.setdefault(part, {})
    tree[parts[-1]] = copy.deepcopy(value)


def live_agent(label: str) -> dict[str, Any]:
    """A GET body as ElevenLabs returns it for an agent built to spec."""
    spec = a.load_spec()
    live = copy.deepcopy(a.payload(label, spec))
    live["name"] = "Agent-1"
    for path, value in spec["observed_defaults"].items():
        _set(live, path, value)
    live.update(
        {
            "agent_id": "agent_live",
            "version_id": "agtvrsn_1",
            "access_info": {
                "creator_email": "someone@example.com",
                "creator_name": "x",
            },
            "access_permissions": [{"role": "admin"}],
            "phone_numbers": [],
            "whatsapp_accounts": [],
        }
    )
    return live


# ------------------------------------------------------------------ payloads


def test_en_payloads_are_evas_table_and_nothing_else() -> None:
    for label, voice in (
        ("en_f", "g6xIsTj2HwM6VR4iXFCw"),
        ("en_m", "cjVigY5qzO86Huf0OWal"),
    ):
        body = a.payload(label)
        assert set(body) == {"name", "conversation_config"}  # no platform_settings
        leaves = a._leaves(body["conversation_config"], "conversation_config")
        for path, value in EVA_TABLE.items():
            assert leaves[path] == value, path
        assert leaves["conversation_config.tts.voice_id"] == voice
        assert leaves["conversation_config.agent.language"] == "en"
        assert (
            len(leaves) == len(EVA_TABLE) + 4
        )  # + voice, language, description, params


def test_end_call_description_is_evas_verbatim() -> None:
    tool = a.payload("en_m")["conversation_config"]["agent"]["prompt"][
        "built_in_tools"
    ]["end_call"]
    text = tool["description"]
    assert text.startswith(
        "Use this to end the phone call and hang up.\n\nCall this function"
    )
    # The trailing space and the two blank lines are EVA's (experiment_setup.md:180-183).
    assert "additional information. \n\n\nBefore calling this tool" in text
    assert text.endswith("always say a brief goodbye first.\n")
    assert tool["params"] == {"system_tool_type": "end_call"}


def test_the_female_voice_deviation_is_recorded() -> None:
    agent = a.load_spec()["agents"]["en_f"]
    assert agent["eva_voice_id"] == "KpTQ5yzwazQWLkvnK59A"
    assert agent["conversation_config"]["tts"]["voice_id"] == "g6xIsTj2HwM6VR4iXFCw"
    assert "moderation" in agent["deviation"]


def test_french_agents_are_placeholders() -> None:
    agents = a.load_spec()["agents"]
    assert {k: v["status"] for k, v in agents.items()} == {
        "en_f": "active",
        "en_m": "active",
        "fr_f": "placeholder",
        "fr_m": "placeholder",
    }


def test_unknown_label() -> None:
    with pytest.raises(KeyError, match="en_f"):
        a.payload("de_f")


def test_the_spec_holds_no_ids_or_account_fields() -> None:
    text = a.SPEC_PATH.read_text(encoding="utf-8")
    assert not re.search(r"agent_[0-9a-z]{20,}", text)
    assert not re.search(r"agt(brch|vrsn)_", text)
    assert "@" not in text.replace("(@039efa0a)", "")
    for field in ("creator", "access_info", "access_permissions", "workspace"):
        assert field not in text


# ---------------------------------------------------------------------- diff


def test_an_agent_built_to_spec_has_no_findings() -> None:
    assert a.diff("en_f", live_agent("en_f")) == []


def test_diff_flags_mismatch_drift_and_missing() -> None:
    live = live_agent("en_m")
    live["conversation_config"]["agent"]["prompt"]["llm"] = "gpt-4o"
    live["conversation_config"]["agent"]["prompt"]["temperature"] = 0.7
    del live["conversation_config"]["tts"]["voice_id"]
    live["conversation_config"]["turn"]["turn_timeout"] = 15.0  # same value: no finding
    by_path = {f.path: f for f in a.diff("en_m", live)}
    llm = by_path["conversation_config.agent.prompt.llm"]
    assert (llm.kind, llm.expected, llm.live) == ("mismatch", "gpt-5.1", "gpt-4o")
    temp = by_path["conversation_config.agent.prompt.temperature"]
    assert (temp.kind, temp.expected, temp.live) == ("drift", 0.0, 0.7)
    voice = by_path["conversation_config.tts.voice_id"]
    assert (voice.kind, voice.note) == ("mismatch", "missing")
    assert "conversation_config.turn.turn_timeout" not in by_path


def test_the_real_read_back_shape_is_handled() -> None:
    """The live agents carry ``end_call`` twice (built_in_tools and tools[]); the
    spec checks the built-in one."""
    live = live_agent("en_f")
    live["conversation_config"]["agent"]["prompt"]["tools"] = [
        copy.deepcopy(
            live["conversation_config"]["agent"]["prompt"]["built_in_tools"]["end_call"]
        )
    ]
    assert a.diff("en_f", live) == []


# ----------------------------------------------------------------------- API


class Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v1/convai/agents/create":
            return httpx.Response(200, json={"agent_id": "agent_new"})
        agent_id = request.url.path.rsplit("/", 1)[-1]
        label = "en_f" if agent_id.endswith("f") else "en_m"
        return httpx.Response(200, json=live_agent(label))

    def api(self) -> AgentsAPI:
        return AgentsAPI(
            api_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(self))
        )


async def test_create_posts_the_payload() -> None:
    rec = Recorder()
    assert await a.create("en_m", rec.api()) == "agent_new"
    (request,) = rec.requests
    assert request.method == "POST"
    assert request.url.path == "/v1/convai/agents/create"
    assert dict(request.url.params) == {}  # enable_versioning unset, not sent
    assert request.headers["xi-api-key"] == "k"
    assert json.loads(request.content) == a.payload("en_m")


async def test_a_placeholder_is_not_created() -> None:
    rec = Recorder()
    with pytest.raises(ValueError, match="placeholder"):
        await a.create("fr_f", rec.api())
    assert rec.requests == []


async def test_read_gets_the_agent() -> None:
    rec = Recorder()
    live = await a.read("agent_f", rec.api())
    assert live["version_id"] == "agtvrsn_1"
    assert rec.requests[0].method == "GET"
    assert rec.requests[0].url.path == "/v1/convai/agents/agent_f"


async def test_snapshot_stores_configs_without_account_fields(tmp_path: Path) -> None:
    rec = Recorder()
    configs = await a.snapshot(
        {"en_f": "agent_f", "en_m": "agent_m"}, tmp_path, rec.api()
    )
    assert set(configs) == {"en_f", "en_m"}
    stored = json.loads(
        (tmp_path / "elevenlabs_agent_en_f.json").read_text(encoding="utf-8")
    )
    assert "access_info" not in stored and "access_permissions" not in stored
    assert stored["conversation_config"]["tts"]["voice_id"] == "g6xIsTj2HwM6VR4iXFCw"
    summary = json.loads(
        (tmp_path / "elevenlabs_agents.json").read_text(encoding="utf-8")
    )
    assert summary["en_m"]["agent_id"] == "agent_m"
    assert summary["en_m"]["version_id"] == "agtvrsn_1"
    assert summary["en_m"]["findings"] == []
    assert "someone@example.com" not in (
        tmp_path / "elevenlabs_agent_en_m.json"
    ).read_text(encoding="utf-8")


# ------------------------------------------------------------- which agent


def test_env_vars_are_evas() -> None:
    assert a.agent_env_var("en", "F") == "EVA_EN_USER_F"
    assert (
        a.agent_env_var("en", 2) == "EVA_EN_USER_M"
    )  # persona 2 → M (elevenlabs.py:31)
    assert a.agent_env_var("pt-br", 1) == "EVA_PT_BR_USER_F"
    assert a.agent_env_var("en", "M", accent="british") == "EVA_BRITISH_ACCENT_USER_M"
    assert (
        a.agent_env_var("en", "F", behavior="elderly_slow") == "EVA_ELDERLY_SLOW_USER_F"
    )
    assert a.label_for("EN", 1) == "en_f"


def test_agent_id_from_env() -> None:
    assert (
        a.agent_id_from_env("en", "M", environ={"EVA_EN_USER_M": "agent_m"})
        == "agent_m"
    )
    with pytest.raises(ValueError, match="EVA_FR_USER_F"):
        a.agent_id_from_env("fr", "F", environ={})


# ----------------------------------------------------------------------- CLI


def test_cli_show_is_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert a.main_agents(["show"]) == 0
    out = capsys.readouterr().out
    assert "g6xIsTj2HwM6VR4iXFCw" in out and "cjVigY5qzO86Huf0OWal" in out
    assert "1hIScOW98xkqE5ttC10C" not in out  # placeholders are not shown by default


def test_cli_create_needs_yes_and_sends_nothing_without_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def no_api() -> AgentsAPI:
        raise AssertionError("no API client without --yes")

    assert a.main_agents(["create", "--agent", "en_f"], api_factory=no_api) == 2
    assert "add --yes" in capsys.readouterr().err


def test_cli_create_with_yes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rec = Recorder()
    assert (
        a.main_agents(["create", "--agent", "en_m", "--yes"], api_factory=rec.api) == 0
    )
    assert "agent_new" in capsys.readouterr().out
    assert len(rec.requests) == 1


def test_cli_diff_of_a_saved_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps(live_agent("en_f")), encoding="utf-8")
    assert a.main_agents(["diff", "--agent", "en_f", "--live", str(good)]) == 0
    bad_agent = live_agent("en_f")
    bad_agent["conversation_config"]["turn"]["turn_eagerness"] = "patient"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bad_agent), encoding="utf-8")
    assert a.main_agents(["diff", "--agent", "en_f", "--live", str(bad)]) == 1
    assert "turn_eagerness" in capsys.readouterr().out


def test_cli_snapshot_reads_ids_from_evas_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVA_EN_USER_F", "agent_f")
    monkeypatch.setenv("EVA_EN_USER_M", "agent_m")
    rec = Recorder()
    assert a.main_agents(["snapshot", "--out", str(tmp_path)], api_factory=rec.api) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "elevenlabs_agent_en_f.json",
        "elevenlabs_agent_en_m.json",
        "elevenlabs_agents.json",
    ]
