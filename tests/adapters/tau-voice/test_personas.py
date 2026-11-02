"""tau-voice voices: tau2's caller voices made with tau2's own requests."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from tau_voice import personas
from tau_voice.main import PERSONAS
from tau_voice.main import VOICE_ENV
from tau_voice.main import main

TAU2_CHECKOUT = Path(__file__).resolve().parents[3].parent / "tau2-bench"
TAU2_COMMIT = "2174a603f6d014ef94473ffa95957f6ce27100db"


def _tau2_file(path: str) -> str:
    if not (TAU2_CHECKOUT / ".git").exists():
        pytest.skip("no tau2-bench checkout beside this repo to compare with")
    result = subprocess.run(
        ["git", "-C", str(TAU2_CHECKOUT), "show", f"{TAU2_COMMIT}:{path}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode:
        pytest.skip(f"tau2-bench checkout lacks {TAU2_COMMIT[:7]}")
    return result.stdout


def test_the_personas_are_tau2s_verbatim() -> None:
    tree = ast.parse(_tau2_file("src/tau2/data_model/voice_personas.py"))
    tau2 = [
        {
            key.arg: ast.literal_eval(key.value)
            for key in node.value.keywords
            if key.arg
            in ("name", "display_name", "complexity", "short_description", "prompt")
        }
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", "") == "VoicePersona"
    ]
    assert personas.load_spec()["personas"] == tau2


def test_the_design_settings_are_tau2s() -> None:
    tree = ast.parse(_tau2_file("src/tau2/voice/scripts/setup_voices.py"))
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.isupper()
    }  # fmt: skip
    spec = personas.load_spec()
    assert spec["voice_name_prefix"] == constants["VOICE_NAME_PREFIX"]
    assert spec["design"] == {
        "model_id": constants["VOICE_DESIGN_MODEL"],
        "text": constants["SAMPLE_TEXT"],
        "auto_generate_text": False,
        "loudness": constants["VOICE_DESIGN_LOUDNESS"],
        "seed": constants["VOICE_DESIGN_SEED"],
        "guidance_scale": constants["VOICE_DESIGN_GUIDANCE_SCALE"],
    }


def test_every_persona_a_task_reads_has_a_voice_to_make() -> None:
    names = [p["name"] for p in personas.personas()]
    assert tuple(names) == PERSONAS
    assert tuple(personas.env_var(name) for name in names) == VOICE_ENV
    assert [p["name"] for p in personas.personas("control")] == [
        "matt_delaney",
        "lisa_brenner",
    ]
    assert len(personas.personas("regular")) == 5


class FakeElevenLabs:
    """The four endpoints, recording every request."""

    def __init__(self, voices: list[dict[str, str]]) -> None:
        self.voices, self.requests, self.made = list(voices), [], 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        assert request.headers["xi-api-key"] == "k"
        if (request.method, request.url.path) == ("GET", "/v1/voices"):
            return httpx.Response(200, json={"voices": self.voices})
        if request.url.path == "/v1/text-to-voice/design":
            return httpx.Response(200, json={"previews": [
                {"generated_voice_id": f"gen-{len(self.requests)}"},
                {"generated_voice_id": "not-this-one"}]})  # fmt: skip
        if request.url.path == "/v1/text-to-voice":
            self.made += 1
            return httpx.Response(200, json={"voice_id": f"new-{self.made}"})
        if request.method == "DELETE":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)


def test_create_makes_whats_missing_with_tau2s_requests() -> None:
    fake = FakeElevenLabs([
        {"voice_id": "old-matt", "name": "tau2_matt_delaney"},
        {"voice_id": "mine", "name": "someone_else"}])  # fmt: skip
    api = personas.VoicesAPI(api_key="k", transport=httpx.MockTransport(fake))

    ids, failed = personas.create(api, "control", pause_s=0)

    assert failed == []
    assert ids == {"matt_delaney": "old-matt", "lisa_brenner": "new-1"}
    spec = personas.load_spec()
    lisa = personas.personas("control", spec)[1]
    assert fake.requests == [
        ("GET", "/v1/voices", None),
        ("POST", "/v1/text-to-voice/design", {"voice_description": lisa["prompt"],
                                               **spec["design"]}),
        ("POST", "/v1/text-to-voice", {
            "voice_name": "tau2_lisa_brenner",
            "voice_description": lisa["short_description"],
            "generated_voice_id": "gen-2"}),  # the first preview, as tau2 keeps
    ]  # fmt: skip
    assert personas.env_lines(ids) == (
        "TAU2_VOICE_ID_MATT_DELANEY=old-matt\nTAU2_VOICE_ID_LISA_BRENNER=new-1\n"
    )


def test_replace_deletes_the_old_voice_first() -> None:
    fake = FakeElevenLabs([{"voice_id": "old-matt", "name": "tau2_matt_delaney"}])
    api = personas.VoicesAPI(api_key="k", transport=httpx.MockTransport(fake))
    ids, _ = personas.create(api, "control", replace=True, pause_s=0)
    assert ("DELETE", "/v1/voices/old-matt", None) in fake.requests
    assert ids == {"matt_delaney": "new-1", "lisa_brenner": "new-2"}


def test_a_failed_persona_is_reported_and_the_rest_go_on() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/v1/text-to-voice/design"
            and b"Midwest" in request.content
        ):
            return httpx.Response(422, json={"detail": "bad"})
        return FakeElevenLabs([])(request)

    api = personas.VoicesAPI(api_key="k", transport=httpx.MockTransport(handler))
    ids, failed = personas.create(api, "control", pause_s=0)
    assert failed == ["matt_delaney"] and list(ids) == ["lisa_brenner"]


def test_create_writes_nothing_without_yes(monkeypatch, capsys) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(SystemExit) as done:
        main(["voices", "create", "--complexity", "control"])
    assert done.value.code == 2  # no key needed: nothing was sent
    out = capsys.readouterr()
    assert "tau2_matt_delaney" in out.out and "--yes" in out.err
