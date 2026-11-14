"""An omni model (audio in, text out) and a TTS as one agent."""

from __future__ import annotations

import base64
import io
import json
import wave

import httpx
import pytest

from harbor_voice.audio.stages import Stage
from harbor_voice.cascade import CascadeAdapter
from harbor_voice.cascade.omni import OmniCascadeTransport
from harbor_voice.cascade.transport import CascadeError
from harbor_voice.conformance import silence_stimulus
from harbor_voice.capabilities import Pipeline
from harbor_voice.ids import PartyId
from harbor_voice.providers import ToneTTS
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.registry import provider_registry
from harbor_voice.providers.selfhosted.omni import OmniLLM
from harbor_voice.providers.selfhosted.omni import OmniTurn
from harbor_voice.transport import SessionConfig
from harbor_voice.trial import AgentSpec
from harbor_voice.trial import VoiceTaskConfig
from harbor_voice.trial import VoiceTrial
from harbor_voice.trial import required_env

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
AGENT = PartyId("agent")


def _completion(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _omni(handler, **kw) -> tuple[OmniLLM, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return OmniLLM(model="m", base_url="http://omni/v1", client=client, **kw), seen


class TestTheOmniModel:
    async def test_a_reply_that_is_not_json_is_malformed(self) -> None:
        """A proxy in front of a server can answer 200 with an HTML page."""
        omni, _ = _omni(lambda _: httpx.Response(200, text="<html>loading</html>"))
        with pytest.raises(ProviderError) as caught:
            await omni.reply(b"\x00\x00" * 10)
        assert caught.value.kind is ProviderFailure.MALFORMED
        assert "loading" in str(caught.value)

    async def test_it_sends_the_audio_as_wav_and_reads_the_text(self) -> None:
        omni, seen = _omni(
            lambda _: httpx.Response(200, json=_completion(" Sure. ")),
            api_key="k",
        )
        reply = await omni.reply(b"\x01\x00" * 2400)
        assert reply.text == "Sure."
        (request,) = seen
        assert request.url == "http://omni/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer k"
        body = json.loads(request.content)
        assert body["messages"][0]["role"] == "system"
        (part,) = body["messages"][1]["content"]
        assert part["type"] == "input_audio"
        assert part["input_audio"]["format"] == "wav"
        with wave.open(io.BytesIO(base64.b64decode(part["input_audio"]["data"]))) as w:
            assert (w.getnchannels(), w.getsampwidth()) == (1, 2)
            assert w.getnframes() == 2400

    async def test_earlier_turns_go_back_as_audio_then_reply(self) -> None:
        omni, seen = _omni(lambda _: httpx.Response(200, json=_completion("Ok.")))
        await omni.reply(
            b"\x00\x00" * 10, [OmniTurn(audio=b"\x00\x00" * 10, reply="Hi.")]
        )
        roles = [m["role"] for m in json.loads(seen[0].content)["messages"]]
        assert roles == ["system", "user", "assistant", "user"]
        assert "authorization" not in seen[0].headers

    async def test_an_http_error_names_the_provider(self) -> None:
        omni, _ = _omni(lambda _: httpx.Response(401, text="no"))
        with pytest.raises(ProviderError) as info:
            await omni.reply(b"\x00\x00")
        assert info.value.kind is ProviderFailure.AUTH

    def test_both_routes_are_registered_as_omni(self, monkeypatch) -> None:
        monkeypatch.setenv("SELFHOSTED_OMNI_BASE_URL", "http://omni")
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        registry = provider_registry()
        for spec in ("selfhosted.omni/Qwen/Qwen3-Omni", "openrouter.omni/qwen/omni"):
            assert isinstance(registry.create(spec, modality=Modality.OMNI), OmniLLM)


class _FakeOmni:
    name = "fake.omni"
    model = "m"

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, int]] = []

    async def reply(self, audio, history=()):
        from harbor_voice.providers.selfhosted.omni import OmniReply

        self.calls.append((audio, len(history)))
        return OmniReply(text=f"reply {len(self.calls)}", latency_s=0.0, usage={})


class TestTheOmniCascade:
    async def test_the_omni_reply_is_spoken_and_remembered(self) -> None:
        omni = _FakeOmni()
        t = OmniCascadeTransport(
            omni=omni,  # ty: ignore[invalid-argument-type]
            tts=ToneTTS(),
            party=AGENT,
            input_idle_s=0.05,
        )
        a = CascadeAdapter(t, party=AGENT, idle_s=0.1, turn_timeout_s=3.0)
        await a.open(SessionConfig(party=AGENT))
        first = await a.call(silence_stimulus(seconds=0.1, party=CALLER))
        await a.call(silence_stimulus(seconds=0.1, party=CALLER))
        await a.close()
        assert first.received
        assert [n for _, n in omni.calls] == [0, 1]
        assert t.said == ["reply 1", "reply 2"]
        # No ASR seam, so no HEARD side: not an empty line per turn.
        assert t.heard == []
        assert first.stage_at(Stage.LLM_DONE) is not None
        assert first.stage_at(Stage.ASR_FINAL) is None
        assert t.ref.model == "fake.omni/m+tone-tts"

    def test_it_declares_no_asr_seam(self) -> None:
        t = OmniCascadeTransport(
            omni=_FakeOmni(),  # ty: ignore[invalid-argument-type]
            tts=ToneTTS(),
            party=AGENT,
        )
        caps = t.capabilities
        assert caps.pipeline is Pipeline.CASCADE
        assert not caps.exposes_asr
        assert Stage.ASR_FINAL not in caps.exposes_stages

    def test_tools_are_refused(self) -> None:
        with pytest.raises(CascadeError, match="no tools"):
            OmniCascadeTransport(
                omni=_FakeOmni(),  # ty: ignore[invalid-argument-type]
                tts=ToneTTS(),
                party=AGENT,
                tools=object(),
            )


class TestTheAgentShape:
    def test_omni_and_tts_is_a_cascade(self) -> None:
        spec = AgentSpec(pipeline=Pipeline.CASCADE, omni="selfhosted.omni/m", tts="t")
        assert spec.omni == "selfhosted.omni/m"

    @pytest.mark.parametrize(
        "extra",
        [{"stt": "s"}, {"llm": "l"}],
    )
    def test_stt_or_llm_beside_omni_is_refused(self, extra: dict) -> None:
        with pytest.raises(ValueError, match="omni"):
            AgentSpec(pipeline=Pipeline.CASCADE, omni="o", tts="t", **extra)

    def test_omni_needs_a_tts(self) -> None:
        with pytest.raises(ValueError, match="tts"):
            AgentSpec(pipeline=Pipeline.CASCADE, omni="o")

    def test_s2s_with_omni_is_refused(self) -> None:
        with pytest.raises(ValueError):
            AgentSpec(pipeline=Pipeline.S2S, s2s="x", omni="o")

    def test_a_driven_benchmark_refuses_it(self) -> None:
        config = VoiceTaskConfig.model_validate(
            {
                "agent": {
                    "pipeline": "cascade",
                    "omni": "selfhosted.omni/m",
                    "tts": "offline.tone-tts",
                },
                "caller": {"source": "script", "turns": ["hi"]},
            }
        )
        with pytest.raises(ValueError, match="own call loop"):
            VoiceTrial(config).build_agent()

    def test_the_omni_slot_s_keys_reach_the_sandbox(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        spec = AgentSpec(
            pipeline=Pipeline.CASCADE,
            omni="openrouter.omni/qwen/omni",
            tts="offline.tone-tts",
        )
        assert required_env(spec).get("OPENROUTER_API_KEY") == "or-key"
