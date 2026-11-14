"""The duplex cascade: turns found on an open line, tau2's cascaded turn-taking."""

from __future__ import annotations

import asyncio
from urllib.parse import parse_qsl

import numpy as np
import pytest

from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.capabilities import OptionalOp
from harbor_voice.capabilities import Pipeline
from harbor_voice.cascade import DuplexCascadeTransport
from harbor_voice.cascade.duplex import LINE
from harbor_voice.cascade.transport import CascadeError
from harbor_voice.ids import PartyId
from harbor_voice.providers import EchoLLM
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers import ScriptedLLM
from harbor_voice.providers import ScriptedSTT
from harbor_voice.providers import ToneTTS
from harbor_voice.providers.deepgram import DeepgramCallStream
from harbor_voice.providers.deepgram import DeepgramSTT
from harbor_voice.providers.deepgram.stt import Heard
from harbor_voice.providers.offline import ToolCallingEchoLLM
from harbor_voice.providers.tools import ChatRound
from harbor_voice.providers.tools import MappingExecutor
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.providers.tools import ToolSpec
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.stream import InputSpeech
from harbor_voice.transport.stream import InputTranscript
from harbor_voice.transport.stream import OutputAudio
from harbor_voice.transport.stream import OutputTranscript
from harbor_voice.transport.stream import StreamEvent
from harbor_voice.trial import AgentSpec
from harbor_voice.trial import TrialConfigError
from harbor_voice.trial import VoiceTaskConfig
from harbor_voice.trial import VoiceTrial
from tests.unit.voice.fakes import Closed
from tests.unit.voice.fakes import FakeDeepgram
from tests.unit.voice.fakes import refusing

pytestmark = pytest.mark.unit

AGENT = PartyId("agent")
CALLER = PartyId("caller")
STEP_MS = 200
SAID = ["I need to change a flight", "it is H, K, L"]


def _line(ms: int, *, speech: bool) -> bytes:
    """16 kHz PCM16: a 220 Hz tone for speech, else silence."""
    n = LINE.sample_rate_hz * ms // 1000
    t = np.arange(n) / LINE.sample_rate_hz
    x = 0.3 * np.sin(2 * np.pi * 220 * t) if speech else np.zeros(n)
    return (x * 32767).astype("<i2").tobytes()


class Line:
    """A caller's line into an open cascade, 200 ms at a time, never without
    frames; everything the cascade puts on its stream is collected."""

    def __init__(self, cascade: DuplexCascadeTransport, deepgram: FakeDeepgram) -> None:
        self.cascade = cascade
        self.deepgram = deepgram
        self.events: list[StreamEvent] = []
        self._seq = 0
        self._reader = asyncio.create_task(self._read())

    async def send(self, ms: int, *, speech: bool) -> None:
        for _ in range(ms // STEP_MS):
            await self.cascade.tx(
                AudioFrame(
                    pcm=_line(STEP_MS, speech=speech),
                    seq=self._seq,
                    party=CALLER,
                    t_produced=0.0,
                    fmt=LINE,
                )
            )
            self._seq += 1
            await asyncio.sleep(0.01)

    async def close(self) -> None:
        """Let the last reply finish, then hang up."""
        await asyncio.sleep(0.05)
        await self.cascade.close()
        await self._reader

    async def _read(self) -> None:
        async for event in self.cascade.events():
            self.events.append(event)

    def kinds(self) -> list[str]:
        """Event kinds in order, consecutive audio collapsed to one."""
        kinds = [
            "InterimTranscript"
            if isinstance(e, InputTranscript) and not e.is_final
            else type(e).__name__
            for e in self.events
        ]
        return [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]]


def _deepgram(socket: FakeDeepgram) -> DeepgramSTT:
    return DeepgramSTT(api_key="test-key", connect=socket.connect())


async def _open(cascade: DuplexCascadeTransport) -> Line:
    await cascade.open(SessionConfig(party=AGENT, audio_in=LINE))
    return Line(cascade, getattr(cascade._deepgram._connect, "socket"))


def _cascade(
    llm: object | None = None,
    *,
    socket: FakeDeepgram | None = None,
    **kw: object,
) -> DuplexCascadeTransport:
    socket = socket or FakeDeepgram(list(SAID))
    return DuplexCascadeTransport(
        stt=_deepgram(socket),
        llm=llm or EchoLLM(prefix="you said "),  # type: ignore[arg-type]
        tts=ToneTTS(),
        party=AGENT,
        system_prompt="You are an airline agent.",
        **kw,  # type: ignore[arg-type]
    )


async def test_deepgram_finds_the_turn_while_the_line_keeps_streaming() -> None:
    """SpeechStarted opens the turn; a final with speech_final ends it."""
    line = await _open(_cascade())
    await line.send(1000, speech=False)
    await line.send(800, speech=True)
    # Under Deepgram's endpointing of silence: still the caller's turn.
    await line.send(200, speech=False)
    await asyncio.sleep(0.05)
    assert line.kinds() == ["InputSpeech", "InterimTranscript"]
    assert line.events[0] == InputSpeech(True, 1000, line.events[0].t)

    await line.send(1000, speech=False)
    await line.close()

    # tau2's order: speech start, transcripts, speech end, then the reply.
    assert line.kinds() == [
        "InputSpeech",
        "InterimTranscript",
        "InputTranscript",
        "InputSpeech",
        "OutputTranscript",
        "OutputAudio",
    ]
    final = line.events[2]
    assert final == InputTranscript(None, SAID[0], final.t)
    ended = line.events[3]
    assert isinstance(ended, InputSpeech) and not ended.started
    reply = line.events[4]
    assert reply == OutputTranscript("utt_0", f"you said {SAID[0]}", reply.t)
    assert {e.utterance for e in line.events if isinstance(e, OutputAudio)} == {"utt_0"}
    assert [t.text for t in line.cascade.transcripts] == [SAID[0]]


async def test_the_stream_is_opened_with_tau2s_settings() -> None:
    line = await _open(_cascade())
    await line.send(400, speech=True)
    await line.close()
    socket = line.deepgram
    url, query = socket.url.split("?")
    assert url == "wss://api.deepgram.com/v1/listen"
    assert dict(parse_qsl(query)) == {
        "model": "nova-3",
        "punctuate": "true",
        "smart_format": "false",
        "no_delay": "true",
        "interim_results": "true",
        "encoding": "linear16",
        "vad_events": "true",
        "sample_rate": "16000",
        "channels": "1",
        "endpointing": "350",
        "filler_words": "true",
        "profanity_filter": "false",
        "numerals": "false",
        "mip_opt_out": "false",
        "language": "en-US",
    }
    assert socket.headers == {"Authorization": "Token test-key"}
    # 50 ms messages, as LiveKit's plugin sends them; CloseStream at the end.
    assert {len(chunk) for chunk in socket.audio} == {1600}
    assert socket.controls[0] == {"type": "KeepAlive"}
    assert socket.controls[-1] == {"type": "CloseStream"}
    assert socket.closed


class _Clock(SessionClock):
    """A clock the test moves."""

    def __init__(self) -> None:
        super().__init__()
        self.t = 0.0

    def now(self) -> float:
        return self.t


async def test_a_final_waiting_for_an_end_that_never_comes_is_answered() -> None:
    """tau2's utterance_end_ms fallback: noise keeps Deepgram's VAD open, so
    no speech_final arrives; 2 s after the last transcript the reply starts."""
    socket = FakeDeepgram()
    clock = _Clock()
    line = await _open(_cascade(socket=socket, clock=clock))
    socket.push({"type": "SpeechStarted", "timestamp": 0.5})
    socket.push(socket.results("I need help", final=True, speech_final=False))
    await line.send(200, speech=False)
    await asyncio.sleep(0.02)
    assert line.kinds() == ["InputSpeech", "InputTranscript"]

    clock.t = 1.9  # under 2 s since the transcript: still waiting
    await line.send(200, speech=False)
    await asyncio.sleep(0.02)
    assert not any(isinstance(e, OutputTranscript) for e in line.events)

    clock.t = 2.1
    await line.send(200, speech=False)
    await line.close()
    assert line.kinds() == [
        "InputSpeech",
        "InputTranscript",
        "OutputTranscript",
        "OutputAudio",
    ]
    assert line.cascade.messages[1] == {"role": "user", "content": "I need help"}


async def test_the_fallback_waits_while_a_reply_is_being_generated() -> None:
    """tau2 checks it only while LISTENING, not while the LLM runs."""
    release = asyncio.Event()

    class _Held(EchoLLM):
        async def chat(self, messages, tools):  # type: ignore[override]
            await release.wait()
            return ChatRound(text="ok")

    socket = FakeDeepgram()
    clock = _Clock()
    line = await _open(_cascade(_Held(prefix=""), socket=socket, clock=clock))
    socket.push(socket.results("first", final=True, speech_final=True))
    await line.send(200, speech=False)
    await asyncio.sleep(0.02)  # the first reply is now generating, held
    socket.push(socket.results("second", final=True, speech_final=False))
    await line.send(200, speech=False)
    clock.t = 5.0
    await line.send(200, speech=False)
    await asyncio.sleep(0.02)
    assert [m["content"] for m in line.cascade.messages if m["role"] == "user"] == [
        "first"
    ]
    release.set()
    await asyncio.sleep(0.02)
    await line.send(200, speech=False)
    await line.close()
    assert [m["content"] for m in line.cascade.messages if m["role"] == "user"] == [
        "first",
        "second",
    ]


async def test_the_reply_is_not_cancelled_when_the_caller_speaks_again() -> None:
    """tau2 never interrupts its provider: the consumer drops what is unplayed."""
    line = await _open(_cascade())
    await line.send(400, speech=False)
    await line.send(600, speech=True)
    await line.send(600, speech=False)
    await line.send(600, speech=True)
    await line.send(600, speech=False)
    await line.close()

    replies = [e for e in line.events if isinstance(e, OutputTranscript)]
    assert [r.utterance for r in replies] == ["utt_0", "utt_1"]
    heard = [
        e.text for e in line.events if isinstance(e, InputTranscript) and e.is_final
    ]
    assert heard == ["I need to change a flight", "it is H, K, L"]


async def test_tool_calls_and_results_stay_in_the_conversation() -> None:
    """tau2's ConversationContext keeps every tool exchange for later turns."""
    executor = MappingExecutor(
        specs=[ToolSpec("find_user", "Look a customer up.")],
        handlers={"find_user": lambda user_id: {"text": f"{user_id}: Jane Doe"}},
    )
    llm = ToolCallingEchoLLM(tool="find_user", arguments={"user_id": "u1"})
    line = await _open(_cascade(llm, tools=ToolRouter(executor)))
    await line.send(400, speech=False)
    for _ in range(2):
        await line.send(600, speech=True)
        await line.send(600, speech=False)
    await line.close()

    messages = line.cascade.messages
    assert [m["role"] for m in messages] == [
        "system",
        *["user", "assistant", "tool", "assistant"] * 2,
    ]
    call = messages[2]["tool_calls"][0]
    assert call["function"] == {"name": "find_user", "arguments": '{"user_id": "u1"}'}
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": call["id"],
        "content": "u1: Jane Doe",
    }
    assert messages[4]["content"] == "the tool said: u1: Jane Doe"
    # The round that only called a tool was not spoken.
    replies = [e for e in line.events if isinstance(e, OutputTranscript)]
    assert [r.delta for r in replies] == ["the tool said: u1: Jane Doe"] * 2


class _FailingLLM(EchoLLM):
    """An LLM whose every call fails the same way."""

    def __init__(self, kind: ProviderFailure) -> None:
        super().__init__(prefix="")
        self.kind = kind

    async def chat(self, messages, tools):  # type: ignore[override]
        raise ProviderError(self.kind, "refused", provider="test.llm")


async def test_a_failure_that_cannot_clear_ends_the_call() -> None:
    """A refused request fails every turn alike; the next audio raises it."""
    line = await _open(_cascade(_FailingLLM(ProviderFailure.MALFORMED)))
    await line.send(400, speech=False)
    await line.send(600, speech=True)
    with pytest.raises(ProviderError, match="refused"):
        await line.send(1000, speech=False)
    await line.close()
    assert line.cascade.errors and line.cascade.failure is not None


async def test_a_failure_that_can_clear_lets_the_call_go_on() -> None:
    """A rate limit is recorded and the call carries on, as in tau2."""
    line = await _open(_cascade(_FailingLLM(ProviderFailure.THROTTLED)))
    await line.send(400, speech=False)
    await line.send(600, speech=True)
    await line.send(1000, speech=False)
    await line.close()
    assert line.cascade.errors and line.cascade.failure is None


class _StrictLLM(EchoLLM):
    """Calls a tool on the first words, answers plainly after; refuses a
    history where a tool call has no result, as OpenAI does."""

    def __init__(self) -> None:
        super().__init__(prefix="")
        self.calls = 0

    async def chat(self, messages, tools):  # type: ignore[override]
        answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
        for m in messages:
            for call in m.get("tool_calls") or ():
                if call["id"] not in answered:
                    raise ProviderError(
                        ProviderFailure.MALFORMED, "call without result", provider="t"
                    )
        self.calls += 1
        if self.calls == 1:
            return ChatRound(calls=(ToolInvocation("call_1", "find_user", {}),))
        return ChatRound(text="ok")


class _SlowRouter(ToolRouter):
    """Holds every tool call until released."""

    def __init__(self, executor: MappingExecutor) -> None:
        super().__init__(executor)
        self.release = asyncio.Event()

    async def invoke(self, invocation, *, at_s=0.0):  # type: ignore[override]
        await self.release.wait()
        return await super().invoke(invocation, at_s=at_s)


async def test_a_reply_started_during_a_tool_call_never_sees_it_unanswered() -> None:
    """The caller speaks again while a tool runs; the second reply's history
    must not hold the first reply's call without its result."""
    executor = MappingExecutor(
        specs=[ToolSpec("find_user", "Look a customer up.")],
        handlers={"find_user": lambda: {"text": "Jane Doe"}},
    )
    router = _SlowRouter(executor)
    line = await _open(_cascade(_StrictLLM(), tools=router))
    await line.send(400, speech=False)
    await line.send(600, speech=True)
    await line.send(600, speech=False)  # first reply: a tool call, held
    await line.send(600, speech=True)
    await line.send(600, speech=False)  # second reply starts meanwhile
    router.release.set()
    await line.send(400, speech=False)
    await line.close()

    assert line.cascade.errors == []
    messages = line.cascade.messages
    calls = [i for i, m in enumerate(messages) if m.get("tool_calls")]
    assert calls and all(messages[i + 1]["role"] == "tool" for i in calls)


def test_an_llm_that_cannot_hold_the_conversation_is_refused() -> None:
    with pytest.raises(CascadeError, match="ChatLLM"):
        _cascade(ScriptedLLM(replies=["hi"]))


def test_it_declares_the_event_stream_on_a_full_duplex_line() -> None:
    caps = _cascade().capabilities
    assert caps.pipeline is Pipeline.CASCADE
    assert caps.supports(OptionalOp.EVENT_STREAM)
    assert not caps.supports(OptionalOp.TRUNCATE)
    assert caps.audio_in == LINE


def test_an_stt_other_than_deepgram_is_refused() -> None:
    """Turn detection is Deepgram's, as in tau2: no other recogniser finds turns."""
    with pytest.raises(CascadeError, match="turn detection is Deepgram's"):
        DuplexCascadeTransport(
            stt=ScriptedSTT(turns=["hi"]),
            llm=EchoLLM(),
            tts=ToneTTS(),
            party=AGENT,
        )


async def test_a_dropped_deepgram_stream_ends_the_call() -> None:
    """Nothing is heard after it drops, so the next audio raises."""
    socket = FakeDeepgram(close_with=Closed(1011))
    line = await _open(_cascade(socket=socket))
    await line.send(200, speech=False)
    socket.push(None)
    await asyncio.sleep(0.02)
    with pytest.raises(ProviderError) as ended:
        await line.send(200, speech=False)
    assert ended.value.kind is ProviderFailure.STREAM_BROKEN
    assert line.cascade.failure is ended.value
    assert line.cascade.errors and line.cascade.errors[0].startswith("stt:")
    await line.close()


async def test_a_server_that_closes_mid_call_ends_it_too() -> None:
    socket = FakeDeepgram()
    line = await _open(_cascade(socket=socket))
    socket.push(None)
    await asyncio.sleep(0.02)
    with pytest.raises(ProviderError, match="closed the stream"):
        await line.send(200, speech=False)
    await line.close()


async def test_a_refused_connection_fails_the_open() -> None:
    from tests.unit.voice.fakes import Rejected

    cascade = DuplexCascadeTransport(
        stt=DeepgramSTT(api_key="bad", connect=refusing(Rejected(401))),
        llm=EchoLLM(),
        tts=ToneTTS(),
        party=AGENT,
    )
    with pytest.raises(ProviderError) as refused:
        await cascade.open(SessionConfig(party=AGENT, audio_in=LINE))
    assert refused.value.kind is ProviderFailure.AUTH
    assert not cascade.is_open


async def test_usage_is_metered_per_slot_and_the_ref_names_all_three() -> None:
    class _Counted(EchoLLM):
        model = "gpt-test"

        async def chat(self, messages, tools):  # type: ignore[override]
            return ChatRound(
                text="okay then",
                usage={
                    "prompt_tokens": 120,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 100},
                },
            )

    line = await _open(_cascade(_Counted(prefix="")))
    await line.send(400, speech=False)
    for _ in range(2):
        await line.send(600, speech=True)
        await line.send(600, speech=False)
    await line.close()

    stt, llm, tts = line.cascade.usage_records()
    assert (stt.component, stt.model, stt.seconds) == ("agent.stt", "nova-3", 2.8)
    assert (llm.component, llm.model, llm.requests) == ("agent.llm", "gpt-test", 2)
    assert (llm.input_tokens, llm.cached_input_tokens, llm.output_tokens) == (
        240,
        200,
        14,
    )
    assert (tts.component, tts.characters) == ("agent.tts", 2 * len("okay then"))
    ref = line.cascade.ref
    assert ref.provider == "cascade"
    assert ref.model == "deepgram.stt/nova-3+echo-llm/gpt-test+tone-tts"


class TestCallStream:
    """DeepgramCallStream maps Deepgram's messages as LiveKit's plugin does."""

    @staticmethod
    async def _heard(*messages: dict) -> list[tuple[Heard, str]]:
        socket = FakeDeepgram()
        stream = DeepgramCallStream(
            model="nova-3", fmt=LINE, api_key="k", connect=socket.connect()
        )
        await stream.open()
        for message in messages:
            socket.push(message)
        await asyncio.sleep(0.02)
        await stream.close()
        return [(e.kind, e.text) async for e in stream.events()]

    async def test_a_transcript_without_speech_started_starts_speech(self) -> None:
        socket = FakeDeepgram()
        heard = await self._heard(
            socket.results("hello", final=False, speech_final=False),
            socket.results("hello there", final=True, speech_final=True),
        )
        assert heard == [
            (Heard.SPEECH_STARTED, ""),
            (Heard.INTERIM, "hello"),
            (Heard.FINAL, "hello there"),
            (Heard.SPEECH_ENDED, ""),
        ]

    async def test_a_second_speech_started_and_a_stray_endpoint_are_ignored(
        self,
    ) -> None:
        socket = FakeDeepgram()
        heard = await self._heard(
            socket.results("", final=True, speech_final=True),
            {"type": "SpeechStarted", "timestamp": 0.1},
            {"type": "SpeechStarted", "timestamp": 0.3},
            {"type": "UtteranceEnd"},
            {"type": "Metadata"},
        )
        assert heard == [(Heard.SPEECH_STARTED, "")]

    async def test_it_keeps_the_stream_alive_through_silence(self) -> None:
        socket = FakeDeepgram()
        stream = DeepgramCallStream(
            model="nova-3",
            fmt=LINE,
            api_key="k",
            connect=socket.connect(),
            keepalive_s=0.01,
        )
        await stream.open()
        await asyncio.sleep(0.05)
        await stream.close()
        keepalives = [c for c in socket.controls if c == {"type": "KeepAlive"}]
        assert len(keepalives) >= 3
        assert socket.controls[-1] == {"type": "CloseStream"}

    async def test_an_error_message_fails_the_stream(self) -> None:
        socket = FakeDeepgram()
        stream = DeepgramCallStream(
            model="nova-3", fmt=LINE, api_key="k", connect=socket.connect()
        )
        await stream.open()
        socket.push({"type": "Error", "variant": "bad", "description": "nope"})
        with pytest.raises(ProviderError, match="nope"):
            async for _ in stream.events():
                pass
        with pytest.raises(ProviderError, match="nope"):
            await stream.send(b"\0\0" * 800)
        await stream.close()


class TestBuildAgent:
    """A driver's build_agent: a cascade comes back ready for a full-duplex call."""

    @staticmethod
    def _trial(stt: str = "deepgram.stt/nova-3") -> VoiceTrial:
        spec = AgentSpec.model_validate(
            {
                "pipeline": "cascade",
                "stt": stt,
                "llm": "offline.echo-llm",
                "tts": "offline.tone-tts",
            }
        )
        config = VoiceTaskConfig.model_validate(
            {"call": {"driver": "fake:drive"}, "agent": spec.model_dump()}
        )
        return VoiceTrial(config)

    def test_a_cascade_gets_the_drivers_system_prompt(self) -> None:
        agent = self._trial().build_agent(system_prompt="Follow the policy.")
        assert isinstance(agent, DuplexCascadeTransport)
        assert agent._system_prompt == "Follow the policy."

    def test_speech_to_speech_settings_are_refused(self) -> None:
        with pytest.raises(TrialConfigError, match="settings"):
            self._trial().build_agent(system_prompt="p", settings={"voice": "x"})

    def test_an_stt_other_than_deepgram_is_refused_before_anything_is_built(
        self,
    ) -> None:
        trial = self._trial("offline.scripted-stt")
        with pytest.raises(TrialConfigError, match="must be deepgram.stt"):
            trial.build_agent(system_prompt="p")
