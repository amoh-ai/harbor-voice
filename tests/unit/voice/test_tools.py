"""Tool calling, and L6 acting on it.

The property most of this file is defending: **a tool call is always
answered.** AWS documents that Nova Sonic waits forever for a result it was
promised, and a cascade's conversation cannot continue without one either, so
an unanswered call is a hung agent rather than a failed tool — and the timing
numbers from that run would be about our injector.
"""

from __future__ import annotations

import asyncio

import pytest

from harbor_voice.capabilities import Pipeline
from harbor_voice.perturbation import SeedSet
from harbor_voice.perturbation.world import LatencyProfile
from harbor_voice.perturbation.world import Outage
from harbor_voice.perturbation.world import ToolFailure
from harbor_voice.perturbation.world import WorldPlan
from harbor_voice.providers import MappingExecutor
from harbor_voice.providers import ToolCallingLLM
from harbor_voice.providers import ToolError
from harbor_voice.providers import ToolInvocation
from harbor_voice.providers import ToolRouter
from harbor_voice.providers import ToolSpec
from harbor_voice.providers.offline import ToolCallingEchoLLM
from harbor_voice.trial import AgentSpec
from harbor_voice.trial import CallerSpec
from harbor_voice.trial import VoiceTaskConfig
from harbor_voice.trial import VoiceTrial

pytestmark = pytest.mark.unit

SPEC = ToolSpec(
    name="lookup_reservation",
    description="Find a booking by confirmation code",
    input_schema={
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
)
CALL = ToolInvocation(
    tool_use_id="t-1", name="lookup_reservation", arguments={"code": "ZK3FFW"}
)


def _executor(**over: object) -> MappingExecutor:
    def lookup(code: str = "ZK3FFW") -> dict[str, object]:
        return {"code": code, "flight": "SK621", "status": "confirmed"}

    handlers: dict[str, object] = {"lookup_reservation": lookup}
    handlers.update(over)
    return MappingExecutor(specs=[SPEC], handlers=handlers)


def _router(world: WorldPlan | None = None, **over: object) -> ToolRouter:
    """A router that does not actually sleep, so tests stay fast.

    A real run *does* sleep: a voice agent's silence is the perturbation.
    """
    return ToolRouter(_executor(**over), world=world, sleep=lambda _s: asyncio.sleep(0))


# --------------------------------------------------------------- declaring


class TestDeclaringATool:
    def test_a_tool_without_a_description_is_refused(self) -> None:
        """Both providers select tools by description; an empty one makes the
        tool present and effectively uncallable."""
        with pytest.raises(ToolError, match="no description"):
            ToolSpec(name="t", description="  ")

    def test_a_tool_without_a_name_is_refused(self) -> None:
        with pytest.raises(ToolError, match="needs a name"):
            ToolSpec(name="", description="d")

    def test_the_bedrock_wire_shape(self) -> None:
        wire = SPEC.as_bedrock()["toolSpec"]
        assert wire["name"] == "lookup_reservation"
        assert wire["inputSchema"]["json"]["required"] == ["code"]

    def test_nova_takes_the_schema_as_a_serialized_string(self) -> None:
        """Pinned from a live failure, not from the docs, which disagree.

        The tool-configuration guide prints `inputSchema.json` as an object;
        the input-events protocol reference types it `"json": "{}"`. Sending
        the object gets `ValidationException: Unable to parse input chunk`,
        naming no field. Bedrock's Converse takes the object. Two vendors,
        two wire formats, even where they look alike.
        """
        import json

        wire = SPEC.as_nova()["toolSpec"]
        assert isinstance(wire["inputSchema"]["json"], str)
        assert json.loads(wire["inputSchema"]["json"])["type"] == "object"

    def test_bedrock_still_takes_the_schema_as_an_object(self) -> None:
        assert isinstance(SPEC.as_bedrock()["toolSpec"]["inputSchema"]["json"], dict)

    def test_an_empty_schema_still_declares_an_object(self) -> None:
        """`inputSchema.json` must be a schema; omitting the type is not one."""
        import json

        bare = ToolSpec(name="ping", description="no arguments")
        assert bare.as_bedrock()["toolSpec"]["inputSchema"]["json"] == {
            "type": "object"
        }
        assert json.loads(bare.as_nova()["toolSpec"]["inputSchema"]["json"]) == {
            "type": "object"
        }

    def test_declared_but_unimplemented_is_refused(self) -> None:
        """The model would be offered a tool that cannot run, then marked
        wrong for calling it."""
        with pytest.raises(ToolError, match="declared but unimplemented"):
            MappingExecutor(specs=[SPEC], handlers={})


class TestParsingWhatTheModelSent:
    def test_arguments_arrive_as_a_json_string(self) -> None:
        call = ToolInvocation.from_json(
            tool_use_id="t", name="x", raw='{"code": "ABC123"}'
        )
        assert call.arguments == {"code": "ABC123"}

    def test_malformed_json_becomes_empty_arguments_not_an_exception(self) -> None:
        """The model asked for something and the call must be answered; an
        answer saying the arguments were unreadable beats a crashed turn."""
        call = ToolInvocation.from_json(tool_use_id="t", name="x", raw='{"code"')
        assert call.arguments == {}

    def test_empty_input_is_no_arguments(self) -> None:
        assert (
            ToolInvocation.from_json(tool_use_id="t", name="x", raw="").arguments == {}
        )


# ----------------------------------------------------------- always answer


class TestACallIsAlwaysAnswered:
    async def test_a_clean_call_returns_the_payload(self) -> None:
        result = await _router().invoke(CALL)
        assert not result.is_error
        assert result.content["flight"] == "SK621"

    async def test_an_injected_failure_still_returns_a_result(self) -> None:
        """Silence would hang the model. The docs are explicit about it."""
        world = WorldPlan(
            error_rate=1.0, failures=(ToolFailure.TIMEOUT,), seeds=SeedSet.derive(1)
        )
        result = await _router(world).invoke(CALL)
        assert result.is_error
        assert result.content["error"] == "timeout"
        assert result.content["retryable"] is True

    async def test_an_outage_still_returns_a_result(self) -> None:
        world = WorldPlan(outages=(Outage(start_s=0.0, duration_s=10.0),))
        result = await _router(world).invoke(CALL, at_s=3.0)
        assert result.is_error
        assert "outage" in result.content["message"]

    async def test_a_tool_that_raises_still_returns_a_result(self) -> None:
        """A broken tool is a fact about the task, not a harness error."""

        def boom(**_: object) -> dict[str, object]:
            raise RuntimeError("salesforce is down")

        result = await _router(lookup_reservation=boom).invoke(CALL)
        assert result.is_error
        assert result.content["message"] == "salesforce is down"

    async def test_a_tool_the_model_invented_still_returns_a_result(self) -> None:
        result = await _router().invoke(
            ToolInvocation(tool_use_id="t", name="teleport")
        )
        assert result.content["error"] == "unknown_tool"

    async def test_the_retryable_bit_travels_with_the_failure(self) -> None:
        """An agent retrying a timeout is behaving; retrying a not_found is
        looping. The result has to say which."""
        world = WorldPlan(
            error_rate=1.0, failures=(ToolFailure.NOT_FOUND,), seeds=SeedSet.derive(1)
        )
        result = await _router(world).invoke(CALL)
        assert result.content["retryable"] is False


class TestWhatL6AddedVersusWhatTheToolTook:
    async def test_injected_delay_is_reported_apart_from_the_total(self) -> None:
        """Folding them together makes every L6 latency run unfalsifiable."""
        world = WorldPlan(
            latency={"*": LatencyProfile(p50_ms=300.0, p99_ms=300.0)},
            seeds=SeedSet.derive(1),
        )
        result = await _router(world).invoke(CALL)
        assert result.injected_delay_ms == pytest.approx(300.0, abs=1.0)
        assert result.latency_ms >= result.injected_delay_ms
        assert result.was_injected

    async def test_a_clean_call_reports_no_injection(self) -> None:
        router = _router()
        result = await router.invoke(CALL)
        assert not result.was_injected
        assert router.records[0].injected_latency_ms is None

    async def test_none_and_zero_are_different_answers(self) -> None:
        """None means L6 was not in the path; 0.0 means it was and added
        nothing."""
        world = WorldPlan(outages=(Outage(start_s=0.0, duration_s=5.0),))
        router = _router(world)
        await router.invoke(CALL, at_s=1.0)
        assert router.records[0].injected_latency_ms == 0.0

    async def test_the_router_keeps_a_record_per_call(self) -> None:
        router = _router()
        await router.invoke(CALL)
        await router.invoke(CALL)
        assert [r.name for r in router.records] == ["lookup_reservation"] * 2
        assert router.records[0].arguments == {"code": "ZK3FFW"}

    async def test_records_can_be_drained_per_turn(self) -> None:
        router = _router()
        await router.invoke(CALL)
        assert len(router.take_records()) == 1
        assert router.records == ()


class TestTheWireFormats:
    async def test_a_success_goes_back_to_bedrock_as_success(self) -> None:
        result = await _router().invoke(CALL)
        block = result.as_bedrock()["toolResult"]
        assert block["status"] == "success"
        assert block["toolUseId"] == "t-1"
        assert block["content"] == [{"json": result.content}]

    async def test_a_failure_goes_back_as_error(self) -> None:
        world = WorldPlan(
            error_rate=1.0, failures=(ToolFailure.UNAVAILABLE,), seeds=SeedSet.derive(1)
        )
        result = await _router(world).invoke(CALL)
        assert result.as_bedrock()["toolResult"]["status"] == "error"

    async def test_nova_takes_a_stringified_json_object(self) -> None:
        import json

        result = await _router().invoke(CALL)
        assert json.loads(result.as_nova_content())["flight"] == "SK621"


# ------------------------------------------------------- declared optional


class TestTheLLMSlotDeclaresWhetherItCanCallTools:
    def test_a_tool_calling_provider_is_recognised(self) -> None:
        assert isinstance(ToolCallingEchoLLM(tool="x"), ToolCallingLLM)

    def test_a_plain_one_is_not(self) -> None:
        from harbor_voice.providers import EchoLLM

        assert not isinstance(EchoLLM(), ToolCallingLLM)

    async def test_a_cascade_refuses_tools_its_llm_cannot_use(self) -> None:
        """Scoring a tool-calling benchmark against an agent that was
        never offered its tools is a number about nothing."""
        from harbor_voice.cascade.transport import CascadeError
        from harbor_voice.cascade.transport import CascadeTransport
        from harbor_voice.ids import PartyId
        from harbor_voice.providers import EchoLLM
        from harbor_voice.providers import ScriptedSTT
        from harbor_voice.providers import ToneTTS
        from harbor_voice.transport import SessionConfig

        transport = CascadeTransport(
            stt=ScriptedSTT(turns=("hello",)),
            llm=EchoLLM(),
            tts=ToneTTS(),
            party=PartyId("agent"),
            tools=_router(),
        )
        await transport.open(SessionConfig(party=PartyId("agent")))
        try:
            with pytest.raises(CascadeError, match="ToolCallingLLM"):
                transport._reply("hello")  # noqa: SLF001 - the guard under test
        finally:
            await transport.close()


# ------------------------------------------------------------- end to end


def _tool_config(**over: object) -> VoiceTaskConfig:
    base: dict[str, object] = {
        "agent": AgentSpec(
            pipeline=Pipeline.CASCADE,
            stt="offline.scripted-stt",
            llm="offline.tool-echo-llm",
            tts="offline.tone-tts",
            config={
                "stt": {"turns": ["what is my reservation status"]},
                "llm": {
                    "tool": "lookup_reservation",
                    "arguments": {"code": "ZK3FFW"},
                },
            },
        ),
        "caller": CallerSpec(source="script", turns=("what is my reservation status",)),
        "profile": "clean",
        "max_turns": 1,
        "idle_s": 0.05,
        "turn_timeout_s": 5.0,
    }
    return VoiceTaskConfig(**(base | over))  # type: ignore[arg-type]


class TestAToolCallThroughAWholeTrial:
    async def test_the_agent_calls_the_tool_and_answers(self) -> None:
        seen: list[str] = []

        def lookup(code: str = "") -> dict[str, object]:
            seen.append(code)
            return {"flight": "SK621", "status": "confirmed"}

        trial = VoiceTrial(
            _tool_config(),
            tools=MappingExecutor(
                specs=[SPEC], handlers={"lookup_reservation": lookup}
            ),
        )
        result = await trial.run()
        assert seen == ["ZK3FFW"], "the tool was never called"
        assert result.turns[0].received, "the agent produced no audio"

    async def test_the_call_lands_in_the_trajectory(self) -> None:
        """`Recorder.tool_call` existed before anything fed it."""
        from harbor_voice.trajectory.model import EventType

        trial = VoiceTrial(_tool_config(), tools=_executor())
        result = await trial.run()
        calls = [e for e in result.trajectory.timeline if e.type is EventType.TOOL_CALL]
        assert [e.meta["name"] for e in calls] == ["lookup_reservation"]

    async def test_l6_latency_reaches_the_trajectory(self) -> None:
        from harbor_voice.trajectory.model import EventType

        trial = VoiceTrial(_tool_config(profile="clean"), tools=_executor())
        trial.tools._world = WorldPlan(  # noqa: SLF001 - no profile ships L6 yet
            latency={"*": LatencyProfile(p50_ms=50.0, p99_ms=50.0)},
            seeds=SeedSet.derive(1),
        )
        result = await trial.run()
        (call,) = [
            e for e in result.trajectory.timeline if e.type is EventType.TOOL_CALL
        ]
        assert call.meta["injected_latency_ms"] == pytest.approx(50.0, abs=5.0)

    async def test_a_trial_without_tools_has_no_router(self) -> None:
        trial = VoiceTrial(_tool_config())
        assert trial.tools is None


class TestNovaSonicDeclaresTools:
    def test_no_router_means_no_tool_fields(self) -> None:
        """An empty `toolConfiguration` is a different declaration from none,
        and there is no reason to find out which way the provider reads it."""
        from harbor_voice.providers.aws.nova_sonic import _tool_config

        assert _tool_config(None) == {}

    def test_a_router_with_no_tools_also_declares_nothing(self) -> None:
        from harbor_voice.providers.aws.nova_sonic import _tool_config

        empty = ToolRouter(MappingExecutor(specs=[], handlers={}))
        assert _tool_config(empty) == {}

    def test_the_prompt_start_fields(self) -> None:
        from harbor_voice.providers.aws.nova_sonic import _tool_config

        fields = _tool_config(_router())
        assert fields["toolUseOutputConfiguration"] == {"mediaType": "application/json"}
        tools = fields["toolConfiguration"]["tools"]
        assert tools[0]["toolSpec"]["name"] == "lookup_reservation"
        assert isinstance(tools[0]["toolSpec"]["inputSchema"]["json"], str)
        assert fields["toolConfiguration"]["toolChoice"] == {"auto": {}}
