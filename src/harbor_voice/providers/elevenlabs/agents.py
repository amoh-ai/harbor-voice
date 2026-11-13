"""ElevenLabs Agents as a :class:`~harbor_voice.transport.base.VoiceTransport`.

An ElevenLabs agent is a whole voice party run by ElevenLabs: its speech
recognition, its LLM, its voice and its turn-taking. harbor-voice is a client
of it, as any SDK user is. EVA's caller *is* such an agent, so this module
speaks to it the way EVA does, and it can equally sit on the agent side of a
call later: nothing here is caller-specific.

The protocol is the Python SDK's, version 2.53.0 — EVA's pin — spoken
directly with ``websockets`` and ``httpx`` rather than through the SDK. The
SDK is a thread-per-conversation client with callbacks; harbor's transports
are asyncio with one ordered event stream. The wire is what has to match, so
every message below is built exactly as
``elevenlabs/conversational_ai/conversation.py`` builds it (cited by line),
and the tests compare the bytes:

- Connect. ``GET /v1/convai/conversation/get-signed-url?agent_id=…``,
  then ``source=python_sdk&version=2.53.0`` appended to the signed URL
  (``_get_signed_url``, :435-445); a websocket with a 16 MiB message limit
  (``_run``, :823). Then one ``conversation_initiation_client_data`` message
  with the dynamic variables (``_create_initiation_message``, :460-473).
  :meth:`open` returns after that message is sent, as the SDK starts sending
  audio then — it does not wait for the server's metadata.
- Send. Audio as ``{"user_audio_chunk": <base64>}``, one message per
  chunk handed in (``input_callback``, :830-843). A failed send ends the
  session rather than raising, as the SDK's does. ``{"type": "user_activity"}``
  (:755-771), ``contextual_update`` (:773-793) and ``user_message`` (:736-753)
  are each one call.
- Receive (``_handle_message_core``, :475-548), surfaced by
  :meth:`events` in arrival order:

  - ``conversation_initiation_metadata`` sets :attr:`conversation_id`;
  - ``audio`` → :class:`~harbor_voice.transport.stream.OutputAudio`, dropped
    when its ``event_id`` is at or below the last interruption's, as the SDK
    drops it;
  - ``agent_response`` → :class:`AgentResponse`, ``agent_response_correction``
    → :class:`AgentResponseCorrection`, ``user_transcript`` →
    :class:`UserTranscript`, each ``.strip()``-ed as the SDK strips them;
  - ``interruption`` → :class:`Interruption`;
  - ``ping`` is answered with ``pong`` at once, before the next message is
    read — the SDK's timing — and is not surfaced;
  - ``client_tool_call`` → :class:`ClientToolCall`, and answered as the SDK
    answers it: by ``tools`` when given one, otherwise with the SDK's
    ``Tool '…' is not registered`` error result;
  - anything else is ignored, as the SDK ignores it, and counted in
    :attr:`ignored` so a recorded session shows what was skipped.

  A message the SDK would fail on (a missing field, bad JSON) ends the
  session, as the SDK's receive loop does. Every end yields one
  :class:`SessionEnded` and then the stream stops.
- After the call (REST): :meth:`conversation_record` polls
  ``GET /v1/convai/conversations/{id}`` on EVA's schedule; the record says
  how the call ended and what it cost (:func:`usage_records`).
  :meth:`fetch_audio` gets the server's recording.

The REST requests carry the SDK's headers and no retries, as SDK 2.53 sends
them by default (``max_retries`` 0): a failed poll or fetch is an exception
for the caller of this module to judge, as it is for EVA.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import urllib.parse
from collections import Counter
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from typing import override

import httpx

from harbor_voice._deps import websockets as _websockets
from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Capabilities
from harbor_voice.capabilities import Duplex
from harbor_voice.capabilities import OptionalOp
from harbor_voice.capabilities import Pipeline
from harbor_voice.capabilities import TurnControl
from harbor_voice.ids import PartyId
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import HTTPSession
from harbor_voice.providers.base import PooledHTTP
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.providers.base import ProviderRef
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.tools import ToolInvocation
from harbor_voice.providers.tools import ToolRouter
from harbor_voice.trajectory.model import UsageRecord
from harbor_voice.transport.base import SessionConfig
from harbor_voice.transport.base import TransportNotConnectedError
from harbor_voice.transport.base import VoiceTransport
from harbor_voice.transport.stream import OutputAudio

NAME = "elevenlabs.agents"
API_BASE = "https://api.elevenlabs.io"

SDK_VERSION = "2.53.0"
"""The ElevenLabs Python SDK whose wire protocol this reproduces: EVA's pin
(``uv.lock`` @039efa0a). It is sent where the SDK sends its own version — the
websocket query, ``source_info`` and the REST headers — so ElevenLabs sees
the same client on both sides of a parity run."""

SOURCE = "python_sdk"
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
"""``connect(ws_url, max_size=16 * 1024 * 1024)``, ``conversation.py:823``."""

HTTP_TIMEOUT_S = 30.0
"""EVA's client: ``httpx.Client(timeout=30.0)``, ``elevenlabs.py:136``."""

SDK_HEADERS: dict[str, str] = {
    "User-Agent": f"elevenlabs/{SDK_VERSION}",
    "X-Fern-Language": "Python",
    "X-Fern-SDK-Name": "elevenlabs",
    "X-Fern-SDK-Version": SDK_VERSION,
}
"""``core/client_wrapper.py::get_headers``, less the key (added per request)."""

POLL_BUDGET_S = 300.0
"""EVA's ``END_CALL_CHECK_BUDGET_SECONDS`` (``elevenlabs.py:29``)."""
POLL_FIRST_DELAY_S = 2.0
POLL_MAX_DELAY_S = 10.0
"""EVA's poll: wait 2 s, then double up to 10 s (``elevenlabs.py:289-327``)."""

AUDIO_FORMATS: dict[str, AudioFormat] = {
    "ulaw_8000": AudioFormat(sample_rate_hz=8_000, encoding="mulaw"),
    "pcm_8000": AudioFormat(sample_rate_hz=8_000),
    "pcm_16000": AudioFormat(sample_rate_hz=16_000),
    "pcm_22050": AudioFormat(sample_rate_hz=22_050),
    "pcm_24000": AudioFormat(sample_rate_hz=24_000),
    "pcm_44100": AudioFormat(sample_rate_hz=44_100),
    "pcm_48000": AudioFormat(sample_rate_hz=48_000),
}
"""ElevenLabs' audio format names. An agent's are part of its config
(``asr.user_input_audio_format``, ``tts.agent_output_audio_format``); EVA's
callers take ``ulaw_8000`` and speak ``pcm_16000``."""

DEFAULT_INPUT_FORMAT = "ulaw_8000"
DEFAULT_OUTPUT_FORMAT = "pcm_16000"


# --------------------------------------------------------------------- events


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """What the agent said: ``agent_response``, the text it spoke."""

    text: str
    t: float


@dataclass(frozen=True, slots=True)
class AgentResponseCorrection:
    """The agent's last response, corrected after it was cut short."""

    original: str
    corrected: str
    t: float


@dataclass(frozen=True, slots=True)
class UserTranscript:
    """ElevenLabs' transcript of the audio *we* sent (its speech recognition)."""

    text: str
    t: float


@dataclass(frozen=True, slots=True)
class Interruption:
    """The agent was interrupted; its audio up to ``event_id`` is void."""

    t: float
    event_id: int | None = None


@dataclass(frozen=True, slots=True)
class ClientToolCall:
    """The agent called a client tool. The transport answers it itself."""

    tool_call_id: str
    name: str
    parameters: dict[str, Any]
    t: float


@dataclass(frozen=True, slots=True)
class SessionEnded:
    """The conversation is over; nothing follows this event.

    ``reason`` is ``server_closed`` (ElevenLabs closed the socket cleanly — how
    a call ends after the agent's ``end_call``), ``connection_lost``,
    ``protocol_error`` (a message the SDK would have failed on),
    ``send_failed`` or ``client_closed``. The SDK ends the session on every
    one of them alike; the name is for diagnosis.
    """

    reason: str
    t: float
    code: int | None = None
    """The websocket close code, when there was one."""


AgentsEvent = (
    OutputAudio
    | AgentResponse
    | AgentResponseCorrection
    | UserTranscript
    | Interruption
    | ClientToolCall
    | SessionEnded
)
"""Everything :meth:`ElevenLabsAgentsTransport.events` yields."""


# ----------------------------------------------------------------------- REST


class AgentsAPI(PooledHTTP):
    """The REST half of ElevenLabs Agents: the requests SDK 2.53 makes.

    No retries, as the SDK makes none by default; errors come back as
    :class:`~harbor_voice.providers.base.ProviderError` so the caller can
    route them.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str = API_BASE,
        timeout_s: float = HTTP_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._timeout_s = timeout_s
        self._http = HTTPSession(client)

    @property
    def api_base(self) -> str:
        return self._api_base

    def require_key(self) -> str:
        """The API key, or an ``auth`` failure naming the variable (never the value)."""
        key = self._api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            raise ProviderError(
                ProviderFailure.AUTH, "ELEVENLABS_API_KEY is not set", provider=NAME
            )
        return key

    def _headers(self) -> dict[str, str]:
        return {**SDK_HEADERS, "xi-api-key": self.require_key()}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        headers = self._headers()
        client = self._http.client(self._timeout_s)
        try:
            response = await client.request(
                method,
                f"{self._api_base}/{path}",
                params=dict(params) if params else None,
                json=dict(body) if body is not None else None,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                ProviderFailure.TIMEOUT, f"{method} /{path}: {exc}", provider=NAME
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                ProviderFailure.STREAM_BROKEN, f"{method} /{path}: {exc}", provider=NAME
            ) from exc
        if not 200 <= response.status_code < 300:
            raise _error(response, f"{method} /{path}")
        return response

    async def _json(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        response = await self._request(method, path, **kw)
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"{method} /{path} returned no JSON: {response.text[:300]}",
                provider=NAME,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"{method} /{path} returned {type(data).__name__}, not an object",
                provider=NAME,
            )
        return data

    async def signed_url(self, agent_id: str) -> str:
        """``conversations.get_signed_url(agent_id=…)``; unset params are not sent."""
        data = await self._json(
            "GET",
            "v1/convai/conversation/get-signed-url",
            params={"agent_id": agent_id},
        )
        url = data.get("signed_url")
        if not isinstance(url, str) or not url:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"get-signed-url returned no signed_url: {str(data)[:300]}",
                provider=NAME,
            )
        return url

    async def conversation(self, conversation_id: str) -> dict[str, Any]:
        """``conversations.get(id)``: the conversation record, as JSON."""
        return await self._json(
            "GET", f"v1/convai/conversations/{_path(conversation_id)}"
        )

    async def conversation_audio(self, conversation_id: str) -> bytes:
        """``conversations.audio.get(id)``: the server's recording of the call."""
        response = await self._request(
            "GET", f"v1/convai/conversations/{_path(conversation_id)}/audio"
        )
        return response.content

    async def agent(self, agent_id: str) -> dict[str, Any]:
        """``agents.get(id)``: an agent's full live configuration."""
        return await self._json("GET", f"v1/convai/agents/{_path(agent_id)}")

    async def create_agent(self, body: Mapping[str, Any]) -> str:
        """``agents.create(...)``: returns the new agent's ID."""
        data = await self._json("POST", "v1/convai/agents/create", body=body)
        agent_id = data.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"agents/create returned no agent_id: {str(data)[:300]}",
                provider=NAME,
            )
        return agent_id


def _path(segment: str) -> str:
    return urllib.parse.quote(segment, safe="")


def _error(response: httpx.Response, what: str) -> ProviderError:
    status = response.status_code
    detail = response.text[:300]
    if status in (401, 403):
        kind = ProviderFailure.AUTH
    elif status == 404:
        kind = ProviderFailure.NOT_ENABLED
    elif status == 429:
        kind = ProviderFailure.THROTTLED
    elif status >= 500:
        kind = ProviderFailure.STREAM_BROKEN
    else:
        kind = ProviderFailure.MALFORMED
    return ProviderError(kind, f"{what} → {status}: {detail}", provider=NAME)


def with_sdk_source(signed_url: str) -> str:
    """The signed URL with ``source`` and ``version`` appended, as the SDK does.

    ``conversation.py:441-445``, step for step: the existing query is parsed
    keeping blanks and re-encoded with ``quote``, so the result is the SDK's
    byte for byte even where re-encoding changes the server's own escaping.
    """
    parsed = urllib.parse.urlparse(signed_url)
    params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    params.extend([("source", SOURCE), ("version", SDK_VERSION)])
    return urllib.parse.urlunparse(
        parsed._replace(
            query=urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        )
    )


def public_url(agent_id: str, *, api_base: str = API_BASE) -> str:
    """The URL for an agent that needs no auth (``_get_wss_url``, :421-433)."""
    parsed = urllib.parse.urlparse(api_base)
    scheme = {"https": "wss", "http": "ws"}.get(parsed.scheme, parsed.scheme)
    path = "/".join(
        urllib.parse.quote(seg, safe="") for seg in ("v1", "convai", "conversation")
    )
    params = [("agent_id", agent_id), ("source", SOURCE), ("version", SDK_VERSION)]
    return urllib.parse.urlunparse(
        (
            scheme,
            parsed.netloc,
            parsed.path.rstrip("/") + "/" + path,
            "",
            urllib.parse.urlencode(params, quote_via=urllib.parse.quote),
            "",
        )
    )


# ------------------------------------------------------------ the record


@dataclass(frozen=True, slots=True)
class PollOutcome:
    """How a :func:`poll_conversation` ended."""

    record: dict[str, Any] | None
    """The last record fetched, if any was."""
    verdict: bool | None
    """What ``decide`` answered; ``None`` if the budget ran out first."""
    attempts: int
    waited_s: float


Decide = Callable[[dict[str, Any]], bool | None]
"""Reads a record; ``True``/``False`` stops the poll, ``None`` keeps going."""


def finished(record: dict[str, Any]) -> bool | None:
    """The default :data:`Decide`: stop once ElevenLabs has finished the record."""
    status = record.get("status")
    if status == "done":
        return True
    if status == "failed":
        return False
    return None


async def poll_conversation(
    fetch: Callable[[], Awaitable[dict[str, Any]]],
    decide: Decide = finished,
    *,
    budget_s: float = POLL_BUDGET_S,
    first_delay_s: float = POLL_FIRST_DELAY_S,
    max_delay_s: float = POLL_MAX_DELAY_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> PollOutcome:
    """Fetch a record until ``decide`` answers, on EVA's schedule.

    EVA's ``_check_end_call_via_api`` (``elevenlabs.py:289-332``), loop for
    loop: *sleep first*, then fetch; the delay starts at 2 s and doubles to a
    10 s ceiling; the budget counts the sleeps only (2, 6, 14, 24, …, 304 s, so
    32 fetches at most). ``on_record`` sees every record fetched, before
    ``decide`` does — EVA rewrites its details file on each one. A fetch that
    raises ends the poll by raising, as it does in EVA.
    """
    delay = first_delay_s
    waited = 0.0
    attempts = 0
    record: dict[str, Any] | None = None
    while waited < budget_s:
        attempts += 1
        await sleep(delay)
        waited += delay
        record = await fetch()
        if on_record is not None:
            on_record(record)
        verdict = decide(record)
        if verdict is not None:
            return PollOutcome(record, verdict, attempts, waited)
        delay = min(delay * 2, max_delay_s)
    return PollOutcome(record, None, attempts, waited)


def _tokens(usage: Mapping[str, Any], category: str) -> int:
    entry = usage.get(category) or {}
    return int(entry.get("tokens") or 0) if isinstance(entry, Mapping) else 0


LLM_GENERATIONS: tuple[tuple[str, str], ...] = (
    ("irreversible_generation", "llm"),
    ("initiated_generation", "llm.initiated"),
)
"""``metadata.charging.llm_usage`` has two views (``LlmCategoryUsage``):
generations that went through, and every generation started — speculative
turns included. Both are kept, under separate components, rather than
guessing which one ElevenLabs bills."""


def usage_records(
    record: Mapping[str, Any], *, component: str = "caller"
) -> tuple[UsageRecord, ...]:
    """What one conversation consumed, from its record.

    - ``component``: call seconds (``metadata.call_duration_secs``), model =
      the agent ID — the unit ElevenLabs bills a call in;
    - ``component.llm`` / ``component.llm.initiated``: tokens per LLM, from
      ``metadata.charging.llm_usage`` (``LlmUsageOutput.model_usage``). More
      than one model here means a backup LLM answered some turn. Input is
      ``input`` + ``input_cache_read`` + ``input_cache_write``; cached input is
      ``input_cache_read``;
    - ``component.tts``: characters and seconds spoken;
    - ``component.asr``: transcription calls and seconds heard.

    Absent blocks give no record: the field is optional in SDK 2.53's types.
    """
    meta = record.get("metadata") or {}
    out: list[UsageRecord] = []
    seconds = meta.get("call_duration_secs")
    out.append(
        UsageRecord(
            component=component,
            model=record.get("agent_id"),
            requests=1,
            seconds=float(seconds) if seconds is not None else None,
        )
    )
    charging = meta.get("charging") or {}
    llm = charging.get("llm_usage") or {}
    for key, suffix in LLM_GENERATIONS:
        per_model = (llm.get(key) or {}).get("model_usage") or {}
        for model, usage in per_model.items():
            if not isinstance(usage, Mapping):
                continue
            cached = _tokens(usage, "input_cache_read")
            out.append(
                UsageRecord(
                    component=f"{component}.{suffix}",
                    model=str(model),
                    input_tokens=_tokens(usage, "input")
                    + cached
                    + _tokens(usage, "input_cache_write"),
                    cached_input_tokens=cached,
                    output_tokens=_tokens(usage, "output_total"),
                )
            )
    tts = charging.get("tts_usage") or {}
    if tts:
        out.append(
            UsageRecord(
                component=f"{component}.tts",
                model=tts.get("primary_tts_model"),
                characters=tts.get("total_characters"),
                seconds=tts.get("total_audio_output_seconds"),
            )
        )
    asr = charging.get("asr_usage") or {}
    if asr:
        out.append(
            UsageRecord(
                component=f"{component}.asr",
                model=asr.get("asr_model"),
                requests=asr.get("total_transcription_calls"),
                seconds=asr.get("total_audio_input_seconds"),
            )
        )
    return tuple(out)


# ------------------------------------------------------------------ transport


class ElevenLabsAgentsTransport(VoiceTransport):
    """One ElevenLabs Agents conversation over a websocket.

    Args:
        agent_id: The agent to talk to.
        dynamic_variables: Sent in the initiation message. EVA sends one,
            ``prompt``; the agent's system prompt is ``{{prompt}}``. Settable
            until :meth:`open`.
        requires_auth: Fetch a signed URL first (EVA's ``requires_auth=True``)
            rather than connecting to the public URL.
        input_format, output_format: The agent's configured formats (ElevenLabs
            names). They decide :attr:`capabilities`; nothing is converted
            here, as the SDK converts nothing.
        tools: Answers ``client_tool_call``. Without it every call gets the
            SDK's "not registered" error result — what EVA's callers, which
            have no client tools, would get.
        api: The REST client; built from ``api_key``/``api_base`` if omitted.
        connect: Replaces ``websockets.connect`` (tests).
    """

    def __init__(
        self,
        *,
        agent_id: str,
        party: PartyId = PartyId("caller"),
        dynamic_variables: Mapping[str, str] | None = None,
        conversation_config_override: Mapping[str, Any] | None = None,
        custom_llm_extra_body: Mapping[str, Any] | None = None,
        user_id: str | None = None,
        requires_auth: bool = True,
        input_format: str = DEFAULT_INPUT_FORMAT,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        tools: ToolRouter | None = None,
        api: AgentsAPI | None = None,
        api_key: str | None = None,
        api_base: str = API_BASE,
        clock: SessionClock | None = None,
        connect: Any = None,
    ) -> None:
        for name, fmt in (
            ("input_format", input_format),
            ("output_format", output_format),
        ):
            if fmt not in AUDIO_FORMATS:
                raise ValueError(
                    f"{NAME}: unknown {name} {fmt!r}; expected one of {sorted(AUDIO_FORMATS)}"
                )
        self.agent_id = agent_id
        self._party = party
        self.dynamic_variables: dict[str, str] = dict(dynamic_variables or {})
        self.conversation_config_override: dict[str, Any] = dict(
            conversation_config_override or {}
        )
        self.custom_llm_extra_body: dict[str, Any] = dict(custom_llm_extra_body or {})
        self.user_id = user_id
        self.requires_auth = requires_auth
        self.input_format = input_format
        self.output_format = output_format
        self._tools = tools
        self.api = api or AgentsAPI(api_key=api_key, api_base=api_base)
        self._clock = clock or SessionClock()
        self._connect = connect

        self._url: str | None = None
        self._ws: Any = None
        self._open = False
        self._closing = False
        self._pump: asyncio.Task[None] | None = None
        self._inbox: asyncio.Queue[AgentsEvent | None] = asyncio.Queue()
        self._tool_tasks: set[asyncio.Task[None]] = set()
        self._last_interrupt_id = 0
        self._seq = 0

        self.conversation_id: str | None = None
        self.initiation_metadata: dict[str, Any] = {}
        """The server's ``conversation_initiation_metadata_event``."""
        self.ended: SessionEnded | None = None
        self.ignored: Counter[str] = Counter()
        """Message types received and ignored, as the SDK ignores them."""
        self.pings = 0
        self.ping_ms: list[int] = []
        """``ping_event.ping_ms`` as reported, where it was."""
        self.errors: list[str] = []
        """Things that are wrong but not fatal (a format the server disagrees on)."""
        self.pump_error: Exception | None = None
        self.sent_chunks = 0
        self.dropped_chunks = 0
        """Audio handed in after the session ended, and dropped."""

    # ------------------------------------------------------------ declaration

    @property
    @override
    def capabilities(self) -> Capabilities:
        return Capabilities(
            duplex=frozenset({Duplex.FULL}),
            # Opaque to harbor: ElevenLabs composes the recogniser, the LLM
            # and the voice, so no seam is ours to observe or swap.
            pipeline=Pipeline.S2S,
            turn_control=TurnControl.PROVIDER_ENDPOINTING,
            # ``user_transcript`` is what its recogniser heard of our audio.
            exposes_asr=True,
            exposes_stages=frozenset({Stage.ASR_FINAL}),
            can_be_interrupted=True,
            optional_ops=frozenset({OptionalOp.EVENT_STREAM, OptionalOp.SEND_TEXT}),
            audio_in=AUDIO_FORMATS[self.input_format],
            audio_out=AUDIO_FORMATS[self.output_format],
        )

    @property
    @override
    def is_open(self) -> bool:
        return self._open

    @property
    def ref(self) -> ProviderRef:
        return ProviderRef.build(
            provider=NAME,
            model=self.agent_id,
            config={
                "requires_auth": self.requires_auth,
                "input_format": self.input_format,
                "output_format": self.output_format,
                "dynamic_variables": sorted(self.dynamic_variables),
                "sdk_version": SDK_VERSION,
            },
            capabilities=SPEC.capabilities,
        )

    # ------------------------------------------------------------- messages

    def initiation_message(self) -> str:
        """``conversation_initiation_client_data``, as ``_create_initiation_message``.

        ``conversation.py:460-473``. The SDK adds ``text_only`` to the
        override only when it has no audio interface (:410-411); a voice
        session has one, so the override goes out as given.
        """
        return json.dumps(
            {
                "type": "conversation_initiation_client_data",
                "custom_llm_extra_body": self.custom_llm_extra_body,
                "conversation_config_override": self.conversation_config_override,
                "dynamic_variables": self.dynamic_variables,
                "source_info": {"source": SOURCE, "version": SDK_VERSION},
                **({"user_id": self.user_id} if self.user_id else {}),
            }
        )

    @staticmethod
    def audio_message(payload: bytes) -> str:
        """One ``user_audio_chunk`` (``input_callback``, :830-838)."""
        return json.dumps({"user_audio_chunk": base64.b64encode(payload).decode()})

    @staticmethod
    def pong_message(event: Mapping[str, Any]) -> str:
        """The answer to one ``ping_event`` (``handle_ping``, :898-906)."""
        return json.dumps({"type": "pong", "event_id": event["event_id"]})

    # ------------------------------------------------------------- lifecycle

    async def prepare(self) -> str:
        """Resolve the websocket URL: a signed URL, fetched once.

        Separate from :meth:`open` because the SDK's ``start_session`` returns
        once it has this URL — the socket connects on its own thread after —
        and EVA logs ``session_started`` at that point.
        """
        if self._url is None:
            if self.requires_auth:
                self._url = with_sdk_source(await self.api.signed_url(self.agent_id))
            else:
                self._url = public_url(self.agent_id, api_base=self.api.api_base)
        return self._url

    @override
    async def open(self, config: SessionConfig | None = None) -> None:
        """Connect and send the initiation message.

        ``config.options`` may carry ``dynamic_variables``, merged over the
        constructor's. Returns once the initiation message is out: the SDK
        starts streaming audio then, without waiting for the server's
        metadata, which arrives on :meth:`events`'s pump.
        """
        if self._open:
            return
        if self.ended is not None:
            raise TransportNotConnectedError(f"{NAME}: this session has ended")
        if config is not None:
            extra = config.options.get("dynamic_variables") or {}
            self.dynamic_variables.update({str(k): str(v) for k, v in extra.items()})
        url = await self.prepare()
        connect = self._connect or _websockets().connect
        try:
            self._ws = await connect(url, max_size=MAX_MESSAGE_BYTES)
            await self._ws.send(self.initiation_message())
        except Exception as exc:
            await self._drop()
            raise _classify(exc, self.agent_id) from exc
        self._open = True
        self._pump = asyncio.create_task(self._receive())

    async def send_audio(self, payload: bytes) -> None:
        """Send one chunk as the SDK's ``input_callback`` does.

        Never raises once open: a send that fails ends the session (the SDK
        calls ``end_session``), and audio after the end is dropped and counted.
        """
        if self.ended is not None or self._closing:
            self.dropped_chunks += 1
            return
        self.require_open()
        try:
            await self._ws.send(self.audio_message(payload))
        except Exception:  # noqa: BLE001 - the SDK ends the session, so do we
            self.dropped_chunks += 1
            await self._end_session("send_failed")
            return
        self.sent_chunks += 1

    @override
    async def tx(self, frame: AudioFrame) -> None:
        """Send one frame, which must already be in :attr:`capabilities`' ``audio_in``."""
        expected = AUDIO_FORMATS[self.input_format]
        if frame.fmt != expected:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"{NAME} takes {self.input_format} ({expected}), got {frame.fmt}; "
                f"convert before the transport, as EVA's bridge does",
                provider=NAME,
                model=self.agent_id,
            )
        await self.send_audio(frame.pcm)

    async def _send_json(self, message: Mapping[str, Any]) -> None:
        if self._ws is None or self.ended is not None:
            # The SDK's ``RuntimeError("Session not started or websocket not
            # connected.")`` for the same case.
            raise TransportNotConnectedError(
                f"{NAME}: session not started or websocket not connected"
            )
        await self._ws.send(json.dumps(dict(message)))

    async def send_user_activity(self) -> None:
        """``register_user_activity``: ``{"type": "user_activity"}``.

        Resets ElevenLabs' idle timer. Raises when there is no open socket, as
        the SDK does; EVA's keep-alive catches that and carries on.
        """
        await self._send_json({"type": "user_activity"})

    async def send_contextual_update(self, text: str) -> None:
        """``send_contextual_update``: context the agent sees without replying."""
        await self._send_json({"type": "contextual_update", "text": text})

    @override
    async def send_text(self, text: str) -> None:
        """``send_user_message``: a user turn as text."""
        await self._send_json({"type": "user_message", "text": text})

    # --------------------------------------------------------------- stream

    @override
    async def events(self) -> AsyncIterator[AgentsEvent]:  # ty: ignore[invalid-method-override]
        """Every event in arrival order, ending with one :class:`SessionEnded`.

        A superset of the agent-side :data:`~harbor_voice.transport.stream.StreamEvent`
        vocabulary: an ElevenLabs agent reports its own speech as text and
        what it heard of ours, which an agent under test does not. Hence the
        wider return type (and the ignore): ``VoiceTransport.events`` is typed
        for the agent side only, and a reader of this transport dispatches on
        these types by ``isinstance`` anyway.
        """
        while True:
            event = await self._inbox.get()
            if event is None:
                self._inbox.put_nowait(None)  # stays ended for a second reader
                return
            yield event

    @override
    async def rx(self) -> AsyncIterator[AudioFrame]:
        fmt = AUDIO_FORMATS[self.output_format]
        async for event in self.events():
            if isinstance(event, OutputAudio) and event.pcm:
                frame = AudioFrame(
                    pcm=event.pcm,
                    seq=self._seq,
                    party=self._party,
                    t_produced=event.t,
                    fmt=fmt,
                )
                self._seq += 1
                yield frame

    @override
    async def close(self) -> None:
        """Close the socket (normal closure) and end the stream. Idempotent."""
        if self._closing:
            return
        self._closing = True
        self._open = False
        for task in list(self._tool_tasks):
            task.cancel()
        if self._tool_tasks:
            await asyncio.gather(*self._tool_tasks, return_exceptions=True)
        await self._drop()
        if self._pump is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._pump), timeout=5.0)
            except (TimeoutError, asyncio.CancelledError):
                self._pump.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._pump
        await self._end_session("client_closed")

    # ------------------------------------------------------------ after call

    async def fetch_conversation(self) -> dict[str, Any]:
        """The conversation record now (one ``GET``)."""
        return await self.api.conversation(self._require_conversation_id())

    async def conversation_record(
        self,
        decide: Decide = finished,
        *,
        budget_s: float = POLL_BUDGET_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_record: Callable[[dict[str, Any]], None] | None = None,
    ) -> PollOutcome:
        """Poll the record on EVA's schedule until ``decide`` answers.

        See :func:`poll_conversation`. EVA's caller passes its ``end_call``
        check as ``decide``; the default waits for ``status`` ``done``.
        """
        return await poll_conversation(
            self.fetch_conversation,
            decide,
            budget_s=budget_s,
            sleep=sleep,
            on_record=on_record,
        )

    async def fetch_audio(self) -> bytes:
        """The server's recording of the conversation (one ``GET``)."""
        return await self.api.conversation_audio(self._require_conversation_id())

    def usage_records(
        self, record: Mapping[str, Any], *, component: str = "caller"
    ) -> tuple[UsageRecord, ...]:
        return usage_records(record, component=component)

    def _require_conversation_id(self) -> str:
        if not self.conversation_id:
            raise TransportNotConnectedError(
                f"{NAME}: no conversation_id (the server never sent its metadata)"
            )
        return self.conversation_id

    # ------------------------------------------------------------- internals

    async def _drop(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

    async def _end_session(self, reason: str, code: int | None = None) -> None:
        """Record the end once, close the socket, end the stream."""
        if self.ended is not None:
            return
        self.ended = SessionEnded(reason, self._clock.now(), code)
        self._open = False
        self._inbox.put_nowait(self.ended)
        self._inbox.put_nowait(None)
        if not self._closing:
            await self._drop()

    async def _receive(self) -> None:
        ws = self._ws
        reason = "server_closed"
        try:
            async for raw in ws:
                await self._handle(json.loads(raw))
        except asyncio.CancelledError:
            reason = "client_closed"
            raise
        except Exception as exc:  # noqa: BLE001 - the SDK ends the session on any error
            self.pump_error = exc
            closed = _websockets().exceptions.ConnectionClosed
            reason = "connection_lost" if isinstance(exc, closed) else "protocol_error"
        finally:
            if self._closing and reason == "server_closed":
                reason = "client_closed"
            await self._end_session(reason, getattr(ws, "close_code", None))

    async def _handle(self, message: dict[str, Any]) -> None:
        """``_handle_message_core``, case for case (``conversation.py:475-548``).

        Field access is as strict as the SDK's: a missing key raises, which
        ends the session as it does there.
        """
        kind = message["type"]
        now = self._clock.now()
        if kind == "conversation_initiation_metadata":
            event = message["conversation_initiation_metadata_event"]
            if self.conversation_id is not None:
                raise ProviderError(
                    ProviderFailure.MALFORMED,
                    "a second conversation_initiation_metadata",
                    provider=NAME,
                    model=self.agent_id,
                )
            self.conversation_id = event["conversation_id"]
            self.initiation_metadata = dict(event)
            self._check_formats(event)
        elif kind == "audio":
            event = message["audio_event"]
            event_id = int(event["event_id"])
            if event_id <= self._last_interrupt_id:
                return
            pcm = base64.b64decode(event["audio_base_64"])
            self._inbox.put_nowait(OutputAudio(pcm, str(event_id), now))
        elif kind == "agent_response":
            event = message["agent_response_event"]
            self._inbox.put_nowait(AgentResponse(event["agent_response"].strip(), now))
        elif kind == "agent_response_correction":
            event = message["agent_response_correction_event"]
            self._inbox.put_nowait(
                AgentResponseCorrection(
                    event["original_agent_response"].strip(),
                    event["corrected_agent_response"].strip(),
                    now,
                )
            )
        elif kind == "user_transcript":
            event = message["user_transcription_event"]
            self._inbox.put_nowait(
                UserTranscript(event["user_transcript"].strip(), now)
            )
        elif kind == "interruption":
            event = message["interruption_event"]
            self._last_interrupt_id = int(event["event_id"])
            self._inbox.put_nowait(Interruption(now, self._last_interrupt_id))
        elif kind == "ping":
            event = message["ping_event"]
            ws = self._ws
            if ws is not None:
                await ws.send(self.pong_message(event))
            self.pings += 1
            if event.get("ping_ms"):
                self.ping_ms.append(int(event["ping_ms"]))
        elif kind == "client_tool_call":
            tool_call = message.get("client_tool_call", {})
            name = tool_call.get("tool_name")
            parameters = {
                "tool_call_id": tool_call["tool_call_id"],
                **tool_call.get("parameters", {}),
            }
            self._inbox.put_nowait(
                ClientToolCall(
                    str(tool_call["tool_call_id"]),
                    str(name),
                    {k: v for k, v in parameters.items() if k != "tool_call_id"},
                    now,
                )
            )
            task = asyncio.create_task(self._answer_tool(name, parameters))
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)
        else:
            self.ignored[str(kind)] += 1

    def _check_formats(self, event: Mapping[str, Any]) -> None:
        for key, ours in (
            ("user_input_audio_format", self.input_format),
            ("agent_output_audio_format", self.output_format),
        ):
            theirs = event.get(key)
            if theirs is not None and theirs != ours:
                self.errors.append(
                    f"server says {key}={theirs!r}, transport expects {ours!r}"
                )

    async def _answer_tool(self, name: str | None, parameters: dict[str, Any]) -> None:
        """``ClientTools.execute_tool`` (``conversation.py:296-325``).

        The result message is the SDK's, including its default text for an
        empty result and its error text for an unregistered tool.
        """
        call_id = parameters.get("tool_call_id")
        tools = self._tools
        try:
            if tools is None or name not in {spec.name for spec in tools.specs}:
                raise ValueError(f"Tool '{name}' is not registered")
            invocation = ToolInvocation(
                tool_use_id=str(call_id),
                name=str(name),
                arguments={k: v for k, v in parameters.items() if k != "tool_call_id"},
            )
            result = await tools.invoke(invocation, at_s=self._clock.now())
            response = {
                "type": "client_tool_result",
                "tool_call_id": call_id,
                "result": result.as_text()
                or f"Client tool: {name} called successfully.",
                "is_error": result.is_error,
            }
        except Exception as exc:  # noqa: BLE001 - becomes the tool's error result
            response = {
                "type": "client_tool_result",
                "tool_call_id": call_id,
                "result": str(exc),
                "is_error": True,
            }
        if self._ws is not None and self.ended is None and not self._closing:
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps(response))


def _classify(exc: Exception, agent_id: str) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    text = str(exc)
    if "401" in text or "403" in text:
        kind = ProviderFailure.AUTH
    elif "404" in text:
        kind = ProviderFailure.NOT_ENABLED
    elif "429" in text:
        kind = ProviderFailure.THROTTLED
    elif isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        kind = ProviderFailure.TIMEOUT
    else:
        kind = ProviderFailure.STREAM_BROKEN
    return ProviderError(
        kind, f"{type(exc).__name__}: {text[:300]}", provider=NAME, model=agent_id
    )


SPEC = ProviderSpec(
    name=NAME,
    modality=Modality.S2S,
    capabilities=ProviderCapabilities(
        modality=Modality.S2S,
        streams=True,
        emits_partials=False,
        word_timings=False,
        diarizes=False,
    ),
    credential_provider="elevenlabs",
    requires_extra="voice",
    default_model=None,
)
"""The spec string names the agent: ``elevenlabs.agents/<agent_id>``."""


def build(*, model: str, **config: Any) -> ElevenLabsAgentsTransport:
    """Registry builder. ``model`` is the agent ID unless ``agent_id`` is given.

    ``clock`` and ``party`` are taken as the other S2S transports take them.
    """
    party = config.pop("party", None) or PartyId("caller")
    agent_id = config.pop("agent_id", None) or model
    return ElevenLabsAgentsTransport(
        agent_id=str(agent_id), party=PartyId(str(party)), **config
    )


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = ((SPEC, build),)
