"""The caller: EVA's ElevenLabs user simulator, session logic ported.

EVA's caller is an ElevenLabs agent; ElevenLabs runs its recogniser, LLM,
voice and turn-taking. What EVA's own code decides is everything *around*
that conversation, in ``eva/user_simulator/elevenlabs.py`` (@039efa0a) and its
base class: when the call counts as over, why, and what is written down. That
is what this module ports, line for line, on harbor's ``elevenlabs.agents``
transport instead of the SDK. It feeds ``conversation_valid_end`` — the last
line of ``user_simulator_events.jsonl`` must be ``connection_state`` with
reason ``goodbye`` — and so EVA's rerun gate, which is why no rule here is
approximated.

**Who does what.** The call loop (``call.py``) owns the audio: it reads the
transport's events, routes :class:`~harbor_voice.transport.stream.OutputAudio`
through the bridge, and hands everything else to :meth:`ElevenLabsCaller.on_event`.
This class owns the session:

1. :meth:`~ElevenLabsCaller.start` — ``connected``, the signed URL,
   ``session_started``, the keep-alive and the time limit, then the socket
   (``elevenlabs.py:130-183``), then ``bridge.start(transport.send_audio)`` as
   the SDK does once connected. Wire the bridge with
   ``record_callback=caller.record_audio`` and
   ``conversation_done_callback=caller.end``, as EVA's simulator wires it
   (``elevenlabs.py:121-128``).
2. :meth:`~ElevenLabsCaller.on_event` — EVA's three SDK callbacks
   (``:404-464``) and the SDK's own end of session.
3. :attr:`~ElevenLabsCaller.should_end` / :meth:`~ElevenLabsCaller.wait` —
   EVA's ``wait_for(conversation_done, timeout)`` (``:185-201``).
4. :meth:`~ElevenLabsCaller.finish` — end the session, the post-hoc
   ``end_call`` check, the server audio, ``session_ended``, then EVA's
   teardown: latencies, the clean caller track, the 4 s grace
   (``:203-265``). Close the agent and save the event log *after* it returns,
   as EVA's ``stop_async`` and ``event_logger.save()`` come after.

**End reasons** (what :attr:`~ElevenLabsCaller.end_reason` can be):

- ``goodbye`` — the record shows ``end_call`` (in ``termination_reason`` or a
  turn's tool results), whatever ended the session;
- ``end_call_unverified`` — the socket closed but the record was not
  finished within 300 s, so ``end_call`` could not be checked;
- ``session_ended`` — the socket closed and the finished record has no
  ``end_call`` (e.g. the agent's 600 s maximum);
- ``inactivity_timeout`` — 12 keep-alives (10 s apart) with no caller speech,
  no transcript of the agent and no agent audio playing;
- ``timeout`` — the simulator's own time limit, logged as a ``timeout``
  event. In EVA it never survives as the final reason: ending the SDK session
  afterwards fires the audio interface's ``stop()``, which reports
  ``session_ended`` into a conversation not yet marked done (``audio_bridge.py:224-239``),
  and the post-hoc check then decides. Ported as is;
- ``elevenlabs_disconnect`` — the call loop's agent socket closed mid-call
  (EVA's bridge default, ``audio_bridge.py:120``, ``:738-744``); the call
  loop reports it with :meth:`~ElevenLabsCaller.end`;
- ``error`` — something raised; the message is logged as an ``error`` event
  after the teardown, as in EVA.

The worker's guard (``time_limit_exceeded``, ``worker.py:166-176``) is the
driver's: it wraps the whole call, agent connect to :meth:`finish`, in
:data:`GUARD_TIMEOUT_S`. Since it starts before this class's own timer and
has the same length, it is the one that fires, as in EVA. :meth:`finish` is
cancellation-safe the same way EVA's is: cancelled mid-poll, it still writes
the latencies and the clean track and waits out the grace, and logs no
``session_ended``.

**Deviations, all deliberate.** A websocket that cannot connect is an
``error`` at once; the SDK's connect thread would die silently and EVA would
wait out its 620 s timeout. ``elevenlabs_conversation_details.json`` holds the
record as ElevenLabs sent it, where EVA dumps the SDK's pydantic model of it
(every declared field, absent ones as ``null``); nothing in EVA reads the file.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import wave
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Protocol

from harbor_voice.providers.elevenlabs.agents import AgentResponse
from harbor_voice.providers.elevenlabs.agents import AgentResponseCorrection
from harbor_voice.providers.elevenlabs.agents import AgentsEvent
from harbor_voice.providers.elevenlabs.agents import ElevenLabsAgentsTransport
from harbor_voice.providers.elevenlabs.agents import SessionEnded
from harbor_voice.providers.elevenlabs.agents import UserTranscript
from harbor_voice.trajectory.model import UsageRecord

logger = logging.getLogger(__name__)

# ------------------------------------------------------------- EVA's numbers

PROVIDER = "elevenlabs"
"""``provider=`` on EVA's event logger (``elevenlabs.py:78``)."""

END_CALL_CHECK_BUDGET_SECONDS = 300.0
"""``elevenlabs.py:29``."""

KEEPALIVE_INTERVAL_S = 10.0
"""``elevenlabs.py:369``."""

MAX_CONSECUTIVE_KEEPALIVES = 12
"""``elevenlabs.py:85``: 12 × 10 s = 2 minutes without activity."""

AUDIO_FETCH_ATTEMPTS = 5
AUDIO_FETCH_FIRST_DELAY_S = 2.0
AUDIO_FETCH_MAX_DELAY_S = 10.0
"""``_fetch_elevenlabs_audio``, ``elevenlabs.py:334-352``."""

STT_GRACE_S = 4.0
"""``elevenlabs.py:258-262``: the assistant's STT gets 4 s for the last turn."""

CONVERSATION_TIME_LIMIT_S = 600
"""``conversation_time_limit_seconds`` default (``models/config.py:735-736``)."""

USER_SIMULATOR_SHUTDOWN_GRACE_S = 20
"""``worker.py:23``."""

SIMULATOR_TIMEOUT_S = CONVERSATION_TIME_LIMIT_S + USER_SIMULATOR_SHUTDOWN_GRACE_S
"""The ``timeout`` EVA's worker hands the simulator (``worker.py:386``, :398-400)."""

GUARD_TIMEOUT_S = SIMULATOR_TIMEOUT_S
"""The worker's own guard around the whole call: the same 620 s (``worker.py:166-171``)."""

ELEVENLABS_OUTPUT_RATE = 16_000
"""The caller's audio, and its clean track's rate (``audio_bridge.py:37``)."""

# ------------------------------------------------------------ end reasons

GOODBYE = "goodbye"
END_CALL_UNVERIFIED = "end_call_unverified"
SESSION_ENDED = "session_ended"
TIMEOUT = "timeout"
INACTIVITY_TIMEOUT = "inactivity_timeout"
ELEVENLABS_DISCONNECT = "elevenlabs_disconnect"
ERROR = "error"
TIME_LIMIT_EXCEEDED = "time_limit_exceeded"
"""The driver's guard, not this class's (``worker.py:175``)."""
UNKNOWN = "unknown"
"""Before anything has ended (``base.py:71``)."""

# --------------------------------------------------------------- the files

DETAILS_FILE = "elevenlabs_conversation_details.json"
AUDIO_FILE = "elevenlabs_audio_recording.mp3"
LATENCIES_FILE = "response_latencies.json"
CLEAN_AUDIO_FILE = "audio_user_clean.wav"


# ---------------------------------------------------------- what we drive


class EventLog(Protocol):
    """The part of EVA's ``UserSimulatorEventLogger`` this class writes through."""

    def log_event(self, event_type: str, data: dict[str, Any]) -> None: ...

    def log_connection_state(
        self, state: str, details: dict[str, Any] | None = None
    ) -> None: ...

    def log_error(self, error: str, details: dict[str, Any] | None = None) -> None: ...


class Bridge(Protocol):
    """The part of EVA's ``BotToBotAudioBridge`` the simulator and SDK touch.

    ``start`` and ``stop`` are the SDK audio interface's. The SDK calls
    ``start(input_callback)`` once its socket is up (``conversation.py:845-846``)
    — the caller hears the agent from then on — and ``stop()`` from
    ``end_session``; EVA's sets ``running = False`` and reports
    ``session_ended`` (``audio_bridge.py:224-239``). This class calls both
    where the SDK would. ``eva_adapter.bridge.Bridge`` is the implementation.
    """

    def start(self, input_callback: Callable[[bytes], Awaitable[None]]) -> None: ...

    def notify_user_utterance_complete(self) -> None: ...

    def is_assistant_playing(self) -> bool: ...

    def get_latencies(self) -> list[float]: ...

    def stop(self) -> None: ...


# ----------------------------------------------------------- pure decisions


def end_reason_after_check(reason: str, end_call_found: bool | None) -> str:
    """``_end_reason_after_check`` (``elevenlabs.py:267-274``)."""
    if end_call_found:
        return GOODBYE
    if end_call_found is None and reason == SESSION_ENDED:
        return END_CALL_UNVERIFIED
    return reason


def end_call_verdict(record: dict[str, Any]) -> bool | None:
    """One poll of ``_check_end_call_via_api`` (``elevenlabs.py:307-322``).

    ``True``: ``end_call`` is in ``metadata.termination_reason`` (set while
    the record is still processing) or in a turn's ``tool_results``.
    ``False``: the transcript is there, the status is ``done``, and no
    ``end_call``. ``None``: keep polling — including a ``done`` record with an
    *empty* transcript, which EVA keeps polling too.
    """
    metadata = record.get("metadata")
    reason = metadata.get("termination_reason") if isinstance(metadata, dict) else None
    if "end_call" in str(reason or ""):
        return True
    transcript = record.get("transcript")
    if transcript:
        for turn in transcript:
            results = turn.get("tool_results") if isinstance(turn, dict) else None
            for result in results or ():
                if isinstance(result, dict) and result.get("tool_name") == "end_call":
                    return True
        if record.get("status") == "done":
            return False
    return None


# ---------------------------------------------------------------- the caller


class ElevenLabsCaller:
    """One EVA caller session on an :class:`ElevenLabsAgentsTransport`.

    Args:
        transport: The caller agent's transport, not yet open.
        prompt: The caller prompt EVA builds (``_build_prompt``), sent as the
            ``prompt`` dynamic variable, the agent's whole system prompt.
        events: Where EVA's simulator events go (``eva_adapter.events.EventLog``).
        out_dir: The record directory: the details, server audio, latencies
            and clean caller track land here.
        bridge: The audio bridge. May be attached later (:attr:`bridge`),
            before the first event.
        timeout_s: The simulator's own limit; EVA's worker passes 620.
        server_url: Logged with ``connected``, where EVA logs its assistant
            server's URL; pass something naming the agent side.
        on_conversation_ending: EVA's hook to the assistant
            (``base.py:74-78``), told the moment the call is known to be over.
        sleep: Every wait but the time limit (tests replace it).
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        transport: ElevenLabsAgentsTransport,
        prompt: str,
        events: EventLog,
        out_dir: Path,
        bridge: Bridge | None = None,
        timeout_s: float = SIMULATOR_TIMEOUT_S,
        server_url: str = "harbor-voice",
        on_conversation_ending: Callable[[str | None], None] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.transport = transport
        self.prompt = prompt
        self.events = events
        self.out_dir = Path(out_dir)
        self.bridge = bridge
        self.timeout_s = timeout_s
        self.server_url = server_url
        self.on_conversation_ending = on_conversation_ending
        self._sleep = sleep

        self._end_reason = UNKNOWN
        self._done = asyncio.Event()
        """EVA's ``_conversation_done``."""
        self._ended = asyncio.Event()
        """Set once the wait for the end is over: done, timed out or failed."""
        self._ending_signaled = False
        self._consecutive_keepalive_count = 0
        self._user_clean_audio = bytearray()
        self._keepalive_task: asyncio.Task[None] | None = None
        self._wait_task: asyncio.Task[None] | None = None
        self._session_closed = False
        self._finished = False
        self.error: Exception | None = None
        """What made the call an ``error``, if something did."""
        self.timed_out = False
        self.keepalives_sent = 0
        self.conversation_record: dict[str, Any] | None = None
        """The last conversation record fetched by the ``end_call`` check."""
        self.end_call_found: bool | None = None

    # --------------------------------------------------------------- state

    @property
    def end_reason(self) -> str:
        return self._end_reason

    @property
    def should_end(self) -> bool:
        """Whether the call loop should stop and call :meth:`finish`."""
        return self._ended.is_set()

    async def wait(self) -> None:
        """Until :attr:`should_end`."""
        await self._ended.wait()

    # ----------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Open the caller, as EVA does once its bridge is connected.

        ``connected`` (``elevenlabs.py:132``), the prompt as the only dynamic
        variable (:144), the signed URL — where the SDK's ``start_session``
        returns — then ``session_started`` (:180), the keep-alive (:183) and
        the time limit (:186), then the socket itself.

        Raises only for a missing API key (EVA raises before logging anything,
        :98-102; here it is a configuration failure that fails the job). Any
        other failure makes the call an ``error``: :attr:`should_end` turns
        true and :meth:`finish` completes EVA's error path.
        """
        self.transport.api.require_key()
        self.events.log_connection_state("connected", {"server_url": self.server_url})
        try:
            self.transport.dynamic_variables = {"prompt": self.prompt}
            await self.transport.prepare()
            self.events.log_connection_state("session_started")
            self._keepalive_task = asyncio.create_task(self._keep_alive())
            self._wait_task = asyncio.create_task(self._wait_for_end())
            await self.transport.open()
            # The SDK hands the audio interface its input callback once the
            # socket is up and the initiation sent (conversation.py:845-846):
            # the agent's audio reaches the caller from here on.
            if self.bridge is not None:
                self.bridge.start(self.transport.send_audio)
        except Exception as exc:  # noqa: BLE001 - EVA's error path, completed by finish()
            logger.error("ElevenLabs conversation error: %s", exc, exc_info=True)
            self.error = exc
            self._end_reason = ERROR
            self._ended.set()

    def on_event(self, event: AgentsEvent) -> None:
        """One transport event; audio is the call loop's and is ignored here.

        - ``AgentResponse`` → ``_on_user_speaks`` (``elevenlabs.py:404-426``):
          reset the inactivity count, tell the bridge the utterance is complete,
          log ``user_speech``;
        - ``AgentResponseCorrection`` → ``user_speech_correction`` (:428-444);
        - ``UserTranscript`` → reset, log ``assistant_speech`` (:446-464);
        - ``SessionEnded`` → what the SDK's ``end_session`` does when its
          socket closes: the bridge's ``stop()`` and ``session_ended``;
        - ``Interruption``, ``ClientToolCall``: nothing. EVA's bridge ignores
          interruptions (``audio_bridge.py:323-330``) and the transport
          answers tool calls as the SDK does.

        Once :meth:`finish` has ended the session nothing is taken, as the
        SDK's thread stops calling back after ``end_session``.
        """
        if self._session_closed:
            return
        if isinstance(event, AgentResponse):
            self._on_user_speaks(event.text)
        elif isinstance(event, AgentResponseCorrection):
            self._on_user_response_correction(event.original, event.corrected)
        elif isinstance(event, UserTranscript):
            self._on_assistant_speaks(event.text)
        elif isinstance(event, SessionEnded):
            self._stop_bridge()
            self.end(SESSION_ENDED)

    def end(self, reason: str = GOODBYE) -> None:
        """``_on_conversation_end`` (``base.py:141-146``): the first reason wins."""
        if not self._done.is_set():
            self._end_reason = reason
            self._done.set()
            logger.info("Conversation end signaled: %s", reason)
            self.signal_conversation_ending(reason)

    def signal_conversation_ending(self, reason: str | None = None) -> None:
        """``base.py:123-139``: tell the agent side once; never raise."""
        if self._ending_signaled:
            return
        self._ending_signaled = True
        if self.on_conversation_ending is None:
            return
        try:
            self.on_conversation_ending(reason or self._end_reason)
        except Exception as exc:  # noqa: BLE001 - advisory, as in EVA
            logger.warning(
                "Failed to signal conversation ending to the assistant: %s", exc
            )

    def record_audio(self, source: str, audio: bytes) -> None:
        """EVA's ``_record_audio`` (``base.py:162-174``): the bridge's record callback.

        Only ``user_clean`` is kept — the caller's audio before any
        perturbation, 16 kHz — and :meth:`finish` writes it.
        """
        if source == "user_clean":
            self._user_clean_audio.extend(audio)

    async def finish(self) -> str:
        """End the call as EVA does, and return the end reason.

        The session part (``elevenlabs.py:203-227``), then the teardown that
        runs whatever happened (:233-263), then the ``error`` event if
        something raised (:106-110). Call it once :attr:`should_end`; called
        earlier it ends the session as if ElevenLabs had closed it.
        """
        if self._finished:
            return self._end_reason
        self._finished = True
        error = self.error
        try:
            if error is None:
                await self._end_of_session()
        except Exception as exc:  # noqa: BLE001 - EVA's inner handler: reason "error"
            logger.error("Error in ElevenLabs conversation: %s", exc, exc_info=True)
            error = exc
            self._end_reason = ERROR
        finally:
            await self._teardown()
        if error is not None:
            # run_conversation's handler (elevenlabs.py:106-110), after the teardown.
            self.error = error
            self._end_reason = ERROR
            self.events.log_error(str(error))
        return self._end_reason

    def usage_records(self) -> tuple[UsageRecord, ...]:
        """What the call consumed, from the last record fetched (may be partial)."""
        if self.conversation_record is None:
            return ()
        return self.transport.usage_records(self.conversation_record)

    # ------------------------------------------------------------ internals

    def _reset_keepalive_counter(self) -> None:
        self._consecutive_keepalive_count = 0

    def _on_user_speaks(self, text: str) -> None:
        self._reset_keepalive_counter()
        logger.info("User (ElevenLabs): %s", text)
        # The authoritative end-of-turn cue for the bridge (elevenlabs.py:414-418).
        if self.bridge is not None:
            self.bridge.notify_user_utterance_complete()
        self.events.log_event("user_speech", {"text": text, "source": "simulated_user"})

    def _on_user_response_correction(self, original: str, corrected: str) -> None:
        self.events.log_event(
            "user_speech_correction", {"original": original, "corrected": corrected}
        )

    def _on_assistant_speaks(self, text: str) -> None:
        self._reset_keepalive_counter()
        logger.info("Assistant: %s", text)
        self.events.log_event("assistant_speech", {"text": text, "source": "assistant"})

    def _stop_bridge(self) -> None:
        if self.bridge is not None:
            try:
                self.bridge.stop()
            except Exception as exc:  # noqa: BLE001 - the SDK's stop path never raises out
                logger.warning("Bridge stop failed: %s", exc)

    async def _wait_for_end(self) -> None:
        """``wait_for(conversation_done, timeout)`` (``elevenlabs.py:185-201``)."""
        try:
            await asyncio.wait_for(self._done.wait(), timeout=self.timeout_s)
            logger.info("Conversation ended: %s", self._end_reason)
        except TimeoutError:
            logger.info("Conversation timed out after %ss", self.timeout_s)
            self.timed_out = True
            self._end_reason = TIMEOUT
            self.events.log_event("timeout", {"duration": self.timeout_s})
            # _on_conversation_end never fired on this path, so signal here too.
            self.signal_conversation_ending(TIMEOUT)
        finally:
            await self._cancel(self._keepalive_task)
        self._ended.set()

    async def _keep_alive(self) -> None:
        """``_keep_alive_task`` (``elevenlabs.py:358-402``).

        Every 10 s, ``user_activity``; then the inactivity count goes up
        unless the agent's audio is playing at that moment. Speech either way
        resets it (:404-464). A ``user_activity`` that cannot be sent (the
        socket is gone) neither counts nor resets.
        """
        while not self._done.is_set():
            await self._sleep(KEEPALIVE_INTERVAL_S)
            if self._done.is_set():
                continue
            try:
                await self.transport.send_user_activity()
                self.keepalives_sent += 1
                if self.bridge is not None and self.bridge.is_assistant_playing():
                    self._reset_keepalive_counter()
                else:
                    self._consecutive_keepalive_count += 1
                if self._consecutive_keepalive_count >= MAX_CONSECUTIVE_KEEPALIVES:
                    logger.warning(
                        "Ending conversation: %s consecutive keep-alives without "
                        "user/agent activity",
                        MAX_CONSECUTIVE_KEEPALIVES,
                    )
                    self.end(INACTIVITY_TIMEOUT)
                    break
            except Exception as exc:  # noqa: BLE001 - EVA logs and carries on
                logger.warning("Failed to send keep-alive ping: %s", exc)

    async def _end_session(self) -> None:
        """The SDK's ``end_session`` (``conversation.py:713-722``).

        The audio interface's ``stop()`` first — EVA's reports
        ``session_ended``, which only sticks if nothing ended the call before
        (the time-limit path) — then the socket.
        """
        self._stop_bridge()
        self.end(SESSION_ENDED)
        self._session_closed = True
        await self.transport.close()

    async def _end_of_session(self) -> None:
        if not self._ended.is_set():
            # finish() before the end: stop waiting, as a cancelled wait would.
            await self._cancel(self._wait_task)
            self._ended.set()
        await self._cancel(self._keepalive_task)
        # Signal first (idempotent): the polling below can take minutes.
        self.signal_conversation_ending()
        await self._end_session()
        conversation_id = self.transport.conversation_id
        if conversation_id:
            try:
                self.end_call_found = await self._check_end_call()
                self._end_reason = end_reason_after_check(
                    self._end_reason, self.end_call_found
                )
            except Exception as exc:  # noqa: BLE001 - elevenlabs.py:219-220
                logger.warning(
                    "Failed to check conversation history for end_call: %s", exc
                )
            try:
                await self._fetch_audio()
            except Exception as exc:  # noqa: BLE001 - elevenlabs.py:224-225
                logger.warning("Failed to fetch ElevenLabs server audio: %s", exc)
        self.events.log_connection_state("session_ended", {"reason": self._end_reason})

    async def _check_end_call(self) -> bool | None:
        """``_check_end_call_via_api`` (``elevenlabs.py:276-332``) on the transport's poll."""
        details_path = self.out_dir / DETAILS_FILE

        def dump(record: dict[str, Any]) -> None:
            self.conversation_record = record
            try:
                details_path.parent.mkdir(parents=True, exist_ok=True)
                details_path.write_text(
                    json.dumps(record, indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001 - elevenlabs.py:304-305
                logger.warning(
                    "Failed to write conversation details to %s: %s", details_path, exc
                )

        outcome = await self.transport.conversation_record(
            end_call_verdict,
            budget_s=END_CALL_CHECK_BUDGET_SECONDS,
            sleep=self._sleep,
            on_record=dump,
        )
        if outcome.verdict is None:
            logger.warning(
                "ElevenLabs had not finished processing the conversation after %.0fs; "
                "end_call could not be verified",
                outcome.waited_s,
            )
        return outcome.verdict

    async def _fetch_audio(self) -> None:
        """``_fetch_elevenlabs_audio`` (``elevenlabs.py:334-352``): 5 tries, 2 s doubling to 10 s."""
        delay = AUDIO_FETCH_FIRST_DELAY_S
        for attempt in range(AUDIO_FETCH_ATTEMPTS):
            try:
                audio = await self.transport.fetch_audio()
                path = self.out_dir / AUDIO_FILE
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(audio)
                return
            except Exception as exc:  # noqa: BLE001 - retried, then logged
                if attempt < AUDIO_FETCH_ATTEMPTS - 1:
                    await self._sleep(delay)
                    delay = min(delay * 2, AUDIO_FETCH_MAX_DELAY_S)
                else:
                    logger.warning(
                        "Failed to fetch ElevenLabs server audio after %s attempts: %s",
                        AUDIO_FETCH_ATTEMPTS,
                        exc,
                    )

    async def _teardown(self) -> None:
        """EVA's ``finally`` (``elevenlabs.py:233-263``), minus the bridge's ``stop_async``."""
        self.signal_conversation_ending()
        await self._cancel(self._keepalive_task)
        await self._cancel(self._wait_task)
        if not self._session_closed:
            # An error before the session part: the socket still has to go.
            self._session_closed = True
            with contextlib.suppress(Exception):
                await self.transport.close()
        self._save_latencies()
        self._save_clean_user_audio()
        logger.info(
            "Waiting %ss for assistant STT to finish last utterance...", STT_GRACE_S
        )
        await self._sleep(STT_GRACE_S)

    def _save_latencies(self) -> None:
        if self.bridge is None:
            return
        latencies = self.bridge.get_latencies()
        if not latencies:
            return
        path = self.out_dir / LATENCIES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "latencies": latencies,
                    "mean": sum(latencies) / len(latencies),
                    "max": max(latencies),
                    "count": len(latencies),
                },
                f,
                indent=2,
            )

    def _save_clean_user_audio(self) -> None:
        """``_save_clean_user_audio`` → ``save_pcm_as_wav`` (``audio_utils.py:203-237``)."""
        if not self._user_clean_audio:
            return
        path = self.out_dir / CLEAN_AUDIO_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(ELEVENLABS_OUTPUT_RATE)
                wav.writeframes(bytes(self._user_clean_audio))
        except Exception as exc:  # noqa: BLE001 - audio_utils.py:218-219
            logger.error("Error saving audio to %s: %s", path, exc)

    @staticmethod
    async def _cancel(task: asyncio.Task[None] | None) -> None:
        if task is None or task.done():
            return
        if task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
