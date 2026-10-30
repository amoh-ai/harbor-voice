"""EVA's ``BotToBotAudioBridge``, ported: the caller's audio path and turn detection.

Every EVA timing metric comes from this class. ``turn_taking`` and
``response_speed`` are computed from the ``audio_start`` / ``audio_end``
events it logs, so it is ported line for line from
``eva/user_simulator/audio_bridge.py`` @039efa0a, as used with the
ElevenLabs simulator (``ElevenLabsAudioInterface``) against an S2S assistant.
Comments cite EVA's file:line. Its constants, its branches (including the
disabled ones) and its oddities are kept, because each of them moves a
timestamp:

- **Caller → assistant** (:meth:`Bridge._send_to_assistant`): the caller's
  16 kHz PCM is queued as it arrives and sent in 20 ms chunks on absolute
  real-time targets, each converted to 8 kHz μ-law with fresh ``ratecv``
  state. The first chunk of an utterance logs ``audio_start`` and sends
  ``user_speech_start``. The utterance ends when a leftover partial chunk has
  waited 600 ms (it is padded and sent), or, if the audio drained on a chunk
  boundary, 600 ms after the caller's end-of-utterance cue
  (:meth:`Bridge.notify_user_utterance_complete`). ``audio_end`` is
  back-dated to the last *full* chunk sent; ``user_speech_stop`` carries the
  event-loop time, not epoch time, as EVA's does.
- **Silence to the assistant.** Once the assistant has spoken and stopped, the
  bridge streams 20 ms silence frames while waiting for the caller. While
  waiting for the *assistant* (the caller has stopped), the input stream
  instead sends **18** silence frames at once on every 250 ms tick in which no
  assistant audio arrived. Nothing is sent before the assistant first speaks.
- **Assistant → caller** (:meth:`Bridge._continuous_input_stream`): the
  assistant's μ-law is buffered and handed to the caller every 250 ms, padded
  with μ-law silence when the assistant is quiet, and *not* padded (so
  shorter, possibly empty) while it is speaking. That is what EVA's
  ElevenLabs SDK sends, as ``user_audio_chunk``.
- **Assistant end** is detected on those 250 ms ticks once no media has
  arrived for 200 ms, back-dated to the last chunk received, and followed by
  200 ms of catch-up silence to the assistant (paced 20 ms, on the input
  stream's task).

What harbor replaces is only the transport. EVA's bridge talks to the
assistant server over a local WebSocket; here the "socket" is the
:class:`AssistantWire` the recorder implements, with direct calls, and the
assistant's media arrives by :meth:`Bridge.receive_media` instead of a
receive loop. The caller's audio arrives by :meth:`Bridge.output`, called
from the call loop rather than from the ElevenLabs SDK's thread.

The perturbation seam is EVA's: a :class:`Perturbator` applied to caller PCM
in :meth:`Bridge.output` (the clean copy recorded separately) and supplying
ambient noise in place of silence. ``None`` (the default) is EVA's
unperturbed path.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Protocol

from eva_adapter.audio import ASSISTANT_SAMPLE_RATE
from eva_adapter.audio import PCM_SAMPLE_WIDTH
from eva_adapter.audio import SEND_CHUNK_DURATION_MS
from eva_adapter.audio import SEND_CHUNK_SIZE_PCM
from eva_adapter.audio import pcm16k_to_mulaw8k
from eva_adapter.audio import white_noise_pcm
from eva_adapter.events import EventLog

logger = logging.getLogger(__name__)

# Timing constants (audio_bridge.py:44-70).
SILENCE_DETECTION_THRESHOLD_S = 0.2
"""200 ms without assistant media ends the assistant's audio."""
USER_END_DETECTION_DELAY_INTERVALS = 30
"""600 ms (30 x 20 ms) before a caller utterance is called finished."""
USER_CATCHUP_SILENCE_CHUNKS = 0
"""No catch-up silence after the caller stops (EVA: let the VAD see real gaps)."""
USER_TRAILING_SILENCE_CHUNKS = 0
"""The trailing white-noise block, disabled in EVA @039efa0a (was 100 = 2 s)."""
ASSISTANT_CATCHUP_SILENCE_CHUNKS = 10
"""200 ms of silence to the assistant after its audio ends."""
ASSISTANT_SILENCE_BURST_CHUNKS = 18
"""Silence frames sent per 250 ms tick while waiting for the assistant
(``audio_bridge.py:669-671``: "250ms / 20ms = 12.5, round up to 18")."""
FAST_POLL_TIMEOUT_S = 0.005
NORMAL_POLL_TIMEOUT_S = 0.01
IDLE_POLL_TIMEOUT_S = 0.1

# Logging intervals, in chunks (audio_bridge.py:72-76).
LOG_INTERVAL_SILENCE = 50
LOG_INTERVAL_AUDIO_SEND = 200
LOG_INTERVAL_AUDIO_RECV = 100
LOG_INTERVAL_INPUT_STREAM = 4

DISCONNECT_REASON = "elevenlabs_disconnect"
"""The end reason when the assistant side closes during the call
(``audio_bridge.py:120``; EVA's name, whichever side actually dropped)."""


class AssistantWire(Protocol):
    """The bridge's connection to the assistant server.

    EVA's is a WebSocket carrying Twilio-framed JSON (``audio_bridge.py:185-206``,
    ``:414-416``, ``:490-543``). Here it is the recorder, called directly;
    the order of calls is the order of EVA's messages.
    """

    @property
    def is_open(self) -> bool:
        """EVA's ``websocket.state == WebSocketState.OPEN``."""
        ...

    def send_media(self, mulaw: bytes) -> None:
        """A ``media`` message: 8 kHz μ-law."""
        ...

    def send_user_speech_start(self, timestamp_ms: str) -> None:
        """A ``user_speech_start`` message (epoch ms, as a string)."""
        ...

    def send_user_speech_stop(self, timestamp_ms: str) -> None:
        """A ``user_speech_stop`` message (event-loop ms, as a string: EVA's)."""
        ...

    def send_stop(self) -> None:
        """A ``stop`` message."""
        ...

    def close(self) -> None:
        """Close the socket from the bridge's side."""
        ...


class Perturbator(Protocol):
    """EVA's ``AudioPerturbator`` surface, as the bridge uses it
    (``perturbation.py:116-166``; :mod:`eva_adapter.vendored.perturbation`)."""

    @property
    def has_ambient_noise(self) -> bool: ...

    def apply(self, pcm_bytes: bytes) -> bytes: ...

    def get_ambient_chunk(self, n_bytes: int) -> bytes: ...


RecordCallback = Callable[[str, bytes], None]
"""``(source, pcm)``: the simulator's ``_record_audio``, which keeps only
``user_clean`` (the caller's ``record_audio``)."""
InputCallback = Callable[[bytes], Awaitable[None]]
"""Hands one chunk of assistant μ-law to the caller. Must not raise: EVA's
SDK callback catches its own send errors and ends the session."""


class Bridge:
    """Port of ``BotToBotAudioBridge`` (``audio_bridge.py:98-955``).

    Lifecycle, as EVA's simulator drives it: :meth:`start_async` (loops run,
    caller audio accepted), :meth:`start` (the caller is connected and hears
    the assistant), :meth:`stop` (the caller's session ended: loops wind
    down, the assistant stays connected), :meth:`stop_async` (the socket to
    the assistant closes).
    """

    INPUT_FRAMES_PER_BUFFER = 4000
    """250 ms at 16 kHz (``audio_bridge.py:109``)."""
    INPUT_CHUNK_DURATION = 0.25
    """The caller hears the assistant in 250 ms chunks (``audio_bridge.py:110``)."""

    def __init__(
        self,
        wire: AssistantWire,
        conversation_id: str,
        record_callback: RecordCallback | None = None,
        event_logger: EventLog | None = None,
        conversation_done_callback: Callable[[str], None] | None = None,
        perturbator: Perturbator | None = None,
        disconnect_reason: str = DISCONNECT_REASON,
    ) -> None:
        self._wire_target = wire
        self.conversation_id = conversation_id
        self.record_callback = record_callback
        self.event_logger = event_logger
        self.conversation_done_callback = conversation_done_callback
        self._perturbator = perturbator
        self.disconnect_reason = disconnect_reason

        self.websocket: AssistantWire | None = None
        self.running = False
        self.send_task: asyncio.Task[None] | None = None
        self.input_stream_task: asyncio.Task[None] | None = None

        self.input_callback: InputCallback | None = None
        self.send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.audio_buffer: asyncio.Queue[bytes] = asyncio.Queue()

        # Audio timing state (audio_bridge.py:151-171).
        self._user_audio_active = False
        self._assistant_audio_active = False
        self._user_audio_ended_time: float | None = None
        self._assistant_audio_ended_time: float | None = None
        self.assistant_audio_starts = 0
        self._last_user_audio_send_time: float | None = None
        self._user_turn_complete = False
        self._pending_trailing_noise_chunks = 0

        self._stopping = False
        self._send_errors_logged = 0
        self._latency_measurements: list[float] = []

        # The receive loop's state: EVA's _receive_from_assistant is a task;
        # here media is pushed to receive_media(), so its counter and its
        # `finally` live on the instance.
        self._audio_chunks_received = 0
        self._receive_done = False

    # ------------------------------------------------------------ lifecycle

    async def start_async(self) -> None:
        """Connect to the assistant and start the loops (``audio_bridge.py:180-213``).

        EVA sends ``connected`` and ``start`` messages here; the assistant
        server only reads a stream id from them, so the wire has no
        equivalent.
        """
        self.running = True
        self.websocket = self._wire_target
        self.send_task = asyncio.create_task(self._send_to_assistant())
        self.input_stream_task = asyncio.create_task(self._continuous_input_stream())

    def start(self, input_callback: InputCallback) -> None:
        """The caller is connected; it hears the assistant through ``input_callback``.

        EVA's SDK calls this from its thread once its socket is up
        (``audio_bridge.py:215-222``). Until then the input stream still runs
        and the assistant's audio is collected and dropped.
        """
        self.input_callback = input_callback

    def stop(self) -> None:
        """The caller's session ended (``audio_bridge.py:224-239``).

        Signals the end, but keeps the assistant connected so it can finish
        the last utterance; :meth:`stop_async` closes it.
        """
        self.running = False
        if self.conversation_done_callback:
            self.conversation_done_callback("session_ended")

    async def stop_async(self) -> None:
        """Cancel the loops and close the assistant socket (``audio_bridge.py:261-291``)."""
        self._stopping = True
        self.running = False
        # EVA cancels its receive task first; its `finally` runs then.
        self._finish_receive()
        for task in (self.send_task, self.input_stream_task):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self.websocket:
            with contextlib.suppress(Exception):
                self.websocket.send_stop()
            self.websocket.close()
            self.websocket = None

    # ------------------------------------------------------- caller audio in

    def output(self, audio: bytes) -> None:
        """Queue caller PCM (16 kHz) for the assistant (``audio_bridge.py:293-308``).

        The perturbed copy goes on the wire, the clean copy to the recording.
        """
        if self.running:
            clean_audio = audio
            if self._perturbator is not None:
                audio = self._perturbator.apply(audio)
            self.send_queue.put_nowait(audio)
            if self.record_callback:
                self.record_callback("user_clean", clean_audio)

    def notify_user_utterance_complete(self) -> None:
        """The caller finished its utterance (``audio_bridge.py:505-512``).

        EVA's simulator calls this on ElevenLabs' ``agent_response``. It arms
        the end of the caller's audio for when the queue drains on a chunk
        boundary, where the partial-chunk detector never fires.
        """
        self._user_turn_complete = True

    def is_assistant_playing(self) -> bool:
        """Assistant audio is currently active."""
        return self._assistant_audio_active

    @property
    def assistant_audio_ended_at(self) -> float | None:
        """Event-loop time the assistant's audio last ended."""
        return self._assistant_audio_ended_time

    def interrupt(self) -> None:
        """A no-op: the caller is a person, and keeps talking (``audio_bridge.py:323-330``)."""

    def get_latencies(self) -> list[float]:
        """Caller end → assistant start, seconds, per response (``audio_bridge.py:946-955``)."""
        return self._latency_measurements.copy()

    # --------------------------------------------------- assistant audio in

    def receive_media(self, mulaw: bytes) -> None:
        """One ``media`` message from the assistant (``audio_bridge.py:683-714``).

        The body of EVA's receive loop. After :meth:`stop` the next message
        ends the loop, as EVA's ``if not self.running: break`` does, and its
        ``finally`` runs.
        """
        if self._receive_done:
            return
        if not self.running:
            self._finish_receive()
            return
        if not mulaw:
            return
        if not self._assistant_audio_active:
            self._on_assistant_audio_start()
        self._audio_chunks_received += 1
        self._assistant_audio_ended_time = asyncio.get_running_loop().time()
        if self._audio_chunks_received % LOG_INTERVAL_AUDIO_RECV == 1:
            logger.debug(
                "<- received audio chunk %d (%d bytes)",
                self._audio_chunks_received,
                len(mulaw),
            )
        # μ-law goes to the caller as is (it is configured for μ-law 8 kHz).
        self.audio_buffer.put_nowait(mulaw)

    def wire_closed(self) -> None:
        """The assistant side closed the socket (``audio_bridge.py:723-744``)."""
        self._finish_receive()

    def _finish_receive(self) -> None:
        """The receive loop's ``finally`` (``audio_bridge.py:731-744``), once."""
        if self._receive_done:
            return
        self._receive_done = True
        if self._assistant_audio_active and self.event_logger:
            self._assistant_audio_active = False
            self.event_logger.log_audio_end("assistant")
            logger.info("Assistant audio END (connection closed)")
        if self.running and self.conversation_done_callback:
            logger.warning(
                "Assistant socket closed during the call; signalling disconnect"
            )
            self.conversation_done_callback(self.disconnect_reason)

    # ------------------------------------------------------------ turn state

    def _should_send_assistant_silence(self) -> bool:
        """Waiting for the assistant: the caller stopped more recently (``audio_bridge.py:360-373``)."""
        if self._user_audio_active or self._assistant_audio_active:
            return False
        if self._user_audio_ended_time is None:
            return False
        if self._assistant_audio_ended_time is not None:
            return self._user_audio_ended_time > self._assistant_audio_ended_time
        return True

    def _should_send_user_silence(self) -> bool:
        """Waiting for the caller: the assistant stopped more recently (``audio_bridge.py:375-388``)."""
        if self._user_audio_active or self._assistant_audio_active:
            return False
        if self._assistant_audio_ended_time is None:
            return False
        if self._user_audio_ended_time is not None:
            return self._assistant_audio_ended_time > self._user_audio_ended_time
        return True

    def _should_send_ambient_noise(self) -> bool:
        """Ambient noise streams whenever the caller is quiet (``audio_bridge.py:390-397``)."""
        return (
            not self._user_audio_active
            and self._perturbator is not None
            and self._perturbator.has_ambient_noise
        )

    # --------------------------------------------------------------- sending

    async def _send_audio_frame(self, mulaw_data: bytes) -> bool:
        """One frame to the assistant, if the socket is open (``audio_bridge.py:399-420``)."""
        if not self.websocket or not mulaw_data:
            return False
        if self._stopping or not self.websocket.is_open:
            return False
        try:
            self.websocket.send_media(mulaw_data)
            return True
        except Exception as e:  # noqa: BLE001 - EVA logs and carries on
            logger.warning("Error sending audio frame: %s", e)
            return False

    async def _send_silence_frame(self, chunk_size: int = SEND_CHUNK_SIZE_PCM) -> bool:
        """One frame of silence, or ambient noise (``audio_bridge.py:422-439``)."""
        if self._perturbator is not None and self._perturbator.has_ambient_noise:
            silence_pcm = self._perturbator.get_ambient_chunk(chunk_size)
        else:
            silence_pcm = b"\x00" * chunk_size
        silence_mulaw = pcm16k_to_mulaw8k(silence_pcm)
        if not silence_mulaw:
            return False
        return await self._send_audio_frame(silence_mulaw)

    async def _send_white_noise_frame(
        self, chunk_size: int = SEND_CHUNK_SIZE_PCM
    ) -> bool:
        """One frame of -45 dBFS white noise (``audio_bridge.py:441-454``); unused at 0 chunks."""
        noise_mulaw = pcm16k_to_mulaw8k(white_noise_pcm(chunk_size))
        if not noise_mulaw:
            return False
        return await self._send_audio_frame(noise_mulaw)

    async def _send_catchup_silence(self, source: str, num_chunks: int) -> None:
        """Silence frames 20 ms apart, no delay after the last (``audio_bridge.py:456-471``)."""
        send_interval = SEND_CHUNK_DURATION_MS / 1000.0
        for i in range(num_chunks):
            await self._send_silence_frame(chunk_size=SEND_CHUNK_SIZE_PCM)
            if i < num_chunks - 1:
                await asyncio.sleep(send_interval)

    # ----------------------------------------------------------- turn edges

    async def _on_user_audio_start(self) -> None:
        """The caller started speaking (``audio_bridge.py:473-503``)."""
        self._user_audio_active = True
        self._user_audio_ended_time = None
        self._pending_trailing_noise_chunks = 0
        timestamp_ms = time.time()  # epoch *seconds*, despite the name (EVA's)

        if self._assistant_audio_ended_time is not None:
            silence_duration = (
                asyncio.get_running_loop().time() - self._assistant_audio_ended_time
            )
            logger.info(
                "User audio START - stopping user silence after %.2fs", silence_duration
            )
            self._assistant_audio_ended_time = None
        if self.event_logger:
            self.event_logger.log_audio_start("simulated_user", timestamp_ms)
        logger.info("User audio START")

        if self.websocket and self.websocket.is_open:
            try:
                self.websocket.send_user_speech_start(
                    str(int(round(timestamp_ms * 1000)))
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Error sending user_speech_start event: %s", e)

    async def _on_user_audio_end(self, current_time: float) -> None:
        """The caller stopped speaking (``audio_bridge.py:514-560``)."""
        self._user_audio_ended_time = current_time
        self._user_audio_active = False
        self._user_turn_complete = False
        if self.event_logger:
            # Detection lags the real end by ~600 ms: stamp at the last full chunk.
            real_end_unix = (
                time.time() - (current_time - self._last_user_audio_send_time)
                if self._last_user_audio_send_time is not None
                else time.time()
            )
            self.event_logger.log_audio_end("simulated_user", real_end_unix)
        logger.info("User audio END")

        # EVA stamps this with the event-loop clock, not epoch time
        # (audio_bridge.py:538). The OpenAI Realtime server ignores it.
        if self.websocket and self.websocket.is_open:
            try:
                self.websocket.send_user_speech_stop(
                    str(int(round(current_time * 1000)))
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Error sending user_speech_stop event: %s", e)
        if USER_CATCHUP_SILENCE_CHUNKS > 0:
            await self._send_catchup_silence("assistant", USER_CATCHUP_SILENCE_CHUNKS)
        self._pending_trailing_noise_chunks = USER_TRAILING_SILENCE_CHUNKS

    def _on_assistant_audio_start(self) -> None:
        """The assistant started speaking (``audio_bridge.py:562-581``)."""
        loop_time = asyncio.get_running_loop().time()
        if self._user_audio_ended_time is not None:
            latency = loop_time - self._user_audio_ended_time
            self._latency_measurements.append(latency)
            logger.info("Assistant responded after %.2fs", latency)
            self._user_audio_ended_time = None
        if self._assistant_audio_ended_time is not None:
            self._assistant_audio_ended_time = None
        self._assistant_audio_active = True
        self.assistant_audio_starts += 1
        self._pending_trailing_noise_chunks = 0
        if self.event_logger:
            self.event_logger.log_audio_start("assistant")
        logger.info("Assistant audio START")

    async def _on_assistant_audio_end(self) -> None:
        """The assistant stopped speaking (``audio_bridge.py:583-602``)."""
        self._assistant_audio_active = False
        # On entry this still holds the last received chunk's loop time.
        loop_now = asyncio.get_running_loop().time()
        real_end_unix = (
            time.time() - (loop_now - self._assistant_audio_ended_time)
            if self._assistant_audio_ended_time is not None
            else time.time()
        )
        if self.event_logger:
            self.event_logger.log_audio_end("assistant", real_end_unix)
        self._assistant_audio_ended_time = loop_now
        logger.info("Assistant audio END (silence detected)")
        if (
            ASSISTANT_CATCHUP_SILENCE_CHUNKS > 0
            and not self._should_send_ambient_noise()
        ):
            await self._send_catchup_silence("user", ASSISTANT_CATCHUP_SILENCE_CHUNKS)

    # ----------------------------------------------------------------- loops

    async def _continuous_input_stream(self) -> None:
        """Hand the caller the assistant's audio every 250 ms (``audio_bridge.py:604-681``)."""
        samples_per_chunk = int(ASSISTANT_SAMPLE_RATE * self.INPUT_CHUNK_DURATION)
        loop = asyncio.get_running_loop()
        consecutive_empty_chunks = 0

        while self.running:
            start_time = loop.time()

            audio_chunk = b""
            try:
                while len(audio_chunk) < samples_per_chunk:
                    remaining_time = self.INPUT_CHUNK_DURATION - (
                        loop.time() - start_time
                    )
                    if consecutive_empty_chunks > 0:
                        timeout = FAST_POLL_TIMEOUT_S
                    else:
                        timeout = max(NORMAL_POLL_TIMEOUT_S, remaining_time)
                    try:
                        chunk = await asyncio.wait_for(
                            self.audio_buffer.get(), timeout=timeout
                        )
                        audio_chunk += chunk
                        consecutive_empty_chunks = 0
                    except TimeoutError:
                        if (
                            consecutive_empty_chunks > 0
                            and remaining_time > NORMAL_POLL_TIMEOUT_S
                        ):
                            continue
                        break
            except Exception as e:  # noqa: BLE001
                logger.error("Error getting audio from buffer: %s", e)

            assistant_silence = False
            if self._assistant_audio_active and self._assistant_audio_ended_time:
                current_time = loop.time()
                if (
                    current_time - self._assistant_audio_ended_time
                    > SILENCE_DETECTION_THRESHOLD_S
                ):
                    await self._on_assistant_audio_end()
            else:
                # Pad with μ-law silence (0xFF) only when the assistant is quiet.
                if len(audio_chunk) < samples_per_chunk:
                    padding_needed = samples_per_chunk - len(audio_chunk)
                    consecutive_empty_chunks += 1
                    audio_chunk += b"\xff" * padding_needed
                    if padding_needed == samples_per_chunk:
                        assistant_silence = True

            if self.input_callback:
                await self.input_callback(audio_chunk)

            if (
                assistant_silence
                and self._should_send_assistant_silence()
                and not self._should_send_ambient_noise()
            ):
                for _ in range(ASSISTANT_SILENCE_BURST_CHUNKS):
                    await self._send_silence_frame(chunk_size=SEND_CHUNK_SIZE_PCM)
                if consecutive_empty_chunks % LOG_INTERVAL_INPUT_STREAM == 1:
                    logger.debug("Sending silence assistant")

            elapsed = loop.time() - start_time
            sleep_time = max(0, self.INPUT_CHUNK_DURATION - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _send_to_assistant(self) -> None:
        """Send caller audio in 20 ms real-time chunks (``audio_bridge.py:746-944``)."""
        loop = asyncio.get_running_loop()
        audio_chunks_sent = 0
        pcm_chunk_size = SEND_CHUNK_SIZE_PCM
        send_interval = SEND_CHUNK_DURATION_MS / 1000.0

        pending_audio = b""
        stream_start_time: float | None = None
        silence_start_time: float | None = None
        silence_chunks_sent = 0
        next_send_time = loop.time()

        while self.running:
            try:
                current_time = loop.time()

                time_until_next_send = max(0, next_send_time - current_time)
                if (
                    self._user_audio_ended_time is not None
                    or self._assistant_audio_ended_time is not None
                    or pending_audio
                ):
                    timeout = max(0.001, time_until_next_send)
                else:
                    timeout = IDLE_POLL_TIMEOUT_S
                try:
                    pcm_audio = await asyncio.wait_for(
                        self.send_queue.get(), timeout=timeout
                    )
                    pending_audio += pcm_audio
                    # A new stream after idle or silence restarts the schedule.
                    if stream_start_time is None or not self._user_audio_active:
                        stream_start_time = loop.time()
                        next_send_time = stream_start_time
                        audio_chunks_sent = 0
                        silence_start_time = None
                        silence_chunks_sent = 0
                except TimeoutError:
                    pass

                current_time = loop.time()

                if (
                    len(pending_audio) >= pcm_chunk_size
                    and current_time >= next_send_time
                ):
                    chunk = pending_audio[:pcm_chunk_size]
                    pending_audio = pending_audio[pcm_chunk_size:]

                    if self.websocket:
                        if not self._user_audio_active:
                            await self._on_user_audio_start()

                        mulaw_audio = pcm16k_to_mulaw8k(chunk)
                        if mulaw_audio and await self._send_audio_frame(mulaw_audio):
                            audio_chunks_sent += 1
                            self._last_user_audio_send_time = current_time
                            assert stream_start_time is not None
                            next_send_time = stream_start_time + (
                                audio_chunks_sent * send_interval
                            )
                            if audio_chunks_sent % LOG_INTERVAL_AUDIO_SEND == 0:
                                logger.debug(
                                    "-> sent audio chunk %d", audio_chunks_sent
                                )

                elif len(pending_audio) > 0 and len(pending_audio) < pcm_chunk_size:
                    # A partial chunk: wait for more, or end the utterance after 600 ms.
                    time_since_last_send = current_time - next_send_time + send_interval
                    if (
                        time_since_last_send
                        >= send_interval * USER_END_DETECTION_DELAY_INTERVALS
                    ):
                        if self.websocket and pending_audio:
                            original_len = len(pending_audio)
                            aligned_len = (
                                original_len // PCM_SAMPLE_WIDTH
                            ) * PCM_SAMPLE_WIDTH
                            padded_chunk = pending_audio[:aligned_len] + b"\x00" * (
                                pcm_chunk_size - aligned_len
                            )
                            mulaw_audio = pcm16k_to_mulaw8k(padded_chunk)
                            if mulaw_audio and await self._send_audio_frame(
                                mulaw_audio
                            ):
                                audio_chunks_sent += 1
                                logger.info(
                                    "Sent padded chunk (%d bytes padded to %d) - end of utterance",
                                    original_len,
                                    pcm_chunk_size,
                                )
                                pending_audio = b""
                                if self._user_audio_active:
                                    await self._on_user_audio_end(current_time)
                                    stream_start_time = None
                                    next_send_time = current_time + send_interval

                elif (
                    self._user_audio_active
                    and self._user_turn_complete
                    and not pending_audio
                    and self.send_queue.empty()
                ):
                    # Drained on a chunk boundary after the caller's cue: end
                    # after the same detection delay.
                    time_since_last_send = current_time - next_send_time + send_interval
                    if (
                        time_since_last_send
                        >= send_interval * USER_END_DETECTION_DELAY_INTERVALS
                    ):
                        await self._on_user_audio_end(current_time)
                        stream_start_time = None
                        next_send_time = current_time + send_interval

                # The trailing white-noise block (0 chunks in EVA @039efa0a).
                if (
                    self._pending_trailing_noise_chunks > 0
                    and not pending_audio
                    and not self._user_audio_active
                ):
                    if silence_start_time is None:
                        silence_start_time = current_time
                        silence_chunks_sent = 0
                        next_send_time = silence_start_time
                    if (
                        current_time >= next_send_time
                        and await self._send_white_noise_frame()
                    ):
                        self._pending_trailing_noise_chunks -= 1
                        silence_chunks_sent += 1
                        next_send_time = silence_start_time + (
                            silence_chunks_sent * send_interval
                        )
                        if self.record_callback:
                            aligned_silence = b"\x00" * pcm_chunk_size
                            self.record_callback("assistant", aligned_silence)
                            self.record_callback("user_clean", aligned_silence)

                # Silence (or ambient noise) while waiting for the caller.
                elif (
                    self._should_send_ambient_noise()
                    or self._should_send_user_silence()
                ):
                    if silence_start_time is None:
                        silence_start_time = current_time
                        silence_chunks_sent = 0
                        next_send_time = silence_start_time

                    can_send = (
                        self._should_send_ambient_noise()
                        or self._assistant_audio_ended_time is not None
                    )
                    if current_time >= next_send_time and can_send:
                        silence_pcm = b"\x00" * pcm_chunk_size
                        if await self._send_silence_frame():
                            silence_chunks_sent += 1
                            next_send_time = silence_start_time + (
                                silence_chunks_sent * send_interval
                            )
                            # Recorded only once sent, so a retry cannot double it.
                            if self.record_callback:
                                self.record_callback("user_clean", silence_pcm)
                            if silence_chunks_sent % LOG_INTERVAL_SILENCE == 0:
                                actual_elapsed = current_time - silence_start_time
                                expected_elapsed = silence_chunks_sent * send_interval
                                logger.debug(
                                    "Sending silence user: chunks=%d actual=%.3fs expected=%.3fs",
                                    silence_chunks_sent,
                                    actual_elapsed,
                                    expected_elapsed,
                                )
                else:
                    if silence_start_time is not None:
                        silence_start_time = None
                        silence_chunks_sent = 0

                # Sleep until the next scheduled send.
                if (
                    not pending_audio
                    and self._user_audio_ended_time is None
                    and self._assistant_audio_ended_time is None
                    and not self._should_send_ambient_noise()
                ):
                    await asyncio.sleep(NORMAL_POLL_TIMEOUT_S)
                else:
                    sleep_time = max(0, next_send_time - loop.time())
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

            except Exception as e:  # noqa: BLE001 - EVA logs and keeps the loop alive
                if self.running:
                    logger.error("Error sending to assistant: %s", e)

        # Whatever is left goes out on shutdown, unpadded (audio_bridge.py:937-944).
        if pending_audio and self.websocket:
            try:
                mulaw_audio = pcm16k_to_mulaw8k(pending_audio)
                if mulaw_audio and await self._send_audio_frame(mulaw_audio):
                    logger.info("Sent final %d bytes on shutdown", len(pending_audio))
            except Exception as e:  # noqa: BLE001
                logger.warning("Error sending final audio: %s", e)
