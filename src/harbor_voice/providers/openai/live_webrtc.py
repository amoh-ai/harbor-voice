"""The WebRTC connection under an OpenAI Live session (the ``live`` extra).

tau2's ``OpenAILiveProvider`` media, kept as it is: a 48 kHz mono input track
that clocks queued caller audio out in 20 ms frames (zero-filled when there is
none, never gated on events), remote audio resampled to 24 kHz PCM16, events
as JSON on one data channel, audio NACK on, no ICE servers. Session creation
is the documented ``POST /v1/live/sessions`` with the SDP offer, with only the
``Authorization`` header (see :mod:`~harbor_voice.providers.openai.live`).
"""

from __future__ import annotations

import asyncio
import audioop
import contextlib
import json
from collections.abc import Callable
from fractions import Fraction
from typing import Any
from typing import override

import httpx
import numpy as np
from aiortc import AudioStreamTrack
from aiortc import RTCConfiguration
from aiortc import RTCPeerConnection
from aiortc import RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcrtpparameters import RTCRtcpFeedback
from aiortc.rtcrtpreceiver import NackGenerator
from aiortc.sdp import SessionDescription
from av import AudioFrame
from av.audio.resampler import AudioResampler

OUTPUT_RATE = 24_000
CREATE_TIMEOUT_S = 25.0
CHANNEL_TIMEOUT_S = 20.0


class _InputTrack(AudioStreamTrack):
    """Clock queued PCM at 20 ms intervals (tau2's ``_LiveInputAudioTrack``)."""

    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        self._resample_state: Any = None
        self._timestamp = 0
        self._started_at: float | None = None
        self.sample_rate = 48_000
        self.samples_per_frame = 960

    def append(self, audio: bytes, sample_rate: int = OUTPUT_RATE) -> None:
        if not audio:
            return
        resampled, self._resample_state = audioop.ratecv(
            audio, 2, 1, sample_rate, self.sample_rate, self._resample_state
        )
        self._buffer.extend(resampled)

    @override
    async def recv(self) -> AudioFrame:
        loop = asyncio.get_running_loop()
        if self._started_at is None:
            self._started_at = loop.time()
        target = self._started_at + self._timestamp / self.sample_rate
        await asyncio.sleep(max(0.0, target - loop.time()))
        size = self.samples_per_frame * 2
        audio = bytes(self._buffer[:size]).ljust(size, b"\x00")
        del self._buffer[:size]
        frame = AudioFrame.from_ndarray(
            np.frombuffer(audio, dtype=np.int16).reshape(1, self.samples_per_frame),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._timestamp
        self._timestamp += self.samples_per_frame
        return frame


class WebRTCConnection:
    """One peer connection; see :class:`~harbor_voice.providers.openai.live.LiveConnection`."""

    def __init__(
        self,
        on_event: Callable[[dict[str, Any]], None],
        on_audio: Callable[[bytes], None],
        on_error: Callable[[str], None],
    ) -> None:
        self._on_event = on_event
        self._on_audio = on_audio
        self._on_error = on_error
        self._peer: RTCPeerConnection | None = None
        self._channel: Any = None
        self._track: _InputTrack | None = None
        self._channel_open = asyncio.Event()
        self._audio_tasks: set[asyncio.Task[None]] = set()
        self._closing = False

    async def start(
        self, session: dict[str, Any], *, api_key: str, url: str
    ) -> dict[str, Any]:
        peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        self._peer = peer
        self._track = _InputTrack()
        peer.addTrack(self._track)
        # aiortc 1.15 has no public audio NACK switch; tau2 enables the
        # receiver's existing generator the same way.
        peer.getReceivers()[0]._RTCRtpReceiver__nack_generator = NackGenerator()  # ty: ignore[unresolved-attribute]
        channel = peer.createDataChannel("")
        self._channel = channel

        @channel.on("open")
        def on_open() -> None:
            self._channel_open.set()

        @channel.on("message")
        def on_message(message: str | bytes) -> None:
            self._on_event(json.loads(message))

        @peer.on("connectionstatechange")
        def on_state() -> None:
            if not self._closing and peer.connectionState in {"failed", "closed"}:
                self._on_error(f"peer connection {peer.connectionState}")

        @peer.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "audio":
                task = asyncio.create_task(self._receive(track))
                self._audio_tasks.add(task)
                task.add_done_callback(self._audio_done)

        offer = await peer.createOffer()
        description = SessionDescription.parse(offer.sdp)
        for media in description.media:
            if media.kind == "audio":
                for codec in media.rtp.codecs:
                    codec.rtcpFeedback.append(RTCRtcpFeedback(type="nack"))
        await peer.setLocalDescription(
            RTCSessionDescription(sdp=str(description), type=offer.type)
        )
        async with httpx.AsyncClient(timeout=CREATE_TIMEOUT_S) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "session": session,
                    "transport": {"type": "webrtc", "sdp": peer.localDescription.sdp},
                },
            )
        if not 200 <= response.status_code < 300:
            raise RuntimeError(
                f"Live session creation failed (HTTP {response.status_code}): "
                f"{response.text[:600]}"
            )
        created = response.json()
        await peer.setRemoteDescription(
            RTCSessionDescription(sdp=created["transport"]["sdp"], type="answer")
        )
        await asyncio.wait_for(self._channel_open.wait(), CHANNEL_TIMEOUT_S)
        return created["session"]

    def send(self, event: dict[str, Any]) -> None:
        if self._channel is None or self._channel.readyState != "open":
            raise ConnectionError("Live data channel is not open")
        self._channel.send(json.dumps(event))

    def append(self, pcm: bytes) -> None:
        if self._track is None:
            raise ConnectionError("Live audio track is not connected")
        self._track.append(pcm)

    async def close(self) -> None:
        self._closing = True
        for task in list(self._audio_tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._peer is not None:
            await self._peer.close()
            self._peer = None

    async def _receive(self, track: Any) -> None:
        resampler = AudioResampler(format="s16", layout="mono", rate=OUTPUT_RATE)
        while not self._closing:
            try:
                frame = await track.recv()
            except MediaStreamError:
                if not self._closing:
                    raise ConnectionError("Live remote audio track ended") from None
                return
            for converted in resampler.resample(frame):
                pcm = (
                    converted.to_ndarray()
                    .reshape(-1)
                    .astype(np.int16, copy=False)
                    .tobytes()
                )
                self._on_audio(pcm)

    def _audio_done(self, task: asyncio.Task[None]) -> None:
        self._audio_tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            self._on_error(f"audio receive failed: {error}")
