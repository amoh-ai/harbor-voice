"""UniversalSimulator, streamed: prepare/emit/finish and the wire encoder."""

from __future__ import annotations

import audioop

import numpy as np
import pytest

from harbor_voice.audio.format import AudioFormat
from harbor_voice.ids import PartyId
from harbor_voice.perturbation import G711Encoder
from harbor_voice.perturbation import GilbertElliottLoss
from harbor_voice.perturbation import Jitter
from harbor_voice.perturbation import MuLawCodec
from harbor_voice.perturbation import PerturbationPipeline
from harbor_voice.perturbation import SeedSet
from harbor_voice.providers import ToneTTS
from harbor_voice.simulator import ScriptContent
from harbor_voice.simulator import SimulatorError
from harbor_voice.simulator import UniversalSimulator
from harbor_voice.simulator.content import Content

pytestmark = pytest.mark.unit

CALLER = PartyId("caller")
LINE = "I need to change the date of my flight."


def _simulator(
    *, layers=(), encoder=None, fmt=None, seed: int = 7
) -> UniversalSimulator:
    kwargs = {"fmt": fmt} if fmt is not None else {}
    pipeline = PerturbationPipeline(
        list(layers), seeds=SeedSet.derive(seed), encoder=encoder, **kwargs
    )
    return UniversalSimulator(
        party=CALLER,
        content=ScriptContent(turns=[LINE]),
        tts=ToneTTS(),
        pipeline=pipeline,
        seeds=SeedSet.derive(seed),
        **kwargs,
    )


def _layers():
    return [MuLawCodec(), GilbertElliottLoss(rate=0.2, burst_duration_ms=60.0)]


class TestEmitMatchesTheBatchPath:
    async def test_frame_by_frame_equals_a_whole_utterance(self) -> None:
        batch = _simulator(layers=_layers())
        turn = await batch.next_utterance()
        assert turn is not None

        streamed = _simulator(layers=_layers())
        line = await streamed.prepare(Content(text=LINE))
        size = 960  # 20 ms of canonical 24 kHz PCM16
        frames = [
            streamed.emit(line.pcm[i : i + size]) for i in range(0, len(line.pcm), size)
        ]
        assert [f.pcm for f in frames] == [f.pcm for f in turn.frames]
        assert line.said == turn.said
        assert line.clean_digest == turn.clean_digest

    async def test_silence_goes_through_the_chain_and_advances_the_clock(self) -> None:
        sim = _simulator(layers=_layers())
        before = sim.pipeline.as_record().frames_processed
        frame = sim.emit(bytes(960), is_speech=False)
        assert sim.pipeline.as_record().frames_processed == before + 1
        assert frame.clean_ref is not None  # degraded, with its clean twin


class TestTheWireEncoder:
    async def test_g711_changes_format_but_not_duration(self) -> None:
        sim = _simulator(encoder=G711Encoder())
        line = await sim.prepare(LINE)
        frames = [sim.emit(line.pcm[i : i + 960]) for i in range(0, len(line.pcm), 960)]
        assert {f.fmt.encoding for f in frames} == {"mulaw"}
        assert {f.fmt.sample_rate_hz for f in frames} == {8000}
        total_out = sum(f.duration_s for f in frames)
        total_in = len(line.pcm) / (24000 * 2)
        assert abs(total_out - total_in) <= 1 / 8000
        for frame in frames:
            sim.pipeline.verify(frame)  # aligned by time, not bytes

    async def test_an_encoder_alone_makes_the_pipeline_unclean_and_is_recorded(
        self,
    ) -> None:
        sim = _simulator(encoder=G711Encoder())
        assert not sim.pipeline.is_clean
        sim.emit(bytes(960))
        record = sim.pipeline.as_record().as_dict()
        assert record["layers"]["L4.g711"]["params"] == {
            "encoding": "mulaw",
            "sample_rate_hz": 8000,
        }

    def test_streamed_encoding_equals_encoding_the_whole_buffer(self) -> None:
        """Resampler state carries across frames: chunking changes nothing."""
        rng = np.random.default_rng(3)
        pcm = (rng.standard_normal(16000) * 3000).astype("<i2").tobytes()
        whole, _ = audioop.ratecv(pcm, 2, 1, 16000, 8000, None)
        encoder = G711Encoder()
        fmt = AudioFormat(sample_rate_hz=16000)
        streamed = b"".join(
            encoder.encode(pcm[i : i + 640], fmt) for i in range(0, len(pcm), 640)
        )
        assert streamed == audioop.lin2ulaw(whole, 2)


class TestPartialLines:
    async def test_a_yielded_line_is_recorded_as_partial(self) -> None:
        sim = _simulator()
        line = await sim.prepare(LINE)
        sent = [sim.emit(line.pcm[i : i + 960]) for i in range(0, 960 * 3, 960)]
        utterance = sim.finish(line, sent, truncated=True)
        assert utterance.truncated
        assert utterance.said == LINE  # what it meant to say
        assert len(utterance.frames) == 3  # what went out
        assert sim.utterances[-1] is utterance


def test_a_pipeline_with_jitter_refuses_to_stream() -> None:
    sim = _simulator(layers=[Jitter(mean_ms=20.0, std_ms=5.0)])
    with pytest.raises(SimulatorError, match="frame-stream"):
        sim.emit(bytes(960))
