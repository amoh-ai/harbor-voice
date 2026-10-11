"""The cascade adapter."""

from __future__ import annotations

from typing import override

from harbor_voice.adapter import VoiceAgentAdapter
from harbor_voice.audio.stages import StageMark
from harbor_voice.capabilities import Pipeline
from harbor_voice.cascade.transport import CascadeTransport


class CascadeAdapter(VoiceAgentAdapter):
    """Turn semantics over a composed STT -> LLM -> TTS pipeline.

    The only thing it adds over the base is stage enrichment: the base can see
    the two response boundaries, but a cascade owns the five seams between
    them, so the transport's marks are merged in.
    """

    @property
    @override
    def pipeline(self) -> Pipeline:
        return Pipeline.CASCADE

    @property
    def cascade(self) -> CascadeTransport:
        """The underlying pipeline, for ablation wiring and transcripts."""
        transport = self.transport
        if not isinstance(transport, CascadeTransport):
            raise TypeError(
                f"CascadeAdapter needs a CascadeTransport, got "
                f"{type(transport).__name__}"
            )
        return transport

    @override
    def _mark_stages(
        self, first_rx: float | None, last_rx: float | None
    ) -> tuple[StageMark, ...]:
        """Take the pipeline's own seams rather than inferring two from frames.

        A cascade knows when ASR finished and when the first token arrived; it
        does not need to guess them from when audio showed up.
        """
        marks = self.cascade.take_stages()
        if marks:
            return marks
        return super()._mark_stages(first_rx, last_rx)
