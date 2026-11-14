"""An omni model and a TTS behind the transport interface.

The omni model hears the caller's audio and writes the reply; the TTS speaks
it. Turn detection, the TTS seam and the stage marks are the cascade's own
(:class:`~harbor_voice.cascade.transport.CascadeTransport`); only hearing and
reasoning are one call instead of two. So there is no ASR seam: the agent's
own transcript of the caller does not exist, and said-vs-heard has no HEARD
side to read.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import override

from harbor_voice.audio.clock import SessionClock
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.audio.stages import Stage
from harbor_voice.capabilities import Capabilities
from harbor_voice.cascade.transport import CascadeError
from harbor_voice.cascade.transport import CascadeTransport
from harbor_voice.cascade.transport import _slot_name
from harbor_voice.ids import PartyId
from harbor_voice.providers.base import ProviderRef
from harbor_voice.providers.selfhosted.omni import OmniLLM
from harbor_voice.providers.selfhosted.omni import OmniTurn


class OmniCascadeTransport(CascadeTransport):
    """Caller audio -> omni model -> text -> TTS -> agent audio."""

    def __init__(
        self,
        *,
        omni: OmniLLM,
        tts: Any,
        party: PartyId,
        clock: SessionClock | None = None,
        tools: Any = None,
        **kwargs: Any,
    ) -> None:
        if tools is not None:
            raise CascadeError(
                f"{omni.name!r} is offered no tools: an omni cascade does not run "
                f"a tool loop, and a task that configured tools would be scored "
                f"against an agent that never saw them"
            )
        super().__init__(
            # No recogniser (the omni model hears), and the omni model in the
            # llm slot only so close() reaches it; _run_pipeline never calls either.
            stt=None,  # ty: ignore[invalid-argument-type]
            llm=omni,  # ty: ignore[invalid-argument-type]
            tts=tts,
            party=party,
            clock=clock,
            **kwargs,
        )
        self._omni = omni
        self._turns: list[OmniTurn] = []

    @property
    @override
    def capabilities(self) -> Capabilities:
        base = super().capabilities
        stages = base.exposes_stages - {Stage.ASR_FINAL, Stage.ASR_PARTIAL}
        return base.model_copy(
            update={
                "exposes_asr": False,
                "exposes_partials": False,
                "exposes_stages": frozenset(stages),
            }
        )

    @override
    async def _run_pipeline(self, audio: list[AudioFrame]) -> None:
        self._mark(Stage.VAD_SPEECH_END, self._clock.now())
        pcm = b"".join(frame.pcm for frame in audio)
        reply = await self._omni.reply(pcm, self._turns)
        # One call returns the whole reply: its first token and its end are
        # the same instant, as far as anything outside the model can see.
        now = self._clock.now()
        self._mark(Stage.LLM_FIRST_TOKEN, now)
        self._mark(Stage.LLM_DONE, now, text=reply.text)
        self._turns.append(OmniTurn(audio=pcm, reply=reply.text))
        self._history.extend(("", reply.text))
        async for frame in self._synthesize(reply.text):
            await self._outbound.put(frame)

    @property
    @override
    def ref(self) -> ProviderRef:
        return ProviderRef(
            provider="cascade",
            model="+".join(_slot_name(slot) for slot in (self._omni, self._tts)),
        )

    @property
    def omni_turns(self) -> Sequence[OmniTurn]:
        return tuple(self._turns)
