"""The all-Bedrock stack — STT, LLM and TTS without Polly or Transcribe.

A fallback that turned out to be better than expected. For an AWS account
that grants Bedrock but **not** Polly or Transcribe, the cascade's STT and
TTS slots have no provider. Bedrock alone can fill them:

===========  ==========================  ==========================
Slot         Model                       How
===========  ==========================  ==========================
``stt``      ``mistral.voxtral-*``       Converse, ``audio`` block
``llm``      any Bedrock text model      ConverseStream
``tts``      ``amazon.nova-2-5-sonic``   bidirectional stream
===========  ==========================  ==========================

Measured on arrival, against audio this stack produced itself:

===========================  =====
Condition                    WER
===========================  =====
clean                        0.000
telephony (SNR 15, 2% loss)  0.000
noisy (SNR 5, 5% loss)       0.111
severe (SNR 0, 15% loss)     0.111
===========================  =====

The instrument responds to the channel, which is the only property that makes
it usable for a robustness curve. The flat top end is a floor effect on a
nine-word sentence, where a single dropped word *is* 0.111.

**What this stack costs us.** One STT vendor, not two. The reference-cascade ablation —
hold the LLM fixed, swap the STT — needs something to swap *to*, and without
Transcribe there is only voxtral. The mini and small variants are not
an ablation: they share a family and, on this audio, transcribed identically.
So the axis is present in the architecture and empty in practice until a
second vendor is reachable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.audio.frame import AudioFrame
from harbor_voice.ids import PartyId
from harbor_voice.providers.aws.nova_sonic import MAX_SESSION_S
from harbor_voice.providers.aws.nova_sonic import NOVA_OUTPUT_FORMAT
from harbor_voice.providers.aws.nova_sonic import NovaSonicTransport
from harbor_voice.providers.aws.nova_sonic import _classify
from harbor_voice.providers.base import ProviderError
from harbor_voice.providers.base import ProviderFailure
from harbor_voice.perturbation.persona import Gender
from harbor_voice.perturbation.persona import VoiceSelection
from harbor_voice.providers.base import ProviderSpec
from harbor_voice.providers.base import Usage
from harbor_voice.providers.capabilities import Modality
from harbor_voice.providers.capabilities import ProviderCapabilities
from harbor_voice.providers.protocols import Transcript
from harbor_voice.transport.base import SessionConfig

VOXTRAL_MINI = "mistral.voxtral-mini-3b-2507"
VOXTRAL_SMALL = "mistral.voxtral-small-24b-2507"
NOVA_TTS_MODEL = "amazon.nova-2-5-sonic"
DEFAULT_LLM = "amazon.nova-pro-v1:0"
"""ON_DEMAND and streaming in us-east-1.

Claude on Bedrock is **not** a usable default here: the bare model id
raises "Invocation ... with on-demand throughput isn't supported. Retry
with ... an inference profile", so it needs a ``us.``-prefixed profile id.
Pass one explicitly if you want Claude; the failure is classified as a
config error so it fails the job rather than every trial separately.
"""

TRANSCRIBE_PROMPT = (
    "Transcribe the audio verbatim. Output only the transcript, with no "
    "commentary, no quotation marks and no explanation. If the audio is "
    "unintelligible, output nothing."
)

VERBATIM_PROMPT = (
    "You are a text-to-speech engine. Say the user's text back word for word, "
    "exactly as written. Add nothing. Omit nothing."
)

_AWS_EXTRA = "aws"


def _runtime(region: str | None) -> Any:
    """A ``bedrock-runtime`` client, imported lazily."""
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - exercised by the message test
        raise ProviderError(
            ProviderFailure.MALFORMED,
            f"needs boto3. Install:  pip install 'harbor-voice[{_AWS_EXTRA}]'",
            provider="aws.bedrock",
        ) from exc
    return boto3.client("bedrock-runtime", region_name=region)


def _wav(pcm: bytes, fmt: AudioFormat) -> bytes:
    from harbor_voice.trajectory.render import to_wav

    return to_wav(pcm, fmt)


# ------------------------------------------------------------------- STT


@dataclass(slots=True)
class VoxtralSTT:
    """Speech to text through Bedrock's Converse API.

    **One-shot, not streaming.** Converse takes a whole audio block and
    returns a whole transcript, so there are no partials and no meaningful
    first-byte mark. ``streams=False`` in the spec says so, and the
    applicability machinery turns that into an abstention rather than a
    fabricated latency.

    That makes it a good **measuring** STT — the reference transcriber runs
    offline over stored audio, where first-byte latency is meaningless — and a
    poor live cascade STT for anything that gates on partials.
    """

    model: str = VOXTRAL_MINI
    region: str | None = None
    prompt: str = TRANSCRIBE_PROMPT
    fmt: AudioFormat = CANONICAL
    name: str = "aws.voxtral"
    usage: Usage = field(default_factory=Usage)
    _client: Any = None

    async def transcribe(
        self, frames: Sequence[AudioFrame]
    ) -> AsyncIterator[Transcript]:
        """Yield exactly one final transcript. Never a partial."""
        pcm = b"".join(f.pcm for f in frames)
        duration = self.fmt.duration_s(len(pcm))
        if not pcm:
            yield Transcript(text="", is_final=True, audio_duration_s=0.0)
            return

        fmt = frames[0].fmt if frames else self.fmt
        text, usage = await asyncio.to_thread(self._converse, _wav(pcm, fmt))
        self.usage = usage
        yield Transcript(text=text, is_final=True, audio_duration_s=duration)

    def _converse(self, wav: bytes) -> tuple[str, Usage]:
        if self._client is None:
            self._client = _runtime(self.region)
        try:
            response = self._client.converse(
                modelId=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"audio": {"format": "wav", "source": {"bytes": wav}}},
                            {"text": self.prompt},
                        ],
                    }
                ],
                inferenceConfig={"maxTokens": 512, "temperature": 0.0},
            )
        except Exception as exc:
            raise _classify(exc, self.model, "aws.voxtral") from exc

        blocks = response["output"]["message"]["content"]
        text = "".join(b.get("text", "") for b in blocks).strip()
        counts = response.get("usage", {})
        return text, _usage_from_converse(counts)


def _usage_from_converse(counts: dict[str, Any]) -> Usage:
    from harbor.models.metric import UsageInfo

    return Usage(
        tokens=UsageInfo(
            prompt_tokens=int(counts.get("inputTokens", 0) or 0),
            completion_tokens=int(counts.get("outputTokens", 0) or 0),
            cache_tokens=int(counts.get("cacheReadInputTokens", 0) or 0),
            cost_usd=0.0,
        )
    )


# ------------------------------------------------------------------- LLM


@dataclass(slots=True)
class BedrockLLM:
    """Streamed text through Bedrock's ConverseStream.

    Genuinely streaming, so ``llm_first_token`` is a real mark rather than a
    restatement of total latency.
    """

    model: str = DEFAULT_LLM
    region: str | None = None
    system: str = (
        "You are a helpful voice assistant. Keep replies to one or two sentences."
    )
    temperature: float = 0.2
    max_tokens: int = 512
    name: str = "aws.bedrock-llm"
    usage: Usage = field(default_factory=Usage)
    _client: Any = None

    async def respond(self, prompt: str, history: Sequence[str]) -> AsyncIterator[str]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def pump() -> None:
            try:
                for chunk in self._stream(prompt, history):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        loop = asyncio.get_running_loop()
        worker = asyncio.create_task(asyncio.to_thread(pump))
        try:
            while (chunk := await queue.get()) is not None:
                yield chunk
        finally:
            await worker

    def _stream(self, prompt: str, history: Sequence[str]):  # noqa: ANN202 - generator
        if self._client is None:
            self._client = _runtime(self.region)
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": [{"text": h}]}
            for i, h in enumerate(history)
        ]
        messages.append({"role": "user", "content": [{"text": prompt}]})
        try:
            response = self._client.converse_stream(
                modelId=self.model,
                messages=messages,
                system=[{"text": self.system}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
        except Exception as exc:
            raise _classify(exc, self.model, "aws.bedrock-llm") from exc

        for event in response["stream"]:
            if "contentBlockDelta" in event:
                text = event["contentBlockDelta"]["delta"].get("text")
                if text:
                    yield text
            elif "metadata" in event:
                self.usage = _usage_from_converse(event["metadata"].get("usage", {}))


# ------------------------------------------------------------------- TTS


@dataclass(frozen=True, slots=True)
class SpokenAudio:
    """Audio, plus what the model says it actually said."""

    pcm: bytes
    spoken_text: str
    requested_text: str

    @property
    def is_verbatim(self) -> bool:
        """Whether the model read the text rather than paraphrasing it."""
        return _normalise(self.spoken_text) == _normalise(self.requested_text)


def _normalise(text: str) -> str:
    """Case, punctuation and whitespace are not paraphrase.

    The guard first fired on ``"Sure , I'll reschedule  your"`` versus
    ``"Sure ,  I'll reschedule your"`` — identical words, different spacing.
    A guard that cries wolf on whitespace gets switched off, and then it is
    not guarding anything.
    """
    import re

    stripped = re.sub(r"[^\w\s]", "", text).casefold()
    return " ".join(stripped.split())


class NovaSonicTTS:
    """Text to speech by coercing a conversational S2S model to read aloud.

    **This is a workaround and the docstring says so**, because the risk it
    carries is not obvious from the call site. Polly is a text-to-speech
    engine: give it a string and it speaks that string. Nova Sonic is a
    *conversational* model being instructed, in a system prompt, to behave
    like one. Instructions are not guarantees, and a paraphrase here would
    silently corrupt ``said`` — the reference side of said-vs-heard —
    making every WER in the run wrong in a direction nothing would reveal.

    So every synthesis is **checked**. Nova Sonic returns the text it spoke,
    and :meth:`synthesize_checked` compares it against what was requested.
    A mismatch is surfaced, not smoothed over. Measured over bring-up it read
    text back verbatim, but "it worked when we tried it" is not a property a
    benchmark can rest a reference transcript on.
    """

    def __init__(
        self,
        *,
        model: str = NOVA_TTS_MODEL,
        region: str | None = None,
        voice: str = "matthew",
        party: PartyId = PartyId("caller"),
        settle_s: float = 8.0,
    ) -> None:
        self.model = model
        self.region = region
        self.voice = voice
        self.party = party
        self.settle_s = settle_s
        self.name = "aws.nova-tts"
        self.last: SpokenAudio | None = None

    @property
    def audio_format(self) -> AudioFormat:
        return NOVA_OUTPUT_FORMAT

    # ---------------------------------------------------------- L1 voice

    def select_voice(self, selection: VoiceSelection) -> bool:
        """Adopt an L1 voice selection, or report that it cannot be served.

        Implements :class:`~harbor_voice.perturbation.persona.VoiceSelectable`.
        Nova Sonic takes a ``voiceId``, which covers gender and language, and
        the transport has always passed one. What it has **no** notion of is
        an *accent* as a separate axis — EVA's condition A varies french,
        indian, spanish and chinese accented English, and a voice catalogue
        indexed by language cannot express that.

        So an accent request returns ``False`` and the run records
        ``voice_honoured: 0``. That is the whole reason this returns a bool
        rather than raising: the run is still worth having, it simply may not
        claim the condition.

        An explicit ``voice_id`` is taken at face value — a task that pins a
        provider-specific voice has already decided this question.
        """
        if selection.voice_id is not None:
            self.voice = selection.voice_id
            return True
        if selection.accent is not None:
            return False
        voice = _VOICES.get((selection.language, selection.gender))
        if voice is None:
            return False
        self.voice = voice
        return True

    async def synthesize_checked(self, text: str) -> SpokenAudio:
        """Speak ``text`` and report what was actually said."""
        transport = NovaSonicTransport(
            party=self.party,
            model=self.model,
            region=self.region,
            voice=self.voice,
            system_prompt=VERBATIM_PROMPT,
            temperature=0.0,
        )
        await transport.open(SessionConfig(party=self.party))
        frames: list[AudioFrame] = []

        async def drain() -> None:
            async for frame in transport.rx():
                frames.append(frame)

        pump = asyncio.create_task(drain())
        try:
            await transport.send_text(text)
            # Silence is what commits the turn: the audio stream drives
            # endpointing, and with nothing flowing the model never replies.
            # Found the hard way — text alone produces only usage events.
            await self._flush_silence(transport)
            await asyncio.sleep(self.settle_s)
        finally:
            # close() drains its own receive pump first, so by the time it
            # returns the inbox has been closed and this drain ends on its own.
            await transport.close()
            try:
                await asyncio.wait_for(pump, timeout=5.0)
            except TimeoutError:
                pump.cancel()

        spoken = SpokenAudio(
            pcm=b"".join(f.pcm for f in frames),
            spoken_text=" ".join(transport.said).strip(),
            requested_text=text,
        )
        self.last = spoken
        return spoken

    async def _flush_silence(self, transport: NovaSonicTransport) -> None:
        size = CANONICAL.frame_bytes()
        for seq in range(60):
            await transport.tx(
                AudioFrame(
                    pcm=bytes(size),
                    party=self.party,
                    seq=seq,
                    t_produced=seq * 0.02,
                )
            )
            await asyncio.sleep(0.016)

    async def synthesize(self, text: str, *, party: PartyId) -> AsyncIterator[bytes]:
        """The :class:`TTSProvider` shape.

        Raises:
            ProviderError: If the model paraphrased. A silent paraphrase would
                corrupt the reference transcript, so this fails loudly rather
                than returning audio that does not match ``said``.
        """
        del party
        if not text:
            return
        spoken = await self.synthesize_checked(text)
        if not spoken.is_verbatim:
            raise ProviderError(
                ProviderFailure.MALFORMED,
                f"asked for {text!r} but the model said "
                f"{spoken.spoken_text!r}; an S2S model used as a TTS can "
                f"paraphrase, and a paraphrased reference silently corrupts "
                f"every said-vs-heard number in the run",
                provider="aws.nova-tts",
                model=self.model,
            )
        size = self.audio_format.frame_bytes()
        for start in range(0, len(spoken.pcm), size):
            yield spoken.pcm[start : start + size]


# ----------------------------------------------- the second recogniser


class NovaSonicSTT:
    """Speech to text from Nova Sonic's transcript of its **own input**.

    Nova Sonic reports what it heard (``textOutput`` with ``role: USER``)
    alongside what it said, and
    :class:`~harbor_voice.providers.aws.nova_sonic.NovaSonicTransport` has
    been collecting it since bring-up — that is why ``Stage.ASR_FINAL`` is an
    S2S-optional stage. This exposes it as an ordinary
    :class:`~harbor_voice.providers.protocols.STTProvider`.

    **Why it earns its place: it is a second vendor.**
    :meth:`~harbor_voice.providers.measurement.ReferenceStack.check_independence`
    refuses a reference transcriber that shares a provider with the agent's,
    because two recognisers from one vendor share an acoustic frontend and
    their errors cancel instead of revealing anything. With Voxtral (Mistral)
    and this (Amazon), both reachable through one Bedrock account, the
    reference stack is satisfiable in both directions:

    - a **cascade** agent on ``aws.voxtral``, measured by ``aws.nova-stt``
    - an **S2S** agent on ``aws.nova-sonic``, measured by ``aws.voxtral``

    An S2S agent measured by this would be refused, correctly, since it would
    be the same model grading its own hearing.

    Note:
        A full conversational session per transcript is an expensive way to
        run an ASR, and the model may answer the audio as well as transcribe
        it. Neither matters for the job this does: the reference transcriber
        runs offline over stored audio, where latency is meaningless, and the
        reply is discarded. ``streams=False`` says the first is not available
        rather than inventing a mark for it.
    """

    def __init__(
        self,
        *,
        model: str = NOVA_TTS_MODEL,
        region: str | None = None,
        party: PartyId = PartyId("reference"),
        settle_s: float = 4.0,
        fmt: AudioFormat = CANONICAL,
    ) -> None:
        self.model = model
        self.region = region
        self.party = party
        self.settle_s = settle_s
        self.fmt = fmt
        self.name = "aws.nova-stt"
        self.usage: Usage = Usage()

    async def transcribe(
        self, frames: Sequence[AudioFrame]
    ) -> AsyncIterator[Transcript]:
        """Yield exactly one final transcript. Never a partial."""
        duration = sum(f.duration_s for f in frames)
        if not frames:
            yield Transcript(text="", is_final=True, audio_duration_s=0.0)
            return

        text = await self._listen(frames)
        yield Transcript(text=text, is_final=True, audio_duration_s=duration)

    async def _listen(self, frames: Sequence[AudioFrame]) -> str:
        transport = NovaSonicTransport(
            party=self.party,
            model=self.model,
            region=self.region,
            system_prompt=LISTEN_PROMPT,
            temperature=0.0,
        )
        await transport.open(SessionConfig(party=self.party))

        async def drain() -> None:
            async for _frame in transport.rx():
                pass  # the reply is not wanted; only the input transcript is

        pump = asyncio.create_task(drain())
        try:
            for frame in frames:
                await transport.tx(frame)
                await asyncio.sleep(0.016)
            # Silence commits the turn, exactly as it does for the TTS path:
            # endpointing is driven by the audio stream, so a stream that
            # simply stops leaves the model waiting for more.
            await self._flush_silence(transport, start=len(frames))
            await asyncio.sleep(self.settle_s)
        finally:
            await transport.close()
            try:
                await asyncio.wait_for(pump, timeout=5.0)
            except TimeoutError:
                pump.cancel()

        self.usage = transport.usage
        return " ".join(transport.heard).strip()

    async def _flush_silence(
        self, transport: NovaSonicTransport, *, start: int
    ) -> None:
        size = CANONICAL.frame_bytes()
        for offset in range(40):
            seq = start + offset
            await transport.tx(
                AudioFrame(
                    pcm=bytes(size),
                    party=self.party,
                    seq=seq,
                    t_produced=seq * 0.02,
                )
            )
            await asyncio.sleep(0.016)


_VOICES: dict[tuple[str, Gender], str] = {
    ("en", Gender.UNSPECIFIED): "matthew",
    ("en", Gender.MALE): "matthew",
    ("en", Gender.FEMALE): "tiffany",
}
"""Nova Sonic voice ids by (language, gender).

Deliberately short and **verified** rather than transcribed from memory. A
mapping that names a voice the service does not have fails at session open,
mid-run, with a vendor error — far worse than :meth:`select_voice` returning
``False`` and the plan recording that the request went unserved. Entries are
added as they are confirmed against the live service.
"""


LISTEN_PROMPT = (
    "You are a transcription service. Listen and say nothing. Do not answer, "
    "acknowledge, or comment on what you hear."
)
"""The reply is discarded either way; this just keeps the session cheap."""


# -------------------------------------------------------------- registry


VOXTRAL_SPEC = ProviderSpec(
    name="aws.voxtral",
    modality=Modality.STT,
    capabilities=ProviderCapabilities(
        modality=Modality.STT,
        streams=False,
        emits_partials=False,
        word_timings=False,
        diarizes=False,
    ),
    credential_provider="aws",
    requires_extra=_AWS_EXTRA,
    default_model=VOXTRAL_MINI,
)

BEDROCK_LLM_SPEC = ProviderSpec(
    name="aws.bedrock-llm",
    modality=Modality.LLM,
    capabilities=ProviderCapabilities(modality=Modality.LLM, streams=True),
    credential_provider="aws",
    requires_extra=_AWS_EXTRA,
    default_model=DEFAULT_LLM,
)

NOVA_STT_SPEC = ProviderSpec(
    name="aws.nova-stt",
    modality=Modality.STT,
    capabilities=ProviderCapabilities(
        modality=Modality.STT,
        streams=False,
        emits_partials=False,
        word_timings=False,
        diarizes=False,
    ),
    credential_provider="aws",
    requires_extra=_AWS_EXTRA,
    default_model=NOVA_TTS_MODEL,
)
"""Registered under its own name, which is the point.

``check_independence`` compares provider names, so ``aws.nova-stt`` and
``aws.voxtral`` are independent while two models under one name are not. That
is also right on the merits: Voxtral is Mistral's recogniser and this is
Amazon's.
"""

NOVA_TTS_SPEC = ProviderSpec(
    name="aws.nova-tts",
    modality=Modality.TTS,
    capabilities=ProviderCapabilities(modality=Modality.TTS, streams=False),
    credential_provider="aws",
    requires_extra=_AWS_EXTRA,
    default_model=NOVA_TTS_MODEL,
)
"""``streams=False`` on purpose.

Audio does arrive incrementally from the stream, but this provider withholds
all of it until the verbatim check passes. A first-byte mark taken after a
whole-utterance round trip would be a fiction, and declaring the capability
is what the applicability gate trusts.
"""


def build_stt(*, model: str = VOXTRAL_MINI, **config: Any) -> VoxtralSTT:
    return VoxtralSTT(model=model, **config)


def build_llm(*, model: str = DEFAULT_LLM, **config: Any) -> BedrockLLM:
    return BedrockLLM(model=model, **config)


def build_tts(*, model: str = NOVA_TTS_MODEL, **config: Any) -> NovaSonicTTS:
    return NovaSonicTTS(model=model, **config)


def build_nova_stt(*, model: str = NOVA_TTS_MODEL, **config: Any) -> NovaSonicSTT:
    return NovaSonicSTT(model=model, **config)


REGISTRATIONS: tuple[tuple[ProviderSpec, Any], ...] = (
    (VOXTRAL_SPEC, build_stt),
    (BEDROCK_LLM_SPEC, build_llm),
    (NOVA_TTS_SPEC, build_tts),
    (NOVA_STT_SPEC, build_nova_stt),
)

MAX_TTS_SESSION_S = MAX_SESSION_S
