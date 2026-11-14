"""A cascade agent on tau-voice: refused unless the run opts in.

Once opted in, it is prompted as tau2 prompts one.

tau-voice scores realtime speech-to-speech agents, as tau2's leaderboard
does. tau2 can also run a cascade (STT -> LLM -> TTS, its ``livekit``
provider), and so can this adapter, but only when the run says so:
``allow_cascade`` in the driver's config (``tau-voice job-config
--allow-cascade``, or ``call_config`` in the call agent's kwargs). A cascade
is a different kind of agent, and a score for one must not land beside
speech-to-speech scores by accident.

tau2 gives a cascade its own instructions: it hears transcribed speech, so
it is told to expect misspellings and missing punctuation, and to answer
without markdown. The sidecar renders the speech-to-speech prompt; the
instructions at its head are swapped for the cascade's here, from tau2's
own text (``tau2/agent/discrete_time_audio_native_agent.py`` at ``2174a60``,
MIT), rather than changing the sidecar and its published images.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

from harbor_voice.capabilities import Pipeline
from harbor_voice.cascade.duplex import stt_refusal
from harbor_voice.trial import AgentSpec
from harbor_voice.trial import TrialConfigError

ALLOW_CASCADE = "allow_cascade"
"""The driver config key that lets a cascade agent run."""

AUDIO_NATIVE_VOICE_INSTRUCTION = """
You are a customer service agent handling a VOICE CALL with a customer.

# Important Voice Call Considerations

1. Respond naturally and conversationally as you would in a real phone call

2. Try to be helpful and always follow the policy.

# User authentication and user information collection

1. When collecting customer information (e.g. names, emails, IDs), ask the customer to spell it out letter by letter (e.g. "J, O, H, N") to ensure you have the correct information and accomodate for customer audio being unclear or background noise.

2. If authenticating the user fails based on user provided information, ALWAYS explicitly ask the customer to SPELL THINGS OUT or provide information LETTER BY LETTER (e.g. "first name J, O, H, N last name S, M, I, T, H").
""".strip()
"""tau2's instructions for a speech-to-speech agent, verbatim."""

CASCADED_MODEL_INSTRUCTION = """
You are a customer service agent handling a VOICE CALL with a customer.

# Important Voice Call Considerations

1. For the conversation, you will see transcribed speech, not written text. Expect:
- Misspellings of names, emails, or technical terms
- Missing or incorrect punctuation (periods, commas)
- Run-on sentences or incomplete thoughts

2. Respond naturally and conversationally as you would in a real phone call. Do not use bullets (numbered or unnumbered) or markdown formatting.

3. Try to be helpful and always follow the policy.

# User authentication and user information collection

1. When collecting customer information (e.g. names, emails, IDs), ask the customer to spell it out letter by letter (e.g. "J, O, H, N") to ensure you have the correct information and accomodate for customer audio being unclear or background noise.

2. If authenticating the user fails based on user provided information, ALWAYS explicitly ask the customer to SPELL THINGS OUT or provide information LETTER BY LETTER (e.g. "first name J, O, H, N last name S, M, I, T, H").
""".strip()
"""tau2's instructions for a cascaded agent, verbatim."""


def is_cascade(agent: AgentSpec | None, config: Mapping[str, Any]) -> bool:
    """Whether the agent under test is a cascade; refuses one not opted into,
    and one whose STT is not ``deepgram.stt`` (Deepgram finds the turns, as in
    tau2)."""
    if agent is None or agent.pipeline is not Pipeline.CASCADE:
        return False
    if not config.get(ALLOW_CASCADE):
        raise TrialConfigError(
            f"tau-voice runs realtime speech-to-speech agents; "
            f"{agent.stt}+{agent.llm}+{agent.tts} is a cascade (STT -> LLM -> "
            f"TTS), which is off by default. To run it anyway, opt in: "
            f"'tau-voice job-config --cascade STT LLM TTS --allow-cascade', or "
            f"pass the call agent 'call_config' = {{\"{ALLOW_CASCADE}\": true}} "
            f"(--ak 'call_config={{\"{ALLOW_CASCADE}\": true}}')"
        )
    if (refusal := stt_refusal(str(agent.stt))) is not None:
        raise TrialConfigError(refusal)
    return True


def cascade_prompt(system_prompt: str) -> str:
    """tau2's prompt for a cascade, from the speech-to-speech one the sidecar
    rendered: the same policy under the cascade's instructions."""
    if not system_prompt.startswith(AUDIO_NATIVE_VOICE_INSTRUCTION):
        raise ValueError(
            "the sidecar's agent prompt does not start with tau2's "
            "speech-to-speech instructions, so the cascade's cannot replace them"
        )
    return CASCADED_MODEL_INSTRUCTION + system_prompt.removeprefix(
        AUDIO_NATIVE_VOICE_INSTRUCTION
    )
