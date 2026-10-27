"""tau-voice adapter: tau2-bench's full-duplex voice mode on harbor-voice.

Everything here is tau2's rules, ported and checked against tau2's own code
(reference outputs in ``tests/fixtures``): the lockstep call, the caller's
turn-taking, the per-step handling of agent audio, the phone line, and the
client for the tau2 sidecar (``sidecar/``). harbor-voice core supplies the
generic parts — transports, providers, the G.711 codec.

"""

from __future__ import annotations

from tau_voice.client import Sidecar
from tau_voice.client import SidecarError
from tau_voice.content import STOP_TOKENS
from tau_voice.content import Tau2Content
from tau_voice.run import run_task
from tau_voice.run import run_trial

__all__ = [
    "STOP_TOKENS",
    "Sidecar",
    "SidecarError",
    "Tau2Content",
    "run_task",
    "run_trial",
]
