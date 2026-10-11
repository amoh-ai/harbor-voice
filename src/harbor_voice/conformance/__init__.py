"""Portable provider conformance.

Shipped in main code, not test code, so a third-party adapter can import it
into its own pytest. Attestation is what this buys: a capability is only
legitimately listed in ``Capabilities.attested`` once a check here verified it.
"""

from __future__ import annotations

from harbor_voice.conformance.suite import DEFAULT_TIMEOUT_S
from harbor_voice.conformance.suite import Check
from harbor_voice.conformance.suite import CheckContext
from harbor_voice.conformance.suite import CheckResult
from harbor_voice.conformance.suite import CheckSkipped
from harbor_voice.conformance.suite import ConformanceFailure
from harbor_voice.conformance.suite import ConformanceReport
from harbor_voice.conformance.suite import Outcome
from harbor_voice.conformance.suite import all_checks
from harbor_voice.conformance.suite import conformance_check
from harbor_voice.conformance.suite import run_conformance
from harbor_voice.conformance.util import silence_stimulus

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "Check",
    "CheckContext",
    "CheckResult",
    "CheckSkipped",
    "ConformanceFailure",
    "ConformanceReport",
    "Outcome",
    "all_checks",
    "conformance_check",
    "run_conformance",
    "silence_stimulus",
]
