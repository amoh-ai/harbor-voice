"""Identifier aliases shared across harbor-voice.

These are ``NewType`` rather than bare ``str`` so that a party id cannot be
passed where a content digest is expected. They carry no runtime cost.
"""

from __future__ import annotations

from typing import NewType

PartyId = NewType("PartyId", str)
"""Stable id of a participant in a call, e.g. ``caller``, ``agent``."""

PlanId = NewType("PlanId", str)
"""Id of the perturbation plan that produced a frame."""

Digest = NewType("Digest", str)
"""Content hash, written ``sha256:<hex>``."""

TrialId = NewType("TrialId", str)
"""Id of the trial a trajectory belongs to."""


def sha256_digest(data: bytes) -> Digest:
    """Return the canonical ``sha256:<hex>`` digest of ``data``."""
    import hashlib

    return Digest(f"sha256:{hashlib.sha256(data).hexdigest()}")
