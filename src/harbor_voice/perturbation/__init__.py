"""Perturbation layers.

Applied in physical order (persona, speech, environment, channel), every layer
seeded, the clean reference retained, and the plan written into the trajectory.
"""

from __future__ import annotations

from harbor_voice.perturbation.base import ApplyContext
from harbor_voice.perturbation.base import Perturbation
from harbor_voice.perturbation.base import RunningMean
from harbor_voice.perturbation.base import SeedSet
from harbor_voice.perturbation.channel import TELEPHONY_HZ
from harbor_voice.perturbation.channel import GilbertElliottLoss
from harbor_voice.perturbation.channel import MuLawCodec
from harbor_voice.perturbation.conversation import ConversationError
from harbor_voice.perturbation.conversation import ConversationPlan
from harbor_voice.perturbation.conversation import Landing
from harbor_voice.perturbation.conversation import Probe
from harbor_voice.perturbation.conversation import ProbeInjector
from harbor_voice.perturbation.conversation import ProbeKind
from harbor_voice.perturbation.conversation import ProbeResult
from harbor_voice.perturbation.conversation import ProbeSpec
from harbor_voice.perturbation.conversation import injector_for
from harbor_voice.perturbation.environment import BackgroundBed
from harbor_voice.perturbation.environment import synthetic_bed
from harbor_voice.perturbation.plan import CleanReferenceError
from harbor_voice.perturbation.plan import CleanStore
from harbor_voice.perturbation.plan import PerturbationPipeline
from harbor_voice.perturbation.plan import PlanRecord
from harbor_voice.perturbation.profiles import PROFILES
from harbor_voice.perturbation.profiles import Profile
from harbor_voice.perturbation.profiles import ProfileNotBuildable
from harbor_voice.perturbation.profiles import ProfileSummary
from harbor_voice.perturbation.profiles import UnknownProfile
from harbor_voice.perturbation.profiles import get_profile
from harbor_voice.perturbation.profiles import profile_names
from harbor_voice.perturbation.profiles import summarise

__all__ = [
    "PROFILES",
    "Profile",
    "ProfileNotBuildable",
    "ProfileSummary",
    "UnknownProfile",
    "get_profile",
    "profile_names",
    "summarise",
    "TELEPHONY_HZ",
    "ApplyContext",
    "BackgroundBed",
    "CleanReferenceError",
    "CleanStore",
    "ConversationError",
    "ConversationPlan",
    "GilbertElliottLoss",
    "Landing",
    "MuLawCodec",
    "Perturbation",
    "PerturbationPipeline",
    "PlanRecord",
    "Probe",
    "ProbeInjector",
    "ProbeKind",
    "ProbeResult",
    "ProbeSpec",
    "RunningMean",
    "SeedSet",
    "injector_for",
    "synthetic_bed",
]
