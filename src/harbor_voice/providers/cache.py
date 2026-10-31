"""Two caches, deliberately not one.

In most codebases these would be a single class with a flag. That is the bug.

:class:`BlobCache`
    Caches the simulator's synthesis. The caller's voice is an *input*, and
    an input that changes between runs is not a benchmark. A hit is the normal
    case and the cache is permanent.
:class:`Cassette`
    Caches agent-side provider calls. The agent is the system under test,
    so a cached answer is not a measurement. It exists to make CI and debugging
    cheap, and every run that touches it is stamped ``replayed``.

Merge them and you create a path by which a replayed run can be published as a
result. :func:`require_live` is what closes it, and the
``measurement.json`` writer calls it.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from harbor_voice.audio.format import CANONICAL
from harbor_voice.audio.format import AudioFormat
from harbor_voice.ids import Digest
from harbor_voice.ids import sha256_digest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from harbor_voice.providers.base import Usage

CACHE_SCHEMA = "voice-blob.1"
"""Bumped when the synthesis pipeline changes in a way that alters bytes.

Part of every key, so a change invalidates the whole cache rather than serving
a mixture of old and new audio that nothing can tell apart.
"""


@dataclass(frozen=True, slots=True)
class SynthesisKey:
    """Everything that changes the synthesised bytes, and nothing that does not.

    The honest part is
    :attr:`provider_version`: AWS can change a Polly voice under a stable name,
    so where a version is exposed we key on it, and where it is not we record
    that the key is :attr:`incomplete` rather than pretending it is sound. A
    cache that silently serves pre-change audio alongside post-change audio
    produces an incomparability nobody can see.
    """

    provider: str
    model: str
    """The voice, for a TTS. ``Joanna``, not ``polly``."""

    text: str
    audio_format: AudioFormat = CANONICAL
    speaking_rate: float = 1.0
    provider_version: str | None = None
    schema: str = CACHE_SCHEMA

    @property
    def digest(self) -> Digest:
        parts = (
            self.schema,
            self.provider,
            self.model,
            f"{self.audio_format}",
            f"{self.speaking_rate:.6f}",
            self.provider_version or "",
            self.text,
        )
        return sha256_digest("\x1f".join(parts).encode("utf-8"))

    @property
    def is_complete(self) -> bool:
        """Whether the key pins everything that can change the bytes."""
        return self.provider_version is not None

    @property
    def filename(self) -> str:
        return f"{self.digest.removeprefix('sha256:')}.pcm"


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    incomplete_keys: int = 0
    """Entries whose provider exposed no version — see :attr:`SynthesisKey.is_complete`."""

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        """``None`` when nothing was looked up — not ``0.0``, which would read
        as a cache that is failing rather than one that was never asked."""
        return self.hits / self.lookups if self.lookups else None


class BlobCache:
    """Content-addressed store for simulator synthesis.

    Legitimate and permanent: it caches an *input*. Backed by a directory when
    one is given — shared across trials in a job — and by memory otherwise,
    which is what the offline test suite uses.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root
        self._memory: dict[Digest, bytes] = {}
        self.stats = CacheStats()
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path | None:
        return self._root

    def _path(self, key: SynthesisKey) -> Path | None:
        return self._root / key.filename if self._root is not None else None

    def get(self, key: SynthesisKey) -> bytes | None:
        """Fetch synthesised audio, or ``None``."""
        blob = self._memory.get(key.digest)
        if blob is None:
            path = self._path(key)
            if path is not None and path.exists():
                blob = path.read_bytes()
                self._memory[key.digest] = blob

        if blob is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return blob

    def put(self, key: SynthesisKey, pcm: bytes) -> Digest:
        """Store synthesised audio and return its content digest."""
        self._memory[key.digest] = pcm
        path = self._path(key)
        if path is not None:
            path.write_bytes(pcm)
        self.stats.writes += 1
        if not key.is_complete:
            self.stats.incomplete_keys += 1
        return key.digest

    def __contains__(self, key: object) -> bool:
        """Membership without counting a hit or a miss."""
        if not isinstance(key, SynthesisKey):
            return False
        if key.digest in self._memory:
            return True
        path = self._path(key)
        return path is not None and path.exists()

    def __len__(self) -> int:
        return len(self._memory)


# -------------------------------------------------------------------- cassette


class CassetteMode(StrEnum):
    """What a cassette does with agent-side calls."""

    OFF = "off"
    """Pass through. The only mode a published run may use."""

    RECORD = "record"
    """Call upstream and keep the response."""

    REPLAY = "replay"
    """Serve the stored response. Stamps the run ``replayed``."""


class ReplayedRunError(RuntimeError):
    """A replayed run was about to be published as a measurement."""


Chunk = bytes | str
"""One element of a recorded stream — audio frames or text deltas."""


@dataclass(slots=True)
class Cassette:
    """Record and replay agent-side provider calls. Test facility only.

    The agent is the system under test, so a replayed answer measures the
    cassette rather than the agent. That is useful for debugging and for CI
    without cloud spend, and it is never a result — so :attr:`used_replay`
    follows the run into :func:`require_live`.
    """

    mode: CassetteMode = CassetteMode.OFF
    _takes: dict[str, list[Chunk]] = field(default_factory=dict, repr=False)
    used_replay: bool = False

    @property
    def is_active(self) -> bool:
        return self.mode is not CassetteMode.OFF

    def record(self, key: str, chunks: Iterable[Chunk]) -> list[Chunk]:
        """Keep a stream. Returns it so a caller can record while yielding."""
        kept = list(chunks)
        self._takes[key] = kept
        return kept

    def replay(self, key: str) -> list[Chunk] | None:
        """Serve a stored stream, marking the run.

        Returns ``None`` when nothing was recorded for ``key`` — a miss, which
        the caller must treat as "go upstream", not as an empty response. An
        empty response from a provider is a real and different event.
        """
        if self.mode is not CassetteMode.REPLAY:
            return None
        take = self._takes.get(key)
        if take is None:
            return None
        self.used_replay = True
        return list(take)

    def __contains__(self, key: object) -> bool:
        return key in self._takes

    def __len__(self) -> int:
        return len(self._takes)


def require_live(usages: Sequence[Usage], *, cassette: Cassette | None = None) -> None:
    """Refuse to publish a run that was replayed.

    The ``measurement.json`` writer calls this. Enforced in code rather
    than left to convention, because the agent-side cache exists to make CI
    cheap and the moment it can feed a reported number it has become a way to
    fabricate results.

    Note:
        ``cache_hit`` is not checked. A simulator-side hit is legitimate:
        it replays the caller's voice, which is an input, not the behaviour
        under measurement.

    Raises:
        ReplayedRunError: If any call was served from a cassette.
    """
    replayed = sum(1 for u in usages if u.replayed)
    if cassette is not None and cassette.used_replay:
        replayed = max(replayed, 1)
    if replayed:
        raise ReplayedRunError(
            f"{replayed} provider call(s) were replayed from a cassette; a "
            f"replayed run is not a measurement. Re-run with "
            f"CassetteMode.OFF to publish a number."
        )
