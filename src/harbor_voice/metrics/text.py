"""Word-level comparison for the divergence family.

The pairing being measured is **said vs heard**: what the other party's words
were, against what the system transcribed from them. That is explicitly *a
channel measurement, not human transcription truth* — and the only WER
definition that survives perturbation, because the
reference is generated rather than annotated and therefore exact.

Normalisation is deliberately shallow: case and surrounding punctuation only.
Aggressive normalisation (number words to digits, contraction expansion) hides
exactly the errors a degraded channel produces, so it would make the metric
look better as the line gets worse.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_STRIP = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def normalise(text: str) -> list[str]:
    """Split into comparable words: case-folded, edge punctuation removed."""
    words: list[str] = []
    for raw in unicodedata.normalize("NFKC", text).split():
        word = _STRIP.sub("", raw).casefold()
        if word:
            words.append(word)
    return words


@dataclass(frozen=True, slots=True)
class EditCounts:
    """The three error types, kept apart.

    A single WER number conflates them, and they have different causes: a
    noisy line produces substitutions, a dropped packet produces deletions,
    and an over-eager recogniser produces insertions. Reporting the mix costs
    nothing and tells you which one you have — the same argument as precision
    versus recall.
    """

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_words: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        """Errors per reference word.

        ``None`` when the reference is empty: WER is undefined there, and
        returning ``0.0`` would report a perfect transcription of nothing.
        Can exceed 1.0 when the hypothesis is longer than the reference, which
        is a real outcome and is not clamped.
        """
        if self.reference_words == 0:
            return None
        return self.errors / self.reference_words


def edit_counts(reference: str, hypothesis: str) -> EditCounts:
    """Levenshtein over words, with the operation mix kept.

    Standard dynamic programme, O(len(ref) * len(hyp)) and backtracked once to
    separate substitutions from deletions and insertions.
    """
    ref = normalise(reference)
    hyp = normalise(hypothesis)
    n, m = len(ref), len(hyp)

    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(
                    cost[i - 1][j - 1],  # substitute
                    cost[i - 1][j],  # delete
                    cost[i][j - 1],  # insert
                )

    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return EditCounts(
        substitutions=subs, deletions=dels, insertions=ins, reference_words=n
    )


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """WER, or ``None`` when the reference is empty."""
    return edit_counts(reference, hypothesis).rate
