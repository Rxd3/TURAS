"""Deterministic Levenshtein alignment counts (reference to hypothesis).

Ties break by backtracing from the end and taking the first move that reaches
the cell's optimal cost: the diagonal (hit or substitution), then deletion,
then insertion. So `a b` against `b a` is two substitutions.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def ref_len(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def __add__(self, other: "EditCounts") -> "EditCounts":
        if not isinstance(other, EditCounts):
            return NotImplemented
        return EditCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.hits + other.hits,
        )


def _cost_table(ref: Sequence[str], hyp: Sequence[str]) -> list[list[int]]:
    # ponytail: O(n*m) pure Python; vectorize if evaluation of a full test set exceeds a few minutes
    table = [list(range(len(hyp) + 1))]
    for i in range(1, len(ref) + 1):
        row = [i]
        for j in range(1, len(hyp) + 1):
            diagonal = table[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            row.append(min(diagonal, table[i - 1][j] + 1, row[j - 1] + 1))
        table.append(row)
    return table


def align_counts(ref: Sequence[str], hyp: Sequence[str]) -> EditCounts:
    """Count S, D, I and hits on one optimal alignment with the fixed tie-break."""
    table = _cost_table(ref, hyp)
    counts = {"S": 0, "D": 0, "I": 0, "H": 0}
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        cost = table[i][j]
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            if table[i - 1][j - 1] + (not same) == cost:
                counts["H" if same else "S"] += 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and table[i - 1][j] + 1 == cost:
            counts["D"] += 1
            i -= 1
            continue
        counts["I"] += 1
        j -= 1
    return EditCounts(counts["S"], counts["D"], counts["I"], counts["H"])
