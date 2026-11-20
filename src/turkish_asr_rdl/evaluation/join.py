"""Strict join of prediction rows to evaluator-only reference rows.

Every check runs before any metric: rows must be objects with string IDs, IDs
must be unique on each side, the prediction IDs must equal the reference IDs of
the requested split, and audio hash, dataset release and (when given) the
split manifest hash must agree. Problems list IDs or row numbers, never text.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

REFERENCE_FIELDS = (
    "utterance_id",
    "audio_sha256",
    "reference_text",
    "leakage_group_id",
    "split",
    "dataset_release",
)
MAX_LISTED = 10

Numbered = Sequence[tuple[int, Any]]


class JoinError(ValueError):
    """The predictions cannot be joined to the references; `problems` lists why."""

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = tuple(problems)
        first = f"; first: {self.problems[0]}" if self.problems else ""
        super().__init__(
            f"predictions do not match the references ({len(self.problems)} problem(s)){first}"
        )


@dataclass(frozen=True)
class JoinedRow:
    utterance_id: str
    leakage_group_id: str
    reference: str
    hypothesis: str
    status: str


def _listed(items: Sequence[str]) -> str:
    shown = ", ".join(items[:MAX_LISTED])
    hidden = len(items) - MAX_LISTED
    return f"{shown}, and {hidden} more" if hidden > 0 else shown


def _id_problem(label: str, ids: Iterable[str]) -> list[str]:
    ordered = sorted(set(ids))
    if not ordered:
        return []
    return [f"{label}: {len(ordered)} ID(s): {_listed([repr(i) for i in ordered])}"]


def _row_problem(label: str, numbers: Sequence[int]) -> list[str]:
    if not numbers:
        return []
    shown = _listed([f"row {n}" for n in numbers])
    return [f"{label} in {len(numbers)} row(s): {shown}"]


def _field_problems(side: str, numbered: Numbered, fields: Sequence[str]) -> list[str]:
    problems: list[str] = []
    for field in fields:
        bad = [n for n, row in numbered if not isinstance(row.get(field), str)]
        problems += _row_problem(f"{side}: {field} is missing or not a string", bad)
    return problems


def _objects(side: str, rows: Sequence[Any]) -> tuple[list[str], Numbered]:
    numbered = list(enumerate(rows, start=1))
    not_objects = [n for n, row in numbered if not isinstance(row, Mapping)]
    problems = _row_problem(f"{side}: rows are not JSON objects", not_objects)
    return problems, [(n, row) for n, row in numbered if isinstance(row, Mapping)]


def _structure(
    predictions: Sequence[Any], references: Sequence[Any], split: str
) -> tuple[list[str], Numbered, Numbered]:
    pred_problems, preds = _objects("predictions", predictions)
    ref_problems, refs = _objects("references", references)
    problems = pred_problems + ref_problems
    problems += _field_problems("predictions", preds, ("utterance_id", "status"))
    problems += _field_problems("references", refs, ("utterance_id", "split"))
    if problems:
        return problems, preds, refs
    in_split = [(n, row) for n, row in refs if row["split"] == split]
    problems += _field_problems("references", in_split, REFERENCE_FIELDS)
    no_text = [
        n for n, row in preds
        if row["status"] != "error" and not isinstance(row.get("hypothesis_text"), str)
    ]
    label = "predictions: hypothesis_text is not a string on a non-error row"
    problems += _row_problem(label, no_text)
    return problems, preds, refs


def _duplicates(side: str, rows: Iterable[Mapping[str, Any]]) -> list[str]:
    counts = Counter(row["utterance_id"] for row in rows)
    return _id_problem(f"duplicate utterance_id in {side}", [i for i, c in counts.items() if c > 1])


def _mismatches(
    by_pred: Mapping[str, Mapping[str, Any]],
    by_ref: Mapping[str, Mapping[str, Any]],
    split_manifest_sha256: str | None,
) -> list[str]:
    common = by_pred.keys() & by_ref.keys()
    problems: list[str] = []
    for field in ("audio_sha256", "dataset_release"):
        differ = [i for i in common if by_pred[i].get(field) != by_ref[i][field]]
        problems += _id_problem(f"{field} differs between prediction and reference", differ)
    if split_manifest_sha256 is not None:
        differ = [
            i for i, row in by_pred.items()
            if row.get("split_manifest_sha256") != split_manifest_sha256
        ]
        problems += _id_problem("split_manifest_sha256 differs from the split manifest", differ)
    return problems


def _joined(pred: Mapping[str, Any], ref: Mapping[str, Any]) -> JoinedRow:
    status = pred["status"]
    return JoinedRow(
        utterance_id=ref["utterance_id"],
        leakage_group_id=ref["leakage_group_id"],
        reference=ref["reference_text"],
        hypothesis="" if status == "error" else pred["hypothesis_text"],
        status=status,
    )


def join(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    split: str,
    split_manifest_sha256: str | None = None,
) -> tuple[JoinedRow, ...]:
    """Join predictions to the `split` references by utterance ID, ordered by ID.

    `error` rows join with an empty hypothesis and keep status `error`.
    Raises JoinError listing every problem found.
    """
    problems, preds, refs = _structure(predictions, references, split)
    if problems:
        raise JoinError(problems)
    pred_rows = [row for _, row in preds]
    by_pred = {row["utterance_id"]: row for row in pred_rows}
    by_ref = {row["utterance_id"]: row for _, row in refs if row["split"] == split}
    problems = _duplicates("predictions", pred_rows)
    problems += _duplicates("references", [row for _, row in refs])
    missing, extra = by_ref.keys() - by_pred.keys(), by_pred.keys() - by_ref.keys()
    problems += _id_problem(f"missing predictions for split {split!r}", missing)
    problems += _id_problem(f"extra predictions not in split {split!r}", extra)
    problems += _mismatches(by_pred, by_ref, split_manifest_sha256)
    if problems:
        raise JoinError(problems)
    return tuple(_joined(by_pred[i], by_ref[i]) for i in sorted(by_ref))
