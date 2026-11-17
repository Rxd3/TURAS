"""Validator for N-best JSONL artifacts (contract 0.1 plus the proposed 0.2 keys).

Rows may carry an optional `dedup_map` (removed raw candidate ID -> kept candidate
ID); candidate scores that are null need a reason in `score_details.null_reasons`.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from turkish_asr_rdl.schemas.predictions import (
    check_expected_ids,
    content_problems,
    duplicate_id_errors,
    hash_problems,
    is_number,
    preview,
    row_label,
    schema_version_problems,
    string_problems,
)
from turkish_asr_rdl.schemas.report import ValidationReport, combine

NBEST_ROW_FIELDS = (
    "schema_version", "run_id", "system_id", "dataset_release", "split", "split_manifest_sha256",
    "utterance_id", "audio_sha256", "model_artifact_sha256", "decoder_config_sha256",
    "requested_n", "actual_n", "score_convention_id", "candidates",
)
CANDIDATE_FIELDS = (
    "candidate_id", "rank", "text", "token_ids", "token_count", "acoustic_log_score",
    "acoustic_score_kind", "decoder_sequence_score", "lm_log_score", "rdl_log_score",
    "combined_score", "score_details",
)
NULLABLE_SCORES = ("decoder_sequence_score", "lm_log_score", "rdl_log_score", "combined_score")
DUPLICATE_TEXT_MESSAGE = "duplicate candidate text; deduplicate before export and record it in dedup_map"

_ROW_STRINGS = ("run_id", "system_id", "dataset_release", "split", "utterance_id", "score_convention_id")
_ROW_HASHES = ("split_manifest_sha256", "audio_sha256", "model_artifact_sha256", "decoder_config_sha256")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return is_number(value) and math.isfinite(value)


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _count_problems(row: Mapping[str, Any]) -> list[str]:
    requested, actual, candidates = row.get("requested_n"), row.get("actual_n"), row.get("candidates")
    problems = []
    if "requested_n" in row and not (_is_int(requested) and requested >= 1):
        problems.append("requested_n must be an integer of at least 1")
    if "candidates" in row and not isinstance(candidates, list):
        problems.append("candidates must be an array")
    if "actual_n" in row and not _is_int(actual):
        problems.append("actual_n must be an integer")
    elif isinstance(candidates, list) and _is_int(actual) and actual != len(candidates):
        problems.append(f"actual_n is {actual} but there are {len(candidates)} candidates")
    if _is_int(actual) and _is_int(requested) and actual > requested >= 1:
        problems.append(f"actual_n {actual} exceeds requested_n {requested}; never pad an N-best list")
    return problems


def _token_problems(candidate: Mapping[str, Any]) -> list[str]:
    token_ids, count = candidate.get("token_ids"), candidate.get("token_count")
    problems = []
    ids_ok = isinstance(token_ids, list) and all(_is_int(t) for t in token_ids)
    if "token_ids" in candidate and not ids_ok:
        problems.append("token_ids must be an array of integers")
    if "token_count" in candidate and not _is_int(count):
        problems.append("token_count must be an integer")
    elif ids_ok and "token_count" in candidate and count != len(token_ids):
        problems.append(f"token_count is {count} but token_ids has {len(token_ids)} entries")
    return problems


def _score_problems(candidate: Mapping[str, Any]) -> list[str]:
    details = candidate.get("score_details")
    reasons = details.get("null_reasons") if isinstance(details, Mapping) else None
    problems = []
    if "score_details" in candidate and not isinstance(details, Mapping):
        problems.append("score_details must be an object")
    elif isinstance(details, Mapping) and "null_reasons" in details and not isinstance(reasons, Mapping):
        problems.append("score_details.null_reasons must be an object")
    if "acoustic_log_score" in candidate and not _is_finite_number(candidate["acoustic_log_score"]):
        problems.append("acoustic_log_score must be a finite number; the model score is mandatory")
    for field in (f for f in NULLABLE_SCORES if f in candidate):
        value = candidate[field]
        reason = reasons.get(field) if isinstance(reasons, Mapping) else None
        if value is None and not _non_empty_str(reason):
            problems.append(f"{field} is null without a non-empty reason in score_details.null_reasons")
        elif value is not None and not _is_finite_number(value):
            problems.append(f"{field} must be a finite number or null")
    return problems


def _candidate_problems(candidate: Any) -> list[str]:
    if not isinstance(candidate, Mapping):
        return ["must be a JSON object"]
    problems = []
    missing = [f for f in CANDIDATE_FIELDS if f not in candidate]
    if missing:
        problems.append(f"missing field(s): {', '.join(missing)}")
    for field in ("candidate_id", "acoustic_score_kind"):
        if field in candidate and not _non_empty_str(candidate[field]):
            problems.append(f"{field} must be a non-empty string")
    if "text" in candidate and not isinstance(candidate["text"], str):
        problems.append("text must be a string")
    if "rank" in candidate and not _is_int(candidate["rank"]):
        problems.append("rank must be an integer")
    return problems + _token_problems(candidate) + _score_problems(candidate)


def _candidate_name(candidate: Any, position: int) -> str:
    cid = candidate.get("candidate_id") if isinstance(candidate, Mapping) else None
    return cid if _non_empty_str(cid) else f"#{position}"


def _rank_problems(candidates: list[Any], requested: Any) -> list[str]:
    ranks = [c["rank"] for c in candidates if isinstance(c, Mapping) and _is_int(c.get("rank"))]
    problems = []
    if any(later <= earlier for earlier, later in zip(ranks, ranks[1:])):
        problems.append("candidate rank values must be strictly increasing")
    upper = requested if _is_int(requested) and requested >= 1 else None
    if any(r < 1 or (upper is not None and r > upper) for r in ranks):
        problems.append(f"candidate rank values must lie between 1 and requested_n ({requested})")
    return problems


def _texts_by_value(candidates: list[Any], key: Callable[[str], str]) -> dict[str, list[tuple[str, str]]]:
    groups: dict[str, list[tuple[str, str]]] = {}
    for position, c in enumerate(candidates, start=1):
        if isinstance(c, Mapping) and isinstance(c.get("text"), str):
            groups.setdefault(key(c["text"]), []).append((_candidate_name(c, position), c["text"]))
    return groups


def _list_problems(candidates: list[Any], requested: Any) -> list[str]:
    names = [c["candidate_id"] for c in candidates if isinstance(c, Mapping) and _non_empty_str(c.get("candidate_id"))]
    repeated = sorted(n for n, count in Counter(names).items() if count > 1)
    problems = [f"duplicate candidate_id: {preview(repeated)}"] if repeated else []
    problems += _rank_problems(candidates, requested)
    for members in _texts_by_value(candidates, str).values():
        if len(members) > 1:
            problems.append(f"{DUPLICATE_TEXT_MESSAGE} (candidates {', '.join(n for n, _ in members)})")
    return problems


def _normalized_warnings(candidates: list[Any], normalize: Callable[[str], str]) -> list[str]:
    warnings = []
    for members in _texts_by_value(candidates, normalize).values():
        if len({text for _, text in members}) > 1:
            warnings.append(f"candidates {', '.join(n for n, _ in members)} are identical after normalization")
    return warnings


def _dedup_map_problems(row: Mapping[str, Any], candidates: list[Any]) -> list[str]:
    if "dedup_map" not in row:
        return []
    dedup_map = row["dedup_map"]
    if not isinstance(dedup_map, Mapping):
        return ["dedup_map must be an object mapping removed candidate IDs to kept candidate IDs"]
    kept = {c.get("candidate_id") for c in candidates if isinstance(c, Mapping) and _non_empty_str(c.get("candidate_id"))}
    problems = [f"dedup_map key {k!r} must be a removed candidate ID, not a kept one" for k in dedup_map if k in kept]
    problems += [
        f"dedup_map value for {k!r} must be the ID of a kept candidate"
        for k, v in dedup_map.items()
        if not (isinstance(v, str) and v in kept)
    ]
    return problems


def _row_report(row: Any, index: int, normalize: Callable[[str], str] | None) -> ValidationReport:
    if not isinstance(row, Mapping):
        return ValidationReport(errors=(f"row {index + 1}: must be a JSON object",))
    label = row_label(row, index)
    candidates = row["candidates"] if isinstance(row.get("candidates"), list) else []
    missing = [f for f in NBEST_ROW_FIELDS if f not in row]
    problems = [f"missing field(s): {', '.join(missing)}"] if missing else []
    problems += [
        *content_problems(row),
        *schema_version_problems(row),
        *string_problems(row, _ROW_STRINGS),
        *hash_problems(row, _ROW_HASHES),
        *_count_problems(row),
        *_list_problems(candidates, row.get("requested_n")),
        *_dedup_map_problems(row, candidates),
    ]
    for position, candidate in enumerate(candidates, start=1):
        name = _candidate_name(candidate, position)
        problems += [f"candidate {position} ({name}): {p}" for p in _candidate_problems(candidate)]
    warnings = _normalized_warnings(candidates, normalize) if normalize is not None else []
    return ValidationReport(
        errors=tuple(f"{label}: {p}" for p in problems),
        warnings=tuple(f"{label}: {w}" for w in warnings),
    )


def validate_nbest(
    rows: Sequence[Mapping[str, Any]],
    expected_ids: Sequence[str] | None = None,
    normalize: Callable[[str], str] | None = None,
) -> ValidationReport:
    """Check N-best rows; normalized duplicates are warnings, raw duplicates are errors."""
    reports = [_row_report(row, i, normalize) for i, row in enumerate(rows)]
    reports.append(ValidationReport(errors=tuple(duplicate_id_errors(rows))))
    if expected_ids is not None:
        reports.append(check_expected_ids(rows, expected_ids))
    return combine(*reports)
