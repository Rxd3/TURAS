"""Validators for prediction JSONL rows and strict inference inputs (contract 0.1).

The rule helpers below (`row_label`, `is_number`, `preview`, `content_problems`,
`string_problems`, `hash_problems`, `schema_version_problems`,
`duplicate_id_errors`, `check_expected_ids`) are shared with the N-best validator.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from turkish_asr_rdl.schemas.report import ValidationReport, combine

SCHEMA_VERSION = "0.1"
PREDICTION_FIELDS: tuple[str, ...] = (
    "schema_version", "run_id", "system_id", "dataset_release", "split", "split_manifest_sha256",
    "utterance_id", "audio_sha256", "status", "hypothesis_text", "selected_candidate_id",
    "nbest_artifact_sha256", "model_artifact_sha256", "decoder_config_sha256",
    "normalization_version", "timing_record_id", "error",
)
INFERENCE_INPUT_FIELDS = (
    "schema_version", "dataset_release", "split_manifest_sha256", "utterance_id", "audio_uri",
    "audio_sha256", "duration_s", "sample_rate_hz", "channels", "language",
)
FORBIDDEN_KEYS = frozenset({
    "reference_text", "reference", "references", "gold", "gold_text", "target", "targets",
    "label", "labels", "transcript_correction", "reference_token_ids",
})
STATUSES = ("ok", "empty", "error")
RUN_CONSTANT_FIELDS = ("run_id", "system_id", "dataset_release", "split", "split_manifest_sha256")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_LISTED = 10
_PREDICTION_STRINGS = (
    "run_id", "system_id", "dataset_release", "split", "utterance_id",
    "normalization_version", "timing_record_id",
)
_PREDICTION_HASHES = ("split_manifest_sha256", "audio_sha256", "model_artifact_sha256", "decoder_config_sha256")
_INPUT_STRINGS = ("dataset_release", "utterance_id", "audio_uri", "language")
_INPUT_HASHES = ("split_manifest_sha256", "audio_sha256")


def row_label(row: Any, index: int) -> str:
    """'row N (utterance_id)' with a 1-based N, or 'row N' when the ID is unusable."""
    uid = row.get("utterance_id") if isinstance(row, Mapping) else None
    return f"row {index + 1} ({uid})" if isinstance(uid, str) and uid else f"row {index + 1}"


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def preview(values: Sequence[Any]) -> str:
    """Up to ten values joined by commas, plus a count of the rest."""
    shown = ", ".join(str(v) for v in values[:_MAX_LISTED])
    rest = len(values) - _MAX_LISTED
    return f"{shown} and {rest} more" if rest > 0 else shown


def _child_items(value: Any, path: str) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        return [(f"{path}.{key}" if path else str(key), child) for key, child in value.items()]
    if isinstance(value, (list, tuple)):
        return [(f"{path}[{i}]", child) for i, child in enumerate(value)]
    return []


def forbidden_key_paths(value: Any, path: str = "") -> list[str]:
    """Paths of every forbidden key at any depth of objects and lists."""
    found = []
    if isinstance(value, Mapping):
        found = [f"{path}.{key}" if path else str(key) for key in value if key in FORBIDDEN_KEYS]
    for child_path, child in _child_items(value, path):
        found.extend(forbidden_key_paths(child, child_path))
    return found


def non_finite_paths(value: Any, path: str = "") -> list[str]:
    """Paths of every NaN or infinite float at any depth of objects and lists."""
    if isinstance(value, float) and not math.isfinite(value):
        return [path or "value"]
    return [p for child_path, child in _child_items(value, path) for p in non_finite_paths(child, child_path)]


def field_set_problems(row: Mapping[str, Any], required: Sequence[str], allowed: Sequence[str]) -> list[str]:
    """Missing required fields and unknown fields (forbidden keys are reported separately)."""
    problems = []
    missing = [f for f in required if f not in row]
    unknown = [str(k) for k in row if k not in allowed and k not in FORBIDDEN_KEYS]
    if missing:
        problems.append(f"missing field(s): {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown field(s): {', '.join(unknown)}")
    return problems


def content_problems(row: Mapping[str, Any]) -> list[str]:
    """Forbidden reference keys and non-finite numbers anywhere in a row."""
    try:
        forbidden, non_finite = forbidden_key_paths(row), non_finite_paths(row)
    except RecursionError:
        return ["row is nested too deeply to check"]
    return [
        f"forbidden key {p}: reference data must never appear in inference or prediction artifacts"
        for p in forbidden
    ] + [f"non-finite number at {p}; all numbers must be finite" for p in non_finite]


def string_problems(row: Mapping[str, Any], fields: Sequence[str]) -> list[str]:
    return [f"{f} must be a non-empty string" for f in fields if f in row and not (isinstance(row[f], str) and row[f])]


def hash_problems(row: Mapping[str, Any], fields: Sequence[str], nullable: bool = False) -> list[str]:
    return [
        f"{f} must be 64 lowercase hex characters" + (" or null" if nullable else "")
        for f in fields
        if f in row and not (is_sha256(row[f]) or (nullable and row[f] is None))
    ]


def schema_version_problems(row: Mapping[str, Any]) -> list[str]:
    if "schema_version" in row and row["schema_version"] != SCHEMA_VERSION:
        return [f'schema_version must be "{SCHEMA_VERSION}"']
    return []


def _ids(rows: Sequence[Any]) -> list[str]:
    return [r["utterance_id"] for r in rows if isinstance(r, Mapping) and isinstance(r.get("utterance_id"), str)]


def duplicate_id_errors(rows: Sequence[Any]) -> list[str]:
    duplicates = sorted(uid for uid, count in Counter(_ids(rows)).items() if count > 1)
    return [f"duplicate utterance_id ({len(duplicates)}): {preview(duplicates)}"] if duplicates else []


def check_expected_ids(rows: Sequence[Any], expected_ids: Sequence[str]) -> ValidationReport:
    """Missing IDs, extra IDs and the first position where the row order differs."""
    found = list(dict.fromkeys(_ids(rows)))
    expected = list(dict.fromkeys(expected_ids))
    found_set, expected_set = set(found), set(expected)
    missing = [uid for uid in expected if uid not in found_set]
    extra = [uid for uid in found if uid not in expected_set]
    errors = []
    if missing:
        errors.append(f"missing {len(missing)} expected utterance IDs: {preview(missing)}")
    if extra:
        errors.append(f"{len(extra)} utterance IDs not in the expected list: {preview(extra)}")
    in_rows = [uid for uid in found if uid in expected_set]
    in_expected = [uid for uid in expected if uid in found_set]
    for position, (got, want) in enumerate(zip(in_rows, in_expected), start=1):
        if got != want:
            errors.append(
                f"rows are not in the expected order: first difference at position {position} "
                f"(expected {want}, found {got})"
            )
            break
    return ValidationReport(errors=tuple(errors))


def _run_constant_errors(rows: Sequence[Any]) -> list[str]:
    errors = []
    for field in RUN_CONSTANT_FIELDS:
        values = sorted({r[field] for r in rows if isinstance(r, Mapping) and isinstance(r.get(field), str)})
        if len(values) > 1:
            errors.append(f"{field} must be the same in every row; found {len(values)} values: {preview(values)}")
    return errors


def _status_problems(row: Mapping[str, Any]) -> list[str]:
    status = row.get("status")
    if "status" not in row:
        return []
    if not isinstance(status, str) or status not in STATUSES:
        return [f'status must be "ok", "empty" or "error", got {status!r}']
    problems = _text_problems(row, status)
    if "error" in row:
        problems.extend(_error_object_problems(row["error"], status))
    return problems


def _text_problems(row: Mapping[str, Any], status: str) -> list[str]:
    if "hypothesis_text" not in row:
        return []
    text = row["hypothesis_text"]
    if status == "ok" and not isinstance(text, str):
        return ['status "ok" needs hypothesis_text as a string']
    if status == "ok" and text == "":
        return ['status "ok" needs a non-empty hypothesis_text; use status "empty" for an empty transcript']
    if status == "empty" and text != "":
        return ['status "empty" needs hypothesis_text ""']
    if status == "error" and text is not None:
        return ['status "error" needs hypothesis_text null']
    return []


def _error_object_problems(error: Any, status: str) -> list[str]:
    if status != "error":
        return [] if error is None else [f'status "{status}" needs error null']
    if isinstance(error, Mapping) and isinstance(error.get("code"), str) and isinstance(error.get("message"), str):
        return []
    return ['status "error" needs error as an object with string code and message']


def _prediction_row_errors(row: Any, index: int) -> list[str]:
    if not isinstance(row, Mapping):
        return [f"row {index + 1}: must be a JSON object"]
    problems = [
        *field_set_problems(row, PREDICTION_FIELDS, PREDICTION_FIELDS),
        *content_problems(row),
        *schema_version_problems(row),
        *string_problems(row, _PREDICTION_STRINGS),
        *hash_problems(row, _PREDICTION_HASHES),
        *hash_problems(row, ("nbest_artifact_sha256",), nullable=True),
        *_status_problems(row),
    ]
    if "selected_candidate_id" in row and not isinstance(row["selected_candidate_id"], (str, type(None))):
        problems.append("selected_candidate_id must be a string or null")
    label = row_label(row, index)
    return [f"{label}: {p}" for p in problems]


def validate_predictions(
    rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str] | None = None
) -> ValidationReport:
    """Check prediction rows against the predictions contract; never raises on bad rows."""
    row_errors = [e for i, row in enumerate(rows) for e in _prediction_row_errors(row, i)]
    report = ValidationReport(errors=tuple(row_errors + duplicate_id_errors(rows) + _run_constant_errors(rows)))
    if expected_ids is None:
        return report
    return combine(report, check_expected_ids(rows, expected_ids))


def _positive_problems(row: Mapping[str, Any]) -> list[str]:
    problems = []
    duration = row.get("duration_s")
    if "duration_s" in row and not (is_number(duration) and math.isfinite(duration) and duration > 0):
        problems.append("duration_s must be a positive finite number")
    for field in ("sample_rate_hz", "channels"):
        value = row.get(field)
        if field in row and not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
            problems.append(f"{field} must be a positive integer")
    return problems


def _input_row_errors(row: Any, index: int) -> list[str]:
    if not isinstance(row, Mapping):
        return [f"row {index + 1}: must be a JSON object"]
    problems = [
        *field_set_problems(row, INFERENCE_INPUT_FIELDS, INFERENCE_INPUT_FIELDS),
        *content_problems(row),
        *schema_version_problems(row),
        *string_problems(row, _INPUT_STRINGS),
        *hash_problems(row, _INPUT_HASHES),
        *_positive_problems(row),
    ]
    label = row_label(row, index)
    return [f"{label}: {p}" for p in problems]


def validate_inference_inputs(rows: Sequence[Mapping[str, Any]]) -> ValidationReport:
    """Check that inference inputs hold exactly the allowlisted fields with valid values."""
    return ValidationReport(errors=tuple(e for i, row in enumerate(rows) for e in _input_row_errors(row, i)))
