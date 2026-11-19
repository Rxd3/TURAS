"""Turn adapter output into contract rows (predictions, N-best) and a run manifest."""
from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from turkish_asr_rdl.decoding.interface import Candidate, Hypothesis, InferenceInput
from turkish_asr_rdl.schemas.nbest import NULLABLE_SCORES
from turkish_asr_rdl.schemas.predictions import SCHEMA_VERSION

NOT_PRODUCED = "not produced by this system"
DUPLICATE_POLICY = "exact raw-text match; first (lowest-rank) occurrence kept; removed IDs listed in dedup_map"
COPIED_HYPOTHESIS_MESSAGE = (
    "decoder returned copies of one hypothesis (same text and token_ids); use the core GenerationMixin beam path, "
    "not WhisperForConditionalGeneration.generate with num_return_sequences"
)
MANIFEST_GROUPS = (
    "identity", "code", "environment", "inputs", "configuration", "randomness",
    "model_knowledge", "selection", "outputs", "evaluation_timing", "limitations",
)
_GIT_TIMEOUT_S = 10
_PACKAGE_DIR = Path(__file__).resolve().parent


class CopiedHypothesisError(ValueError):
    """Two candidates of one result share text and token_ids: copies of one hypothesis, not a beam."""


@dataclass(frozen=True)
class RunContext:
    """Identity shared by every row of one run."""

    run_id: str
    system_id: str
    dataset_release: str
    split: str
    split_manifest_sha256: str
    model_artifact_sha256: str
    decoder_config_sha256: str
    normalization_version: str = "tr-normalized-v1"


def _identity(hyp: Hypothesis, inp: InferenceInput, ctx: RunContext) -> dict[str, Any]:
    if hyp.utterance_id != inp.utterance_id:
        raise ValueError(f"hypothesis for {hyp.utterance_id!r} does not belong to input {inp.utterance_id!r}")
    if inp.dataset_release != ctx.dataset_release or inp.split_manifest_sha256 != ctx.split_manifest_sha256:
        raise ValueError(f"input {inp.utterance_id!r} comes from a different dataset release or split manifest")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": ctx.run_id,
        "system_id": ctx.system_id,
        "dataset_release": ctx.dataset_release,
        "split": ctx.split,
        "split_manifest_sha256": ctx.split_manifest_sha256,
        "utterance_id": inp.utterance_id,
        "audio_sha256": inp.audio_sha256,
    }


def prediction_row(
    hyp: Hypothesis,
    inp: InferenceInput,
    ctx: RunContext,
    timing_record_id: str,
    nbest_artifact_sha256: str | None,
) -> dict:
    """One prediction row in contract field order; the text stays raw."""
    return {
        **_identity(hyp, inp, ctx),
        "status": hyp.status,
        "hypothesis_text": hyp.text,
        "selected_candidate_id": _selected_candidate_id(hyp, nbest_artifact_sha256 is not None),
        "nbest_artifact_sha256": nbest_artifact_sha256,
        "model_artifact_sha256": ctx.model_artifact_sha256,
        "decoder_config_sha256": ctx.decoder_config_sha256,
        "normalization_version": ctx.normalization_version,
        "timing_record_id": timing_record_id,
        "error": _error_object(hyp),
    }


def _error_object(hyp: Hypothesis) -> dict[str, Any] | None:
    return {"code": hyp.error_code, "message": hyp.error_message} if hyp.status == "error" else None


def _candidate_row(candidate: Candidate) -> dict[str, Any]:
    scores = {
        "decoder_sequence_score": candidate.decoder_sequence_score,
        "lm_log_score": candidate.lm_log_score,
        "rdl_log_score": candidate.rdl_log_score,
        "combined_score": candidate.combined_score,
    }
    if not isinstance(candidate.score_details, Mapping):
        raise ValueError(f"candidate {candidate.candidate_id!r}: score_details must be an object")
    given = candidate.score_details.get("null_reasons", {})
    if not isinstance(given, Mapping):
        raise ValueError(f"candidate {candidate.candidate_id!r}: score_details.null_reasons must be an object")
    defaults = {name: NOT_PRODUCED for name in NULLABLE_SCORES if scores[name] is None}
    details = candidate.score_details
    return {
        "candidate_id": candidate.candidate_id,
        "rank": candidate.rank,
        "text": candidate.text,
        "token_ids": list(candidate.token_ids),
        "token_count": len(candidate.token_ids),
        "acoustic_log_score": candidate.acoustic_log_score,
        "acoustic_score_kind": candidate.acoustic_score_kind,
        **scores,
        "score_details": {
            **details,
            "null_reasons": {**defaults, **given},
            "duplicate_policy": details.get("duplicate_policy", DUPLICATE_POLICY),
        },
    }


def _dedupe(candidates: Sequence[Candidate]) -> tuple[tuple[Candidate, ...], dict[str, str]]:
    """Keep the first candidate of each exact raw text; map every removed candidate_id to the kept one."""
    kept: dict[Any, Candidate] = {}
    dedup_map: dict[str, str] = {}
    for candidate in candidates:
        first = kept.setdefault(candidate.text, candidate)
        if first is candidate:
            continue
        removed_id = candidate.candidate_id
        if not isinstance(removed_id, str) or not removed_id:
            raise ValueError(f"removed duplicate candidate_id {removed_id!r} must be a non-empty string")
        if removed_id in dedup_map:
            raise ValueError(f"removed duplicate candidate_id {removed_id!r} appears more than once")
        dedup_map[removed_id] = first.candidate_id
    return tuple(kept.values()), dedup_map


def _refuse_copies(candidates: Sequence[Candidate]) -> None:
    """Beam search never returns one token sequence twice, so a repeated (text, token_ids) pair is a copy."""
    seen: set[tuple[Any, tuple[Any, ...]]] = set()
    for candidate in candidates:
        key = (candidate.text, tuple(candidate.token_ids))
        if key in seen:
            raise CopiedHypothesisError(COPIED_HYPOTHESIS_MESSAGE)
        seen.add(key)


def _candidate_rows(hyp: Hypothesis) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not isinstance(hyp.candidates, (tuple, list)):
        raise ValueError(f"result for {hyp.utterance_id!r}: candidates must be a tuple or list")
    _refuse_copies(hyp.candidates)
    kept, dedup_map = _dedupe(hyp.candidates)
    return [_candidate_row(c) for c in kept], dedup_map


def _selected_candidate_id(hyp: Hypothesis, has_nbest: bool) -> str | None:
    """The rescored pick: one candidate written to the N-best artifact whose text is the hypothesis text."""
    selected = hyp.selected_candidate_id
    if selected is None:
        return None
    if not isinstance(selected, str) or hyp.status == "error":
        raise ValueError(f"selected_candidate_id {selected!r} must be a string naming a candidate "
                         "of a non-error result")
    if not has_nbest:
        raise ValueError(f"selected_candidate_id {selected!r} needs an N-best artifact; this run writes none")
    kept, _ = _candidate_rows(hyp)
    matches = [row for row in kept if row["candidate_id"] == selected]
    if len(matches) != 1 or matches[0]["text"] != hyp.text:
        raise ValueError(f"selected_candidate_id {selected!r} must name one candidate kept after deduplication "
                         "whose text is the hypothesis text")
    return selected


def _candidate_fields(hyp: Hypothesis) -> dict[str, Any]:
    """The kept candidate rows, plus dedup_map only when a candidate was removed."""
    rows, dedup_map = _candidate_rows(hyp)
    return {"candidates": rows, "dedup_map": dedup_map} if dedup_map else {"candidates": rows}


def nbest_row(
    hyp: Hypothesis, inp: InferenceInput, ctx: RunContext, requested_n: int, score_convention_id: str
) -> dict:
    """One N-best row; scores the system does not produce are null with a reason."""
    candidates = _candidate_fields(hyp)
    return {
        **_identity(hyp, inp, ctx),
        "model_artifact_sha256": ctx.model_artifact_sha256,
        "decoder_config_sha256": ctx.decoder_config_sha256,
        "requested_n": requested_n,
        "actual_n": len(candidates["candidates"]),
        "score_convention_id": score_convention_id,
        **candidates,
    }


def adapter_result_fields(hyp: Hypothesis, requested_n: int) -> dict[str, Any]:
    """Every value the prediction row, and the N-best row when requested_n > 0, takes from one result."""
    fields = {
        "status": hyp.status,
        "hypothesis_text": hyp.text,
        "selected_candidate_id": _selected_candidate_id(hyp, requested_n > 0),
        "error": _error_object(hyp),
    }
    return {**fields, **_candidate_fields(hyp)} if requested_n > 0 else fields


def _git(repo_dir: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=repo_dir, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=_GIT_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def git_state(repo_dir: Path) -> dict:
    """Commit and dirty flag of the repository holding repo_dir; None values without git."""
    commit = _git(repo_dir, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git(repo_dir, "status", "--porcelain")
    return {"commit": commit.strip(), "dirty": None if status is None else bool(status.strip())}


def run_manifest(
    ctx: RunContext,
    *,
    command: str,
    environment: Mapping[str, Any],
    configuration: Mapping[str, Any],
    outputs: Mapping[str, str],
    timing: Mapping[str, Any],
    limitations: Sequence[str],
) -> dict:
    """The run manifest with the artifacts contract's eleven groups as top-level keys."""
    return {
        "identity": {
            "schema_version": SCHEMA_VERSION,
            "run_id": ctx.run_id,
            "system_id": ctx.system_id,
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "code": {"command": command, **git_state(_PACKAGE_DIR)},
        "environment": dict(environment),
        "inputs": {
            "dataset_release": ctx.dataset_release,
            "split": ctx.split,
            "split_manifest_sha256": ctx.split_manifest_sha256,
        },
        "configuration": {"decoder_config_sha256": ctx.decoder_config_sha256, "resolved": dict(configuration)},
        "randomness": {},
        "model_knowledge": {"model_artifact_sha256": ctx.model_artifact_sha256},
        "selection": {},
        "outputs": dict(outputs),
        "evaluation_timing": {"normalization_version": ctx.normalization_version, **timing},
        "limitations": list(limitations),
    }
