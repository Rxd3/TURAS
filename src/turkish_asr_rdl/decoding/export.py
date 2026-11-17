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
MANIFEST_GROUPS = (
    "identity", "code", "environment", "inputs", "configuration", "randomness",
    "model_knowledge", "selection", "outputs", "evaluation_timing", "limitations",
)
_GIT_TIMEOUT_S = 10
_PACKAGE_DIR = Path(__file__).resolve().parent


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
        "selected_candidate_id": None,
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
        "lm_log_score": None,
        "rdl_log_score": None,
        "combined_score": None,
    }
    if not isinstance(candidate.score_details, Mapping):
        raise ValueError(f"candidate {candidate.candidate_id!r}: score_details must be an object")
    given = candidate.score_details.get("null_reasons", {})
    if not isinstance(given, Mapping):
        raise ValueError(f"candidate {candidate.candidate_id!r}: score_details.null_reasons must be an object")
    defaults = {name: NOT_PRODUCED for name in NULLABLE_SCORES if scores[name] is None}
    return {
        "candidate_id": candidate.candidate_id,
        "rank": candidate.rank,
        "text": candidate.text,
        "token_ids": list(candidate.token_ids),
        "token_count": len(candidate.token_ids),
        "acoustic_log_score": candidate.acoustic_log_score,
        "acoustic_score_kind": candidate.acoustic_score_kind,
        **scores,
        "score_details": {**candidate.score_details, "null_reasons": {**defaults, **given}},
    }


def nbest_row(
    hyp: Hypothesis, inp: InferenceInput, ctx: RunContext, requested_n: int, score_convention_id: str
) -> dict:
    """One N-best row; scores the system does not produce are null with a reason."""
    candidates = _candidate_rows(hyp)
    return {
        **_identity(hyp, inp, ctx),
        "model_artifact_sha256": ctx.model_artifact_sha256,
        "decoder_config_sha256": ctx.decoder_config_sha256,
        "requested_n": requested_n,
        "actual_n": len(candidates),
        "score_convention_id": score_convention_id,
        "candidates": candidates,
    }


def _candidate_rows(hyp: Hypothesis) -> list[dict[str, Any]]:
    return [_candidate_row(c) for c in hyp.candidates]


def adapter_result_fields(hyp: Hypothesis, requested_n: int) -> dict[str, Any]:
    """Every value the prediction row, and the N-best row when requested_n > 0, takes from one result."""
    fields = {"status": hyp.status, "hypothesis_text": hyp.text, "error": _error_object(hyp)}
    return {**fields, "candidates": _candidate_rows(hyp)} if requested_n > 0 else fields


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
