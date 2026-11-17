"""Adapter interface, export rows and run manifest. All texts and IDs are SYNTHETIC."""
import dataclasses
import subprocess

import numpy as np
import pytest

import turkish_asr_rdl.decoding.export as export_module
from turkish_asr_rdl.decoding.export import (
    MANIFEST_GROUPS,
    NOT_PRODUCED,
    git_state,
    nbest_row,
    prediction_row,
    run_manifest,
)
from turkish_asr_rdl.decoding.interface import AudioItem, Candidate, Hypothesis, InferenceInput
from turkish_asr_rdl.schemas.nbest import NBEST_ROW_FIELDS, NULLABLE_SCORES, validate_nbest
from turkish_asr_rdl.schemas.predictions import INFERENCE_INPUT_FIELDS, PREDICTION_FIELDS, validate_predictions

ELEVEN_GROUPS = {
    "identity", "code", "environment", "inputs", "configuration", "randomness",
    "model_knowledge", "selection", "outputs", "evaluation_timing", "limitations",
}


def ok_hyp(uid: str = "utt_a") -> Hypothesis:
    return Hypothesis(utterance_id=uid, status="ok", text="SYNTHETIC merhaba")


def scored_hyp(uid: str = "utt_a") -> Hypothesis:
    candidates = (
        Candidate("c1", 1, "SYNTHETIC bir", (5, 6), -1.0, "seq2seq_teacher_forced_log_prob", -0.4),
        Candidate("c2", 2, "SYNTHETIC iki", (5, 7, 8), -2.5, "seq2seq_teacher_forced_log_prob",
                  score_details={"length_penalty": 1.0, "null_reasons": {"lm_log_score": "LM disabled"}}),
    )
    return Hypothesis(utterance_id=uid, status="ok", text="SYNTHETIC bir", candidates=candidates)


def test_inference_input_fields_follow_the_allowlist():
    assert tuple(f.name for f in dataclasses.fields(InferenceInput)) == INFERENCE_INPUT_FIELDS


def test_from_row_builds_input(make_input):
    row = dataclasses.asdict(make_input("utt_a"))
    assert InferenceInput.from_row(row) == make_input("utt_a")


def test_from_row_rejects_reference_text(make_input):
    row = {**dataclasses.asdict(make_input("utt_a")), "reference_text": "SYNTHETIC x"}
    with pytest.raises(ValueError, match="reference_text"):
        InferenceInput.from_row(row)


def test_from_row_rejects_missing_field(make_input):
    row = {k: v for k, v in dataclasses.asdict(make_input("utt_a")).items() if k != "audio_uri"}
    with pytest.raises(ValueError, match="audio_uri"):
        InferenceInput.from_row(row)


def test_dataclasses_are_frozen(make_input):
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_input("utt_a").utterance_id = "utt_b"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ok_hyp().status = "error"


def test_audio_items_compare_by_id_without_array_truth_errors():
    a = AudioItem("utt_a", np.zeros(4, np.float32))
    assert a == AudioItem("utt_a", np.ones(4, np.float32))


def test_prediction_row_passes_validator(make_input, run_context):
    row = prediction_row(ok_hyp(), make_input("utt_a"), run_context, "t1", None)
    assert validate_predictions([row]).ok
    assert tuple(row) == PREDICTION_FIELDS


def test_prediction_row_copies_identity_and_links(make_input, run_context):
    row = prediction_row(ok_hyp(), make_input("utt_a"), run_context, "t7", "c" * 64)
    assert row["timing_record_id"] == "t7" and row["nbest_artifact_sha256"] == "c" * 64
    assert row["run_id"] == "run_synthetic" and row["audio_sha256"] == "b" * 64
    assert row["normalization_version"] == "tr-normalized-v1" and row["selected_candidate_id"] is None


def test_empty_hypothesis_row_passes(make_input, run_context):
    hyp = Hypothesis("utt_a", "empty", "")
    assert validate_predictions([prediction_row(hyp, make_input("utt_a"), run_context, "t1", None)]).ok


def test_error_hypothesis_row_passes_validator(make_input, run_context):
    hyp = Hypothesis("utt_a", "error", None, error_code="adapter_exception", error_message="SYNTHETIC failure")
    row = prediction_row(hyp, make_input("utt_a"), run_context, "t1", None)
    assert validate_predictions([row]).ok
    assert row["hypothesis_text"] is None
    assert row["error"] == {"code": "adapter_exception", "message": "SYNTHETIC failure"}


def test_prediction_row_refuses_hypothesis_for_another_utterance(make_input, run_context):
    with pytest.raises(ValueError, match="utt_b"):
        prediction_row(ok_hyp("utt_b"), make_input("utt_a"), run_context, "t1", None)


@pytest.mark.parametrize("change", [{"dataset_release": "other"}, {"split_manifest_sha256": "c" * 64}])
def test_prediction_row_refuses_input_from_another_release(make_input, run_context, change):
    inp = dataclasses.replace(make_input("utt_a"), **change)
    with pytest.raises(ValueError):
        prediction_row(ok_hyp(), inp, run_context, "t1", None)


def test_nbest_row_passes_validator_and_fills_null_reasons(make_input, run_context):
    row = nbest_row(scored_hyp(), make_input("utt_a"), run_context, 5, "nbest-synthetic-v1")
    assert validate_nbest([row]).ok
    assert tuple(row) == NBEST_ROW_FIELDS and row["actual_n"] == 2 and row["requested_n"] == 5
    first, second = row["candidates"]
    assert first["token_ids"] == [5, 6] and first["token_count"] == 2
    assert first["score_details"]["null_reasons"] == {f: NOT_PRODUCED for f in NULLABLE_SCORES[1:]}
    assert second["score_details"]["null_reasons"]["lm_log_score"] == "LM disabled"
    assert second["score_details"]["null_reasons"]["decoder_sequence_score"] == NOT_PRODUCED
    assert second["score_details"]["length_penalty"] == 1.0


def test_nbest_row_does_not_modify_adapter_details(make_input, run_context):
    hyp = scored_hyp()
    before = dict(hyp.candidates[1].score_details["null_reasons"])
    nbest_row(hyp, make_input("utt_a"), run_context, 5, "nbest-synthetic-v1")
    assert hyp.candidates[1].score_details["null_reasons"] == before


def test_nbest_row_refuses_non_object_null_reasons(make_input, run_context):
    bad = Candidate("c1", 1, "SYNTHETIC x", (1,), -1.0, "kind", score_details={"null_reasons": "none"})
    with pytest.raises(ValueError, match="null_reasons"):
        nbest_row(Hypothesis("utt_a", "ok", "SYNTHETIC x", (bad,)), make_input("utt_a"), run_context, 1, "conv")


def test_error_hypothesis_nbest_row_has_no_candidates(make_input, run_context):
    hyp = Hypothesis("utt_a", "error", None, error_code="adapter_exception", error_message="x")
    row = nbest_row(hyp, make_input("utt_a"), run_context, 3, "conv")
    assert row["candidates"] == [] and row["actual_n"] == 0 and validate_nbest([row]).ok


def test_manifest_has_all_groups(run_context):
    manifest = run_manifest(run_context, command="SYNTHETIC cmd", environment={}, configuration={},
                            outputs={}, timing={}, limitations=[])
    assert set(manifest) == ELEVEN_GROUPS == set(MANIFEST_GROUPS)


def test_manifest_records_inputs_outputs_and_timing(run_context):
    manifest = run_manifest(run_context, command="python run.py", environment={"gpu": "none"},
                            configuration={"beam": 5}, outputs={"predictions.jsonl": "c" * 64},
                            timing={"end_to_end_rtf": 0.5}, limitations=["single seed"])
    assert manifest["identity"]["run_id"] == "run_synthetic" and manifest["identity"]["schema_version"] == "0.1"
    assert manifest["code"]["command"] == "python run.py" and set(manifest["code"]) >= {"commit", "dirty"}
    assert manifest["inputs"]["split_manifest_sha256"] == "a" * 64
    assert manifest["configuration"]["resolved"] == {"beam": 5}
    assert manifest["model_knowledge"]["model_artifact_sha256"] == "b" * 64
    assert manifest["outputs"] == {"predictions.jsonl": "c" * 64}
    assert manifest["evaluation_timing"]["end_to_end_rtf"] == 0.5
    assert manifest["limitations"] == ["single seed"]


def test_git_state_outside_repo_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


def test_git_state_without_git_is_none(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(export_module.subprocess, "run", no_git)
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


@pytest.mark.parametrize("status_out,dirty", [("", False), (" M src/x.py\n", True)])
def test_git_state_reads_commit_and_dirty_flag(tmp_path, monkeypatch, status_out, dirty):
    def fake_git(args, **kwargs):
        out = "f" * 40 + "\n" if args[1] == "rev-parse" else status_out
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

    monkeypatch.setattr(export_module.subprocess, "run", fake_git)
    assert git_state(tmp_path) == {"commit": "f" * 40, "dirty": dirty}


def test_git_state_with_failing_status_keeps_commit(tmp_path, monkeypatch):
    def fake_git(args, **kwargs):
        if args[1] == "rev-parse":
            return subprocess.CompletedProcess(args, 0, stdout="e" * 40 + "\n", stderr="")
        raise subprocess.TimeoutExpired(args, 10)

    monkeypatch.setattr(export_module.subprocess, "run", fake_git)
    assert git_state(tmp_path) == {"commit": "e" * 40, "dirty": None}
