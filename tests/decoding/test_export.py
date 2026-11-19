"""Adapter interface, export rows and run manifest. All texts and IDs are SYNTHETIC."""
import dataclasses
import subprocess

import numpy as np
import pytest

import turkish_asr_rdl.decoding.export as export_module
from turkish_asr_rdl.decoding.export import (
    MANIFEST_GROUPS,
    NOT_PRODUCED,
    adapter_result_fields,
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


@pytest.mark.parametrize("details", [None, ["SYNTHETIC"], "SYNTHETIC"])
def test_nbest_row_refuses_non_object_score_details(make_input, run_context, details):
    bad = Candidate("c1", 1, "SYNTHETIC x", (1,), -1.0, "kind", score_details=details)
    with pytest.raises(ValueError, match="'c1': score_details must be an object"):
        nbest_row(Hypothesis("utt_a", "ok", "SYNTHETIC x", (bad,)), make_input("utt_a"), run_context, 1, "conv")


def test_nbest_row_refuses_candidates_that_are_not_a_tuple_or_list(make_input, run_context):
    hyp = Hypothesis("utt_a", "ok", "SYNTHETIC bir", iter(scored_hyp().candidates))
    with pytest.raises(ValueError, match="candidates must be a tuple or list"):
        nbest_row(hyp, make_input("utt_a"), run_context, 5, "conv")


RESULT_KEYS = ("status", "hypothesis_text", "selected_candidate_id", "error")


def test_adapter_result_fields_are_the_values_the_rows_take_from_the_result(make_input, run_context):
    hyp = dataclasses.replace(scored_hyp(), selected_candidate_id="c1")
    pred = prediction_row(hyp, make_input("utt_a"), run_context, "t1", "c" * 64)
    nbest = nbest_row(hyp, make_input("utt_a"), run_context, 5, "conv")
    fields = adapter_result_fields(hyp, 5)
    assert fields == {**{key: pred[key] for key in RESULT_KEYS}, "candidates": nbest["candidates"]}
    assert fields["selected_candidate_id"] == pred["selected_candidate_id"] == "c1"
    plain = prediction_row(scored_hyp(), make_input("utt_a"), run_context, "t1", None)
    assert adapter_result_fields(scored_hyp(), 0) == {key: plain[key] for key in RESULT_KEYS}


def test_adapter_result_fields_refuse_a_selection_the_rows_cannot_hold():
    with pytest.raises(ValueError, match="selected_candidate_id 'c9' must name one candidate"):
        adapter_result_fields(dataclasses.replace(scored_hyp(), selected_candidate_id="c9"), 5)
    with pytest.raises(ValueError, match="selected_candidate_id 'c1' needs an N-best artifact"):
        adapter_result_fields(dataclasses.replace(scored_hyp(), selected_candidate_id="c1"), 0)


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


DUPLICATE_POLICY_TEXT = "exact raw-text match; first (lowest-rank) occurrence kept; removed IDs listed in dedup_map"


def cand(cid, rank: int, text: str, token_ids=None, **changes) -> Candidate:
    tokens = (rank, rank + 100) if token_ids is None else token_ids
    return Candidate(cid, rank, text, tokens, -1.0 * rank, "seq2seq_teacher_forced_log_prob", **changes)


def hyp_of(*candidates: Candidate) -> Hypothesis:
    return Hypothesis("utt_a", "ok", candidates[0].text, candidates)


def test_dedupe_keeps_the_first_occurrence_of_each_text():
    c01, c02, c03, c04, c05 = (cand("c01", 1, "SYNTHETIC a"), cand("c02", 2, "SYNTHETIC b"),
                               cand("c03", 3, "SYNTHETIC a"), cand("c04", 4, "SYNTHETIC b"),
                               cand("c05", 5, "SYNTHETIC c"))
    kept, dedup_map = export_module._dedupe((c01, c02, c03, c04, c05))
    assert kept == (c01, c02, c05)
    assert dedup_map == {"c03": "c01", "c04": "c02"}


def test_nbest_row_drops_repeated_texts_and_keeps_original_ranks(make_input, run_context):
    hyp = hyp_of(cand("c01", 1, "SYNTHETIC a"), cand("c02", 2, "SYNTHETIC b"), cand("c03", 3, "SYNTHETIC a"),
                 cand("c04", 4, "SYNTHETIC b"), cand("c05", 5, "SYNTHETIC c"))
    row = nbest_row(hyp, make_input("utt_a"), run_context, 5, "conv")
    assert [c["candidate_id"] for c in row["candidates"]] == ["c01", "c02", "c05"]
    assert tuple(c["rank"] for c in row["candidates"]) == (1, 2, 5)
    assert row["dedup_map"] == {"c03": "c01", "c04": "c02"}
    assert row["actual_n"] == 3 and tuple(row) == (*NBEST_ROW_FIELDS, "dedup_map")
    assert validate_nbest([row]).ok


def test_every_removed_id_maps_to_the_kept_id_not_the_previous_copy(make_input, run_context):
    hyp = hyp_of(cand("c01", 1, "SYNTHETIC a"), cand("c02", 2, "SYNTHETIC a"), cand("c03", 3, "SYNTHETIC a"),
                 cand("c04", 4, "SYNTHETIC b"), cand("c05", 5, "SYNTHETIC b"))
    row = nbest_row(hyp, make_input("utt_a"), run_context, 5, "conv")
    assert row["dedup_map"] == {"c02": "c01", "c03": "c01", "c05": "c04"}
    assert row["actual_n"] == 2 and validate_nbest([row]).ok


def test_distinct_texts_write_no_dedup_map(make_input, run_context):
    row = nbest_row(scored_hyp(), make_input("utt_a"), run_context, 5, "conv")
    assert "dedup_map" not in row and tuple(row) == NBEST_ROW_FIELDS
    assert "dedup_map" not in adapter_result_fields(scored_hyp(), 5)


def test_texts_are_compared_byte_for_byte(make_input, run_context):
    texts = (" SYNTHETIC a", "SYNTHETIC a", "SYNTHETIC A", "SYNTHETIC a ", "SYNTHETIC \u00e7", "SYNTHETIC c\u0327")
    hyp = hyp_of(*(cand(f"c{k:02d}", k, text) for k, text in enumerate(texts, start=1)))
    row = nbest_row(hyp, make_input("utt_a"), run_context, 6, "conv")
    assert [c["text"] for c in row["candidates"]] == list(texts)
    assert "dedup_map" not in row and row["actual_n"] == 6


def test_adapter_result_fields_hold_the_deduplicated_candidates_and_map(make_input, run_context):
    hyp = hyp_of(cand("c01", 1, "SYNTHETIC a"), cand("c02", 2, "SYNTHETIC a"), cand("c03", 3, "SYNTHETIC b"))
    row = nbest_row(hyp, make_input("utt_a"), run_context, 3, "conv")
    fields = adapter_result_fields(hyp, 3)
    assert list(fields) == ["status", "hypothesis_text", "selected_candidate_id", "error", "candidates", "dedup_map"]
    assert fields["candidates"] == row["candidates"] and fields["dedup_map"] == row["dedup_map"] == {"c02": "c01"}


def test_candidates_carry_the_duplicate_policy_unless_the_adapter_gives_one(make_input, run_context):
    own = {"duplicate_policy": "SYNTHETIC own policy"}
    hyp = hyp_of(cand("c01", 1, "SYNTHETIC a"), cand("c02", 2, "SYNTHETIC b", score_details=own))
    first, second = nbest_row(hyp, make_input("utt_a"), run_context, 2, "conv")["candidates"]
    assert export_module.DUPLICATE_POLICY == DUPLICATE_POLICY_TEXT
    assert first["score_details"]["duplicate_policy"] == DUPLICATE_POLICY_TEXT
    assert second["score_details"]["duplicate_policy"] == "SYNTHETIC own policy"


def test_nbest_row_refuses_copies_of_one_hypothesis(make_input, run_context):
    hyp = hyp_of(cand("c01", 1, "SYNTHETIC a", (7, 8)), cand("c02", 2, "SYNTHETIC a", [7, 8]))
    with pytest.raises(ValueError, match=r"^decoder returned copies of one hypothesis \(same text and token_ids\)"):
        nbest_row(hyp, make_input("utt_a"), run_context, 2, "conv")


def test_shared_token_ids_with_different_texts_are_not_copies(make_input, run_context):
    hyp = hyp_of(cand("c01", 1, "SYNTHETIC a", (7, 8)), cand("c02", 2, "SYNTHETIC a ", (7, 8)))
    row = nbest_row(hyp, make_input("utt_a"), run_context, 2, "conv")
    assert row["actual_n"] == 2 and validate_nbest([row]).ok


@pytest.mark.parametrize("removed_id", [3, None, ""])
def test_removed_candidate_id_must_be_a_non_empty_string(make_input, run_context, removed_id):
    hyp = hyp_of(cand("c01", 1, "SYNTHETIC a"), cand(removed_id, 2, "SYNTHETIC a"))
    with pytest.raises(ValueError, match="candidate_id"):
        nbest_row(hyp, make_input("utt_a"), run_context, 2, "conv")


@pytest.mark.parametrize("texts", [("SYNTHETIC a", "SYNTHETIC a"), ("SYNTHETIC a", "SYNTHETIC b")])
def test_two_removed_candidates_with_one_id_are_refused(make_input, run_context, texts):
    first, second = texts
    hyp = hyp_of(cand("c01", 1, "SYNTHETIC a"), cand("c02", 2, "SYNTHETIC b"),
                 cand("c09", 3, first), cand("c09", 4, second))
    with pytest.raises(ValueError, match=r"^removed duplicate candidate_id 'c09' appears more than once$"):
        nbest_row(hyp, make_input("utt_a"), run_context, 4, "conv")


def test_candidate_and_hypothesis_end_with_the_rescoring_fields():
    candidate = dataclasses.fields(Candidate)
    names = ("score_details", "lm_log_score", "rdl_log_score", "combined_score")
    assert tuple(f.name for f in candidate[-4:]) == names
    assert [f.default for f in candidate[-3:]] == [None, None, None]
    assert (dataclasses.fields(Hypothesis)[-1].name, dataclasses.fields(Hypothesis)[-1].default) == (
        "selected_candidate_id", None)


def test_nbest_row_carries_each_rescoring_score_into_its_own_key(make_input, run_context):
    hyp = hyp_of(cand("c01", 1, "SYNTHETIC a", lm_log_score=-3.5, rdl_log_score=-0.25, combined_score=-1.75),
                 cand("c02", 2, "SYNTHETIC b", combined_score=-2.0),
                 cand("c03", 3, "SYNTHETIC c", rdl_log_score=-0.5,
                      score_details={"null_reasons": {"lm_log_score": "SYNTHETIC LM disabled"}}))
    row = nbest_row(hyp, make_input("utt_a"), run_context, 3, "conv")
    first, second, third = row["candidates"]
    assert (first["lm_log_score"], first["rdl_log_score"], first["combined_score"]) == (-3.5, -0.25, -1.75)
    assert first["score_details"]["null_reasons"] == {"decoder_sequence_score": NOT_PRODUCED}
    assert (second["lm_log_score"], second["rdl_log_score"], second["combined_score"]) == (None, None, -2.0)
    assert second["score_details"]["null_reasons"] == {
        "decoder_sequence_score": NOT_PRODUCED, "lm_log_score": NOT_PRODUCED, "rdl_log_score": NOT_PRODUCED}
    assert third["score_details"]["null_reasons"] == {
        "decoder_sequence_score": NOT_PRODUCED, "lm_log_score": "SYNTHETIC LM disabled",
        "combined_score": NOT_PRODUCED}
    assert validate_nbest([row]).ok


def test_prediction_row_records_the_selected_candidate(make_input, run_context):
    hyp = dataclasses.replace(scored_hyp(), text="SYNTHETIC iki", selected_candidate_id="c2")
    row = prediction_row(hyp, make_input("utt_a"), run_context, "t1", "c" * 64)
    assert row["selected_candidate_id"] == "c2" and validate_predictions([row]).ok
    unset = prediction_row(scored_hyp(), make_input("utt_a"), run_context, "t1", "c" * 64)
    assert unset["selected_candidate_id"] is None


def removed_duplicate() -> Hypothesis:
    candidates = (cand("c01", 1, "SYNTHETIC a"), cand("c02", 2, "SYNTHETIC a"))
    return Hypothesis("utt_a", "ok", "SYNTHETIC a", candidates, selected_candidate_id="c02")


def error_naming_a_candidate() -> Hypothesis:
    return Hypothesis("utt_a", "error", None, scored_hyp().candidates, error_code="adapter_exception",
                      error_message="SYNTHETIC failure", selected_candidate_id="c1")


BAD_SELECTIONS = {
    "unknown_id": (lambda: dataclasses.replace(scored_hyp(), selected_candidate_id="c9"), "c" * 64),
    "removed_duplicate": (removed_duplicate, "c" * 64),
    "text_differs": (lambda: dataclasses.replace(scored_hyp(), selected_candidate_id="c2"), "c" * 64),
    "no_nbest_artifact": (
        lambda: dataclasses.replace(scored_hyp(), text="SYNTHETIC iki", selected_candidate_id="c2"), None),
    "non_string_id": (
        lambda: Hypothesis("utt_a", "ok", "SYNTHETIC a", (cand("1", 1, "SYNTHETIC a"),), selected_candidate_id=1),
        "c" * 64),
    "error_status": (error_naming_a_candidate, "c" * 64),
    "int_id_on_a_candidate": (
        lambda: Hypothesis("utt_a", "ok", "SYNTHETIC a", (cand(1, 1, "SYNTHETIC a"),), selected_candidate_id=1),
        "c" * 64),
    "true_id_on_a_candidate": (
        lambda: Hypothesis("utt_a", "ok", "SYNTHETIC a", (cand(True, 1, "SYNTHETIC a"),),
                           selected_candidate_id=True),
        "c" * 64),
    "error_status_with_the_candidate_text": (
        lambda: dataclasses.replace(error_naming_a_candidate(), text="SYNTHETIC bir"), "c" * 64),
    "error_status_with_no_text_on_either": (
        lambda: dataclasses.replace(error_naming_a_candidate(), candidates=(cand("c1", 1, None),)), "c" * 64),
    "id_on_two_kept_candidates": (
        lambda: Hypothesis("utt_a", "ok", "SYNTHETIC a", (cand("c01", 1, "SYNTHETIC a"), cand("c01", 2, "SYNTHETIC b")),
                           selected_candidate_id="c01"),
        "c" * 64),
}


@pytest.mark.parametrize("build,nbest_sha", BAD_SELECTIONS.values(), ids=BAD_SELECTIONS.keys())
def test_prediction_row_refuses_a_selection_that_is_not_a_kept_candidate(make_input, run_context, build, nbest_sha):
    with pytest.raises(ValueError, match="selected_candidate_id"):
        prediction_row(build(), make_input("utt_a"), run_context, "t1", nbest_sha)


@pytest.mark.parametrize("build,nbest_sha", BAD_SELECTIONS.values(), ids=BAD_SELECTIONS.keys())
def test_adapter_result_fields_refuse_a_selection_that_is_not_a_kept_candidate(build, nbest_sha):
    with pytest.raises(ValueError, match="selected_candidate_id"):
        adapter_result_fields(build(), 5 if nbest_sha else 0)
