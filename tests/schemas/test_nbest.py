"""N-best validator. All candidate texts and IDs are SYNTHETIC."""
import pytest

from turkish_asr_rdl.schemas.nbest import (
    CANDIDATE_FIELDS,
    NBEST_ROW_FIELDS,
    NULLABLE_SCORES,
    validate_nbest,
)
from turkish_asr_rdl.schemas.report import ValidationReport

HEX_A = "a" * 64
HEX_B = "b" * 64
ANY_JSON = [None, True, 1, 1.5, "", [], {}]
DEFAULT_TEXTS = ("SYNTHETIC bir", "SYNTHETIC iki", "SYNTHETIC üç")


def cand(k: int, text: str, rank: int) -> dict:
    return {
        "candidate_id": f"c{k}",
        "rank": rank,
        "text": text,
        "token_ids": [1, 2],
        "token_count": 2,
        "acoustic_log_score": -1.5 * k,
        "acoustic_score_kind": "seq2seq_teacher_forced_log_prob",
        "decoder_sequence_score": -0.5,
        "lm_log_score": None,
        "rdl_log_score": None,
        "combined_score": None,
        "score_details": {"null_reasons": {"lm_log_score": "no LM", "rdl_log_score": "no RDL", "combined_score": "no fusion"}},
    }


def nbest_row(texts=DEFAULT_TEXTS, ranks=None, uid="utt_a", requested_n=5) -> dict:
    ranks = list(ranks) if ranks is not None else list(range(1, len(texts) + 1))
    candidates = [cand(k, t, r) for k, (t, r) in enumerate(zip(texts, ranks), start=1)]
    return {
        "schema_version": "0.1",
        "run_id": "run_synthetic",
        "system_id": "sys_synthetic",
        "dataset_release": "rel_synthetic",
        "split": "test",
        "split_manifest_sha256": HEX_A,
        "utterance_id": uid,
        "audio_sha256": HEX_B,
        "model_artifact_sha256": HEX_A,
        "decoder_config_sha256": HEX_B,
        "requested_n": requested_n,
        "actual_n": len(candidates),
        "score_convention_id": "nbest-synthetic-v1",
        "candidates": candidates,
    }


def with_candidate(row: dict, index: int, **changes) -> dict:
    candidates = [dict(c) for c in row["candidates"]]
    candidates[index] = {**candidates[index], **changes}
    return {**row, "candidates": candidates}


def errors_of(rows, **kwargs) -> tuple[str, ...]:
    return validate_nbest(rows, **kwargs).errors


def test_constants_match_the_contract():
    assert set(nbest_row()) == set(NBEST_ROW_FIELDS)
    assert set(cand(1, "x", 1)) == set(CANDIDATE_FIELDS)
    assert NULLABLE_SCORES == ("decoder_sequence_score", "lm_log_score", "rdl_log_score", "combined_score")


def test_valid_row_passes():
    assert validate_nbest([nbest_row(), nbest_row(uid="utt_b")]) == ValidationReport()


def test_whisper_duplicate_trap_fails():
    row = nbest_row(texts=["merhaba dünya"] * 3)  # SYNTHETIC
    assert any("duplicate candidate text" in e for e in validate_nbest([row]).errors)


def test_duplicate_text_message_names_dedup_map():
    errors = errors_of([nbest_row(texts=["SYNTHETIC a", "SYNTHETIC b", "SYNTHETIC a"])])
    assert any("record it in dedup_map" in e and "c1" in e and "c3" in e for e in errors)


def test_deduplicated_list_with_map_passes():
    row = {**nbest_row(texts=["SYNTHETIC a", "SYNTHETIC b"], ranks=[1, 3]), "dedup_map": {"c_raw2": "c1"}}
    assert validate_nbest([row]).ok


@pytest.mark.parametrize("dedup_map", [{"c_raw2": "c9"}, {"c1": "c2"}, {"c_raw2": 1}, ["c1"], "c1"])
def test_bad_dedup_map_fails(dedup_map):
    assert any("dedup_map" in e for e in errors_of([{**nbest_row(), "dedup_map": dedup_map}]))


def test_rank_gaps_after_dedup_allowed():
    assert validate_nbest([nbest_row(ranks=[1, 3, 4])]).ok


@pytest.mark.parametrize("ranks", [[2, 3, 4], [1, 3, 5]])
def test_ranks_kept_after_dedup_may_start_above_one(ranks):
    assert validate_nbest([nbest_row(ranks=ranks)]).ok


@pytest.mark.parametrize("ranks", [[1, 1, 2], [1, 3, 2], [0, 1, 2], [1, 2, 6]])
def test_bad_rank_sequences_fail(ranks):
    assert any("rank" in e for e in errors_of([nbest_row(ranks=ranks)]))


@pytest.mark.parametrize("rank", [True, 1.0, "1", None])
def test_rank_must_be_an_integer(rank):
    assert any("rank" in e for e in errors_of([with_candidate(nbest_row(), 0, rank=rank)]))


def test_null_score_without_reason_fails():
    row = with_candidate(nbest_row(), 1, score_details={"null_reasons": {"lm_log_score": "no LM"}})
    errors = errors_of([row])
    assert any("rdl_log_score" in e and "null_reasons" in e for e in errors)
    assert any("combined_score" in e for e in errors)


def test_null_score_with_reason_passes():
    row = with_candidate(nbest_row(), 0, decoder_sequence_score=None, score_details={
        "null_reasons": {f: "not produced by this system" for f in NULLABLE_SCORES}})
    assert validate_nbest([row]).ok


@pytest.mark.parametrize("details", [{}, {"null_reasons": {}}, {"null_reasons": {"lm_log_score": ""}},
                                     {"null_reasons": "none"}, None, "x"])
def test_null_reasons_must_be_present_and_non_empty(details):
    assert errors_of([with_candidate(nbest_row(), 0, score_details=details)])


@pytest.mark.parametrize("value", ["x", float("nan"), True])
def test_present_score_must_be_finite_number(value):
    assert any("decoder_sequence_score" in e for e in errors_of([with_candidate(nbest_row(), 0, decoder_sequence_score=value)]))


def test_normalized_duplicates_are_warnings_only():
    r = validate_nbest([nbest_row(texts=["Ankara", "ankara"])], normalize=str.lower)
    assert r.ok and r.warnings


def test_no_warnings_without_normalizer():
    assert validate_nbest([nbest_row(texts=["Ankara", "ankara"])]) == ValidationReport()


def test_actual_n_mismatch_fails():
    assert any("actual_n" in e for e in errors_of([{**nbest_row(), "actual_n": 2}]))


def test_actual_n_above_requested_n_fails():
    assert any("requested_n" in e for e in errors_of([nbest_row(requested_n=2, ranks=[1, 2, 2])]))


@pytest.mark.parametrize("value", [0, -1, True, 2.0, "5", None])
def test_requested_n_must_be_positive_integer(value):
    assert any("requested_n" in e for e in errors_of([{**nbest_row(), "requested_n": value}]))


def test_empty_candidate_list_is_allowed():
    assert validate_nbest([{**nbest_row(texts=()), "actual_n": 0}]).ok


def test_token_count_mismatch_fails():
    assert any("token_count" in e for e in errors_of([with_candidate(nbest_row(), 0, token_count=3)]))


@pytest.mark.parametrize("token_ids", [[1, True], ["1"], "12", None])
def test_token_ids_must_be_integers(token_ids):
    assert any("token_ids" in e for e in errors_of([with_candidate(nbest_row(), 0, token_ids=token_ids)]))


@pytest.mark.parametrize("value", [None, float("nan"), float("-inf"), "x", True])
def test_missing_acoustic_score_fails(value):
    assert any("acoustic_log_score" in e for e in errors_of([with_candidate(nbest_row(), 0, acoustic_log_score=value)]))


def test_absent_acoustic_score_key_fails():
    row = nbest_row()
    first = {k: v for k, v in row["candidates"][0].items() if k != "acoustic_log_score"}
    errors = errors_of([{**row, "candidates": [first, *row["candidates"][1:]]}])
    assert any("acoustic_log_score" in e for e in errors)


def test_candidate_ids_must_be_unique():
    assert any("candidate_id" in e for e in errors_of([with_candidate(nbest_row(), 1, candidate_id="c1")]))


@pytest.mark.parametrize("field", ["candidate_id", "acoustic_score_kind", "text"])
def test_candidate_strings_checked(field):
    assert any(field in e for e in errors_of([with_candidate(nbest_row(), 0, **{field: 5})]))


def test_missing_row_field_fails():
    row = {k: v for k, v in nbest_row().items() if k != "score_convention_id"}
    assert any("missing" in e and "score_convention_id" in e for e in errors_of([row]))


@pytest.mark.parametrize("field,value", [("schema_version", "0.2"), ("audio_sha256", "A" * 64),
                                         ("run_id", ""), ("score_convention_id", None)])
def test_row_identity_fields_checked(field, value):
    assert any(field in e for e in errors_of([{**nbest_row(), field: value}]))


def test_forbidden_key_in_score_details_fails():
    row = with_candidate(nbest_row(), 0, score_details={"null_reasons": {"lm_log_score": "a", "rdl_log_score": "b",
                                                                         "combined_score": "c"}, "gold": "x"})
    assert any("gold" in e for e in errors_of([row]))


def test_duplicate_utterance_rows_fail():
    assert any("duplicate" in e for e in errors_of([nbest_row(), nbest_row()]))


def test_expected_ids_checked_like_predictions():
    rows = [nbest_row(uid="utt_b"), nbest_row(uid="utt_a")]
    errors = errors_of(rows, expected_ids=["utt_a", "utt_b", "utt_c"])
    assert any("order" in e for e in errors) and any("missing" in e and "utt_c" in e for e in errors)


def test_validator_does_not_modify_rows():
    rows = [nbest_row(texts=["SYNTHETIC a", "SYNTHETIC a"])]
    snapshot = [{**r, "candidates": [dict(c) for c in r["candidates"]]} for r in rows]
    validate_nbest(rows, expected_ids=["utt_a"], normalize=str.lower)
    assert rows == snapshot


@pytest.mark.parametrize("field", NBEST_ROW_FIELDS)
@pytest.mark.parametrize("value", ANY_JSON)
def test_row_level_never_raises(field, value):
    report = validate_nbest([{**nbest_row(), field: value}], expected_ids=["utt_a"], normalize=str.lower)
    assert isinstance(report, ValidationReport) and not report.ok


@pytest.mark.parametrize("field", CANDIDATE_FIELDS)
@pytest.mark.parametrize("value", ANY_JSON)
def test_candidate_level_never_raises(field, value):
    report = validate_nbest([with_candidate(nbest_row(), 0, **{field: value})], normalize=str.lower)
    assert isinstance(report, ValidationReport)


@pytest.mark.parametrize("candidate", [None, "c1", [1], 3])
def test_non_object_candidate_fails(candidate):
    row = nbest_row()
    assert any("candidate" in e for e in errors_of([{**row, "candidates": [candidate, *row["candidates"][1:]]}]))


@pytest.mark.parametrize("row", [None, [1], "row"])
def test_non_object_row_fails(row):
    assert any("object" in e for e in errors_of([row]))
