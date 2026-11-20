"""Prediction and inference-input validators. All texts and IDs are SYNTHETIC."""
import pytest

from turkish_asr_rdl.schemas.predictions import (
    FORBIDDEN_KEYS,
    INFERENCE_INPUT_FIELDS,
    PREDICTION_FIELDS,
    SCHEMA_VERSION,
    validate_inference_inputs,
    validate_predictions,
)
from turkish_asr_rdl.schemas.report import ValidationReport

HEX_A = "a" * 64
HEX_B = "b" * 64
ANY_JSON = [None, True, 1, 1.5, "", [], {}]


def ok_row(uid: str) -> dict:
    return {
        "schema_version": "0.1",
        "run_id": "run_synthetic",
        "system_id": "sys_synthetic",
        "dataset_release": "rel_synthetic",
        "split": "test",
        "split_manifest_sha256": HEX_A,
        "utterance_id": uid,
        "audio_sha256": HEX_B,
        "status": "ok",
        "hypothesis_text": "SYNTHETIC merhaba",
        "selected_candidate_id": None,
        "nbest_artifact_sha256": None,
        "model_artifact_sha256": HEX_A,
        "decoder_config_sha256": HEX_B,
        "normalization_version": "tr-normalized-v1",
        "timing_record_id": "t1",
        "error": None,
    }


def empty_row(uid: str) -> dict:
    return {**ok_row(uid), "status": "empty", "hypothesis_text": ""}


def err_row(uid: str) -> dict:
    return {**ok_row(uid), "status": "error", "hypothesis_text": None, "error": {"code": "x", "message": "y"}}


def input_row(uid: str) -> dict:
    return {
        "schema_version": "0.1",
        "dataset_release": "rel_synthetic",
        "split_manifest_sha256": HEX_A,
        "utterance_id": uid,
        "audio_uri": f"audio/{uid}.flac",
        "audio_sha256": HEX_B,
        "duration_s": 1.5,
        "sample_rate_hz": 16000,
        "channels": 1,
        "language": "tr",
    }


def errors_of(rows, **kwargs) -> tuple[str, ...]:
    return validate_predictions(rows, **kwargs).errors


def test_constants_match_the_contract():
    assert SCHEMA_VERSION == "0.1"
    assert len(PREDICTION_FIELDS) == 17 and PREDICTION_FIELDS[0] == "schema_version" and PREDICTION_FIELDS[-1] == "error"
    assert set(ok_row("u")) == set(PREDICTION_FIELDS)
    assert set(input_row("u")) == set(INFERENCE_INPUT_FIELDS)
    assert {"reference_text", "gold", "labels", "reference_token_ids"} <= FORBIDDEN_KEYS


def test_valid_rows_pass():
    assert validate_predictions([ok_row("utt_a"), ok_row("utt_b")]).ok


def test_empty_and_error_rows_pass():
    report = validate_predictions([ok_row("utt_a"), empty_row("utt_b"), err_row("utt_c")])
    assert report == ValidationReport()


def test_no_rows_pass():
    assert validate_predictions([]).ok


def test_ok_with_empty_text_says_use_empty():
    r = validate_predictions([{**ok_row("utt_a"), "hypothesis_text": ""}])
    assert any('status "empty"' in e for e in r.errors)


def test_ok_with_null_text_fails():
    assert errors_of([{**ok_row("utt_a"), "hypothesis_text": None}])


def test_empty_status_needs_empty_string():
    assert errors_of([{**empty_row("utt_a"), "hypothesis_text": "SYNTHETIC x"}])
    assert errors_of([{**empty_row("utt_a"), "hypothesis_text": None}])


@pytest.mark.parametrize("change", [
    {"hypothesis_text": ""},
    {"hypothesis_text": "SYNTHETIC x"},
    {"error": None},
    {"error": {"message": "y"}},
    {"error": {"code": 1, "message": "y"}},
    {"error": {"code": "x", "message": None}},
    {"error": "x"},
])
def test_error_status_needs_null_text_and_error_object(change):
    assert errors_of([{**err_row("utt_a"), **change}])


def test_non_error_rows_need_null_error():
    r = errors_of([{**ok_row("utt_a"), "error": {"code": "x", "message": "y"}}])
    assert any("error null" in e for e in r)


@pytest.mark.parametrize("status", ["OK", "failed", None, 1])
def test_unknown_status_fails(status):
    assert any("status" in e for e in errors_of([{**ok_row("utt_a"), "status": status}]))


def test_missing_field_reported():
    row = ok_row("utt_a")
    del row["timing_record_id"]
    assert any("missing" in e and "timing_record_id" in e for e in errors_of([row]))


def test_unknown_field_reported():
    assert any("unknown" in e and "confidence" in e for e in errors_of([{**ok_row("utt_a"), "confidence": 0.5}]))


def test_reference_text_top_level_is_forbidden():
    r = errors_of([{**ok_row("utt_a"), "reference_text": "SYNTHETIC x"}])
    assert len(r) == 1 and "forbidden" in r[0] and "reference_text" in r[0]


def test_reference_text_nested_is_forbidden():
    r = validate_predictions([{**err_row("utt_a"), "error": {"code": "x", "message": "y", "gold": "z"}}])
    assert any("gold" in e for e in r.errors)


def test_forbidden_key_inside_a_list_is_found():
    r = errors_of([{**err_row("utt_a"), "error": {"code": "x", "message": "y", "trace": [{"labels": [1]}]}}])
    assert any("labels" in e for e in r)


def test_schema_version_must_be_exact():
    assert any("schema_version" in e for e in errors_of([{**ok_row("utt_a"), "schema_version": "0.1.0"}]))


@pytest.mark.parametrize("field", ["split_manifest_sha256", "audio_sha256", "model_artifact_sha256", "decoder_config_sha256"])
@pytest.mark.parametrize("value", ["A" * 64, "a" * 63, "a" * 65, "g" * 64, None, 7])
def test_bad_sha256_rejected(field, value):
    assert any(field in e for e in errors_of([{**ok_row("utt_a"), field: value}]))


def test_nullable_sha256_accepts_hash_and_rejects_bad_text():
    assert validate_predictions([{**ok_row("utt_a"), "nbest_artifact_sha256": HEX_A}]).ok
    assert errors_of([{**ok_row("utt_a"), "nbest_artifact_sha256": "A" * 64}])


@pytest.mark.parametrize("field", ["run_id", "system_id", "utterance_id", "timing_record_id", "normalization_version"])
@pytest.mark.parametrize("value", ["", None, 3])
def test_identity_fields_must_be_non_empty_strings(field, value):
    assert any(field in e for e in errors_of([{**ok_row("utt_a"), field: value}]))


def test_selected_candidate_id_is_string_or_null():
    assert validate_predictions([{**ok_row("utt_a"), "selected_candidate_id": "c1"}]).ok
    assert errors_of([{**ok_row("utt_a"), "selected_candidate_id": 1}])


def test_non_finite_numbers_rejected():
    row = {**err_row("utt_a"), "error": {"code": "x", "message": "y", "seconds": float("nan")}}
    assert any("finite" in e for e in errors_of([row]))


def test_duplicate_utterance_id_rejected():
    assert any("duplicate" in e and "utt_a" in e for e in errors_of([ok_row("utt_a"), ok_row("utt_a")]))


@pytest.mark.parametrize("field", ["run_id", "system_id", "dataset_release", "split", "split_manifest_sha256"])
def test_mixed_run_ids_rejected(field):
    other = {"split_manifest_sha256": HEX_B}.get(field, "other_synthetic")
    r = errors_of([ok_row("utt_a"), {**ok_row("utt_b"), field: other}])
    assert any(field in e and "same" in e for e in r)


def test_missing_and_extra_ids_listed():
    r = errors_of([ok_row("utt_a"), ok_row("utt_x")], expected_ids=["utt_a", "utt_b"])
    assert any("missing" in e and "utt_b" in e for e in r)
    assert any("utt_x" in e and "not in the expected" in e for e in r)


def test_long_id_lists_are_truncated_with_count():
    expected = [f"utt_{i:03d}" for i in range(25)]
    r = errors_of([], expected_ids=expected)
    assert len(r) == 1 and "25" in r[0] and "15 more" in r[0] and "utt_024" not in r[0]


def test_order_must_match_expected_ids():
    r = validate_predictions([ok_row("utt_b"), ok_row("utt_a")], expected_ids=["utt_a", "utt_b"])
    assert any("order" in e for e in r.errors)


def test_order_reports_first_mismatch_position():
    rows = [ok_row("utt_a"), ok_row("utt_c"), ok_row("utt_b")]
    r = errors_of(rows, expected_ids=["utt_a", "utt_b", "utt_c"])
    assert any("order" in e and "position 2" in e for e in r)


def test_expected_ids_in_order_pass():
    assert validate_predictions([ok_row("utt_a"), ok_row("utt_b")], expected_ids=("utt_a", "utt_b")).ok


def test_error_names_utterance_id_or_row_number():
    r = errors_of([ok_row("utt_a"), {**ok_row("utt_b"), "status": "bad"}, {**ok_row(None), "status": "bad"}])
    assert any(e.startswith("row 2 (utt_b):") for e in r)
    assert any(e.startswith("row 3:") for e in r)


def test_validator_does_not_modify_rows():
    rows = [ok_row("utt_b"), err_row("utt_a")]
    snapshot = [dict(r) for r in rows]
    validate_predictions(rows, expected_ids=["utt_a"])
    assert rows == snapshot


@pytest.mark.parametrize("field", PREDICTION_FIELDS)
@pytest.mark.parametrize("value", ANY_JSON)
def test_prediction_validator_never_raises(field, value):
    report = validate_predictions([{**ok_row("utt_a"), field: value}, ok_row("utt_b")], expected_ids=["utt_a"])
    assert isinstance(report, ValidationReport)


def test_deeply_nested_row_is_an_error_not_a_crash():
    nested: list = []
    for _ in range(5000):
        nested = [nested]
    r = errors_of([{**err_row("utt_a"), "error": {"code": "x", "message": "y", "trace": nested}}])
    assert any("nested too deeply" in e for e in r)


@pytest.mark.parametrize("row", [[1, 2], "row", None, 3])
def test_non_object_row_is_an_error(row):
    assert any("object" in e for e in errors_of([row]))


def test_valid_inference_inputs_pass():
    assert validate_inference_inputs([input_row("utt_a"), input_row("utt_b")]) == ValidationReport()


def test_inference_input_extra_key_rejected():
    r = validate_inference_inputs([{**input_row("utt_a"), "reference_text": "SYNTHETIC x"}])
    assert not r.ok and any("reference_text" in e for e in r.errors)


def test_inference_input_unknown_key_rejected():
    r = validate_inference_inputs([{**input_row("utt_a"), "speaker": "s1"}])
    assert any("unknown" in e and "speaker" in e for e in r.errors)


def test_inference_input_missing_key_rejected():
    row = input_row("utt_a")
    del row["language"]
    assert any("missing" in e and "language" in e for e in validate_inference_inputs([row]).errors)


@pytest.mark.parametrize("field,value", [
    ("duration_s", 0), ("duration_s", -1.0), ("duration_s", float("inf")), ("duration_s", True), ("duration_s", "1"),
    ("sample_rate_hz", 0), ("sample_rate_hz", 16000.0), ("sample_rate_hz", True),
    ("channels", -1), ("channels", 1.5), ("channels", None),
    ("audio_uri", ""), ("audio_uri", None),
    ("schema_version", "0.2"), ("audio_sha256", "A" * 64), ("split_manifest_sha256", "x"),
    ("utterance_id", ""), ("language", 1),
])
def test_inference_input_bad_values_rejected(field, value):
    assert any(field in e for e in validate_inference_inputs([{**input_row("utt_a"), field: value}]).errors)


def test_inference_input_integer_duration_accepted():
    assert validate_inference_inputs([{**input_row("utt_a"), "duration_s": 2}]).ok


@pytest.mark.parametrize("field", INFERENCE_INPUT_FIELDS)
@pytest.mark.parametrize("value", ANY_JSON)
def test_inference_validator_never_raises(field, value):
    assert isinstance(validate_inference_inputs([{**input_row("utt_a"), field: value}]), ValidationReport)


def test_inference_non_object_row_is_an_error():
    assert not validate_inference_inputs([["utt_a"]]).ok
