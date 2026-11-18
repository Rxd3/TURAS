"""N-best facts of the 5070 Ti smoke record, read from rows the shared export already deduplicated. SYNTHETIC."""
import pytest

from turkish_asr_rdl.decoding.export import RunContext, nbest_row
from turkish_asr_rdl.decoding.interface import Candidate, Hypothesis, InferenceInput


def _nbest_rows(texts_per_utterance):
    ctx = RunContext("run_syn", "sys_syn", "rel_syn", "train", "a" * 64, "b" * 64, "c" * 64)
    rows = []
    for k, texts in enumerate(texts_per_utterance):
        uid = f"utt_{k}"
        inp = InferenceInput("0.1", "rel_syn", "a" * 64, uid, f"{uid}.flac", "d" * 64, 1.0, 16000, 1, "tr")
        cands = tuple(Candidate(f"{uid}-c{r}", r, t, (r,), -float(r), "seq2seq_token_log_prob_sum")
                      for r, t in enumerate(texts, start=1))
        rows.append(nbest_row(Hypothesis(uid, "ok", texts[0], cands), inp, ctx, 3, "conv_syn"))
    return rows


def test_nbest_facts_count_the_duplicates_the_export_removed(smoke):
    facts = smoke.nbest_facts(_nbest_rows([["SYN a", "SYN b", "SYN c"], ["SYN d", "SYN d"],
                                           ["SYN e", "SYN e", "SYN e"]]), 3)
    assert facts["requested_n"] == 3 and facts["rows"] == 3
    assert (facts["min_candidates"], facts["max_candidates"]) == (1, 3)
    assert facts["utterances_all_distinct"] == 1 and facts["duplicate_texts"] == 3
    assert facts["actual_n_mean"] == pytest.approx(5 / 3) and facts["actual_n_min"] == 1
    assert facts["rows_with_dedup_map"] == 2
    assert facts["validate_nbest_ok"] is True and facts["validation_errors"] == 0
    assert (facts["acoustic_log_score_min"], facts["acoustic_log_score_max"]) == (-3.0, -1.0)
    assert facts["acoustic_log_scores_finite"] is True


def test_nbest_facts_report_an_invalid_row(smoke):
    rows = _nbest_rows([["SYN a", "SYN b"]])
    broken = [{**rows[0], "actual_n": 5}]
    facts = smoke.nbest_facts(broken, 3)
    assert facts["validate_nbest_ok"] is False and facts["validation_errors"] >= 1
    assert 1 <= len(facts["first_errors"]) <= 3 and "actual_n" in facts["first_errors"][0]
    assert facts["actual_n_mean"] == 5.0 and facts["actual_n_min"] == 5


def test_nbest_facts_pass_distinct_lists(smoke):
    facts = smoke.nbest_facts(_nbest_rows([["SYN a", "SYN b"]]), 3)
    assert facts["validate_nbest_ok"] is True and facts["duplicate_texts"] == 0
    assert facts["utterances_all_distinct"] == 1 and facts["rows_with_dedup_map"] == 0
    assert (facts["actual_n_mean"], facts["actual_n_min"]) == (2.0, 2)


def test_nbest_facts_for_systems_without_nbest(smoke):
    assert smoke.nbest_facts([], 0) == {"requested_n": 0}


def test_nbest_facts_count_an_error_row_as_actual_n_zero(smoke):
    rows = _nbest_rows([["SYN a", "SYN b"]])
    error_row = {**rows[0], "utterance_id": "utt_err", "status": "error", "candidates": [], "actual_n": 0}
    facts = smoke.nbest_facts(rows + [error_row], 3)
    assert (facts["actual_n_mean"], facts["actual_n_min"]) == (1.0, 0)


def _inference_record(nbest):
    return {"name": "whisper-small", "kind": "inference", "ok": True, "config": {"precision": "float16"},
            "nbest": nbest}


def _row(report, name):
    return next(line for line in report.splitlines() if line.startswith(f"| {name} |"))


def test_report_shows_mean_and_minimum_actual_n_and_rows_with_a_dedup_map(smoke):
    facts = smoke.nbest_facts(_nbest_rows([["SYN a", "SYN b", "SYN c"], ["SYN d", "SYN d"],
                                           ["SYN e", "SYN e", "SYN e"]]), 3)
    row = _row(smoke.render_report({"whisper-small": _inference_record(facts)}), "whisper-small")
    assert "actual_n mean 1.67, min 1" in row and "2/3 rows with dedup_map" in row


def test_report_renders_a_result_file_written_before_the_export_deduplicated(smoke):
    old = {"requested_n": 10, "rows": 20, "min_candidates": 10, "max_candidates": 10, "utterances_all_distinct": 8,
           "duplicate_texts": 29, "validate_nbest_ok": False, "validation_errors": 24,
           "acoustic_log_scores_finite": True}
    row = _row(smoke.render_report({"whisper-small": _inference_record(old)}), "whisper-small")
    assert "10-10 of 10" in row and "8/20 distinct" in row and "invalid (24 errors)" in row
    assert "actual_n" not in row and "dedup_map" not in row


def test_report_shows_a_dash_when_no_row_has_an_actual_n(smoke):
    facts = smoke.nbest_facts([{**_nbest_rows([["SYN a"]])[0], "actual_n": None}], 3)
    row = _row(smoke.render_report({"whisper-small": _inference_record(facts)}), "whisper-small")
    assert "actual_n mean -, min -," in row and "None" not in row


def test_report_shows_an_actual_n_minimum_of_zero(smoke):
    rows = _nbest_rows([["SYN a", "SYN b"]])
    error_row = {**rows[0], "utterance_id": "utt_err", "status": "error", "candidates": [], "actual_n": 0}
    facts = smoke.nbest_facts(rows + [error_row], 3)
    row = _row(smoke.render_report({"whisper-small": _inference_record(facts)}), "whisper-small")
    assert "actual_n mean 1.00, min 0," in row
