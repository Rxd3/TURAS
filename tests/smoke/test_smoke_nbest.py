"""N-best facts of the 5070 Ti smoke record, read from rows the shared export already deduplicated. SYNTHETIC."""
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
    assert facts["validate_nbest_ok"] is True and facts["validation_errors"] == 0
    assert (facts["acoustic_log_score_min"], facts["acoustic_log_score_max"]) == (-3.0, -1.0)
    assert facts["acoustic_log_scores_finite"] is True


def test_nbest_facts_report_an_invalid_row(smoke):
    rows = _nbest_rows([["SYN a", "SYN b"]])
    broken = [{**rows[0], "actual_n": 5}]
    facts = smoke.nbest_facts(broken, 3)
    assert facts["validate_nbest_ok"] is False and facts["validation_errors"] >= 1
    assert 1 <= len(facts["first_errors"]) <= 3 and "actual_n" in facts["first_errors"][0]


def test_nbest_facts_pass_distinct_lists(smoke):
    facts = smoke.nbest_facts(_nbest_rows([["SYN a", "SYN b"]]), 3)
    assert facts["validate_nbest_ok"] is True and facts["duplicate_texts"] == 0
    assert facts["utterances_all_distinct"] == 1


def test_nbest_facts_for_systems_without_nbest(smoke):
    assert smoke.nbest_facts([], 0) == {"requested_n": 0}
