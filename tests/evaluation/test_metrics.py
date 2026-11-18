"""SYNTHETIC: every string here is a made-up example, never a dataset transcript."""

import math

import pytest

from turkish_asr_rdl.evaluation.align import EditCounts
from turkish_asr_rdl.evaluation.metrics import (
    CorpusRates,
    UtteranceScore,
    corpus_rates,
    rtf,
    rtfx,
    score_pair,
)
from turkish_asr_rdl.evaluation.normalize import RAW_V1, TR_NORMALIZED_V1

NINE = "a b c d e f g h i"  # SYNTHETIC


def test_corpus_wer_is_sum_over_sum():
    scores = [
        score_pair("u1", "a", "b", TR_NORMALIZED_V1),
        score_pair("u2", NINE, NINE, TR_NORMALIZED_V1),
    ]
    rates = corpus_rates(scores)
    assert rates.wer == pytest.approx(0.1)
    assert rates.words == EditCounts(1, 0, 0, 9)
    assert rates.utterances == 2


def test_zero_reference_counted_and_rate_none_when_total_zero():
    rates = corpus_rates([score_pair("u1", "", "x", TR_NORMALIZED_V1)])
    assert rates.zero_reference_utterances == 1
    assert rates.words.insertions == 1
    assert (rates.wer, rates.cer, rates.wacc) == (None, None, None)


def test_zero_reference_adds_insertions_but_no_denominator():
    scores = [
        score_pair("u1", "", "x", TR_NORMALIZED_V1),
        score_pair("u2", "a", "a", TR_NORMALIZED_V1),
    ]
    rates = corpus_rates(scores)
    assert rates.zero_reference_utterances == 1
    assert rates.wer == pytest.approx(1.0)


def test_raw_cer_counts_trailing_space():
    assert score_pair("u", "ab", "ab ", RAW_V1).chars.insertions == 1


def test_normalized_view_ignores_case_and_punct():
    assert score_pair("u", "İZMİR.", "izmir", TR_NORMALIZED_V1).words.errors == 0


def test_raw_view_counts_case_and_punct():
    score = score_pair("u", "İZMİR.", "izmir", RAW_V1)
    assert score.words == EditCounts(1, 0, 0, 0)
    assert score.chars.ref_len == 6


def test_score_pair_keeps_id_view_and_status():
    score = score_pair("u7", "a", "", TR_NORMALIZED_V1, status="error")
    assert score == UtteranceScore(
        "u7", TR_NORMALIZED_V1, "error", EditCounts(0, 1, 0, 0), EditCounts(0, 1, 0, 0)
    )


def test_score_pair_unknown_view_raises():
    with pytest.raises(ValueError, match="unknown normalization version"):
        score_pair("u", "a", "a", "lowercase-v1")


def test_wacc_is_one_minus_wer():
    rates = corpus_rates([score_pair("u", "a b c d", "a b c x", TR_NORMALIZED_V1)])
    assert rates.wer == pytest.approx(0.25)
    assert rates.wacc == pytest.approx(0.75)


def test_rates_may_exceed_one_and_wacc_goes_negative():
    rates = corpus_rates([score_pair("u", "a", "b c d", TR_NORMALIZED_V1)])
    assert rates.wer == pytest.approx(3.0)
    assert rates.wacc == pytest.approx(-2.0)


def test_cer_is_sum_over_sum_of_code_points():
    scores = [
        score_pair("u1", "ab", "ax", TR_NORMALIZED_V1),
        score_pair("u2", "abcdefgh", "abcdefgh", TR_NORMALIZED_V1),
    ]
    assert corpus_rates(scores).cer == pytest.approx(0.1)


def test_error_rows_are_counted():
    scores = [
        score_pair("u1", "a", "", TR_NORMALIZED_V1, status="error"),
        score_pair("u2", "a", "a", TR_NORMALIZED_V1),
    ]
    rates = corpus_rates(scores)
    assert rates.error_rows == 1
    assert rates.wer == pytest.approx(0.5)


def test_corpus_rates_reports_view():
    rates = corpus_rates([score_pair("u", "a", "a", RAW_V1)])
    assert isinstance(rates, CorpusRates)
    assert rates.view == RAW_V1


def test_corpus_rates_rejects_empty_input():
    with pytest.raises(ValueError, match="no utterance scores"):
        corpus_rates([])


def test_corpus_rates_rejects_mixed_views():
    scores = [score_pair("u1", "a", "a", RAW_V1), score_pair("u2", "a", "a", TR_NORMALIZED_V1)]
    with pytest.raises(ValueError, match="one normalization view"):
        corpus_rates(scores)


def test_rtf_and_rtfx():
    assert rtf(5.0, 10.0) == pytest.approx(0.5)
    assert rtfx(5.0, 10.0) == pytest.approx(2.0)
    assert rtf(0.0, 10.0) == 0.0


@pytest.mark.parametrize("audio_s", [0.0, -1.0, math.nan, math.inf])
def test_rtf_rejects_non_positive_or_non_finite_audio(audio_s):
    with pytest.raises(ValueError, match="audio_s"):
        rtf(1.0, audio_s)
    with pytest.raises(ValueError, match="audio_s"):
        rtfx(1.0, audio_s)


@pytest.mark.parametrize("processing_s", [-1.0, math.nan, math.inf])
def test_rtf_rejects_negative_or_non_finite_processing(processing_s):
    with pytest.raises(ValueError, match="processing_s"):
        rtf(processing_s, 10.0)


def test_rtfx_rejects_zero_processing_time():
    with pytest.raises(ValueError, match="processing_s"):
        rtfx(0.0, 10.0)
