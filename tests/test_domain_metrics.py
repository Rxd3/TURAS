"""Synthetic term/quantity cases, never real corpus predictions."""
from types import SimpleNamespace

from turkish_asr_rdl.data.screening import Lexicon
from turkish_asr_rdl.evaluation.domain import evaluate_domain


def lexicon():
    return Lexicon({'lexicon_version': 'synthetic-only', 'topics': [{'topic_id': 't'}], 'terms': [
        {'term_id': 'bank', 'topic_id': 't', 'entity_type': 'concept', 'aliases': ['banka*']},
        {'term_id': 'salary', 'topic_id': 't', 'entity_type': 'concept', 'aliases': ['\u00fccret*'], 'exclude': ['\u00fccretsiz']},
        {'term_id': 'central', 'topic_id': 't', 'entity_type': 'organization', 'aliases': ['merkez banka*']}],
        'quantity_units': {}})


def test_occurrences_are_counted_per_row_and_unsupported_fp_are_visible():
    rows = [SimpleNamespace(utterance_id='fixture-a', reference='bankan\u0131n banka', hypothesis='banka \u00fccretler'),
            SimpleNamespace(utterance_id='fixture-b', reference='merkez bankas\u0131', hypothesis='merkez bankas\u0131')]
    r = evaluate_domain(rows, lexicon())
    assert r['micro']['tp'] == 2 and r['micro']['fn'] == 1 and r['micro']['fp'] == 1
    assert r['per_term']['bank']['reference_support'] == 2
    assert r['per_term']['central']['tp'] == 1
    assert r['unsupported_term_false_positives'] == 1
    assert r['macro_reference_supported']['reference_supported_terms'] == 2
    assert evaluate_domain([SimpleNamespace(utterance_id='empty',reference='\u00fccretsiz',hypothesis='')],lexicon())['micro']['reference_support'] == 0


def test_quantity_units_and_repeat_counts_are_not_merged_across_utterances():
    rows = [SimpleNamespace(utterance_id='one', reference='5 milyon lira ve 5 milyon lira', hypothesis='5000000 lira'),
            SimpleNamespace(utterance_id='two', reference='y\u00fczde 42', hypothesis='%42'),
            SimpleNamespace(utterance_id='three', reference='100 dolar', hypothesis='100 lira')]
    q = evaluate_domain(rows, lexicon())['quantity_diagnostics']['micro']
    assert (q['tp'], q['fp'], q['fn']) == (2, 1, 2)
