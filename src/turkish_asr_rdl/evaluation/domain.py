"""Frozen-lexicon occurrence metrics and separate automatic quantity diagnostics."""
from collections import Counter

from turkish_asr_rdl.data.screening import Lexicon, screening_tokens
from turkish_asr_rdl.data.relational import extract_quantities

VERSION = 'canonical-occurrence-m1-v1'


def _rates(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'reference_support': tp + fn,
            'predicted_support': tp + fp, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def _add(stats, reference, hypothesis):
    for key in reference.keys() | hypothesis.keys():
        counts = stats.setdefault(key, [0, 0, 0])
        matched = min(reference[key], hypothesis[key])
        counts[0] += matched
        counts[1] += hypothesis[key] - matched
        counts[2] += reference[key] - matched


def _total(values):
    return _rates(*(sum(v[i] for v in values) for i in range(3)))


def evaluate_domain(joined, lexicon: Lexicon):
    """Count canonical occurrences within each utterance; never match across rows."""
    terms = {term: [0, 0, 0] for term in lexicon.terms}
    quantities = {}
    containing = []
    for row in joined:
        reference = Counter(m.term_id for m in lexicon.match(screening_tokens(row.reference)))
        hypothesis = Counter(m.term_id for m in lexicon.match(screening_tokens(row.hypothesis)))
        _add(terms, reference, hypothesis)
        if reference:
            containing.append(row.utterance_id)
        def amounts(text):
            return Counter((q['quantifier_type'], q['unit'], q['numeric_value']) for q in extract_quantities(text))
        _add(quantities, amounts(row.reference), amounts(row.hypothesis))
    per_term = {key: _rates(*value) for key, value in sorted(terms.items())}
    supported = [v for v in per_term.values() if v['reference_support']]
    macro = {key: sum(v[key] for v in supported) / len(supported) if supported else 0.0
             for key in ['precision', 'recall', 'f1']}
    macro['reference_supported_terms'] = len(supported)
    topics = {topic: _total([value for term, value in terms.items() if lexicon.topic_of(term) == topic])
              for topic in sorted(lexicon.topics)}
    types = sorted({t.get('entity_type', 'concept') for t in lexicon.terms.values()})
    entities = {kind: _total([value for term, value in terms.items()
                            if lexicon.terms[term].get('entity_type', 'concept') == kind]) for kind in types}
    return {'version': VERSION, 'lexicon_version': lexicon.version,
        'matching': 'frozen M1 raw-text screening_tokens + longest nonoverlapping alias matches; canonical-ID ties, prefixes and exclusions',
        'micro': _total(list(terms.values())), 'macro_reference_supported': macro,
        'per_term': per_term, 'by_topic': topics, 'by_entity_type': entities,
        'unsupported_term_false_positives': sum(v['fp'] for v in per_term.values() if not v['reference_support']),
        'term_containing_utterance_ids': containing,
        'quantity_diagnostics': {'micro': _total(list(quantities.values())),
            'by_type': {kind: _total([v for key, v in quantities.items() if key[0] == kind]) for kind in ['amount', 'percentage']},
            'matching': 'per-utterance exact (type, currency/unit, canonical numeric value) occurrence multisets',
            'limitations': 'automatic digit-based amount/percentage extraction; no spoken-number conversion, counts or dates; not human annotation'}}
