"""SYNTHETIC: token lists here are made-up symbols, never dataset transcripts."""

import random

import jiwer
import pytest

from turkish_asr_rdl.evaluation.align import EditCounts, align_counts


def test_golden_swap_is_two_substitutions():
    assert align_counts(["a", "b"], ["b", "a"]) == EditCounts(2, 0, 0, 0)


def test_empty_hypothesis_is_all_deletions():
    assert align_counts(["a", "b"], []) == EditCounts(0, 2, 0, 0)


def test_empty_reference_is_all_insertions():
    assert align_counts([], ["a"]) == EditCounts(0, 0, 1, 0)


def test_both_empty_is_zero():
    assert align_counts([], []) == EditCounts()


def test_hand_checked():
    assert align_counts(list("kitten"), list("sitting")) == EditCounts(2, 0, 1, 4)


def test_tie_prefers_deletion_over_insertion():
    # SYNTHETIC: two optimal alignments cost 3; an insertion-first backtrace
    # gives EditCounts(2, 0, 1, 1), the deletion-first rule gives 0 S, 1 D, 2 I.
    assert align_counts(["a", "b", "a"], ["b", "c", "a", "b"]) == EditCounts(
        substitutions=0, deletions=1, insertions=2, hits=2
    )


def test_accepts_code_point_strings():
    assert align_counts("ab", "ab ") == EditCounts(0, 0, 1, 2)


def test_properties_and_addition():
    left = EditCounts(1, 2, 3, 4)
    total = left + EditCounts(10, 20, 30, 40)
    assert total == EditCounts(11, 22, 33, 44)
    assert left.ref_len == 7
    assert left.errors == 6
    assert left == EditCounts(1, 2, 3, 4)


def test_adding_a_non_count_raises_type_error():
    with pytest.raises(TypeError):
        EditCounts() + 1  # type: ignore[operator]


def test_edit_counts_is_frozen():
    with pytest.raises(AttributeError):
        EditCounts().hits = 1  # type: ignore[misc]


def _random_tokens(rng: random.Random) -> list[str]:
    return [rng.choice("abcd") for _ in range(rng.randint(0, 8))]


def test_totals_match_jiwer_on_random_pairs():
    rng = random.Random(2026)
    for _ in range(300):
        ref, hyp = _random_tokens(rng), _random_tokens(rng)
        ours = align_counts(ref, hyp)
        theirs = jiwer.process_words(" ".join(ref), " ".join(hyp))
        expected = theirs.substitutions + theirs.deletions + theirs.insertions
        assert ours.errors == expected, (ref, hyp)
        assert ours.ref_len == len(ref)
