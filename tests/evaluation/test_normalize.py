"""SYNTHETIC: every string here is a contract rule example, never a dataset transcript."""

import pytest

from turkish_asr_rdl.evaluation.normalize import (
    RAW_V1,
    TR_NORMALIZED_V1,
    normalize,
    word_tokens,
)

# SYNTHETIC rule examples (contract 0.2), not transcripts.
CASES = [
    ("IŞIK", "ışık"),
    ("İZMİR", "izmir"),
    ("I\u0307", "i"),
    ("i\u0307", "i\u0307"),
    ("Ankara'da", "ankarada"),
    ("Ankara\u2019da", "ankarada"),
    ("Ankara\u02bcda", "ankarada"),
    ("2026'da", "2026 da"),
    ("1.250,50", "1 250 50"),
    ("%5", "%5"),
    ("₺100", "₺100"),
    ("a+b-c/d", "a+b c d"),
    ("a\u200bb", "ab"),
    ("a\u00a0 \u2028b", "a b"),
    ("", ""),
    ("   ", ""),
]


@pytest.mark.parametrize("src,out", CASES)
def test_tr_normalized_v1(src, out):
    assert normalize(src, TR_NORMALIZED_V1) == out


def test_idempotent():
    for src, _ in CASES:
        once = normalize(src, TR_NORMALIZED_V1)
        assert normalize(once, TR_NORMALIZED_V1) == once


@pytest.mark.parametrize("src", ["e\u200b\u0301", "\u1100'\u1161"])
def test_idempotent_when_deletion_leaves_a_composable_pair(src):
    # SYNTHETIC: base letter + ZWSP + combining acute; Hangul L + apostrophe + V jamo.
    once = normalize(src, TR_NORMALIZED_V1)
    assert normalize(once, TR_NORMALIZED_V1) == once


# SYNTHETIC: inputs where a step can leave text that a later pass would change.
TRICKY = [
    "e\u200b\u0301",
    "A\u200b\u0301",
    "\u1100'\u1161",
    "I\u0307",
    "\u0130ZM\u0130R",
    "a'\u02bcb",
    " a\t\u00a0b\u2003\n\u3000c\x1c ",
]


@pytest.mark.parametrize("src", TRICKY)
def test_normalizing_twice_equals_normalizing_once(src):
    once = normalize(src, TR_NORMALIZED_V1)
    assert normalize(once, TR_NORMALIZED_V1) == once


def test_final_step_composes_pairs_left_by_deletion():
    assert normalize("e\u200b\u0301", TR_NORMALIZED_V1) == "\u00e9"
    assert normalize("\u1100'\u1161", TR_NORMALIZED_V1) == "\uac00"


def test_compatibility_characters_are_kept():
    # SYNTHETIC: ligature fi + superscript two; NFKC would give "fi2".
    assert normalize("\ufb01\u00b2", TR_NORMALIZED_V1) == "\ufb01\u00b2"


def test_modifier_letter_apostrophe_next_to_a_digit_becomes_space():
    # SYNTHETIC: U+02BC is category Lm, so only the apostrophe rule turns it into a space.
    assert normalize("2026\u02bcda", TR_NORMALIZED_V1) == "2026 da"


def test_apostrophe_neighbours_come_from_the_text_after_cf_deletion():
    # SYNTHETIC: U+200B next to the apostrophe is gone before step 4 looks.
    assert normalize("a\u200b'b", TR_NORMALIZED_V1) == "ab"


@pytest.mark.parametrize(
    "src",
    [
        "a''b",
        "a\u02bc\u02bcb",
        "a'\u02bcb",
        "a\u02bc'b",
        "a\u02bc\u02bc\u02bcb",
    ],
)
def test_apostrophe_run_between_letters_becomes_space(src):
    # SYNTHETIC: no apostrophe counts as a letter neighbour, U+02BC (Lm) included,
    # so every apostrophe in a run has a non-letter neighbour.
    assert normalize(src, TR_NORMALIZED_V1) == "a b"


def test_other_modifier_letters_still_count_as_letter_neighbours():
    # SYNTHETIC: U+02B0 is category Lm but not an apostrophe.
    assert normalize("a\u02b0'b", TR_NORMALIZED_V1) == "a\u02b0b"


def test_apostrophe_at_edges_becomes_space():
    assert normalize("'a'", TR_NORMALIZED_V1) == "a"


def test_whitespace_includes_information_separators():
    # SYNTHETIC: U+001C..U+001F count as whitespace under str.isspace().
    assert normalize("a\x1cb\x1fc", TR_NORMALIZED_V1) == "a b c"


def test_bom_inside_text_is_deleted_as_format_character():
    assert normalize("\ufeffab", TR_NORMALIZED_V1) == "ab"


def test_raw_is_identity():
    assert normalize(" Aİ, b ", RAW_V1) == " Aİ, b "


def test_unknown_version_raises():
    with pytest.raises(ValueError, match="unknown normalization version"):
        normalize("a", "tr-normalized-v9")


def test_word_tokens_split_on_unicode_whitespace():
    assert word_tokens(" a\u00a0b\tc ") == ["a", "b", "c"]
    assert word_tokens("") == []
