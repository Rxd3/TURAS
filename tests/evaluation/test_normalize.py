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
    ("İ", "i"),
    ("i̇", "i̇"),
    ("Ankara'da", "ankarada"),
    ("Ankara’da", "ankarada"),
    ("Ankaraʼda", "ankarada"),
    ("2026'da", "2026 da"),
    ("1.250,50", "1 250 50"),
    ("%5", "%5"),
    ("₺100", "₺100"),
    ("a+b-c/d", "a+b c d"),
    ("a​b", "ab"),
    ("a   b", "a b"),
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


@pytest.mark.xfail(
    strict=True,
    reason="the six contract steps end without a final NFC, so deleting a Cf or an "
    "apostrophe can leave a pair that NFC composes on a second pass",
)
@pytest.mark.parametrize("src", ["e​́", "ᄀ'ᅡ"])
def test_idempotent_when_deletion_leaves_a_composable_pair(src):
    # SYNTHETIC: base letter + ZWSP + combining acute; Hangul L + apostrophe + V jamo.
    once = normalize(src, TR_NORMALIZED_V1)
    assert normalize(once, TR_NORMALIZED_V1) == once


def test_apostrophe_neighbours_come_from_the_text_after_cf_deletion():
    # SYNTHETIC: U+200B next to the apostrophe is gone before step 4 looks.
    assert normalize("a​'b", TR_NORMALIZED_V1) == "ab"


def test_doubled_apostrophe_becomes_space():
    # SYNTHETIC: each apostrophe has a non-letter neighbour in the original run.
    assert normalize("a''b", TR_NORMALIZED_V1) == "a b"


def test_apostrophe_at_edges_becomes_space():
    assert normalize("'a'", TR_NORMALIZED_V1) == "a"


def test_whitespace_includes_information_separators():
    # SYNTHETIC: U+001C..U+001F count as whitespace under str.isspace().
    assert normalize("a\x1cb\x1fc", TR_NORMALIZED_V1) == "a b c"


def test_bom_inside_text_is_deleted_as_format_character():
    assert normalize("﻿ab", TR_NORMALIZED_V1) == "ab"


def test_raw_is_identity():
    assert normalize(" Aİ, b ", RAW_V1) == " Aİ, b "


def test_unknown_version_raises():
    with pytest.raises(ValueError, match="unknown normalization version"):
        normalize("a", "tr-normalized-v9")


def test_word_tokens_split_on_unicode_whitespace():
    assert word_tokens(" a b\tc ") == ["a", "b", "c"]
    assert word_tokens("") == []
