"""Offline tests for M1 screening, lexicon matching and grouped selection.

All transcript strings below are synthetic fixtures written for these tests;
they are not WorldSpeech or TURAS transcripts.
"""

import json
from pathlib import Path

import pytest

from turkish_asr_rdl.data.screening import (
    Lexicon,
    count_quantities,
    screening_lower,
    screening_tokens,
    transcript_issues,
)

REPO = Path(__file__).resolve().parents[1]
LEXICON = REPO / "configs/datasets/turas-domain-lexicon-v0.1.json"


@pytest.fixture(scope="module")
def lex():
    return Lexicon.load(LEXICON)


def test_turkish_dotted_and_dotless_i():
    assert screening_lower("IŞIK İZMİR") == "ışık izmir"
    assert screening_lower("İ") == "i"  # decomposed dotted capital I


def test_apostrophe_splits_suffix():
    assert screening_tokens("Merkez Bankası'nın faizi") == ["merkez", "bankası", "nın", "faizi"]


@pytest.mark.parametrize(
    "text, codes",
    [
        ("Bütçe görüşmelerine geçiyoruz.", []),
        ("BAŞKAN - Sayın milletvekilleri, bütçe", ["speaker_header"]),
        ("ALİ VELİ (Ankara) – Teşekkürler.", ["parenthetical", "speaker_header"]),
        ("vergi oranları (Alkışlar)", ["parenthetical"]),
        ("ve bu nedenle...", ["trailing_ellipsis"]),
        ("   ", ["empty"]),
    ],
)
def test_transcript_issues(text, codes):
    assert transcript_issues(text) == codes


def test_lexicon_aliases_are_valid(lex):
    data = json.loads(LEXICON.read_text(encoding="utf-8"))
    assert len(lex.terms) == len(data["terms"])
    assert {t["topic_id"] for t in data["terms"]} <= set(lex.topics)


def test_longest_span_wins(lex):
    toks = screening_tokens("Hazine ve Maliye Bakanlığı bütçe açığını açıkladı")
    got = [m.term_id for m in lex.match(toks)]
    assert got == ["PUBFIN_HAZINE_MALIYE_BAKANLIGI", "PUBFIN_BUTCE_ACIGI"]


def test_suffix_prefix_match_and_exclusions(lex):
    def ids(text):
        return [m.term_id for m in lex.match(screening_tokens(text))]

    assert ids("enflasyonun ve faizlerin") == ["MON_ENFLASYON", "MON_FAIZ"]
    assert ids("zaman yasak ücretsiz partiden ihraç") == []  # ambiguous forms excluded
    assert ids("zamlar yasalar ücretler") == ["MON_ZAM", "LEGIS_YASA", "LAB_UCRET"]


def test_quantities(lex):
    text = "5 milyar lira ve yüzde 40 ile %3"
    toks = screening_tokens(text)
    q = count_quantities(text, toks, lex.match(toks), lex)
    assert q == {"amount": 1, "percentage": 2}
