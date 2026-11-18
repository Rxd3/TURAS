"""Offline tests for relational table extraction (synthetic text fixtures)."""

import csv
from pathlib import Path

import pandas as pd
import pytest

from turkish_asr_rdl.data.relational import extract_quantities, write_split_tables, write_static_tables
from turkish_asr_rdl.data.screening import Lexicon

REPO = Path(__file__).resolve().parents[1]
LEX = Lexicon.load(REPO / "configs/datasets/turas-domain-lexicon-v0.1.json")


def test_quantities_decimal_strings():
    q = extract_quantities("Bütçe 1.500 milyar lira, faiz yüzde 42,5 ve %3; 100 dolar")
    got = {(x["quantifier_type"], x["numeric_value"], x["unit"]) for x in q}
    assert got == {
        ("amount", "1500000000000", "TRY"),
        ("percentage", "42.5", "percent"),
        ("percentage", "3", "percent"),
        ("amount", "100", "USD"),
    }


def _read(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_split_tables_are_split_labelled(tmp_path):
    refs = pd.DataFrame({
        "utterance_id": ["utt_a", "utt_b"],
        "reference_text": ["Enflasyon ve faiz 5 milyar lira", "vergi kanunu"],
        "leakage_group_id": ["src_1", "src_2"],
        "split": ["train", "train"],
    })
    counts = write_split_tables(refs, "train", LEX, "rel-x", "0" * 64, tmp_path)
    rels = _read(tmp_path / "relationships.csv")
    assert counts["utterances"] == 2 and all(r["access_split"] == "train" for r in rels)
    assert {r["tail_id"] for r in rels if r["head_id"] == "utt_a" and r["relation_type_id"] == "rel_mentions"} == {
        "MON_ENFLASYON", "MON_FAIZ", "MON_TURK_LIRASI"}
    assert _read(tmp_path / "provenance.csv")[0]["origin_kind"] == "train_annotation"


def test_split_tables_refuse_mixed_splits(tmp_path):
    refs = pd.DataFrame({"utterance_id": ["a", "b"], "reference_text": ["x", "y"],
                         "leakage_group_id": ["g", "h"], "split": ["train", "test"]})
    with pytest.raises(ValueError):
        write_split_tables(refs, "train", LEX, "rel-x", "0" * 64, tmp_path)


def test_static_tables_foreign_keys(tmp_path):
    write_static_tables(LEX, "0" * 64, tmp_path)
    ents = {r["entity_id"] for r in _read(tmp_path / "entities.csv")}
    topics = {r["topic_id"] for r in _read(tmp_path / "topics.csv")}
    for r in _read(tmp_path / "relationships_static.csv"):
        assert r["head_id"] in ents and r["tail_id"] in topics
    assert {r["entity_id"] for r in _read(tmp_path / "entity_aliases.csv")} == ents


@pytest.mark.parametrize("text,expected", [
    ("5 milyon 300 bin lira", ("amount", "5300000", "TRY")),
    ("5 bin 500 lira", ("amount", "5500", "TRY")),
    ("1.5 milyar lira", ("amount", "1500000000", "TRY")),
    ("2,5 milyar TL", ("amount", "2500000000", "TRY")),
    ("1.250,50 TL", ("amount", "1250.5", "TRY")),
    ("12.345.678 lira", ("amount", "12345678", "TRY")),
    ("yüzde 3.5", ("percentage", "3.5", "percent")),
    ("enflasyon 3% oldu", ("percentage", "3", "percent")),
    ("faiz %50,5", ("percentage", "50.5", "percent")),
])
def test_compound_and_decimal_quantities(text, expected):
    assert [(q["quantifier_type"], q["numeric_value"], q["unit"]) for q in extract_quantities(text)] == [expected]


def test_bare_scale_word_gets_no_value():
    assert extract_quantities("bin lira verdi") == []
