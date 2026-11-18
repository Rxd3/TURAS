"""Typed relational CSV tables for RDL (contract docs/contracts/rdl.md 0.1).

Static tables (topics, entities, aliases, relation types, entity->topic edges)
come from the lexicon and are ``external_knowledge``. Utterance evidence
(mentions, quantities, topics, co-mentions) is extracted automatically from
references of ONE split per call, labelled with that split's ``origin_kind``
and ``access_split``, and ``annotation_status = auto_extracted``. Only the
train call may feed graph training; validation/test tables are evaluator-only.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path

import pandas as pd

from .screening import Lexicon, screening_lower, screening_tokens

RELATION_TYPES = [
    # id, name, head_type, tail_type, directed
    ("rel_has_topic", "has_topic", "utterance", "topic", "true"),
    ("rel_mentions", "mentions", "utterance", "entity", "true"),
    ("rel_expresses", "expresses", "utterance", "quantifier", "true"),
    ("rel_belongs_to", "belongs_to", "entity", "topic", "true"),
    ("rel_co_mentioned", "related_to:co_mentioned_in_utterance", "entity", "entity", "false"),
]
ORIGIN = {"train": "train_annotation", "validation": "validation_annotation", "test": "test_annotation"}
PROV_LEXICON = "prov_lexicon"

_SCALE = {"bin": Decimal(10) ** 3, "milyon": Decimal(10) ** 6, "milyar": Decimal(10) ** 9, "trilyon": Decimal(10) ** 12}
_NUM = r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?"
_AMOUNT = re.compile(_NUM + r"\s*(bin|milyon|milyar|trilyon)?\s+(türk lirası|lira\w*|tl|dolar\w*|avro\w*|euro\w*)")
_PERCENT = re.compile(r"(?:yüzde\s*" + _NUM + r"|%\s?" + _NUM + ")")


def _decimal(int_part: str, frac: str | None) -> Decimal | None:
    try:
        return Decimal(int_part.replace(".", "") + ("." + frac if frac else ""))
    except InvalidOperation:
        return None


def _unit(word: str) -> str:
    if word.startswith(("lira", "tl", "türk")):
        return "TRY"
    if word.startswith("dolar"):
        return "USD"
    return "EUR"


def extract_quantities(text: str) -> list[dict]:
    """Amounts with currency and percentages, numeric value as a decimal string."""
    low = screening_lower(text)
    out = []
    for m in _AMOUNT.finditer(low):
        v = _decimal(m.group(1), m.group(2))
        if v is None:
            continue
        if m.group(3):
            v *= _SCALE[m.group(3)]
        out.append({"surface_form": m.group(0), "quantifier_type": "amount",
                    "numeric_value": format(v.normalize(), "f"), "unit": _unit(m.group(4))})
    for m in _PERCENT.finditer(low):
        ip, fr = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        v = _decimal(ip, fr)
        if v is not None:
            out.append({"surface_form": m.group(0), "quantifier_type": "percentage",
                        "numeric_value": format(v.normalize(), "f"), "unit": "percent"})
    return out


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_static_tables(lexicon: Lexicon, lexicon_sha256: str, out_dir: Path) -> None:
    _write(out_dir / "provenance.csv",
           ["provenance_id", "source_uri", "source_revision", "content_sha256", "rights_record_id",
            "origin_kind", "available_at_utc", "allowed_split", "review_status"],
           [[PROV_LEXICON, "configs/datasets/turas-domain-lexicon-v0.1.json", lexicon.version,
             lexicon_sha256, "rights_turas_lexicon", "external_knowledge", "2026-11-18T00:00:00Z",
             "train|validation|test", "proposed"]])
    _write(out_dir / "topics.csv", ["topic_id", "canonical_label", "parent_topic_id", "provenance_id"],
           [[t, v["label_en"], "", PROV_LEXICON] for t, v in sorted(lexicon.topics.items())])
    _write(out_dir / "entities.csv", ["entity_id", "canonical_label", "entity_type", "provenance_id"],
           [[t, v["canonical"], v.get("entity_type", "concept"), PROV_LEXICON]
            for t, v in sorted(lexicon.terms.items())])
    aliases = []
    for t, v in sorted(lexicon.terms.items()):
        for i, a in enumerate(v["aliases"]):
            aliases.append([f"{t}__a{i}", t, a, "screening-lower-v0.1", PROV_LEXICON])
    _write(out_dir / "entity_aliases.csv",
           ["alias_id", "entity_id", "alias_text", "normalization_version", "provenance_id"], aliases)
    _write(out_dir / "relation_types.csv",
           ["relation_type_id", "name", "head_type", "tail_type", "directed", "provenance_id"],
           [[*r, PROV_LEXICON] for r in RELATION_TYPES])
    _write(out_dir / "relationships_static.csv", _REL_HEADER,
           [[f"belongs__{t}", "entity", t, "rel_belongs_to", "topic", v["topic_id"], PROV_LEXICON, "",
             "lexicon_defined", "train|validation|test"] for t, v in sorted(lexicon.terms.items())])


_REL_HEADER = ["relationship_id", "head_type", "head_id", "relation_type_id", "tail_type", "tail_id",
               "evidence_provenance_id", "evidence_utterance_id", "annotation_status", "access_split"]


def write_split_tables(refs: pd.DataFrame, split: str, lexicon: Lexicon, release: str,
                       references_sha256: str, out_dir: Path) -> dict[str, int]:
    """Utterance evidence for one split. ``refs`` needs utterance_id, reference_text,
    leakage_group_id and split columns; rows of other splits are rejected."""
    if set(refs["split"]) - {split}:
        raise ValueError("write_split_tables received rows of another split")
    prov = f"prov_{split}_references"
    _write(out_dir / "provenance.csv",
           ["provenance_id", "source_uri", "source_revision", "content_sha256", "rights_record_id",
            "origin_kind", "available_at_utc", "allowed_split", "review_status"],
           [[prov, f"{release}.references.jsonl#split={split}", release, references_sha256,
             "rights_worldspeech_tr_tr_tbmm", ORIGIN[split], "2026-11-18T00:00:00Z", split, "auto_extracted"]])
    utts, rels, quants = [], [], {}
    co: Counter = Counter()
    co_evidence: dict[tuple[str, str], str] = {}
    for r in refs.sort_values("utterance_id").itertuples(index=False):
        uid = r.utterance_id
        utts.append([uid, release, split, r.leakage_group_id, prov])
        ents = sorted({m.term_id for m in lexicon.match(screening_tokens(r.reference_text))})
        for e in ents:
            rels.append([f"m__{uid}__{e}", "utterance", uid, "rel_mentions", "entity", e, prov, uid,
                         "auto_extracted", split])
        for t in sorted({lexicon.topic_of(e) for e in ents}):
            rels.append([f"t__{uid}__{t}", "utterance", uid, "rel_has_topic", "topic", t, prov, uid,
                         "auto_extracted", split])
        for k, q in enumerate(extract_quantities(r.reference_text)):
            qid = f"q__{q['quantifier_type']}__{q['unit']}__{q['numeric_value']}"
            quants[qid] = [qid, q["surface_form"], q["quantifier_type"], q["numeric_value"], q["unit"], prov]
            rels.append([f"x__{uid}__{k}", "utterance", uid, "rel_expresses", "quantifier", qid, prov, uid,
                         "auto_extracted", split])
        for a, b in combinations(ents, 2):
            co[(a, b)] += 1
            co_evidence.setdefault((a, b), uid)
    for (a, b), n in sorted(co.items()):
        rels.append([f"c__{a}__{b}", "entity", a, "rel_co_mentioned", "entity", b, prov,
                     co_evidence[(a, b)], f"auto_extracted:count={n}", split])
    _write(out_dir / "utterances.csv",
           ["utterance_id", "dataset_release", "split", "leakage_group_id", "provenance_id"], utts)
    _write(out_dir / "quantifiers.csv",
           ["quantifier_id", "surface_form", "quantifier_type", "numeric_value", "unit", "provenance_id"],
           sorted(quants.values()))
    _write(out_dir / "relationships.csv", _REL_HEADER, rels)
    return {"utterances": len(utts), "quantifiers": len(quants), "relationships": len(rels)}
