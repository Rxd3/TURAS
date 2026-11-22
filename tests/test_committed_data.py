"""Check the committed raw/cleaned dataset against the frozen release (standard library only)."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "turas-tr-econ-legis-v0.1"
CLEANED = ROOT / "data/cleaned" / RELEASE
SPLITS = ROOT / "data/splits" / RELEASE
LEDGER = CLEANED / "exclusion_ledger.jsonl.gz"

pytestmark = pytest.mark.skipif(not LEDGER.exists(), reason="committed dataset not checked out")


def manifest() -> dict:
    return json.loads((SPLITS / "split_manifest.json").read_text(encoding="utf-8"))


def ledger_rows() -> list[dict]:
    with gzip.open(LEDGER, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_ids(split: str) -> list[str]:
    return (SPLITS / f"{split}.txt").read_text(encoding="utf-8").split()


def test_removal_log_is_the_frozen_ledger():
    data = gzip.decompress(LEDGER.read_bytes())
    assert hashlib.sha256(data).hexdigest() == manifest()["exclusion_ledger_sha256"]


def test_removal_log_counts_match_the_manifest():
    rows = ledger_rows()
    counts = collections.Counter(r["exclusion_reason"] or "selected" for r in rows)
    frozen = {k: v for k, v in manifest()["exclusion_counts"].items() if v}
    assert dict(counts) == frozen
    assert len(rows) == 239_654
    kept = collections.Counter(r["turas_split"] for r in rows if not r["exclusion_reason"])
    assert kept == {s: len(split_ids(s)) for s in ("train", "validation", "test")}


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_references_cover_exactly_the_split(split):
    path = CLEANED / f"{RELEASE}.references.{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["utterance_id"] for r in rows) == sorted(split_ids(split))
    assert all(r["split"] == split and r["reference_text"].strip() for r in rows)
    assert all((CLEANED / r["audio_uri"]).exists() for r in rows)


def test_every_split_clip_has_audio_and_a_checksum():
    sums = {line.split()[1].lstrip("*").removeprefix("./") for line in
            (CLEANED / "SHA256SUMS").read_text(encoding="utf-8").splitlines()}
    for split in ("train", "validation", "test"):
        for uid in split_ids(split):
            assert f"audio/{uid}.flac" in sums
            assert (CLEANED / "audio" / f"{uid}.flac").exists()


def test_validation_references_are_the_reported_variant():
    path = CLEANED / f"{RELEASE}.references.validation.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "d0c8519ca6aed053f89786bc733541a70faee09cf8d8bdb6cc12065d97dbd958"


def test_no_test_sentence_appears_in_train():
    def texts(split):
        path = CLEANED / f"{RELEASE}.references.{split}.jsonl"
        return {" ".join(json.loads(line)["reference_text"].split()) for line in
                path.read_text(encoding="utf-8").splitlines()}
    assert not texts("test") & texts("train")
