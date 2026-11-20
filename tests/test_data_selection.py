"""Offline tests for recording-grouped selection and release files (synthetic rows)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from turkish_asr_rdl.data import release as R
from turkish_asr_rdl.data.screening import Lexicon
from turkish_asr_rdl.data.selection import leakage_report, screen, select

REPO = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO / "configs/datasets/turas-tr-econ-legis-v0.1.json").read_text(encoding="utf-8"))

ECON = "bütçe ve vergi oranları {i}. madde ile {i} milyar lira"  # synthetic in-domain text


def _rows():
    rows = []
    recs = [("2024-05-01", "train"), ("2024-06-01", "test"), ("2025-10-10", "train"),
            ("2025-11-11", "test"), ("2026-02-02", "train"), ("2026-03-03", "test")]
    k = 0
    for r, (date, up) in enumerate(recs):
        for s in range(6):
            k += 1
            rows.append({
                "human_transcript": ECON.format(i=k), "cer": 0.05, "duration": 15.0,
                "source": "tbmm_parliament", "source_url": f"https://example.invalid/watch?v=r{r}",
                "source_start_s": 15.0 * s, "source_end_s": 15.0 * (s + 1), "session_date": date,
                "segment_id": f"seg_{r}_{s:03d}", "upstream_split": up if s % 3 == 0 else "train",
                "shard": "x.parquet", "row_in_shard": k,
            })
    rows.append({**rows[0], "human_transcript": "BAŞKAN - bütçe vergi", "segment_id": "seg_hdr",
                 "source_start_s": 999.0, "source_end_s": 1014.0, "row_in_shard": 999})
    rows.append({**rows[0], "human_transcript": "Kabul edilmiştir.", "segment_id": "seg_off",
                 "source_start_s": 2000.0, "source_end_s": 2015.0, "row_in_shard": 1000})
    rows.append({**rows[1], "source": "other_source", "segment_id": "seg_src", "row_in_shard": 1001})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def selection():
    cfg = json.loads(json.dumps(CONFIG))
    cfg["splits"]["test"]["target_hours"] = 0.05
    cfg["splits"]["validation"]["target_hours"] = 0.02
    cfg["splits"]["train"]["target_hours"] = 0.1
    lex = Lexicon.load(REPO / CONFIG["lexicon"])
    return select(screen(_rows(), lex), cfg)


def test_every_row_has_split_or_reason(selection):
    t = selection.table
    assert ((t.turas_split != "") ^ (t.exclusion_reason != "")).all()
    reasons = dict(zip(t.segment_id, t.exclusion_reason))
    assert reasons["seg_hdr"] == "non_spoken_or_truncated_text"
    assert reasons["seg_off"] == "not_in_domain"
    assert reasons["seg_src"] == "source_not_selected"


def test_recording_grouping_and_temporal_test(selection):
    t = selection.table
    sel = t[t.turas_split != ""]
    assert sel.groupby("source_url").turas_split.nunique().max() == 1
    assert (sel[sel.turas_split == "test"].session_date >= "2026-01-01").all()
    assert set(sel.turas_split) == {"train", "validation", "test"}
    assert not ((sel.turas_split == "train") & (sel.upstream_split == "test")).any()
    assert all(v == 0 for v in leakage_report(t).values())


def test_upstream_test_rows_in_train_recordings_are_quarantined(selection):
    t = selection.table
    train_recs = set(t[t.turas_split == "train"].source_url)
    q = t[t.source_url.isin(train_recs) & (t.upstream_split == "test")]
    assert len(q) and (q.exclusion_reason == "upstream_test_in_train_recording").all()


def test_ids_are_stable_and_text_independent():
    a = R.utterance_id("u", "seg", 0, 1000)
    assert a == R.utterance_id("u", "seg", 0, 1000) and a.startswith("utt_") and len(a) == 20
    assert a != R.utterance_id("u", "seg", 0, 1001)


def test_split_files_sorted_disjoint(tmp_path, selection):
    chosen = R.assign_ids(selection.table[selection.table.turas_split != ""])
    chosen = chosen.assign(duration_s=chosen.duration)
    info = R.write_split_files(chosen, tmp_path)
    seen = set()
    for split, meta in info.items():
        ids = (tmp_path / meta["file"]).read_text(encoding="utf-8").splitlines()
        assert ids == sorted(ids) and len(ids) == meta["utterances"]
        assert not seen & set(ids)
        seen |= set(ids)
    assert seen == set(chosen.utterance_id)


def test_inference_rows_have_strict_allowlist(selection):
    chosen = R.assign_ids(selection.table[selection.table.turas_split != ""])
    chosen = chosen.assign(audio_uri="audio/x.flac", audio_sha256="0" * 64, sample_rate_hz=16000,
                           channels=1, duration_s=chosen.duration)
    rows = R.inference_rows(chosen, CONFIG, "test", "f" * 64)
    allowed = {"schema_version", "dataset_release", "split_manifest_sha256", "utterance_id", "audio_uri",
               "audio_sha256", "duration_s", "sample_rate_hz", "channels", "language"}
    assert rows and all(set(r) == allowed for r in rows)
