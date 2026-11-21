from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_oracle():
    spec = importlib.util.spec_from_file_location(
        "compute_nbest_oracle", ROOT / "scripts/compute_nbest_oracle.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_nbest_oracle_finds_headroom_and_diversity() -> None:
    mod = _load_oracle()
    refs = {
        "u1": "Merkez Bankası faiz kararı",
        "u2": "enflasyon yüzde elli",
    }
    nbest_rows = [
        {
            "utterance_id": "u1",
            "status": "ok",
            "candidates": [
                {"rank": 2, "text": "Merkez Bankası faiz kararı"},
                {"rank": 1, "text": "Merkez Bankası faiz kar"},
            ],
        },
        {
            "utterance_id": "u2",
            "status": "ok",
            "candidates": [
                {"rank": 1, "text": "enflasyon yüzde elli"},
                {"rank": 2, "text": "Enflasyon, yüzde elli!"},
            ],
        },
    ]
    res = mod.evaluate_nbest_oracle(refs, nbest_rows, view="tr-normalized-v1")
    assert res["utterances"] == 2
    assert res["one_best_words"]["substitutions"] == 1
    assert res["oracle_words"]["substitutions"] == 0
    assert res["oracle_wer_pct"] == 0.0
    assert res["headroom_pp"] > 0.0
    assert res["oracle_beats_rank1_count"] == 1
    assert res["mean_candidates"] == 2.0
    assert res["mean_unique_normalized_candidates"] == 1.5


def test_cli_writes_aggregate_only_json(tmp_path: Path) -> None:
    mod = _load_oracle()
    ref_path = tmp_path / "refs.jsonl"
    nbest_path = tmp_path / "sys.nbest.jsonl"
    out_path = tmp_path / "oracle.json"

    ref_path.write_text(
        json.dumps({"utterance_id": "u1", "reference_text": "SGK primi arttı"}) + "\n",
        encoding="utf-8",
    )
    nbest_path.write_text(
        json.dumps(
            {
                "utterance_id": "u1",
                "status": "ok",
                "candidates": [
                    {"rank": 1, "text": "esgeka primi arttı"},
                    {"rank": 2, "text": "SGK primi arttı"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = mod.main(
        [
            "--references",
            str(ref_path),
            "--nbest",
            f"F1={nbest_path}",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    raw_out = out_path.read_text(encoding="utf-8")
    assert "SGK" not in raw_out
    assert "esgeka" not in raw_out
    data = json.loads(raw_out)
    assert data["systems"]["F1"]["oracle_wer_pct"] == 0.0
    assert data["systems"]["F1"]["oracle_beats_rank1_count"] == 1


def test_evaluate_nbest_oracle_rejects_id_mismatch() -> None:
    mod = _load_oracle()
    with pytest.raises(ValueError, match="utterance_id mismatch"):
        mod.evaluate_nbest_oracle({"u1": "merhaba"}, [{"utterance_id": "u2", "candidates": []}])
