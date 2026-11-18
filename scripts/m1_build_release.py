"""Build the proposed TURAS release from pinned WorldSpeech tr_tr (Member 1).

Usage (network access to huggingface.co required; ~26 GB streamed, ~2 GB kept):

    python scripts/m1_build_release.py --artifact-root ../turas-artifacts
    python scripts/m1_build_release.py --artifact-root ../turas-artifacts --skip-audio

Protected outputs go to --artifact-root (outside Git): audio/, references,
inference exports, exclusion ledger, cached metadata. Public outputs go to the
repository: data/splits/<release>/, data/manifests/<release>.json and
data/metadata/<release>-*.{csv,json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from turkish_asr_rdl.data import release as R  # noqa: E402
from turkish_asr_rdl.data.screening import Lexicon  # noqa: E402
from turkish_asr_rdl.data.selection import leakage_report, screen, select  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/datasets/turas-tr-econ-legis-v0.1.json"))
    ap.add_argument("--artifact-root", required=True, type=Path)
    ap.add_argument("--skip-audio", action="store_true", help="selection, IDs and split files only")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rid, rev = cfg["release_id"], cfg["upstream"]["revision"]
    root: Path = args.artifact_root / rid
    root.mkdir(parents=True, exist_ok=True)

    from turkish_asr_rdl.data import worldspeech as W

    meta_cache = root / f"worldspeech-tr_tr-{rev[:12]}-metadata.parquet"
    if meta_cache.exists():
        meta = pd.read_parquet(meta_cache)
    else:
        meta = W.fetch_metadata(rev)
        meta.to_parquet(meta_cache)
    print(f"metadata rows: {len(meta)}")

    lexicon = Lexicon.load(REPO / cfg["lexicon"])
    sel = select(screen(meta, lexicon), cfg)
    table = sel.table
    leak = leakage_report(table)
    if any(v for v in leak.values()):
        raise SystemExit(f"leakage checks failed: {leak}")

    chosen = R.assign_ids(table[table.turas_split != ""])
    ledger = table.astype({"duration": float, "cer": float})[["segment_id", "upstream_split", "source_url", "session_date", "duration",
                    "cer", "issue_codes", "term_ids", "turas_split", "exclusion_reason"]]
    ledger_sha = R.write_jsonl(root / "exclusion_ledger.jsonl",
                               ledger.sort_values("segment_id").to_dict("records"))

    if args.skip_audio:
        chosen = chosen.assign(audio_uri="", audio_sha256="", sample_rate_hz=16000, channels=1,
                               duration_s=chosen["duration"])
    else:
        read = W.shard_audio_reader(rev)
        chosen = R.write_audio(chosen, root, W.prefetching(read, R.shard_requests(chosen)))
        drift = (chosen["duration_s"] - chosen["duration"]).abs()
        print(f"max |measured - upstream duration| = {drift.max():.4f} s")

    split_dir = REPO / "data/splits" / rid
    split_info = R.write_split_files(chosen, split_dir)
    sources = (chosen.groupby(["source_id", "source_url", "session_date"]).size()
               .rename("selected_utterances").reset_index().sort_values("source_id"))
    meta_dir = REPO / "data/metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    sources.to_csv(meta_dir / f"{rid}-source-registry.csv", index=False, lineterminator="\n")

    ref_sha = R.write_jsonl(root / f"{rid}.references.jsonl", R.reference_rows(chosen, cfg))
    split_manifest = {
        "schema_version": R.SCHEMA_VERSION,
        "dataset_release": rid,
        "status": cfg["status"],
        "upstream": cfg["upstream"],
        "config_sha256": R.sha256_file(Path(args.config)),
        "lexicon_sha256": R.sha256_file(REPO / cfg["lexicon"]),
        "algorithm": "turkish_asr_rdl.data.selection.select (recording-grouped, temporal test, seeded validation)",
        "group_key": cfg["splits"]["group_key"],
        "seed": cfg["splits"]["validation"]["seed"],
        "caps_minutes": {k: round(v / 60, 3) for k, v in sel.caps_s.items()},
        "splits": split_info,
        "leakage_checks": leak,
        "exclusion_counts": table["exclusion_reason"].replace("", "selected").value_counts().to_dict(),
        "speaker_identity": "unknown for every row; speaker-disjointness is NOT claimed",
        "references_sha256": ref_sha,
        "exclusion_ledger_sha256": ledger_sha,
        "audio_written": not args.skip_audio,
    }
    sm_path = split_dir / "split_manifest.json"
    sm_path.write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sm_sha = R.sha256_file(sm_path)

    inf = {s: R.write_jsonl(root / f"{rid}.inference.{s}.jsonl", R.inference_rows(chosen, cfg, s, sm_sha))
           for s in R.SPLITS}
    summary = {
        "dataset_release": rid,
        "split_manifest_sha256": sm_sha,
        "inference_exports_sha256": inf,
        "topic_hours": R.topic_hours(chosen),
        "audio_bytes_total": int(sum((root / u).stat().st_size for u in chosen.audio_uri)) if not args.skip_audio else None,
    }
    (REPO / "data/manifests").mkdir(parents=True, exist_ok=True)
    (REPO / "data/manifests" / f"{rid}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v["duration_s"] / 3600 for k, v in split_info.items()}, indent=1))
    print("split_manifest_sha256", sm_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
