"""Recover a separately identified validation-minutes view; never synthesize gold.

Reads pinned WorldSpeech identity columns and human_transcript only. Original
M1 corrections/review evidence remain unavailable and are never inferred.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from turkish_asr_rdl.data.release import source_id, to_ms, utterance_id
from turkish_asr_rdl.training.acoustic import read_jsonl, safe_audio_path, sha256_file, write_json
from turkish_asr_rdl.schemas.predictions import validate_inference_inputs

VARIANT = "owner-recovered-validation-minutes-v1"
REVISION = "b76b85395039f50332fbc421f7c098ee2f2653d7"
RELEASE = "turas-tr-econ-legis-v0.1"
SPLIT_HASH = "a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8"


def make_reference(upstream: dict, inference: dict, *, shard: str, row_index: int):
    """Bind actual upstream human text to an exactly matching existing audio ID."""
    start, end = to_ms(upstream["source_start_s"]), to_ms(upstream["source_end_s"])
    uid = utterance_id(upstream["source_url"], upstream["segment_id"], start, end)
    if uid != inference["utterance_id"]:
        raise ValueError("upstream identity does not match the validation audio ID")
    text = upstream.get("human_transcript")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("actual human_transcript is required; ASR text is never substituted")
    if abs(float(upstream["duration"]) - inference["duration_s"]) > .02:
        raise ValueError("upstream/validation audio duration mismatch")
    return {"schema_version": "0.1", "dataset_release": RELEASE, "split": "validation",
        "reference_variant": VARIANT, "utterance_id": uid,
        "source_id": source_id(upstream["source_url"]), "source_segment_id": upstream["segment_id"],
        "upstream_dataset": "disco-eth/WorldSpeech", "upstream_config": "tr_tr",
        "upstream_revision": REVISION, "upstream_split": shard.split("-")[0],
        "upstream_shard": shard, "row_in_shard": row_index,
        "source_url": upstream["source_url"], "start_ms": start, "end_ms": end,
        "audio_uri": inference["audio_uri"], "audio_sha256": inference["audio_sha256"],
        "sample_rate_hz": inference["sample_rate_hz"], "channels": inference["channels"],
        "duration_s": inference["duration_s"], "language": "tr_tr", "speaker_ids": None,
        "session_id": source_id(upstream["source_url"]),
        "leakage_group_id": source_id(upstream["source_url"]),
        "reference_text": text, "reference_origin": "upstream_human",
        "verification_status": "unreviewed", "verification_record_id": None,
        "annotation_policy_version": "worldspeech-upstream-minutes-v1",
        "rights_record_id": "rights_worldspeech_tr_tr_tbmm"}


def recover(*, inference: Path, audio_root: Path, train_references: Path,
            output: Path, workers: int = 8):
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    if output.exists():
        raise ValueError("output must be new; frozen/private reference versions are not overwritten")
    split_root = Path("data/splits") / RELEASE
    manifest_path = split_root / "split_manifest.json"
    if sha256_file(manifest_path) != SPLIT_HASH:
        raise ValueError("frozen split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids_path = split_root / "validation.txt"
    if sha256_file(ids_path) != manifest["splits"]["validation"]["sha256"]:
        raise ValueError("validation ID hash mismatch")
    ids = ids_path.read_text(encoding="utf-8").splitlines()
    rows = read_jsonl(inference)
    if not validate_inference_inputs(rows).ok or [r["utterance_id"] for r in rows] != ids:
        raise ValueError("gold-free inference rows must match frozen validation IDs in order")
    if any(r["dataset_release"] != RELEASE or r["split_manifest_sha256"] != SPLIT_HASH for r in rows):
        raise ValueError("inference release/split mismatch")
    for row in rows:
        if sha256_file(safe_audio_path(audio_root, row["audio_uri"])) != row["audio_sha256"]:
            raise ValueError("validation audio hash mismatch")
    train = read_jsonl(train_references)
    if any(r.get("split") != "train" for r in train):
        raise ValueError("source exclusion input must contain train references only")
    train_sources = {r["source_id"] for r in train}
    registry_path = Path("data/metadata") / (RELEASE + "-source-registry.csv")
    with registry_path.open(encoding="utf-8", newline="") as stream:
        registry = list(csv.DictReader(stream))
    sources = {r["source_url"] for r in registry if r["source_id"] not in train_sources
               and r["session_date"] < "2026-01-01"}
    if len(sources) != manifest["splits"]["validation"]["recordings"]:
        raise ValueError("source registry does not identify the exact validation recording count")
    by_id = {r["utterance_id"]: r for r in rows}
    fs = HfFileSystem()
    remote = f"datasets/disco-eth/WorldSpeech@{REVISION}/data/tr_tr"
    paths = sorted(p for p in fs.ls(remote, detail=False) if p.endswith(".parquet"))
    output.mkdir(parents=True, exist_ok=False)
    audit = {"status": "recovering", "reference_variant": VARIANT,
        "dataset_release": RELEASE, "split_manifest_sha256": SPLIT_HASH,
        "upstream_revision": REVISION, "authority": "docs/owner_decisions.md D35",
        "inference_sha256": sha256_file(inference), "train_source_inventory_sha256": sha256_file(train_references),
        "source_registry_sha256": sha256_file(registry_path), "validation_ids_sha256": sha256_file(ids_path),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "m1_private_validation_corrections_available": False,
        "human_listening_verification_performed": False, "asr_transcript_used": False,
        "test_reference_archive_opened": False}
    write_json(output / "reference-manifest.json", audit)
    def one(path):
        for attempt in range(4):
            try:
                with HfFileSystem().open(path, "rb", block_size=65536) as stream:
                    pf = pq.ParquetFile(stream)
                    # Identity comes first. Human text is read only in shards
                    # containing target validation IDs; no ASR/audio column is requested.
                    identity = pf.read(columns=["source_url", "segment_id", "source_start_s", "source_end_s", "duration"]).to_pylist()
                    selected = []
                    for index, row in enumerate(identity):
                        if row["source_url"] not in sources:
                            continue
                        uid = utterance_id(row["source_url"], row["segment_id"], to_ms(row["source_start_s"]), to_ms(row["source_end_s"]))
                        if uid in by_id:
                            selected.append((index, uid, row))
                    if not selected:
                        return []
                    texts = pf.read(columns=["human_transcript"])["human_transcript"]
                    return [make_reference({**row, "human_transcript": texts[index].as_py()}, by_id[uid],
                                shard=path.rsplit("/", 1)[-1], row_index=index) for index, uid, row in selected]
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
    found = {}
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for finished, future in enumerate(as_completed([pool.submit(one, path) for path in paths]), 1):
                for row in future.result():
                    uid = row["utterance_id"]
                    if uid in found:
                        raise ValueError("ambiguous duplicate upstream identity")
                    found[uid] = row
                if finished % 25 == 0 or finished == len(paths):
                    print(json.dumps({"shards_checked": finished, "shards_total": len(paths), "validation_rows_recovered": len(found)}), flush=True)
        if set(found) != set(ids):
            raise ValueError(f"incomplete recovery: {len(found)}/{len(ids)} validation IDs")
        target = output / "references.validation.jsonl"
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            for uid in ids:
                stream.write(json.dumps(found[uid], ensure_ascii=False, allow_nan=False) + "\n")
        audit.update(status="recovered_upstream_minutes_not_m1_corrected_bundle", rows=len(found),
            recordings=len({r["source_id"] for r in found.values()}), references_sha256=sha256_file(target),
            verification_status="all rows unreviewed; private review records not imported",
            limitations=["Edited parliamentary minutes, not verified verbatim speech",
                         "M1 reports three corrected validation references that are unavailable here",
                         "This is a separate preliminary reference version, not M1's original frozen gold",
                         "Underlying parquet column chunks may include unrelated rows; only exact validation IDs are retained",
                         "Never supply this reference file to model inference or acoustic/RDL training"],
            ended_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        audit.update(status="failed", error_type=type(exc).__name__)
        write_json(output / "reference-manifest.json", audit)
        raise
    write_json(output / "reference-manifest.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inference", "audio-root", "train-references", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((Path.cwd() / "data/private").resolve()):
        parser.error("reference outputs must stay under ignored data/private")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be in 1..16")
    print(json.dumps(recover(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
