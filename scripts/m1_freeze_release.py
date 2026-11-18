"""Freeze the TURAS release after listening review (Member 1, with M6 review).

    python scripts/m1_freeze_release.py --artifact-root ../turas-artifacts \
        --status "frozen by M1 on 2026-11-2X; M6 review: <PR link>"

Refuses to run unless every validation and test row is verified, adjudicated
or rejected (``--draft-dir`` writes a trial freeze elsewhere instead).
Then, in one step:

- merges the resolved review status into the references (corrections become
  ``reference_text`` with ``reference_origin = local_human``; IDs unchanged);
- removes rejected rows from the splits and marks them in the exclusion ledger
  with reason ``rejected_in_listening_review``;
- rewrites split files, split manifest (with a verification summary and the
  hashes of the review records), inference exports and release summary.

Run ``scripts/m1_build_relational.py`` afterwards so the relational evidence
matches the frozen references.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import pandas as pd  # noqa: E402

from m1_build_release import _build_info  # noqa: E402
from turkish_asr_rdl.data import release as R  # noqa: E402
from turkish_asr_rdl.data import review as RV  # noqa: E402

RELEASE = "turas-tr-econ-legis-v0.1"
REJECT_REASON = "rejected_in_listening_review"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", required=True, type=Path)
    ap.add_argument("--status", required=True, help="status text written into the manifests")
    ap.add_argument("--draft-dir", type=Path,
                    help="trial run: allow unreviewed rows and write everything under this directory")
    args = ap.parse_args()

    root = args.artifact_root / RELEASE
    refs = {}
    for line in (root / f"{RELEASE}.references.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        refs[r["utterance_id"]] = r
    rec_path = root / "review_records.jsonl"
    records = RV.load_records(rec_path)
    kept, rejected = RV.apply_to_references(refs, records)

    open_rows = [r["utterance_id"] for r in kept if r["split"] in ("validation", "test")
                 and r["verification_status"] not in ("verified", "adjudicated")]
    if open_rows and not args.draft_dir:
        print(f"{len(open_rows)} validation/test rows are not verified or adjudicated yet; "
              f"first: {open_rows[:5]}. Use --draft-dir for a trial freeze.", file=sys.stderr)
        return 1

    out_root = args.draft_dir / RELEASE if args.draft_dir else root
    repo_out = args.draft_dir / "repo" if args.draft_dir else REPO
    out_root.mkdir(parents=True, exist_ok=True)

    # Exclusion ledger: rejected rows leave their split with an explicit reason.
    rejected_segments = {r["source_segment_id"] for r in rejected}
    ledger = [json.loads(x) for x in (root / "exclusion_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in ledger:
        if row["segment_id"] in rejected_segments:
            row["turas_split"], row["exclusion_reason"] = "", REJECT_REASON
    ledger_sha = R.write_jsonl(out_root / "exclusion_ledger.jsonl", ledger)
    ref_sha = R.write_jsonl(out_root / f"{RELEASE}.references.jsonl", kept)

    dates = {}
    with open(REPO / "data/metadata" / f"{RELEASE}-source-registry.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dates[row["source_id"]] = row["session_date"]
    df = pd.DataFrame(kept)
    df["turas_split"] = df["split"]
    df["session_date"] = df["source_id"].map(dates)
    df["topic_ids"] = df["domain_topics"].map("|".join)

    split_dir = repo_out / "data/splits" / RELEASE
    split_info = R.write_split_files(df, split_dir)
    old = json.loads((REPO / "data/splits" / RELEASE / "split_manifest.json").read_text(encoding="utf-8"))
    status_counts = Counter((r["split"], r["verification_status"], r["reference_origin"]) for r in kept)
    reject_counts = Counter(r["split"] for r in rejected)
    manifest = {
        **old,
        "status": args.status,
        "splits": split_info,
        "references_sha256": ref_sha,
        "exclusion_ledger_sha256": ledger_sha,
        "verification": {
            "policy": RV.POLICY_VERSION,
            "review_records_sha256": R.sha256_file(rec_path) if rec_path.exists() else None,
            "review_records": len(records),
            "reviewer_ids": len({r["reviewer_id"] for r in records}),
            "rows_by_split_status_origin": {f"{s}|{v}|{o}": n for (s, v, o) in sorted(status_counts)
                                            for n in [status_counts[(s, v, o)]]},
            "rejected_in_listening_review": {s: reject_counts[s] for s in R.SPLITS},
        },
        "frozen_on": date.today().isoformat() if not args.draft_dir else None,
        "build": _build_info(),
    }
    manifest["exclusion_counts"] = dict(old["exclusion_counts"])
    manifest["exclusion_counts"]["selected"] = len(kept)
    manifest["exclusion_counts"][REJECT_REASON] = len(rejected)
    sm_path = split_dir / "split_manifest.json"
    sm_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sm_sha = R.sha256_file(sm_path)

    cfg = {"release_id": RELEASE}
    inf = {s: R.write_jsonl(out_root / f"{RELEASE}.inference.{s}.jsonl", R.inference_rows(df, cfg, s, sm_sha))
           for s in R.SPLITS}
    summary = {
        "dataset_release": RELEASE,
        "split_manifest_sha256": sm_sha,
        "inference_exports_sha256": inf,
        "topic_hours": R.topic_hours(df),
        "audio_bytes_total": int(sum((root / u).stat().st_size for u in df.audio_uri)),
    }
    (repo_out / "data/manifests").mkdir(parents=True, exist_ok=True)
    (repo_out / "data/manifests" / f"{RELEASE}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"kept": len(kept), "rejected": len(rejected), "open_validation_test": len(open_rows),
                      "hours": {s: round(v["duration_s"] / 3600, 3) for s, v in split_info.items()},
                      "split_manifest_sha256": sm_sha}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
