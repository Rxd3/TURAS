"""Ingest exported listening-review decisions (Member 1).

    python scripts/m1_apply_reviews.py --artifact-root ../turas-artifacts DECISIONS.json [...]

Checks every decision against the protected references (ID and audio hash),
appends contract review records to <artifact-root>/<release>/review_records.jsonl
and writes aggregate counts (no text) to data/verification/<release>-audit.md.
References are not rewritten here; the freeze step merges accepted records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELEASE = "turas-tr-econ-legis-v0.1"
STATUS = {"accept": "verified", "correct": "verified", "reject": "rejected"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", required=True, type=Path)
    ap.add_argument("decisions", nargs="+", type=Path)
    args = ap.parse_args()
    root = args.artifact_root / RELEASE
    refs = {}
    for line in (root / f"{RELEASE}.references.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        refs[r["utterance_id"]] = r
    out_path = root / "review_records.jsonl"
    records, counts, issues, undecided = [], Counter(), Counter(), 0
    for path in args.decisions:
        kit = json.loads(path.read_text(encoding="utf-8"))
        for d in kit["reviews"]:
            ref = refs.get(d["utterance_id"])
            if ref is None or ref["audio_sha256"] != d["audio_sha256"]:
                raise SystemExit(f"{d['utterance_id']}: unknown ID or audio hash mismatch")
            if d["decision"] not in STATUS:
                undecided += 1
                continue
            if d["decision"] == "accept" and d.get("text_changed"):
                raise SystemExit(f"{d['utterance_id']}: text edited but decision is accept")
            key = f"{d['utterance_id']}|{d['reviewer_id']}|{d['review_utc']}"
            records.append({
                "verification_record_id": "rev_" + hashlib.sha256(key.encode()).hexdigest()[:16],
                "utterance_id": d["utterance_id"], "reviewer_id": d["reviewer_id"],
                "review_utc": d["review_utc"], "audio_sha256": d["audio_sha256"],
                "annotation_policy_version": d["annotation_policy_version"], "decision": d["decision"],
                "proposed_status": STATUS[d["decision"]], "corrected_text": d.get("corrected_text"),
                "issue_codes": d.get("issue_codes", []), "note": d.get("note", ""),
                "adjudicator_id": None, "kit": kit.get("kit"),
            })
            counts[(ref["split"], d["decision"])] += 1
            issues.update(d.get("issue_codes", []))
    with open(out_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = [f"# {RELEASE} verification audit (aggregate, no text)", "",
             f"Records appended: {len(records)}; undecided items skipped: {undecided}.", "",
             "| Split | accept | correct | reject |", "| --- | ---: | ---: | ---: |"]
    for s in ("test", "validation", "train"):
        lines.append(f"| {s} | {counts[(s, 'accept')]} | {counts[(s, 'correct')]} | {counts[(s, 'reject')]} |")
    lines += ["", "Issue codes: " + (", ".join(f"{k} {v}" for k, v in issues.most_common()) or "none"), ""]
    (REPO / "data/verification" / f"{RELEASE}-audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
