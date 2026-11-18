"""Ingest exported listening-review decisions (Member 1).

    python scripts/m1_apply_reviews.py --artifact-root ../turas-artifacts DECISIONS.json [...]
    python scripts/m1_apply_reviews.py --artifact-root ../turas-artifacts --role adjudication ADJ.json

Validates every decision against the protected references and the guide
(data/verification/turas-verbatim-v0.1.md); if any is invalid, nothing is
written. Valid new records are appended to
<artifact-root>/<release>/review_records.jsonl (re-ingesting the same export is
a no-op). The current status of every reviewed utterance is rewritten to
<artifact-root>/<release>/review_status.jsonl, and aggregate counts (no text)
to data/verification/<release>-audit.md. References are not rewritten here;
the freeze step merges the resolved statuses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from turkish_asr_rdl.data import review as RV  # noqa: E402

RELEASE = "turas-tr-econ-legis-v0.1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", required=True, type=Path)
    ap.add_argument("--role", choices=RV.ROLES, default="primary")
    ap.add_argument("decisions", nargs="*", type=Path, help="none: only rewrite status and audit")
    args = ap.parse_args()
    root = args.artifact_root / RELEASE
    refs = {}
    for line in (root / f"{RELEASE}.references.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        refs[r["utterance_id"]] = r
    rec_path = root / "review_records.jsonl"
    existing = RV.load_records(rec_path)
    kits = [json.loads(p.read_text(encoding="utf-8")) for p in args.decisions]
    try:
        eq_path = root / "audio_equivalence.jsonl"
        equivalent = {}
        if eq_path.exists():
            for line in eq_path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                equivalent[e["reviewed_audio_sha256"]] = e["release_audio_sha256"]
        new, stats = RV.ingest(refs, kits, existing, args.role, equivalent)
    except RV.ReviewError as exc:
        print(exc, file=sys.stderr)
        return 1
    with open(rec_path, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    records = existing + new
    status = RV.status_table(refs, records)
    with open(root / "review_status.jsonl", "w", encoding="utf-8") as f:
        for row in status.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pilot_file = REPO / "data/verification" / f"{RELEASE}-pilot-ids.txt"
    pilot = set(pilot_file.read_text().split()) if pilot_file.exists() else set()
    md = RV.audit_markdown(RELEASE, refs, status, records, pilot)
    (REPO / "data/verification" / f"{RELEASE}-audit.md").write_text(md, encoding="utf-8")
    print(dict(stats))
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
