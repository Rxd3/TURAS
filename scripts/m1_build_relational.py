"""Write RDL relational CSV tables for the proposed release (Member 1).

    python scripts/m1_build_relational.py --artifact-root ../turas-artifacts

Static lexicon tables (external knowledge) go to data/relational/turas-tr-econ-legis-v0.1/.
Per-split utterance evidence goes to <artifact-root>/<release>/relational/<split>/:
only train/ may feed graph training; validation/ and test/ are evaluator-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from turkish_asr_rdl.data.relational import write_split_tables, write_static_tables  # noqa: E402
from turkish_asr_rdl.data.release import sha256_file  # noqa: E402
from turkish_asr_rdl.data.screening import Lexicon  # noqa: E402

RELEASE = "turas-tr-econ-legis-v0.1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", required=True, type=Path)
    args = ap.parse_args()
    cfg = json.loads((REPO / "configs/datasets" / f"{RELEASE}.json").read_text(encoding="utf-8"))
    lex_path = REPO / cfg["lexicon"]
    lex = Lexicon.load(lex_path)
    root = args.artifact_root / RELEASE
    ref_path = root / f"{RELEASE}.references.jsonl"
    refs = pd.read_json(ref_path, lines=True, dtype=False)

    static_dir = REPO / "data/relational" / RELEASE
    write_static_tables(lex, sha256_file(lex_path), static_dir)
    manifest = {"dataset_release": RELEASE, "schema": "docs/contracts/rdl.md 0.1",
                "references_sha256": sha256_file(ref_path), "static": {}, "splits": {}}
    for p in sorted(static_dir.glob("*.csv")):
        manifest["static"][p.name] = sha256_file(p)
    for split in ("train", "validation", "test"):
        out = root / "relational" / split
        part = refs[refs["split"] == split]
        counts = write_split_tables(part, split, lex, RELEASE, manifest["references_sha256"], out)
        manifest["splits"][split] = {
            "access": "graph training allowed" if split == "train" else "evaluator-only; never in training snapshots",
            "counts": counts,
            "files": {p.name: sha256_file(p) for p in sorted(out.glob("*.csv"))},
        }
    (static_dir / "relational_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({s: v["counts"] for s, v in manifest["splits"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
