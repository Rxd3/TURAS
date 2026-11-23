"""Export the raw WorldSpeech tr_tr text/metadata and the per-clip removal log.

Usage (network access to huggingface.co; reads every column except audio,
about 70 MB over HTTP range requests):

    python scripts/m1_export_raw.py
    python scripts/m1_export_raw.py --metadata-cache path/to/metadata.parquet

Writes, inside the repository:

* ``data/raw/worldspeech-tr_tr-<rev8>/metadata-*.parquet``: every upstream row
  (239,654) with all non-audio columns (human and ASR transcripts, CER, SNR,
  DNSMOS, duration, source, session date, segment ID) plus upstream split, shard
  and row position.
* ``data/cleaned/<release>/exclusion_ledger.jsonl.gz``: one line per upstream
  row with its TURAS split or the first exclusion reason.

The removal log is produced with the same screening and selection code as
``scripts/m1_build_release.py`` and must reproduce the frozen release exactly:
the ledger SHA-256, the exclusion counts and the utterance IDs of every split are
checked against ``data/splits/<release>/``. Any mismatch stops the script
before anything is written.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from turkish_asr_rdl.data import release as R  # noqa: E402
from turkish_asr_rdl.data.screening import Lexicon  # noqa: E402
from turkish_asr_rdl.data.selection import leakage_report, screen, select  # noqa: E402

LEDGER_COLUMNS = ["segment_id", "upstream_split", "source_url", "session_date", "duration",
                  "cer", "issue_codes", "term_ids", "turas_split", "exclusion_reason"]
ROWS_PER_FILE = 60_000  # keeps every parquet file far below GitHub's 100 MB limit


def ledger_bytes(table: pd.DataFrame) -> bytes:
    """Serialize the ledger exactly as m1_build_release.py does."""
    ledger = table.astype({"duration": float, "cer": float})[LEDGER_COLUMNS]
    rows = ledger.sort_values("segment_id").to_dict("records")
    return "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/datasets/turas-tr-econ-legis-v0.1.json"))
    ap.add_argument("--metadata-cache", type=Path, help="reuse a metadata parquet from m1_build_release.py")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rid, rev = cfg["release_id"], cfg["upstream"]["revision"]
    split_dir = REPO / "data/splits" / rid
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))

    if args.metadata_cache and args.metadata_cache.exists():
        meta = pd.read_parquet(args.metadata_cache)
    else:
        from turkish_asr_rdl.data import worldspeech as W

        meta = W.fetch_metadata(rev)
        if args.metadata_cache:
            meta.to_parquet(args.metadata_cache)
    print(f"upstream rows: {len(meta)}")

    # Same screening and selection as the release builder.
    table = select(screen(meta, Lexicon.load(REPO / cfg["lexicon"])), cfg).table
    leak = leakage_report(table)
    if any(leak.values()):
        raise SystemExit(f"leakage checks failed: {leak}")

    counts = {k: int(v) for k, v in table["exclusion_reason"].replace("", "selected").value_counts().items()}
    frozen_counts = {k: v for k, v in manifest["exclusion_counts"].items() if v}  # drop zero-count reasons
    if counts != frozen_counts:
        raise SystemExit(f"exclusion counts differ from the frozen manifest: {counts}")
    chosen = R.assign_ids(table[table.turas_split != ""])
    for split in R.SPLITS:
        frozen = (split_dir / f"{split}.txt").read_text(encoding="utf-8").split()
        if sorted(chosen.loc[chosen.turas_split == split, "utterance_id"]) != sorted(frozen):
            raise SystemExit(f"{split}: utterance IDs differ from the frozen split file")
    data = ledger_bytes(table)
    digest = hashlib.sha256(data).hexdigest()
    expected = manifest.get("exclusion_ledger_sha256")
    if expected and digest != expected:
        raise SystemExit(f"ledger SHA-256 {digest} differs from the frozen {expected}")
    print("frozen release reproduced: counts, split IDs" + (" and ledger SHA-256" if expected else ""))

    raw_dir = REPO / "data/raw" / f"worldspeech-tr_tr-{rev[:8]}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    full = pa.Table.from_pandas(meta, preserve_index=False)
    for i, start in enumerate(range(0, full.num_rows, ROWS_PER_FILE)):
        pq.write_table(full.slice(start, ROWS_PER_FILE), raw_dir / f"metadata-{i:02d}.parquet",
                       compression="zstd")
    out = REPO / "data/cleaned" / rid / "exclusion_ledger.jsonl.gz"
    with gzip.GzipFile(out, "wb", mtime=0) as f:  # mtime=0 keeps the archive bytes reproducible
        f.write(data)
    print(f"wrote {raw_dir.relative_to(REPO)} and {out.relative_to(REPO)} (ledger sha256 {digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
