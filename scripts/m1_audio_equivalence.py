"""Record which review-kit audio is equivalent to the release audio (Member 1).

    python scripts/m1_audio_equivalence.py --artifact-root ../turas-artifacts KIT.html [...]

Rebuilding the FLAC files on another machine can change their bytes (decoder
and encoder rounding) without changing what a listener hears. For every clip
embedded in a review kit, this decodes the kit audio and the release audio and
records the pair in <artifact-root>/<release>/audio_equivalence.jsonl only if
both have the same sample rate and length and no sample differs by more than
1 LSB (16-bit). Clips that fail are listed and not recorded, so decisions on
them are refused by m1_apply_reviews.py and must be reviewed again.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

RELEASE = "turas-tr-econ-legis-v0.1"
MAX_LSB = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", required=True, type=Path)
    ap.add_argument("kits", nargs="+", type=Path)
    args = ap.parse_args()
    root = args.artifact_root / RELEASE
    refs = {}
    for line in (root / f"{RELEASE}.references.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        refs[r["utterance_id"]] = r
    out_path = root / "audio_equivalence.jsonl"
    rows = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            rows[e["reviewed_audio_sha256"]] = e
    failed = []
    for kit in args.kits:
        html = kit.read_text(encoding="utf-8")
        items = json.loads(re.search(r"const ITEMS = (\[.*?\]);\n", html, re.S).group(1))
        for it in items:
            old = base64.b64decode(it["audio"])
            if hashlib.sha256(old).hexdigest() != it["audio_sha256"]:
                raise SystemExit(f"{it['utterance_id']}: kit audio does not match its own hash")
            ref = refs[it["utterance_id"]]
            if it["audio_sha256"] == ref["audio_sha256"]:
                continue
            a, sa = sf.read(io.BytesIO(old), dtype="int16")
            b, sb = sf.read(root / ref["audio_uri"], dtype="int16")
            same = sa == sb and a.shape == b.shape
            max_lsb = int(np.max(np.abs(a.astype(np.int32) - b.astype(np.int32)))) if same else None
            if not same or max_lsb > MAX_LSB:
                failed.append(it["utterance_id"])
                continue
            d = (a.astype(np.float64) - b.astype(np.float64))
            snr = float(10 * np.log10(np.sum(a.astype(np.float64) ** 2) / max(np.sum(d ** 2), 1e-12)))
            rows[it["audio_sha256"]] = {
                "utterance_id": it["utterance_id"], "reviewed_audio_sha256": it["audio_sha256"],
                "release_audio_sha256": ref["audio_sha256"], "sample_rate_hz": sa, "samples": int(a.shape[0]),
                "max_abs_diff_lsb": max_lsb, "snr_db": round(snr, 1), "kit": kit.name,
            }
    with open(out_path, "w", encoding="utf-8") as f:
        for k in sorted(rows):
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    print(f"equivalent pairs recorded: {len(rows)}; failed: {len(failed)}")
    for u in failed:
        print("  not equivalent (re-review):", u)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
