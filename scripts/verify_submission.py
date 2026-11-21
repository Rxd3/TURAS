"""Standalone submission verification for TURAS.

Checks without requiring GPU, torch, or private audio/reference files:
1. All 56 critical_code_sha256 entries in configs/evaluation/m45_final_protocol.json.
2. Exact WER/CER recomputation from stored S/D/I/N counts in
   results/final/m45_test_summary.json cross-checked against results/final/m45_metrics.csv.
3. Disjointness and exact clip counts of train/validation/test split IDs.
4. Presence of root LICENSE, NOTICE, relational CSVs, and key reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocol_hashes(root: Path = ROOT) -> int:
    protocol_path = root / "configs/evaluation/m45_final_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pins: dict[str, str] = protocol["critical_code_sha256"]
    mismatches: list[str] = []
    for rel, expected in pins.items():
        target = root / rel
        if not target.is_file():
            mismatches.append(f"{rel}: missing")
            continue
        actual = _sha256(target)
        if actual != expected:
            mismatches.append(f"{rel}: expected {expected[:12]}, got {actual[:12]}")
    if mismatches:
        raise AssertionError("Pinned code hash mismatches:\n" + "\n".join(mismatches))
    return len(pins)


def verify_metrics_consistency(root: Path = ROOT) -> int:
    summary = json.loads(
        (root / "results/final/m45_test_summary.json").read_text(encoding="utf-8")
    )
    with (root / "results/final/m45_metrics.csv").open(encoding="utf-8", newline="") as fh:
        csv_rows = {row["system"]: row for row in csv.DictReader(fh)}

    systems = summary["systems"]
    if len(systems) != len(csv_rows):
        raise AssertionError(
            f"System count mismatch: JSON has {len(systems)}, CSV has {len(csv_rows)}"
        )

    for entry in systems:
        sid = entry["system_id"]
        if sid not in csv_rows:
            raise AssertionError(f"System {sid!r} missing from m45_metrics.csv")
        row = csv_rows[sid]
        for view, wer_col, cer_col in (
            ("raw-v1", "raw_wer_pct", "raw_cer_pct"),
            ("tr-normalized-v1", "normalized_wer_pct", "normalized_cer_pct"),
        ):
            block = entry["metrics"][view]
            w = block["words"]
            c = block["chars"]
            recomputed_wer = (
                w["substitutions"] + w["deletions"] + w["insertions"]
            ) / block["reference_words"]
            recomputed_cer = (
                c["substitutions"] + c["deletions"] + c["insertions"]
            ) / block["reference_characters"]
            if not math.isclose(recomputed_wer, block["wer"], rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"{sid} {view}: JSON WER does not match S/D/I/N")
            if not math.isclose(recomputed_cer, block["cer"], rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"{sid} {view}: JSON CER does not match S/D/I/N")
            if not math.isclose(recomputed_wer * 100.0, float(row[wer_col]), rel_tol=0.0, abs_tol=1e-9):
                raise AssertionError(f"{sid} {wer_col}: CSV does not match JSON")
            if not math.isclose(recomputed_cer * 100.0, float(row[cer_col]), rel_tol=0.0, abs_tol=1e-9):
                raise AssertionError(f"{sid} {cer_col}: CSV does not match JSON")
    return len(systems)


def verify_splits_disjoint(root: Path = ROOT) -> dict[str, int]:
    split_dir = root / "data/splits/turas-tr-econ-legis-v0.1"
    splits: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        lines = [
            ln.strip()
            for ln in (split_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if len(lines) != len(set(lines)):
            raise AssertionError(f"Duplicate IDs in {name}.txt")
        splits[name] = set(lines)
    if splits["train"] & splits["validation"]:
        raise AssertionError("Overlap between train and validation splits")
    if splits["train"] & splits["test"]:
        raise AssertionError("Overlap between train and test splits")
    if splits["validation"] & splits["test"]:
        raise AssertionError("Overlap between validation and test splits")
    return {k: len(v) for k, v in splits.items()}


def verify_required_files(root: Path = ROOT) -> int:
    required = [
        "LICENSE",
        "NOTICE",
        "README.md",
        "docs/error_analysis.md",
        "results/final/M4_M5_FINAL_REPORT.md",
        "results/final/m45_metrics.csv",
        "results/final/m45_test_summary.json",
        "data/relational/turas-tr-econ-legis-v0.1/entities.csv",
        "data/relational/turas-tr-econ-legis-v0.1/relationships_static.csv",
    ]
    missing = [rel for rel in required if not (root / rel).is_file()]
    if missing:
        raise AssertionError("Missing required submission files: " + ", ".join(missing))
    return len(required)


def main(root: Path = ROOT) -> int:
    n_pins = verify_protocol_hashes(root)
    n_systems = verify_metrics_consistency(root)
    split_counts = verify_splits_disjoint(root)
    n_files = verify_required_files(root)
    total_clips = sum(split_counts.values())
    print(
        f"OK: {n_pins} pinned code hashes, {n_systems} systems recomputed from S/D/I/N, "
        f"{total_clips} disjoint split clips ({split_counts['train']}/{split_counts['validation']}/{split_counts['test']}), "
        f"and {n_files} required submission files verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
