"""Standalone submission verification for TURAS.

Checks without requiring GPU, torch, or private audio/reference files:
1. The frozen protocol SHA-256 and all 56 critical code hashes.
2. Exact WER/CER recomputation from stored S/D/I/N counts in
   results/final/m45_test_summary.json for all 20 conditions, checked against m45_metrics.csv.
3. Frozen split hashes, disjointness and exact train/validation/test clip counts.
4. Presence of the submission guide, presentation, licenses, relational CSVs, and key reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = Path("configs/evaluation/m45_final_protocol.json")
PROTOCOL_SHA256 = "288aebd71d8a3574e13d436748d4161bc258a487e65066f1c3508bc8e835bec1"
SPLIT_DIR = Path("data/splits/turas-tr-econ-legis-v0.1")
EXPECTED_SPLIT_COUNTS = {"train": 4376, "validation": 438, "test": 636}
EXPECTED_PIN_COUNT = 56
EXPECTED_SYSTEM_COUNT = 20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_frozen_protocol(root: Path) -> dict:
    data = (root / PROTOCOL_PATH).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != PROTOCOL_SHA256:
        raise AssertionError(
            f"Frozen protocol SHA-256 mismatch: expected {PROTOCOL_SHA256}, got {actual}"
        )
    return json.loads(data)


def verify_protocol_hashes(root: Path = ROOT) -> int:
    protocol = _read_frozen_protocol(root)
    pins: dict[str, str] = protocol["critical_code_sha256"]
    if len(pins) != EXPECTED_PIN_COUNT:
        raise AssertionError(f"Expected {EXPECTED_PIN_COUNT} code pins, got {len(pins)}")
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
    protocol = _read_frozen_protocol(root)
    expected_ids = {
        entry["system_id"]
        for key in ("systems", "rescore_conditions")
        for entry in protocol[key]
    }
    summary = json.loads(
        (root / "results/final/m45_test_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("protocol_sha256") != PROTOCOL_SHA256:
        raise AssertionError("The test summary does not identify the frozen protocol")
    with (root / "results/final/m45_metrics.csv").open(encoding="utf-8", newline="") as fh:
        csv_records = list(csv.DictReader(fh))

    systems = summary["systems"]
    if len(systems) != EXPECTED_SYSTEM_COUNT or len(csv_records) != EXPECTED_SYSTEM_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_SYSTEM_COUNT} systems in each result file, "
            f"got {len(systems)} in JSON and {len(csv_records)} in CSV"
        )
    csv_rows = {row["system"]: row for row in csv_records}
    if {entry["system_id"] for entry in systems} != expected_ids or set(csv_rows) != expected_ids:
        raise AssertionError("Result system IDs do not match the frozen protocol")

    for entry in systems:
        sid = entry["system_id"]
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
    protocol = _read_frozen_protocol(root)
    split_dir = root / SPLIT_DIR
    manifest_path = split_dir / "split_manifest.json"
    if _sha256(manifest_path) != protocol["split_manifest_sha256"]:
        raise AssertionError("Frozen split manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    splits: dict[str, set[str]] = {}
    for name, expected_count in EXPECTED_SPLIT_COUNTS.items():
        split_path = split_dir / f"{name}.txt"
        data = split_path.read_bytes()
        lines = [line.strip() for line in data.decode("utf-8").splitlines() if line.strip()]
        if len(lines) != expected_count:
            raise AssertionError(f"Expected {expected_count} IDs in {name}.txt, got {len(lines)}")
        if len(lines) != len(set(lines)):
            raise AssertionError(f"Duplicate IDs in {name}.txt")
        if hashlib.sha256(data).hexdigest() != manifest["splits"][name]["sha256"]:
            raise AssertionError(f"Frozen split SHA-256 mismatch for {name}.txt")
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
        "SUBMISSION.md",
        "presentation/TURAS.pptx",
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
        f"OK: frozen protocol and split hashes, {n_pins} pinned code hashes, "
        f"{n_systems} systems recomputed from S/D/I/N, "
        f"{total_clips} disjoint split clips ({split_counts['train']}/{split_counts['validation']}/{split_counts['test']}), "
        f"and {n_files} required submission files verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
