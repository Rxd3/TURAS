"""Stage owner-authorized training/inference bundles without accepting test gold."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tarfile

from turkish_asr_rdl.schemas.predictions import validate_inference_inputs
from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file, write_json


def stage_bundle(archive: Path, destination: Path, *, expected_sha256: str, kind: str,
                 release: str, split_sha256: str, train_ids: set[str]):
    """Hash first, validate every path, then extract only data into a fresh root."""
    if kind not in {"inference", "train"}:
        raise ValueError("only inference and train-reference bundles are accepted")
    if sha256_file(archive) != expected_sha256:
        raise ValueError("bundle SHA-256 mismatch; nothing extracted")
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("destination must be new; existing releases are never overwritten")
    with tarfile.open(archive, "r:*") as stream:
        selected = []
        names = set()
        for member in stream:
            path = PurePosixPath(member.name)
            if path.name.startswith("._") or "__MACOSX" in path.parts:
                continue
            target = (destination / member.name).resolve()
            if (path.is_absolute() or PureWindowsPath(member.name).drive or ".." in path.parts
                    or "\\" in member.name or ":" in member.name or not target.is_relative_to(destination)
                    or not (member.isfile() or member.isdir()) or member.issparse()):
                raise ValueError("unsafe archive member; nothing extracted")
            if target == destination / ".turas-stage-audit.json":
                raise ValueError("archive collides with the staging audit")
            if target in names:
                raise ValueError("duplicate archive path")
            names.add(target)
            if (member.isfile() and path.name != "SHA256SUMS"
                    and path.suffix.lower() not in {".flac", ".json", ".jsonl", ".txt", ".csv", ".md"}):
                raise ValueError("unexpected executable/binary member in a data bundle")
            selected.append(member)
        required = sum(m.size for m in selected if m.isfile())
        if required > shutil.disk_usage(destination.parent if destination.parent.exists() else Path.cwd()).free:
            raise ValueError("insufficient free disk space")
        destination.mkdir(parents=True, exist_ok=False)
        audit = {"status": "staging", "kind": kind, "dataset_release": release,
                 "source_sha256": expected_sha256, "split_manifest_sha256": split_sha256,
                 "reference_text_in_report": False}
        audit_path = destination / ".turas-stage-audit.json"
        try:
            stream.extractall(destination, members=selected, filter="data")
            inventories = []
            reference_sets = []
            for path in sorted(destination.rglob("*.jsonl")):
                rows = read_jsonl(path)
                if kind == "inference":
                    report = validate_inference_inputs(rows)
                    if not report.ok:
                        raise ValueError("inference bundle violates the gold-free contract")
                    if any(r["dataset_release"] != release or r["split_manifest_sha256"] != split_sha256 for r in rows):
                        raise ValueError("inference release/split mismatch")
                else:
                    if not rows or any(r.get("split") != "train" or r.get("dataset_release") != release
                                       or not isinstance(r.get("reference_text"), str) for r in rows):
                        raise ValueError("non-train or malformed reference file in training bundle")
                    ids = [r["utterance_id"] for r in rows]
                    if len(ids) != len(set(ids)) or set(ids) != train_ids:
                        raise ValueError("training references differ from frozen train IDs")
                    reference_sets.append(path)
                inventories.append({"file": path.relative_to(destination).as_posix(), "rows": len(rows), "sha256": sha256_file(path)})
            if not inventories or (kind == "train" and not reference_sets):
                raise ValueError("bundle has no required JSONL data")
            if kind == "train":
                for path in destination.rglob("*.csv"):
                    with path.open(encoding="utf-8", newline="") as source:
                        for row in csv.DictReader(source):
                            if row.get("split", "train") != "train" or row.get("access_split", "train") != "train":
                                raise ValueError("held-out relational evidence in training bundle")
                            if row.get("origin_kind", "train_annotation") != "train_annotation":
                                raise ValueError("unexpected provenance in train-only bundle")
            audit.update(status="staged_contract_checked", jsonl_files=inventories, payload_files=sum(m.isfile() for m in selected))
        except Exception as exc:
            audit.update(status="failed_validation", error_type=type(exc).__name__)
            write_json(audit_path, audit)
            raise
        write_json(audit_path, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description="Stage a frozen M1 training or inference bundle into ignored data/private")
    parser.add_argument("--kind", choices=["inference", "train"], required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, default=Path("data/manifests/turas-tr-econ-legis-v0.1-handoff.json"))
    args = parser.parse_args()
    private = (Path.cwd() / "data/private").resolve()
    if not args.output.resolve().is_relative_to(private):
        parser.error("--output must be under this checkout's ignored data/private directory")
    handoff = json.loads(args.handoff.read_text(encoding="utf-8"))
    suffix = "inference-ALL-MODEL-OWNERS.tar" if args.kind == "inference" else "train-references-M3-M4-M5.tar"
    bundle = next(b for b in handoff["bundles"] if b["file"].endswith(suffix))
    release = handoff["dataset_release"]
    split_root = Path("data/splits") / release
    manifest_path = split_root / "split_manifest.json"
    if sha256_file(manifest_path) != handoff["split_manifest_sha256"]:
        parser.error("public frozen split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids_path = split_root / "train.txt"
    if sha256_file(ids_path) != manifest["splits"]["train"]["sha256"]:
        parser.error("public frozen train ID hash mismatch")
    try:
        audit = stage_bundle(args.archive, args.output, expected_sha256=bundle["sha256"], kind=args.kind,
                             release=release, split_sha256=handoff["split_manifest_sha256"],
                             train_ids=set(ids_path.read_text(encoding="utf-8").splitlines()))
    except (ValueError, OSError, tarfile.TarError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
