"""Restore authorized role packages to a fresh checkout, refusing conflicting bytes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile

RELEASE = "turas-tr-econ-legis-v0.1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def safe_target(root, name):
    parts = PurePosixPath(name)
    if "\\" in name or ":" in name or parts.is_absolute() or ".." in parts.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Archive member escapes destination")
    return target


def write_member(root, name, source):
    target = safe_target(root, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".restore-tmp")
    with temporary.open("wb") as f:
        shutil.copyfileobj(source, f)
    if target.exists():
        if sha(target) != sha(temporary):
            temporary.unlink()
            raise ValueError(f"Refusing to replace different bytes: {target}")
        temporary.unlink()
    else:
        temporary.replace(target)


def unpack(archive, root, *, preserve_m45_metadata=False):
    root.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                safe_target(root, info.filename)
                if not info.is_dir():
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("Symbolic links are not accepted")
                    with z.open(info) as f:
                        write_member(root, info.filename, f)
    else:
        with tarfile.open(archive) as t:
            for info in t:
                safe_target(root, info.name)
                if info.isdir():
                    continue
                if not info.isfile():
                    raise ValueError("Only regular files/directories are accepted")
                name = info.name
                if preserve_m45_metadata and not name.startswith("artifacts/"):
                    name = "artifacts/owner-m45/delivery-metadata/" + name
                with t.extractfile(info) as f:
                    write_member(root, name, f)


def find_payload(root, relative):
    parts = Path(relative).parts
    starts = [0] + [i for i in range(1, len(parts)) if root.name == parts[i - 1]]
    for start in starts:
        path = root.joinpath(*parts[start:])
        if path.is_file():
            return path
    raise FileNotFoundError(relative)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--packages", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--roles", required=True, help="Comma-separated inference,train,validation,evaluator,m3,m45,b2,b3,transducer,gmm")
    args = p.parse_args()
    root = args.workspace.resolve()
    if not (root / "pyproject.toml").is_file():
        p.error("workspace must be an extracted source snapshot or checkout")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    roles = set(args.roles.split(","))
    if roles - {"inference", "train", "validation", "evaluator", "m3", "m45", "b2", "b3", "transducer", "gmm"}:
        p.error("Unknown role")
    selected = [r for r in manifest["files"] if r.get("restore_role") in roles]
    if not selected:
        p.error("No matching packages in manifest")
    missing_roles = roles - {r["restore_role"] for r in selected}
    if missing_roles:
        p.error("No payloads for requested roles: " + ", ".join(sorted(missing_roles)))
    # Verify all requested payloads before the first write.
    located = [(r, find_payload(args.packages, r["path"])) for r in selected]
    for record, source in located:
        if source.stat().st_size != record["bytes"] or sha(source) != record["sha256"]:
            raise ValueError(f"Package checksum mismatch: {source.name}")
    private = root / "data/private" / RELEASE
    for record, source in located:
        action = record.get("restore_action")
        if action == "archive":
            unpack(source, root / record["restore_target"], preserve_m45_metadata=record["restore_role"] == "m45")
        elif action == "copy":
            with source.open("rb") as f:
                write_member(root, record["restore_target"], f)
        print(f"Restored {source.name}")
    if "validation" in roles:
        source = private / "validation-import" / RELEASE / f"{RELEASE}.references.validation.jsonl"
        with source.open("rb") as f:
            write_member(private, "validation-recovered-minutes-v1/references.validation.jsonl", f)
    if "evaluator" in roles:
        source = private / "evaluator-import" / RELEASE / f"{RELEASE}.references.test.jsonl"
        with source.open("rb") as f:
            write_member(private, "evaluator-final/references.test.jsonl", f)
    if "m3" in roles:
        out = root / "artifacts/m3-final-2026-11-22"
        checkpoint = root / "artifacts/b5-gpu-resume-2026-11-22/best-checkpoint"
        for name in ["bilstm_ctc.pt", "vocab.json", "checkpoint-manifest.json"]:
            with (out / "b5/checkpoint" / name).open("rb") as f:
                write_member(checkpoint, name, f)
        for name in ["run-manifest.json", "training-history.json", "parent-validation.json"]:
            with (out / "b5/training" / name).open("rb") as f:
                write_member(checkpoint.parent, name, f)
    print("Restore complete. References remain outside the inference/audio directory.")


if __name__ == "__main__":
    main()
