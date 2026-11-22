"""Standard-library checksum verification for a complete or role-limited delivery."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--allow-missing", action="store_true")
    args = p.parse_args()
    manifest_path = args.manifest or args.root / "TURAS-Public-Submission/00-START-HERE/package-manifest.json"
    if not manifest_path.exists():
        p.error("Supply the downloaded --manifest path")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified, missing = [], []
    for record in manifest["files"]:
        parts = Path(record["path"]).parts
        if ".." in parts or Path(record["path"]).is_absolute():
            raise ValueError("Unsafe manifest path")
        starts = [0] + [i for i in range(1, len(parts)) if args.root.name == parts[i - 1]]
        candidates = [args.root.joinpath(*parts[i:]) for i in starts]
        file = next((f for f in candidates if f.is_file()), None)
        if file is None:
            missing.append(record["path"])
            continue
        if file.stat().st_size != record["bytes"] or sha(file) != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {record['path']}")
        verified.append(record["path"])
    if not verified or (missing and not args.allow_missing):
        raise ValueError(f"Verified {len(verified)} payloads; missing {len(missing)}. Use --allow-missing only for role-limited receipt.")
    print(json.dumps({"status": "passed", "verified_payloads": len(verified), "missing_payloads": len(missing),
                      "verification_scope": "received file bytes only; not cloud access or independent reproduction"}, indent=2))


if __name__ == "__main__":
    main()
