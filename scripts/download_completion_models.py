"""Download exact baseline snapshots and hash every consumed model file."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "b2": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1", "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95"),
    "qwen17": ("Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3", "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1"),
    "qwen06": ("Qwen/Qwen3-ASR-0.6B-hf", "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c", "d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35"),
    "seda": ("atasoglu/seda-v0.1", "5cbbda1b34bde5e3004e79b137dc3871587ea282", None),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def inventory_is_verified(directory, inventory, model, revision, weight_sha):
    if not isinstance(inventory, dict) or inventory.get("model_id") != model or inventory.get("revision") != revision:
        return False
    files = inventory.get("files")
    required = ({"model.safetensors", "config.json", "tokenizer.json"} if weight_sha else
                {"encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"})
    if not isinstance(files, dict) or not required.issubset(files):
        return False
    if weight_sha and files["model.safetensors"].get("sha256") != weight_sha:
        return False
    for name, info in files.items():
        if not isinstance(info, dict) or not name or name in {".", ".."} or any(c in name for c in "/\\:"):
            return False
        path = directory / name
        # Delivered neural packages intentionally omit large upstream weights.
        # Missing files mean "download needed", not an invalid completed run.
        if not path.is_file() or path.stat().st_size != info.get("bytes") or sha(path) != info.get("sha256"):
            return False
    return True


def main():
    from huggingface_hub import snapshot_download
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = p.parse_args()
    base = ROOT / "artifacts/completion-2026-11-22/models"
    for name in args.models:
        model, revision, weight_sha = MODELS[name]
        out = base / name
        if (out / "verified-inventory.json").exists():
            inventory = json.loads((out / "verified-inventory.json").read_text())
            if inventory_is_verified(out, inventory, model, revision, weight_sha):
                print(f"ALREADY VERIFIED {name}", flush=True)
                continue
        patterns = ["*.json", "model.safetensors", "*.txt", "*.jinja", "README.md"]
        prefix = out / "model.safetensors.resume"
        if name == "b2" and prefix.exists():
            patterns.remove("model.safetensors")
        if name == "seda":
            patterns = ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt", "README.md"]
        print(f"DOWNLOAD {name}: {model}@{revision}", flush=True)
        for attempt in range(3):
            try:
                snapshot_download(repo_id=model, revision=revision, local_dir=out, allow_patterns=patterns,
                                  ignore_patterns=["*fp32*"], max_workers=3)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)
        if name == "b2" and prefix.exists():
            expected_size = 3087130976
            while prefix.stat().st_size < expected_size:
                offset = prefix.stat().st_size
                url = f"https://huggingface.co/{model}/resolve/{revision}/model.safetensors?download=true&resume_offset={offset}"
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(request, timeout=60) as response:
                            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                                raise ValueError("Pinned weight server did not honor the resume range")
                            with prefix.open("ab") as f:
                                while True:
                                    data = response.read(1024 * 1024)
                                    if not data:
                                        break
                                    f.write(data)
                                    if f.tell() // (128 * 1024 * 1024) != (f.tell() - len(data)) // (128 * 1024 * 1024):
                                        print(f"RESUME b2 {f.tell()}/{expected_size} bytes", flush=True)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(3)
                        # Restart the HTTP range from the durable prefix after partial progress.
                        offset = prefix.stat().st_size
                        url = f"https://huggingface.co/{model}/resolve/{revision}/model.safetensors?download=true&resume_offset={offset}"
                        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
            if prefix.stat().st_size != expected_size or sha(prefix) != weight_sha:
                raise ValueError("Resumed pinned B2 weight failed full size/SHA verification")
            prefix.replace(out / "model.safetensors")
        files = {path.relative_to(out).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)}
                 for path in sorted(out.iterdir())
                 if path.is_file() and path.name != "verified-inventory.json" and "fp32" not in path.name}
        if weight_sha and files["model.safetensors"]["sha256"] != weight_sha:
            raise ValueError(f"Pinned weight hash mismatch: {name}")
        (out / "verified-inventory.json").write_text(json.dumps({"model_id": model, "revision": revision,
            "files": files, "language_model_downloaded": False if name == "seda" else None}, indent=2) + "\n", encoding="utf-8")
        print(f"VERIFIED {name}: {sum(d['bytes'] for d in files.values())/1e9:.3f} GB", flush=True)


if __name__ == "__main__":
    main()
