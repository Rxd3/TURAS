"""Train-only input validation and a small shared acoustic optimization loop.

No model or dataset is downloaded on import or during preflight. Validation and
test references are intentionally unsupported: M6 evaluates saved checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

INFERENCE_FIELDS = frozenset({
    "schema_version", "dataset_release", "split_manifest_sha256", "utterance_id",
    "audio_uri", "audio_sha256", "duration_s", "sample_rate_hz", "channels", "language",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {number}")
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {number} must be an object")
            rows.append(row)
    return rows


def write_json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_audio_path(root: Path, uri: str) -> Path:
    if not isinstance(uri, str) or not uri or Path(uri).is_absolute() or ":" in uri:
        raise ValueError("audio_uri must be a portable relative file path")
    root = Path(root).resolve()
    path = (root / uri).resolve()
    if not path.is_relative_to(root):
        raise ValueError("audio_uri escapes the approved audio root")
    return path


def load_train_rows(
    inference: Path, references: Path, audio_root: Path, split_manifest: Path,
    train_ids: Path, *, release: str, split_sha256: str, references_sha256: str,
    inference_sha256: str, max_duration_s: float = 30.0,
) -> list[dict]:
    """Validate hashes/IDs before joining train-only references to audio metadata."""
    for path, expected in ((split_manifest, split_sha256), (references, references_sha256),
                           (inference, inference_sha256)):
        if sha256_file(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {Path(path).name}")
    manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    if manifest["dataset_release"] != release:
        raise ValueError("split manifest release mismatch")
    if sha256_file(train_ids) != manifest["splits"]["train"]["sha256"]:
        raise ValueError("frozen train ID file hash mismatch")
    ids = Path(train_ids).read_text(encoding="utf-8").splitlines()
    if not ids or ids != sorted(set(ids)):
        raise ValueError("train IDs must be nonempty, unique and sorted")
    if len(ids) != manifest["splits"]["train"]["utterances"]:
        raise ValueError("train ID count mismatch")
    inputs = read_jsonl(inference)
    refs = read_jsonl(references)
    if [r.get("utterance_id") for r in inputs] != ids:
        raise ValueError("inference rows must match the frozen train ID order exactly")
    ref_map = {}
    for row in refs:
        if row.get("split") != "train" or row.get("dataset_release") != release:
            raise ValueError("only matching train references are permitted")
        uid = row.get("utterance_id")
        if uid in ref_map:
            raise ValueError("duplicate reference utterance ID")
        if not isinstance(row.get("reference_text"), str) or not row["reference_text"].strip():
            raise ValueError("empty or missing training reference")
        if row.get("verification_status") in {"rejected", "in_review"}:
            raise ValueError("rejected or unresolved training reference")
        ref_map[uid] = row
    if set(ref_map) != set(ids):
        raise ValueError("reference IDs differ from frozen train IDs")
    joined = []
    for row in inputs:
        if set(row) != INFERENCE_FIELDS:
            raise ValueError("inference input violates the gold-free field allowlist")
        if row["dataset_release"] != release or row["split_manifest_sha256"] != split_sha256:
            raise ValueError("inference release/split hash mismatch")
        if row["language"] != "tr_tr" or row["sample_rate_hz"] != 16000 or row["channels"] != 1:
            raise ValueError("training requires Turkish, 16 kHz, mono audio")
        duration = row["duration_s"]
        if not isinstance(duration, (float, int)) or isinstance(duration, bool) or not 0 < duration <= max_duration_s:
            raise ValueError("audio requires an approved segmentation policy; no truncation is allowed")
        path = safe_audio_path(audio_root, row["audio_uri"])
        if sha256_file(path) != row["audio_sha256"]:
            raise ValueError("audio hash mismatch: " + row["utterance_id"])
        ref = ref_map[row["utterance_id"]]
        if ref.get("audio_sha256") != row["audio_sha256"]:
            raise ValueError("reference/audio identity mismatch")
        joined.append({**row, "reference_text": ref["reference_text"], "audio_path": path})
    return joined


def load_audio(row: dict):
    import soundfile as sf
    samples, rate = sf.read(row["audio_path"], dtype="float32", always_2d=True)
    if rate != 16000 or samples.shape[1] != 1 or not len(samples):
        raise ValueError("decoded audio is not nonempty 16 kHz mono")
    if abs(len(samples) / rate - row["duration_s"]) > 0.02:
        raise ValueError("decoded duration differs from manifest")
    return samples[:, 0]


def preflight(config: dict) -> dict:
    """Return concrete blockers without importing torch or touching protected data."""
    blockers = []
    for key in ("inference", "references", "audio_root", "split_manifest", "train_ids", "baseline_report"):
        value = config.get("inputs", {}).get(key)
        if not value or not Path(value).exists():
            blockers.append("missing input: " + key)
    for key in ("references_sha256", "inference_sha256"):
        value = config.get("inputs", {}).get(key)
        if not isinstance(value, str) or len(value) != 64:
            blockers.append("missing reviewed hash: " + key)
    if not config.get("protocol_reviewed", False):
        blockers.append("protocol has not been reviewed by the affected owners")
    if not config.get("baseline_reviewed", False):
        blockers.append("corresponding unmodified baseline has not been reviewed")
    return {"status": "blocked" if blockers else "ready_for_input_validation", "blockers": blockers}


def validate_baseline(config: dict) -> None:
    report = json.loads(Path(config["inputs"]["baseline_report"]).read_text(encoding="utf-8"))
    expected = config["model"]
    if (report.get("status") != "completed" or report.get("training_mode") != "pretrained"
            or report.get("model_id") != expected["id"] or report.get("revision") != expected["revision"]
            or report.get("dataset_release") != config["dataset_release"]
            or report.get("split_manifest_sha256") != config["split_manifest_sha256"]):
        raise ValueError("baseline evidence is incompatible with the proposed adaptation")


def fit(model, rows: list[dict], collate, *, epochs: int, accumulation: int,
        learning_rate: float, seed: int, device: str, precision: str, on_epoch):
    """Batch-one AdamW training; correctly scales the final partial accumulation."""
    import torch
    if not rows or epochs < 1 or accumulation < 1 or learning_rate <= 0:
        raise ValueError("invalid training budget")
    if precision not in {"fp32", "bf16"} or (precision == "bf16" and device != "cuda"):
        raise ValueError("use fp32 on CPU or supported bf16 on CUDA")
    if device == "cuda" and (not torch.cuda.is_available() or
                              (precision == "bf16" and not torch.cuda.is_bf16_supported())):
        raise ValueError("requested CUDA precision is unavailable")
    random.seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    history = []
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(rows)))
        rng.shuffle(order)
        total_loss = 0.0
        steps = 0
        for start in range(0, len(order), accumulation):
            group = order[start:start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                batch = {k: v.to(device) for k, v in collate([rows[index]]).items()}
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=precision == "bf16"):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss; checkpoint not exported")
                total_loss += float(loss.detach())
                (loss / len(group)).backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
        facts = {"epoch": epoch, "train_loss": total_loss / len(rows), "optimizer_steps": steps}
        on_epoch(epoch, facts)
        history.append(facts)
    return history


def run_training(config: dict, output: Path, build_model, build_collator) -> dict:
    """Validate every external input before model loading or checkpoint creation."""
    started_at = datetime.now(timezone.utc).isoformat()
    report = preflight(config)
    if report["blockers"]:
        raise ValueError("training blocked: " + "; ".join(report["blockers"]))
    if output.exists():
        raise ValueError("output must be a new directory; existing runs are never overwritten")
    validate_baseline(config)
    inputs = config["inputs"]
    rows = load_train_rows(
        *(Path(inputs[k]) for k in ("inference", "references", "audio_root", "split_manifest", "train_ids")),
        release=config["dataset_release"], split_sha256=config["split_manifest_sha256"],
        references_sha256=inputs["references_sha256"], inference_sha256=inputs["inference_sha256"],
    )
    import torch
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    model, processor = build_model(config)
    collate = build_collator(processor)
    # Validate all targets before the first optimizer step; no hidden row dropping.
    for row in rows:
        collate([row])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "resolved-config.json", config)
    try:
        code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "diff", "HEAD"], text=True)
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("training must run from a versioned checkout") from None
    import importlib.metadata
    import sys
    environment = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    write_json(output / "environment.json", {"python": sys.version, "packages": environment})
    manifest = {"schema_version": "0.1", "run_id": "acoustic-" + uuid.uuid4().hex,
                "started_at_utc": started_at, "status": "running", "dataset_release": config["dataset_release"],
                "split_manifest_sha256": config["split_manifest_sha256"], "model": config["model"],
                "training_mode": config["mode"], "code_commit": code_commit,
                "dirty_diff_sha256": hashlib.sha256(dirty.encode()).hexdigest(), "torch": torch.__version__,
                "config_sha256": sha256_file(output / "resolved-config.json"),
                "environment_sha256": sha256_file(output / "environment.json"),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in model.parameters()), "checkpoints": [],
                "selection": "pending M6 validation; no checkpoint is called best automatically"}
    write_json(output / "run.json", manifest)

    def checkpoint(epoch, facts):
        path = output / f"epoch-{epoch}"
        model.save_pretrained(path, safe_serialization=True)
        processor.save_pretrained(path)
        files = {str(p.relative_to(path)): sha256_file(p) for p in sorted(path.rglob("*")) if p.is_file()}
        item = {**facts, "directory": path.name, "files": files, "parent": config["model"],
                "training_mode": config["mode"], "config_sha256": manifest["config_sha256"],
                "dataset_release": config["dataset_release"],
                "split_manifest_sha256": config["split_manifest_sha256"]}
        write_json(path / "checkpoint-manifest.json", item)
        manifest["checkpoints"].append(item)
        write_json(output / "run.json", manifest)

    try:
        manifest["history"] = fit(model, rows, collate, on_epoch=checkpoint, **config["training"])
        manifest["status"] = "training_completed_selection_pending"
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", manifest)
        raise
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "run.json", manifest)
    return manifest


def training_main(build_model, build_collator):
    import argparse
    parser = argparse.ArgumentParser(description="Train-only acoustic adaptation; M6 owns validation/test scoring")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.preflight:
        print(json.dumps(preflight(config), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for training")
    try:
        result = run_training(config, args.output, build_model, build_collator)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps({"status": result["status"]}))
