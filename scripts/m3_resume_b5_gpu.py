"""Continue the M3 epoch-13 handoff on CUDA using the frozen train/validation IDs.

Private features, checkpoints and run evidence stay in --output. No test data is
read. A latest training checkpoint includes optimizer, scheduler and RNG states.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import torch
import torch.nn.functional as F

from turkish_asr_rdl.baselines.bilstm_ctc import save_bilstm_checkpoint
from turkish_asr_rdl.decoding.audio import load_audio, resolve_audio_path
from turkish_asr_rdl.evaluation.metrics import corpus_rates, score_pair
from turkish_asr_rdl.evaluation.normalize import TR_NORMALIZED_V1, normalize
from turkish_asr_rdl.models.bilstm_ctc import BiLSTMCTC, CharCtcTokenizer, LogMelSpectrogramExtractor
from turkish_asr_rdl.schemas.jsonl import read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "turas-tr-econ-legis-v0.1"
PRIVATE = ROOT / "data/private" / RELEASE
INFER = PRIVATE / "inference" / RELEASE
SPLITS = ROOT / "data/splits" / RELEASE
TRAIN_REFS = PRIVATE / "train" / RELEASE / f"{RELEASE}.references.train.jsonl"
VAL_REFS = PRIVATE / "validation-recovered-minutes-v1/references.validation.jsonl"
EXPECTED_PARENT = "3ce85052d0e2b8ad63c07647d9911ccdb2cb4264baa1737e95c38a514d38a683"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(payload, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def frozen_inputs(split, ref_path):
    ids = (SPLITS / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    rows = read_jsonl(INFER / f"{RELEASE}.inference.{split}.jsonl")
    refs = read_jsonl(ref_path)
    if len({r["utterance_id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate inference IDs in {split}")
    if len({r["utterance_id"] for r in refs}) != len(refs):
        raise ValueError(f"Duplicate reference IDs in {split}")
    by_id = {r["utterance_id"]: r for r in rows}
    ref_map = {r["utterance_id"]: r["reference_text"] for r in refs}
    if set(ids) != set(by_id) or set(ids) != set(ref_map):
        raise ValueError(f"Frozen {split} IDs do not exactly match inputs")
    manifest = json.loads((SPLITS / "split_manifest.json").read_text(encoding="utf-8"))
    if sha(SPLITS / f"{split}.txt") != manifest["splits"][split]["sha256"]:
        raise ValueError(f"Frozen {split} file hash mismatch")
    return [by_id[uid] for uid in ids], ref_map


def prepare_source(archive, out):
    source = out / "source"
    source.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        manifest_bytes = z.read("checkpoint-manifest.json")
        if hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_PARENT:
            raise ValueError("Unexpected parent checkpoint manifest")
        manifest = json.loads(manifest_bytes)
        for name in ["bilstm_ctc.pt", "vocab.json", "checkpoint-manifest.json", "train_b5_and_gmmhmm.py"]:
            data = z.read(name)
            expected = manifest["files"].get(name)
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {name}")
            target = source / name
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"Refusing to replace different source: {target}")
            if not target.exists():
                target.write_bytes(data)
    return source, manifest


def cache_features(rows, split, cache, extractor):
    directory = cache / split
    directory.mkdir(parents=True, exist_ok=True)

    def extract(row):
        path = directory / (row["utterance_id"] + ".pt")
        if not path.exists():
            audio = resolve_audio_path(INFER, row["audio_uri"])
            if sha(audio) != row["audio_sha256"]:
                raise ValueError(f"Audio hash mismatch: {row['utterance_id']}")
            samples = load_audio(audio)
            with torch.inference_mode():
                features, _ = extractor(torch.from_numpy(samples))
            atomic_save({"audio_sha256": row["audio_sha256"], "log_mel": features[0].half()}, path)
        return path

    entries = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (row, path) in enumerate(zip(rows, pool.map(extract, rows)), 1):
            entries.append({**row, "feat_path": str(path)})
            if i % 250 == 0 or i == len(rows):
                print(f"CACHE {split}: {i}/{len(rows)}", flush=True)
    return entries


def load_features(entry):
    payload = torch.load(entry["feat_path"], weights_only=True)
    if payload["audio_sha256"] != entry["audio_sha256"]:
        raise ValueError("Cached audio hash mismatch")
    return payload["log_mel"].float()


def evaluate(model, entries, refs, tokenizer):
    model.eval()
    scores = []
    with torch.inference_mode():
        for i, entry in enumerate(entries, 1):
            f = load_features(entry).unsqueeze(0).cuda()
            logits, _ = model(f, torch.tensor([f.shape[1]], device="cuda"))
            hyp = tokenizer.decode(logits[0].argmax(-1).cpu().tolist())
            scores.append(score_pair(entry["utterance_id"], refs[entry["utterance_id"]], hyp, TR_NORMALIZED_V1))
            if i % 100 == 0:
                print(f"VALIDATION {i}/{len(entries)}", flush=True)
    return dataclasses.asdict(corpus_rates(scores))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--max-hours", type=float, default=2.0)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    status = {"state": "starting", "started_at_utc": now(), "pid": os.getpid()}
    write_json(status_path, status)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
        torch.set_num_threads(1)
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        source, original_manifest = prepare_source(args.source_zip, out)
        parent = torch.load(source / "bilstm_ctc.pt", map_location="cpu", weights_only=True)
        tokenizer = CharCtcTokenizer.load(source / "vocab.json")
        train, train_refs = frozen_inputs("train", TRAIN_REFS)
        val, val_refs = frozen_inputs("validation", VAL_REFS)
        norm_refs = {uid: normalize(text, TR_NORMALIZED_V1) for uid, text in train_refs.items()}
        fitted = CharCtcTokenizer.fit_from_train_texts(norm_refs.values())
        if tokenizer.to_dict() != fitted.to_dict():
            raise ValueError("Handoff vocabulary differs from this frozen training reference inventory")
        model = BiLSTMCTC(**parent["arch"])
        model.load_state_dict(parent["state_dict"], strict=True)
        model.cuda()
        cfg = json.loads((ROOT / "configs/baselines/bilstm_ctc.json").read_text(encoding="utf-8"))
        command = subprocess.list2cmdline([sys.executable, *sys.argv])
        manifest = {
            "schema_version": "0.1", "run_id": out.name, "state": "running", "started_at_utc": now(),
            "owner_member": "M3 local GPU continuation", "purpose": "Continue epoch-13 B5 weights on CUDA",
            "parent_manifest_sha256": EXPECTED_PARENT, "parent_weights_sha256": sha(source / "bilstm_ctc.pt"),
            "dataset_release": RELEASE, "split_manifest_sha256": sha(SPLITS / "split_manifest.json"),
            "split_file_sha256": {s: sha(SPLITS / f"{s}.txt") for s in ["train", "validation"]},
            "reference_sha256": {"train": sha(TRAIN_REFS), "validation": sha(VAL_REFS)},
            "validation_reference_variant": "owner-recovered-validation-minutes-v1",
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            "script_sha256": sha(Path(__file__)), "original_script_sha256": sha(source / "train_b5_and_gmmhmm.py"),
            "command": command, "environment": {"os": platform.platform(), "python": sys.version,
                "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(), "cpu": platform.processor(), "cpu_threads": 1,
                "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()},
            "dependency_lock_sha256": sha(ROOT / "requirements/windows-cu130.txt"),
            "model_arch": parent["arch"], "vocab_sha256": sha(source / "vocab.json"),
            "model_code_sha256": sha(ROOT / "src/turkish_asr_rdl/models/bilstm_ctc.py"),
            "normalizer_sha256": sha(ROOT / "src/turkish_asr_rdl/evaluation/normalize.py"),
            "metrics_sha256": sha(ROOT / "src/turkish_asr_rdl/evaluation/metrics.py"),
            "aligner_sha256": sha(ROOT / "src/turkish_asr_rdl/evaluation/align.py"),
            "optimizer": {"type": "AdamW", "lr": 3e-4, "weight_decay": cfg["training_protocol"]["weight_decay"], "fresh_on_parent_resume": True},
            "scheduler": {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": 2},
            "seed": 42, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "precision": "float32; cached features round-tripped through float16 as in M3 script",
            "batch_size": 16, "max_epochs": args.max_epochs, "early_stopping_patience": args.patience,
            "wall_clock_limit_hours": args.max_hours, "selection_metric": "validation norm CER",
            "limits": ["Single training seed", "Original optimizer/scheduler/RNG not supplied; fresh AdamW continuation",
                "Working validation minutes are unreviewed; not a verified verbatim gold reference",
                "Original M3 reference hash was not supplied; parent is re-evaluated on this pinned local snapshot",
                "No test references or test inference inputs read; no inference timing or GMM-HMM run"]}
        write_json(out / "run-manifest.json", manifest)
        print(f"GPU {torch.cuda.get_device_name()} | parent epoch 13 | {len(train)} train / {len(val)} validation", flush=True)
        if args.preflight_only:
            status.update(state="preflight_passed", ended_at_utc=now())
            write_json(status_path, status)
            return
        start = time.perf_counter()
        deadline = start + args.max_hours * 3600
        extractor = LogMelSpectrogramExtractor()
        cache = out / "feature-cache"
        cache.mkdir(exist_ok=True)
        status.update(state="caching_features")
        write_json(status_path, status)
        val = cache_features(val, "validation", cache, extractor)
        history_path = out / "training-history.json"
        latest = out / "latest-training.pt"
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=cfg["training_protocol"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        if latest.exists():
            saved = torch.load(latest, map_location="cpu", weights_only=True)
            model.load_state_dict(saved["state_dict"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            random.setstate(saved["python_rng"])
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
            history = saved["history"]
            best_cer, best_epoch, stale = saved["best_cer"], saved["best_epoch"], saved["stale"]
            first_epoch = saved["epoch"] + 1
            print(f"RESUME full local training state after epoch {saved['epoch']}", flush=True)
        else:
            status.update(state="evaluating_parent")
            write_json(status_path, status)
            baseline = evaluate(model, val, val_refs, tokenizer)
            best_cer, best_epoch, stale = baseline["cer"], 13, 0
            history = [{"epoch": 13, "parent_re_evaluation": True, "validation": baseline}]
            first_epoch = 14
            write_json(out / "parent-validation.json", baseline)
            print(f"PARENT epoch 13: CER={baseline['cer']:.6f} WER={baseline['wer']:.6f}", flush=True)
            (out / "best-checkpoint").mkdir(exist_ok=True)
            for name in ["bilstm_ctc.pt", "vocab.json", "checkpoint-manifest.json"]:
                shutil.copy2(source / name, out / "best-checkpoint" / name)
        train = cache_features(train, "train", cache, extractor)
        for entry in train:
            entry["target_ids"] = tokenizer.encode(norm_refs[entry["utterance_id"]])
            frames = load_features(entry).shape[0]
            out_frames = ((frames + 1) // 2 + 1) // 2
            targets = entry["target_ids"]
            required = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
            if not targets or out_frames < required:
                raise ValueError(f"Infeasible frozen training clip: {entry['utterance_id']}")
        # Canonical order before each shuffle makes an epoch-boundary restart reproducible.
        stop_reason = "max_epochs"
        for epoch in range(first_epoch, args.max_epochs + 1):
            if time.perf_counter() >= deadline:
                stop_reason = "wall_clock_limit"
                break
            t_epoch = time.perf_counter()
            status.update(state="training", epoch=epoch, best_epoch=best_epoch, best_val_norm_cer=best_cer)
            write_json(status_path, status)
            model.train()
            train.sort(key=lambda x: x["utterance_id"])
            random.shuffle(train)
            train.sort(key=lambda x: x["duration_s"] + random.uniform(-0.5, 0.5))
            losses = []
            torch.cuda.reset_peak_memory_stats()
            for step, offset in enumerate(range(0, len(train), 16), 1):
                batch = train[offset:offset + 16]
                feats = [load_features(b) for b in batch]
                lengths = torch.tensor([f.shape[0] for f in feats], device="cuda")
                padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True).cuda()
                targets = torch.tensor([t for b in batch for t in b["target_ids"]], dtype=torch.long, device="cuda")
                target_lengths = torch.tensor([len(b["target_ids"]) for b in batch], device="cuda")
                optimizer.zero_grad(set_to_none=True)
                logits, out_lengths = model(padded, lengths)
                loss = F.ctc_loss(F.log_softmax(logits.float(), -1).transpose(0, 1), targets,
                                  out_lengths, target_lengths, blank=0, zero_infinity=True)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch {epoch}, step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(norm):
                    raise RuntimeError(f"Nonfinite gradient at epoch {epoch}, step {step}")
                optimizer.step()
                losses.append(loss.item())
                if step % 25 == 0 or offset + 16 >= len(train):
                    print(f"TRAIN epoch={epoch} step={step}/{math.ceil(len(train)/16)} loss={sum(losses)/len(losses):.5f}", flush=True)
            status.update(state="validating", epoch=epoch)
            write_json(status_path, status)
            validation = evaluate(model, val, val_refs, tokenizer)
            improved = validation["cer"] < best_cer
            if improved:
                best_cer, best_epoch, stale = validation["cer"], epoch, 0
                save_bilstm_checkpoint(out / "best-checkpoint", model, tokenizer, metadata={
                    "best_epoch": epoch, "best_val_norm_cer": best_cer, "best_val_norm_wer": validation["wer"],
                    "parent_manifest_sha256": EXPECTED_PARENT, "run_manifest": "../run-manifest.json",
                    "reference_variant": manifest["validation_reference_variant"]})
            else:
                stale += 1
            scheduler.step(validation["cer"])
            record = {"epoch": epoch, "train_loss": sum(losses)/len(losses), "validation": validation,
                      "epoch_seconds": time.perf_counter()-t_epoch, "peak_vram_gib": torch.cuda.max_memory_allocated()/2**30,
                      "lr_next_epoch": optimizer.param_groups[0]["lr"], "improved": improved}
            history.append(record)
            atomic_save({"state_dict": model.state_dict(), "arch": parent["arch"], "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "epoch": epoch, "best_cer": best_cer,
                         "best_epoch": best_epoch, "stale": stale, "history": history,
                         "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all()}, latest)
            write_json(history_path, {"history": history, "best_epoch": best_epoch, "best_val_norm_cer": best_cer})
            status.update(state="epoch_completed", epoch=epoch, best_epoch=best_epoch, best_val_norm_cer=best_cer,
                          last_epoch_seconds=record["epoch_seconds"], updated_at_utc=now())
            write_json(status_path, status)
            print(f"EPOCH {epoch} | CER={validation['cer']:.6f} WER={validation['wer']:.6f} | {record['epoch_seconds']:.1f}s | best={best_epoch} | stale={stale}/{args.patience}", flush=True)
            if stale >= args.patience:
                stop_reason = "early_stopping"
                break
        best = out / "best-checkpoint"
        manifest.update(state="completed", ended_at_utc=now(), best_epoch=best_epoch, best_val_norm_cer=best_cer,
                        stop_reason=stop_reason, selected_checkpoint_frozen_at_utc=now(),
                        outputs={name: {"sha256": sha(best / name), "bytes": (best / name).stat().st_size}
                                 for name in ["bilstm_ctc.pt", "vocab.json", "checkpoint-manifest.json"]})
        write_json(out / "run-manifest.json", manifest)
        status.update(state="completed", stop_reason=stop_reason, best_epoch=best_epoch, best_val_norm_cer=best_cer, ended_at_utc=now())
        write_json(status_path, status)
        print(f"COMPLETE {stop_reason}: best epoch {best_epoch}, CER={best_cer:.6f}; {best}", flush=True)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=now())
        write_json(status_path, status)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
