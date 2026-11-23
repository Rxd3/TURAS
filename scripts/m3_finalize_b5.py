"""Frozen B5 validation/test inference, shared timing and offline gold joins."""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch

from turkish_asr_rdl.baselines.bilstm_ctc import BiLstmCtcBaselineAdapter
from turkish_asr_rdl.decoding.export import RunContext
from turkish_asr_rdl.decoding.interface import InferenceInput
from turkish_asr_rdl.decoding.timing import run_timed, write_run_outputs
from turkish_asr_rdl.evaluation.join import join
from turkish_asr_rdl.evaluation.metrics import corpus_rates, score_pair
from turkish_asr_rdl.evaluation.normalize import RAW_V1, TR_NORMALIZED_V1
from turkish_asr_rdl.schemas.jsonl import read_jsonl, write_json
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.schemas.predictions import validate_predictions

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "turas-tr-econ-legis-v0.1"
PRIVATE = ROOT / "data/private" / RELEASE
AUDIO = PRIVATE / "inference" / RELEASE
SPLITS = ROOT / "data/splits" / RELEASE
PARENT = ROOT / "artifacts/b5-gpu-resume-2026-11-22"
OUT = ROOT / "artifacts/m3-final-2026-11-22/b5"
CHECKPOINT = PARENT / "best-checkpoint"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(split):
    ids = (SPLITS / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    manifest = json.loads((SPLITS / "split_manifest.json").read_text())
    if sha(SPLITS / f"{split}.txt") != manifest["splits"][split]["sha256"]:
        raise ValueError("Frozen split hash mismatch")
    rows = read_jsonl(AUDIO / f"{RELEASE}.inference.{split}.jsonl")
    if [r["utterance_id"] for r in rows] != ids:
        raise ValueError("Frozen inference IDs/order mismatch")
    return [InferenceInput.from_row(r) for r in rows]


class ProgressB5(BiLstmCtcBaselineAdapter):
    completed = 0

    def transcribe(self, batch):
        results = super().transcribe(batch)
        self.completed += len(batch)
        if self.completed % 25 == 0:
            print(f"B5 INFERENCE {self.completed} items including warm-up", flush=True)
        return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "summary.json").exists():
        print("B5 final evaluation already completed", flush=True)
        return
    write_json(OUT / "status.json", {"state": "running", "started_at_utc": now()})
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    train_status = json.loads((PARENT / "status.json").read_text())
    if train_status["state"] != "completed" or train_status["best_epoch"] != 22:
        raise ValueError("Expected completed, frozen epoch-22 training")
    checkpoint_sha = sha(CHECKPOINT / "checkpoint-manifest.json")
    cfg = json.loads((ROOT / "configs/baselines/bilstm_ctc.json").read_text())
    warmup = inputs("train")[:3]
    split_sha = sha(SPLITS / "split_manifest.json")
    frozen = {"schema_version": "0.1", "state": "frozen_before_test_scoring", "frozen_at_utc": now(),
              "system_id": "baseline-bilstm-ctc", "best_epoch": 22, "checkpoint_manifest_sha256": checkpoint_sha,
              "dataset_release": RELEASE, "split_manifest_sha256": split_sha, "config": cfg,
              "script_sha256": sha(__file__), "decoder": "greedy_ctc", "requested_n": 1,
              "timing": {"subset": "m45-timing-30min.txt", "repeats": 3, "batch_size": 1, "warmup_batches": 3},
              "limitations": ["Single training seed", "Working parliamentary minutes, not fully verified verbatim gold",
                              "GMM-HMM is a separate stage; no B5 settings selected from test scores"]}
    write_json(OUT / "frozen-protocol.json", frozen)
    completed = {}
    for mode, split in [("validation", "validation"), ("test", "test"), ("timing", "test")]:
        directory = OUT / mode
        if (directory / "completed.json").exists():
            done = json.loads((directory / "completed.json").read_text())
            for name, digest in done["files"].items():
                if sha(directory / name) != digest:
                    raise ValueError("Existing B5 output hash mismatch")
            completed[mode] = done
            continue
        rows = inputs(split)
        if mode == "timing":
            ids = (SPLITS / "m45-timing-30min.txt").read_text().splitlines()
            by_id = {i.utterance_id: i for i in rows}
            rows = [by_id[uid] for uid in ids]
        else:
            ids = [i.utterance_id for i in rows]
        adapter = ProgressB5(cfg, checkpoint_dir=CHECKPOINT, model_artifact_sha256=checkpoint_sha, device="cuda")
        run_id = f"m3-b5-epoch22-{mode}-20261122"
        print(f"B5 START {mode}: {len(rows)} clips", flush=True)
        write_json(OUT / "status.json", {"state": "inference", "mode": mode, "rows": len(rows), "updated_at_utc": now()})
        run = run_timed(adapter, rows, warmup, audio_root=AUDIO, run_id=run_id,
                        repeats=3 if mode == "timing" else 1, batch_size=1)
        context = RunContext(run_id, adapter.system_id, RELEASE, split, split_sha,
                             adapter.model_artifact_sha256, adapter.decoder_config_sha256)
        files = write_run_outputs(directory, run, rows, context, adapter, command="scripts/m3_finalize_b5.py")
        for validator, name in [(validate_predictions, "predictions.jsonl"), (validate_nbest, "nbest.jsonl")]:
            report = validator(read_jsonl(directory / name), ids)
            if not report.ok:
                raise ValueError(f"{name}: {report.errors}")
        errors = sum(h.status == "error" for h in run.hypotheses)
        if errors:
            raise ValueError(f"B5 {mode}: {errors} inference errors")
        done = {"state": "completed", "rows": len(rows), "files": files,
                "rtf": run.end_to_end_rtf, "repeats": 3 if mode == "timing" else 1,
                "thermal_flags": run.thermal_flags, "vram_spill_suspected": run.vram.vram_spill_suspected,
                "timing_claim": "headline 30-minute subset" if mode == "timing" else "accuracy inference only"}
        write_json(directory / "completed.json", done)
        completed[mode] = done
        print(f"B5 COMPLETE {mode}: {len(rows)} rows, zero errors, RTF={run.end_to_end_rtf:.5f}", flush=True)
        del adapter, run
        gc.collect()
        torch.cuda.empty_cache()
    # Protected references are read only after all B5 inference and timing are frozen.
    results = {}
    refs_paths = {"validation": PRIVATE / "validation-recovered-minutes-v1/references.validation.jsonl",
                  "test": PRIVATE / "evaluator-final/references.test.jsonl"}
    for split, path in refs_paths.items():
        print(f"B5 SCORING {split}", flush=True)
        write_json(OUT / "status.json", {"state": "scoring", "split": split, "updated_at_utc": now()})
        predictions = read_jsonl(OUT / split / "predictions.jsonl")
        joined = join(predictions, read_jsonl(path), split=split, split_manifest_sha256=split_sha)
        scores = {}
        for view in [RAW_V1, TR_NORMALIZED_V1]:
            scores[view] = dataclasses.asdict(corpus_rates([
                score_pair(r.utterance_id, r.reference, r.hypothesis, view, r.status) for r in joined]))
        results[split] = {"rows": len(joined), "reference_sha256": sha(path), "metrics": scores}
        write_json(OUT / f"metrics.{split}.json", results[split])
        print(f"B5 {split} WER={scores[TR_NORMALIZED_V1]['wer']:.6f} CER={scores[TR_NORMALIZED_V1]['cer']:.6f}", flush=True)
    summary = {"state": "completed", "completed_at_utc": now(), "system_id": "baseline-bilstm-ctc",
               "best_epoch": 22, "checkpoint_manifest_sha256": checkpoint_sha, "scores": results,
               "headline_timing": completed["timing"], "limitations": frozen["limitations"]}
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "status.json", {"state": "completed", "ended_at_utc": now()})
    print("B5 FINAL EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "status.json", {"state": "failed", "error": str(exc), "at_utc": now()})
        traceback.print_exc()
        sys.exit(1)
