"""Run the supplied fixed three-pass character Gaussian-HMM baseline and evaluate."""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import gc
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from m3_finalize_b5 import AUDIO, PRIVATE, RELEASE, ROOT, SPLITS, inputs, now, sha
from m3_resume_b5_gpu import frozen_inputs
from turkish_asr_rdl.baselines.m3_gaussian_hmm import FEATURE_RECIPE, GaussianHmmAdapter, M3Features, train
from turkish_asr_rdl.decoding.audio import load_audio, resolve_audio_path
from turkish_asr_rdl.decoding.export import RunContext
from turkish_asr_rdl.decoding.memory import NoVram
from turkish_asr_rdl.decoding.timing import run_timed, write_run_outputs
from turkish_asr_rdl.evaluation.join import join
from turkish_asr_rdl.evaluation.metrics import corpus_rates, score_pair
from turkish_asr_rdl.evaluation.normalize import RAW_V1, TR_NORMALIZED_V1, normalize
from turkish_asr_rdl.models.bilstm_ctc import CharCtcTokenizer
from turkish_asr_rdl.schemas.jsonl import read_jsonl, write_json
from turkish_asr_rdl.schemas.predictions import validate_predictions

OUT = ROOT / "artifacts/m3-final-2026-11-22/gaussian-hmm"
B5_OUT = OUT.parent / "b5"
TRAIN_REFS = PRIVATE / "train" / RELEASE / f"{RELEASE}.references.train.jsonl"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "summary.json").exists():
        print("HMM final evaluation already complete", flush=True)
        return
    torch.set_num_threads(1)
    split_sha = sha(SPLITS / "split_manifest.json")
    frozen = {"state": "fixed_before_test_scoring", "frozen_at_utc": now(), "dataset_release": RELEASE,
              "split_manifest_sha256": split_sha, "feature_recipe": FEATURE_RECIPE, "em_passes": 3,
              "states_per_unit": 3, "gaussians_per_state": 1, "external_lm": False,
              "script_sha256": sha(__file__), "model_code_sha256": sha(ROOT / "src/turkish_asr_rdl/baselines/m3_gaussian_hmm.py"),
              "source_script_sha256": sha(ROOT / "artifacts/b5-gpu-resume-2026-11-22/source/train_b5_and_gmmhmm.py"),
              "feature_threads_during_preparation": 4, "cpu_threads_during_training_inference": 1,
              "training_reference_sha256": sha(TRAIN_REFS), "normalizer": TR_NORMALIZED_V1,
              "limitations": ["Supplied stage is named GMM-HMM but implements one Gaussian per character state",
                  "DCT of normalized log-Mel features is the supplied recipe, not standard MFCC computation",
                  "Infeasible training clips are explicitly counted; evaluation retains all frozen IDs",
                  "Gaussian quadratic expansion can introduce float32 rounding differences from the source difference tensor",
                  "No supported scored N-best; requested_n=0", "Working minutes are not fully verified verbatim references"]}
    write_json(OUT / "frozen-protocol.json", frozen)
    write_json(OUT / "status.json", {"state": "waiting_for_b5_serial_timing", "started_at_utc": now(), "pid": os.getpid()})
    print("HMM waiting for B5 serial timing to finish (avoids competing benchmark workload)", flush=True)
    while not (B5_OUT / "timing/completed.json").exists():
        state_file = B5_OUT / "status.json"
        if state_file.exists() and json.loads(state_file.read_text())["state"] == "failed":
            raise RuntimeError("B5 evaluation failed; repair it before proceeding")
        time.sleep(10)
    checkpoint = OUT / "gaussian_hmm_character.npz"
    if not checkpoint.exists():
        write_json(OUT / "status.json", {"state": "preparing_train_features", "updated_at_utc": now(), "pid": os.getpid()})
        rows, references = frozen_inputs("train", TRAIN_REFS)
        texts = {uid: normalize(text, TR_NORMALIZED_V1) for uid, text in references.items()}
        tokenizer = CharCtcTokenizer.fit_from_train_texts(texts.values())
        units = tokenizer.id_to_token[2:]
        mapping = {char: i for i, char in enumerate(units)}
        feature_cache = OUT / "feature-cache/train"
        feature_cache.mkdir(parents=True, exist_ok=True)
        extractor = M3Features()

        def cached(row):
            path = feature_cache / (row["utterance_id"] + ".npy")
            if path.exists():
                return np.load(path, allow_pickle=False).astype(np.float32)
            audio = resolve_audio_path(AUDIO, row["audio_uri"])
            if sha(audio) != row["audio_sha256"]:
                raise ValueError("HMM training audio hash mismatch")
            features = extractor(load_audio(audio))
            with path.with_suffix(".tmp").open("wb") as f:
                np.save(f, features.astype(np.float16), allow_pickle=False)
            os.replace(path.with_suffix(".tmp"), path)
            return features

        items, used, dropped = [], [], []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, (row, features) in enumerate(zip(rows, pool.map(cached, rows)), 1):
                sequence = [mapping[char] for char in texts[row["utterance_id"]] if char in mapping]
                if not sequence or len(features) < len(sequence) * 3:
                    dropped.append(row["utterance_id"])
                else:
                    items.append((features, sequence))
                    used.append(row["utterance_id"])
                if index % 250 == 0 or index == len(rows):
                    print(f"HMM TRAIN CACHE {index}/{len(rows)}", flush=True)
        evidence = {"frozen_train_clips": len(rows), "used_train_clips": len(items), "infeasible_train_clips": len(dropped),
                    "used_ids": used, "infeasible_ids": dropped, "feature_recipe": FEATURE_RECIPE}
        write_json(OUT / "training-evidence.json", evidence)
        print(f"HMM train: {len(items)} feasible / {len(rows)} frozen clips; {len(dropped)} infeasible", flush=True)
        write_json(OUT / "status.json", {"state": "training", "train_clips": len(items), "updated_at_utc": now(), "pid": os.getpid()})
        means, variances, history = train(items, units)
        if not np.isfinite(means).all() or not np.isfinite(variances).all() or not (variances > 0).all():
            raise ValueError("Invalid learned HMM parameters")
        with (OUT / "checkpoint.tmp").open("wb") as f:
            np.savez(f, means=means, variances=variances, units=np.array(units))
        os.replace(OUT / "checkpoint.tmp", checkpoint)
        write_json(OUT / "training-history.json", {"history": history, "checkpoint_sha256": sha(checkpoint)})
        del items
        gc.collect()
    frozen.update(state="checkpoint_frozen", checkpoint_sha256=sha(checkpoint), checkpoint_frozen_at_utc=now())
    write_json(OUT / "frozen-protocol.json", frozen)
    warmup = inputs("train")[:3]
    completed = {}
    for mode, split in [("validation", "validation"), ("test", "test"), ("timing", "test")]:
        directory = OUT / mode
        if (directory / "completed.json").exists():
            done = json.loads((directory / "completed.json").read_text())
            for name, digest in done["files"].items():
                if sha(directory / name) != digest:
                    raise ValueError("Existing HMM export hash mismatch")
            completed[mode] = done
            continue
        rows = inputs(split)
        if mode == "timing":
            by_id = {r.utterance_id: r for r in rows}
            rows = [by_id[uid] for uid in (SPLITS / "m45-timing-30min.txt").read_text().splitlines()]
        ids = [r.utterance_id for r in rows]
        adapter = GaussianHmmAdapter(checkpoint)
        run_id = f"m3-gaussian-hmm-{mode}-20261122"
        print(f"HMM START {mode}: {len(rows)} clips", flush=True)
        write_json(OUT / "status.json", {"state": "inference", "mode": mode, "rows": len(rows), "updated_at_utc": now(), "pid": os.getpid()})
        # This adapter uses CPU exclusively; no CUDA sync or GPU memory claim.
        def cpu_sync():
            pass
        run = run_timed(adapter, rows, warmup, audio_root=AUDIO, run_id=run_id, repeats=3 if mode == "timing" else 1,
                        batch_size=1, sync=cpu_sync, memory=NoVram())
        context = RunContext(run_id, adapter.system_id, RELEASE, split, split_sha,
                             adapter.model_artifact_sha256, adapter.decoder_config_sha256)
        files = write_run_outputs(directory, run, rows, context, adapter, command="scripts/m3_run_gaussian_hmm.py")
        report = validate_predictions(read_jsonl(directory / "predictions.jsonl"), ids)
        if not report.ok or any(h.status == "error" for h in run.hypotheses):
            raise ValueError(f"HMM inference/contract failure: {report.errors}")
        done = {"state": "completed", "rows": len(rows), "files": files, "rtf": run.end_to_end_rtf,
                "repeats": 3 if mode == "timing" else 1, "requested_n": 0,
                "timing_claim": "CPU serial 30-minute subset" if mode == "timing" else "accuracy inference only"}
        write_json(directory / "completed.json", done)
        completed[mode] = done
        print(f"HMM COMPLETE {mode}: zero errors, RTF={run.end_to_end_rtf:.5f}", flush=True)
    results = {}
    for split, reference in [("validation", PRIVATE / "validation-recovered-minutes-v1/references.validation.jsonl"),
                             ("test", PRIVATE / "evaluator-final/references.test.jsonl")]:
        print(f"HMM SCORING {split}", flush=True)
        write_json(OUT / "status.json", {"state": "scoring", "split": split, "updated_at_utc": now(), "pid": os.getpid()})
        joined = join(read_jsonl(OUT / split / "predictions.jsonl"), read_jsonl(reference), split=split,
                      split_manifest_sha256=split_sha)
        scores = {view: dataclasses.asdict(corpus_rates([
            score_pair(r.utterance_id, r.reference, r.hypothesis, view, r.status) for r in joined]))
                  for view in [RAW_V1, TR_NORMALIZED_V1]}
        results[split] = {"rows": len(joined), "reference_sha256": sha(reference), "metrics": scores}
        write_json(OUT / f"metrics.{split}.json", results[split])
        print(f"HMM {split}: WER={scores[TR_NORMALIZED_V1]['wer']:.6f} CER={scores[TR_NORMALIZED_V1]['cer']:.6f}", flush=True)
    write_json(OUT / "summary.json", {"state": "completed", "system_id": GaussianHmmAdapter.system_id,
               "ended_at_utc": now(), "scores": results, "headline_timing": completed["timing"],
               "checkpoint_sha256": sha(checkpoint), "limitations": frozen["limitations"]})
    write_json(OUT / "status.json", {"state": "completed", "ended_at_utc": now()})
    print("HMM FINAL EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "status.json", {"state": "failed", "error": str(exc), "at_utc": now()})
        traceback.print_exc()
        raise
