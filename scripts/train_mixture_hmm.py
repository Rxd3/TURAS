"""Train a separate four-component diagonal GMM-HMM on frozen train references."""
from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

from turkish_asr_rdl.baselines.mixture_hmm import (
    FEATURE_CONFIG, Mfcc39, component_responsibilities, forced_align, mixture_log_densities, update_parameters)
from turkish_asr_rdl.decoding.audio import load_audio, resolve_audio_path
from turkish_asr_rdl.evaluation.normalize import normalize, TR_NORMALIZED_V1
from turkish_asr_rdl.models.bilstm_ctc import CharCtcTokenizer
from turkish_asr_rdl.schemas.jsonl import read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/completion-2026-11-22"
OUT = BASE / "gmm"
RELEASE = "turas-tr-econ-legis-v0.1"
SPLITS = ROOT / "data/splits" / RELEASE
AUDIO = ROOT / "data/private" / RELEASE / "inference" / RELEASE
REFERENCES = ROOT / "data/private" / RELEASE / "train" / RELEASE / f"{RELEASE}.references.train.jsonl"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def checkpoint(path, means, variances, weights, units, priors):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez(f, means=means, variances=variances, weights=weights, units=np.array(units), unit_priors=priors)
    temp.replace(path)


def main():
    from sklearn.mixture import GaussianMixture
    from sklearn.exceptions import ConvergenceWarning
    import librosa
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "checkpoint.npz").exists():
        manifest = json.loads((OUT / "model-manifest.json").read_text())
        if manifest["checkpoint_sha256"] != sha(OUT / "checkpoint.npz"):
            raise ValueError("Completed GMM checkpoint hash mismatch")
        print("GMM training already complete", flush=True)
        return
    ids = (SPLITS / "train.txt").read_text().splitlines()
    rows = read_jsonl(AUDIO / f"{RELEASE}.inference.train.jsonl")
    refs = read_jsonl(REFERENCES)
    ref_map = {r["utterance_id"]: r["reference_text"] for r in refs}
    if len(ref_map) != len(refs) or [r["utterance_id"] for r in rows] != ids or set(ref_map) != set(ids):
        raise ValueError("Frozen training IDs/reference coverage mismatch")
    split = json.loads((SPLITS / "split_manifest.json").read_text())
    if sha(SPLITS / "train.txt") != split["splits"]["train"]["sha256"]:
        raise ValueError("Frozen training split changed")
    texts = {uid: normalize(ref_map[uid], TR_NORMALIZED_V1) for uid in ids}
    tokenizer = CharCtcTokenizer.fit_from_train_texts(texts.values())
    units = tokenizer.id_to_token[2:]
    mapping = {ch: i for i, ch in enumerate(units)}
    counts = np.full(len(units), 0.1, dtype=np.float64)
    for text in texts.values():
        for char in text:
            counts[mapping[char]] += 1
    priors = (counts / counts.sum()).astype(np.float32)
    frozen = {"state": "frozen_train_only_protocol", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_release": RELEASE, "split_manifest_sha256": sha(SPLITS / "split_manifest.json"),
              "train_ids_sha256": sha(SPLITS / "train.txt"), "train_reference_sha256": sha(REFERENCES),
              "feature_config": FEATURE_CONFIG, "librosa_version": librosa.__version__, "seed": 42,
              "states_per_unit": 3, "mixtures_per_state": 4, "em_passes": 5, "covariance": "diagonal",
              "variance_floor": 0.01, "mixture_smoothing": 0.01, "initialization_samples_per_state": 2048,
              "initialization": "train-only uniform alignment; sklearn diagonal GMM, 15 iterations, seed 42",
              "training": "Viterbi state alignment with soft Gaussian-component responsibilities",
              "model_selection": "fixed five-pass budget; no test or validation tuning of mixture count",
              "script_sha256": sha(__file__), "model_code_sha256": sha(ROOT / "src/turkish_asr_rdl/baselines/mixture_hmm.py"),
              "limitations": ["Grapheme-state HMM, not a phoneme pronunciation-lexicon system", "No external language model",
                  "Working edited-minutes references; single training seed", "Separate follow-up experiment after prior project test exposure"]}
    write_json(OUT / "frozen-training.json", frozen)
    status = {"state": "features", "pid": os.getpid(), "train_rows": len(rows)}
    write_json(OUT / "status.json", status)
    cache = OUT / "feature-cache"
    cache.mkdir(exist_ok=True)
    feature_identity = {"config": FEATURE_CONFIG, "librosa": librosa.__version__}
    identity_file = cache / "identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != feature_identity:
        raise ValueError("Feature cache recipe/version mismatch")
    write_json(identity_file, feature_identity)
    extractor = Mfcc39()

    def features(row):
        path = cache / (row["utterance_id"] + ".npy")
        if path.exists():
            result = np.load(path, allow_pickle=False)
        else:
            audio = resolve_audio_path(AUDIO, row["audio_uri"])
            if sha(audio) != row["audio_sha256"]:
                raise ValueError("Training audio hash mismatch")
            result = extractor(load_audio(audio))
            with path.with_suffix(".tmp").open("wb") as f:
                np.save(f, result, allow_pickle=False)
            path.with_suffix(".tmp").replace(path)
        if result.ndim != 2 or result.shape[1] != 39 or not np.isfinite(result).all():
            raise ValueError("Invalid MFCC cache")
        return result

    items, dropped = [], []
    samples = [[] for _ in range(len(units) * 3)]
    sample_counts = np.zeros(len(samples), dtype=np.int32)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (row, x) in enumerate(zip(rows, pool.map(features, rows)), 1):
            sequence = np.array([mapping[c] * 3 + s for c in texts[row["utterance_id"]] for s in range(3)], dtype=np.int32)
            if len(sequence) == 0 or len(x) < len(sequence):
                dropped.append(row["utterance_id"])
                continue
            items.append((x, sequence))
            aligned = sequence[np.linspace(0, len(sequence) - 1, len(x)).astype(np.int32)]
            for state in np.unique(aligned):
                remaining = 2048 - sample_counts[state]
                if remaining > 0:
                    selected = x[aligned == state][:remaining].copy()
                    samples[state].append(selected)
                    sample_counts[state] += len(selected)
            if index % 250 == 0 or index == len(rows):
                print(f"GMM MFCC CACHE {index}/{len(rows)}", flush=True)
    write_json(OUT / "training-coverage.json", {"frozen_rows": len(rows), "used_rows": len(items), "infeasible_ids": dropped,
               "state_initialization_frame_counts": sample_counts.tolist()})
    if not items:
        raise ValueError("No feasible training examples")
    states = len(units) * 3
    means = np.zeros((states, 4, 39), dtype=np.float32)
    variances = np.ones_like(means)
    weights = np.full((states, 4), 0.25, dtype=np.float32)
    fallback = np.concatenate([x[:min(200, len(x))] for x, _ in items[:20]], axis=0)
    converged = 0
    for state, parts in enumerate(samples):
        x = np.concatenate(parts, axis=0) if sample_counts[state] >= 4 else fallback
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = GaussianMixture(n_components=4, covariance_type="diag", reg_covar=0.01,
                                     max_iter=15, n_init=1, random_state=42).fit(x)
        means[state], variances[state], weights[state] = model.means_, model.covariances_, model.weights_
        converged += int(model.converged_)
    print(f"GMM initialized {states} states, four Gaussians each; {len(items)} feasible clips", flush=True)
    status.update(state="training", initialization_converged=converged)
    write_json(OUT / "status.json", status)
    history = []
    for iteration in range(1, 6):
        counts = np.zeros((states, 4), dtype=np.float64)
        sums = np.zeros((states, 4, 39), dtype=np.float64)
        squares = np.zeros_like(sums)
        total_score, total_frames = 0., 0
        for index, (x, sequence) in enumerate(items, 1):
            density, components = mixture_log_densities(x, means, variances, weights)
            aligned, score = forced_align(density, sequence)
            responsibility = component_responsibilities(components[np.arange(len(x)), aligned])
            joint = aligned[:, None] * 4 + np.arange(4)[None, :]
            np.add.at(counts.reshape(-1), joint.ravel(), responsibility.ravel())
            np.add.at(sums.reshape(-1, 39), joint.ravel(), (responsibility[:, :, None] * x[:, None, :]).reshape(-1, 39))
            np.add.at(squares.reshape(-1, 39), joint.ravel(), (responsibility[:, :, None] * (x[:, None, :] ** 2)).reshape(-1, 39))
            total_score += score
            total_frames += len(x)
            if index % 250 == 0 or index == len(items):
                status.update(iteration=iteration, processed=index, total=len(items))
                write_json(OUT / "status.json", status)
                print(f"GMM VITERBI-EM {iteration}/5: {index}/{len(items)}", flush=True)
        means, variances, weights = update_parameters(counts, sums, squares, means, variances)
        if not np.isfinite(means).all() or not np.isfinite(variances).all() or not np.isfinite(weights).all():
            raise ValueError("Nonfinite GMM parameters")
        history.append({"iteration": iteration, "mean_frame_transcript_conditional_path_score": total_score / total_frames})
        checkpoint(OUT / f"checkpoint.iter-{iteration}.npz", means, variances, weights, units, priors)
        write_json(OUT / "training-history.json", {"history": history})
    checkpoint(OUT / "checkpoint.npz", means, variances, weights, units, priors)
    write_json(OUT / "model-manifest.json", {"state": "completed", "checkpoint_sha256": sha(OUT / "checkpoint.npz"),
                "trained_states": states, "mixtures_per_state": 4, "em_passes": 5, "train_rows": len(items),
                "infeasible_train_rows": len(dropped), "frozen_protocol_sha256": sha(OUT / "frozen-training.json"),
                "training_history_sha256": sha(OUT / "training-history.json")})
    write_json(OUT / "status.json", {**status, "state": "completed", "ended_at_utc": datetime.now(timezone.utc).isoformat()})
    print("GMM-HMM TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
