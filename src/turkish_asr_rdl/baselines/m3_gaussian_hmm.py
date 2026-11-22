"""M3's supplied single-diagonal-Gaussian, three-state character HMM.

The supplied stage is named GMM-HMM but has one Gaussian per state. Its feature
recipe is DCT of normalized log-Mel bins, not standard raw-Mel MFCC extraction.
Emission expansion and vectorized transitions avoid a large T x S x 39 tensor.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from turkish_asr_rdl.decoding.interface import Hypothesis
from turkish_asr_rdl.models.bilstm_ctc import LogMelSpectrogramExtractor

FEATURE_RECIPE = "m3-dct-normalized-logmel-first40-every2-delta39-fp16-v1"


class M3Features:
    def __init__(self):
        self.extractor = LogMelSpectrogramExtractor()
        basis = np.zeros((40, 13), dtype=np.float32)
        for k in range(13):
            for n in range(40):
                basis[n, k] = math.cos(math.pi * k * (2 * n + 1) / 80.0)
        basis *= math.sqrt(2.0 / 40)
        basis[:, 0] *= math.sqrt(0.5)
        self.basis = torch.from_numpy(basis)

    def __call__(self, samples):
        with torch.inference_mode():
            mel, _ = self.extractor(torch.from_numpy(samples))
            mfcc = mel[0, ::2, :40] @ self.basis
            d1 = F.pad(mfcc[1:] - mfcc[:-1], (0, 0, 1, 0))
            d2 = F.pad(d1[1:] - d1[:-1], (0, 0, 1, 0))
            result = torch.cat([mfcc, d1, d2], dim=-1)
            result = (result - result.mean(0, keepdim=True)) / result.std(0, keepdim=True).clamp(min=1e-5)
            return result.half().float().numpy()


def emissions(features, means, variances):
    """Diagonal Gaussian log densities; algebraically the supplied difference sum."""
    inverse = 1.0 / variances
    constant = -0.5 * np.sum(np.log(2.0 * np.pi * variances), axis=1)
    return constant[None, :] - 0.5 * (
        (features ** 2) @ inverse.T - 2 * features @ (means * inverse).T
        + np.sum(means ** 2 * inverse, axis=1)[None, :])


def forced_path(log_b, states):
    """Source transcript-constrained left-to-right Viterbi alignment."""
    count, width = log_b.shape[0], len(states)
    previous = np.full(width, -1e18, dtype=np.float32)
    previous[0] = log_b[0, states[0]]
    back = np.zeros((count, width), dtype=np.bool_)
    for t in range(1, count):
        stay = previous - 0.35
        step = np.full(width, -1e18, dtype=np.float32)
        step[1:] = previous[:-1] - 1.2
        choose_step = step > stay
        previous = np.where(choose_step, step, stay) + log_b[t, states]
        back[t] = choose_step
    path = np.zeros(count, dtype=np.int32)
    path[-1] = width - 1
    for t in range(count - 1, 0, -1):
        path[t - 1] = path[t] - int(back[t, path[t]])
    return states[path], float(previous[-1]) / count


def decode_path(log_b):
    """Source ergodic character-state decoder with vectorized within-unit steps."""
    count, width = log_b.shape
    indices = np.arange(width, dtype=np.int32)
    entries = indices[::3]
    exits = indices[2::3]
    destinations = indices[indices % 3 != 0]
    previous = np.full(width, -1e18, dtype=np.float32)
    previous[entries] = log_b[0, entries]
    back = np.zeros((count, width), dtype=np.int32)
    for t in range(1, count):
        score = previous - 0.35
        source = indices.copy()
        candidate = previous[destinations - 1] - 1.0
        better = candidate > score[destinations]
        score[destinations] = np.where(better, candidate, score[destinations])
        source[destinations] = np.where(better, destinations - 1, source[destinations])
        best_exit = exits[int(np.argmax(previous[exits]))]
        cross_score = previous[best_exit] - 2.5
        better_entry = cross_score > score[entries]
        score[entries] = np.where(better_entry, cross_score, score[entries])
        source[entries] = np.where(better_entry, best_exit, source[entries])
        previous = score + log_b[t]
        back[t] = source
    path = np.zeros(count, dtype=np.int32)
    path[-1] = int(np.argmax(previous))
    for t in range(count - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def train(items, units, *, progress=print):
    """Three Viterbi EM passes, preserving supplied initialization and floors.

    items contains (feature array, train-only character-index sequence).
    Infeasible clips are counted by the caller, never removed from evaluation.
    """
    states = len(units) * 3
    sums = np.zeros((states, 39), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(states, dtype=np.float64)
    prepared = []
    for features, sequence in items:
        state_sequence = np.array([u * 3 + s for u in sequence for s in range(3)], dtype=np.int32)
        aligned = state_sequence[np.linspace(0, len(state_sequence) - 1, len(features)).astype(np.int32)]
        np.add.at(sums, aligned, features)
        np.add.at(squares, aligned, features ** 2)
        np.add.at(counts, aligned, 1.0)
        prepared.append((features, state_sequence))
    if not prepared:
        raise ValueError("No feasible HMM training clips")
    global_mean = sums.sum(0) / counts.sum()
    global_var = np.maximum(squares.sum(0) / counts.sum() - global_mean ** 2, 0.1)
    means = np.tile(global_mean, (states, 1)).astype(np.float32)
    variances = np.tile(global_var, (states, 1)).astype(np.float32)
    for state in np.flatnonzero(counts >= 10):
        means[state] = sums[state] / counts[state]
        variances[state] = np.maximum(squares[state] / counts[state] - (sums[state] / counts[state]) ** 2, 0.05)
    history = []
    for iteration in range(1, 4):
        sums.fill(0)
        squares.fill(0)
        counts.fill(0)
        likelihood = 0.0
        for index, (features, sequence) in enumerate(prepared, 1):
            aligned, ll = forced_path(emissions(features, means, variances), sequence)
            likelihood += ll
            np.add.at(sums, aligned, features)
            np.add.at(squares, aligned, features ** 2)
            np.add.at(counts, aligned, 1.0)
            if index % 250 == 0 or index == len(prepared):
                progress(f"HMM EM {iteration}/3: {index}/{len(prepared)}", flush=True)
        for state in np.flatnonzero(counts >= 10):
            m = sums[state] / counts[state]
            means[state] = m
            variances[state] = np.maximum(squares[state] / counts[state] - m ** 2, 0.05)
        history.append({"iteration": iteration, "mean_frame_log_likelihood": likelihood / len(prepared)})
    return means, variances, history


class GaussianHmmAdapter:
    system_id = "baseline-m3-single-gaussian-character-hmm"
    requested_n = 0
    score_convention_id = None

    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        self.model_artifact_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.config = {"model": "single diagonal Gaussian per character state", "states_per_unit": 3,
                       "feature_recipe": FEATURE_RECIPE, "decoder": "ergodic_character_viterbi",
                       "lm_rescoring": False, "device": "cpu", "precision": "float32",
                       "whitespace_policy": "supplied decoder collapses and strips whitespace", "requested_n": 0}
        self.decoder_config_sha256 = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.completed = 0

    def load(self):
        if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != self.model_artifact_sha256:
            raise ValueError("HMM checkpoint hash mismatch")
        with np.load(self.checkpoint, allow_pickle=False) as data:
            self.means, self.variances, self.units = data["means"], data["variances"], data["units"].tolist()
        self.features = M3Features()

    def transcribe(self, batch):
        results = []
        for item in batch:
            features = self.features(item.samples)
            path = decode_path(emissions(features, self.means, self.variances))
            chars = []
            prev_unit, prev_sub = -1, -1
            for state in path:
                unit, sub = int(state) // 3, int(state) % 3
                if unit != prev_unit or (sub == 0 and prev_sub == 2):
                    chars.append(self.units[unit])
                prev_unit, prev_sub = unit, sub
            text = " ".join("".join(chars).split())
            results.append(Hypothesis(item.utterance_id, "ok" if text else "empty", text))
            self.completed += 1
            if self.completed % 50 == 0:
                print(f"HMM INFERENCE {self.completed} items including warm-up", flush=True)
        return results
