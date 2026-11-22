"""Diagonal mixture-Gaussian, three-state grapheme HMM with standard MFCC39.

This is a separate experiment from the supplied single-Gaussian/DCT baseline.
There are four Gaussians per state, normalized transition probabilities and
soft component responsibilities within transcript-constrained Viterbi EM.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

from turkish_asr_rdl.decoding.interface import Hypothesis

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda function: function

STAY = float(np.log(0.65))
ADVANCE = float(np.log(0.35))
FEATURE_CONFIG = {"recipe": "mfcc39-standard-v1", "sample_rate": 16000, "preemphasis": 0.97,
                  "n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 26, "n_mfcc": 13,
                  "dct_type": 2, "dct_norm": "ortho", "lifter": 22, "htk": True, "center": False,
                  "delta_width": 5, "delta_mode": "nearest", "cmvn": "per-utterance after stacking"}


class Mfcc39:
    def __call__(self, samples):
        import librosa
        x = np.asarray(samples, dtype=np.float32)
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - np.float32(0.97) * x[:-1]
        static = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13, n_fft=512, win_length=400,
                    hop_length=160, n_mels=26, dct_type=2, norm="ortho", lifter=22, htk=True, center=False)
        delta = librosa.feature.delta(static, width=5, order=1, mode="nearest")
        acceleration = librosa.feature.delta(static, width=5, order=2, mode="nearest")
        result = np.concatenate([static, delta, acceleration], axis=0).T.astype(np.float32)
        result = (result - result.mean(0)) / np.maximum(result.std(0), np.float32(1e-5))
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite MFCC features")
        return result


def component_log_densities(features, means, variances, weights):
    states, mixtures, dimension = means.shape
    flat_mean = means.reshape(-1, dimension)
    inverse = (1.0 / variances).reshape(-1, dimension)
    constant = -0.5 * np.sum(np.log(2.0 * np.pi * variances), axis=-1).reshape(-1)
    quadratic = (features ** 2) @ inverse.T - 2.0 * features @ (flat_mean * inverse).T
    quadratic += np.sum(flat_mean ** 2 * inverse, axis=1)[None, :]
    return (constant[None, :] - 0.5 * quadratic).reshape(len(features), states, mixtures) + np.log(weights)[None, :, :]


def mixture_log_densities(features, means, variances, weights):
    components = component_log_densities(features, means, variances, weights)
    return np.logaddexp.reduce(components, axis=2), components


def component_responsibilities(components):
    return np.exp(components - np.logaddexp.reduce(components, axis=-1, keepdims=True))


@njit(cache=True)
def forced_align(log_emissions, sequence):
    frames = log_emissions.shape[0]
    width = len(sequence)
    previous = np.full(width, -1e30, dtype=np.float32)
    previous[0] = log_emissions[0, sequence[0]]
    back = np.zeros((frames, width), dtype=np.bool_)
    for t in range(1, frames):
        current = np.full(width, -1e30, dtype=np.float32)
        for state in range(width):
            best = previous[state] + STAY
            if state > 0 and previous[state - 1] + ADVANCE > best:
                best = previous[state - 1] + ADVANCE
                back[t, state] = True
            current[state] = best + log_emissions[t, sequence[state]]
        previous = current
    path = np.zeros(frames, dtype=np.int32)
    position = width - 1
    path[-1] = sequence[position]
    for t in range(frames - 1, 0, -1):
        position -= int(back[t, position])
        path[t - 1] = sequence[position]
    return path, float(previous[-1])


@njit(cache=True)
def decode_states(log_emissions, unit_log_priors):
    frames, states = log_emissions.shape
    previous = np.full(states, -1e30, dtype=np.float32)
    for unit in range(len(unit_log_priors)):
        previous[3 * unit] = unit_log_priors[unit] + log_emissions[0, 3 * unit]
    back = np.zeros((frames, states), dtype=np.int32)
    for t in range(1, frames):
        best_exit = 2
        for state in range(5, states, 3):
            if previous[state] > previous[best_exit]:
                best_exit = state
        current = np.empty(states, dtype=np.float32)
        for state in range(states):
            source = state
            best = previous[state] + STAY
            if state % 3:
                candidate = previous[state - 1] + ADVANCE
                if candidate > best:
                    best, source = candidate, state - 1
            else:
                candidate = previous[best_exit] + ADVANCE + unit_log_priors[state // 3]
                if candidate > best:
                    best, source = candidate, best_exit
            current[state] = best + log_emissions[t, state]
            back[t, state] = source
        previous = current
    path = np.zeros(frames, dtype=np.int32)
    path[-1] = np.argmax(previous)
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def update_parameters(counts, sums, squares, old_means, old_variances, floor=0.01):
    means, variances = old_means.copy(), old_variances.copy()
    valid = counts >= 1.0
    means[valid] = sums[valid] / counts[valid, None]
    variances[valid] = np.maximum(squares[valid] / counts[valid, None] - means[valid] ** 2, floor)
    smoothed = counts + 0.01
    weights = smoothed / smoothed.sum(axis=1, keepdims=True)
    return means.astype(np.float32), variances.astype(np.float32), weights.astype(np.float32)


class MixtureHmmAdapter:
    system_id = "baseline-gmm-hmm-mfcc39-4mix"
    requested_n = 0
    score_convention_id = None

    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        self.model_artifact_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.config = {"model": {"type": "grapheme Gaussian-mixture HMM", "states_per_unit": 3,
                      "mixtures_per_state": 4, "covariance": "diagonal", "precision": "float32", "device": "cpu", "threads": 1},
                      "features": FEATURE_CONFIG, "decoding": {"algorithm": "Viterbi", "stay_probability": 0.65,
                      "advance_probability": 0.35, "unit_priors": "smoothed train-character frequencies",
                      "external_lm": False, "text_postprocessing": "none", "requested_n": 0}}
        self.decoder_config_sha256 = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()

    def load(self):
        if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != self.model_artifact_sha256:
            raise ValueError("GMM checkpoint changed")
        with np.load(self.checkpoint, allow_pickle=False) as data:
            self.means, self.variances, self.weights = data["means"], data["variances"], data["weights"]
            self.units, self.priors = data["units"].tolist(), data["unit_priors"]
        if self.means.shape[1] != 4 or self.means.shape[2] != 39:
            raise ValueError("Expected four Gaussian components and 39 MFCC dimensions")
        self.extractor = Mfcc39()
        # Compile the native decoder during cold load, before warm-up/timed input.
        decode_states(np.zeros((3, len(self.units) * 3), dtype=np.float32), np.log(self.priors))

    def transcribe(self, batch):
        results = []
        for item in batch:
            features = self.extractor(item.samples)
            densities, _ = mixture_log_densities(features, self.means, self.variances, self.weights)
            path = decode_states(densities, np.log(self.priors))
            chars, previous_unit, previous_sub = [], -1, -1
            for state in path:
                unit, sub = int(state) // 3, int(state) % 3
                if unit != previous_unit or (sub == 0 and previous_sub == 2):
                    chars.append(self.units[unit])
                previous_unit, previous_sub = unit, sub
            text = "".join(chars)
            results.append(Hypothesis(item.utterance_id, "ok" if text else "empty", text))
        return results
