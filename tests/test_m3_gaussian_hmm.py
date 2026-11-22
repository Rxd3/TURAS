"""Independent density and exhaustive small-path checks for M3 HMM optimization."""
import itertools

import numpy as np

from turkish_asr_rdl.baselines.m3_gaussian_hmm import decode_path, emissions, forced_path


def test_expanded_gaussian_densities_match_direct_formula():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(12, 39)).astype(np.float32)
    means = rng.normal(size=(6, 39)).astype(np.float32)
    var = rng.uniform(0.05, 2, size=(6, 39)).astype(np.float32)
    direct = -0.5 * np.sum(np.log(2 * np.pi * var), axis=1)[None, :] - 0.5 * np.sum(
        (x[:, None, :] - means[None, :, :]) ** 2 / var[None, :, :], axis=2)
    np.testing.assert_allclose(emissions(x, means, var), direct, rtol=2e-6, atol=5e-5)


def test_ergodic_decoder_matches_exhaustive_path_search():
    rng = np.random.default_rng(42)
    density = rng.normal(size=(5, 6)).astype(np.float32)
    choices = []
    for path in itertools.product(range(6), repeat=5):
        if path[0] % 3 != 0:
            continue
        score = float(density[0, path[0]])
        for t, (a, b) in enumerate(zip(path, path[1:]), 1):
            if b == a:
                score -= 0.35
            elif b == a + 1 and b // 3 == a // 3:
                score -= 1.0
            elif a % 3 == 2 and b % 3 == 0:
                score -= 2.5
            else:
                break
            score += float(density[t, b])
        else:
            choices.append((score, path))
    wanted = max(choices)[1]
    assert tuple(decode_path(density)) == wanted


def test_forced_alignment_matches_exhaustive_path_search():
    rng = np.random.default_rng(2026)
    density = rng.normal(size=(8, 6)).astype(np.float32)
    sequence = np.arange(6, dtype=np.int32)
    choices = []
    for moves in itertools.product([0, 1], repeat=7):
        path = (0, *np.cumsum(moves))
        if path[-1] != 5:
            continue
        score = sum(float(density[t, state]) for t, state in enumerate(path))
        score -= sum(1.2 if move else 0.35 for move in moves)
        choices.append((score, path))
    score, wanted = max(choices)
    path, ll = forced_path(density, sequence)
    assert tuple(path) == wanted
    np.testing.assert_allclose(ll, score / 8, atol=1e-6)
