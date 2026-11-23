"""Check mixture probabilities, weighted EM and exhaustive constrained paths."""
import itertools
import numpy as np
from turkish_asr_rdl.baselines.mixture_hmm import (
    ADVANCE, STAY, component_responsibilities, decode_states, forced_align, mixture_log_densities, update_parameters)


def test_two_component_density_is_weighted_sum_not_mean_gaussian():
    x = np.array([[0.0], [3.0]], dtype=np.float32)
    means = np.array([[[0.0], [3.0]]], dtype=np.float32)
    variance = np.ones_like(means)
    weights = np.array([[0.25, 0.75]], dtype=np.float32)
    actual, components = mixture_log_densities(x, means, variance, weights)
    wanted = 0.25 * np.exp(-0.5 * x[:, 0] ** 2) / np.sqrt(2 * np.pi)
    wanted += 0.75 * np.exp(-0.5 * (x[:, 0] - 3) ** 2) / np.sqrt(2 * np.pi)
    np.testing.assert_allclose(np.exp(actual[:, 0]), wanted, rtol=1e-6)
    np.testing.assert_allclose(component_responsibilities(components).sum(-1), 1, atol=1e-6)


def test_m_step_updates_distinct_components_and_normalizes_weights():
    means = np.zeros((1, 2, 1), dtype=np.float32)
    var = np.ones_like(means)
    m, v, w = update_parameters(np.array([[2., 1.]]), np.array([[[2.], [5.]]]),
                                np.array([[[4.], [25.]]]), means, var)
    np.testing.assert_allclose(m[:, :, 0], [[1., 5.]])
    np.testing.assert_allclose(v[:, :, 0], [[1., 0.01]], atol=1e-6)
    np.testing.assert_allclose(w.sum(1), 1.)


def test_forced_hmm_alignment_matches_exhaustive_search():
    density = np.random.default_rng(42).normal(size=(6, 3)).astype(np.float32)
    paths = []
    for moves in itertools.product([0, 1], repeat=5):
        states = (0, *np.cumsum(moves))
        if states[-1] != 2:
            continue
        score = sum(float(density[t, state]) for t, state in enumerate(states))
        score += sum(ADVANCE if move else STAY for move in moves)
        paths.append((score, states))
    expected_score, expected_path = max(paths)
    actual, score = forced_align(density, np.arange(3, dtype=np.int32))
    assert tuple(actual) == expected_path
    np.testing.assert_allclose(score, expected_score, atol=1e-5)


def test_free_decoder_matches_exhaustive_normalized_character_transitions():
    """Check the fast exit-state shortcut against the full small HMM graph."""
    emissions = np.random.default_rng(19).normal(size=(7, 6)).astype(np.float32)
    priors = np.log(np.array([0.8, 0.2], dtype=np.float32))
    paths = [(float(priors[u] + emissions[0, 3 * u]), (3 * u,)) for u in range(2)]
    for frame in range(1, len(emissions)):
        expanded = []
        for score, path in paths:
            state = path[-1]
            choices = [(state, np.log(0.65))]
            if state % 3 < 2:
                choices.append((state + 1, np.log(0.35)))
            else:
                choices.extend((3 * unit, np.log(0.35) + priors[unit]) for unit in range(2))
            for destination, transition in choices:
                expanded.append((score + transition + float(emissions[frame, destination]), (*path, destination)))
        paths = expanded
    _, expected = max(paths)
    actual = decode_states(emissions, priors)
    assert tuple(actual) == expected
