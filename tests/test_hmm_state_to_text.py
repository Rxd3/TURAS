"""State-path to text rules behind the published B6 and B7 figures.

Each grapheme unit has three states (unit * 3 + sub-state). A unit is emitted when
the path enters a new unit, or re-enters the same unit from its last state
(sub 2 -> sub 0), which is how a doubled letter is decoded. The adapters are run
with stubbed features and decoders, so the pinned model sources stay unchanged.
"""
from types import SimpleNamespace

import numpy as np
import pytest

UNITS = ["a", "b", " "]
DOUBLED_A_THEN_B = [0, 1, 2, 0, 1, 2, 3, 4, 5]   # "aab": re-entry emits a second "a"
STAYS = [0, 0, 1, 1, 2, 2, 6, 7, 8, 3, 4, 5]       # "a b": self-loops add nothing
NO_REENTRY = [0, 1, 2, 2, 1, 2]                     # never back to sub 0: one "a"


def item():
    return SimpleNamespace(utterance_id="utt_x", samples=np.zeros(1600, dtype=np.float32))


def test_mixture_hmm_state_path_to_text(monkeypatch):
    mixture = pytest.importorskip("turkish_asr_rdl.baselines.mixture_hmm")
    adapter = object.__new__(mixture.MixtureHmmAdapter)
    adapter.units, adapter.priors = UNITS, np.ones(len(UNITS))
    adapter.means = adapter.variances = adapter.weights = None
    adapter.extractor = lambda samples: samples
    monkeypatch.setattr(mixture, "mixture_log_densities", lambda x, *a: (x, None))
    for path, text in [(DOUBLED_A_THEN_B, "aab"), (STAYS, "a b"), (NO_REENTRY, "a")]:
        monkeypatch.setattr(mixture, "decode_states", lambda d, p, path=path: path)
        [hyp] = adapter.transcribe([item()])
        assert (hyp.status, hyp.text) == ("ok", text)


def test_single_gaussian_hmm_state_path_to_text(monkeypatch):
    pytest.importorskip("torch")
    m3 = pytest.importorskip("turkish_asr_rdl.baselines.m3_gaussian_hmm")
    adapter = object.__new__(m3.GaussianHmmAdapter)
    adapter.units, adapter.completed = UNITS, 0
    adapter.means = adapter.variances = None
    adapter.features = lambda samples: samples
    monkeypatch.setattr(m3, "emissions", lambda f, *a: f)
    # This adapter also collapses repeated spaces.
    for path, text in [(DOUBLED_A_THEN_B, "aab"), (STAYS, "a b"), (NO_REENTRY, "a")]:
        monkeypatch.setattr(m3, "decode_path", lambda e, path=path: path)
        [hyp] = adapter.transcribe([item()])
        assert (hyp.status, hyp.text) == ("ok", text)
