"""Timing harness: adapter configuration checks before loading. SYNTHETIC inputs only."""
import pytest

import turkish_asr_rdl.decoding.timing as timing_module
from turkish_asr_rdl.decoding.interface import Hypothesis
from turkish_asr_rdl.decoding.timing import run_timed

WARM = ("warm_1", "warm_2", "warm_3")
TIMED = ("utt_a", "utt_b")


class ConfigAdapter:
    """Counts load calls; its configuration is whatever the test gives."""

    system_id = "sys_synthetic"
    model_artifact_sha256 = "b" * 64
    decoder_config_sha256 = "a" * 64

    def __init__(self, requested_n=0, score_convention_id=None, config=...):
        self.requested_n = requested_n
        self.score_convention_id = score_convention_id
        self.config = {"beam": 1} if config is ... else config
        self.loads = 0

    def load(self):
        self.loads += 1

    def transcribe(self, batch):
        return [Hypothesis(item.utterance_id, "ok", "SYNTHETIC") for item in batch]


def no_default(*args, **kwargs):
    raise AssertionError("a default device helper ran before the adapter configuration was checked")


@pytest.fixture
def refused(make_input, tmp_path, monkeypatch):
    """Run the harness and return the ValueError it raises; the default sync must not run first."""
    monkeypatch.setattr(timing_module, "default_sync", no_default)

    def _refused(adapter):
        timed, warm = [make_input(u) for u in TIMED], [make_input(u) for u in WARM]
        with pytest.raises(ValueError) as info:
            run_timed(adapter, timed, warm, audio_root=tmp_path, run_id="run_synthetic", sync=None,
                      probe=no_default)
        assert adapter.loads == 0
        message = str(info.value)
        assert "\n" not in message
        return message

    return _refused


@pytest.mark.parametrize("requested_n", [-1, True, False, 2.0, "3", None])
def test_requested_n_must_be_an_int_of_at_least_zero(refused, requested_n):
    message = refused(ConfigAdapter(requested_n=requested_n, score_convention_id="nbest-synthetic-v1"))
    assert "requested_n" in message


@pytest.mark.parametrize("convention", [None, "", 5])
def test_nbest_needs_a_score_convention_id(refused, convention):
    assert "score_convention_id" in refused(ConfigAdapter(requested_n=2, score_convention_id=convention))


@pytest.mark.parametrize("config", [
    {"length_penalty": float("nan")},
    {"beam": float("inf")},
    {"device": object()},
    {1: "int key", "1": "str key"},
    None,
    [("beam", 5)],
])
def test_config_must_be_strict_json(refused, config):
    assert "adapter.config" in refused(ConfigAdapter(config=config))
