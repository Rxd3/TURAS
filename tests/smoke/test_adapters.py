"""Torch-free parts of the 5070 Ti smoke adapters and probes. SYNTHETIC token IDs and text only."""
import re
import sys
import types
from pathlib import Path

import pytest

from turkish_asr_rdl.decoding.interface import AudioItem, Hypothesis

HEX64 = re.compile(r"[0-9a-f]{64}")


def test_collapse_ctc_merges_repeats_and_drops_blanks(common):
    assert common.collapse_ctc([0, 5, 5, 0, 5, 7, 7, 0], blank=0) == (5, 5, 7)
    assert common.collapse_ctc([0, 0, 0], blank=0) == ()


def test_cut_at_stop_keeps_the_stop_token(common):
    assert common.cut_at_stop([11, 12, 9, 9], frozenset({9})) == (11, 12, 9)
    assert common.cut_at_stop([11, 12], frozenset({9})) == (11, 12)


def test_beam_hypothesis_sums_scores_through_the_stop_token(common):
    hyp = common.beam_hypothesis(
        "u1",
        [" SYNTHETIC a", " SYNTHETIC b"],
        [[11, 12, 9, 9], [11, 13, 14, 9]],
        [[-0.5, -0.25, -0.25, 0.0], [-1.0, -1.0, -1.0, -1.0]],
        [-0.3, -1.0],
        kind="seq2seq_token_log_prob_sum",
        details={"nbest_algorithm": "beam search"},
        stop_ids=frozenset({9}),
    )
    assert (hyp.status, hyp.text) == ("ok", " SYNTHETIC a")
    first, second = hyp.candidates
    assert (first.candidate_id, first.rank, first.token_ids) == ("u1-c01", 1, (11, 12, 9))
    assert first.acoustic_log_score == pytest.approx(-1.0) and second.acoustic_log_score == pytest.approx(-4.0)
    assert first.decoder_sequence_score == pytest.approx(-0.3)
    assert first.score_details["nbest_algorithm"] == "beam search"


def test_beam_hypothesis_gives_a_reason_when_there_is_no_beam_score(common):
    hyp = common.beam_hypothesis("u1", ["SYN"], [[11, 9]], [[-0.5, -0.5]], [None], kind="k",
                                   details={}, stop_ids=frozenset({9}))
    (only,) = hyp.candidates
    assert only.decoder_sequence_score is None
    assert only.score_details["null_reasons"]["decoder_sequence_score"]


def test_make_hypothesis_treats_whitespace_as_empty(common):
    hyp = common.make_hypothesis("u1", ())
    assert (hyp.status, hyp.text) == ("empty", "")
    blank = common.beam_hypothesis("u2", ["  "], [[9]], [[-0.1]], [None], kind="k", details={},
                                     stop_ids=frozenset({9}))
    assert (blank.status, blank.text, len(blank.candidates)) == ("empty", "", 1)


def test_weights_sha256_covers_names_and_bytes(common, tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"SYNTHETIC one")
    (tmp_path / "b.safetensors").write_bytes(b"SYNTHETIC two")
    first = common.weights_sha256(tmp_path, "*.safetensors")
    assert HEX64.fullmatch(first) and first == common.weights_sha256(tmp_path, "*.safetensors")
    (tmp_path / "b.safetensors").write_bytes(b"SYNTHETIC changed")
    assert common.weights_sha256(tmp_path, "*.safetensors") != first
    with pytest.raises(FileNotFoundError, match=r"\*\.bin"):
        common.weights_sha256(tmp_path, "*.bin")


EXPECTED = {
    "whisper-small": ("WhisperAdapter", 10, "float16"),
    "whisper-large-v3": ("WhisperAdapter", 10, "float16"),
    "qwen3-asr-1.7b": ("QwenAsrAdapter", 5, "bfloat16"),
    "qwen3-asr-0.6b": ("QwenAsrAdapter", 5, "bfloat16"),
    "xlsr-tr-cv7": ("XlsrAdapter", 1, "float32"),
    "xlsr-tr-cv8": ("XlsrAdapter", 1, "float32"),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_build_adapter_picks_the_family_without_loading(adapters, monkeypatch, name):
    def fake_fetch(model_name):
        repo_id, revision, _ = adapters.MODEL_REPOS[model_name]
        return adapters.ModelFiles(model_name, repo_id, revision, Path("unused"), "e" * 64)

    monkeypatch.setattr(adapters, "fetch_model", fake_fetch)
    adapter = adapters.build_adapter(name)
    class_name, requested_n, precision = EXPECTED[name]
    assert type(adapter).__name__ == class_name and adapter.requested_n == requested_n
    assert adapter.config["precision"] == precision and adapter.config["repo_id"] == adapters.MODEL_REPOS[name][0]
    assert adapter.system_id == f"smoke-{name}" and adapter.model_artifact_sha256 == "e" * 64
    assert HEX64.fullmatch(adapter.decoder_config_sha256) and adapter.score_convention_id
    assert adapter.extra_facts([])["decode_path"] == "not run"


def _files(adapters):
    return adapters.ModelFiles("synthetic", "synthetic/model", "f" * 40, Path("unused"), "e" * 64)


def test_beam_adapter_falls_back_to_greedy_once_and_keeps_the_traceback(adapters):
    class Flaky(adapters.BeamAdapter):
        beams_tried = 0

        def load(self):
            return None

        def beam(self, item):
            self.beams_tried += 1
            raise RuntimeError("SYNTHETIC beam failure")

        def greedy(self, item):
            return Hypothesis(item.utterance_id, "ok", "SYNTHETIC greedy")

    adapter = Flaky(_files(adapters), {"num_beams": 5})
    items = [AudioItem("u1", None), AudioItem("u2", None)]
    hyps = adapter.transcribe(items)
    assert [h.text for h in hyps] == ["SYNTHETIC greedy"] * 2 and adapter.beams_tried == 1
    facts = adapter.extra_facts(items)
    assert facts["decode_path"].startswith("greedy")
    assert "SYNTHETIC beam failure" in facts["beam_error"] and "Traceback" in facts["beam_traceback"]


def test_beam_adapter_reports_the_beam_path_when_it_works(adapters):
    class Steady(adapters.BeamAdapter):
        def load(self):
            return None

        def beam(self, item):
            return Hypothesis(item.utterance_id, "ok", "SYNTHETIC beam")

        def greedy(self, item):
            raise AssertionError("greedy must not run")

    adapter = Steady(_files(adapters), {"num_beams": 5})
    assert adapter.transcribe([AudioItem("u1", None)])[0].text == "SYNTHETIC beam"
    facts = adapter.extra_facts([])
    assert facts["decode_path"] == "beam 5" and facts["beam_error"] is None


def test_transcribe_keeps_the_first_traceback_and_reraises(adapters):
    class Broken(adapters.SmokeAdapter):
        calls = 0

        def load(self):
            return None

        def decode(self, item):
            self.calls += 1
            raise ValueError(f"SYNTHETIC failure {self.calls}")

    adapter = Broken(_files(adapters), {})
    for _ in range(2):
        with pytest.raises(ValueError):
            adapter.transcribe([AudioItem("u1", None)])
    assert "SYNTHETIC failure 1" in adapter.extra_facts([])["first_traceback"]


def test_train_probe_needs_enough_clips_before_touching_the_gpu(probes):
    with pytest.raises(ValueError, match="needs 8 clips"):
        probes.train_probe("train-whisper-small-full", [(None, "SYNTHETIC")], vram=None)


def test_model_tables_cover_the_cli_names(smoke, adapters, probes):
    assert set(adapters.MODEL_REPOS) == set(smoke.INFERENCE_MODELS)
    assert set(probes.TRAINING_PROBES) == set(smoke.TRAINING_MODELS)
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for _, revision, _ in adapters.MODEL_REPOS.values())


@pytest.mark.parametrize("name, kept, skipped", [("whisper-large-v3", "model.safetensors", "*.bin"),
                                                 ("xlsr-tr-cv7", "pytorch_model.bin", "language_model/*")])
def test_fetch_model_pins_the_revision_and_skips_unused_files(adapters, monkeypatch, tmp_path, name, kept, skipped):
    calls = []

    def snapshot_download(repo_id, revision, ignore_patterns):
        calls.append((repo_id, revision, ignore_patterns))
        (tmp_path / kept).write_bytes(b"SYNTHETIC weights")
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    files = adapters.fetch_model(name)
    repo_id, revision, _ = adapters.MODEL_REPOS[name]
    assert calls[0][:2] == (repo_id, revision) and skipped in calls[0][2] and "*fp32*" in calls[0][2]
    assert files.path == tmp_path and HEX64.fullmatch(files.artifact_sha256)
    if kept.endswith(".bin"):
        assert "*.bin" not in calls[0][2]


def test_cuda_torch_refuses_a_machine_without_cuda(common, monkeypatch):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        common.cuda_torch()


@pytest.mark.parametrize("fp32, bf16, outcome", [(True, True, "completed"), (True, False, "fp32 only"),
                                                 (False, True, "failed"), (False, False, "failed")])
def test_pyg_outcome_fails_only_when_fp32_fails(probes, fp32, bf16, outcome):
    assert probes._pyg_outcome({"fp32": {"ok": fp32}, "bf16": {"ok": bf16}}) == outcome
