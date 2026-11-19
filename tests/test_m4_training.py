"""Offline synthetic fixtures only; no pretrained model or dataset downloads."""
import json
from pathlib import Path

import pytest

from turkish_asr_rdl.training.acoustic import load_train_rows, preflight, sha256_file


@pytest.fixture
def bundle(tmp_path):
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"synthetic manifest fixture; audio decoding is tested separately")
    ids = tmp_path / "train.txt"
    ids.write_text("fixture-utterance\n", encoding="utf-8")
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(json.dumps({"dataset_release": "fixture-only", "splits": {
        "train": {"sha256": sha256_file(ids), "utterances": 1}}}), encoding="utf-8")
    split_hash = sha256_file(manifest)
    row = dict(schema_version="0.1", dataset_release="fixture-only", split_manifest_sha256=split_hash,
               utterance_id="fixture-utterance", audio_uri="audio.flac", audio_sha256=sha256_file(audio),
               duration_s=1.0, sample_rate_hz=16000, channels=1, language="tr_tr")
    inference = tmp_path / "inference.train.jsonl"
    refs = tmp_path / "references.train.jsonl"
    inference.write_text(json.dumps(row) + "\n", encoding="utf-8")
    reference = {**row, "split": "train", "reference_text": "synthetic test fixture", "verification_status": "unreviewed"}
    refs.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    return dict(inference=inference, references=refs, audio_root=tmp_path, split_manifest=manifest,
                train_ids=ids, release="fixture-only", split_sha256=split_hash,
                references_sha256=sha256_file(refs), inference_sha256=sha256_file(inference))


def test_train_join_accepts_only_matching_frozen_audio(bundle):
    rows = load_train_rows(**bundle)
    assert rows[0]["utterance_id"] == "fixture-utterance"
    Path(rows[0]["audio_path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="audio hash"):
        load_train_rows(**bundle)


def test_test_references_are_rejected_even_with_matching_ids(bundle):
    refs = bundle["references"]
    row = json.loads(refs.read_text())
    row["split"] = "test"
    refs.write_text(json.dumps(row) + "\n")
    bundle["references_sha256"] = sha256_file(refs)
    with pytest.raises(ValueError, match="only matching train"):
        load_train_rows(**bundle)


def test_gold_field_in_inference_is_rejected(bundle):
    path = bundle["inference"]
    row = json.loads(path.read_text())
    row["reference_text"] = "forbidden fixture field"
    path.write_text(json.dumps(row) + "\n")
    bundle["inference_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="gold-free"):
        load_train_rows(**bundle)


def test_no_silent_truncation_or_path_escape(bundle):
    path = bundle["inference"]
    row = json.loads(path.read_text())
    row["duration_s"] = 31.0
    path.write_text(json.dumps(row) + "\n")
    bundle["inference_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="no truncation"):
        load_train_rows(**bundle)
    row.update(duration_s=1.0, audio_uri="../outside.flac")
    path.write_text(json.dumps(row) + "\n")
    bundle["inference_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="escapes"):
        load_train_rows(**bundle)


def test_supplied_profiles_block_before_loading_models():
    root = Path(__file__).resolve().parents[1]
    for mode in ("lora", "full"):
        config = json.loads((root / f"configs/fine-tuning/whisper_small_{mode}.json").read_text())
        report = preflight(config)
        assert report["status"] == "blocked"
        assert "missing input: references" in report["blockers"]


def test_baseline_gate_requires_actual_matching_validation_evidence(tmp_path):
    from turkish_asr_rdl.training.acoustic import validate_baseline
    path = tmp_path / "synthetic-baseline-report.json"
    config = {"model": {"id": "fixture-only", "revision": "a" * 40},
              "dataset_release": "fixture-only", "split_manifest_sha256": "b" * 64,
              "inputs": {"baseline_report": str(path)}}
    report = {"status": "completed", "training_mode": "pretrained", "split": "validation",
              "model_id": "fixture-only", "revision": "a" * 40,
              "dataset_release": "fixture-only", "split_manifest_sha256": "b" * 64}
    path.write_text(json.dumps(report))
    validate_baseline(config)
    report["split"] = "test"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="baseline validation evidence"):
        validate_baseline(config)


def test_pending_metrics_need_explicit_protocol_and_real_hashed_outputs(tmp_path):
    from turkish_asr_rdl.training.acoustic import validate_baseline
    path = tmp_path / "baseline-evidence.json"
    config = {"model": {"id": "fixture-only", "revision": "a" * 40},
              "dataset_release": "fixture-only", "split_manifest_sha256": "b" * 64,
              "inputs": {"baseline_report": str(path)}}
    report = {"status": "completed", "scope": "validation_inference_only", "training_mode": "pretrained",
              "split": "validation", "model_id": "fixture-only", "revision": "a" * 40,
              "dataset_release": "fixture-only", "split_manifest_sha256": "b" * 64, "files": {}}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="explicit D32"):
        validate_baseline(config)
    config["allow_fixed_budget_with_pending_validation_metrics"] = True
    with pytest.raises(ValueError, match="actual inference artifacts"):
        validate_baseline(config)
    for name in ["predictions.jsonl", "nbest.jsonl", "manifest.json"]:
        artifact = tmp_path / name
        artifact.write_text("synthetic artifact for integrity test")
        report["files"][name] = sha256_file(artifact)
    path.write_text(json.dumps(report))
    validate_baseline(config)
    (tmp_path / "predictions.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="artifact path/hash"):
        validate_baseline(config)


def test_pretrained_adapter_shares_the_adapted_decoder_and_requires_pins():
    from turkish_asr_rdl.decoding.adapted import AdaptedAsrAdapter, PretrainedAsrAdapter
    parent = dict(id="fixture-only", revision="a" * 40, weight_file="model.safetensors", weight_sha256="b" * 64)
    adapter = PretrainedAsrAdapter(parent, family="whisper", system_id="fixture-only", device="cpu", precision="fp32")
    assert adapter.nbest.__func__ is AdaptedAsrAdapter.nbest
    assert adapter.transcribe.__func__ is AdaptedAsrAdapter.transcribe
    assert adapter.model_artifact_sha256 == parent["weight_sha256"]
    with pytest.raises(ValueError, match="pinned"):
        PretrainedAsrAdapter({**parent, "revision": "main"}, family="whisper", system_id="fixture-only")


def tiny_whisper():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(3)
    config = transformers.WhisperConfig(
        vocab_size=12, num_mel_bins=4, d_model=8, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16,
        max_source_positions=8, max_target_positions=16, decoder_start_token_id=1,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, suppress_tokens=[], begin_suppress_tokens=[],
    )
    return transformers.WhisperForConditionalGeneration(config)


def test_lora_changes_only_adapters_and_saves_compatible_weights(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import PeftModel
    from turkish_asr_rdl.training.whisper import configure_whisper
    from turkish_asr_rdl.training.acoustic import fit
    model = configure_whisper(tiny_whisper(), "lora", {"rank": 2, "alpha": 4, "dropout": 0.0})
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    batch = {"input_features": torch.randn(1, 4, 16), "labels": torch.tensor([[3, 4, 2]])}
    epochs = []
    history = fit(model, [{}] * 3, lambda rows: batch, epochs=1, accumulation=2,
                  learning_rate=0.01, seed=3, device="cpu", precision="fp32",
                  on_epoch=lambda epoch, facts: epochs.append(facts))
    changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], p)]
    assert changed and all("lora_" in name for name in changed)
    assert epochs[0]["optimizer_steps"] == 2  # includes the last partial group
    assert history[0]["train_loss"] > 0
    model.save_pretrained(tmp_path)
    loaded = PeftModel.from_pretrained(tiny_whisper(), tmp_path)
    assert any("lora_" in name for name, _ in loaded.named_parameters())


def test_full_mode_has_gradients_on_base_model():
    torch = pytest.importorskip("torch")
    from turkish_asr_rdl.training.whisper import configure_whisper
    model = configure_whisper(tiny_whisper(), "full", {})
    loss = model(input_features=torch.randn(1, 4, 16), labels=torch.tensor([[3, 4, 2]])).loss
    loss.backward()
    assert all(p.requires_grad for p in model.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())


def test_teacher_forced_score_uses_original_logits():
    torch = pytest.importorskip("torch")
    from turkish_asr_rdl.decoding.whisper import teacher_forced_score, scored_token_ids
    model = tiny_whisper().eval()
    features = torch.randn(1, 4, 16)
    ids = [1, 3, 4, 5, 6, 2]
    logits = model(input_features=features, decoder_input_ids=torch.tensor([ids[:-1]])).logits
    expected = logits[0].log_softmax(-1)[3, 6] + logits[0].log_softmax(-1)[4, 2]
    assert teacher_forced_score(model, features, ids, 4) == pytest.approx(float(expected.detach()), abs=1e-6)
    assert scored_token_ids([1, 3, 4, 5, 6, 2, 0, 0], 4, 2) == [6, 2]


def test_actual_tiny_whisper_beam_search_returns_scored_candidates():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from turkish_asr_rdl.decoding.whisper import whisper_nbest
    class Tokenizer:
        eos_token_id = 2
        def convert_tokens_to_ids(self, tokens):
            return [1, 3, 4, 5]
        def decode(self, tokens, skip_special_tokens):
            return " ".join(str(t) for t in tokens if t != 2)
    model = tiny_whisper().eval()
    candidates, dedup = whisper_nbest(model, SimpleNamespace(tokenizer=Tokenizer()),
                                     torch.randn(1, 4, 16), requested_n=2, max_new_tokens=2)
    assert 1 <= len(candidates) <= 2
    assert len(candidates) + len(dedup) == 2
    assert all(c["acoustic_log_score"] <= 0 and c["token_count"] > 0 for c in candidates)
