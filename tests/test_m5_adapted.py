"""Safe checkpoint loading tests with a locally initialized tiny model."""
import json
from types import SimpleNamespace

import pytest

from turkish_asr_rdl.training.acoustic import sha256_file


def test_saved_full_whisper_loads_and_generates_without_pretrained_download(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    transformers = pytest.importorskip("transformers")
    from turkish_asr_rdl.decoding.adapted import AdaptedAsrAdapter
    config = transformers.WhisperConfig(
        vocab_size=12, num_mel_bins=4, d_model=8, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16,
        max_source_positions=8, max_target_positions=16, decoder_start_token_id=1,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, suppress_tokens=[], begin_suppress_tokens=[])
    transformers.WhisperForConditionalGeneration(config).save_pretrained(tmp_path)
    manifest = {"training_mode": "full", "parent": {"id": "synthetic-fixture-only"},
                "files": {p.name: sha256_file(p) for p in tmp_path.iterdir() if p.is_file()}}
    path = tmp_path / "checkpoint-manifest.json"
    path.write_text(json.dumps(manifest))
    class Tokenizer:
        eos_token_id = 2
        def convert_tokens_to_ids(self, tokens):return [1, 3, 4, 5]
        def decode(self, tokens, skip_special_tokens):return " ".join(str(t) for t in tokens if t != 2)
    features = transformers.WhisperFeatureExtractor(feature_size=4, sampling_rate=16000,
                                                    chunk_length=1, hop_length=1000, n_fft=2000)
    processor = SimpleNamespace(tokenizer=Tokenizer(), feature_extractor=features)
    monkeypatch.setattr(transformers.WhisperProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    adapter = AdaptedAsrAdapter(tmp_path, sha256_file(path), system_id="synthetic-fixture-only",
                                requested_n=2, max_new_tokens=2)
    adapter.load()
    rows, _ = adapter.nbest(np.zeros(16000, dtype=np.float32))
    assert rows and all(row["acoustic_log_score"] <= 0 for row in rows)
    with pytest.raises(ValueError, match="at most 30 seconds"):
        adapter.nbest(np.zeros(480001, dtype=np.float32))
    # Any changed weight/config fails before a new load or network access.
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        adapter.load()
