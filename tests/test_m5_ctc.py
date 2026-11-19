"""Hand-computed/synthetic CTC tests; no corpus predictions are produced."""
from itertools import product
import math

import pytest

from turkish_asr_rdl.decoding.ctc import prefix_beam_search, sequence_log_probability, logadd
from turkish_asr_rdl.decoding.rescoring import rescore
from turkish_asr_rdl.training.ctc import label_text


def collapse(path):
    return tuple(token for i, token in enumerate(path) if token != 0 and (i == 0 or token != path[i - 1]))


def test_exact_ctc_probability_matches_every_alignment_including_repeats():
    probabilities = [[0.2, 0.5, 0.3], [0.4, 0.4, 0.2], [0.1, 0.6, 0.3]]
    frames = [[math.log(v) for v in frame] for frame in probabilities]
    expected = {}
    for path in product(range(3), repeat=3):
        transcript = collapse(path)
        mass = math.prod(probabilities[t][token] for t, token in enumerate(path))
        expected[transcript] = expected.get(transcript, 0) + mass
    for transcript, mass in expected.items():
        assert sequence_log_probability(frames, transcript, 0) == pytest.approx(math.log(mass), abs=1e-12)
    beams = dict(prefix_beam_search(frames, blank_id=0, beam_size=30))
    for transcript, mass in expected.items():
        assert beams[transcript] == pytest.approx(math.log(mass), abs=1e-12)


def test_impossible_repeated_target_is_negative_infinity():
    assert sequence_log_probability([[math.log(.5), math.log(.5)]], [1, 1], 0) == -math.inf
    assert logadd(-math.inf, -math.inf) == -math.inf
    with pytest.raises(ValueError, match="sum to one"):
        prefix_beam_search([[-1, -1]], blank_id=0)


def test_joint_selection_disabled_null_terms_and_original_rank_tie():
    candidates = [dict(candidate_id="b", rank=2, token_count=2, acoustic_log_score=-4,
                       lm_log_score=None, rdl_log_score=-.2),
                  dict(candidate_id="a", rank=1, token_count=1, acoustic_log_score=-2,
                       lm_log_score=None, rdl_log_score=-1)]
    winner, rows = rescore(candidates)
    assert winner == "a"
    assert "combined_score" not in candidates[0]
    winner, _ = rescore(candidates, rdl_weight=1)
    assert winner == "b"
    with pytest.raises(ValueError, match="unavailable"):
        rescore(candidates, lm_weight=1)
    assert rows[0]["rank"] == 2


def test_turkish_training_label_view_preserves_dotted_dotless_i_and_numbers():
    assert label_text("IŞIK İZMİR, 30!") == "ışık izmir 30"
    # Numbers remain visible for the tokenizer OOV gate, never silently removed.


def test_real_tiny_ctc_has_finite_loss_and_frozen_frontend_gradients():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Wav2Vec2Config(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
        conv_dim=(4, 4), conv_stride=(2, 2), conv_kernel=(3, 3),
        num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
        vocab_size=6, pad_token_id=0, apply_spec_augment=False,
        ctc_loss_reduction="mean", ctc_zero_infinity=False,
    )
    model = transformers.Wav2Vec2ForCTC(config)
    model.freeze_feature_encoder()
    loss = model(input_values=torch.randn(1, 128), labels=torch.tensor([[1, 1, 2]])).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert model.lm_head.weight.grad is not None
    assert all(p.grad is None for p in model.wav2vec2.feature_extractor.parameters())


def test_real_ctc_tokenizer_rejects_unknown_numeric_training_labels(tmp_path):
    import json
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    transformers = pytest.importorskip("transformers")
    from turkish_asr_rdl.training.ctc import build_collator
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3}))
    tokenizer = transformers.Wav2Vec2CTCTokenizer(str(vocab), pad_token="[PAD]", unk_token="[UNK]", word_delimiter_token="|")
    processor = transformers.Wav2Vec2Processor(transformers.Wav2Vec2FeatureExtractor(sampling_rate=16000), tokenizer)
    audio = tmp_path / "synthetic.flac"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="OOV"):
        build_collator(processor)([{"audio_path": audio, "duration_s": 1.0, "reference_text": "30"}])
    from turkish_asr_rdl.training.ctc import audit_targets
    audit = audit_targets([{"split": "train", "utterance_id": "fixture-only", "reference_text": "30"}], tokenizer)
    assert audit["unsupported_characters"] == {"0": 1, "3": 1}
    assert audit["affected_utterance_ids"] == ["fixture-only"]
    with pytest.raises(ValueError, match="training references only"):
        audit_targets([{"split": "test"}], tokenizer)


def test_explicit_extension_preserves_old_rows_ids_and_roundtrips(tmp_path):
    import json
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from turkish_asr_rdl.training.ctc import extend_vocabulary
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "<s>": 4, "</s>": 5}))
    tokenizer = transformers.Wav2Vec2CTCTokenizer(str(vocab))
    model = transformers.Wav2Vec2ForCTC(transformers.Wav2Vec2Config(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
        conv_dim=(4,), conv_stride=(2,), conv_kernel=(3,), num_conv_pos_embedding_groups=2,
        vocab_size=len(tokenizer), pad_token_id=0, apply_spec_augment=False))
    old_weights = model.lm_head.weight.detach().clone()
    old_bias = model.lm_head.bias.detach().clone()
    old_vocab = tokenizer.get_vocab().copy()
    info = extend_vocabulary(model, tokenizer, ["3", "0", "\u00e2"])
    assert model.config.vocab_size == len(tokenizer) == info["new_vocab_size"] == 9
    assert torch.equal(model.lm_head.weight[:6], old_weights)
    assert torch.equal(model.lm_head.bias[:6], old_bias)
    assert all(tokenizer.get_vocab()[k] == v for k, v in old_vocab.items())
    text = "a30 \u00e2a"
    assert tokenizer.unk_token_id not in tokenizer(text).input_ids
    assert tokenizer.decode(tokenizer(text).input_ids, group_tokens=False) == text
    saved = tmp_path / "extended"
    tokenizer.save_pretrained(saved)
    restored = transformers.Wav2Vec2CTCTokenizer.from_pretrained(saved)
    assert restored.get_vocab() == tokenizer.get_vocab()
    assert restored.decode(restored(text).input_ids, group_tokens=False) == text
