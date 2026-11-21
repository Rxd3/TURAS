"""Offline synthetic unit tests for Member 3 B4 (Wav2Vec2 CTC) and B5 (BiLSTM-CTC) baselines."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from turkish_asr_rdl.baselines.bilstm_ctc import BiLstmCtcBaselineAdapter, save_bilstm_checkpoint
from turkish_asr_rdl.baselines.ctc import (
    Wav2Vec2CtcBaselineAdapter,
    ctc_loss_cross_check,
)
from turkish_asr_rdl.decoding.interface import AudioItem, check_adapter
from turkish_asr_rdl.models.bilstm_ctc import (
    BiLSTMCTC,
    CharCtcTokenizer,
    LogMelSpectrogramExtractor,
    check_ctc_feasibility,
    run_3step_memory_probe,
)


class _DummyCharTokenizer:
    """38-token dummy CTC tokenizer mimicking mpoyraz/wav2vec2-xls-r-300m-cv7-turkish."""

    def __init__(self) -> None:
        chars = list("abcçdefgğhıijklmnoöprsştuüvyz'")
        self.vocab = ["<pad>", "<s>", "</s>", "<unk>", "|"] + chars  # 38 tokens
        self.pad_token_id = 0

    def __len__(self) -> int:
        return len(self.vocab)

    def decode(self, token_ids, *, group_tokens: bool = True, skip_special_tokens: bool = True) -> str:
        out: list[str] = []
        prev = None
        for idx in token_ids:
            if group_tokens and idx == prev:
                continue
            prev = idx
            if idx == 0:
                continue
            if idx in (1, 2, 3) and skip_special_tokens:
                continue
            tok = self.vocab[idx]
            out.append(" " if tok == "|" else tok)
        return "".join(out)


class _DummyFeatureExtractor:
    def __call__(self, samples: np.ndarray, *, sampling_rate: int = 16000, return_attention_mask: bool = True, return_tensors: str = "pt"):
        wav = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
        mask = torch.ones_like(wav, dtype=torch.long)
        return {"input_values": wav, "attention_mask": mask}


class _DummyWav2Vec2Model:
    """Produces deterministic bfloat16 logits encoding ' merhaba ' (with leading/trailing spaces)."""

    def __init__(self, tokenizer: _DummyCharTokenizer) -> None:
        self.dtype = torch.bfloat16
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(vocab_size=len(tokenizer), pad_token_id=0)
        self._tok_to_id = {ch: idx for idx, ch in enumerate(tokenizer.vocab)}

    def __call__(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        seq = ["|", "m", "e", "r", "h", "a", "b", "a", "|"]
        logits = torch.full((1, len(seq), self.config.vocab_size), -12.0, dtype=torch.float32)
        for t, ch in enumerate(seq):
            logits[0, t, self._tok_to_id[ch]] = 8.0
        return SimpleNamespace(logits=logits.to(dtype=torch.bfloat16))


def test_b4_config_loads_and_matches_adapter_contract() -> None:
    cfg_path = Path("configs/baselines/wav2vec2_xlsr_turkish.json")
    tok = _DummyCharTokenizer()
    adapter = Wav2Vec2CtcBaselineAdapter.from_config_file(
        cfg_path,
        device="cpu",
        model_loader=lambda: _DummyWav2Vec2Model(tok),
        tokenizer_loader=lambda: tok,
        feature_extractor_loader=lambda: _DummyFeatureExtractor(),
    )
    check_adapter(adapter)
    adapter.load()

    audio = np.zeros(16000, dtype=np.float32)
    hyps = adapter.transcribe(
        [
            AudioItem("utt-ok", audio),
            AudioItem("utt-empty", np.zeros(0, dtype=np.float32)),
            AudioItem("utt-nan", np.array([0.0, np.nan], dtype=np.float32)),
        ]
    )
    assert len(hyps) == 3
    assert hyps[0].status == "ok"
    assert hyps[0].text == " merhaba "
    assert hyps[0].selected_candidate_id == "ctc-0001"
    assert len(hyps[0].candidates) == 5
    assert hyps[0].candidates[0].text == " merhaba "
    assert hyps[1].status == "empty"
    assert hyps[2].status == "error"


def test_b4_rejects_lm_rescoring_and_bad_hash() -> None:
    cfg_path = Path("configs/baselines/wav2vec2_xlsr_turkish.json")
    with pytest.raises(ValueError, match="LFS OID"):
        Wav2Vec2CtcBaselineAdapter.from_config_file(
            cfg_path,
            model_artifact_sha256="f" * 64,
        )


def test_ctc_loss_cross_check_finite_for_feasible_sequence() -> None:
    logits = torch.randn((12, 38), dtype=torch.float32)
    loss_val = ctc_loss_cross_check(logits, [5, 6, 7], blank_id=0)
    assert np.isfinite(loss_val) and loss_val > 0.0


def test_b5_vocab_train_only_subsampling_4x_and_feasibility() -> None:
    tok = CharCtcTokenizer.fit_from_train_texts(["merhaba dünya", "ekonomi bütçe"])
    assert tok.blank_token_id == 0
    assert tok.unk_token_id == 1
    assert tok.space_token_id == 2
    assert tok.encode("merhaba!")[-1] == 1
    assert tok.decode(tok.encode("merhaba dünya"), group_tokens=False) == "merhaba dünya"

    extractor = LogMelSpectrogramExtractor(sample_rate_hz=16000, n_mels=80)
    wav = torch.randn((2, 16000), dtype=torch.float32)
    feats, feat_lens = extractor(wav)
    assert feats.shape[0] == 2 and feats.shape[2] == 80

    model = BiLSTMCTC(vocab_size=len(tok), n_mels=80, hidden_size=32, num_layers=2, dropout=0.1)
    logits, out_lens = model(feats, feat_lens)
    assert logits.shape == (2, int(out_lens[0].item()), len(tok))
    assert int(out_lens[0].item()) == (((int(feat_lens[0].item()) + 1) // 2) + 1) // 2

    assert check_ctc_feasibility([10, 4, 15], [8, 6, 15]) == ([0, 2], 1)
    assert check_ctc_feasibility(
        output_lengths=[10, 4, 15], target_lengths=[8, 6, 15]
    ) == ([0, 2], 1)


def test_b5_ctc_feasibility_requires_blanks_between_adjacent_repeats() -> None:
    targets = [[1, 1], [1, 1], [1, 2, 1], [1, 1, 1]]
    valid, dropped = check_ctc_feasibility([2, 3, 3, 4], [2, 2, 3, 3], targets=targets)

    assert valid == [1, 2]
    assert dropped == 2


def test_b5_ctc_feasibility_validates_target_lengths() -> None:
    with pytest.raises(ValueError, match="same number of items"):
        check_ctc_feasibility([3], [2], targets=[])
    with pytest.raises(ValueError, match="Target 0 length"):
        check_ctc_feasibility([3], [2], targets=[[1]])


def test_b5_3step_probe_and_adapter_roundtrip(tmp_path: Path) -> None:
    tok = CharCtcTokenizer.fit_from_train_texts(["vergi kanunu", "bütçe dengesi"])
    model = BiLSTMCTC(vocab_size=len(tok), n_mels=80, hidden_size=32, num_layers=2, dropout=0.1)
    extractor = LogMelSpectrogramExtractor()

    probe = run_3step_memory_probe(
        model,
        extractor,
        batch_size=2,
        clip_seconds=1.0,
        target_length=8,
        device="cpu",
        steps=3,
    )
    assert probe.steps_completed == 3
    assert len(probe.losses) == 3
    assert all(np.isfinite(x) for x in probe.losses)

    ckpt_dir = tmp_path / "b5_ckpt"
    manifest_sha = save_bilstm_checkpoint(ckpt_dir, model, tok, metadata={"epoch": 1})

    adapter = BiLstmCtcBaselineAdapter.from_config_file(
        "configs/baselines/bilstm_ctc.json",
        checkpoint_dir=ckpt_dir,
        model_artifact_sha256=manifest_sha,
        device="cpu",
    )
    check_adapter(adapter)
    adapter.load()

    hyps = adapter.transcribe(
        [
            AudioItem("utt-1", np.random.default_rng(42).standard_normal(16000).astype(np.float32)),
            AudioItem("utt-2", np.zeros(0, dtype=np.float32)),
        ]
    )
    assert len(hyps) == 2
    assert hyps[0].status in {"ok", "empty"}
    assert hyps[1].status == "empty"