"""Track B5 from-scratch BiLSTM-CTC model, feature extractor, train-only vocab, and probe utilities."""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCtcTokenizer:
    """Character-level CTC tokenizer built strictly from frozen train transcripts.

    Special tokens match configs/baselines/bilstm_ctc.json:
      0 -> <blank>
      1 -> <unk>
      2 -> ' ' (space)
    """

    BLANK_TOKEN = "<blank>"
    UNK_TOKEN = "<unk>"
    SPACE_TOKEN = " "

    def __init__(self, id_to_token: Sequence[str]) -> None:
        if len(id_to_token) < 3:
            raise ValueError("Vocabulary must contain at least <blank>, <unk>, and space")
        if id_to_token[0] != self.BLANK_TOKEN or id_to_token[1] != self.UNK_TOKEN or id_to_token[2] != self.SPACE_TOKEN:
            raise ValueError("Special token IDs must be 0:<blank>, 1:<unk>, 2:' '")
        self.id_to_token: list[str] = list(id_to_token)
        self.token_to_id: dict[str, int] = {tok: idx for idx, tok in enumerate(self.id_to_token)}
        self.pad_token_id: int = 0
        self.blank_token_id: int = 0
        self.unk_token_id: int = 1
        self.space_token_id: int = 2

    def __len__(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def fit_from_train_texts(cls, train_texts: Iterable[str]) -> CharCtcTokenizer:
        """Build sorted character inventory strictly from train-split transcripts."""
        chars: set[str] = set()
        for text in train_texts:
            for ch in text:
                if ch != " ":
                    chars.add(ch)
        vocab = [cls.BLANK_TOKEN, cls.UNK_TOKEN, cls.SPACE_TOKEN] + sorted(chars)
        return cls(vocab)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(ch, self.unk_token_id) for ch in text]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        group_tokens: bool = True,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs; compatible with turkish_asr_rdl.decoding.ctc.ctc_nbest."""
        chars: list[str] = []
        prev: int | None = None
        for raw_id in token_ids:
            idx = int(raw_id)
            if group_tokens and idx == prev:
                continue
            prev = idx
            if idx == self.blank_token_id:
                continue
            if idx == self.unk_token_id:
                if not skip_special_tokens:
                    chars.append(self.UNK_TOKEN)
                continue
            if 0 <= idx < len(self.id_to_token):
                chars.append(self.id_to_token[idx])
        return "".join(chars)

    def to_dict(self) -> dict[str, Any]:
        return {"id_to_token": list(self.id_to_token)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CharCtcTokenizer:
        return cls(payload["id_to_token"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharCtcTokenizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class LogMelSpectrogramExtractor(nn.Module):
    """Pure-PyTorch 80-bin log-Mel extractor (16 kHz, 25 ms window, 10 ms hop) with per-utterance CMVN."""

    def __init__(
        self,
        sample_rate_hz: int = 16000,
        n_mels: int = 80,
        win_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        f_min: float = 20.0,
        f_max: float = 7600.0,
    ) -> None:
        super().__init__()
        self.sample_rate_hz = sample_rate_hz
        self.n_mels = n_mels
        self.win_length = int(round(sample_rate_hz * win_length_ms / 1000.0))  # 400
        self.hop_length = int(round(sample_rate_hz * hop_length_ms / 1000.0))  # 160
        self.n_fft = 512
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer(
            "mel_fb",
            self._build_mel_filterbank(sample_rate_hz, self.n_fft, n_mels, f_min, f_max),
            persistent=False,
        )

    @staticmethod
    def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @classmethod
    def _build_mel_filterbank(
        cls,
        sr: int,
        n_fft: int,
        n_mels: int,
        f_min: float,
        f_max: float,
    ) -> torch.Tensor:
        n_freqs = n_fft // 2 + 1
        fft_freqs = torch.linspace(0.0, sr / 2.0, n_freqs)
        mel_pts = torch.linspace(
            float(cls._hz_to_mel(torch.tensor(f_min))),
            float(cls._hz_to_mel(torch.tensor(f_max))),
            n_mels + 2,
        )
        hz_pts = cls._mel_to_hz(mel_pts)
        fb = torch.zeros((n_freqs, n_mels), dtype=torch.float32)
        for m in range(n_mels):
            left, center, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
            up = (fft_freqs - left) / torch.clamp(center - left, min=1e-6)
            down = (right - fft_freqs) / torch.clamp(right - center, min=1e-6)
            fb[:, m] = torch.clamp(torch.minimum(up, down), min=0.0)
        return fb

    def frame_count(self, num_samples: int) -> int:
        if num_samples < self.win_length:
            return 0
        return (num_samples - self.win_length) // self.hop_length + 1

    def forward(self, waveforms: torch.Tensor, sample_lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized (B, T_mel, n_mels) log-Mel features and valid frame lengths."""
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        bsz, max_len = waveforms.shape
        if max_len < self.win_length:
            waveforms = F.pad(waveforms, (0, self.win_length - max_len))
            max_len = waveforms.shape[1]

        stft = torch.stft(
            waveforms.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = stft.abs().pow(2.0).transpose(1, 2)  # (B, T_mel, n_freqs)
        mel = torch.matmul(power, self.mel_fb)
        log_mel = torch.log(torch.clamp(mel, min=1e-10))

        t_mel = log_mel.shape[1]
        if sample_lengths is None:
            feat_lengths = torch.full((bsz,), t_mel, dtype=torch.long, device=waveforms.device)
        else:
            feat_lengths = torch.tensor(
                [max(1, min(t_mel, self.frame_count(int(s)))) for s in sample_lengths.tolist()],
                dtype=torch.long,
                device=waveforms.device,
            )

        # Per-utterance CMVN over valid frames
        out = torch.zeros_like(log_mel)
        for idx in range(bsz):
            valid_t = int(feat_lengths[idx].item())
            seg = log_mel[idx, :valid_t]
            mean = seg.mean(dim=0, keepdim=True)
            std = seg.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-5)
            out[idx, :valid_t] = (seg - mean) / std
        return out, feat_lengths


class Conv2dSubsampler4x(nn.Module):
    """Two stride-2 Conv2D layers yielding exact 4x temporal subsampling."""

    SUBSAMPLING_FACTOR = 4

    def __init__(self, in_mels: int = 80, channels: int = 32) -> None:
        super().__init__()
        self.in_mels = in_mels
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        sub_freq = math.ceil(math.ceil(in_mels / 2.0) / 2.0)
        self.out_dim = channels * sub_freq

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        t1 = torch.div(input_lengths + 1, 2, rounding_mode="floor")
        t2 = torch.div(t1 + 1, 2, rounding_mode="floor")
        return torch.clamp(t2, min=1)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # features: (B, T, F) -> (B, 1, T, F)
        x = self.conv(features.unsqueeze(1))
        bsz, ch, t_sub, f_sub = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(bsz, t_sub, ch * f_sub)
        return x, self.output_lengths(lengths)


class BiLSTMCTC(nn.Module):
    """From-scratch BiLSTM-CTC acoustic model (Track B5)."""

    def __init__(
        self,
        vocab_size: int,
        *,
        n_mels: int = 80,
        conv_channels: int = 32,
        hidden_size: int = 320,
        num_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if vocab_size < 3:
            raise ValueError("vocab_size must be at least 3")
        self.vocab_size = vocab_size
        self.n_mels = n_mels
        self.conv_channels = conv_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.subsampling_factor = Conv2dSubsampler4x.SUBSAMPLING_FACTOR

        self.subsampler = Conv2dSubsampler4x(in_mels=n_mels, channels=conv_channels)
        self.input_proj = nn.Linear(self.subsampler.out_dim, hidden_size)
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sub_x, out_lengths = self.subsampler(features, feature_lengths)
        proj = torch.relu(self.input_proj(sub_x))
        enc_out, _ = self.encoder(proj)
        logits = self.classifier(self.dropout_layer(enc_out))
        return logits, out_lengths


def check_ctc_feasibility(
    output_lengths: Sequence[int],
    target_lengths: Sequence[int],
) -> tuple[list[int], int]:
    """Enforce T_out >= L_target CTC feasibility rule and count dropped clips."""
    valid_indices: list[int] = []
    dropped = 0
    for idx, (t_out, l_tgt) in enumerate(zip(output_lengths, target_lengths)):
        if l_tgt > 0 and t_out >= l_tgt:
            valid_indices.append(idx)
        else:
            dropped += 1
    return valid_indices, dropped


@dataclass(frozen=True)
class ProbeResult:
    """Summary of B5 mandatory 3-step memory/timing probe."""

    steps_completed: int
    peak_vram_gib: float
    mean_step_ms: float
    losses: tuple[float, ...]


def run_3step_memory_probe(
    model: BiLSTMCTC,
    extractor: LogMelSpectrogramExtractor,
    *,
    batch_size: int = 16,
    clip_seconds: float = 15.0,
    target_length: int = 80,
    device: str = "cpu",
    steps: int = 3,
) -> ProbeResult:
    """Execute B5's mandatory 3-step forward/backward memory & timing probe."""
    import time

    dev = torch.device(device)
    model = model.to(dev).train()
    extractor = extractor.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_samples = int(clip_seconds * extractor.sample_rate_hz)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    losses: list[float] = []
    t0 = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        wav = torch.randn((batch_size, num_samples), device=dev)
        feats, feat_lens = extractor(wav)
        logits, out_lens = model(feats, feat_lens)
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        targets = torch.randint(2, model.vocab_size, (batch_size, target_length), device=dev)
        tgt_lens = torch.full((batch_size,), target_length, dtype=torch.long, device=dev)
        loss = F.ctc_loss(log_probs, targets, out_lens, tgt_lens, blank=0, zero_infinity=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    peak_gib = (
        float(torch.cuda.max_memory_allocated(dev)) / (1024.0**3)
        if dev.type == "cuda"
        else 0.0
    )
    return ProbeResult(
        steps_completed=steps,
        peak_vram_gib=round(peak_gib, 4),
        mean_step_ms=round(elapsed_ms / max(steps, 1), 2),
        losses=tuple(losses),
    )