"""Track B5 from-scratch BiLSTM-CTC baseline adapter implementing AsrAdapter."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from turkish_asr_rdl.baselines.ctc import greedy_ctc_decode
from turkish_asr_rdl.baselines.whisper import get_shared_interface
from turkish_asr_rdl.decoding.ctc import ctc_nbest, sequence_log_probability
from turkish_asr_rdl.models.bilstm_ctc import (
    BiLSTMCTC,
    CharCtcTokenizer,
    LogMelSpectrogramExtractor,
)


def save_bilstm_checkpoint(
    directory: str | Path,
    model: BiLSTMCTC,
    tokenizer: CharCtcTokenizer,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save B5 weights + tokenizer + manifest and return manifest SHA-256."""
    out_dir = Path(directory).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / "bilstm_ctc.pt"
    vocab_path = out_dir / "vocab.json"
    manifest_path = out_dir / "checkpoint-manifest.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": {
                "vocab_size": model.vocab_size,
                "n_mels": model.n_mels,
                "conv_channels": model.conv_channels,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "dropout": model.dropout,
            },
        },
        weights_path,
    )
    tokenizer.save(vocab_path)

    manifest = {
        "architecture": "BiLSTM-CTC",
        "files": {
            "bilstm_ctc.pt": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "vocab.json": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        },
        "metadata": dict(metadata or {}),
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return hashlib.sha256(manifest_bytes).hexdigest()


class BiLstmCtcBaselineAdapter:
    """Reusable AsrAdapter implementation for Track B5 BiLSTM-CTC baseline."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        checkpoint_dir: str | Path | None = None,
        model_artifact_sha256: str | None = None,
        device: str | None = None,
        requested_n: int | None = None,
        beam_size: int = 10,
        model: BiLSTMCTC | None = None,
        tokenizer: CharCtcTokenizer | None = None,
        shared_path: str | Path | None = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise ValueError(f"config must be a mapping, got {type(config).__name__}")

        self.system_id: str = str(config.get("system_id", "baseline-bilstm-ctc"))
        model_cfg = config.get("model", {})
        feat_cfg = model_cfg.get("input_features", {})
        enc_cfg = model_cfg.get("provisional_encoder", {})
        dec_cfg = config.get("decoding_policy", {})

        self.sampling_rate_hz: int = int(feat_cfg.get("sample_rate_hz", 16000))
        if self.sampling_rate_hz != 16000:
            raise ValueError(f"BiLSTM-CTC requires 16000 Hz audio, got {self.sampling_rate_hz}")

        self.n_mels: int = int(feat_cfg.get("n_mels", 80))
        self.win_length_ms: float = float(feat_cfg.get("win_length_ms", 25.0))
        self.hop_length_ms: float = float(feat_cfg.get("hop_length_ms", 10.0))
        self.hidden_size: int = int(enc_cfg.get("hidden_size", 320))
        self.num_layers: int = int(enc_cfg.get("num_layers", 4))
        self.dropout: float = float(enc_cfg.get("dropout", 0.2))
        self.blank_token_id: int = int(model_cfg.get("ctc_head", {}).get("blank_id", 0))

        if dec_cfg.get("lm_rescoring", False):
            raise ValueError("LM rescoring is excluded from B5 baseline")

        self.decode_mode: str = str(dec_cfg.get("primary_decode_mode", "greedy_ctc"))
        if self.decode_mode not in {"greedy_ctc", "ctc_prefix_beam_search"}:
            raise ValueError(f"Unsupported B5 decode mode: {self.decode_mode!r}")

        cfg_req_n = int(dec_cfg.get("num_return_sequences", 1))
        req_n = requested_n if requested_n is not None else cfg_req_n
        if not isinstance(req_n, int) or isinstance(req_n, bool) or req_n < 1:
            raise ValueError(f"requested_n must be >= 1, got {req_n}")
        self.requested_n: int = req_n
        self.beam_size: int = max(beam_size, self.requested_n)

        self.device: str = device or str(model_cfg.get("device", "cuda"))
        self.precision: str = "float32"
        self.score_convention_id: str | None = (
            "ctc-greedy-v1" if self.decode_mode == "greedy_ctc" else "ctc-exact-forward-logprob-v1"
        )

        self.checkpoint_dir: Path | None = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        if model_artifact_sha256 is not None:
            if not (
                isinstance(model_artifact_sha256, str)
                and len(model_artifact_sha256) == 64
                and all(c in "0123456789abcdef" for c in model_artifact_sha256)
            ):
                raise ValueError("model_artifact_sha256 must be a 64-character lowercase hex SHA-256")
            self.model_artifact_sha256 = model_artifact_sha256
        else:
            # Deterministic sentinel for unpersisted in-memory unit-test instances
            self.model_artifact_sha256 = "0" * 64

        self.config: Mapping[str, Any] = {
            "system_id": self.system_id,
            "model": {
                "architecture": "BiLSTM-CTC",
                "n_mels": self.n_mels,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "precision": self.precision,
                "device": self.device,
            },
            "decoding": {
                "primary_decode_mode": self.decode_mode,
                "requested_n": self.requested_n,
                "beam_size": self.beam_size,
                "lm_rescoring": False,
                "strip_leading_trailing_whitespace_in_raw": False,
                "score_convention_id": self.score_convention_id,
            },
        }
        self.decoder_config_sha256: str = hashlib.sha256(
            json.dumps(self.config["decoding"], sort_keys=True).encode("utf-8")
        ).hexdigest()

        self.extractor = LogMelSpectrogramExtractor(
            sample_rate_hz=self.sampling_rate_hz,
            n_mels=self.n_mels,
            win_length_ms=self.win_length_ms,
            hop_length_ms=self.hop_length_ms,
        )
        self.model: BiLSTMCTC | None = model
        self.tokenizer: CharCtcTokenizer | None = tokenizer
        self._shared_path = shared_path

    @classmethod
    def from_config_file(cls, path: str | Path, **kwargs: Any) -> BiLstmCtcBaselineAdapter:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data, **kwargs)

    def load(self) -> None:
        """Load verified B5 checkpoint from disk or prepare injected in-memory model."""
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA is not available on this host; cannot load {self.system_id} on {self.device}")

        if self.model is not None and self.tokenizer is not None:
            self.extractor.to(self.device).eval()
            self.model.to(self.device).eval()
            return

        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when model/tokenizer are not injected")

        manifest_path = self.checkpoint_dir / "checkpoint-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint-manifest.json in {self.checkpoint_dir}")
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != self.model_artifact_sha256:
            raise ValueError("B5 checkpoint-manifest.json SHA-256 mismatch")

        manifest = json.loads(manifest_bytes.decode("utf-8"))
        for fname, expected_sha in manifest["files"].items():
            member = (self.checkpoint_dir / fname).resolve()
            if not member.is_relative_to(self.checkpoint_dir) or not member.is_file():
                raise ValueError(f"Invalid checkpoint member: {fname}")
            if hashlib.sha256(member.read_bytes()).hexdigest() != expected_sha:
                raise ValueError(f"Checkpoint file hash mismatch for {fname}")

        self.tokenizer = CharCtcTokenizer.load(self.checkpoint_dir / "vocab.json")
        ckpt = torch.load(self.checkpoint_dir / "bilstm_ctc.pt", map_location="cpu", weights_only=True)
        arch = ckpt["arch"]
        self.model = BiLSTMCTC(
            vocab_size=arch["vocab_size"],
            n_mels=arch["n_mels"],
            conv_channels=arch["conv_channels"],
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            dropout=arch["dropout"],
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.extractor.to(self.device).eval()
        self.model.to(self.device).eval()

    def nbest(self, samples: np.ndarray, *, deduplicate: bool = False) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Decode one 16 kHz mono utterance using greedy CTC or prefix beam search."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("BiLstmCtcBaselineAdapter must be loaded before inference")
        if not isinstance(samples, np.ndarray) or samples.ndim != 1 or len(samples) == 0 or not np.isfinite(samples).all():
            raise ValueError("Expected non-empty finite 1D numpy array")

        with torch.inference_mode():
            wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).to(self.device)
            feats, feat_lens = self.extractor(wav)
            logits, _ = self.model(feats, feat_lens)
            frames = logits[0].float().log_softmax(dim=-1).cpu().tolist()

        if self.decode_mode == "greedy_ctc":
            text, tok_ids = greedy_ctc_decode(frames, self.tokenizer, blank_id=self.blank_token_id)
            seq_score = sequence_log_probability(frames, tok_ids, self.blank_token_id)
            return (
                [
                    {
                        "candidate_id": "ctc-0001",
                        "rank": 1,
                        "text": text,
                        "token_ids": tok_ids,
                        "token_count": len(tok_ids),
                        "acoustic_log_score": seq_score,
                        "acoustic_score_kind": "ctc_sequence_log_probability",
                        "decoder_sequence_score": seq_score,
                        "lm_log_score": None,
                        "rdl_log_score": None,
                        "combined_score": None,
                        "score_details": {
                            "log_base": "e",
                            "blank_id": self.blank_token_id,
                            "nbest_algorithm": "greedy_ctc",
                            "beam_size": 1,
                            "sequence_score_exact": True,
                            "beam_score_approximate": False,
                        },
                    }
                ],
                {},
            )

        return ctc_nbest(
            frames,
            self.tokenizer,
            blank_id=self.blank_token_id,
            requested_n=self.requested_n,
            beam_size=self.beam_size,
            deduplicate=deduplicate,
        )

    def transcribe(self, batch: Sequence[Any]) -> list[Any]:
        """Transcribe a batch of reference-free AudioItems preserving order and raw whitespace."""
        _, Candidate, Hypothesis, _, _, _ = get_shared_interface(self._shared_path)

        results: list[Any] = []
        for item in batch:
            utt_id = getattr(item, "utterance_id", None)
            if not isinstance(utt_id, str) or not utt_id:
                raise ValueError("Each AudioItem must have a non-empty string utterance_id")

            samples = getattr(item, "samples", None)
            if isinstance(samples, np.ndarray) and samples.ndim == 1 and len(samples) == 0:
                results.append(
                    Hypothesis(
                        utterance_id=utt_id,
                        status="empty",
                        text="",
                        candidates=(),
                        selected_candidate_id=None,
                    )
                )
                continue

            if not isinstance(samples, np.ndarray) or samples.ndim != 1 or not np.isfinite(samples).all():
                results.append(
                    Hypothesis(
                        utterance_id=utt_id,
                        status="error",
                        text=None,
                        candidates=(),
                        error_code="InvalidAudio",
                        error_message="Audio samples must be a 1D numpy array of finite floats",
                        selected_candidate_id=None,
                    )
                )
                continue

            if len(samples) > 480000:
                results.append(
                    Hypothesis(
                        utterance_id=utt_id,
                        status="error",
                        text=None,
                        candidates=(),
                        error_code="AudioTooLong",
                        error_message=f"Audio exceeds 30 s limit (480,000 samples): got {len(samples)}",
                        selected_candidate_id=None,
                    )
                )
                continue

            try:
                raw_cands, _ = self.nbest(samples, deduplicate=False)
                if not raw_cands:
                    results.append(
                        Hypothesis(
                            utterance_id=utt_id,
                            status="empty",
                            text="",
                            candidates=(),
                            selected_candidate_id=None,
                        )
                    )
                    continue

                candidates = tuple(
                    Candidate(
                        candidate_id=c["candidate_id"],
                        rank=c["rank"],
                        text=c["text"],
                        token_ids=tuple(c["token_ids"]),
                        acoustic_log_score=c["acoustic_log_score"],
                        acoustic_score_kind=c["acoustic_score_kind"],
                        decoder_sequence_score=c.get("decoder_sequence_score"),
                        score_details=copy.deepcopy(c.get("score_details", {})),
                        lm_log_score=c.get("lm_log_score"),
                        rdl_log_score=c.get("rdl_log_score"),
                        combined_score=c.get("combined_score"),
                    )
                    for c in raw_cands
                )
                top_text = candidates[0].text
                status = "ok" if top_text else "empty"
                selected_id = candidates[0].candidate_id if status == "ok" else None
                results.append(
                    Hypothesis(
                        utterance_id=utt_id,
                        status=status,
                        text=top_text,
                        candidates=candidates,
                        selected_candidate_id=selected_id,
                    )
                )
            except Exception as exc:
                results.append(
                    Hypothesis(
                        utterance_id=utt_id,
                        status="error",
                        text=None,
                        candidates=(),
                        error_code=type(exc).__name__,
                        error_message=f"B5 BiLSTM-CTC decoding failed: {exc}",
                        selected_candidate_id=None,
                    )
                )

        return results