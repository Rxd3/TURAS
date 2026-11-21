"""Track B4 CTC baseline adapter for pinned Wav2Vec2/XLS-R Turkish checkpoints.

Implements the shared AsrAdapter protocol using explicit Wav2Vec2CTCTokenizer +
Wav2Vec2FeatureExtractor loading (bypassing KenLM / Wav2Vec2ProcessorWithLM),
bfloat16 forward passes cast to float32 log-probabilities, and M5's shared
turkish_asr_rdl.decoding.ctc prefix beam search.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from turkish_asr_rdl.baselines.whisper import get_shared_interface
from turkish_asr_rdl.decoding.ctc import ctc_nbest

PINNED_CTC_REVISIONS: dict[str, str] = {
    "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish": "708639f50559d7970f462e13ec64d3f059ca89f6",
    "mpoyraz/wav2vec2-xls-r-300m-cv8-turkish": "bdd0bb878d8bf59509611631e334eb073ba57773",
}

PINNED_CTC_LFS_OIDS: dict[str, str] = {
    "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish": "008d69c82bf45c82ec5485e550abc86ddabefe75080f22e1ebfdc1699889d82e",
    "mpoyraz/wav2vec2-xls-r-300m-cv8-turkish": "dbd7a9f776a232c0dd4cbbee44025c8056246d83ad4de930b049e649e2fe402c",
}

ALLOWED_CTC_PRECISIONS: frozenset[str] = frozenset(
    {"bfloat16", "bf16", "float16", "fp16", "float32", "fp32"}
)


def greedy_ctc_decode(
    log_probs: np.ndarray | Sequence[Sequence[float]],
    tokenizer: Any,
    *,
    blank_id: int = 0,
) -> tuple[str, list[int]]:
    """Diagnostic greedy CTC collapse (collapse consecutive duplicates, drop blanks)."""
    arr = np.asarray(log_probs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return "", []
    raw_ids = arr.argmax(axis=-1).tolist()
    collapsed: list[int] = []
    prev: int | None = None
    for tok in raw_ids:
        if tok != prev and tok != blank_id:
            collapsed.append(int(tok))
        prev = int(tok)
    text = tokenizer.decode(collapsed, group_tokens=False, skip_special_tokens=True)
    return text, collapsed


def ctc_loss_cross_check(
    logits_fp32: Any,
    target_ids: Sequence[int],
    *,
    blank_id: int = 0,
) -> float:
    """Compute per-utterance CTCLoss(reduction='none', zero_infinity=False) cross-check."""
    import torch
    import torch.nn.functional as F

    log_probs = F.log_softmax(logits_fp32.float(), dim=-1).transpose(0, 1)  # (T, B=1, V)
    targets = torch.tensor(list(target_ids), dtype=torch.long, device=log_probs.device)
    input_lengths = torch.tensor([log_probs.shape[0]], dtype=torch.long, device=log_probs.device)
    target_lengths = torch.tensor([len(target_ids)], dtype=torch.long, device=log_probs.device)
    loss = F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_id,
        reduction="none",
        zero_infinity=False,
    )
    return float(loss.squeeze().item())


class Wav2Vec2CtcBaselineAdapter:
    """Reusable AsrAdapter implementation for Track B4 Turkish XLS-R CTC baselines."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        device: str | None = None,
        precision: str | None = None,
        requested_n: int | None = None,
        checkpoint_path: str | Path | None = None,
        model_artifact_sha256: str | None = None,
        model_loader: Callable[..., Any] | None = None,
        tokenizer_loader: Callable[..., Any] | None = None,
        feature_extractor_loader: Callable[..., Any] | None = None,
        nbest_fn: Callable[..., Any] | None = None,
        shared_path: str | Path | None = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise ValueError(f"config must be a mapping, got {type(config).__name__}")

        self.system_id: str = str(config.get("system_id", "B4-xlsr-tr-cv7"))
        model_cfg = config.get("model", {})
        proc_cfg = config.get("processor", {})
        dec_cfg = config.get("decoding_policy", config.get("decoding", {}))
        nbest_cfg = dec_cfg.get("nbest", config.get("nbest", {}))

        self.model_id: str = str(model_cfg.get("id", ""))
        self.model_revision: str = str(model_cfg.get("revision", ""))
        if not self.model_id:
            raise ValueError("config['model']['id'] is required")
        if not self.model_revision:
            raise ValueError("config['model']['revision'] is required")

        if self.model_id not in PINNED_CTC_REVISIONS:
            raise ValueError(
                f"Unsupported B4 CTC model ID: {self.model_id!r}. "
                f"Supported pinned models: {sorted(PINNED_CTC_REVISIONS.keys())}"
            )
        expected_rev = PINNED_CTC_REVISIONS[self.model_id]
        if self.model_revision != expected_rev:
            raise ValueError(
                f"Model revision mismatch for {self.model_id}: expected {expected_rev}, got {self.model_revision}"
            )

        self.processor_id: str = str(proc_cfg.get("id", self.model_id))
        self.processor_revision: str = str(proc_cfg.get("revision", self.model_revision))
        if self.processor_id != self.model_id or self.processor_revision != self.model_revision:
            raise ValueError("Processor id/revision must match model id/revision")

        self.sampling_rate_hz: int = int(proc_cfg.get("sample_rate_hz", proc_cfg.get("sampling_rate_hz", 16000)))
        if self.sampling_rate_hz != 16000:
            raise ValueError(f"Wav2Vec2 CTC requires 16000 Hz audio, got {self.sampling_rate_hz}")

        self.blank_token_id: int = int(proc_cfg.get("blank_token_id", 0))
        self.expected_vocab_size: int = int(proc_cfg.get("vocab_size", 38))

        self.device: str = device or str(model_cfg.get("device", "cuda"))
        self.precision: str = precision or str(model_cfg.get("precision", "bfloat16"))
        if self.precision not in ALLOWED_CTC_PRECISIONS:
            raise ValueError(f"Unsupported precision: {self.precision!r}")

        if dec_cfg.get("lm_rescoring", False):
            raise ValueError("KenLM / LM rescoring is strictly excluded from B4 baseline")
        if dec_cfg.get("numeral_or_diacritic_postprocessing", False):
            raise ValueError("Numeral/diacritic post-processing is forbidden for B4 baseline")

        self.decode_mode: str = str(dec_cfg.get("primary_decode_mode", "ctc_prefix_beam_search"))
        if self.decode_mode not in {"ctc_prefix_beam_search", "float32_greedy_ctc"}:
            raise ValueError(f"Unsupported B4 decode mode: {self.decode_mode!r}")

        cfg_req_n = int(nbest_cfg.get("requested_n", 5))
        req_n = requested_n if requested_n is not None else cfg_req_n
        self.beam_size: int = int(nbest_cfg.get("beam_size", 10))
        if not isinstance(req_n, int) or isinstance(req_n, bool) or not 1 <= req_n <= self.beam_size:
            raise ValueError(f"requested_n must satisfy 1 <= requested_n <= beam_size ({self.beam_size}), got {req_n}")
        self.requested_n: int = req_n

        self.score_convention_id: str | None = str(
            nbest_cfg.get("score_convention_id", "ctc-exact-forward-logprob-v1")
        )

        weight_sha = model_artifact_sha256 or model_cfg.get("weight_sha256")
        if not (isinstance(weight_sha, str) and len(weight_sha) == 64 and all(c in "0123456789abcdef" for c in weight_sha)):
            raise ValueError(f"model_artifact_sha256 must be a 64-char lowercase hex SHA-256, got {weight_sha!r}")
        self.model_artifact_sha256: str = weight_sha

        expected_oid = PINNED_CTC_LFS_OIDS.get(self.model_id)
        if model_loader is None and expected_oid and self.model_artifact_sha256.lower() != expected_oid.lower():
            raise ValueError(
                f"model_artifact_sha256 {self.model_artifact_sha256} does not match pinned LFS OID {expected_oid}"
            )

        self.checkpoint_path: Path | None = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.config: Mapping[str, Any] = {
            "system_id": self.system_id,
            "model": {
                "id": self.model_id,
                "revision": self.model_revision,
                "weight_file": str(model_cfg.get("weight_file", "pytorch_model.bin")),
                "weight_sha256": self.model_artifact_sha256,
                "precision": self.precision,
                "device": self.device,
            },
            "processor": {
                "construction": "explicit_Wav2Vec2CTCTokenizer_and_Wav2Vec2FeatureExtractor_no_lm",
                "sample_rate_hz": self.sampling_rate_hz,
                "vocab_size": self.expected_vocab_size,
                "blank_token_id": self.blank_token_id,
            },
            "decoding": {
                "primary_decode_mode": self.decode_mode,
                "requested_n": self.requested_n,
                "beam_size": self.beam_size,
                "lm_rescoring": False,
                "strip_leading_trailing_whitespace_in_raw": False,
                "numeral_or_diacritic_postprocessing": False,
                "score_convention_id": self.score_convention_id,
            },
        }
        self.decoder_config_sha256: str = hashlib.sha256(
            json.dumps(self.config["decoding"], sort_keys=True).encode("utf-8")
        ).hexdigest()

        self.model: Any = None
        self.tokenizer: Any = None
        self.feature_extractor: Any = None
        self._model_loader = model_loader
        self._tokenizer_loader = tokenizer_loader
        self._feature_extractor_loader = feature_extractor_loader
        self._nbest_fn = nbest_fn or ctc_nbest
        self._shared_path = shared_path

    @classmethod
    def from_config_file(cls, path: str | Path, **kwargs: Any) -> Wav2Vec2CtcBaselineAdapter:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data, **kwargs)

    def load(self) -> None:
        """Load Wav2Vec2ForCTC + explicit tokenizer/feature_extractor (no KenLM)."""
        if (
            self._model_loader is not None
            and self._tokenizer_loader is not None
            and self._feature_extractor_loader is not None
        ):
            self.model = self._model_loader()
            self.tokenizer = self._tokenizer_loader()
            self.feature_extractor = self._feature_extractor_loader()
            return

        import torch
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA is not available on this host; cannot load {self.system_id} on {self.device}")

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map[self.precision]
        weight_filename = self.config["model"]["weight_file"]

        if self.checkpoint_path is not None:
            ckpt_p = Path(self.checkpoint_path).resolve()
            weight_file = ckpt_p if ckpt_p.is_file() else ckpt_p / weight_filename
            snapshot_dir = weight_file.parent
            if not weight_file.exists():
                raise FileNotFoundError(f"Weight file not found: {weight_file}")
            if not (snapshot_dir / "config.json").exists():
                raise ValueError(f"Incomplete snapshot directory {snapshot_dir}: missing config.json")
        else:
            from huggingface_hub import hf_hub_download

            weight_file = Path(
                hf_hub_download(
                    repo_id=self.model_id,
                    filename=weight_filename,
                    revision=self.model_revision,
                    local_files_only=True,
                )
            ).resolve()
            snapshot_dir = weight_file.parent

        actual_sha = hashlib.sha256(weight_file.read_bytes()).hexdigest()
        if actual_sha.lower() != self.model_artifact_sha256.lower():
            raise ValueError(f"Weight SHA-256 mismatch: expected {self.model_artifact_sha256}, got {actual_sha}")

        # Explicit component loading avoids Wav2Vec2ProcessorWithLM / KenLM dependency
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(str(snapshot_dir), local_files_only=True)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(snapshot_dir), local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            str(snapshot_dir),
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        if len(self.tokenizer) != self.expected_vocab_size or self.model.config.vocab_size != self.expected_vocab_size:
            raise ValueError(
                f"Expected vocab_size={self.expected_vocab_size}, got tokenizer={len(self.tokenizer)} "
                f"and model={self.model.config.vocab_size}"
            )
        if self.tokenizer.pad_token_id != self.blank_token_id:
            raise ValueError(
                f"Expected pad_token_id={self.blank_token_id}, got {self.tokenizer.pad_token_id}"
            )
        self.model.to(self.device).eval()

    def nbest(self, samples: np.ndarray, *, deduplicate: bool = False) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Decode one 16 kHz mono utterance into N-best CTC candidates."""
        import torch

        if not isinstance(samples, np.ndarray) or samples.ndim != 1 or len(samples) == 0 or not np.isfinite(samples).all():
            raise ValueError("Expected non-empty finite 1D numpy array")

        with torch.inference_mode():
            extracted = self.feature_extractor(
                samples,
                sampling_rate=self.sampling_rate_hz,
                return_attention_mask=True,
                return_tensors="pt",
            )
            model_dtype = getattr(self.model, "dtype", None)
            model_device = getattr(self.model, "device", self.device)
            inputs: dict[str, Any] = {}
            for key, val in extracted.items():
                if hasattr(val, "to"):
                    if key == "input_values" and model_dtype is not None:
                        inputs[key] = val.to(device=model_device, dtype=model_dtype)
                    else:
                        inputs[key] = val.to(device=model_device)
                else:
                    inputs[key] = val

            logits = self.model(**inputs).logits[0]
            # Cast to float32 before log_softmax so validate_log_probs sum-to-one tolerance (1e-4) holds under bf16
            frames = logits.float().log_softmax(dim=-1).cpu().tolist()

        if self.decode_mode == "float32_greedy_ctc":
            text, tok_ids = greedy_ctc_decode(frames, self.tokenizer, blank_id=self.blank_token_id)
            from turkish_asr_rdl.decoding.ctc import sequence_log_probability

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

        return self._nbest_fn(
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
                        error_message=f"B4 CTC decoding failed: {exc}",
                        selected_candidate_id=None,
                    )
                )

        return results