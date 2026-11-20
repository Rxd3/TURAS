"""Reusable Whisper baseline adapter for B1 Whisper-small and B2 Whisper-large-v3.

Implements the shared AsrAdapter protocol for unmodified pretrained Whisper baselines.
Uses pinned Hugging Face model and processor revisions, reference-free 16 kHz mono audio,
and the shared core Whisper N-best decoding helper from integration revision b6afa13.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np


# Pinned revisions audited for TURAS baselines
PINNED_WHISPER_REVISIONS: dict[str, str] = {
    "openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
    "openai/whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
}

PINNED_WHISPER_LFS_OIDS: dict[str, str] = {
    "openai/whisper-small": "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b",
    "openai/whisper-large-v3": "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
}

ALLOWED_WHISPER_PRECISIONS: frozenset[str] = frozenset(
    {"bfloat16", "bf16", "float16", "fp16", "float32", "fp32"}
)

DEFAULT_PROMPT_TOKENS: list[str] = [
    "<|startoftranscript|>",
    "<|tr|>",
    "<|transcribe|>",
    "<|notimestamps|>",
]


def connect_shared_decoding(shared_path: str | Path | None = None) -> bool:
    """Connect to shared decoding and schema modules from integration revision b6afa13.

    If not already importable in the environment, checks TURAS_SHARED_DECODER_PATH
    or the supplied shared_path, extending turkish_asr_rdl subpackage __path__ entries
    to resolve modules outside the working tree.
    """
    try:
        from turkish_asr_rdl.decoding.interface import AsrAdapter, AudioItem, Candidate, Hypothesis
        from turkish_asr_rdl.decoding.whisper import whisper_nbest
        return True
    except ImportError:
        pass

    target = shared_path or os.environ.get("TURAS_SHARED_DECODER_PATH")
    if not target:
        return False

    target_path = Path(target)
    if not target_path.is_dir():
        return False

    if (target_path / "turkish_asr_rdl").is_dir():
        export_pkg = target_path / "turkish_asr_rdl"
        export_root = target_path
    elif target_path.name == "turkish_asr_rdl":
        export_pkg = target_path
        export_root = target_path.parent
    else:
        export_pkg = target_path
        export_root = target_path

    if str(export_root) not in sys.path:
        sys.path.insert(0, str(export_root))

    try:
        import turkish_asr_rdl
        if str(export_pkg) not in turkish_asr_rdl.__path__:
            turkish_asr_rdl.__path__.append(str(export_pkg))

        for subpkg in ("decoding", "schemas", "utils"):
            sub_dir = export_pkg / subpkg
            if sub_dir.is_dir():
                try:
                    mod = __import__(f"turkish_asr_rdl.{subpkg}", fromlist=[subpkg])
                    if str(sub_dir) not in mod.__path__:
                        mod.__path__.append(str(sub_dir))
                except ImportError:
                    pass

        from turkish_asr_rdl.decoding.interface import AsrAdapter, AudioItem, Candidate, Hypothesis
        from turkish_asr_rdl.decoding.whisper import whisper_nbest
        return True
    except ImportError:
        return False


def _verify_shared_schema(candidate_cls: Any, hypothesis_cls: Any) -> None:
    """Verify shared interface classes have all required fields per integration contract."""
    import dataclasses

    if dataclasses.is_dataclass(candidate_cls):
        cand_fields = {f.name for f in dataclasses.fields(candidate_cls)}
    else:
        cand_fields = set(getattr(candidate_cls, "__annotations__", {}).keys())
    missing_cand = {"lm_log_score", "rdl_log_score", "combined_score"} - cand_fields
    if missing_cand:
        raise ImportError(
            f"Shared Candidate schema missing required fields: {sorted(missing_cand)}. "
            f"Requires full integration interface from revision b6afa13."
        )

    if dataclasses.is_dataclass(hypothesis_cls):
        hyp_fields = {f.name for f in dataclasses.fields(hypothesis_cls)}
    else:
        hyp_fields = set(getattr(hypothesis_cls, "__annotations__", {}).keys())
    missing_hyp = {"selected_candidate_id"} - hyp_fields
    if missing_hyp:
        raise ImportError(
            f"Shared Hypothesis schema missing required fields: {sorted(missing_hyp)}. "
            f"Requires full integration interface from revision b6afa13."
        )


def get_shared_interface(shared_path: str | Path | None = None):
    """Obtain shared interface classes and helper from integration revision b6afa13.

    Uses standard production package imports when available; falls back to opt-in
    test/development path wiring (via shared_path or TURAS_SHARED_DECODER_PATH)
    prior to integration PR merge.
    """
    try:
        from turkish_asr_rdl.decoding.interface import AsrAdapter, AudioItem, Candidate, Hypothesis, check_adapter
        from turkish_asr_rdl.decoding.whisper import whisper_nbest
        _verify_shared_schema(Candidate, Hypothesis)
        return AudioItem, Candidate, Hypothesis, AsrAdapter, check_adapter, whisper_nbest
    except ImportError:
        pass

    if not connect_shared_decoding(shared_path):
        raise ImportError(
            "Shared decoding contract (from integration revision b6afa13063f7fa90bb1bc190c67b4529e40e91a5) "
            "is not available. Set TURAS_SHARED_DECODER_PATH to the isolated export directory or pass shared_path."
        )
    from turkish_asr_rdl.decoding.interface import AsrAdapter, AudioItem, Candidate, Hypothesis, check_adapter
    from turkish_asr_rdl.decoding.whisper import whisper_nbest
    _verify_shared_schema(Candidate, Hypothesis)
    return AudioItem, Candidate, Hypothesis, AsrAdapter, check_adapter, whisper_nbest


class WhisperBaselineAdapter:
    """Reusable AsrAdapter implementation for B1 and B2 Whisper baselines."""

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
        processor_loader: Callable[..., Any] | None = None,
        nbest_fn: Callable[..., Any] | None = None,
        shared_path: str | Path | None = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise ValueError(f"config must be a mapping, got {type(config).__name__}")

        self.system_id: str = str(config.get("system_id", "baseline-whisper"))
        model_cfg = config.get("model", {})
        proc_cfg = config.get("processor", {})
        dec_cfg = config.get("decoding", {})
        nbest_cfg = config.get("nbest", {})

        self.model_id: str = str(model_cfg.get("id", ""))
        self.model_revision: str = str(model_cfg.get("revision", ""))
        if not self.model_id:
            raise ValueError("config['model']['id'] is required")
        if not self.model_revision:
            raise ValueError("config['model']['revision'] is required")

        if self.model_id not in PINNED_WHISPER_REVISIONS:
            raise ValueError(
                f"Unsupported Whisper model ID: {self.model_id!r}. "
                f"Supported pinned models: {sorted(PINNED_WHISPER_REVISIONS.keys())}"
            )

        expected_rev = PINNED_WHISPER_REVISIONS[self.model_id]
        if self.model_revision != expected_rev:
            raise ValueError(
                f"Model revision mismatch for {self.model_id}: "
                f"expected {expected_rev}, got {self.model_revision}"
            )

        self.processor_id: str = str(proc_cfg.get("id", ""))
        self.processor_revision: str = str(proc_cfg.get("revision", ""))
        if self.processor_id != self.model_id:
            raise ValueError(
                f"Processor ID ({self.processor_id}) must match Model ID ({self.model_id})"
            )
        if self.processor_revision != self.model_revision:
            raise ValueError(
                f"Processor revision ({self.processor_revision}) must match Model revision ({self.model_revision})"
            )

        self.sampling_rate_hz: int = int(proc_cfg.get("sampling_rate_hz", 16000))
        if self.sampling_rate_hz != 16000:
            raise ValueError(f"Whisper requires 16000 Hz audio, got {self.sampling_rate_hz}")

        self.device: str = device or str(model_cfg.get("device", "cuda"))
        self.precision: str = precision or str(model_cfg.get("precision", "bfloat16"))
        if self.precision not in ALLOWED_WHISPER_PRECISIONS:
            raise ValueError(
                f"Unsupported precision: {self.precision!r}. "
                f"Allowed precisions: {sorted(ALLOWED_WHISPER_PRECISIONS)}"
            )

        # Checkpoint path
        self.checkpoint_path: Path | None = Path(checkpoint_path).resolve() if checkpoint_path else None

        # Requested N and beam width handling
        cfg_req_n = dec_cfg.get("requested_n", 5)
        cfg_num_beams = dec_cfg.get("num_beams")
        if cfg_num_beams is not None and cfg_req_n is not None and cfg_num_beams != cfg_req_n:
            raise ValueError(
                f"The target whisper_nbest helper at revision b6afa13 ties beam width to requested_n; "
                f"cannot use num_beams={cfg_num_beams} with requested_n={cfg_req_n}"
            )

        req_n = requested_n if requested_n is not None else cfg_req_n
        if not isinstance(req_n, int) or isinstance(req_n, bool) or req_n < 1:
            raise ValueError(f"requested_n must be a positive integer, got {req_n}")
        self.requested_n: int = req_n

        # Decoding options validation: honor documented options, reject unsupported overrides
        algo = dec_cfg.get("algorithm", "beam")
        if algo != "beam":
            raise ValueError(f"WhisperBaselineAdapter only supports algorithm='beam', got {algo!r}")

        self.do_sample: bool = bool(dec_cfg.get("do_sample", False))
        if self.do_sample:
            raise ValueError("Whisper baseline decoding requires do_sample=False (deterministic beam search)")

        temp = dec_cfg.get("temperature")
        if temp is not None and temp != 0.0:
            raise ValueError(f"temperature is not supported when do_sample=False; set temperature to null, got {temp}")
        self.temperature: float | None = None

        self.task: str = str(dec_cfg.get("task", "transcribe"))
        if self.task != "transcribe":
            raise ValueError(f"Whisper baseline decoding requires task='transcribe', got {self.task!r}")

        self.language: str = str(dec_cfg.get("language", "tr"))
        if self.language != "tr":
            raise ValueError(f"Whisper baseline decoding requires language='tr', got {self.language!r}")

        self.max_new_tokens: int = int(dec_cfg.get("max_new_tokens", 440))
        self.forced_prefix_tokens: list[str] = list(dec_cfg.get("forced_prefix_tokens", DEFAULT_PROMPT_TOKENS))
        if self.forced_prefix_tokens != DEFAULT_PROMPT_TOKENS:
            raise ValueError(
                f"forced_prefix_tokens must match helper prompt tokens {DEFAULT_PROMPT_TOKENS}, "
                f"got {self.forced_prefix_tokens}"
            )

        length_penalty = dec_cfg.get("length_penalty", 1.0)
        if length_penalty != 1.0:
            raise ValueError(
                f"The target whisper_nbest helper at revision b6afa13 requires length_penalty=1.0, got {length_penalty}"
            )

        if dec_cfg.get("early_stopping", False):
            raise ValueError("early_stopping=True is not supported in baseline decoding")

        if dec_cfg.get("repetition_penalty", 1.0) != 1.0:
            raise ValueError("repetition_penalty != 1.0 is not supported in baseline decoding")

        if dec_cfg.get("no_repeat_ngram_size", 0) != 0:
            raise ValueError("no_repeat_ngram_size != 0 is not supported in baseline decoding")

        self.score_convention_id: str | None = nbest_cfg.get(
            "score_convention_id", "whisper-original-logits-generated-plus-eos-v1"
        )
        if self.score_convention_id != "whisper-original-logits-generated-plus-eos-v1":
            raise ValueError(
                f"score_convention_id must be 'whisper-original-logits-generated-plus-eos-v1', "
                f"got {self.score_convention_id!r}"
            )

        self.acoustic_score_kind: str = str(
            nbest_cfg.get("acoustic_score_kind", "seq2seq_token_log_prob_sum")
        )
        if self.acoustic_score_kind != "seq2seq_token_log_prob_sum":
            raise ValueError(
                f"acoustic_score_kind must be 'seq2seq_token_log_prob_sum', "
                f"got {self.acoustic_score_kind!r}"
            )

        self.deduplication_policy: str = str(
            nbest_cfg.get(
                "deduplication_policy",
                "exact raw-text match; first (lowest-rank) occurrence kept; removed IDs listed in dedup_map",
            )
        )

        # model_artifact_sha256 must follow artifact contract: 64-char lowercase hex of the model weight file
        weight_sha = model_artifact_sha256 or model_cfg.get("weight_sha256")
        if not (isinstance(weight_sha, str) and len(weight_sha) == 64 and all(c in "0123456789abcdef" for c in weight_sha)):
            raise ValueError(
                f"model_artifact_sha256 for {self.system_id} must be a 64-character lowercase hex SHA-256 string "
                f"(under the artifact contract, recording the weight artifact SHA-256; not a placeholder or model name hash). "
                f"Got: {weight_sha!r}"
            )
        self.model_artifact_sha256: str = weight_sha

        expected_oid = PINNED_WHISPER_LFS_OIDS.get(self.model_id)
        if model_loader is None and expected_oid and self.model_artifact_sha256.lower() != expected_oid.lower():
            raise ValueError(
                f"model_artifact_sha256 {self.model_artifact_sha256} does not match pinned LFS OID "
                f"{expected_oid} for {self.model_id}"
            )

        # Strict JSON-serializable config mapping satisfying check_adapter
        self.config: Mapping[str, Any] = {
            "system_id": self.system_id,
            "model": {
                "id": self.model_id,
                "revision": self.model_revision,
                "weight_file": str(model_cfg.get("weight_file", "model.safetensors")),
                "weight_sha256": self.model_artifact_sha256,
                "weight_sha256_kind": str(model_cfg.get("weight_sha256_kind", "Hub LFS oid; bytes verified at download")),
                "precision": self.precision,
                "device": self.device,
            },
            "processor": {
                "id": self.processor_id,
                "revision": self.processor_revision,
                "sampling_rate_hz": self.sampling_rate_hz,
            },
            "audio": {
                "sampling_rate_hz": self.sampling_rate_hz,
                "clip_duration_constraint": "10-20 s clips from TURAS (M1 source audit: data/metadata/worldspeech-tr_tr-source-audit.md:52); fits within Whisper 30 s window (480,000 samples), no chunking needed",
            },
            "decoding": {
                "algorithm": "beam",
                "requested_n": self.requested_n,
                "num_beams": self.requested_n,
                "num_return_sequences": self.requested_n,
                "max_new_tokens": self.max_new_tokens,
                "language": self.language,
                "task": self.task,
                "do_sample": False,
                "temperature": None,
                "length_penalty": 1.0,
                "early_stopping": False,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "forced_prefix_tokens": self.forced_prefix_tokens,
            },
            "nbest": {
                "score_convention_id": self.score_convention_id,
                "acoustic_score_kind": self.acoustic_score_kind,
                "deduplication_policy": self.deduplication_policy,
            },
        }

        # Include decoding config, precision, score convention, and acoustic score kind in hash
        hash_payload = {
            **self.config["decoding"],
            "precision": self.precision,
            "score_convention_id": self.score_convention_id,
            "acoustic_score_kind": self.acoustic_score_kind,
        }
        self.decoder_config_sha256: str = hashlib.sha256(
            json.dumps(hash_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self.model: Any = None
        self.processor: Any = None
        self._model_loader = model_loader
        self._processor_loader = processor_loader
        self._nbest_fn = nbest_fn
        self._shared_path = shared_path

    @classmethod
    def from_config_file(cls, path: str | Path, **kwargs: Any) -> WhisperBaselineAdapter:
        """Instantiate adapter from a JSON baseline configuration file."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, **kwargs)

    def load(self) -> None:
        """Load weights and processor into memory.

        Uses injected loaders if provided (for offline testing), otherwise resolves
        the verified local snapshot bytes and loads model/processor with local_files_only=True.
        """
        if self._model_loader is not None and self._processor_loader is not None:
            self.model = self._model_loader()
            self.processor = self._processor_loader()
            return

        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

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
        torch_dtype = dtype_map.get(self.precision, torch.bfloat16)

        # Resolve exact local snapshot path and verify weight integrity
        weight_filename = self.config["model"].get("weight_file", "model.safetensors")
        if self.checkpoint_path is not None:
            ckpt_p = Path(self.checkpoint_path).resolve()
            if ckpt_p.is_file():
                weight_file = ckpt_p
                snapshot_dir = ckpt_p.parent
            else:
                snapshot_dir = ckpt_p
                weight_file = snapshot_dir / weight_filename
            if not weight_file.exists():
                raise FileNotFoundError(f"Weight file not found in checkpoint: {weight_file}")
            if not (snapshot_dir / "config.json").exists():
                raise ValueError(f"Incomplete snapshot directory {snapshot_dir}: missing config.json")
        else:
            try:
                from huggingface_hub import hf_hub_download
                resolved = Path(
                    hf_hub_download(
                        repo_id=self.model_id,
                        filename=weight_filename,
                        revision=self.model_revision,
                        local_files_only=True,
                    )
                ).resolve()
            except Exception as exc:
                raise FileNotFoundError(
                    f"Direct loading from Hugging Face cache failed for {self.model_id} (revision {self.model_revision}) "
                    f"with local_files_only=True: {exc}"
                ) from exc
            weight_file = resolved
            snapshot_dir = resolved.parent
            if not (snapshot_dir / "config.json").exists():
                raise ValueError(f"Incomplete snapshot directory {snapshot_dir}: missing config.json")

        actual_sha = hashlib.sha256(weight_file.read_bytes()).hexdigest()
        if actual_sha.lower() != self.model_artifact_sha256.lower():
            raise ValueError(
                f"Weight SHA-256 mismatch for {weight_file.name}: "
                f"expected {self.model_artifact_sha256}, got {actual_sha}"
            )
        pinned_oid = PINNED_WHISPER_LFS_OIDS.get(self.model_id)
        if pinned_oid and actual_sha.lower() != pinned_oid.lower():
            raise ValueError(
                f"Weight LFS OID mismatch for {self.model_id}: "
                f"expected {pinned_oid}, got {actual_sha}"
            )

        self.processor = WhisperProcessor.from_pretrained(
            str(snapshot_dir),
            local_files_only=True,
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(snapshot_dir),
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
        )
        if not isinstance(self.model, WhisperForConditionalGeneration):
            raise TypeError(
                f"Loaded model is of type {type(self.model).__name__}, expected WhisperForConditionalGeneration"
            )
        self.model.to(self.device).eval()

    def transcribe(self, batch: Sequence[Any]) -> list[Any]:
        """Transcribe a batch of reference-free AudioItems.

        Returns exactly one Hypothesis per AudioItem in the same order, preserving
        utterance IDs, handling empty and error results without dropping items, and
        preserving raw decoded text (including leading whitespace).
        """
        AudioItem, Candidate, Hypothesis, AsrAdapter, check_adapter, default_nbest = get_shared_interface(
            self._shared_path
        )
        nbest_fn = self._nbest_fn or default_nbest

        results: list[Any] = []
        for item in batch:
            utt_id = getattr(item, "utterance_id", None)
            if not isinstance(utt_id, str) or not utt_id:
                raise ValueError("Each AudioItem in batch must have a non-empty string utterance_id")

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
                        error_message=(
                            f"Audio duration exceeds 30 s limit (480,000 samples at 16 kHz): "
                            f"got {len(samples)} samples. Expected 10-20 s clips per M1 audit "
                            f"(data/metadata/worldspeech-tr_tr-source-audit.md:52)."
                        ),
                        selected_candidate_id=None,
                    )
                )
                continue

            try:
                features = self.processor.feature_extractor(
                    samples,
                    sampling_rate=self.sampling_rate_hz,
                    return_tensors="pt",
                ).input_features

                if hasattr(features, "to"):
                    if hasattr(self.model, "device"):
                        target_device = self.model.device
                    else:
                        target_device = self.device

                    if hasattr(self.model, "dtype"):
                        target_dtype = self.model.dtype
                    else:
                        target_dtype = None

                    if target_dtype is not None:
                        features = features.to(device=target_device, dtype=target_dtype)
                    else:
                        features = features.to(device=target_device)
            except Exception as exc:
                results.append(
                    Hypothesis(
                        utterance_id=utt_id,
                        status="error",
                        text=None,
                        candidates=(),
                        error_code=type(exc).__name__,
                        error_message=f"Feature extraction failed: {exc}",
                        selected_candidate_id=None,
                    )
                )
                continue

            try:
                raw_cands, _ = nbest_fn(
                    self.model,
                    self.processor,
                    features,
                    requested_n=self.requested_n,
                    max_new_tokens=self.max_new_tokens,
                    deduplicate=False,
                )

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

                candidates_list = []
                for c in raw_cands:
                    cand_obj = Candidate(
                        candidate_id=c["candidate_id"],
                        rank=c["rank"],
                        text=c["text"],  # Verbatim decoded text (whitespace preserved)
                        token_ids=tuple(c["token_ids"]),
                        acoustic_log_score=c["acoustic_log_score"],
                        acoustic_score_kind=c["acoustic_score_kind"],
                        decoder_sequence_score=c.get("decoder_sequence_score"),  # None preserved as None
                        score_details=copy.deepcopy(c.get("score_details", {})),
                        lm_log_score=c.get("lm_log_score"),
                        rdl_log_score=c.get("rdl_log_score"),
                        combined_score=c.get("combined_score"),
                    )
                    candidates_list.append(cand_obj)

                cand_tuple = tuple(candidates_list)
                top_text = cand_tuple[0].text
                status = "ok" if top_text else "empty"
                selected_candidate_id = cand_tuple[0].candidate_id if (cand_tuple and status == "ok") else None

                results.append(
                    Hypothesis(
                        utterance_id=utt_id,
                        status=status,
                        text=top_text,
                        candidates=cand_tuple,
                        selected_candidate_id=selected_candidate_id,
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
                        error_message=f"Decoding failed: {exc}",
                        selected_candidate_id=None,
                    )
                )

        return results
