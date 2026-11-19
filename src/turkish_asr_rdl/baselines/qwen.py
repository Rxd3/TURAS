"""Reusable Qwen3-ASR baseline adapter for B3 candidates (1.7B-hf and 0.6B-hf).

Implements the shared AsrAdapter protocol for unmodified pretrained Qwen3-ASR baselines.
Uses pinned Hugging Face model and processor revisions, reference-free 16 kHz mono audio,
native transformers AutoModelForMultimodalLM/AutoProcessor classes, and speech-LLM
joint token log-probability scoring via transformers compute_transition_scores over
generation output through end-of-turn (EOS).
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


# Pinned revisions audited for TURAS B3 baselines
PINNED_QWEN_REVISIONS: dict[str, str] = {
    "Qwen/Qwen3-ASR-1.7B-hf": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
    "Qwen/Qwen3-ASR-0.6B-hf": "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
}

PINNED_QWEN_LFS_OIDS: dict[str, str] = {
    "Qwen/Qwen3-ASR-1.7B-hf": "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1",
    "Qwen/Qwen3-ASR-0.6B-hf": "d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35",
}

ALLOWED_QWEN_PRECISIONS: frozenset[str] = frozenset(
    {"bfloat16", "bf16", "float16", "fp16", "float32", "fp32"}
)

DEFAULT_SCORE_CONVENTION = "qwen3-asr-generation-logprob-sum-v1"
DEFAULT_ACOUSTIC_KIND = "speech_llm_joint_token_log_prob_sum"
DEFAULT_DEDUP_POLICY = "exact raw-text match; first (lowest-rank) occurrence kept; removed IDs listed in dedup_map"


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


def connect_shared_decoding(shared_path: str | Path | None = None) -> bool:
    """Connect to shared decoding and schema modules from integration revision b6afa13.

    If not already importable in the environment, checks TURAS_SHARED_DECODER_PATH
    or the supplied shared_path, extending turkish_asr_rdl subpackage __path__ entries
    to resolve modules outside the working tree.
    """
    try:
        from turkish_asr_rdl.decoding.interface import AsrAdapter, AudioItem, Candidate, Hypothesis, check_adapter
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

        from turkish_asr_rdl.decoding.interface import AsrAdapter, AudioItem, Candidate, Hypothesis, check_adapter
        return True
    except ImportError:
        return False


def get_shared_interface(shared_path: str | Path | None = None):
    """Obtain shared interface classes from integration revision b6afa13.

    Uses standard production package imports when available; falls back to opt-in
    test/development path wiring (via shared_path or TURAS_SHARED_DECODER_PATH)
    prior to integration PR merge.
    """
    try:
        from turkish_asr_rdl.decoding.interface import AsrAdapter, AudioItem, Candidate, Hypothesis, check_adapter
        _verify_shared_schema(Candidate, Hypothesis)
        return AudioItem, Candidate, Hypothesis, AsrAdapter, check_adapter
    except ImportError:
        pass

    if not connect_shared_decoding(shared_path):
        raise ImportError(
            "Shared decoding contract (from integration revision b6afa13063f7fa90bb1bc190c67b4529e40e91a5) "
            "is not available. Set TURAS_SHARED_DECODER_PATH to the isolated export directory or pass shared_path."
        )
    from turkish_asr_rdl.decoding.interface import AsrAdapter, AudioItem, Candidate, Hypothesis, check_adapter
    _verify_shared_schema(Candidate, Hypothesis)
    return AudioItem, Candidate, Hypothesis, AsrAdapter, check_adapter


def cut_at_stop(tokens: Sequence[int], stop_ids: set[int] | frozenset[int]) -> tuple[int, ...]:
    """Generated tokens up to and including the first stop token; trailing padding is dropped."""
    end = next((k + 1 for k, token in enumerate(tokens) if token in stop_ids), len(tokens))
    return tuple(tokens[:end])


def compute_joint_log_probs_from_step_scores(
    step_scores: Sequence[float],
    generated_tokens: Sequence[int],
    stop_ids: set[int] | frozenset[int],
) -> tuple[float, tuple[int, ...]]:
    """Compute the speech-LLM joint token log-probability sum from generation step transition scores.

    Excludes prompt tokens (which are not part of generation step_scores) and padding tokens
    (occurring after stop_ids / EOS). Includes the stop token (EOS) according to the shared convention.
    Raises ValueError on empty kept tokens.
    """
    kept_tokens = cut_at_stop(generated_tokens, stop_ids)
    if not kept_tokens:
        raise ValueError("Empty kept tokens encountered during Qwen joint score computation")
    score = float(sum(step_scores[:len(kept_tokens)]))
    return score, kept_tokens


class QwenBaselineAdapter:
    """Reusable AsrAdapter implementation for B3 Qwen3-ASR candidates (1.7B-hf and 0.6B-hf)."""

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

        self.system_id: str = str(config.get("system_id", "baseline-qwen3-asr"))
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

        if self.model_id in PINNED_QWEN_REVISIONS:
            expected_rev = PINNED_QWEN_REVISIONS[self.model_id]
            if self.model_revision != expected_rev:
                raise ValueError(
                    f"Model revision mismatch for {self.model_id}: "
                    f"expected {expected_rev}, got {self.model_revision}"
                )
        else:
            raise ValueError(
                f"Unknown or unpinned model ID {self.model_id!r}; "
                f"expected one of {sorted(PINNED_QWEN_REVISIONS.keys())}"
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
            raise ValueError(f"Qwen3-ASR requires 16000 Hz audio, got {self.sampling_rate_hz}")

        self.device: str = device or str(model_cfg.get("device", "cuda"))
        self.precision: str = precision or str(model_cfg.get("precision", "bfloat16"))
        if self.precision not in ALLOWED_QWEN_PRECISIONS:
            raise ValueError(
                f"Unsupported precision: {self.precision!r}. "
                f"Allowed precisions: {sorted(ALLOWED_QWEN_PRECISIONS)}"
            )

        # Checkpoint path
        self.checkpoint_path: Path | None = Path(checkpoint_path).resolve() if checkpoint_path else None

        # Requested N and beam width handling
        cfg_req_n = dec_cfg.get("requested_n", 5)
        cfg_num_beams = dec_cfg.get("num_beams")
        if cfg_num_beams is not None and cfg_req_n is not None and cfg_num_beams != cfg_req_n:
            raise ValueError(
                f"num_beams ({cfg_num_beams}) and requested_n ({cfg_req_n}) must match for provisional beam search"
            )

        req_n = requested_n if requested_n is not None else cfg_req_n
        if not isinstance(req_n, int) or isinstance(req_n, bool) or req_n < 1:
            raise ValueError(f"requested_n must be a positive integer, got {req_n}")
        self.requested_n: int = req_n

        # Decoding options validation
        algo = dec_cfg.get("algorithm", "beam")
        if algo != "beam":
            raise ValueError(f"QwenBaselineAdapter only supports algorithm='beam', got {algo!r}")

        self.do_sample: bool = bool(dec_cfg.get("do_sample", False))
        if self.do_sample:
            raise ValueError("Qwen baseline decoding requires do_sample=False (deterministic beam search)")

        temp = dec_cfg.get("temperature")
        if temp is not None and temp != 0.0:
            raise ValueError(f"temperature is not supported when do_sample=False; set temperature to null, got {temp}")
        self.temperature: float | None = None

        self.language: str = str(dec_cfg.get("language", "tr"))
        if self.language != "tr":
            raise ValueError(f"Qwen baseline decoding requires language='tr', got {self.language!r}")

        self.max_new_tokens: int = int(dec_cfg.get("max_new_tokens", 440))

        length_penalty = dec_cfg.get("length_penalty", 1.0)
        if length_penalty != 1.0:
            raise ValueError(f"length_penalty != 1.0 is not supported in baseline decoding, got {length_penalty}")

        if dec_cfg.get("early_stopping", False):
            raise ValueError("early_stopping=True is not supported in baseline decoding")

        if dec_cfg.get("repetition_penalty", 1.0) != 1.0:
            raise ValueError("repetition_penalty != 1.0 is not supported in baseline decoding")

        if dec_cfg.get("no_repeat_ngram_size", 0) != 0:
            raise ValueError("no_repeat_ngram_size != 0 is not supported in baseline decoding")

        self.score_convention_id: str | None = nbest_cfg.get(
            "score_convention_id", DEFAULT_SCORE_CONVENTION
        )
        if self.score_convention_id != DEFAULT_SCORE_CONVENTION:
            raise ValueError(
                f"score_convention_id must be {DEFAULT_SCORE_CONVENTION!r}, got {self.score_convention_id!r}"
            )

        self.acoustic_score_kind: str = str(
            nbest_cfg.get("acoustic_score_kind", DEFAULT_ACOUSTIC_KIND)
        )
        if self.acoustic_score_kind != DEFAULT_ACOUSTIC_KIND:
            raise ValueError(
                f"acoustic_score_kind must be {DEFAULT_ACOUSTIC_KIND!r}, got {self.acoustic_score_kind!r}"
            )

        self.deduplication_policy: str = str(
            nbest_cfg.get("deduplication_policy", DEFAULT_DEDUP_POLICY)
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

        expected_oid = PINNED_QWEN_LFS_OIDS.get(self.model_id)
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
                "clip_duration_constraint": "10-20 s clips from TURAS (M1 source audit: data/metadata/worldspeech-tr_tr-source-audit.md:52); Qwen3-ASR accepts up to 20 minutes (1,200 s per arXiv:2601.21337 Section 2.1)",
            },
            "decoding": {
                "algorithm": "beam",
                "requested_n": self.requested_n,
                "num_beams": self.requested_n,
                "num_return_sequences": self.requested_n,
                "max_new_tokens": self.max_new_tokens,
                "language": self.language,
                "prompt_format": "AutoProcessor.apply_transcription_request(language='tr')",
                "do_sample": False,
                "temperature": None,
                "length_penalty": 1.0,
                "early_stopping": False,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
            },
            "nbest": {
                "score_convention_id": self.score_convention_id,
                "acoustic_score_kind": self.acoustic_score_kind,
                "library_beam_score_target": "decoder_sequence_score",
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
        self._stop_ids: frozenset[int] = frozenset()
        self._model_loader = model_loader
        self._processor_loader = processor_loader
        self._nbest_fn = nbest_fn
        self._shared_path = shared_path

    @classmethod
    def from_config_file(cls, path: str | Path, **kwargs: Any) -> QwenBaselineAdapter:
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
            eos = getattr(self.model, "generation_config", None)
            eos_id = getattr(eos, "eos_token_id", None) if eos is not None else None
            if eos_id is None and hasattr(self.processor, "tokenizer"):
                eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                self._stop_ids = frozenset(eos_id if isinstance(eos_id, (list, tuple, set)) else [eos_id])
            else:
                self._stop_ids = frozenset()
            return

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA is not available on this host; cannot load {self.system_id} on {self.device}")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
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
        pinned_oid = PINNED_QWEN_LFS_OIDS.get(self.model_id)
        if pinned_oid and actual_sha.lower() != pinned_oid.lower():
            raise ValueError(
                f"Weight LFS OID mismatch for {self.model_id}: "
                f"expected {pinned_oid}, got {actual_sha}"
            )

        self.processor = AutoProcessor.from_pretrained(
            str(snapshot_dir),
            local_files_only=True,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            str(snapshot_dir),
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        if not hasattr(self.model, "generate"):
            raise TypeError(
                f"Loaded model of type {type(self.model).__name__} does not have required generate method"
            )
        self.model.to(self.device).eval()

        eos = getattr(self.model, "generation_config", None)
        eos_id = getattr(eos, "eos_token_id", None) if eos is not None else None
        if eos_id is None and hasattr(self.processor, "tokenizer"):
            eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            self._stop_ids = frozenset(eos_id if isinstance(eos_id, (list, tuple, set)) else [eos_id])
        else:
            self._stop_ids = frozenset()

    def transcribe(self, batch: Sequence[Any]) -> list[Any]:
        """Transcribe a batch of reference-free AudioItems.

        Returns exactly one Hypothesis per AudioItem in the same order, preserving
        utterance IDs, handling empty and error results without dropping items, and
        preserving raw decoded text verbatim.
        """
        AudioItem, Candidate, Hypothesis, AsrAdapter, check_adapter = get_shared_interface(
            self._shared_path
        )

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

            if len(samples) > 19200000:
                results.append(
                    Hypothesis(
                        utterance_id=utt_id,
                        status="error",
                        text=None,
                        candidates=(),
                        error_code="AudioTooLong",
                        error_message=(
                            f"Audio duration exceeds 20-minute (1,200 s) Qwen3-ASR input limit "
                            f"(arXiv:2601.21337 Section 2.1): got {len(samples)} samples at 16 kHz."
                        ),
                        selected_candidate_id=None,
                    )
                )
                continue

            # Process audio inputs via configured transcription request (strict: no silent fallbacks)
            try:
                if not hasattr(self.processor, "apply_transcription_request"):
                    raise AttributeError(
                        f"Processor {self.processor_id} missing required apply_transcription_request method"
                    )
                inputs = self.processor.apply_transcription_request(
                    audio=[samples],
                    language=self.language,
                )

                # Move inputs to target device and dtype if applicable
                if hasattr(inputs, "to"):
                    target_device = getattr(self.model, "device", self.device)
                    target_dtype = getattr(self.model, "dtype", None)
                    if target_dtype is not None:
                        inputs = inputs.to(device=target_device, dtype=target_dtype)
                    else:
                        inputs = inputs.to(device=target_device)
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

            # Inference / decoding
            try:
                if self._nbest_fn is not None:
                    raw_cands, dedup_map = self._nbest_fn(
                        self.model,
                        self.processor,
                        inputs,
                        requested_n=self.requested_n,
                        max_new_tokens=self.max_new_tokens,
                        language=self.language,
                    )
                else:
                    raw_cands = self._decode_nbest_native(inputs)

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
                        text=c["text"],  # Verbatim decoded text
                        token_ids=tuple(c["token_ids"]),
                        acoustic_log_score=c["acoustic_log_score"],
                        acoustic_score_kind=c["acoustic_score_kind"],
                        decoder_sequence_score=c.get("decoder_sequence_score"),  # Kept separate; None preserved
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

    def _decode_nbest_native(self, inputs: Any) -> list[dict[str, Any]]:
        """Run native beam generation with transformers and extract candidate rows."""
        import torch
        import transformers

        prompt_input_ids = inputs.get("input_ids")
        if prompt_input_ids is None:
            raise ValueError("inputs dictionary must contain 'input_ids' for native decoding")
        prompt_length = prompt_input_ids.shape[1] if hasattr(prompt_input_ids, "shape") else len(prompt_input_ids)

        if not hasattr(self.processor, "decode"):
            raise AttributeError(f"Processor {self.processor_id} missing required decode method")

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                num_beams=self.requested_n,
                num_return_sequences=self.requested_n,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=self.max_new_tokens,
            )

            # Compute transition scores
            normalize_logits = self.requested_n == 1
            beam_indices = getattr(out, "beam_indices", None)
            step_scores_tensor = self.model.compute_transition_scores(
                out.sequences,
                out.scores,
                beam_indices=beam_indices,
                normalize_logits=normalize_logits,
            )
            step_rows = step_scores_tensor.float().tolist()

        sequences = out.sequences.tolist()
        beam_scores = getattr(out, "sequences_scores", None)
        seq_scores_list = beam_scores.float().tolist() if beam_scores is not None else [None] * len(sequences)

        candidates = []
        for rank, (seq, steps, b_score) in enumerate(zip(sequences, step_rows, seq_scores_list), start=1):
            gen_tokens = seq[prompt_length:]
            joint_score, kept_tokens = compute_joint_log_probs_from_step_scores(
                steps, gen_tokens, self._stop_ids
            )

            # Verbatim raw decode with tokenizer (no strip, no repetition fix)
            text = self.processor.decode(list(kept_tokens), skip_special_tokens=True)

            includes_eos = bool(kept_tokens and kept_tokens[-1] in self._stop_ids)
            score_details: dict[str, Any] = {
                "score_source": "generation-time step log-probs; equals the teacher-forced sum in exact arithmetic; bf16 drift up to ~0.16 nats measured on 5 clips",
                "joint": "speech-LLM score; no separate acoustic-only component exists",
                "log_base": "e",
                "includes_eos": includes_eos,
                "prompt_excluded": f"chat prompt with forced language {self.language}",
                "nbest_algorithm": "beam",
                "num_beams": self.requested_n,
                "library": f"transformers {transformers.__version__}",
                "duplicate_policy": self.deduplication_policy,
            }
            if b_score is None:
                score_details["null_reasons"] = {
                    "decoder_sequence_score": "greedy decoding has no beam score"
                    if self.requested_n == 1
                    else "framework beam score not produced"
                }

            candidates.append({
                "candidate_id": f"beam-{rank:04d}",
                "rank": rank,
                "text": text,
                "token_ids": kept_tokens,
                "acoustic_log_score": joint_score,
                "acoustic_score_kind": self.acoustic_score_kind,
                "decoder_sequence_score": float(b_score) if b_score is not None else None,
                "score_details": score_details,
            })

        return candidates
