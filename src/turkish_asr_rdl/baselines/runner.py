"""Baseline batch inference runner and shared exporter integration for Member 2.

Supports B1 (Whisper-small), B2 (Whisper-large-v3), and B3 (Qwen3-ASR 1.7B/0.6B) baselines.
Validates local checkpoint artifacts against pinned inventory before loading,
enforces reference-free input manifests, preserves batch order and utterance IDs across
success, empty, and error outcomes, and writes validated predictions, N-best, and
run manifest artifacts to the specified external directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Pinned checkpoint artifact inventory audited for TURAS baselines
PINNED_CHECKPOINT_INVENTORY: dict[str, dict[str, Any]] = {
    "openai/whisper-small": {
        "weight_file": "model.safetensors",
        "sha256": "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b",
        "size_bytes": 966995080,
        "sharded": False,
    },
    "openai/whisper-large-v3": {
        "weight_file": "model.safetensors",
        "sha256": "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
        "size_bytes": 3087130976,
        "sharded": False,
    },
    "Qwen/Qwen3-ASR-1.7B-hf": {
        "weight_file": "model.safetensors",
        "sha256": "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1",
        "size_bytes": 4076193080,
        "sharded": False,
    },
    "Qwen/Qwen3-ASR-0.6B-hf": {
        "weight_file": "model.safetensors",
        "sha256": "d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35",
        "size_bytes": 1564928088,
        "sharded": False,
    },
}

DEFAULT_LIMITATIONS = [
    "Baseline inference only; no task-specific fine-tuning applied.",
    "Reference-free transcription; references are strictly protected and never accessed during decoding.",
    "Whisper baselines emit leading space under raw-v1 text emission; edge whitespace preserved verbatim.",
    "Qwen3-ASR sequence scores originate from compute_transition_scores over generation output through EOS.",
    "Pretraining contamination risk documented in experiments/m2-baseline-notes.md.",
]


class CheckpointValidationError(ValueError):
    """Raised when local checkpoint files fail integrity, existence, or inventory checks."""


class InputManifestError(ValueError):
    """Raised when an inference input manifest is malformed or violates contract allowlists."""


@dataclass(frozen=True)
class BatchRunResult:
    """Summary of batch inference execution and written output artifacts."""

    run_id: str
    system_id: str
    total_utterances: int
    ok_count: int
    empty_count: int
    error_count: int
    predictions_path: Path
    predictions_sha256: str
    nbest_path: Path | None
    nbest_sha256: str | None
    manifest_path: Path
    manifest_sha256: str
    validation_ok: bool
    validation_errors: tuple[str, ...]


def connect_shared_decoding(shared_path: str | Path | None = None) -> bool:
    """Ensure shared decoding and schemas from revision b6afa13 are importable."""
    try:
        from turkish_asr_rdl.decoding.export import RunContext, nbest_row, prediction_row, run_manifest
        from turkish_asr_rdl.decoding.interface import AudioItem, Candidate, Hypothesis, InferenceInput
        from turkish_asr_rdl.schemas.nbest import validate_nbest
        from turkish_asr_rdl.schemas.predictions import validate_predictions
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

        from turkish_asr_rdl.decoding.export import RunContext, nbest_row, prediction_row, run_manifest
        from turkish_asr_rdl.decoding.interface import AudioItem, Candidate, Hypothesis, InferenceInput
        from turkish_asr_rdl.schemas.nbest import validate_nbest
        from turkish_asr_rdl.schemas.predictions import validate_predictions
        return True
    except ImportError:
        return False


def get_shared_modules(shared_path: str | Path | None = None):
    """Retrieve shared decoding and schema interfaces, connecting fallback paths if needed."""
    if not connect_shared_decoding(shared_path):
        raise ImportError(
            "Shared decoding and schema modules (from integration revision b6afa13063f7fa90bb1bc190c67b4529e40e91a5) "
            "are not available. Set TURAS_SHARED_DECODER_PATH to the isolated export directory or pass shared_path."
        )
    from turkish_asr_rdl.decoding.export import RunContext, nbest_row, prediction_row, run_manifest
    from turkish_asr_rdl.decoding.interface import AudioItem, Candidate, Hypothesis, InferenceInput, check_adapter
    from turkish_asr_rdl.schemas.jsonl import read_jsonl, sha256_file, write_json, write_jsonl
    from turkish_asr_rdl.schemas.nbest import validate_nbest
    from turkish_asr_rdl.schemas.predictions import validate_predictions

    return {
        "RunContext": RunContext,
        "prediction_row": prediction_row,
        "nbest_row": nbest_row,
        "run_manifest": run_manifest,
        "AudioItem": AudioItem,
        "Candidate": Candidate,
        "Hypothesis": Hypothesis,
        "InferenceInput": InferenceInput,
        "check_adapter": check_adapter,
        "read_jsonl": read_jsonl,
        "sha256_file": sha256_file,
        "write_json": write_json,
        "write_jsonl": write_jsonl,
        "validate_predictions": validate_predictions,
        "validate_nbest": validate_nbest,
    }


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a local file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_checkpoint_inventory(
    checkpoint_dir_or_file: str | Path,
    expected_weight_file: str,
    expected_sha256: str,
    model_id: str | None = None,
) -> Path:
    """Validate that local checkpoint files exist and match the expected SHA-256 artifact hash.

    Enforces local-only paths (rejects remote URIs). If sharded weights are present
    without an explicit verified inventory for each shard, fails safely.
    """
    raw_str = str(checkpoint_dir_or_file)
    if raw_str.startswith(("http://", "https://", "ftp://", "s3://", "gs://")):
        raise CheckpointValidationError(f"Checkpoint must be a local path, got remote URL: {raw_str!r}")

    check_path = Path(checkpoint_dir_or_file).resolve()
    if not check_path.exists():
        raise CheckpointValidationError(f"Checkpoint path does not exist: {check_path}")

    # Resolve target weight file and snapshot directory
    if check_path.is_file():
        weight_file = check_path
        snapshot_dir = check_path.parent
    else:
        snapshot_dir = check_path
        # Check if sharded weights exist without single weight file
        index_file = check_path / f"{expected_weight_file}.index.json"
        target_file = check_path / expected_weight_file
        if index_file.exists() and not target_file.exists():
            raise CheckpointValidationError(
                f"Checkpoint at {check_path} contains sharded weights ({index_file.name}), "
                f"but model configuration specifies single weight file {expected_weight_file}. "
                f"Unresolved shard inventory: all shards must be verified against pinned shard inventory."
            )
        if not target_file.exists():
            raise CheckpointValidationError(
                f"Expected weight file {expected_weight_file!r} not found under checkpoint directory {check_path}"
            )
        weight_file = target_file

    if not (snapshot_dir / "config.json").exists():
        raise CheckpointValidationError(
            f"Checkpoint directory {snapshot_dir} is incomplete: missing config.json"
        )

    # Compute and verify SHA-256
    actual_sha = compute_file_sha256(weight_file)
    expected_clean = expected_sha256.strip().lower()
    if actual_sha != expected_clean:
        raise CheckpointValidationError(
            f"Checkpoint SHA-256 mismatch for {weight_file.name}: "
            f"expected {expected_clean}, got {actual_sha}"
        )

    return weight_file


def load_audio_clip(audio_path: Path) -> np.ndarray:
    """Decode local audio file to float32 mono 16 kHz numpy array."""
    raw_str = str(audio_path)
    if raw_str.startswith(("http://", "https://", "ftp://")):
        raise ValueError(f"Audio must be a local file path, got remote URL: {raw_str!r}")

    resolved = audio_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Audio file not found: {resolved}")

    # Attempt to use shared audio module if present
    try:
        from turkish_asr_rdl.decoding.audio import load_audio
        return load_audio(resolved)
    except ImportError:
        pass

    # Standard fallback via soundfile and soxr
    import soundfile as sf
    import soxr

    with open(resolved, "rb") as f:
        data, sample_rate = sf.read(f, dtype="float32", always_2d=True)

    mono = data.mean(axis=1, dtype=np.float32) if data.ndim == 2 else data.astype(np.float32)
    if sample_rate != 16000:
        mono = soxr.resample(mono, sample_rate, 16000, quality="HQ")

    return np.ascontiguousarray(mono, dtype=np.float32)


def read_inference_manifest(
    manifest_path: str | Path,
    shared_path: str | Path | None = None,
) -> list[Any]:
    """Read and validate a reference-free inference input manifest JSONL.

    Enforces allowlisted fields and strictly forbids reference transcripts.
    Verifies that all rows share the same dataset release and split manifest SHA-256.
    """
    shared = get_shared_modules(shared_path)
    InferenceInput = shared["InferenceInput"]

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Inference manifest file not found: {path}")

    inputs: list[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                row_dict = json.loads(line_str)
            except json.JSONDecodeError as exc:
                raise InputManifestError(f"Malformed JSON on line {line_no} of {path}: {exc}") from exc

            if not isinstance(row_dict, dict):
                raise InputManifestError(f"Line {line_no} of {path} must be a JSON object, got {type(row_dict).__name__}")

            try:
                inp_obj = InferenceInput.from_row(row_dict)
            except ValueError as exc:
                raise InputManifestError(f"Line {line_no} ({row_dict.get('utterance_id', 'unknown')}) failed validation: {exc}") from exc

            inputs.append(inp_obj)

    if not inputs:
        raise InputManifestError(f"Inference manifest {path} contains no input rows")

    # Verify constant dataset identity across all rows
    ref_release = inputs[0].dataset_release
    ref_split_sha = inputs[0].split_manifest_sha256
    for inp in inputs:
        if inp.dataset_release != ref_release:
            raise InputManifestError(
                f"Inconsistent dataset_release in manifest: expected {ref_release!r}, got {inp.dataset_release!r} on {inp.utterance_id}"
            )
        if inp.split_manifest_sha256 != ref_split_sha:
            raise InputManifestError(
                f"Inconsistent split_manifest_sha256 in manifest: expected {ref_split_sha!r}, got {inp.split_manifest_sha256!r} on {inp.utterance_id}"
            )

    return inputs


def run_baseline_batch(
    config: Mapping[str, Any] | str | Path,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    audio_root: str | Path | None = None,
    device: str | None = None,
    requested_n: int | None = None,
    shared_path: str | Path | None = None,
    model_loader: Callable[..., Any] | None = None,
    processor_loader: Callable[..., Any] | None = None,
    nbest_fn: Callable[..., Any] | None = None,
    audio_loader: Callable[[Path], np.ndarray] | None = None,
    command_line: str | None = None,
    split_name: str | None = None,
) -> BatchRunResult:
    """Execute end-to-end baseline batch inference and export validated artifacts.

    Args:
        config: Baseline configuration mapping or path to JSON config file.
        checkpoint_path: Local path to model weights file or checkpoint folder.
        manifest_path: Path to reference-free input manifest JSONL.
        output_dir: Destination folder outside repository for exported artifacts.
        audio_root: Root directory for relative audio_uri resolution (defaults to manifest dir).
        device: Device override (e.g. 'cpu', 'cuda').
        requested_n: N-best candidate count override.
        shared_path: Path to shared decoding package export.
        model_loader: Optional injected model factory for mock testing.
        processor_loader: Optional injected processor factory for mock testing.
        audio_loader: Optional injected audio loader for mock testing.
        command_line: Command line invocation string for run manifest.
        split_name: Split name override (e.g. 'test', 'validation').

    Returns:
        BatchRunResult containing output paths, hashes, and validation status.
    """
    shared = get_shared_modules(shared_path)
    AudioItem = shared["AudioItem"]
    Candidate = shared["Candidate"]
    Hypothesis = shared["Hypothesis"]
    RunContext = shared["RunContext"]
    prediction_row = shared["prediction_row"]
    nbest_row = shared["nbest_row"]
    run_manifest = shared["run_manifest"]
    check_adapter = shared["check_adapter"]
    write_json = shared["write_json"]
    write_jsonl = shared["write_jsonl"]
    sha256_file = shared["sha256_file"]
    validate_predictions = shared["validate_predictions"]
    validate_nbest = shared["validate_nbest"]

    # 1. Parse and validate configuration
    if isinstance(config, (str, Path)):
        cfg_path = Path(config)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg_data = json.load(f)
    elif isinstance(config, Mapping):
        cfg_data = dict(config)
    else:
        raise ValueError(f"config must be a path or mapping, got {type(config).__name__}")

    model_cfg = cfg_data.get("model", {})
    model_id = str(model_cfg.get("id", ""))
    weight_file = str(model_cfg.get("weight_file", "model.safetensors"))
    weight_sha = str(model_cfg.get("weight_sha256", ""))

    # 2. Checkpoint artifact integrity validation (performed before any model loading)
    verify_checkpoint_inventory(checkpoint_path, weight_file, weight_sha, model_id=model_id)

    # 3. Read and validate input manifest
    manifest_p = Path(manifest_path).resolve()
    inputs = read_inference_manifest(manifest_p, shared_path=shared_path)
    audio_base = Path(audio_root).resolve() if audio_root else manifest_p.parent

    # 4. Initialize adapter
    if model_id.startswith("openai/whisper"):
        from turkish_asr_rdl.baselines.whisper import WhisperBaselineAdapter
        adapter = WhisperBaselineAdapter(
            cfg_data,
            device=device,
            requested_n=requested_n,
            checkpoint_path=checkpoint_path,
            model_artifact_sha256=weight_sha,
            model_loader=model_loader,
            processor_loader=processor_loader,
            nbest_fn=nbest_fn,
            shared_path=shared_path,
        )
    elif model_id.startswith("Qwen/Qwen3-ASR"):
        from turkish_asr_rdl.baselines.qwen import QwenBaselineAdapter
        adapter = QwenBaselineAdapter(
            cfg_data,
            device=device,
            requested_n=requested_n,
            checkpoint_path=checkpoint_path,
            model_artifact_sha256=weight_sha,
            model_loader=model_loader,
            processor_loader=processor_loader,
            nbest_fn=nbest_fn,
            shared_path=shared_path,
        )
    else:
        raise ValueError(f"Unsupported model family for baseline runner: {model_id!r}")

    check_adapter(adapter)
    adapter.load()

    # 5. Load audio and prepare batches preserving order and IDs
    audio_items: list[Any] = []
    valid_indices: list[int] = []
    hypotheses: list[Any | None] = [None] * len(inputs)

    start_time = time.perf_counter()
    for idx, inp in enumerate(inputs):
        audio_file = (audio_base / inp.audio_uri).resolve()
        try:
            if audio_loader is not None:
                samples = audio_loader(audio_file)
            else:
                samples = load_audio_clip(audio_file)
            audio_items.append(AudioItem(inp.utterance_id, samples))
            valid_indices.append(idx)
        except Exception as exc:
            # Per-clip error preserves utterance ID in exact position without dropping
            hypotheses[idx] = Hypothesis(
                utterance_id=inp.utterance_id,
                status="error",
                text=None,
                candidates=(),
                error_code=type(exc).__name__,
                error_message=f"Audio loading failed: {exc}",
                selected_candidate_id=None,
            )

    # 6. Transcribe valid items
    if audio_items:
        transcribed_hyps = adapter.transcribe(audio_items)
        if len(transcribed_hyps) != len(audio_items):
            raise RuntimeError(
                f"Adapter returned {len(transcribed_hyps)} hypotheses for {len(audio_items)} audio items"
            )
        for original_idx, hyp in zip(valid_indices, transcribed_hyps):
            hypotheses[original_idx] = hyp

    elapsed_s = time.perf_counter() - start_time

    # Verify every input has a hypothesis
    final_hypotheses: list[Any] = []
    for idx, h in enumerate(hypotheses):
        if h is None:
            raise RuntimeError(f"Missing hypothesis for input index {idx} ({inputs[idx].utterance_id})")
        final_hypotheses.append(h)

    # 7. Setup RunContext and output directories
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"run-{adapter.system_id}-{timestamp_str}"
    split = split_name or manifest_p.stem

    ctx = RunContext(
        run_id=run_id,
        system_id=adapter.system_id,
        dataset_release=inputs[0].dataset_release,
        split=split,
        split_manifest_sha256=inputs[0].split_manifest_sha256,
        model_artifact_sha256=adapter.model_artifact_sha256,
        decoder_config_sha256=adapter.decoder_config_sha256,
        normalization_version="tr-normalized-v1",
    )

    pairs = list(zip(final_hypotheses, inputs))
    hashes: dict[str, str] = {}
    nbest_path: Path | None = None
    nbest_sha: str | None = None
    validation_errors: list[str] = []

    # 8. Export N-best if requested_n > 0
    if adapter.requested_n > 0:
        nbest_rows = [
            nbest_row(h, i, ctx, adapter.requested_n, adapter.score_convention_id)
            for h, i in pairs
        ]
        nbest_report = validate_nbest(nbest_rows)
        if not nbest_report.ok:
            validation_errors.extend(f"N-best: {err}" for err in nbest_report.errors)

        nbest_path = out_dir / "nbest.jsonl"
        nbest_sha = hashes["nbest.jsonl"] = write_jsonl(nbest_path, nbest_rows)

    # 9. Export Predictions
    record_ids = {i.utterance_id: f"{run_id}-rec-{idx:05d}" for idx, (_, i) in enumerate(pairs, start=1)}
    prediction_rows = [
        prediction_row(h, i, ctx, record_ids[i.utterance_id], nbest_sha)
        for h, i in pairs
    ]
    pred_report = validate_predictions(prediction_rows)
    if not pred_report.ok:
        validation_errors.extend(f"Predictions: {err}" for err in pred_report.errors)

    pred_path = out_dir / "predictions.jsonl"
    pred_sha = hashes["predictions.jsonl"] = write_jsonl(pred_path, prediction_rows)

    # 10. Export Run Manifest
    resolved_cmd = command_line or f"{Path(sys.executable).name} scripts/run_baseline_batch.py"
    env_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "device": adapter.device,
        "precision": adapter.precision,
    }
    timing_info = {
        "elapsed_seconds": round(elapsed_s, 4),
        "total_utterances": len(inputs),
        "note": "batch wall time, not RTF (official RTF measured by M6 on 5070 Ti)",
    }

    manifest_obj = run_manifest(
        ctx,
        command=resolved_cmd,
        environment=env_info,
        configuration=dict(adapter.config),
        outputs=dict(hashes),
        timing=timing_info,
        limitations=list(DEFAULT_LIMITATIONS),
    )
    manifest_path_out = out_dir / "manifest.json"
    manifest_sha = write_json(manifest_path_out, manifest_obj)

    # Counts
    ok_count = sum(1 for h in final_hypotheses if h.status == "ok")
    empty_count = sum(1 for h in final_hypotheses if h.status == "empty")
    error_count = sum(1 for h in final_hypotheses if h.status == "error")

    return BatchRunResult(
        run_id=run_id,
        system_id=adapter.system_id,
        total_utterances=len(inputs),
        ok_count=ok_count,
        empty_count=empty_count,
        error_count=error_count,
        predictions_path=pred_path,
        predictions_sha256=pred_sha,
        nbest_path=nbest_path,
        nbest_sha256=nbest_sha,
        manifest_path=manifest_path_out,
        manifest_sha256=manifest_sha,
        validation_ok=len(validation_errors) == 0,
        validation_errors=tuple(validation_errors),
    )
