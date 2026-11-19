"""Thin CLI runner for Whisper rank-1 vs single-result smoke comparison.

Compares rank 1 from the shared N-best decoder with standard single-result
generation (num_return_sequences=1) at the same beam width.
Supports synthetic fixtures as well as real-audio verification with local checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from turkish_asr_rdl.baselines.runner import (
    CheckpointValidationError,
    load_audio_clip,
    verify_checkpoint_inventory,
)
from turkish_asr_rdl.baselines.whisper import (
    connect_shared_decoding,
)
from turkish_asr_rdl.baselines.whisper_rank1 import (
    DEFAULT_WHISPER_PROMPT_TOKENS,
    Rank1ComparisonResult,
    compare_rank1_outputs,
    run_whisper_rank1_comparison,
)


@dataclass(frozen=True)
class ClipComparisonOutcome:
    """Outcome of rank-1 parity comparison on a single clip."""

    clip_id: str
    result: Rank1ComparisonResult


def resolve_shared_whisper_nbest(shared_decoder_path: str | Path | None = None) -> Callable[..., Any]:
    """Resolve the shared whisper_nbest helper from the export directory."""
    if not shared_decoder_path:
        shared_decoder_path = os.environ.get("TURAS_SHARED_DECODER_PATH")

    if not shared_decoder_path:
        raise ValueError(
            "Shared decoder path not supplied. Provide --shared-decoder-path or set TURAS_SHARED_DECODER_PATH."
        )

    export_path = Path(shared_decoder_path).resolve()
    if not export_path.is_dir():
        raise FileNotFoundError(f"Shared decoder directory does not exist: {export_path}")

    whisper_candidates = [
        export_path / "turkish_asr_rdl" / "decoding" / "whisper.py",
        export_path / "decoding" / "whisper.py",
        export_path / "whisper.py",
    ]
    whisper_path = next((p for p in whisper_candidates if p.is_file()), None)
    if not whisper_path:
        raise FileNotFoundError(f"whisper.py not found under {export_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("shared_whisper_decoding", whisper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create importlib spec for {whisper_path}")
    whisper_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(whisper_mod)
    return getattr(whisper_mod, "whisper_nbest")


def run_synthetic_smoke_check(
    shared_decoder_path: str | Path | None = None,
    beam_width: int = 2,
    max_new_tokens: int = 10,
) -> Rank1ComparisonResult:
    """Run comparison on a synthetic Whisper model using genuine torch/transformers."""
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    try:
        whisper_nbest = resolve_shared_whisper_nbest(shared_decoder_path)
    except Exception as err:
        return Rank1ComparisonResult(
            matched=False,
            status="error",
            details=f"Shared whisper_nbest helper could not be loaded: {err}",
            error=str(err),
        )

    # Local fixture tokenizer and processor
    class LocalTokenizer:
        def __init__(self):
            self.eos_token_id = 50257
            self.vocab = {
                "<|startoftranscript|>": 50258,
                "<|tr|>": 50359,
                "<|transcribe|>": 50363,
                "<|notimestamps|>": 50365,
                "<|endoftranscript|>": 50257,
            }
            self.id_to_word = {v: k for k, v in self.vocab.items()}

        def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
            return [self.vocab.get(t, 50257) for t in tokens]

        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            words = []
            for tid in token_ids:
                if skip_special_tokens and tid in (50258, 50359, 50363, 50365, 50257):
                    continue
                words.append(self.id_to_word.get(tid, f"[{tid}]"))
            return " ".join(words) if words else ""

    class LocalProcessor:
        def __init__(self):
            self.tokenizer = LocalTokenizer()

    torch.manual_seed(42)
    config = WhisperConfig(
        vocab_size=51865,
        d_model=64,
        encoder_layers=2,
        encoder_attention_heads=2,
        decoder_layers=2,
        decoder_attention_heads=2,
        decoder_ffn_dim=128,
        encoder_ffn_dim=128,
        forced_decoder_ids=None,
        max_source_positions=1500,
        max_target_positions=448,
    )
    model = WhisperForConditionalGeneration(config)
    model.eval()
    processor = LocalProcessor()
    input_features = torch.randn(1, 80, 3000)

    return run_whisper_rank1_comparison(
        model,
        processor,
        input_features,
        nbest_fn=whisper_nbest,
        beam_width=beam_width,
        max_new_tokens=max_new_tokens,
    )


def run_real_audio_smoke_check(
    config_path: str | Path,
    checkpoint_path: str | Path,
    audio_paths_or_manifest: Sequence[Path] | Path,
    *,
    audio_root: str | Path | None = None,
    shared_decoder_path: str | Path | None = None,
    beam_width: int | None = None,
    max_new_tokens: int | None = None,
    device: str | None = None,
    model_loader: Callable[..., Any] | None = None,
    processor_loader: Callable[..., Any] | None = None,
    audio_loader: Callable[[Path], Any] | None = None,
    nbest_fn: Callable[..., Any] | None = None,
) -> list[ClipComparisonOutcome]:
    """Execute Whisper rank-1 smoke comparison on real local audio clips."""
    # 1. Enforce local paths only
    for path_str in [str(config_path), str(checkpoint_path)]:
        if path_str.startswith(("http://", "https://", "ftp://")):
            raise ValueError(f"Only local paths are accepted, got remote URL: {path_str!r}")

    cfg_p = Path(config_path).resolve()
    with cfg_p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = cfg.get("model", {})
    weight_file = str(model_cfg.get("weight_file", "model.safetensors"))
    weight_sha = str(model_cfg.get("weight_sha256", ""))
    model_id = str(model_cfg.get("id", ""))
    dec_cfg = cfg.get("decoding", {})

    resolved_beam = beam_width or dec_cfg.get("num_beams", 10)
    resolved_max_tokens = max_new_tokens or dec_cfg.get("max_new_tokens", 440)
    target_device = device or model_cfg.get("device", "cpu")

    # 2. Checkpoint integrity verification before loading
    if model_loader is None:
        verify_checkpoint_inventory(checkpoint_path, weight_file, weight_sha, model_id=model_id)

    # 3. Resolve shared helper
    if nbest_fn is not None:
        shared_nbest = nbest_fn
    else:
        shared_nbest = resolve_shared_whisper_nbest(shared_decoder_path)

    # 4. Load model and processor
    if model_loader is not None and processor_loader is not None:
        model = model_loader()
        processor = processor_loader()
    else:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        ckpt_dir = Path(checkpoint_path).resolve()
        if ckpt_dir.is_file():
            ckpt_dir = ckpt_dir.parent
        if not (ckpt_dir / "config.json").exists():
            raise CheckpointValidationError(f"Incomplete snapshot directory {ckpt_dir}: missing config.json")

        processor = WhisperProcessor.from_pretrained(str(ckpt_dir), local_files_only=True)
        torch_dtype = torch.bfloat16 if str(cfg.get("model", {}).get("precision")) in ("bfloat16", "bf16") else (
            torch.float16 if target_device.startswith("cuda") else torch.float32
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            str(ckpt_dir), torch_dtype=torch_dtype, use_safetensors=True, local_files_only=True
        )
        model = model.to(target_device).eval()

    # 5. Collect audio clips to compare
    clips: list[tuple[str, Path]] = []
    if isinstance(audio_paths_or_manifest, Path) and audio_paths_or_manifest.is_file():
        if audio_paths_or_manifest.suffix == ".jsonl":
            # Manifest JSONL: read audio_uri and utterance_id
            base_dir = Path(audio_root).resolve() if audio_root else audio_paths_or_manifest.parent
            with audio_paths_or_manifest.open("r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    row = json.loads(line_s)
                    uid = row["utterance_id"]
                    uri = row["audio_uri"]
                    clips.append((uid, (base_dir / uri).resolve()))
        else:
            # Single audio file
            clips.append((audio_paths_or_manifest.stem, audio_paths_or_manifest))
    elif isinstance(audio_paths_or_manifest, Path) and audio_paths_or_manifest.is_dir():
        # Directory of audio files
        for ext in ("*.flac", "*.wav", "*.mp3"):
            for p in sorted(audio_paths_or_manifest.glob(ext)):
                clips.append((p.stem, p))
    elif isinstance(audio_paths_or_manifest, Sequence):
        for p in audio_paths_or_manifest:
            p_obj = Path(p)
            clips.append((p_obj.stem, p_obj))
    else:
        raise ValueError(f"Unsupported audio input specification: {audio_paths_or_manifest}")

    if not clips:
        raise ValueError("No audio clips found to compare")

    # 6. Execute per-clip comparison
    outcomes: list[ClipComparisonOutcome] = []
    for uid, clip_path in clips:
        try:
            if audio_loader is not None:
                samples = audio_loader(clip_path)
            else:
                samples = load_audio_clip(clip_path)

            if len(samples) > 480000:
                raise ValueError(
                    f"Audio clip {uid} exceeds 30 s limit (480,000 samples at 16 kHz): "
                    f"got {len(samples)} samples. Expected 10-20 s clips per M1 audit "
                    f"(data/metadata/worldspeech-tr_tr-source-audit.md:52)."
                )

            features = processor(
                samples,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features

            if hasattr(features, "to"):
                m_dev = getattr(model, "device", target_device)
                m_dtype = getattr(model, "dtype", None)
                if m_dtype is not None:
                    features = features.to(device=m_dev, dtype=m_dtype)
                else:
                    features = features.to(device=m_dev)

            res = run_whisper_rank1_comparison(
                model=model,
                processor=processor,
                input_features=features,
                nbest_fn=shared_nbest,
                beam_width=resolved_beam,
                max_new_tokens=resolved_max_tokens,
                language="tr",
                task="transcribe",
            )
            outcomes.append(ClipComparisonOutcome(clip_id=uid, result=res))
        except Exception as exc:
            err_res = Rank1ComparisonResult(
                matched=False,
                status="error",
                details=f"Comparison failed on clip {uid}: {exc}",
                error=str(exc),
            )
            outcomes.append(ClipComparisonOutcome(clip_id=uid, result=err_res))

    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper Rank-1 vs Single-Result Smoke Check")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic Whisper fixture")
    parser.add_argument("--config", type=str, default=None, help="Path to baseline configuration JSON (B1 or B2)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to local checkpoint directory or weight file")
    parser.add_argument("--audio", type=str, default=None, help="Path to local audio file or folder of audio files")
    parser.add_argument("--manifest", type=str, default=None, help="Path to reference-free input manifest JSONL")
    parser.add_argument("--audio-root", type=str, default=None, help="Root folder for relative audio_uri in manifest")
    parser.add_argument("--shared-decoder-path", type=str, default=None, help="Path to shared decoder checkout/export")
    parser.add_argument("--beam-width", type=int, default=None, help="Beam width and requested_n override")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens override")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cpu, cuda)")
    parser.add_argument("--show-transcripts", action="store_true", help="Print decoded hypothesis text (hidden by default)")
    args = parser.parse_args()

    if args.synthetic:
        beam = args.beam_width or 2
        max_tokens = args.max_new_tokens or 10
        result = run_synthetic_smoke_check(
            shared_decoder_path=args.shared_decoder_path,
            beam_width=beam,
            max_new_tokens=max_tokens,
        )
        print("=== Whisper Rank-1 Synthetic Smoke Check ===")
        print(f"Status:             {result.status}")
        print(f"Matched:            {result.matched}")
        print(f"Text exact match:   {result.text_exact_match}")
        print(f"Tokens exact match: {result.tokens_exact_match}")
        if args.show_transcripts:
            print(f"Single text:        {result.single_text!r}")
            print(f"N-best rank 1 text: {result.nbest_rank1_text!r}")
        print(f"Single tokens:      {result.single_token_ids}")
        print(f"N-best tokens:      {result.nbest_rank1_token_ids}")
        if result.token_diff:
            print(f"Token diff:         {result.token_diff}")
        print(f"Details:            {result.details}")
        if result.error:
            print(f"Error:              {result.error}")
        return 0 if result.matched else 1

    if not args.config or not args.checkpoint or not (args.audio or args.manifest):
        parser.error(
            "Real-audio smoke check requires --config, --checkpoint, and either --audio or --manifest. "
            "Use --synthetic to run the synthetic fixture comparison."
        )

    audio_target = Path(args.manifest) if args.manifest else Path(args.audio)
    try:
        outcomes = run_real_audio_smoke_check(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            audio_paths_or_manifest=audio_target,
            audio_root=args.audio_root,
            shared_decoder_path=args.shared_decoder_path,
            beam_width=args.beam_width,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    except CheckpointValidationError as c_err:
        print(f"CHECKPOINT ERROR: {c_err}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"EXECUTION ERROR: {exc}", file=sys.stderr)
        return 1

    print("=== Whisper Rank-1 Real-Audio Smoke Check ===")
    matches = 0
    mismatches = 0
    errors = 0

    for outcome in outcomes:
        res = outcome.result
        uid = outcome.clip_id
        if res.status == "match":
            matches += 1
            verdict = "MATCH"
        elif res.status == "error":
            errors += 1
            verdict = "ERROR"
        else:
            mismatches += 1
            verdict = "MISMATCH"

        # Report status without printing transcripts by default
        print(f"Clip {uid}: {verdict} | Text Match: {res.text_exact_match} | Tokens Match: {res.tokens_exact_match}")
        if res.token_diff:
            print(f"  Token diff: {res.token_diff}")
        if res.error:
            print(f"  Error: {res.error}")
        if args.show_transcripts:
            print(f"  Single text: {res.single_text!r}")
            print(f"  N-best text: {res.nbest_rank1_text!r}")

    total = len(outcomes)
    print("\n=== Smoke Check Summary ===")
    print(f"Total clips: {total}")
    print(f"Matches:     {matches}")
    print(f"Mismatches:  {mismatches}")
    print(f"Errors:      {errors}")
    verdict_str = "PASSED" if (matches == total and total > 0) else "FAILED"
    print(f"Verdict:     {verdict_str}")

    return 0 if verdict_str == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
