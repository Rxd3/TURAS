"""Thin CLI runner for Whisper rank-1 vs single-result smoke comparison.

Compares rank 1 from the shared N-best decoder with standard single-result
generation (num_return_sequences=1) at the same beam width.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from turkish_asr_rdl.baselines.whisper_rank1 import (
    DEFAULT_WHISPER_PROMPT_TOKENS,
    Rank1ComparisonResult,
    compare_rank1_outputs,
    run_whisper_rank1_comparison,
)


def run_synthetic_smoke_check(
    shared_decoder_path: str | Path | None = None,
    beam_width: int = 2,
    max_new_tokens: int = 10,
) -> Rank1ComparisonResult:
    """Run comparison on a synthetic Whisper model using genuine torch/transformers."""
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    if not shared_decoder_path:
        shared_decoder_path = os.environ.get("TURAS_SHARED_DECODER_PATH")

    if not shared_decoder_path:
        return Rank1ComparisonResult(
            matched=False,
            status="error",
            details="Shared decoder path not supplied. Provide --shared-decoder-path or set TURAS_SHARED_DECODER_PATH.",
            error="Missing shared decoder path",
        )

    export_path = Path(shared_decoder_path)
    if not export_path.is_dir():
        return Rank1ComparisonResult(
            matched=False,
            status="error",
            details=f"Shared decoder directory does not exist: {export_path}",
            error="Directory not found",
        )

    whisper_candidates = [
        export_path / "turkish_asr_rdl" / "decoding" / "whisper.py",
        export_path / "decoding" / "whisper.py",
        export_path / "whisper.py",
    ]
    whisper_path = next((p for p in whisper_candidates if p.is_file()), None)
    if not whisper_path:
        return Rank1ComparisonResult(
            matched=False,
            status="error",
            details=f"whisper.py not found under {export_path}",
            error="whisper.py missing",
        )

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("shared_whisper_decoding", whisper_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create importlib spec for {whisper_path}")
        whisper_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(whisper_mod)
        whisper_nbest = getattr(whisper_mod, "whisper_nbest")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper Rank-1 vs Single-Result Smoke Check")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic Whisper fixture")
    parser.add_argument("--shared-decoder-path", type=str, default=None, help="Path to shared decoder checkout/export")
    parser.add_argument("--beam-width", type=int, default=2, help="Beam width and requested_n")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Max new tokens to generate")
    args = parser.parse_args()

    if args.synthetic:
        result = run_synthetic_smoke_check(
            shared_decoder_path=args.shared_decoder_path,
            beam_width=args.beam_width,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        print("No input provided. Use --synthetic to run the synthetic fixture comparison.")
        return 0

    print("=== Whisper Rank-1 Smoke Check Result ===")
    print(f"Status:             {result.status}")
    print(f"Matched:            {result.matched}")
    print(f"Text exact match:   {result.text_exact_match}")
    print(f"Tokens exact match: {result.tokens_exact_match}")
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


if __name__ == "__main__":
    sys.exit(main())
