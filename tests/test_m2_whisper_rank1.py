"""Tests for Whisper rank-1 comparison against single-result generation.

Validates the Member 2 rank-1 smoke comparison logic using deterministic fixtures
for matches, mismatches, missing candidates, and decoding errors, as well as
a real framework synthetic test using a randomly initialized Whisper model.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from turkish_asr_rdl.baselines.whisper_rank1 import (
    DEFAULT_WHISPER_PROMPT_TOKENS,
    Rank1ComparisonResult,
    compare_rank1_outputs,
    run_whisper_rank1_comparison,
    strip_known_prompt,
)


class DeterministicWhisperRank1ComparisonTests(unittest.TestCase):
    """Offline unit tests using deterministic fixtures without model downloads."""

    def test_exact_match_fixture(self):
        """When single-result and rank-1 match identically, return match status."""
        single_res = {
            "status": "ok",
            "text": " Turkiye Buyuk Millet Meclisi",
            "token_ids": [101, 102, 103],
        }
        nbest_cands = [
            {
                "candidate_id": "beam-0001",
                "rank": 1,
                "text": " Turkiye Buyuk Millet Meclisi",
                "token_ids": [101, 102, 103],
                "acoustic_log_score": -12.5,
            },
            {
                "candidate_id": "beam-0002",
                "rank": 2,
                "text": " Turkiye Buyuk Millet",
                "token_ids": [101, 102],
                "acoustic_log_score": -14.2,
            },
        ]

        result = compare_rank1_outputs(single_res, nbest_cands)
        self.assertTrue(result.matched)
        self.assertEqual(result.status, "match")
        self.assertTrue(result.text_exact_match)
        self.assertTrue(result.tokens_exact_match)
        self.assertIsNone(result.token_diff)
        self.assertEqual(result.single_text, " Turkiye Buyuk Millet Meclisi")
        self.assertEqual(result.nbest_rank1_text, " Turkiye Buyuk Millet Meclisi")

    def test_text_and_token_mismatch_fixture(self):
        """When rank 1 differs from single-result, report mismatch and token diff."""
        single_res = {
            "status": "ok",
            "text": " birinci secenek",
            "token_ids": [201, 202],
        }
        nbest_cands = [
            {
                "candidate_id": "beam-0001",
                "rank": 1,
                "text": " ikinci secenek",
                "token_ids": [203, 202],
                "acoustic_log_score": -10.0,
            }
        ]

        result = compare_rank1_outputs(single_res, nbest_cands)
        self.assertFalse(result.matched)
        self.assertEqual(result.status, "mismatch")
        self.assertFalse(result.text_exact_match)
        self.assertFalse(result.tokens_exact_match)
        self.assertIsNotNone(result.token_diff)
        self.assertEqual(result.token_diff.first_diff_index, 0)
        self.assertEqual(result.token_diff.single_token_at_diff, 201)
        self.assertEqual(result.token_diff.nbest_token_at_diff, 203)

    def test_verbatim_whitespace_preservation_no_stripping(self):
        """Leading/trailing whitespace differences must NOT be stripped or normalized."""
        single_res = {
            "status": "ok",
            "text": " Meclis",  # Leading space as emitted by raw Whisper
            "token_ids": [301],
        }
        nbest_cands = [
            {
                "candidate_id": "beam-0001",
                "rank": 1,
                "text": "Meclis",  # Stripped text
                "token_ids": [301],
            }
        ]

        result = compare_rank1_outputs(single_res, nbest_cands)
        self.assertFalse(result.matched, "Leading whitespace difference must prevent match")
        self.assertEqual(result.status, "mismatch")
        self.assertFalse(result.text_exact_match)
        self.assertTrue(result.tokens_exact_match)

    def test_missing_rank1_candidate_fixture(self):
        """Report missing_rank1 when candidate list is empty or lacks rank 1."""
        single_res = {
            "status": "ok",
            "text": " test",
            "token_ids": [401],
        }

        # Case A: Empty candidate list
        res_empty = compare_rank1_outputs(single_res, [])
        self.assertFalse(res_empty.matched)
        self.assertEqual(res_empty.status, "missing_rank1")
        self.assertIn("empty", res_empty.details)

        # Case B: List with candidates starting at rank 2
        res_no_rank1 = compare_rank1_outputs(single_res, [{"candidate_id": "c2", "rank": 2, "text": " b"}])
        self.assertFalse(res_no_rank1.matched)
        self.assertEqual(res_no_rank1.status, "missing_rank1")
        self.assertIn("rank == 1", res_no_rank1.details)

    def test_decode_failure_error_reporting(self):
        """Report error status separately from mismatches when either path fails."""
        # Case A: Single-result failed
        single_err = {
            "status": "error",
            "error": "CUDA out of memory",
            "text": None,
            "token_ids": [],
        }
        nbest_ok = [{"candidate_id": "c1", "rank": 1, "text": " ok", "token_ids": [501]}]
        res_a = compare_rank1_outputs(single_err, nbest_ok)
        self.assertFalse(res_a.matched)
        self.assertEqual(res_a.status, "error")
        self.assertIn("CUDA out of memory", res_a.details)

        # Case B: N-best failed
        single_ok = {"status": "ok", "text": " ok", "token_ids": [501]}
        nbest_err = {"status": "error", "error": "Model forward pass failed"}
        res_b = compare_rank1_outputs(single_ok, nbest_err)
        self.assertFalse(res_b.matched)
        self.assertEqual(res_b.status, "error")
        self.assertIn("Model forward pass failed", res_b.details)

    def test_prompt_token_stripping(self):
        """Prompt tokens at the beginning of raw sequence are stripped before comparison."""
        prompt = [50258, 50359, 50363, 50365]
        tokens_with_prompt = prompt + [10, 20, 30]
        cleaned = strip_known_prompt(tokens_with_prompt, prompt_token_ids=prompt)
        self.assertEqual(cleaned, [10, 20, 30])

        # If prompt is not at the start, don't strip
        tokens_no_prompt = [99] + prompt
        cleaned_no = strip_known_prompt(tokens_no_prompt, prompt_token_ids=prompt)
        self.assertEqual(cleaned_no, [99] + prompt)


class RealFrameworkSyntheticRank1Check(unittest.TestCase):
    """Exercise rank-1 comparison on a synthetic Whisper model using genuine packages."""

    def test_synthetic_whisper_rank1_execution(self):
        """Opt-in integration test: run_whisper_rank1_comparison with real torch and shared decoder."""
        shared_decoder_path = os.environ.get("TURAS_SHARED_DECODER_PATH")
        if not shared_decoder_path:
            self.skipTest(
                "Synthetic shared-decoder integration test skipped: set TURAS_SHARED_DECODER_PATH "
                "to an explicit path to run with a separate checkout of the shared decoder."
            )

        export_path = Path(shared_decoder_path)
        if not export_path.is_dir():
            self.skipTest(f"Configured TURAS_SHARED_DECODER_PATH does not exist: {export_path}")

        whisper_candidates = [
            export_path / "turkish_asr_rdl" / "decoding" / "whisper.py",
            export_path / "decoding" / "whisper.py",
            export_path / "whisper.py",
        ]
        whisper_path = next((p for p in whisper_candidates if p.is_file()), None)
        if not whisper_path:
            self.skipTest(f"whisper.py not found under TURAS_SHARED_DECODER_PATH ({export_path})")

        try:
            import importlib.util
            import torch
            from transformers import WhisperConfig, WhisperForConditionalGeneration

            spec = importlib.util.spec_from_file_location("shared_whisper_decoding", whisper_path)
            if spec is None or spec.loader is None:
                self.skipTest(f"Could not create importlib spec for {whisper_path}")
            whisper_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(whisper_mod)
            whisper_nbest = getattr(whisper_mod, "whisper_nbest")
        except Exception as err:
            self.skipTest(f"Required package or shared helper unavailable: {err}")

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

        # Run comparison with beam_width=2
        result = run_whisper_rank1_comparison(
            model,
            processor,
            input_features,
            nbest_fn=whisper_nbest,
            beam_width=2,
            max_new_tokens=10,
        )

        # Check that the comparison returned a valid result object without crashing
        self.assertIsInstance(result, Rank1ComparisonResult)
        self.assertIn(result.status, ("match", "mismatch", "missing_rank1", "error"))
        self.assertIsInstance(result.text_exact_match, bool)
        self.assertIsInstance(result.tokens_exact_match, bool)

        # Direct verification of candidate score against independent teacher-forced sum
        raw_cands, _ = whisper_nbest(
            model,
            processor,
            input_features,
            requested_n=2,
            max_new_tokens=10,
            deduplicate=False,
        )
        self.assertGreater(len(raw_cands), 0)
        prompt_tokens = [50258, 50359, 50363, 50365]
        for c in raw_cands:
            full_tokens = torch.tensor([prompt_tokens + list(c["token_ids"])])
            with torch.no_grad():
                logits = model(input_features=input_features, decoder_input_ids=full_tokens).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            p_len = len(prompt_tokens)
            tf_scores = [log_probs[0, p_len - 1 + i, tok].item() for i, tok in enumerate(c["token_ids"])]
            tf_sum = sum(tf_scores)
            diff = abs(c["acoustic_log_score"] - tf_sum)
            self.assertLess(diff, 1e-3, f"Whisper acoustic_log_score diff {diff} exceeds 1e-3")
            if c.get("decoder_sequence_score") is not None:
                self.assertNotEqual(c["decoder_sequence_score"], c["acoustic_log_score"])

        # Record outcome in test assertions
        print(f"\n[Real Framework Synthetic Rank-1 Check Outcome] Status: {result.status} | Text Match: {result.text_exact_match} | Tokens Match: {result.tokens_exact_match}")
        print(f"  Single result text: {result.single_text!r} | Tokens: {result.single_token_ids}")
        print(f"  N-best rank-1 text: {result.nbest_rank1_text!r} | Tokens: {result.nbest_rank1_token_ids}")


if __name__ == "__main__":
    unittest.main()
