"""Unit and integration tests for Member 2 QwenBaselineAdapter (B3 candidates).

Tests cover:
1. Deterministic scoring exercising compute_transition_scores and step scores to verify
   exclusion of prompt and padding tokens and exact inclusion of EOS under the shared convention.
2. Deterministic configuration parsing and strict validation for both pinned Qwen3-ASR
   candidates (1.7B-hf and 0.6B-hf) without selecting either candidate.
3. Decoding parameter validation and rejection of unsupported overrides.
4. Opt-in integration tests with the shared decoding contract from integration revision b6afa13:
   - check_adapter protocol conformance.
   - Reference-free audio-only transcription preserving batch order and utterance IDs.
   - Turkish transcription request wiring (language="tr").
   - Empty audio array and decoding error handling without dropping IDs.
   - Verbatim raw text preservation.
   - Score separation (speech_llm_joint_token_log_prob_sum vs decoder_sequence_score).
   - Raw candidate ordering and deduplication delegation to shared export.
   - Export pipeline integration with prediction_row, nbest_row, and validate_predictions.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from turkish_asr_rdl.baselines.qwen import (
    DEFAULT_ACOUSTIC_KIND,
    DEFAULT_SCORE_CONVENTION,
    PINNED_QWEN_LFS_OIDS,
    PINNED_QWEN_REVISIONS,
    QwenBaselineAdapter,
    compute_joint_log_probs_from_step_scores,
    connect_shared_decoding,
    cut_at_stop,
    get_shared_interface,
)


class QwenDeterministicScoringTests(unittest.TestCase):
    """Deterministic verification of speech-LLM joint token log-prob sum calculation."""

    def test_cut_at_stop_logic(self):
        """Tokens are sliced through the first stop token; trailing tokens are dropped."""
        stop_ids = frozenset({99})
        # Normal sequence terminated with EOS
        self.assertEqual(cut_at_stop([10, 20, 99, 0, 0], stop_ids), (10, 20, 99))
        # Sequence starting with EOS
        self.assertEqual(cut_at_stop([99, 0, 0], stop_ids), (99,))
        # Sequence without EOS
        self.assertEqual(cut_at_stop([10, 20, 30], stop_ids), (10, 20, 30))
        # Empty sequence
        self.assertEqual(cut_at_stop([], stop_ids), ())

    def test_compute_joint_from_step_scores(self):
        """Per-step scores sum only through EOS; prompt (excluded upfront) and padding are ignored."""
        stop_ids = frozenset({99})
        gen_tokens = [101, 102, 99, 0, 0]  # 99 is EOS, 0 is PAD
        step_scores = [-1.5, -2.5, -0.5, -99.0, -99.0]

        score, kept = compute_joint_log_probs_from_step_scores(step_scores, gen_tokens, stop_ids)
        self.assertEqual(kept, (101, 102, 99))
        self.assertAlmostEqual(score, -4.5, places=5)

    def test_deterministic_scoring_via_compute_transition_scores_greedy(self):
        """Deterministic test exercising the exact adapter path for greedy generation scoring.

        Verifies that:
        1. compute_transition_scores with normalize_logits=True scores each generation step.
        2. Prompt tokens are strictly excluded (compute_transition_scores only scores generation steps).
        3. compute_joint_log_probs_from_step_scores includes EOS and truncates padding tokens after EOS.
        4. The computed speech_llm_joint_token_log_prob_sum matches the exact log-probability sum.
        """
        try:
            import torch
            from transformers.generation.utils import GenerationMixin
        except ImportError:
            self.skipTest("PyTorch and Transformers are required for deterministic scoring test")

        class DummyTextConfig:
            vocab_size = 10

            def get_text_config(self):
                return self

        class DummyModel(GenerationMixin):
            def __init__(self):
                self.config = DummyTextConfig()

        model = DummyModel()
        vocab_size = 10
        eos_id = 9
        pad_id = 0
        stop_ids = frozenset({eos_id})

        prompt_tokens = [1, 2, 3]  # length 3
        generated_tokens = [4, 5, eos_id, pad_id, pad_id]  # length 5
        prompt_t = torch.tensor([prompt_tokens], dtype=torch.long)
        gen_t = torch.tensor([generated_tokens], dtype=torch.long)
        sequences = torch.cat([prompt_t, gen_t], dim=1)  # shape [1, 8]

        # 5 generation step score tensors, each shape [1, vocab_size]
        scores = []
        for t in range(5):
            s = torch.zeros(1, vocab_size)
            if t == 0:
                s[0, 4] = 10.0
            elif t == 1:
                s[0, 5] = 10.0
            elif t == 2:
                s[0, eos_id] = 10.0
            elif t == 3:
                s[0, pad_id] = -50.0
            elif t == 4:
                s[0, pad_id] = -50.0
            scores.append(s)

        # Exact path called in QwenBaselineAdapter._decode_nbest_native for greedy (requested_n=1)
        transition_scores = model.compute_transition_scores(
            sequences, tuple(scores), normalize_logits=True
        )
        step_rows = transition_scores.float().tolist()
        gen_tokens_extracted = sequences[0, len(prompt_tokens):].tolist()

        score, kept = compute_joint_log_probs_from_step_scores(
            step_rows[0], gen_tokens_extracted, stop_ids
        )

        # Verify kept tokens are strictly [4, 5, 9] (EOS included, padding excluded)
        self.assertEqual(kept, (4, 5, eos_id))

        # Expected score: sum of log_softmax for token 4 at step 0, token 5 at step 1, and token 9 at step 2
        p_val = math.exp(10.0) / (math.exp(10.0) + 9.0)
        expected_token_log_prob = math.log(p_val)
        expected_total = 3 * expected_token_log_prob

        self.assertAlmostEqual(score, expected_total, places=4)
        # If padding tokens were accidentally included, score would include values < -50.0
        self.assertGreater(score, -10.0)

    def test_deterministic_scoring_via_compute_transition_scores_beam_search(self):
        """Deterministic test exercising the exact adapter path for beam search scoring.

        Verifies that:
        1. compute_transition_scores with beam_indices and normalize_logits=False extracts beam transitions.
        2. Beams finishing early (with negative beam index and trailing pad steps) have pad steps excluded.
        3. EOS is included for all beams, and scores match exact step sums through EOS.
        4. Sequences scores (framework beam score) stay separate from acoustic_log_score.
        """
        try:
            import torch
            from transformers.generation.utils import GenerationMixin
        except ImportError:
            self.skipTest("PyTorch and Transformers are required for deterministic scoring test")

        class DummyTextConfig:
            vocab_size = 10

            def get_text_config(self):
                return self

        class DummyModel(GenerationMixin):
            def __init__(self):
                self.config = DummyTextConfig()

        model = DummyModel()
        vocab_size = 10
        eos_id = 9
        pad_id = 0
        stop_ids = frozenset({eos_id})

        prompt_tokens = [1, 2]  # length 2
        # Beam 0: [4, eos_id, pad_id] (finishes early at step 1)
        # Beam 1: [5, 6, eos_id] (finishes at step 2)
        prompt_b = torch.tensor([prompt_tokens, prompt_tokens], dtype=torch.long)
        gen_b = torch.tensor([[4, eos_id, pad_id], [5, 6, eos_id]], dtype=torch.long)
        sequences = torch.cat([prompt_b, gen_b], dim=1)  # shape [2, 5]

        # Beam indices: beam 0 finished early, so step 2 has -1
        beam_indices = torch.tensor([[0, 0, -1], [1, 1, 1]], dtype=torch.long)

        # 3 generation step score tensors (log-probs), shape [2, vocab_size]
        scores = []
        for t in range(3):
            s = torch.zeros(2, vocab_size)
            if t == 0:
                s[0, 4] = -0.15
                s[1, 5] = -0.25
            elif t == 1:
                s[0, eos_id] = -0.35
                s[1, 6] = -0.45
            elif t == 2:
                s[0, pad_id] = -99.0
                s[1, eos_id] = -0.55
            scores.append(s)

        # Exact path called in QwenBaselineAdapter._decode_nbest_native for beam search (requested_n > 1)
        transition_scores = model.compute_transition_scores(
            sequences, tuple(scores), beam_indices=beam_indices, normalize_logits=False
        )
        step_rows = transition_scores.float().tolist()

        # Simulated framework sequences_scores (beam scores with length penalty)
        framework_beam_scores = [-0.38, -0.92]

        # Verify Beam 0 (finished early)
        gen_tokens_0 = sequences[0, len(prompt_tokens):].tolist()
        score_0, kept_0 = compute_joint_log_probs_from_step_scores(
            step_rows[0], gen_tokens_0, stop_ids
        )
        self.assertEqual(kept_0, (4, eos_id))
        self.assertAlmostEqual(score_0, -0.15 + -0.35, places=5)
        # Verify framework beam score stays separate
        self.assertNotEqual(score_0, framework_beam_scores[0])

        # Verify Beam 1 (full length)
        gen_tokens_1 = sequences[1, len(prompt_tokens):].tolist()
        score_1, kept_1 = compute_joint_log_probs_from_step_scores(
            step_rows[1], gen_tokens_1, stop_ids
        )
        self.assertEqual(kept_1, (5, 6, eos_id))
        self.assertAlmostEqual(score_1, -0.25 + -0.45 + -0.55, places=5)
        self.assertNotEqual(score_1, framework_beam_scores[1])


class QwenBaselineAdapterConfigTests(unittest.TestCase):
    """Deterministic offline tests for Qwen baseline configuration parsing and validation."""

    def setUp(self) -> None:
        self.c1_path = REPO_ROOT / "configs" / "baselines" / "qwen3_asr_1_7b.json"
        self.c2_path = REPO_ROOT / "configs" / "baselines" / "qwen3_asr_0_6b.json"
        with self.c1_path.open("r", encoding="utf-8") as f:
            self.c1_config = json.load(f)
        with self.c2_path.open("r", encoding="utf-8") as f:
            self.c2_config = json.load(f)

    def test_b3_candidate_1_config_binding(self):
        """Candidate 1 (1.7B-hf) loads with exact pinned revisions, 16 kHz, and unselected status."""
        adapter = QwenBaselineAdapter.from_config_file(self.c1_path)
        self.assertEqual(adapter.system_id, "baseline-qwen3-asr-1.7b")
        self.assertEqual(adapter.model_id, "Qwen/Qwen3-ASR-1.7B-hf")
        self.assertEqual(adapter.model_revision, PINNED_QWEN_REVISIONS["Qwen/Qwen3-ASR-1.7B-hf"])
        self.assertEqual(adapter.processor_id, "Qwen/Qwen3-ASR-1.7B-hf")
        self.assertEqual(adapter.processor_revision, PINNED_QWEN_REVISIONS["Qwen/Qwen3-ASR-1.7B-hf"])
        self.assertEqual(adapter.sampling_rate_hz, 16000)
        self.assertEqual(adapter.requested_n, 5)
        self.assertEqual(adapter.language, "tr")
        self.assertEqual(adapter.score_convention_id, DEFAULT_SCORE_CONVENTION)
        self.assertEqual(adapter.acoustic_score_kind, DEFAULT_ACOUSTIC_KIND)
        self.assertEqual(adapter.model_artifact_sha256, PINNED_QWEN_LFS_OIDS["Qwen/Qwen3-ASR-1.7B-hf"])
        self.assertEqual(len(adapter.decoder_config_sha256), 64)
        self.assertEqual(self.c1_config.get("candidate_status"), "unselected-candidate")

    def test_b3_candidate_2_config_binding(self):
        """Candidate 2 (0.6B-hf) loads with exact pinned revisions, 16 kHz, and unselected status."""
        adapter = QwenBaselineAdapter.from_config_file(self.c2_path)
        self.assertEqual(adapter.system_id, "baseline-qwen3-asr-0.6b")
        self.assertEqual(adapter.model_id, "Qwen/Qwen3-ASR-0.6B-hf")
        self.assertEqual(adapter.model_revision, PINNED_QWEN_REVISIONS["Qwen/Qwen3-ASR-0.6B-hf"])
        self.assertEqual(adapter.processor_id, "Qwen/Qwen3-ASR-0.6B-hf")
        self.assertEqual(adapter.processor_revision, PINNED_QWEN_REVISIONS["Qwen/Qwen3-ASR-0.6B-hf"])
        self.assertEqual(adapter.sampling_rate_hz, 16000)
        self.assertEqual(adapter.requested_n, 5)
        self.assertEqual(adapter.language, "tr")
        self.assertEqual(adapter.score_convention_id, DEFAULT_SCORE_CONVENTION)
        self.assertEqual(adapter.acoustic_score_kind, DEFAULT_ACOUSTIC_KIND)
        self.assertEqual(adapter.model_artifact_sha256, PINNED_QWEN_LFS_OIDS["Qwen/Qwen3-ASR-0.6B-hf"])
        self.assertEqual(len(adapter.decoder_config_sha256), 64)
        self.assertEqual(self.c2_config.get("candidate_status"), "unselected-candidate")

    def test_provisional_requested_n_configuration(self):
        """Adapter supports requested_n=5 (smoke default) and provisional overrides."""
        adapter_5 = QwenBaselineAdapter(self.c1_config, requested_n=5)
        self.assertEqual(adapter_5.requested_n, 5)
        self.assertEqual(adapter_5.config["decoding"]["requested_n"], 5)
        self.assertEqual(adapter_5.config["decoding"]["num_beams"], 5)

        adapter_10 = QwenBaselineAdapter(self.c1_config, requested_n=10)
        self.assertEqual(adapter_10.requested_n, 10)
        self.assertEqual(adapter_10.config["decoding"]["requested_n"], 10)
        self.assertEqual(adapter_10.config["decoding"]["num_beams"], 10)

    def test_invalid_requested_n_raises(self):
        """Non-positive or non-integer requested_n raises ValueError."""
        for invalid_n in (0, -1, -5, True, False, "5", 5.0):
            with self.subTest(invalid_n=invalid_n):
                with self.assertRaises(ValueError):
                    QwenBaselineAdapter(self.c1_config, requested_n=invalid_n)

    def test_unpinned_revision_raises(self):
        """Altering pinned model revision raises ValueError."""
        bad_cfg = copy.deepcopy(self.c1_config)
        bad_cfg["model"]["revision"] = "0" * 40
        bad_cfg["processor"]["revision"] = "0" * 40
        with self.assertRaises(ValueError) as ctx:
            QwenBaselineAdapter(bad_cfg)
        self.assertIn("Model revision mismatch", str(ctx.exception))

    def test_unknown_model_id_raises(self):
        """Unknown or unpinned model ID raises ValueError."""
        bad_cfg = copy.deepcopy(self.c1_config)
        bad_cfg["model"]["id"] = "Qwen/Qwen3-ASR-7B-hf"
        with self.assertRaises(ValueError) as ctx:
            QwenBaselineAdapter(bad_cfg)
        self.assertIn("Unknown or unpinned model ID", str(ctx.exception))

    def test_model_and_processor_id_mismatch_raises(self):
        """Mismatched processor ID raises ValueError."""
        bad_cfg = copy.deepcopy(self.c1_config)
        bad_cfg["processor"]["id"] = "Qwen/Qwen3-ASR-0.6B-hf"
        with self.assertRaises(ValueError) as ctx:
            QwenBaselineAdapter(bad_cfg)
        self.assertIn("Processor ID", str(ctx.exception))

    def test_non_16k_sampling_rate_raises(self):
        """Sampling rate other than 16000 Hz raises ValueError."""
        bad_cfg = copy.deepcopy(self.c1_config)
        bad_cfg["processor"]["sampling_rate_hz"] = 8000
        with self.assertRaises(ValueError) as ctx:
            QwenBaselineAdapter(bad_cfg)
        self.assertIn("16000 Hz", str(ctx.exception))

    def test_unsupported_decoding_overrides_raise(self):
        """Adapter enforces documented baseline decoding options and rejects unsupported overrides."""
        overrides = [
            ({"do_sample": True}, "do_sample=False"),
            ({"temperature": 0.5}, "temperature is not supported"),
            ({"language": "en"}, "language='tr'"),
            ({"length_penalty": 1.5}, "length_penalty != 1.0 is not supported"),
            ({"early_stopping": True}, "early_stopping=True is not supported"),
            ({"repetition_penalty": 1.2}, "repetition_penalty != 1.0 is not supported"),
            ({"no_repeat_ngram_size": 3}, "no_repeat_ngram_size != 0 is not supported"),
            ({"algorithm": "greedy"}, "algorithm='beam'"),
        ]
        for dec_override, expected_msg in overrides:
            with self.subTest(override=dec_override):
                bad_cfg = copy.deepcopy(self.c1_config)
                bad_cfg["decoding"].update(dec_override)
                with self.assertRaises(ValueError) as ctx:
                    QwenBaselineAdapter(bad_cfg)
                self.assertIn(expected_msg, str(ctx.exception))

    def test_model_artifact_sha256_validation(self):
        """model_artifact_sha256 requires 64-char lowercase hex; rejects placeholders."""
        bad_cfg = copy.deepcopy(self.c1_config)
        bad_cfg["model"]["weight_sha256"] = None
        with self.assertRaises(ValueError) as ctx:
            QwenBaselineAdapter(bad_cfg)
        self.assertIn("model_artifact_sha256", str(ctx.exception))

        # Explicit fixture parameter works cleanly
        explicit_sha = "e" * 64
        adapter = QwenBaselineAdapter(bad_cfg, model_artifact_sha256=explicit_sha)
        self.assertEqual(adapter.model_artifact_sha256, explicit_sha)

    def test_mock_loaders_execution(self):
        """Injected model and processor loaders are invoked without real downloads."""
        dummy_model = object()
        dummy_processor = object()
        adapter = QwenBaselineAdapter(
            self.c1_config,
            model_loader=lambda: dummy_model,
            processor_loader=lambda: dummy_processor,
        )
        adapter.load()
        self.assertIs(adapter.model, dummy_model)
        self.assertIs(adapter.processor, dummy_processor)


class QwenSharedDecoderIntegrationTests(unittest.TestCase):
    """Opt-in integration tests using the shared decoding contract from revision b6afa13."""

    @classmethod
    def setUpClass(cls) -> None:
        shared_path = os.environ.get("TURAS_SHARED_DECODER_PATH")
        if not shared_path or not os.path.exists(shared_path):
            cls.shared_available = False
            cls.skip_reason = (
                "Shared decoding contract integration test skipped: set TURAS_SHARED_DECODER_PATH "
                "to an explicit path to run with a separate checkout of the shared decoder."
            )
            return

        cls.shared_path = shared_path
        if not connect_shared_decoding(shared_path):
            cls.shared_available = False
            cls.skip_reason = f"Could not import shared modules from {shared_path}"
            return

        cls.shared_available = True
        try:
            from turkish_asr_rdl.decoding.export import nbest_row, prediction_row, RunContext
            from turkish_asr_rdl.decoding.interface import (
                AudioItem,
                Candidate,
                Hypothesis,
                InferenceInput,
                check_adapter,
            )
            from turkish_asr_rdl.schemas.predictions import validate_predictions

            cls.AudioItem = AudioItem
            cls.Candidate = Candidate
            cls.Hypothesis = Hypothesis
            cls.InferenceInput = InferenceInput
            cls.check_adapter = staticmethod(check_adapter)
            cls.prediction_row = staticmethod(prediction_row)
            cls.nbest_row = staticmethod(nbest_row)
            cls.RunContext = RunContext
            cls.validate_predictions = staticmethod(validate_predictions)
        except Exception as exc:
            cls.shared_available = False
            cls.skip_reason = f"Failed to import shared contract components: {exc}"

    def setUp(self) -> None:
        if not getattr(self, "shared_available", False):
            self.skipTest(self.skip_reason)

        self.c1_path = REPO_ROOT / "configs" / "baselines" / "qwen3_asr_1_7b.json"
        self.c2_path = REPO_ROOT / "configs" / "baselines" / "qwen3_asr_0_6b.json"
        with self.c1_path.open("r", encoding="utf-8") as f:
            self.c1_config = json.load(f)
        with self.c2_path.open("r", encoding="utf-8") as f:
            self.c2_config = json.load(f)

    def test_check_adapter_protocol_conformance(self):
        """QwenBaselineAdapter passes check_adapter protocol verification for both candidate configs."""
        adapter_c1 = QwenBaselineAdapter(self.c1_config, shared_path=self.shared_path)
        adapter_c2 = QwenBaselineAdapter(self.c2_config, shared_path=self.shared_path)

        self.check_adapter(adapter_c1)
        self.check_adapter(adapter_c2)

    def test_reference_free_batch_order_and_id_preservation(self):
        """Transcribe batch of AudioItems reference-free, preserving exact order and IDs."""
        u_ids = ["qwen-utt-1", "qwen-utt-2", "qwen-utt-3"]

        class MockProcessor:
            def __init__(self):
                self.calls = []

            def apply_transcription_request(self, audio, language):
                self.calls.append({"audio_len": len(audio), "language": language})
                return {"input_ids": np.zeros((1, 5), dtype=np.int64)}

        def dummy_nbest(model, processor, inputs, requested_n, max_new_tokens, language):
            return [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": "Turkiye Buyuk Millet Meclisi",
                    "token_ids": [101, 102],
                    "acoustic_log_score": -3.2,
                    "acoustic_score_kind": DEFAULT_ACOUSTIC_KIND,
                    "decoder_sequence_score": -0.32,
                    "score_details": {},
                }
            ], {}

        proc = MockProcessor()
        adapter = QwenBaselineAdapter(
            self.c1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: proc,
            nbest_fn=dummy_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        batch = [
            self.AudioItem(utterance_id=uid, samples=np.ones(1600, dtype=np.float32) * 0.01)
            for uid in u_ids
        ]

        hypotheses = adapter.transcribe(batch)
        self.assertEqual(len(hypotheses), len(u_ids))
        for hyp, uid in zip(hypotheses, u_ids):
            self.assertEqual(hyp.utterance_id, uid)
            self.assertEqual(hyp.status, "ok")
            self.assertEqual(hyp.text, "Turkiye Buyuk Millet Meclisi")
            self.assertEqual(hyp.selected_candidate_id, "beam-0001")
            self.assertEqual(len(hyp.candidates), 1)

        # Verify Turkish prompt wiring
        self.assertEqual(len(proc.calls), 3)
        for call in proc.calls:
            self.assertEqual(call["language"], "tr")

    def test_empty_audio_array_and_decode_error_without_dropping_ids(self):
        """Empty audio array returns status='empty' and errors return status='error' without dropping IDs."""
        class MockProcessor:
            def apply_transcription_request(self, audio, language):
                return {"input_ids": np.zeros((1, 5), dtype=np.int64)}

        call_count = 0

        def failing_nbest(model, processor, inputs, requested_n, max_new_tokens, language):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    {
                        "candidate_id": "beam-0001",
                        "rank": 1,
                        "text": "gecerli metin",
                        "token_ids": [201],
                        "acoustic_log_score": -2.0,
                        "acoustic_score_kind": DEFAULT_ACOUSTIC_KIND,
                        "decoder_sequence_score": None,
                    }
                ], {}
            raise RuntimeError("Synthetic Qwen decoder failure")

        adapter = QwenBaselineAdapter(
            self.c1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: MockProcessor(),
            nbest_fn=failing_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        batch = [
            self.AudioItem(utterance_id="q1", samples=np.ones(1600, dtype=np.float32) * 0.05),
            self.AudioItem(utterance_id="q2", samples=np.array([], dtype=np.float32)),
            self.AudioItem(utterance_id="q3", samples=np.array([0.1, np.nan, 0.2], dtype=np.float32)),
            self.AudioItem(utterance_id="q4", samples=np.ones(1600, dtype=np.float32) * 0.05),
        ]

        hypotheses = adapter.transcribe(batch)
        self.assertEqual(len(hypotheses), 4)

        # q1: ok
        self.assertEqual(hypotheses[0].utterance_id, "q1")
        self.assertEqual(hypotheses[0].status, "ok")
        self.assertEqual(hypotheses[0].text, "gecerli metin")

        # q2: empty
        self.assertEqual(hypotheses[1].utterance_id, "q2")
        self.assertEqual(hypotheses[1].status, "empty")
        self.assertEqual(hypotheses[1].text, "")
        self.assertEqual(hypotheses[1].candidates, ())

        # q3: error (InvalidAudio)
        self.assertEqual(hypotheses[2].utterance_id, "q3")
        self.assertEqual(hypotheses[2].status, "error")
        self.assertIsNone(hypotheses[2].text)
        self.assertEqual(hypotheses[2].error_code, "InvalidAudio")

        # q4: error (RuntimeError)
        self.assertEqual(hypotheses[3].utterance_id, "q4")
        self.assertEqual(hypotheses[3].status, "error")
        self.assertIsNone(hypotheses[3].text)
        self.assertEqual(hypotheses[3].error_code, "RuntimeError")

    def test_verbatim_raw_whitespace_preservation(self):
        """Decoded text is preserved verbatim including emitted leading whitespace without stripping."""
        class MockProcessor:
            def apply_transcription_request(self, audio, language):
                return {"input_ids": np.zeros((1, 5), dtype=np.int64)}

        raw_text_with_spaces = "  Sayin Baskan, degerli milletvekilleri  "

        def dummy_nbest(model, processor, inputs, requested_n, max_new_tokens, language):
            return [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": raw_text_with_spaces,
                    "token_ids": [301, 302],
                    "acoustic_log_score": -1.8,
                    "acoustic_score_kind": DEFAULT_ACOUSTIC_KIND,
                    "decoder_sequence_score": -0.18,
                }
            ], {}

        adapter = QwenBaselineAdapter(
            self.c1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: MockProcessor(),
            nbest_fn=dummy_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        batch = [self.AudioItem(utterance_id="q_ws", samples=np.ones(1600, dtype=np.float32))]
        results = adapter.transcribe(batch)

        self.assertEqual(results[0].text, raw_text_with_spaces)
        self.assertEqual(results[0].candidates[0].text, raw_text_with_spaces)

    def test_score_separation_and_none_preservation(self):
        """speech_llm_joint_token_log_prob_sum is kept in acoustic_log_score; beam score in decoder_sequence_score."""
        class MockProcessor:
            def apply_transcription_request(self, audio, language):
                return {"input_ids": np.zeros((1, 5), dtype=np.int64)}

        def missing_beam_score_nbest(model, processor, inputs, requested_n, max_new_tokens, language):
            return [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": "ornek",
                    "token_ids": [401],
                    "acoustic_log_score": -7.75,
                    "acoustic_score_kind": DEFAULT_ACOUSTIC_KIND,
                    "decoder_sequence_score": None,  # Preserved as None
                }
            ], {}

        adapter = QwenBaselineAdapter(
            self.c1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: MockProcessor(),
            nbest_fn=missing_beam_score_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        batch = [self.AudioItem(utterance_id="q_score", samples=np.ones(1600, dtype=np.float32))]
        results = adapter.transcribe(batch)
        cand = results[0].candidates[0]

        self.assertEqual(cand.acoustic_log_score, -7.75)
        self.assertEqual(cand.acoustic_score_kind, DEFAULT_ACOUSTIC_KIND)
        self.assertIsNone(cand.decoder_sequence_score)

        # Check export row formatting
        ctx = self.RunContext(
            run_id="run-qwen-score",
            system_id="baseline-qwen3-asr-1.7b",
            dataset_release="release-2026-v1",
            split="test",
            split_manifest_sha256="c" * 64,
            model_artifact_sha256=adapter.model_artifact_sha256,
            decoder_config_sha256=adapter.decoder_config_sha256,
        )
        inp = self.InferenceInput(
            schema_version="0.1",
            dataset_release="release-2026-v1",
            split_manifest_sha256="c" * 64,
            utterance_id="q_score",
            audio_uri="file:///audio.wav",
            audio_sha256="d" * 64,
            duration_s=1.0,
            sample_rate_hz=16000,
            channels=1,
            language="tr",
        )
        nb = self.nbest_row(results[0], inp, ctx, requested_n=5, score_convention_id=adapter.score_convention_id)
        self.assertIsNone(nb["candidates"][0]["decoder_sequence_score"])
        self.assertEqual(
            nb["candidates"][0]["score_details"]["null_reasons"]["decoder_sequence_score"],
            "not produced by this system",
        )

    def test_raw_candidate_order_and_deduplication_delegation(self):
        """Adapter returns unpadded raw candidates; shared export owns deduplication."""
        class MockProcessor:
            def apply_transcription_request(self, audio, language):
                return {"input_ids": np.zeros((1, 5), dtype=np.int64)}

        def duplicate_cands_nbest(model, processor, inputs, requested_n, max_new_tokens, language):
            return [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": "Madde 1",
                    "token_ids": [501],
                    "acoustic_log_score": -1.2,
                    "acoustic_score_kind": DEFAULT_ACOUSTIC_KIND,
                    "decoder_sequence_score": -0.12,
                },
                {
                    "candidate_id": "beam-0002",
                    "rank": 2,
                    "text": "Madde 1",  # Duplicate text
                    "token_ids": [502],
                    "acoustic_log_score": -2.3,
                    "acoustic_score_kind": DEFAULT_ACOUSTIC_KIND,
                    "decoder_sequence_score": -0.23,
                },
            ], {}

        adapter = QwenBaselineAdapter(
            self.c1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: MockProcessor(),
            nbest_fn=duplicate_cands_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        batch = [self.AudioItem(utterance_id="q_dup", samples=np.ones(1600, dtype=np.float32))]
        results = adapter.transcribe(batch)
        hyp = results[0]

        self.assertEqual(len(hyp.candidates), 2)
        self.assertEqual(hyp.candidates[0].candidate_id, "beam-0001")
        self.assertEqual(hyp.candidates[1].candidate_id, "beam-0002")

        # Exporter deduplicates
        ctx = self.RunContext(
            run_id="run-q-dup",
            system_id="baseline-qwen3-asr-1.7b",
            dataset_release="release-2026-v1",
            split="test",
            split_manifest_sha256="1" * 64,
            model_artifact_sha256=adapter.model_artifact_sha256,
            decoder_config_sha256=adapter.decoder_config_sha256,
        )
        inp = self.InferenceInput(
            schema_version="0.1",
            dataset_release="release-2026-v1",
            split_manifest_sha256="1" * 64,
            utterance_id="q_dup",
            audio_uri="file:///audio.wav",
            audio_sha256="2" * 64,
            duration_s=1.0,
            sample_rate_hz=16000,
            channels=1,
            language="tr",
        )
        nb_row = self.nbest_row(hyp, inp, ctx, requested_n=5, score_convention_id=adapter.score_convention_id)
        self.assertEqual(len(nb_row["candidates"]), 1)
        self.assertEqual(nb_row["dedup_map"], {"beam-0002": "beam-0001"})

    def test_full_predictions_schema_validation(self):
        """Adapter output hypotheses produce prediction rows that pass validate_predictions."""
        class MockProcessor:
            def apply_transcription_request(self, audio, language):
                return {"input_ids": np.zeros((1, 5), dtype=np.int64)}

        def dummy_nbest(model, processor, inputs, requested_n, max_new_tokens, language):
            return [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": "gorusmelere baslanmistir",
                    "token_ids": [601],
                    "acoustic_log_score": -4.5,
                    "acoustic_score_kind": DEFAULT_ACOUSTIC_KIND,
                    "decoder_sequence_score": -0.45,
                }
            ], {}

        adapter = QwenBaselineAdapter(
            self.c1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: MockProcessor(),
            nbest_fn=dummy_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        u_ids = ["qwen-val-1", "qwen-val-2"]
        batch = [
            self.AudioItem(utterance_id=uid, samples=np.ones(1600, dtype=np.float32) * 0.02)
            for uid in u_ids
        ]
        results = adapter.transcribe(batch)

        ctx = self.RunContext(
            run_id="run-qwen-valid",
            system_id="baseline-qwen3-asr-1.7b",
            dataset_release="release-2026-v1",
            split="test",
            split_manifest_sha256="3" * 64,
            model_artifact_sha256=adapter.model_artifact_sha256,
            decoder_config_sha256=adapter.decoder_config_sha256,
        )

        prediction_rows = []
        for hyp in results:
            inp = self.InferenceInput(
                schema_version="0.1",
                dataset_release="release-2026-v1",
                split_manifest_sha256="3" * 64,
                utterance_id=hyp.utterance_id,
                audio_uri=f"file:///{hyp.utterance_id}.wav",
                audio_sha256="4" * 64,
                duration_s=1.0,
                sample_rate_hz=16000,
                channels=1,
                language="tr",
            )
            row = self.prediction_row(hyp, inp, ctx, timing_record_id=f"t-{hyp.utterance_id}", nbest_artifact_sha256=None)
            prediction_rows.append(row)

        report = self.validate_predictions(prediction_rows, expected_ids=u_ids)
        self.assertTrue(report.ok, f"Validation failed with errors: {report.errors}")
        self.assertEqual(len(report.errors), 0)


if __name__ == "__main__":
    unittest.main()
