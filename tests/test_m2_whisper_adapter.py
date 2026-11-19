"""Unit and integration tests for Member 2 WhisperBaselineAdapter.

Tests cover:
1. Deterministic configuration parsing and strict validation for B1 (whisper-small)
   and B2 (whisper-large-v3).
2. Decoding parameter validation and support for both requested_n=10 (M2) and requested_n=5 (M4).
3. Opt-in integration tests with the shared decoding contract from integration revision b6afa13:
   - check_adapter protocol conformance.
   - Reference-free audio-only transcription preserving batch order and utterance IDs.
   - Empty audio array and decoding error handling without dropping IDs.
   - Verbatim raw text preservation (including emitted leading whitespace).
   - Raw candidate ordering and deduplication delegation to shared export.
   - Score separation (teacher-forced acoustic_log_score vs decoder_sequence_score).
   - Export pipeline integration with prediction_row, nbest_row, and validation_report.
   - Synthetic end-to-end execution with randomly initialized Whisper model and genuine whisper_nbest.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from turkish_asr_rdl.baselines.whisper import (
    PINNED_WHISPER_LFS_OIDS,
    PINNED_WHISPER_REVISIONS,
    WhisperBaselineAdapter,
    connect_shared_decoding,
    get_shared_interface,
)


class WhisperBaselineAdapterConfigTests(unittest.TestCase):
    """Deterministic offline tests for configuration parsing, binding, and validation."""

    def setUp(self) -> None:
        self.b1_path = REPO_ROOT / "configs" / "baselines" / "whisper_small.json"
        self.b2_path = REPO_ROOT / "configs" / "baselines" / "whisper_large_v3.json"
        with self.b1_path.open("r", encoding="utf-8") as f:
            self.b1_config = json.load(f)
        with self.b2_path.open("r", encoding="utf-8") as f:
            self.b2_config = json.load(f)

    def test_b1_whisper_small_config_binding(self):
        """B1 Whisper-small config loads with exact pinned revisions and 16 kHz sampling."""
        adapter = WhisperBaselineAdapter.from_config_file(self.b1_path)
        self.assertEqual(adapter.system_id, "baseline-whisper-small")
        self.assertEqual(adapter.model_id, "openai/whisper-small")
        self.assertEqual(
            adapter.model_revision,
            "973afd24965f72e36ca33b3055d56a652f456b4d",
        )
        self.assertEqual(adapter.processor_id, "openai/whisper-small")
        self.assertEqual(
            adapter.processor_revision,
            "973afd24965f72e36ca33b3055d56a652f456b4d",
        )
        self.assertEqual(adapter.sampling_rate_hz, 16000)
        self.assertEqual(adapter.requested_n, 5)
        self.assertEqual(
            adapter.score_convention_id,
            "whisper-original-logits-generated-plus-eos-v1",
        )
        self.assertEqual(
            adapter.model_artifact_sha256,
            "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b",
        )
        self.assertEqual(len(adapter.decoder_config_sha256), 64)
        self.assertIsInstance(adapter.config, dict)

    def test_b2_whisper_large_v3_config_binding(self):
        """B2 Whisper-large-v3 config loads with exact pinned revisions and 16 kHz sampling."""
        adapter = WhisperBaselineAdapter.from_config_file(self.b2_path)
        self.assertEqual(adapter.system_id, "baseline-whisper-large-v3")
        self.assertEqual(adapter.model_id, "openai/whisper-large-v3")
        self.assertEqual(
            adapter.model_revision,
            "06f233fe06e710322aca913c1bc4249a0d71fce1",
        )
        self.assertEqual(adapter.processor_id, "openai/whisper-large-v3")
        self.assertEqual(
            adapter.processor_revision,
            "06f233fe06e710322aca913c1bc4249a0d71fce1",
        )
        self.assertEqual(adapter.sampling_rate_hz, 16000)
        self.assertEqual(adapter.requested_n, 5)
        self.assertEqual(
            adapter.score_convention_id,
            "whisper-original-logits-generated-plus-eos-v1",
        )
        # B2 config has verified Hub LFS OID
        self.assertEqual(
            adapter.model_artifact_sha256,
            "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
        )
        self.assertEqual(len(adapter.model_artifact_sha256), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in adapter.model_artifact_sha256))
        self.assertEqual(len(adapter.decoder_config_sha256), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in adapter.decoder_config_sha256))

    def test_unsupported_decoding_overrides_raise(self):
        """Adapter enforces documented baseline decoding options and rejects unsupported overrides."""
        overrides = [
            ({"do_sample": True}, "do_sample=False"),
            ({"temperature": 0.5}, "temperature is not supported"),
            ({"task": "translate"}, "task='transcribe'"),
            ({"language": "en"}, "language='tr'"),
            ({"length_penalty": 1.5}, "length_penalty=1.0"),
            ({"early_stopping": True}, "early_stopping=True is not supported"),
            ({"repetition_penalty": 1.2}, "repetition_penalty != 1.0 is not supported"),
            ({"no_repeat_ngram_size": 3}, "no_repeat_ngram_size != 0 is not supported"),
            ({"algorithm": "greedy"}, "algorithm='beam'"),
        ]
        for dec_override, expected_msg in overrides:
            with self.subTest(override=dec_override):
                bad_cfg = copy.deepcopy(self.b1_config)
                bad_cfg["decoding"].update(dec_override)
                with self.assertRaises(ValueError) as ctx:
                    WhisperBaselineAdapter(bad_cfg)
                self.assertIn(expected_msg, str(ctx.exception))

    def test_model_artifact_sha256_validation(self):
        """model_artifact_sha256 requires 64-char lowercase hex; rejects placeholders and hashes of model name."""
        bad_cfg = copy.deepcopy(self.b1_config)
        bad_cfg["model"]["weight_sha256"] = None
        with self.assertRaises(ValueError) as ctx:
            WhisperBaselineAdapter(bad_cfg)
        self.assertIn("model_artifact_sha256", str(ctx.exception))

        # Explicit fixture parameter matching pinned LFS OID works cleanly
        pinned_sha = "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b"
        adapter = WhisperBaselineAdapter(bad_cfg, model_artifact_sha256=pinned_sha)
        self.assertEqual(adapter.model_artifact_sha256, pinned_sha)

        # Mismatched LFS OID raises ValueError
        with self.assertRaises(ValueError) as ctx:
            WhisperBaselineAdapter(bad_cfg, model_artifact_sha256="0" * 64)
        self.assertIn("does not match pinned LFS OID", str(ctx.exception))

    def test_configurable_requested_n_for_m2_and_m4(self):
        """Adapter supports requested_n=10 (M2 default) and requested_n=5 (M4 contract)."""
        adapter_m2 = WhisperBaselineAdapter(self.b1_config, requested_n=10)
        self.assertEqual(adapter_m2.requested_n, 10)
        self.assertEqual(adapter_m2.config["decoding"]["requested_n"], 10)
        self.assertEqual(adapter_m2.config["decoding"]["num_beams"], 10)

        adapter_m4 = WhisperBaselineAdapter(self.b1_config, requested_n=5)
        self.assertEqual(adapter_m4.requested_n, 5)
        self.assertEqual(adapter_m4.config["decoding"]["requested_n"], 5)
        self.assertEqual(adapter_m4.config["decoding"]["num_beams"], 5)

    def test_invalid_requested_n_raises(self):
        """Negative, zero, boolean, or non-integer requested_n values raise ValueError."""
        for invalid_n in (0, -1, -5, True, False, "5", 5.0):
            with self.subTest(invalid_n=invalid_n):
                with self.assertRaises(ValueError):
                    WhisperBaselineAdapter(self.b1_config, requested_n=invalid_n)

    def test_beam_width_mismatch_in_config_raises(self):
        """Explicitly mismatched num_beams and requested_n in decoding config raises ValueError."""
        bad_cfg = copy.deepcopy(self.b1_config)
        bad_cfg["decoding"]["num_beams"] = 8
        bad_cfg["decoding"]["requested_n"] = 5
        with self.assertRaises(ValueError) as ctx:
            WhisperBaselineAdapter(bad_cfg)
        self.assertIn("whisper_nbest helper at revision b6afa13 ties beam width", str(ctx.exception))

    def test_unpinned_model_revision_raises(self):
        """Altering pinned model revision raises ValueError."""
        bad_cfg = copy.deepcopy(self.b1_config)
        bad_cfg["model"]["revision"] = "0000000000000000000000000000000000000000"
        bad_cfg["processor"]["revision"] = "0000000000000000000000000000000000000000"
        with self.assertRaises(ValueError) as ctx:
            WhisperBaselineAdapter(bad_cfg)
        self.assertIn("Model revision mismatch", str(ctx.exception))

    def test_model_and_processor_id_mismatch_raises(self):
        """Mismatched processor ID raises ValueError."""
        bad_cfg = copy.deepcopy(self.b1_config)
        bad_cfg["processor"]["id"] = "openai/whisper-base"
        with self.assertRaises(ValueError) as ctx:
            WhisperBaselineAdapter(bad_cfg)
        self.assertIn("Processor ID", str(ctx.exception))

    def test_non_16k_sampling_rate_raises(self):
        """Sampling rate other than 16000 Hz raises ValueError."""
        bad_cfg = copy.deepcopy(self.b1_config)
        bad_cfg["processor"]["sampling_rate_hz"] = 8000
        with self.assertRaises(ValueError) as ctx:
            WhisperBaselineAdapter(bad_cfg)
        self.assertIn("16000 Hz", str(ctx.exception))

    def test_non_mapping_config_raises(self):
        """Passing non-mapping object raises ValueError."""
        with self.assertRaises(ValueError):
            WhisperBaselineAdapter(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_mock_loaders_execution(self):
        """Injected model and processor loaders are invoked without real downloads."""
        dummy_model = object()
        dummy_processor = object()
        adapter = WhisperBaselineAdapter(
            self.b1_config,
            model_loader=lambda: dummy_model,
            processor_loader=lambda: dummy_processor,
        )
        adapter.load()
        self.assertIs(adapter.model, dummy_model)
        self.assertIs(adapter.processor, dummy_processor)

    def test_explicit_checkpoint_loading_and_hub_isolation(self):
        """Valid explicit local checkpoint makes both model and processor loaders receive the same local snapshot directory with local_files_only=True and never calls Hub."""
        try:
            import torch
            import transformers
        except ImportError:
            self.skipTest("PyTorch and Transformers are required for loader tests")

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td).resolve()
            (tdp / "config.json").write_text("{}")
            (tdp / "model.safetensors").write_bytes(b"dummy")

            adapter = WhisperBaselineAdapter(self.b1_config, checkpoint_path=tdp, device="cpu")
            pinned_oid = PINNED_WHISPER_LFS_OIDS[adapter.model_id]

            with (
                patch("hashlib.sha256") as mock_sha,
                patch("transformers.WhisperProcessor.from_pretrained") as mock_proc,
                patch("transformers.WhisperForConditionalGeneration.from_pretrained") as mock_model,
                patch("huggingface_hub.hf_hub_download") as mock_hub,
            ):
                mock_sha.return_value.hexdigest.return_value = pinned_oid
                mock_inst = MagicMock()
                from transformers import WhisperForConditionalGeneration
                mock_inst.__class__ = WhisperForConditionalGeneration
                mock_model.return_value = mock_inst
                mock_inst.to.return_value.eval.return_value = mock_inst

                adapter.load()

                mock_proc.assert_called_once_with(str(tdp), local_files_only=True)
                mock_model.assert_called_once()
                model_dir, model_kwargs = mock_model.call_args
                self.assertEqual(model_dir[0], str(tdp))
                self.assertTrue(model_kwargs.get("local_files_only"))
                mock_hub.assert_not_called()

    def test_explicit_checkpoint_hash_mismatch_fails_before_loaders(self):
        """Weight hash mismatch fails before model or processor loaders run and never calls Hub."""
        try:
            import torch
            import transformers
        except ImportError:
            self.skipTest("PyTorch and Transformers are required for loader tests")

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td).resolve()
            (tdp / "config.json").write_text("{}")
            (tdp / "model.safetensors").write_bytes(b"dummy")

            adapter = WhisperBaselineAdapter(self.b1_config, checkpoint_path=tdp, device="cpu")

            with (
                patch("hashlib.sha256") as mock_sha,
                patch("transformers.WhisperProcessor.from_pretrained") as mock_proc,
                patch("transformers.WhisperForConditionalGeneration.from_pretrained") as mock_model,
                patch("huggingface_hub.hf_hub_download") as mock_hub,
            ):
                mock_sha.return_value.hexdigest.return_value = "0" * 64
                with self.assertRaises(ValueError) as ctx:
                    adapter.load()
                self.assertIn("Weight SHA-256 mismatch", str(ctx.exception))
                mock_proc.assert_not_called()
                mock_model.assert_not_called()
                mock_hub.assert_not_called()

    def test_explicit_checkpoint_missing_config_fails_without_hub(self):
        """Explicit checkpoint missing config.json fails before loaders and never calls Hub."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td).resolve()
            (tdp / "model.safetensors").write_bytes(b"dummy")
            adapter = WhisperBaselineAdapter(self.b1_config, checkpoint_path=tdp, device="cpu")
            mock_hub = MagicMock()
            mock_hub_mod = MagicMock(hf_hub_download=mock_hub)
            mock_tf_mod = MagicMock()
            mock_torch_mod = MagicMock()
            with patch.dict(
                sys.modules,
                {
                    "huggingface_hub": mock_hub_mod,
                    "transformers": mock_tf_mod,
                    "torch": mock_torch_mod,
                },
            ):
                with self.assertRaises(ValueError) as ctx:
                    adapter.load()
                self.assertIn("missing config.json", str(ctx.exception))
                mock_hub.assert_not_called()
                mock_tf_mod.WhisperProcessor.from_pretrained.assert_not_called()
                mock_tf_mod.WhisperForConditionalGeneration.from_pretrained.assert_not_called()
                self.assertIsNone(adapter.model)
                self.assertIsNone(adapter.processor)


class SharedDecoderIntegrationTests(unittest.TestCase):
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
            from turkish_asr_rdl.decoding.whisper import whisper_nbest
            from turkish_asr_rdl.schemas.predictions import validate_predictions

            cls.AudioItem = AudioItem
            cls.Candidate = Candidate
            cls.Hypothesis = Hypothesis
            cls.InferenceInput = InferenceInput
            cls.check_adapter = staticmethod(check_adapter)
            cls.whisper_nbest = staticmethod(whisper_nbest)
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

        self.b1_path = REPO_ROOT / "configs" / "baselines" / "whisper_small.json"
        self.b2_path = REPO_ROOT / "configs" / "baselines" / "whisper_large_v3.json"
        with self.b1_path.open("r", encoding="utf-8") as f:
            self.b1_config = json.load(f)
        with self.b2_path.open("r", encoding="utf-8") as f:
            self.b2_config = json.load(f)

    def test_check_adapter_protocol_conformance(self):
        """Adapter passes check_adapter protocol verification for B1 and B2."""
        adapter_b1 = WhisperBaselineAdapter(self.b1_config, shared_path=self.shared_path)
        adapter_b2 = WhisperBaselineAdapter(self.b2_config, shared_path=self.shared_path)

        # check_adapter raises ValueError if adapter fails protocol or cannot write strict JSON
        self.check_adapter(adapter_b1)
        self.check_adapter(adapter_b2)

    def test_reference_free_batch_order_and_id_preservation(self):
        """Transcribe batch of AudioItems reference-free, preserving exact order and IDs."""
        u_ids = ["utt-alpha", "utt-beta", "utt-gamma"]

        class DummyFeatureExtractor:
            def __call__(self, samples, sampling_rate, return_tensors):
                class DummyTensors:
                    input_features = np.zeros((1, 80, 100), dtype=np.float32)
                return DummyTensors()

        class DummyProcessor:
            feature_extractor = DummyFeatureExtractor()

        def dummy_nbest(model, processor, features, requested_n, max_new_tokens, deduplicate):
            return [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": " Meclis tutanagi",
                    "token_ids": [501, 502],
                    "acoustic_log_score": -8.5,
                    "acoustic_score_kind": "seq2seq_token_log_prob_sum",
                    "decoder_sequence_score": -0.85,
                    "score_details": {},
                }
            ], {}

        adapter = WhisperBaselineAdapter(
            self.b1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: DummyProcessor(),
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
            self.assertEqual(hyp.text, " Meclis tutanagi")
            self.assertEqual(hyp.selected_candidate_id, "beam-0001")
            self.assertEqual(len(hyp.candidates), 1)

    def test_empty_audio_array_and_decode_error_without_dropping_ids(self):
        """Empty audio array returns status='empty' and errors return status='error' without dropping IDs."""
        class DummyFeatureExtractor:
            def __call__(self, samples, sampling_rate, return_tensors):
                class DummyTensors:
                    input_features = np.zeros((1, 80, 100), dtype=np.float32)
                return DummyTensors()

        class DummyProcessor:
            feature_extractor = DummyFeatureExtractor()

        call_count = 0

        def failing_nbest(model, processor, features, requested_n, max_new_tokens, deduplicate):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    {
                        "candidate_id": "beam-0001",
                        "rank": 1,
                        "text": " gecerli audio",
                        "token_ids": [101],
                        "acoustic_log_score": -5.0,
                        "acoustic_score_kind": "seq2seq_token_log_prob_sum",
                        "decoder_sequence_score": None,
                    }
                ], {}
            raise RuntimeError("Synthetic decoder failure")

        adapter = WhisperBaselineAdapter(
            self.b1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: DummyProcessor(),
            nbest_fn=failing_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        batch = [
            # 1. Normal valid audio
            self.AudioItem(utterance_id="u1", samples=np.ones(1600, dtype=np.float32) * 0.05),
            # 2. Empty audio array
            self.AudioItem(utterance_id="u2", samples=np.array([], dtype=np.float32)),
            # 3. Invalid audio (non-finite float)
            self.AudioItem(utterance_id="u3", samples=np.array([0.1, np.nan, 0.2], dtype=np.float32)),
            # 4. Decoder failure
            self.AudioItem(utterance_id="u4", samples=np.ones(1600, dtype=np.float32) * 0.05),
        ]

        hypotheses = adapter.transcribe(batch)
        self.assertEqual(len(hypotheses), 4)

        # u1: ok
        self.assertEqual(hypotheses[0].utterance_id, "u1")
        self.assertEqual(hypotheses[0].status, "ok")
        self.assertEqual(hypotheses[0].text, " gecerli audio")

        # u2: empty
        self.assertEqual(hypotheses[1].utterance_id, "u2")
        self.assertEqual(hypotheses[1].status, "empty")
        self.assertEqual(hypotheses[1].text, "")
        self.assertEqual(hypotheses[1].candidates, ())

        # u3: error (InvalidAudio)
        self.assertEqual(hypotheses[2].utterance_id, "u3")
        self.assertEqual(hypotheses[2].status, "error")
        self.assertIsNone(hypotheses[2].text)
        self.assertEqual(hypotheses[2].error_code, "InvalidAudio")
        self.assertIn("1D numpy array of finite floats", hypotheses[2].error_message)

        # u4: error (RuntimeError)
        self.assertEqual(hypotheses[3].utterance_id, "u4")
        self.assertEqual(hypotheses[3].status, "error")
        self.assertIsNone(hypotheses[3].text)
        self.assertEqual(hypotheses[3].error_code, "RuntimeError")
        self.assertIn("Synthetic decoder failure", hypotheses[3].error_message)

    def test_audio_longer_than_30s_returns_audiotoolong(self):
        """Audio clips exceeding 30 s (480,000 samples at 16 kHz) return status='error' with AudioTooLong."""
        adapter = WhisperBaselineAdapter(
            self.b1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: object(),
            shared_path=self.shared_path,
        )
        adapter.load()
        dummy_item = type("DummyItem", (), {"utterance_id": "long-01", "samples": np.ones(480001, dtype=np.float32)})()
        results = adapter.transcribe([dummy_item])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "error")
        self.assertEqual(results[0].error_code, "AudioTooLong")
        self.assertIn("480,000 samples", results[0].error_message)
        self.assertIn("data/metadata/worldspeech-tr_tr-source-audit.md:52", results[0].error_message)

    def test_verbatim_raw_whitespace_preservation(self):
        """Decoded text is preserved verbatim including emitted leading whitespace without stripping."""
        class DummyFeatureExtractor:
            def __call__(self, samples, sampling_rate, return_tensors):
                class DummyTensors:
                    input_features = np.zeros((1, 80, 100), dtype=np.float32)
                return DummyTensors()

        class DummyProcessor:
            feature_extractor = DummyFeatureExtractor()

        raw_text_with_spaces = "   Merhaba Sayin Baskan  "

        def dummy_nbest(model, processor, features, requested_n, max_new_tokens, deduplicate):
            return [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": raw_text_with_spaces,
                    "token_ids": [301, 302],
                    "acoustic_log_score": -4.2,
                    "acoustic_score_kind": "seq2seq_token_log_prob_sum",
                    "decoder_sequence_score": -0.42,
                }
            ], {}

        adapter = WhisperBaselineAdapter(
            self.b1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: DummyProcessor(),
            nbest_fn=dummy_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        batch = [self.AudioItem(utterance_id="u_ws", samples=np.ones(1600, dtype=np.float32))]
        results = adapter.transcribe(batch)

        self.assertEqual(results[0].text, raw_text_with_spaces)
        self.assertEqual(results[0].candidates[0].text, raw_text_with_spaces)
        self.assertTrue(results[0].text.startswith("   "))
        self.assertTrue(results[0].text.endswith("  "))

    def test_raw_candidate_order_and_deduplication_ownership(self):
        """Adapter returns unpadded raw candidates; shared export owns deduplication."""
        class DummyFeatureExtractor:
            def __call__(self, samples, sampling_rate, return_tensors):
                class DummyTensors:
                    input_features = np.zeros((1, 80, 100), dtype=np.float32)
                return DummyTensors()

        class DummyProcessor:
            feature_extractor = DummyFeatureExtractor()

        # Two candidates with duplicate text at ranks 1 and 2
        def duplicate_cands_nbest(model, processor, features, requested_n, max_new_tokens, deduplicate):
            return [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": " Turkiye",
                    "token_ids": [101],
                    "acoustic_log_score": -2.0,
                    "acoustic_score_kind": "seq2seq_token_log_prob_sum",
                    "decoder_sequence_score": -0.2,
                },
                {
                    "candidate_id": "beam-0002",
                    "rank": 2,
                    "text": " Turkiye",  # Duplicate text
                    "token_ids": [102],
                    "acoustic_log_score": -3.5,
                    "acoustic_score_kind": "seq2seq_token_log_prob_sum",
                    "decoder_sequence_score": -0.35,
                },
                {
                    "candidate_id": "beam-0003",
                    "rank": 3,
                    "text": " Meclisi",
                    "token_ids": [103],
                    "acoustic_log_score": -5.0,
                    "acoustic_score_kind": "seq2seq_token_log_prob_sum",
                    "decoder_sequence_score": -0.5,
                },
            ], {}

        adapter = WhisperBaselineAdapter(
            self.b1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: DummyProcessor(),
            nbest_fn=duplicate_cands_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        batch = [self.AudioItem(utterance_id="u_dup", samples=np.ones(1600, dtype=np.float32))]
        results = adapter.transcribe(batch)
        hyp = results[0]

        # The adapter itself must NOT deduplicate or drop candidates
        self.assertEqual(len(hyp.candidates), 3)
        self.assertEqual(hyp.candidates[0].candidate_id, "beam-0001")
        self.assertEqual(hyp.candidates[1].candidate_id, "beam-0002")
        self.assertEqual(hyp.candidates[2].candidate_id, "beam-0003")

        # Now pass to shared exporter: export owns deduplication
        ctx = self.RunContext(
            run_id="run-test",
            system_id="baseline-whisper-small",
            dataset_release="release-2026-v1",
            split="test",
            split_manifest_sha256="a" * 64,
            model_artifact_sha256=adapter.model_artifact_sha256,
            decoder_config_sha256=adapter.decoder_config_sha256,
        )
        inp = self.InferenceInput(
            schema_version="0.1",
            dataset_release="release-2026-v1",
            split_manifest_sha256="a" * 64,
            utterance_id="u_dup",
            audio_uri="file:///audio.wav",
            audio_sha256="b" * 64,
            duration_s=1.0,
            sample_rate_hz=16000,
            channels=1,
            language="tr",
        )

        nb_row = self.nbest_row(hyp, inp, ctx, requested_n=10, score_convention_id=adapter.score_convention_id)
        # Shared deduplication kept 2 candidates and mapped beam-0002 -> beam-0001
        self.assertEqual(len(nb_row["candidates"]), 2)
        self.assertEqual(nb_row["dedup_map"], {"beam-0002": "beam-0001"})

    def test_score_separation_and_none_preservation(self):
        """Teacher-forced acoustic_log_score is separated from decoder_sequence_score (missing is None)."""
        class DummyFeatureExtractor:
            def __call__(self, samples, sampling_rate, return_tensors):
                class DummyTensors:
                    input_features = np.zeros((1, 80, 100), dtype=np.float32)
                return DummyTensors()

        class DummyProcessor:
            feature_extractor = DummyFeatureExtractor()

        def missing_beam_score_nbest(model, processor, features, requested_n, max_new_tokens, deduplicate):
            return [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": " ornek metin",
                    "token_ids": [201],
                    "acoustic_log_score": -11.25,
                    "acoustic_score_kind": "seq2seq_token_log_prob_sum",
                    "decoder_sequence_score": None,  # Missing framework beam score
                }
            ], {}

        adapter = WhisperBaselineAdapter(
            self.b1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: DummyProcessor(),
            nbest_fn=missing_beam_score_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        batch = [self.AudioItem(utterance_id="u_score", samples=np.ones(1600, dtype=np.float32))]
        results = adapter.transcribe(batch)
        cand = results[0].candidates[0]

        self.assertEqual(cand.acoustic_log_score, -11.25)
        self.assertEqual(cand.acoustic_score_kind, "seq2seq_token_log_prob_sum")
        self.assertIsNone(cand.decoder_sequence_score)

        # Check export row formatting
        ctx = self.RunContext(
            run_id="run-score",
            system_id="baseline-whisper-small",
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
            utterance_id="u_score",
            audio_uri="file:///audio.wav",
            audio_sha256="d" * 64,
            duration_s=1.0,
            sample_rate_hz=16000,
            channels=1,
            language="tr",
        )
        pred = self.prediction_row(results[0], inp, ctx, timing_record_id="t-1", nbest_artifact_sha256="e" * 64)
        nb = self.nbest_row(results[0], inp, ctx, requested_n=10, score_convention_id=adapter.score_convention_id)

        self.assertEqual(pred["status"], "ok")
        self.assertIsNone(nb["candidates"][0]["decoder_sequence_score"])
        self.assertEqual(
            nb["candidates"][0]["score_details"]["null_reasons"]["decoder_sequence_score"],
            "not produced by this system",
        )

    def test_full_predictions_schema_validation(self):
        """Adapter output hypotheses produce prediction rows that pass validate_predictions."""
        class DummyFeatureExtractor:
            def __call__(self, samples, sampling_rate, return_tensors):
                class DummyTensors:
                    input_features = np.zeros((1, 80, 100), dtype=np.float32)
                return DummyTensors()

        class DummyProcessor:
            feature_extractor = DummyFeatureExtractor()

        def dummy_nbest(model, processor, features, requested_n, max_new_tokens, deduplicate):
            return [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": " kanun teklifi",
                    "token_ids": [401],
                    "acoustic_log_score": -6.5,
                    "acoustic_score_kind": "seq2seq_token_log_prob_sum",
                    "decoder_sequence_score": -0.65,
                }
            ], {}

        adapter = WhisperBaselineAdapter(
            self.b1_config,
            model_loader=lambda: object(),
            processor_loader=lambda: DummyProcessor(),
            nbest_fn=dummy_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        u_ids = ["utt-01", "utt-02"]
        batch = [
            self.AudioItem(utterance_id=uid, samples=np.ones(1600, dtype=np.float32) * 0.02)
            for uid in u_ids
        ]
        results = adapter.transcribe(batch)

        ctx = self.RunContext(
            run_id="run-valid",
            system_id="baseline-whisper-small",
            dataset_release="release-2026-v1",
            split="test",
            split_manifest_sha256="1" * 64,
            model_artifact_sha256=adapter.model_artifact_sha256,
            decoder_config_sha256=adapter.decoder_config_sha256,
        )

        prediction_rows = []
        for hyp in results:
            inp = self.InferenceInput(
                schema_version="0.1",
                dataset_release="release-2026-v1",
                split_manifest_sha256="1" * 64,
                utterance_id=hyp.utterance_id,
                audio_uri=f"file:///{hyp.utterance_id}.wav",
                audio_sha256="2" * 64,
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

    def test_synthetic_end_to_end_with_whisper_nbest(self):
        """Synthetic end-to-end test with tiny randomly initialized Whisper model and whisper_nbest."""
        try:
            import torch
            from transformers import WhisperConfig, WhisperForConditionalGeneration
        except ImportError:
            self.skipTest("PyTorch and Transformers are required for synthetic model integration test")

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

        class LocalFeatureExtractor:
            def __init__(self):
                self.saved_features = torch.randn(1, 80, 3000)

            def __call__(self, samples, sampling_rate, return_tensors):
                class Tensors:
                    input_features = self.saved_features
                return Tensors()

        class LocalProcessor:
            def __init__(self):
                self.tokenizer = LocalTokenizer()
                self.feature_extractor = LocalFeatureExtractor()

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

        adapter = WhisperBaselineAdapter(
            self.b1_config,
            device="cpu",
            precision="float32",
            requested_n=2,
            model_loader=lambda: model,
            processor_loader=lambda: processor,
            nbest_fn=self.whisper_nbest,
            shared_path=self.shared_path,
        )
        adapter.load()

        batch = [
            self.AudioItem(utterance_id="synth-01", samples=np.zeros(16000, dtype=np.float32)),
        ]

        hypotheses = adapter.transcribe(batch)
        self.assertEqual(len(hypotheses), 1)
        hyp = hypotheses[0]
        self.assertEqual(hyp.utterance_id, "synth-01")
        self.assertIn(hyp.status, ("ok", "empty"))
        self.assertEqual(len(hyp.candidates), 2)
        self.assertEqual(hyp.candidates[0].rank, 1)
        self.assertEqual(hyp.candidates[1].rank, 2)

        prompt_tokens = [50258, 50359, 50363, 50365]
        for c in hyp.candidates:
            self.assertTrue(np.isfinite(c.acoustic_log_score))
            self.assertEqual(c.acoustic_score_kind, "seq2seq_token_log_prob_sum")
            # Independent teacher-forced score recompute
            full_tokens = torch.tensor([prompt_tokens + list(c.token_ids)])
            with torch.no_grad():
                logits = model(
                    input_features=processor.feature_extractor.saved_features,
                    decoder_input_ids=full_tokens,
                ).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            p_len = len(prompt_tokens)
            tf_scores = [log_probs[0, p_len - 1 + i, tok].item() for i, tok in enumerate(c.token_ids)]
            tf_sum = sum(tf_scores)
            diff = abs(c.acoustic_log_score - tf_sum)
            self.assertLess(diff, 1e-3, f"Whisper acoustic_log_score diff {diff} exceeds 1e-3")
            if c.decoder_sequence_score is not None:
                self.assertNotEqual(c.decoder_sequence_score, c.acoustic_log_score)


if __name__ == "__main__":
    unittest.main()
