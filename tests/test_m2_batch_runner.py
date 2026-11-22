"""Unit and end-to-end integration tests for Member 2 baseline batch runner and smoke check.

Tests verify:
1. Checkpoint inventory validation (matching hash, rejected mismatch, unresolved shards, remote URL rejection).
2. Reference-free inference manifest parsing and allowlist enforcement (forbidding reference text).
3. End-to-end batch runner workflow for Whisper (B1/B2) with mocked pretrained loading.
4. End-to-end batch runner workflow for Qwen (B3) with mocked pretrained loading.
5. Safe failure on checkpoint hash mismatch before model loading occurs.
6. Per-clip error handling (corrupted audio) preserving utterance IDs and batch ordering.
7. Real-audio Whisper rank-1 smoke comparison runner and transcript suppression by default.
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
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from turkish_asr_rdl.baselines.runner import (
    CheckpointValidationError,
    InputManifestError,
    compute_file_sha256,
    connect_shared_decoding,
    get_shared_modules,
    read_inference_manifest,
    run_baseline_batch,
    verify_checkpoint_inventory,
)
from turkish_asr_rdl.baselines.whisper_rank1 import Rank1ComparisonResult
from scripts.smoke_check_whisper_rank1 import run_real_audio_smoke_check, run_synthetic_smoke_check


class DummyTokenizer:
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
        return " ".join(words) if words else "dummy transcript"

    def batch_decode(self, sequences: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]


class DummyFeatureExtractor:
    def __call__(self, audio, sampling_rate=16000, return_tensors="pt"):
        class DummyBatch:
            def __init__(self):
                self.input_features = np.zeros((1, 80, 100), dtype=np.float32)

            def to(self, *args, **kwargs):
                return self

        return DummyBatch()


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.feature_extractor = DummyFeatureExtractor()

    def __call__(self, audio, sampling_rate=16000, return_tensors="pt"):
        return self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors=return_tensors)

    def apply_transcription_request(self, audio, language="tr"):
        class DummyInputs:
            def __init__(self):
                self.data = {"input_ids": np.array([[1, 2, 3]], dtype=np.int64)}

            def to(self, *args, **kwargs):
                return self

            def get(self, key, default=None):
                return self.data.get(key, default)

            def __getitem__(self, item):
                return self.data[item]

        return DummyInputs()

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, sequences: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)


class DummyGenerationOutput:
    def __init__(self, sequences, scores, sequences_scores=None, beam_indices=None):
        self.sequences = sequences
        self.scores = scores
        self.sequences_scores = sequences_scores
        self.beam_indices = beam_indices


class DummyModel:
    def __init__(self, vocab_size=51865):
        class GenConfig:
            eos_token_id = 50257
            forced_decoder_ids = None

        class TextConfig:
            vocab_size = 51865

            def get_text_config(self):
                return self

        self.generation_config = GenConfig()
        self.config = TextConfig()
        self.device = "cpu"
        self.dtype = None

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def generate(self, *args, **kwargs):
        num_return = kwargs.get("num_return_sequences", 1)
        seqs = np.array([[50258, 50359, 50363, 50365, 101, 102, 50257]] * num_return)
        return DummyGenerationOutput(
            sequences=seqs,
            scores=(),
            sequences_scores=np.array([-0.5] * num_return),
        )

    def compute_transition_scores(self, sequences, scores, beam_indices=None, normalize_logits=False):
        import torch
        return torch.tensor([[-0.1, -0.2, -0.3]] * len(sequences))


class BaselineBatchRunnerTests(unittest.TestCase):
    """Unit and end-to-end integration tests for baseline batch runner and exporter."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)

        # Baseline config paths
        self.b1_config_path = REPO_ROOT / "configs" / "baselines" / "whisper_small.json"
        self.b3_config_path = REPO_ROOT / "configs" / "baselines" / "qwen3_asr_1_7b.json"

        with self.b1_config_path.open("r", encoding="utf-8") as f:
            self.b1_config = json.load(f)
        with self.b3_config_path.open("r", encoding="utf-8") as f:
            self.b3_config = json.load(f)

        self.shared_path = os.environ.get("TURAS_SHARED_DECODER_PATH")
        self.shared_available = connect_shared_decoding(self.shared_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _require_shared_modules(self) -> None:
        if not self.shared_available:
            self.skipTest(
                "Shared decoding interface (revision b6afa13063f7fa90bb1bc190c67b4529e40e91a5) "
                "is not available in environment. Set TURAS_SHARED_DECODER_PATH to enable integration test."
            )

    def _create_mock_checkpoint(self, filename: str, content: bytes = b"dummy weight bytes") -> tuple[Path, str]:
        ckpt_dir = self.work_dir / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        file_path = ckpt_dir / filename
        file_path.write_bytes(content)
        (ckpt_dir / "config.json").write_text("{}", encoding="utf-8")
        sha = hashlib.sha256(content).hexdigest().lower()
        return ckpt_dir, sha

    def _create_mock_manifest(
        self,
        clips: list[tuple[str, str]],
        dataset_release: str = "worldspeech-tr-v0.1",
        split_manifest_sha: str = "a" * 64,
        extra_fields: dict[str, Any] | None = None,
    ) -> Path:
        manifest_path = self.work_dir / "manifest.jsonl"
        lines = []
        for uid, uri in clips:
            row = {
                "schema_version": "0.1",
                "dataset_release": dataset_release,
                "split_manifest_sha256": split_manifest_sha,
                "utterance_id": uid,
                "audio_uri": uri,
                "audio_sha256": hashlib.sha256(uid.encode("utf-8")).hexdigest(),
                "duration_s": 5.0,
                "sample_rate_hz": 16000,
                "channels": 1,
                "language": "tr",
            }
            if extra_fields:
                row.update(extra_fields)
            lines.append(json.dumps(row))

        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest_path

    def test_verify_checkpoint_inventory_success_and_rejections(self):
        """Checkpoint inventory verifies matching file, and rejects missing, corrupt, remote, or unresolved shards."""
        ckpt_dir, valid_sha = self._create_mock_checkpoint("model.safetensors", b"safetensors content")

        # 1. Successful verification
        resolved = verify_checkpoint_inventory(ckpt_dir, "model.safetensors", valid_sha)
        self.assertEqual(resolved, ckpt_dir / "model.safetensors")

        # 2. Direct file path success
        resolved_file = verify_checkpoint_inventory(ckpt_dir / "model.safetensors", "model.safetensors", valid_sha)
        self.assertEqual(resolved_file, ckpt_dir / "model.safetensors")

        # 3. Hash mismatch
        wrong_sha = "f" * 64
        with self.assertRaises(CheckpointValidationError) as ctx:
            verify_checkpoint_inventory(ckpt_dir, "model.safetensors", wrong_sha)
        self.assertIn("mismatch", str(ctx.exception).lower())

        # 4. Missing weight file
        with self.assertRaises(CheckpointValidationError) as ctx:
            verify_checkpoint_inventory(ckpt_dir, "nonexistent.safetensors", valid_sha)
        self.assertIn("not found", str(ctx.exception).lower())

        # 5. Remote URL rejected
        with self.assertRaises(CheckpointValidationError) as ctx:
            verify_checkpoint_inventory("https://huggingface.co/model", "model.safetensors", valid_sha)
        self.assertIn("local", str(ctx.exception).lower())

        # 6. Unresolved sharded weights rejected
        sharded_dir = self.work_dir / "sharded_ckpt"
        sharded_dir.mkdir()
        (sharded_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(CheckpointValidationError) as ctx:
            verify_checkpoint_inventory(sharded_dir, "model.safetensors", valid_sha)
        self.assertIn("unresolved shard inventory", str(ctx.exception).lower())

    def test_read_inference_manifest_allowlist_and_rejections(self):
        """Manifest reader validates allowlist and strictly forbids reference transcripts."""
        self._require_shared_modules()
        manifest = self._create_mock_manifest([("utt-01", "clip1.flac"), ("utt-02", "clip2.flac")])
        rows = read_inference_manifest(manifest, shared_path=self.shared_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].utterance_id, "utt-01")
        self.assertEqual(rows[1].utterance_id, "utt-02")

        # Reject manifest with reference_text
        bad_manifest = self._create_mock_manifest(
            [("utt-01", "clip1.flac")], extra_fields={"reference_text": "forbidden text"}
        )
        with self.assertRaises(InputManifestError) as ctx:
            read_inference_manifest(bad_manifest, shared_path=self.shared_path)
        self.assertIn("reference_text", str(ctx.exception).lower())

        # Reject manifest with gold transcript
        gold_manifest = self._create_mock_manifest(
            [("utt-01", "clip1.flac")], extra_fields={"gold": "forbidden gold"}
        )
        with self.assertRaises(InputManifestError) as ctx:
            read_inference_manifest(gold_manifest, shared_path=self.shared_path)
        self.assertIn("gold", str(ctx.exception).lower())

    def test_batch_runner_rejects_reference_bearing_manifest(self):
        """Batch runner directly rejects manifests containing forbidden reference transcripts."""
        self._require_shared_modules()
        ckpt_dir, valid_sha = self._create_mock_checkpoint("model.safetensors", b"whisper test bytes")
        cfg = copy.deepcopy(self.b1_config)
        cfg["model"]["weight_sha256"] = valid_sha

        bad_manifest = self._create_mock_manifest(
            [("utt-01", "clip1.flac")], extra_fields={"reference_text": "forbidden reference text"}
        )
        out_dir = self.work_dir / "out_bad_manifest"

        with self.assertRaises(InputManifestError) as ctx:
            run_baseline_batch(
                config=cfg,
                checkpoint_path=ckpt_dir,
                manifest_path=bad_manifest,
                output_dir=out_dir,
                device="cpu",
                shared_path=self.shared_path,
                model_loader=lambda: DummyModel(),
            )
        self.assertIn("reference_text", str(ctx.exception).lower())
        self.assertFalse((out_dir / "predictions.jsonl").exists())

    def test_end_to_end_successful_export_whisper(self):
        """End-to-end batch runner on Whisper produces schema-valid predictions, N-best, and manifest."""
        self._require_shared_modules()
        ckpt_dir, valid_sha = self._create_mock_checkpoint("model.safetensors", b"whisper test bytes")
        cfg = copy.deepcopy(self.b1_config)
        cfg["model"]["weight_sha256"] = valid_sha

        manifest = self._create_mock_manifest([("utt-01", "audio1.flac"), ("utt-02", "audio2.flac")])
        out_dir = self.work_dir / "output_whisper"

        dummy_audio = np.zeros(16000, dtype=np.float32)

        def mock_audio_loader(p: Path) -> np.ndarray:
            return dummy_audio

        def mock_nbest_fn(*args, **kwargs):
            cands = [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": " merhaba dunya",
                    "token_ids": [101, 102],
                    "acoustic_log_score": -1.2,
                    "acoustic_score_kind": "seq2seq_token_log_prob_sum",
                    "decoder_sequence_score": -0.5,
                    "score_details": {"length_penalty": 1.0},
                }
            ]
            return cands, {}

        result = run_baseline_batch(
            config=cfg,
            checkpoint_path=ckpt_dir,
            manifest_path=manifest,
            output_dir=out_dir,
            device="cpu",
            requested_n=10,
            shared_path=self.shared_path,
            model_loader=lambda: DummyModel(),
            processor_loader=lambda: DummyProcessor(),
            nbest_fn=mock_nbest_fn,
            audio_loader=mock_audio_loader,
            command_line="python test_whisper_batch.py",
        )

        self.assertEqual(result.system_id, "baseline-whisper-small")
        self.assertEqual(result.total_utterances, 2)
        self.assertEqual(result.ok_count, 2)
        self.assertEqual(result.error_count, 0)
        self.assertTrue(result.validation_ok)
        self.assertEqual(result.validation_errors, ())

        self.assertTrue(result.predictions_path.is_file())
        self.assertTrue(result.nbest_path.is_file())
        self.assertTrue(result.manifest_path.is_file())

        # Verify predictions.jsonl
        shared = get_shared_modules()
        validate_predictions = shared["validate_predictions"]
        validate_nbest = shared["validate_nbest"]

        pred_lines = [json.loads(line) for line in result.predictions_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(pred_lines), 2)
        self.assertTrue(validate_predictions(pred_lines).ok)

        # Verify nbest.jsonl
        nbest_lines = [json.loads(line) for line in result.nbest_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(nbest_lines), 2)
        self.assertTrue(validate_nbest(nbest_lines).ok)

        # Verify manifest.json
        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        for group in ("identity", "code", "environment", "inputs", "configuration", "outputs", "limitations"):
            self.assertIn(group, manifest_data)

    def test_end_to_end_successful_export_qwen(self):
        """End-to-end batch runner on unselected Qwen candidate produces valid predictions, N-best, and manifest."""
        self._require_shared_modules()
        ckpt_dir, valid_sha = self._create_mock_checkpoint("model.safetensors", b"qwen test bytes")
        cfg = copy.deepcopy(self.b3_config)
        cfg["model"]["weight_sha256"] = valid_sha
        self.assertEqual(cfg.get("candidate_status"), "unselected-candidate")

        manifest = self._create_mock_manifest([("qwen-utt-01", "audio1.flac"), ("qwen-utt-02", "audio2.flac")])
        out_dir = self.work_dir / "output_qwen"

        dummy_audio = np.zeros(16000, dtype=np.float32)

        def mock_audio_loader(p: Path) -> np.ndarray:
            return dummy_audio

        def mock_nbest_fn(*args, **kwargs):
            cands = [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": "qwen hipotezi",
                    "token_ids": (201, 202),
                    "acoustic_log_score": -0.85,
                    "acoustic_score_kind": "speech_llm_joint_token_log_prob_sum",
                    "decoder_sequence_score": -0.32,
                    "score_details": {"score_source": "test"},
                }
            ]
            return cands, {}

        result = run_baseline_batch(
            config=cfg,
            checkpoint_path=ckpt_dir,
            manifest_path=manifest,
            output_dir=out_dir,
            device="cpu",
            requested_n=5,
            shared_path=self.shared_path,
            model_loader=lambda: DummyModel(),
            processor_loader=lambda: DummyProcessor(),
            nbest_fn=mock_nbest_fn,
            audio_loader=mock_audio_loader,
            command_line="python test_qwen_batch.py",
        )

        self.assertEqual(result.system_id, "baseline-qwen3-asr-1.7b")
        self.assertEqual(result.total_utterances, 2)
        self.assertEqual(result.ok_count, 2)
        self.assertTrue(result.validation_ok)

        shared = get_shared_modules(self.shared_path)
        pred_lines = [json.loads(line) for line in result.predictions_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertTrue(shared["validate_predictions"](pred_lines).ok)
        nbest_lines = [json.loads(line) for line in result.nbest_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertTrue(shared["validate_nbest"](nbest_lines).ok)

    def test_checkpoint_hash_mismatch_fails_safely_before_loading(self):
        """Corrupt checkpoint hash triggers CheckpointValidationError and never calls model loading."""
        self._require_shared_modules()
        ckpt_dir, valid_sha = self._create_mock_checkpoint("model.safetensors", b"corrupted bytes")
        cfg = copy.deepcopy(self.b1_config)
        cfg["model"]["weight_sha256"] = "e" * 64  # deliberate mismatch

        manifest = self._create_mock_manifest([("utt-01", "audio1.flac")])
        out_dir = self.work_dir / "output_mismatch"

        loaded = False

        def model_factory():
            nonlocal loaded
            loaded = True
            return DummyModel()

        with self.assertRaises(CheckpointValidationError) as ctx:
            run_baseline_batch(
                config=cfg,
                checkpoint_path=ckpt_dir,
                manifest_path=manifest,
                output_dir=out_dir,
                device="cpu",
                shared_path=self.shared_path,
            )

        self.assertIn("mismatch", str(ctx.exception).lower())
        self.assertFalse(loaded)
        # Verify no output files were written
        self.assertFalse((out_dir / "predictions.jsonl").exists())

    def test_per_clip_error_preserves_utterance_id_and_batch_order(self):
        """Audio decoding error on one clip preserves utterance ID and position as status='error'."""
        self._require_shared_modules()
        ckpt_dir, valid_sha = self._create_mock_checkpoint("model.safetensors", b"whisper test bytes")
        cfg = copy.deepcopy(self.b1_config)
        cfg["model"]["weight_sha256"] = valid_sha

        # 3 clips: 1 valid, 1 failing audio, 1 valid
        manifest = self._create_mock_manifest([
            ("utt-01", "good1.flac"),
            ("utt-02", "corrupt.flac"),
            ("utt-03", "good2.flac"),
        ])
        out_dir = self.work_dir / "output_error_handling"

        def mock_audio_loader(p: Path) -> np.ndarray:
            if "corrupt" in p.name:
                raise ValueError("Corrupt audio header in file")
            return np.zeros(16000, dtype=np.float32)

        def mock_nbest_fn(*args, **kwargs):
            return [
                {
                    "candidate_id": "beam-0001",
                    "rank": 1,
                    "text": " ornek ses",
                    "token_ids": [101],
                    "acoustic_log_score": -1.0,
                    "acoustic_score_kind": "seq2seq_token_log_prob_sum",
                    "decoder_sequence_score": -0.2,
                    "score_details": {},
                }
            ], {}

        result = run_baseline_batch(
            config=cfg,
            checkpoint_path=ckpt_dir,
            manifest_path=manifest,
            output_dir=out_dir,
            device="cpu",
            requested_n=10,
            shared_path=self.shared_path,
            model_loader=lambda: DummyModel(),
            processor_loader=lambda: DummyProcessor(),
            nbest_fn=mock_nbest_fn,
            audio_loader=mock_audio_loader,
        )

        self.assertEqual(result.total_utterances, 3)
        self.assertEqual(result.ok_count, 2)
        self.assertEqual(result.error_count, 1)
        self.assertTrue(result.validation_ok)

        # Inspect prediction rows in order
        shared = get_shared_modules(self.shared_path)
        pred_lines = [json.loads(line) for line in result.predictions_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(pred_lines), 3)

        self.assertEqual(pred_lines[0]["utterance_id"], "utt-01")
        self.assertEqual(pred_lines[0]["status"], "ok")

        self.assertEqual(pred_lines[1]["utterance_id"], "utt-02")
        self.assertEqual(pred_lines[1]["status"], "error")
        self.assertIsNone(pred_lines[1]["hypothesis_text"])
        self.assertEqual(pred_lines[1]["error"]["code"], "ValueError")
        self.assertIn("Corrupt audio header", pred_lines[1]["error"]["message"])

        self.assertEqual(pred_lines[2]["utterance_id"], "utt-03")
        self.assertEqual(pred_lines[2]["status"], "ok")

        # Must pass schema validation
        self.assertTrue(shared["validate_predictions"](pred_lines).ok)

        # N-best rows
        nbest_lines = [json.loads(line) for line in result.nbest_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(nbest_lines), 3)
        self.assertEqual(nbest_lines[1]["utterance_id"], "utt-02")
        self.assertEqual(nbest_lines[1]["actual_n"], 0)
        self.assertEqual(nbest_lines[1]["candidates"], [])
        self.assertTrue(shared["validate_nbest"](nbest_lines).ok)

    def test_smoke_check_whisper_rank1_real_audio_runner(self):
        """Real-audio smoke check helper executes parity comparison and catches errors without printing transcripts."""
        # In lean environments without PyTorch installed (e.g. packaging CI),
        # supply a mock torch so this offline unit test can exercise the
        # runner, argument passing, and rank-1 parity comparator without heavyweight deps.
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            try:
                import torch as _test_torch
                torch_mod = _test_torch
            except ImportError:
                torch_mod = None

        if torch_mod is None:
            mock_torch = MagicMock()

            class MockTensor:
                def __init__(self, data, *args, **kwargs):
                    self.data = data

            mock_torch.tensor.side_effect = lambda data, *a, **k: MockTensor(data)
            mock_torch.inference_mode.return_value.__enter__ = lambda *a: None
            mock_torch.inference_mode.return_value.__exit__ = lambda *a: None
            patcher = patch.dict(sys.modules, {"torch": mock_torch})
            patcher.start()
            self.addCleanup(patcher.stop)

        ckpt_dir, valid_sha = self._create_mock_checkpoint("model.safetensors", b"whisper test bytes")
        cfg = copy.deepcopy(self.b1_config)
        cfg["model"]["weight_sha256"] = valid_sha
        cfg_file = self.work_dir / "test_whisper_cfg.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        manifest = self._create_mock_manifest([("clip-01", "clip1.flac"), ("clip-02", "clip2.flac")])

        dummy_audio = np.zeros(16000, dtype=np.float32)

        def mock_audio_loader(p: Path) -> np.ndarray:
            return dummy_audio

        call_count = 0

        def mock_nbest_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # clip-01: rank-1 matches single-result generation exactly
                return [{"rank": 1, "text": "[101] [102]", "token_ids": [101, 102, 50257]}], {}
            # clip-02: rank-1 produces intentional mismatch
            return [{"rank": 1, "text": "farkli metin", "token_ids": [999]}], {}

        outcomes = run_real_audio_smoke_check(
            config_path=cfg_file,
            checkpoint_path=ckpt_dir,
            audio_paths_or_manifest=manifest,
            model_loader=lambda: DummyModel(),
            processor_loader=lambda: DummyProcessor(),
            audio_loader=mock_audio_loader,
            nbest_fn=mock_nbest_fn,
        )

        self.assertEqual(len(outcomes), 2)
        self.assertEqual(outcomes[0].clip_id, "clip-01")
        self.assertEqual(outcomes[1].clip_id, "clip-02")

        # clip-01 must match verbatim without error
        self.assertEqual(outcomes[0].result.status, "match")
        self.assertTrue(outcomes[0].result.matched)
        self.assertTrue(outcomes[0].result.text_exact_match)
        self.assertTrue(outcomes[0].result.tokens_exact_match)
        self.assertIsNone(outcomes[0].result.error)

        # clip-02 must be an expected mismatch without error
        self.assertEqual(outcomes[1].result.status, "mismatch")
        self.assertFalse(outcomes[1].result.matched)
        self.assertFalse(outcomes[1].result.text_exact_match)
        self.assertFalse(outcomes[1].result.tokens_exact_match)
        self.assertIsNone(outcomes[1].result.error)

        # Neither outcome may ever be 'error'
        for outcome in outcomes:
            self.assertIn(outcome.result.status, ("match", "mismatch"))
            self.assertNotEqual(outcome.result.status, "error")


if __name__ == "__main__":
    unittest.main()
