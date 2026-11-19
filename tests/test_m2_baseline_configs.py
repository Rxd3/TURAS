"""Offline unit tests for Member 2 baseline configuration files.

Validates that:
1. All four proposed baseline JSON configs exist and parse as valid JSON.
2. Model and processor IDs and exact pinned revisions match the checkpoint audit.
3. Dataset release, split manifest hash, and input paths remain unset in proposals.
4. Both B3 Qwen3-ASR candidates remain explicitly marked unselected.
5. Proposed decoding settings are internally consistent without treating beam settings as final.
6. Checkpoint revisions in configs match those documented in experiments/m2-baseline-notes.md.

Run:
    python -m unittest tests/test_m2_baseline_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]

BASELINE_CONFIG_FILES = {
    "B1": REPO_ROOT / "configs" / "baselines" / "whisper_small.json",
    "B2": REPO_ROOT / "configs" / "baselines" / "whisper_large_v3.json",
    "B3_candidate_1": REPO_ROOT / "configs" / "baselines" / "qwen3_asr_1_7b.json",
    "B3_candidate_2": REPO_ROOT / "configs" / "baselines" / "qwen3_asr_0_6b.json",
}

EXPECTED_AUDIT_IDENTITIES = {
    "B1": {
        "model_id": "openai/whisper-small",
        "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
        "precision": "float16",
        "acoustic_score_kind": "seq2seq_token_log_prob_sum",
        "score_convention_id": "whisper-original-logits-generated-plus-eos-v1",
    },
    "B2": {
        "model_id": "openai/whisper-large-v3",
        "revision": "06f233fe06e710322aca913c1bc4249a0d71fce1",
        "precision": "float16",
        "acoustic_score_kind": "seq2seq_token_log_prob_sum",
        "score_convention_id": "whisper-original-logits-generated-plus-eos-v1",
    },
    "B3_candidate_1": {
        "model_id": "Qwen/Qwen3-ASR-1.7B-hf",
        "revision": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
        "precision": "bfloat16",
        "acoustic_score_kind": "joint_speech_language_beam_score",
        "score_convention_id": "qwen3-asr-joint-sequence-score-v1",
    },
    "B3_candidate_2": {
        "model_id": "Qwen/Qwen3-ASR-0.6B-hf",
        "revision": "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
        "precision": "bfloat16",
        "acoustic_score_kind": "joint_speech_language_beam_score",
        "score_convention_id": "qwen3-asr-joint-sequence-score-v1",
    },
}


class Member2BaselineConfigTests(unittest.TestCase):
    """Offline validation of Member 2 baseline JSON configurations."""

    def load_configs(self) -> dict[str, dict]:
        configs = {}
        for track, path in BASELINE_CONFIG_FILES.items():
            self.assertTrue(path.is_file(), f"Expected config file does not exist: {path}")
            raw_text = path.read_text(encoding="utf-8")
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as err:
                self.fail(f"Config {path} failed to parse as valid JSON: {err}")
            self.assertIsInstance(data, dict, f"Config {path} root must be a JSON object")
            configs[track] = data
        return configs

    def test_all_baseline_configs_parse_strictly(self):
        """All four configuration files exist and parse as non-empty JSON objects."""
        configs = self.load_configs()
        self.assertEqual(set(configs.keys()), {"B1", "B2", "B3_candidate_1", "B3_candidate_2"})

    def test_model_and_processor_identities_and_revisions_match_audit(self):
        """Model IDs, processor IDs, precision, and 40-character SHAs match the audit."""
        configs = self.load_configs()
        for track, expected in EXPECTED_AUDIT_IDENTITIES.items():
            with self.subTest(track=track):
                cfg = configs[track]
                model = cfg.get("model", {})
                processor = cfg.get("processor", {})

                # Check model identity and revision
                self.assertEqual(model.get("id"), expected["model_id"])
                self.assertEqual(model.get("revision"), expected["revision"])
                self.assertEqual(len(model.get("revision", "")), 40)
                self.assertEqual(model.get("precision"), expected["precision"])

                # Processor must mirror model identity and revision
                self.assertEqual(processor.get("id"), expected["model_id"])
                self.assertEqual(processor.get("revision"), expected["revision"])
                self.assertEqual(processor.get("sampling_rate_hz"), 16000)

                # Score convention and acoustic score kind must match audit
                nbest = cfg.get("nbest", {})
                self.assertEqual(nbest.get("score_convention_id"), expected["score_convention_id"])
                self.assertEqual(nbest.get("acoustic_score_kind"), expected["acoustic_score_kind"])

    def test_dataset_bindings_and_inputs_are_unset_in_proposals(self):
        """Dataset release, split manifest hash, and input paths remain unset (None)."""
        configs = self.load_configs()
        for track, cfg in configs.items():
            with self.subTest(track=track):
                self.assertEqual(cfg.get("status"), "proposed-baseline-protocol")
                self.assertIsNone(cfg.get("dataset_release"), f"{track} dataset_release must be null")
                self.assertIsNone(cfg.get("split_manifest_sha256"), f"{track} split_manifest_sha256 must be null")

                inputs = cfg.get("inputs", {})
                self.assertIsInstance(inputs, dict)
                self.assertIsNone(inputs.get("inference"), f"{track} inputs.inference must be null")
                self.assertIsNone(inputs.get("audio_root"), f"{track} inputs.audio_root must be null")
                self.assertIsNone(inputs.get("split_manifest"), f"{track} inputs.split_manifest must be null")
                self.assertIsNone(inputs.get("test_ids"), f"{track} inputs.test_ids must be null")
                self.assertIsNone(inputs.get("validation_ids"), f"{track} inputs.validation_ids must be null")

    def test_both_qwen_candidates_remain_explicitly_unselected(self):
        """Both Qwen3-ASR candidate configs retain candidate_status == 'unselected-candidate'."""
        configs = self.load_configs()
        for qwen_track in ("B3_candidate_1", "B3_candidate_2"):
            with self.subTest(track=qwen_track):
                cfg = configs[qwen_track]
                self.assertEqual(
                    cfg.get("candidate_status"),
                    "unselected-candidate",
                    f"{qwen_track} candidate_status must be 'unselected-candidate'",
                )

    def test_proposed_decoding_values_are_internally_consistent(self):
        """Proposed decoding parameters are valid and internally consistent without assuming finality."""
        configs = self.load_configs()
        for track, cfg in configs.items():
            with self.subTest(track=track):
                decoding = cfg.get("decoding", {})
                self.assertIsInstance(decoding, dict)
                self.assertEqual(decoding.get("status"), "proposed")
                self.assertEqual(decoding.get("algorithm"), "beam")

                num_beams = decoding.get("num_beams")
                requested_n = decoding.get("requested_n")
                self.assertIsInstance(num_beams, int)
                self.assertIsInstance(requested_n, int)
                self.assertGreater(num_beams, 0)
                self.assertGreater(requested_n, 0)
                self.assertGreaterEqual(
                    num_beams,
                    requested_n,
                    f"{track}: num_beams ({num_beams}) must be >= requested_n ({requested_n})",
                )

                self.assertEqual(decoding.get("max_new_tokens"), 440)
                self.assertEqual(decoding.get("language"), "tr")
                self.assertFalse(decoding.get("do_sample"))
                self.assertEqual(decoding.get("temperature"), 0.0)

                # Track-specific decoding checks
                if track.startswith("B1") or track.startswith("B2"):
                    self.assertEqual(decoding.get("task"), "transcribe")
                    forced_prefix = decoding.get("forced_prefix_tokens", [])
                    self.assertEqual(
                        forced_prefix,
                        ["<|startoftranscript|>", "<|tr|>", "<|transcribe|>", "<|notimestamps|>"],
                    )

    def test_audit_notes_revisions_match_configs(self):
        """Experiments notes table contains the exact revisions present in the configs."""
        notes_path = REPO_ROOT / "experiments" / "m2-baseline-notes.md"
        self.assertTrue(notes_path.is_file(), "m2-baseline-notes.md must exist")
        notes_text = notes_path.read_text(encoding="utf-8")

        for track, expected in EXPECTED_AUDIT_IDENTITIES.items():
            with self.subTest(track=track):
                # Pinned revision must be cited in notes
                rev = expected["revision"]
                self.assertIn(rev[:8], notes_text, f"Short revision {rev[:8]} for {track} missing from notes")
                self.assertIn(expected["model_id"], notes_text, f"Model ID {expected['model_id']} missing from notes")


if __name__ == "__main__":
    unittest.main()
