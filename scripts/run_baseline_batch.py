"""Thin CLI entry point for baseline batch inference and shared artifact export.

Runs Whisper (B1/B2) or Qwen (B3) baselines on reference-free inference input manifests,
validating checkpoint artifacts and exporting validated predictions.jsonl, nbest.jsonl,
and manifest.json to the specified destination directory outside the repository.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from turkish_asr_rdl.baselines.runner import (
    CheckpointValidationError,
    InputManifestError,
    run_baseline_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="TURAS Baseline Batch Inference & Exporter")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline config JSON (B1, B2, or B3)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to local checkpoint directory or weight file")
    parser.add_argument("--manifest", type=str, required=True, help="Path to reference-free input manifest JSONL")
    parser.add_argument("--output-dir", type=str, required=True, help="Output folder outside repository")
    parser.add_argument("--audio-root", type=str, default=None, help="Root folder for relative audio_uri in manifest")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda, cpu)")
    parser.add_argument("--requested-n", type=int, default=None, help="N-best candidate count override")
    parser.add_argument("--shared-path", type=str, default=None, help="Path to shared decoder checkout/export")
    parser.add_argument("--split", type=str, default=None, help="Split name override (e.g. test, validation)")
    args = parser.parse_args()

    cmd_line = " ".join([Path(sys.executable).name, *sys.argv])

    try:
        result = run_baseline_batch(
            config=args.config,
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            audio_root=args.audio_root,
            device=args.device,
            requested_n=args.requested_n,
            shared_path=args.shared_path,
            command_line=cmd_line,
            split_name=args.split,
        )
    except CheckpointValidationError as c_err:
        print(f"CHECKPOINT ERROR: {c_err}", file=sys.stderr)
        return 1
    except InputManifestError as m_err:
        print(f"MANIFEST ERROR: {m_err}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"BATCH INFERENCE ERROR: {exc}", file=sys.stderr)
        return 1

    print("=== TURAS Baseline Batch Export Summary ===")
    print(f"Run ID:             {result.run_id}")
    print(f"System ID:          {result.system_id}")
    print(f"Total utterances:   {result.total_utterances}")
    print(f"  OK:               {result.ok_count}")
    print(f"  Empty:            {result.empty_count}")
    print(f"  Error:            {result.error_count}")
    print(f"Predictions:        {result.predictions_path}")
    print(f"  SHA-256:          {result.predictions_sha256}")
    if result.nbest_path:
        print(f"N-best:             {result.nbest_path}")
        print(f"  SHA-256:          {result.nbest_sha256}")
    print(f"Run Manifest:       {result.manifest_path}")
    print(f"  SHA-256:          {result.manifest_sha256}")
    print(f"Validation OK:      {result.validation_ok}")
    if not result.validation_ok:
        print(f"Validation errors:  {result.validation_errors}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
