"""Score a complete frozen split with shared TURAS joining and metrics."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

from turkish_asr_rdl.evaluation.join import join
from turkish_asr_rdl.evaluation.metrics import corpus_rates, score_pair
from turkish_asr_rdl.evaluation.normalize import RAW_V1, TR_NORMALIZED_V1
from turkish_asr_rdl.schemas.jsonl import read_jsonl, write_json
from turkish_asr_rdl.schemas.predictions import validate_predictions


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a fresh output file")
    ids = args.ids.read_text(encoding="utf-8").splitlines()
    predictions = read_jsonl(args.predictions)
    check = validate_predictions(predictions, ids)
    if not check.ok:
        raise ValueError(check.errors)
    manifest_sha = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    joined = join(predictions, read_jsonl(args.references), split=args.split, split_manifest_sha256=manifest_sha)
    report = {"split": args.split, "rows": len(joined), "reference_sha256": hashlib.sha256(args.references.read_bytes()).hexdigest(),
              "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              "metrics": {view: asdict(corpus_rates([score_pair(r.utterance_id, r.reference, r.hypothesis, view, r.status)
                                                    for r in joined])) for view in [RAW_V1, TR_NORMALIZED_V1]}}
    write_json(args.output, report)
    print(f"{len(joined)} rows; normalized WER={report['metrics'][TR_NORMALIZED_V1]['wer']:.9f}")


if __name__ == "__main__":
    main()
