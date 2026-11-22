"""Offline shared scoring of frozen baseline exports, with validation-only Qwen selection."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import jiwer
import numpy as np

from turkish_asr_rdl.data.screening import Lexicon
from turkish_asr_rdl.evaluation.domain import evaluate_domain
from turkish_asr_rdl.evaluation.join import join
from turkish_asr_rdl.evaluation.metrics import corpus_rates, score_pair, UtteranceScore
from turkish_asr_rdl.evaluation.normalize import RAW_V1, TR_NORMALIZED_V1, normalize, word_tokens
from turkish_asr_rdl.schemas.jsonl import read_jsonl, write_json
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.schemas.nbest import validate_nbest
from run_completion_baseline import BASE, ROOT, RELEASE, SPLITS, sha


def score(system, split):
    directory = BASE / "runs" / system / split
    done = json.loads((directory / "completed.json").read_text())
    if split == "test" and not (BASE / "runs" / system / "timing/completed.json").exists():
        raise ValueError("Finish this model's frozen repeated timing before opening its test scoring")
    for name, digest in done["files"].items():
        if sha(directory / name) != digest:
            raise ValueError("Frozen inference output hash mismatch")
    ids = (SPLITS / f"{split}.txt").read_text().splitlines()
    predictions = read_jsonl(directory / "predictions.jsonl")
    check = validate_predictions(predictions, ids)
    if not check.ok:
        raise ValueError(check.errors)
    candidate_path = directory / "nbest.jsonl"
    if candidate_path.exists() and not validate_nbest(read_jsonl(candidate_path), ids).ok:
        raise ValueError("N-best contract mismatch")
    private = ROOT / "data/private" / RELEASE
    refpath = private / ("evaluator-final/references.test.jsonl" if split == "test" else "validation-recovered-minutes-v1/references.validation.jsonl")
    references = read_jsonl(refpath)
    joined = join(predictions, references, split=split, split_manifest_sha256=sha(SPLITS / "split_manifest.json"))
    scores = {view: [score_pair(r.utterance_id, r.reference, r.hypothesis, view, r.status) for r in joined]
              for view in [RAW_V1, TR_NORMALIZED_V1]}
    metrics = {view: asdict(corpus_rates(values)) for view, values in scores.items()}
    for view in scores:
        refs = [normalize(r.reference, view) for r in joined]
        hyps = [normalize(r.hypothesis, view) for r in joined]
        words = jiwer.process_words([" ".join(word_tokens(s)) for s in refs], [" ".join(word_tokens(s)) for s in hyps])
        transform = jiwer.ReduceToListOfListOfChars()
        chars = jiwer.process_characters(refs, hyps, reference_transform=transform, hypothesis_transform=transform)
        if abs(words.wer - metrics[view]["wer"]) > 1e-12 or abs(chars.cer - metrics[view]["cer"]) > 1e-12:
            raise ValueError("Independent JiWER audit failed")
    lexicon_path = ROOT / "configs/datasets/turas-domain-lexicon-v0.1.json"
    lexicon = Lexicon.load(lexicon_path)
    domain = evaluate_domain(joined, lexicon)
    containing = set(domain.pop("term_containing_utterance_ids"))
    domain["term_containing_utterances"] = {"rows": len(containing), "normalized_metrics": asdict(corpus_rates(
        [s for s in scores[TR_NORMALIZED_V1] if s.utterance_id in containing])) if containing else None}
    groups = sorted({r.leakage_group_id for r in joined})
    aggregated = []
    for group in groups:
        group_rates = corpus_rates([s for s, row in zip(scores[TR_NORMALIZED_V1], joined) if row.leakage_group_id == group])
        aggregated.append(UtteranceScore(group, TR_NORMALIZED_V1, "ok", group_rates.words, group_rates.chars))
    draws = np.random.default_rng(2026).integers(0, len(groups), size=(2000, len(groups)))
    wers = [corpus_rates([aggregated[i] for i in draw]).wer for draw in draws]
    cers = [corpus_rates([aggregated[i] for i in draw]).cer for draw in draws]
    verified = {r["utterance_id"] for r in references if r.get("verification_status") in ["verified", "adjudicated"]}
    subset = [s for s in scores[TR_NORMALIZED_V1] if s.utterance_id in verified]
    result = {"state": "completed", "system_key": system, "system_id": done["system_id"], "split": split,
              "rows": len(joined), "reference_sha256": sha(refpath), "predictions_sha256": sha(directory / "predictions.jsonl"),
              "nbest_sha256": sha(candidate_path) if candidate_path.exists() else None,
              "metrics": metrics, "domain": domain, "reference_status_counts": dict(Counter(r.get("verification_status", "missing") for r in references)),
              "reported_verified_subset": {"rows": len(subset), "normalized_metrics": asdict(corpus_rates(subset)) if subset else None},
              "bootstrap": {"seed": 2026, "replicates": 2000, "recording_groups": len(groups),
                            "normalized_wer_95_ci": np.percentile(wers, [2.5, 97.5]).tolist(),
                            "normalized_cer_95_ci": np.percentile(cers, [2.5, 97.5]).tolist()},
              "independent_jiwer_crosscheck": "passed raw and normalized WER/CER", "scored_at_utc": datetime.now(timezone.utc).isoformat(),
              "limitations": ["Working edited-minutes references", "Follow-up study after earlier working test exposure",
                              "Training-data overlap uncertainty is disclosed in candidate audit"]}
    if split == "test":
        timing_path = BASE / "runs" / system / "timing"
        result["timing"] = json.loads((timing_path / "completed.json").read_text())
        result["timing_manifest_sha256"] = sha(timing_path / "manifest.json")
    output = BASE / "scores" / split / f"{system}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(f"SCORED {system} {split}: WER={metrics[TR_NORMALIZED_V1]['wer']:.6f}, CER={metrics[TR_NORMALIZED_V1]['cer']:.6f}", flush=True)
    return result


def select_qwen():
    path = BASE / "b3-selection.json"
    results = [json.loads((BASE / "scores/validation" / f"{s}.json").read_text()) for s in ["qwen17", "qwen06"]]
    chosen = min(results, key=lambda r: (r["metrics"][TR_NORMALIZED_V1]["cer"], r["metrics"][TR_NORMALIZED_V1]["wer"], r["system_key"]))
    record = {"state": "frozen_before_selected_qwen_test", "selected_system": chosen["system_key"],
              "criterion": "minimum validation normalized CER, then normalized WER, then stable system key",
              "validation_report_sha256": {r["system_key"]: sha(BASE / "scores/validation" / f"{r['system_key']}.json") for r in results},
              "selected_at_utc": datetime.now(timezone.utc).isoformat(), "peer_approval_claimed": False,
              "authority": "project owner goal to audit and execute remaining B3 candidates"}
    if path.exists() and json.loads(path.read_text())["selected_system"] != record["selected_system"]:
        raise ValueError("Refusing to change an already frozen Qwen selection")
    write_json(path, record)
    print(f"SELECTED B3: {record['selected_system']} using validation only", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--systems", nargs="+", default=[])
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--select-qwen", action="store_true")
    args = p.parse_args()
    for system in args.systems:
        score(system, args.split)
    if args.select_qwen:
        select_qwen()
