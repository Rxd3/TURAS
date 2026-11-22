"""Verify M3 exports, add shared domain metrics and package reference-free handoff."""
from __future__ import annotations

from collections import Counter
import dataclasses
import json
from pathlib import Path
import sys
import zipfile

import jiwer
import numpy as np

from m3_finalize_b5 import PRIVATE, ROOT, SPLITS, now, sha
from turkish_asr_rdl.data.screening import Lexicon
from turkish_asr_rdl.evaluation.domain import evaluate_domain
from turkish_asr_rdl.evaluation.join import join
from turkish_asr_rdl.evaluation.metrics import corpus_rates, score_pair, UtteranceScore
from turkish_asr_rdl.evaluation.normalize import RAW_V1, TR_NORMALIZED_V1, normalize, word_tokens
from turkish_asr_rdl.schemas.jsonl import read_jsonl, write_json
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.schemas.predictions import validate_predictions

OUT = ROOT / "artifacts/m3-final-2026-11-22"


def main():
    paths = {name: OUT / name for name in ["b5", "gaussian-hmm"]}
    for path in paths.values():
        if not (path / "summary.json").exists():
            raise RuntimeError(f"Still running: {path.name}")
    ref_path = PRIVATE / "evaluator-final/references.test.jsonl"
    refs = read_jsonl(ref_path)
    ids = (SPLITS / "test.txt").read_text().splitlines()
    if [r["utterance_id"] for r in refs] != ids:
        raise ValueError("Test references do not match frozen IDs/order")
    split_sha = sha(SPLITS / "split_manifest.json")
    parent_protocol = json.loads((ROOT / "configs/evaluation/m45_final_protocol.json").read_text())
    lexicon_spec = parent_protocol["lexicon"]
    if sha(ROOT / lexicon_spec["path"]) != lexicon_spec["sha256"]:
        raise ValueError("Frozen lexicon hash mismatch")
    lexicon = Lexicon.load(ROOT / lexicon_spec["path"])
    systems = {}
    verified = {r["utterance_id"] for r in refs if r.get("verification_status") in ["verified", "adjudicated"]}
    for name, path in paths.items():
        summary = json.loads((path / "summary.json").read_text())
        validations = []
        for mode, split in [("validation", "validation"), ("test", "test"), ("timing", "test")]:
            directory = path / mode
            done = json.loads((directory / "completed.json").read_text())
            for filename, digest in done["files"].items():
                if sha(directory / filename) != digest:
                    raise ValueError(f"Changed export: {directory / filename}")
            expected = (SPLITS / ("m45-timing-30min.txt" if mode == "timing" else f"{split}.txt")).read_text().splitlines()
            for validator, filename in [(validate_predictions, "predictions.jsonl"), (validate_nbest, "nbest.jsonl")]:
                file = directory / filename
                if not file.exists():
                    if filename == "nbest.jsonl" and name == "gaussian-hmm":
                        continue
                    raise ValueError(f"Missing output: {file}")
                check = validator(read_jsonl(file), expected)
                if not check.ok:
                    raise ValueError(f"Contract violation: {check.errors}")
                validations.append({"path": str(file.relative_to(OUT)), "rows": len(expected), "valid": True})
        predictions = read_jsonl(path / "test/predictions.jsonl")
        joined = join(predictions, refs, split="test", split_manifest_sha256=split_sha)
        norm_scores = [score_pair(r.utterance_id, r.reference, r.hypothesis, TR_NORMALIZED_V1, r.status) for r in joined]
        # Audit the primary evaluator with the existing independent jiwer library.
        for view in [RAW_V1, TR_NORMALIZED_V1]:
            rtext = [normalize(r.reference, view) for r in joined]
            htext = [normalize(r.hypothesis, view) for r in joined]
            words = jiwer.process_words([" ".join(word_tokens(t)) for t in rtext], [" ".join(word_tokens(t)) for t in htext])
            transform = jiwer.ReduceToListOfListOfChars()
            chars = jiwer.process_characters(rtext, htext, reference_transform=transform, hypothesis_transform=transform)
            primary = summary["scores"]["test"]["metrics"][view]
            if abs(words.wer - primary["wer"]) > 1e-12 or abs(chars.cer - primary["cer"]) > 1e-12:
                raise ValueError("Independent jiwer WER/CER audit failed")
        domain = evaluate_domain(joined, lexicon)
        containing = set(domain.pop("term_containing_utterance_ids"))
        domain["term_containing_utterances"] = {"count": len(containing), "normalized_metrics": dataclasses.asdict(
            corpus_rates([s for s in norm_scores if s.utterance_id in containing]))}
        groups = sorted({r.leakage_group_id for r in joined})
        aggregated = []
        for group in groups:
            rates = corpus_rates([s for s, r in zip(norm_scores, joined) if r.leakage_group_id == group])
            aggregated.append(UtteranceScore(group, TR_NORMALIZED_V1, "ok", rates.words, rates.chars))
        rng = np.random.default_rng(2026)
        draws = rng.integers(0, len(groups), size=(2000, len(groups)))
        wer_samples = [corpus_rates([aggregated[i] for i in draw]).wer for draw in draws]
        cer_samples = [corpus_rates([aggregated[i] for i in draw]).cer for draw in draws]
        subset = [s for s in norm_scores if s.utterance_id in verified]
        timing_manifest = json.loads((path / "timing/manifest.json").read_text())
        records = [r for r in read_jsonl(path / "timing/timing.jsonl") if r["record_type"] == "batch" and not r["warmup"]]
        measured = [r["end_to_end_s"] for r in records]
        result = {"system_id": summary["system_id"], "test": summary["scores"]["test"],
                  "validation": summary["scores"]["validation"], "domain": domain,
                  "reported_verified_subset": {"rows": len(subset), "normalized_metrics": dataclasses.asdict(corpus_rates(subset)) if subset else None,
                                               "note": "Uses M1 status tags, not an independent listening audit"},
                  "normalized_bootstrap_95_ci": {"groups": len(groups), "replicates": 2000, "seed": 2026,
                       "wer": np.percentile(wer_samples, [2.5, 97.5]).tolist(), "cer": np.percentile(cer_samples, [2.5, 97.5]).tolist()},
                  "timing": {**summary["headline_timing"], "scope": timing_manifest["evaluation_timing"],
                       "environment": timing_manifest["environment"], "request_measurements": len(records),
                       "latency_p50_s": float(np.percentile(measured, 50)), "latency_p95_s": float(np.percentile(measured, 95)),
                       "rtfx": 1 / summary["headline_timing"]["rtf"]},
                  "independent_jiwer_crosscheck": "passed raw and normalized WER/CER", "contract_validation": validations,
                  "limitations": summary["limitations"]}
        write_json(path / "final-report.json", result)
        systems[name] = result
        print(f"VERIFIED {name}: test WER={result['test']['metrics'][TR_NORMALIZED_V1]['wer']:.6f}", flush=True)
    report = {"state": "completed", "completed_at_utc": now(), "dataset_release": "turas-tr-econ-legis-v0.1",
              "test_rows": len(ids), "reference_sha256": sha(ref_path),
              "reference_variant": "m1-test-minutes-archive-01ad5878-v1", "split_manifest_sha256": split_sha,
              "reference_verification_counts": dict(Counter(r.get("verification_status", "missing") for r in refs)),
              "systems": systems, "limitations": ["Results use working parliamentary minutes, not fully verified verbatim gold",
                  "B5 trained with one seed; Gaussian-HMM is an independent classical baseline",
                  "Timing is on this RTX 5070 desktop/CPU and is not a matched comparison with another machine"]}
    write_json(OUT / "final-summary.json", report)
    lines = ["TURAS Member 3 final results", "", f"Completed UTC: {report['completed_at_utc']}",
             "Test: 636 frozen IDs; working parliamentary minutes (not fully verified verbatim speech).", ""]
    for name, result in systems.items():
        metric = result["test"]["metrics"][TR_NORMALIZED_V1]
        lines.extend([name, f"  Test normalized WER: {metric['wer']:.6f} ({100*metric['wer']:.2f}%)",
                      f"  Test normalized CER: {metric['cer']:.6f} ({100*metric['cer']:.2f}%)",
                      f"  Terminology micro F1: {result['domain']['micro']['f1']:.6f}",
                      f"  End-to-end RTF: {result['timing']['rtf']:.6f}",
                      "  Exports passed shared validators; WER/CER independently cross-checked with jiwer.", ""])
    lines.extend(["B5 selected checkpoint: epoch 22 (selection frozen on validation before test scoring).",
                  "The supplied GMM-HMM method has one Gaussian per character state and nonstandard DCT/log-Mel features.",
                  "See final-summary.json and each model's final-report.json for counts, confidence intervals, timing and limitations."])
    (OUT / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    package()


def package():
    paths = {name: OUT / name for name in ["b5", "gaussian-hmm"]}
    archive = OUT / "TURAS-M3-final-2026-11-22.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ["RESULTS.txt", "final-summary.json"]:
            z.write(OUT / name, name)
        for name, path in paths.items():
            for item in sorted(path.rglob("*")):
                if item.is_file() and "feature-cache" not in item.parts:
                    z.write(item, item.relative_to(OUT).as_posix())
        checkpoint = ROOT / "artifacts/b5-gpu-resume-2026-11-22/best-checkpoint"
        for item in sorted(checkpoint.iterdir()):
            z.write(item, "b5/checkpoint/" + item.name)
        for name in ["m3_finalize_b5.py", "m3_run_gaussian_hmm.py", "m3_summarize_final.py", "m3_resume_b5_gpu.py"]:
            z.write(ROOT / "scripts" / name, "code/scripts/" + name)
        z.write(ROOT / "src/turkish_asr_rdl/baselines/m3_gaussian_hmm.py", "code/src/turkish_asr_rdl/baselines/m3_gaussian_hmm.py")
        for name in ["run-manifest.json", "training-history.json", "parent-validation.json"]:
            z.write(ROOT / "artifacts/b5-gpu-resume-2026-11-22" / name, "b5/training/" + name)
        # Preserve the checkpoint metadata's ../run-manifest.json relative link.
        z.write(ROOT / "artifacts/b5-gpu-resume-2026-11-22/run-manifest.json", "b5/run-manifest.json")
        z.write(ROOT / "requirements/windows-cu130.txt", "code/requirements/windows-cu130.txt")
        z.write(ROOT / "tests/test_m3_gaussian_hmm.py", "code/tests/test_m3_gaussian_hmm.py")
    write_json(OUT / "handoff-index.json", {"archive": archive.name, "sha256": sha(archive), "bytes": archive.stat().st_size,
                "reference_text_included": False, "feature_caches_included": False})
    print(f"COMPLETE: {archive}", flush=True)


if __name__ == "__main__":
    package() if "--package-only" in sys.argv else main()
