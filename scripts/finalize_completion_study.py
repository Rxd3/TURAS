"""Verify complete follow-up exports before publishing aggregate-only results.

This command fails closed while any required full-split or repeated-timing run is
missing. It does not rerun inference, choose settings from test, or publish text.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from turkish_asr_rdl.evaluation.normalize import TR_NORMALIZED_V1
from turkish_asr_rdl.schemas.jsonl import read_jsonl, write_json
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.schemas.nbest import validate_nbest
from run_completion_baseline import ROOT, BASE, SPLITS, RELEASE, sha
from verify_submission import main as verify_original_submission


LABELS = {"b2": "B2 Whisper-large-v3", "qwen17": "B3 Qwen3-ASR-1.7B",
          "qwen06": "B3 Qwen3-ASR-0.6B", "seda": "Seda Zipformer2 RNN-T",
          "gmm": "Four-mixture grapheme HMM"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_run(system, mode):
    directory = BASE / "runs" / system / mode
    done = read(directory / "completed.json")
    if done["state"] != "completed":
        raise ValueError(f"Incomplete {system}/{mode}")
    for name, digest in done["files"].items():
        if sha(directory / name) != digest:
            raise ValueError(f"Changed completed export: {system}/{mode}/{name}")
    name = "m45-timing-30min" if mode == "timing" else mode
    ids = (SPLITS / f"{name}.txt").read_text().splitlines()
    for validator, filename in [(validate_predictions, "predictions.jsonl"), (validate_nbest, "nbest.jsonl")]:
        path = directory / filename
        if filename == "nbest.jsonl" and system in ["seda", "gmm"]:
            if path.exists():
                raise ValueError("Unsupported N-best must not be fabricated")
            continue
        report = validator(read_jsonl(path), ids)
        if not report.ok:
            raise ValueError(report.errors)
    predictions = read_jsonl(directory / "predictions.jsonl")
    if any(r["status"] == "error" for r in predictions):
        raise ValueError(f"Inference error in {system}/{mode}")
    manifest = read(directory / "manifest.json")
    frozen = read(directory / "frozen-run.json")
    if frozen["split_manifest_sha256"] != sha(SPLITS / "split_manifest.json"):
        raise ValueError("Split manifest changed")
    if sha(ROOT / "scripts/run_completion_baseline.py") != frozen["script_sha256"]:
        raise ValueError("Runner differs from executed snapshot")
    result = {"completed_sha256": sha(directory / "completed.json"),
              "frozen_run_sha256": sha(directory / "frozen-run.json"),
              "files": done["files"], "rows": len(ids), "contract_validation": "passed"}
    if mode == "timing":
        records = read_jsonl(directory / "timing.jsonl")
        batches = [r for r in records if r["record_type"] == "batch" and not r["warmup"]]
        warm = [r for r in records if r["record_type"] == "batch" and r["warmup"]]
        if len(warm) != 3 or done["repeats"] != 3:
            raise ValueError("Timing warmup/repetition mismatch")
        for repeat in [1, 2, 3]:
            seen = [uid for row in batches if row["repeat"] == repeat for uid in row["utterance_ids"]]
            if seen != ids:
                raise ValueError("Timing coverage/order mismatch")
        if len(batches) != len(ids) * 3 or any(r["error_rows"] or r["batch_size"] != 1 for r in batches):
            raise ValueError("Timing batch/error mismatch")
        rtf = sum(r["end_to_end_s"] for r in batches) / sum(r["audio_duration_s"] for r in batches)
        if abs(rtf - done["rtf"]) > 1e-10:
            raise ValueError("Timing aggregate differs from recorded durations")
        result.update(rtf=rtf, rtfx=1 / rtf, request_measurements=len(batches),
                      latency_p50_s=float(np.percentile([r["end_to_end_s"] for r in batches], 50)),
                      latency_p95_s=float(np.percentile([r["end_to_end_s"] for r in batches], 95)),
                      scope=manifest["evaluation_timing"], environment=manifest["environment"],
                      device=manifest["configuration"]["resolved"]["model"]["device"])
    return result


def main():
    verify_original_submission(ROOT)
    original_protocol_path = ROOT / "configs/evaluation/m45_final_protocol.json"
    original_protocol = read(original_protocol_path)
    lexicon = original_protocol["lexicon"]
    if sha(ROOT / lexicon["path"]) != lexicon["sha256"]:
        raise ValueError("Frozen domain lexicon changed")
    test_reference_sha = original_protocol["test_reference"]["payload_sha256"]
    hardware_path = ROOT / "results/final/completion_hardware_context.json"
    hardware = read(hardware_path)
    if not hardware["cpu_model"] or hardware["ram_os_visible_bytes"] <= 0 or hardware["gpu_count"] != 1:
        raise ValueError("Incomplete follow-up host context")
    protocol = read(ROOT / "configs/evaluation/completion_followup_2026-11-22.json")
    for path, digest in protocol["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Follow-up critical source changed: {path}")
    selection = read(BASE / "b3-selection.json")
    parent = read(BASE / "runs/qwen06/validation/frozen-run.json")["parent_commit"]
    baseline_sources = {}
    for path in ["src/turkish_asr_rdl/baselines/whisper.py", "src/turkish_asr_rdl/baselines/qwen.py",
                 "src/turkish_asr_rdl/decoding/whisper.py"]:
        committed = subprocess.check_output(["git", "show", f"{parent}:{path}"], cwd=ROOT)
        if (ROOT / path).read_bytes() != committed:
            raise ValueError(f"Existing baseline runtime differs from recorded parent: {path}")
        baseline_sources[path] = sha(ROOT / path)
    candidate_reports = {s: read(BASE / "scores/validation" / f"{s}.json") for s in ["qwen17", "qwen06"]}
    validation_reference_sha = candidate_reports["qwen06"]["reference_sha256"]
    for system in candidate_reports:
        verify_run(system, "validation")
        if candidate_reports[system]["reference_sha256"] != validation_reference_sha:
            raise ValueError("Qwen candidates used different validation references")
        if sha(BASE / "scores/validation" / f"{system}.json") != selection["validation_report_sha256"][system]:
            raise ValueError("Qwen selection report changed after selection")
    best = min(candidate_reports, key=lambda s: (candidate_reports[s]["metrics"][TR_NORMALIZED_V1]["cer"],
                                               candidate_reports[s]["metrics"][TR_NORMALIZED_V1]["wer"], s))
    if best != selection["selected_system"]:
        raise ValueError("Qwen selection does not follow frozen validation criterion")
    test_start = read(BASE / "runs" / best / "test/frozen-run.json")["frozen_at_utc"]
    if datetime.fromisoformat(test_start) <= datetime.fromisoformat(selection["selected_at_utc"]):
        raise ValueError("Qwen test began before candidate selection")
    systems = {}
    for system in ["b2", best, "seda", "gmm"]:
        evidence = {mode: verify_run(system, mode) for mode in ["validation", "test", "timing"]}
        test = read(BASE / "scores/test" / f"{system}.json")
        val = read(BASE / "scores/validation" / f"{system}.json")
        if test["reference_sha256"] != test_reference_sha or val["reference_sha256"] != validation_reference_sha:
            raise ValueError("Follow-up comparison changed working reference bytes")
        for split, report in [("test", test), ("validation", val)]:
            if report["rows"] != evidence[split]["rows"] or report["predictions_sha256"] != evidence[split]["files"]["predictions.jsonl"]:
                raise ValueError("Score coverage or prediction identity mismatch")
            if not report["independent_jiwer_crosscheck"].startswith("passed"):
                raise ValueError("Missing independent metric-library check")
        if test["timing_manifest_sha256"] != evidence["timing"]["files"]["manifest.json"]:
            raise ValueError("Score timing identity mismatch")
        systems[system] = {"label": LABELS[system], "test": test, "validation": val, "exports": evidence}
    gm = read(BASE / "gmm/model-manifest.json")
    with np.load(BASE / "gmm/checkpoint.npz", allow_pickle=False) as checkpoint:
        means, variances, weights = checkpoint["means"], checkpoint["variances"], checkpoint["weights"]
        units, priors = checkpoint["units"].tolist(), checkpoint["unit_priors"]
        if (means.shape != (gm["trained_states"], 4, 39) or variances.shape != means.shape
                or weights.shape != means.shape[:2] or not np.allclose(weights.sum(axis=1), 1)):
            raise ValueError("Not a valid four-component diagonal GMM")
        if not all(np.isfinite(x).all() for x in [means, variances, weights]) or not (variances > 0).all() or not (weights > 0).all():
            raise ValueError("Invalid mixture parameters")
        if (len(units) != len(set(units)) or len(units) * 3 != len(means) or priors.shape != (len(units),)
                or not np.isfinite(priors).all() or not (priors > 0).all() or not np.isclose(priors.sum(), 1)):
            raise ValueError("Invalid train-character priors/state inventory")
        pairwise = np.linalg.norm(means[:, :, None, :] - means[:, None, :, :], axis=-1)
        first, second = np.triu_indices(4, 1)
        distinct = np.min(pairwise[:, first, second], axis=1)
        if not (distinct > 1e-6).all():
            raise ValueError("Collapsed identical component means")
        mixture_evidence = {**gm, "all_states_have_distinct_component_means": True,
                            "minimum_pairwise_component_mean_distance": float(distinct.min()),
                            "minimum_component_weight": float(weights.min()), "unit_priors_sum": float(priors.sum()),
                            "parameter_shape": list(means.shape)}
    if sha(BASE / "gmm/checkpoint.npz") != gm["checkpoint_sha256"]:
        raise ValueError("GMM checkpoint hash mismatch")
    proof_path = BASE / "reproduction/fresh-environment-b5-restore.json"
    proof = read(proof_path)
    for flag in ["isolated_new_python_environment", "torch_loaded_from_new_environment", "model_code_imported_from_archived_source",
                 "model_weights_restored_from_handoff", "audio_restored_from_role_package"]:
        if proof[flag] is not True:
            raise ValueError(f"Fresh restoration proof missing: {flag}")
    if proof["state"] != "passed" or proof["system_site_packages"] or proof["validation_transcripts_matching_exactly"] != 3:
        raise ValueError("Fresh-environment execution not verified")
    # The public summary omits the private environment's absolute local paths.
    public_proof = {k: v for k, v in proof.items() if k != "packages"}
    expanded_path = ROOT / "results/final/fresh_environment_restoration.json"
    expanded = read(expanded_path)
    acceptance_path = BASE / "reproduction/m45-fresh-restore/acceptance.json"
    acceptance = read(acceptance_path)
    if expanded["state"] != "passed" or not expanded["isolated_new_python_environment"]:
        raise ValueError("Expanded fresh-environment restoration is unverified")
    if acceptance["status"] != "passed" or any(j["status"] != "passed" or j["exit_code"] for j in acceptance["jobs"]):
        raise ValueError("A restored M4/M5 runtime job failed")
    if sha(acceptance_path) != expanded["m45_restoration"]["acceptance_report_sha256"]:
        raise ValueError("Expanded restoration evidence identity mismatch")
    report = {"state": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_release": RELEASE, "split_manifest_sha256": sha(SPLITS / "split_manifest.json"),
              "test_rows": 636, "followup_protocol_sha256": sha(ROOT / "configs/evaluation/completion_followup_2026-11-22.json"),
              "original_study_protocol_sha256": sha(original_protocol_path), "domain_lexicon_sha256": lexicon["sha256"],
              "test_reference_sha256": test_reference_sha, "validation_reference_sha256": validation_reference_sha,
              "hardware_and_measurement_context": hardware, "hardware_context_sha256": sha(hardware_path),
              "b3_selection": selection, "b3_validation_candidates": candidate_reports, "systems": systems,
              "existing_baseline_runtime_audit": {"recorded_parent_commit": parent,
                  "checked_after_runs": True, "files_match_recorded_git_objects": True, "sha256": baseline_sources},
              "gmm_training": mixture_evidence, "fresh_environment_restore": public_proof,
              "expanded_fresh_environment_restore": expanded,
              "expanded_fresh_environment_report_sha256": sha(expanded_path),
              "fresh_environment_private_proof_sha256": sha(proof_path),
              "limitations": protocol["limitations"], "independent_human_approval": False,
              "cloud_upload_or_recipient_download_claimed": False}
    output = ROOT / "results/final/completion_test_summary.json"
    write_json(output, report)
    with (output.parent / "completion_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "normalized_wer_pct", "normalized_cer_pct", "term_f1_pct", "end_to_end_rtf", "device"])
        for key, system in systems.items():
            rates = system["test"]["metrics"][TR_NORMALIZED_V1]
            writer.writerow([key, 100 * rates["wer"], 100 * rates["cer"], 100 * system["test"]["domain"]["micro"]["f1"],
                             system["exports"]["timing"]["rtf"], system["exports"]["timing"]["device"]])
    lines = ["# Baseline completion report", "",
             "B2 was rerun locally on an RTX 5070 because M2's reported ZIP was not available. "
             "Both Qwen candidates were evaluated on all 438 validation clips; the validation winner alone was run on test. "
             "A Turkish RNN-T and a newly trained genuine mixture-Gaussian grapheme HMM complete the follow-up coverage.", "",
             "These are separate follow-up experiments after previous working-test exposure. "
             "The frozen original M4/M5 study and the supplied single-Gaussian negative result are preserved.", "",
             "## Full frozen test results", "",
             "All four systems cover the same 636 test IDs and unchanged working reference bytes. "
             "WER/CER use the shared Turkish normalization and corpus edit counts, cross-checked against JiWER. "
             "Predictions contain no reference text, and supported N-best lists are separate validated JSONL files.", "",
             "| System | Norm. WER | Norm. CER | Term F1 | End-to-end RTF | Device |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for key, system in systems.items():
        rates = system["test"]["metrics"][TR_NORMALIZED_V1]
        timing = system["exports"]["timing"]
        lines.append(f"| {system['label']} | {100*rates['wer']:.3f}% | {100*rates['cer']:.3f}% | "
                     f"{100*system['test']['domain']['micro']['f1']:.3f}% | {timing['rtf']:.5f} | {timing['device']} |")
    lines += ["", "RTF is measured through shared `run_timed` on 107 frozen clips, three repeats, batch one, "
              "with three train warm-ups. Audio loading/resampling and serialization are included. "
              "Headline jobs run serially after project downloads/training stop. CPU and GPU rows describe different devices, "
              "and these single-machine measurements do not establish a hardware-matched model-efficiency ranking.", "",
              "[Hardware/cache and measurement scope](completion_hardware_context.json) records CPU/RAM, thread counts, "
              "GPU/driver and local NVMe storage. OS caching is enabled and is not forcibly cleared; no cold-filesystem claim is made. "
              "Model loading, warm-ups and output-file writing/flush are excluded from inference RTF.", "",
              "## Validation-only Qwen selection", "",
              "| Candidate | Validation norm. WER | Validation norm. CER | Selected |",
              "| --- | ---: | ---: | --- |"]
    for key, candidate in candidate_reports.items():
        rates = candidate["metrics"][TR_NORMALIZED_V1]
        lines.append(f"| {LABELS[key]} | {100*rates['wer']:.3f}% | {100*rates['cer']:.3f}% | {'yes' if key == best else 'no'} |")
    lines += ["", "The fixed criterion is minimum validation normalized CER, then WER, then stable system key. "
              "The selected report hashes and timestamp precede the winner's test inference. "
              "There is no hotword or external context input. Qwen scores remain labeled speech-language joint scores.", "",
              "## Classical and transducer methods", "",
              "Seda uses the pinned Turkish Zipformer2 RNN-T acoustic model through sherpa-onnx 2.0.2, "
              "greedy decoding, one CPU thread and no optional Wikipedia-trained LM. Its raw spoken-form output has no ITN.", "",
              "The new HMM uses 39-dimensional MFCC/delta/acceleration features, three states per grapheme and "
              "four diagonal Gaussian components per state. All 4,376 train clips were feasible. "
              "Five fixed Viterbi-EM passes update state alignments and soft component responsibilities. "
              "The saved 138 x 4 x 39 parameter arrays have positive normalized weights and distinct component means. "
              "Only 29 of 138 short initializer fits converged within their fixed 15-step cap; this is recorded, "
              "followed by the five full training passes. No test-driven mixture-count or decoder tuning is claimed. "
              "This is a grapheme system without a pronunciation lexicon or external LM. "
              "It does not represent a conventional phoneme/lexicon GMM-HMM recognizer.", "",
              "## Reproduction and limits", "",
              "A newly installed isolated Python environment restored archived B5 source/weights/audio and matched three "
              "archived validation transcripts exactly. A separate fresh-source restoration verified 226 M4/M5 artifact members "
              "and executed three acoustic models, three online joint runs and 15 cached controls on two validation clips, "
              "including cached/joint transcript parity for all three adapted models. "
              "The follow-up runs use newly installed pinned neural/CPU environments. "
              "Replay commands write to fresh directories and compare actual outputs. "
              "This is an owner-assisted same-machine check, not an independent person's review or a fresh OS installation.", "",
              "Working edited-minutes references retain 611 unreviewed and 25 historical single-reviewer test tags. "
              "Recording-group bootstrap intervals are included in the JSON; speaker disjointness and training-seed robustness "
              "remain unproven. Publisher cards do not provide complete clip-level pretraining manifests, so Qwen/Seda "
              "training overlap remains uncertain. Independent reference adjudication and actual team approval remain separate tasks.", "",
              "[Full aggregate evidence](completion_test_summary.json) · [Candidate audit and protocol](../../docs/baseline_completion_followup.md) · "
              "[Replay commands](../../docs/completion_replay_commands.md)", ""]
    (output.parent / "BASELINE_COMPLETION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"VERIFIED follow-up: {', '.join(systems)}; complete validation/test and 107 x 3 timing", flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
