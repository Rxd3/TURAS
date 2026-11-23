"""Verify follow-up public counts, source pins and table consistency without ML dependencies."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message="Follow-up aggregate check failed"):
    # This is a handoff verifier: checks must also execute under Python -O.
    if not condition:
        raise ValueError(message)


def verify(root=ROOT):
    path = root / "results/final/completion_test_summary.json"
    if not path.is_file():
        raise SystemExit("PENDING: results/final/completion_test_summary.json is not in the repository. "
                         "The committed follow-up results are results/final/baseline_completion_summary.json.")
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol_path = root / "configs/evaluation/completion_followup_2026-11-22.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(sha(protocol_path) == report["followup_protocol_sha256"], "Follow-up protocol changed")
    original_path = root / "configs/evaluation/m45_final_protocol.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    require(sha(original_path) == report["original_study_protocol_sha256"])
    require(report["test_reference_sha256"] == original["test_reference"]["payload_sha256"])
    require(sha(root / original["lexicon"]["path"]) == report["domain_lexicon_sha256"] == original["lexicon"]["sha256"])
    for name, digest in protocol["source_sha256"].items():
        require(sha(root / name) == digest, f"Critical source changed: {name}")
    audit = report["existing_baseline_runtime_audit"]
    require(audit["files_match_recorded_git_objects"])
    for name, digest in audit["sha256"].items():
        require(sha(root / name) == digest, f"Existing baseline runtime changed: {name}")
    require(sha(root / "data/splits/turas-tr-econ-legis-v0.1/split_manifest.json") == report["split_manifest_sha256"])
    require(sha(root / "results/final/completion_hardware_context.json") == report["hardware_context_sha256"])
    hardware = report["hardware_and_measurement_context"]
    require(bool(hardware["cpu_model"]) and hardware["ram_os_visible_bytes"] > 0 and hardware["gpu_count"] == 1)
    selected = report["b3_selection"]["selected_system"]
    require(set(report["systems"]) == {"b2", selected, "seda", "gmm"})
    with (root / "results/final/completion_metrics.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 4 and {r["system"] for r in rows} == set(report["systems"]))
    indexed = {r["system"]: r for r in rows}
    for key, system in report["systems"].items():
        for split, expected in [("validation", 438), ("test", 636)]:
            score = system[split]
            require(score["state"] == "completed" and score["rows"] == expected)
            require(score["reference_sha256"] == report[f"{split}_reference_sha256"])
            require(score["predictions_sha256"] == system["exports"][split]["files"]["predictions.jsonl"])
            for metric in score["metrics"].values():
                require(metric["utterances"] == expected and metric["error_rows"] == 0)
                for kind, field in [("words", "wer"), ("chars", "cer")]:
                    counts = metric[kind]
                    require(all(type(v) is int and v >= 0 for v in counts.values()))
                    expected_rate = (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / (counts["hits"] + counts["substitutions"] + counts["deletions"])
                    require(math.isclose(metric[field], expected_rate, rel_tol=0, abs_tol=1e-12))
            require(score["independent_jiwer_crosscheck"].startswith("passed"))
        timing = system["exports"]["timing"]
        require(timing["rows"] == 107 and timing["request_measurements"] == 321)
        require(timing["scope"]["repeats"] == 3 and timing["scope"]["batch_size"] == 1)
        require(timing["rtf"] > 0 and math.isclose(timing["rtfx"], 1 / timing["rtf"]))
        rates = system["test"]["metrics"]["tr-normalized-v1"]
        for column, value in [("normalized_wer_pct", 100 * rates["wer"]), ("normalized_cer_pct", 100 * rates["cer"]),
                              ("term_f1_pct", 100 * system["test"]["domain"]["micro"]["f1"]), ("end_to_end_rtf", timing["rtf"])]:
            require(math.isclose(float(indexed[key][column]), value, rel_tol=0, abs_tol=1e-10))
    candidates = report["b3_validation_candidates"]
    best = min(candidates, key=lambda s: (candidates[s]["metrics"]["tr-normalized-v1"]["cer"], candidates[s]["metrics"]["tr-normalized-v1"]["wer"], s))
    require(selected == best)
    gm = report["gmm_training"]
    require(gm["mixtures_per_state"] == 4 and gm["em_passes"] == 5 and gm["train_rows"] == 4376)
    require(gm["all_states_have_distinct_component_means"] and gm["parameter_shape"] == [138, 4, 39])
    proof = report["fresh_environment_restore"]
    require(proof["state"] == "passed" and proof["isolated_new_python_environment"] and not proof["system_site_packages"])
    require(proof["validation_transcripts_matching_exactly"] == 3)
    expanded_path = root / "results/final/fresh_environment_restoration.json"
    require(sha(expanded_path) == report["expanded_fresh_environment_report_sha256"])
    expanded = report["expanded_fresh_environment_restore"]
    require(expanded["state"] == "passed" and expanded["isolated_new_python_environment"])
    m45 = expanded["m45_restoration"]
    require(m45["all_22_child_jobs_passed"] and m45["restored_files_verified"] == 226)
    require((m45["selected_acoustic_models_executed"], m45["online_joint_models_executed"], m45["cached_controls_executed"]) == (3, 3, 15))
    require(not m45["references_read"] and not m45["test_audio_decoded"])
    print("OK: four follow-up systems, full validation/test counts, 107 x 3 timing, Qwen validation selection, mixture model and fresh restore evidence.")


if __name__ == "__main__":
    verify()
