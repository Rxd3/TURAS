"""Complete remaining jobs when verified model downloads become ready.

PIDs and exit codes are authoritative; no failed job is silently restarted.
All headline timing runs are serial and wait for downloads/training to finish.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from completion_job_state import write_job_state
from run_completion_baseline import sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/completion-2026-11-22"
NEURAL = BASE / "envs/neural/Scripts/python.exe"
TRANSDUCER = BASE / "envs/transducer/Scripts/python.exe"
LOGS = BASE / "jobs/dispatcher"


def matching_processes(script):
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["name"].lower() == "python.exe" and any(Path(part).name == script for part in p.info["cmdline"] or []):
                found.append(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def main():
    if matching_processes("run_completion_baseline.py") or matching_processes("score_completion_study.py"):
        raise RuntimeError("An inference/scoring process is still live; inspect it before starting another controller")
    LOGS.mkdir(parents=True, exist_ok=True)
    state = {"state": "running", "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": []}
    state_path = LOGS / "status.json"

    def save():
        write_job_state(state_path, state)

    def command(name, python, args):
        with (LOGS / f"{name}.log").open("a", encoding="utf-8") as stdout, (LOGS / f"{name}-errors.log").open("a", encoding="utf-8") as stderr:
            environment = dict(os.environ, PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
            child = subprocess.Popen([str(python), "-u", *args], cwd=ROOT, stdout=stdout, stderr=stderr, env=environment)
            job = {"name": name, "pid": child.pid, "state": "running", "command": [str(python), *args]}
            state["jobs"].append(job)
            state.update(state="running", current_job=name)
            save()
            print(f"START {name} PID {child.pid}", flush=True)
            code = child.wait()
            job.update(state="completed" if code == 0 else "failed", exit_code=code)
            save()
            if code:
                state["state"] = "failed"
                save()
                raise RuntimeError(f"Job {name} failed with code {code}; inspect retained logs")

    def inference(system, mode):
        python = TRANSDUCER if system == "seda" else NEURAL
        command(f"{system}-{mode}", python, [str(ROOT / "scripts/run_completion_baseline.py"), "--system", system, "--mode", mode])

    def score(system, split):
        output = BASE / "scores" / split / f"{system}.json"
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            run = BASE / "runs" / system / split
            done = json.loads((run / "completed.json").read_text(encoding="utf-8"))
            for name, digest in done["files"].items():
                if sha(run / name) != digest:
                    raise ValueError("Completed inference changed before reuse")
            expected_rows = 438 if split == "validation" else 636
            ref = ROOT / "data/private/turas-tr-econ-legis-v0.1" / (
                "validation-recovered-minutes-v1/references.validation.jsonl" if split == "validation" else "evaluator-final/references.test.jsonl")
            if (existing["state"] != "completed" or existing["rows"] != expected_rows
                    or existing["predictions_sha256"] != sha(run / "predictions.jsonl")
                    or existing["nbest_sha256"] != (sha(run / "nbest.jsonl") if (run / "nbest.jsonl").exists() else None)
                    or not existing["independent_jiwer_crosscheck"].startswith("passed")
                    or existing["reference_sha256"] != sha(ref)):
                raise ValueError("Existing score report does not match completed outputs/references")
            if split == "test":
                timing = BASE / "runs" / system / "timing"
                timing_done = json.loads((timing / "completed.json").read_text(encoding="utf-8"))
                if existing["timing_manifest_sha256"] != sha(timing / "manifest.json"):
                    raise ValueError("Existing test score has different timing evidence")
                for name, digest in timing_done["files"].items():
                    if sha(timing / name) != digest:
                        raise ValueError("Completed timing changed before reuse")
            print(f"REUSE VERIFIED score {system} {split}", flush=True)
            return
        command(f"score-{system}-{split}", NEURAL, [str(ROOT / "scripts/score_completion_study.py"), "--systems", system, "--split", split])

    save()
    pending = ["b2", "qwen17", "qwen06"]
    while pending:
        ready = next((s for s in pending if (BASE / "models" / s / "verified-inventory.json").exists()), None)
        if ready:
            inference(ready, "probe")
            inference(ready, "validation")
            score(ready, "validation")
            pending.remove(ready)
            continue
        downloaders = matching_processes("download_completion_models.py")
        if not downloaders:
            raise RuntimeError(f"No live model downloader and missing inventories: {pending}")
        state.update(state="waiting_for_verified_model", pending_models=pending, live_downloader_pids=downloaders)
        save()
        time.sleep(10)
    selected_path = BASE / "b3-selection.json"
    if selected_path.exists():
        selected_record = json.loads(selected_path.read_text(encoding="utf-8"))
        reports = {}
        for key, digest in selected_record["validation_report_sha256"].items():
            report_path = BASE / "scores/validation" / f"{key}.json"
            if sha(report_path) != digest:
                raise ValueError("Frozen Qwen selection report changed; do not select again after test")
            reports[key] = json.loads(report_path.read_text(encoding="utf-8"))
        best = min(reports, key=lambda s: (reports[s]["metrics"]["tr-normalized-v1"]["cer"], reports[s]["metrics"]["tr-normalized-v1"]["wer"], s))
        if best != selected_record["selected_system"]:
            raise ValueError("Frozen Qwen choice does not follow the validation criterion")
        print("REUSE VERIFIED frozen Qwen selection", flush=True)
    else:
        command("select-qwen", NEURAL, [str(ROOT / "scripts/score_completion_study.py"), "--select-qwen"])
    selected = json.loads((BASE / "b3-selection.json").read_text())["selected_system"]
    while not (BASE / "gmm/model-manifest.json").exists():
        training = matching_processes("train_mixture_hmm.py")
        if not training:
            raise RuntimeError("GMM training has no live process and no completed manifest")
        state.update(state="waiting_for_gmm_training", live_training_pids=training)
        save()
        time.sleep(10)
    inference("gmm", "probe")
    inference("gmm", "validation")
    score("gmm", "validation")
    while not (BASE / "runs/seda/test/completed.json").exists():
        live = matching_processes("run_completion_sequence.py")
        if not live:
            raise RuntimeError("Seda accuracy sequence stopped before complete test export")
        state.update(state="waiting_for_seda_accuracy", live_sequence_pids=live)
        save()
        time.sleep(10)
    score("seda", "validation")
    systems = ["b2", selected, "seda", "gmm"]
    for system in systems:
        inference(system, "test")
    while matching_processes("download_completion_models.py") or matching_processes("train_mixture_hmm.py"):
        state.update(state="waiting_for_serial_timing", live_downloaders=matching_processes("download_completion_models.py"),
                     live_trainers=matching_processes("train_mixture_hmm.py"))
        save()
        time.sleep(10)
    for system in systems:
        inference(system, "timing")
    for system in systems:
        score(system, "test")
    state.update(state="completed", systems=systems, ended_at_utc=datetime.now(timezone.utc).isoformat())
    save()
    print("BASELINE STUDY JOBS COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
