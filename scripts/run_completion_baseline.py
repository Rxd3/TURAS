"""Reference-free full-split baseline export through the shared timed harness."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

from turkish_asr_rdl.decoding.interface import InferenceInput
from turkish_asr_rdl.decoding.export import RunContext
from turkish_asr_rdl.decoding.timing import run_timed, write_run_outputs
from turkish_asr_rdl.decoding.memory import NoVram
from turkish_asr_rdl.schemas.jsonl import read_jsonl, write_json
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.schemas.nbest import validate_nbest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "turas-tr-econ-legis-v0.1"
BASE = ROOT / "artifacts/completion-2026-11-22"
SPLITS = ROOT / "data/splits" / RELEASE
AUDIO = ROOT / "data/private" / RELEASE / "inference" / RELEASE
CONFIGS = {"b2": "whisper_large_v3", "qwen17": "qwen3_asr_1_7b", "qwen06": "qwen3_asr_0_6b"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def inputs(split):
    path = SPLITS / f"{split}.txt"
    manifest = json.loads((SPLITS / "split_manifest.json").read_text())
    if sha(path) != manifest["splits"][split]["sha256"]:
        raise ValueError("Frozen split file changed")
    ids = path.read_text().splitlines()
    rows = read_jsonl(AUDIO / f"{RELEASE}.inference.{split}.jsonl")
    if [r["utterance_id"] for r in rows] != ids:
        raise ValueError("Frozen inference coverage/order mismatch")
    return [InferenceInput.from_row(r) for r in rows]


def adapter_for(system):
    if system in CONFIGS:
        import torch
        torch.set_num_threads(4)
        config = json.loads((ROOT / "configs/baselines" / (CONFIGS[system] + ".json")).read_text())
        if system == "b2":
            from turkish_asr_rdl.baselines.whisper import WhisperBaselineAdapter as Adapter
        else:
            from turkish_asr_rdl.baselines.qwen import QwenBaselineAdapter as Adapter
        return Adapter(config, checkpoint_path=BASE / "models" / system, device="cuda", precision="bfloat16")
    if system == "seda":
        from turkish_asr_rdl.baselines.seda_transducer import SedaTransducerAdapter
        return SedaTransducerAdapter(BASE / "models/seda")
    if system == "gmm":
        from turkish_asr_rdl.baselines.mixture_hmm import MixtureHmmAdapter
        return MixtureHmmAdapter(BASE / "gmm/checkpoint.npz")
    raise ValueError(system)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--system", choices=[*CONFIGS, "seda", "gmm"], required=True)
    parser.add_argument("--mode", choices=["probe", "validation", "test", "timing"], required=True)
    args = parser.parse_args()
    out = BASE / "runs" / args.system / args.mode
    out.mkdir(parents=True, exist_ok=True)
    done_file = out / "completed.json"
    if done_file.exists():
        done = json.loads(done_file.read_text())
        for name, digest in done["files"].items():
            if sha(out / name) != digest:
                raise ValueError("Completed baseline output changed")
        print(f"ALREADY COMPLETE {args.system} {args.mode}", flush=True)
        return
    state_file = out / "status.json"
    state = {"state": "starting", "system": args.system, "mode": args.mode, "pid": os.getpid(), "started_at_utc": now()}
    write_json(state_file, state)
    adapter = adapter_for(args.system)
    split = "validation" if args.mode in ["probe", "validation"] else "test"
    rows = inputs(split)
    if args.mode == "probe":
        rows = rows[:3]
    if args.mode == "timing":
        by_id = {i.utterance_id: i for i in rows}
        rows = [by_id[uid] for uid in (SPLITS / "m45-timing-30min.txt").read_text().splitlines()]
    warmup = inputs("train")[:3]
    frozen = {"state": "frozen_before_this_run", "frozen_at_utc": now(), "system": args.system, "mode": args.mode,
              "config": adapter.config, "model_artifact_sha256": adapter.model_artifact_sha256,
              "decoder_config_sha256": adapter.decoder_config_sha256, "split_manifest_sha256": sha(SPLITS / "split_manifest.json"),
              "script_sha256": sha(__file__), "python": sys.version, "python_executable": sys.executable,
              "platform": platform.platform(), "parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "environment_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
              "limitations": ["Follow-up working benchmark after earlier project test exposure; no blind/preregistered holdout claim",
                              "No reference text is read by this inference command", "Unknown training overlap is separately audited"]}
    write_json(out / "frozen-run.json", frozen)
    original_transcribe = adapter.transcribe
    counter = 0

    def progress(batch):
        nonlocal counter
        result = original_transcribe(batch)
        counter += len(batch)
        if counter % 10 == 0:
            state.update(state="inference", items_including_warmup=counter, updated_at_utc=now())
            write_json(state_file, state)
            print(f"{args.system} {args.mode}: {counter} items including warmups", flush=True)
        return result

    adapter.transcribe = progress
    run_id = f"completion-{args.system}-{args.mode}-20261122"
    kwargs = {}
    if args.system in ["seda", "gmm"]:
        def cpu_sync():
            pass
        kwargs = {"sync": cpu_sync, "memory": NoVram()}
    run = run_timed(adapter, rows, warmup, audio_root=AUDIO, run_id=run_id, repeats=3 if args.mode == "timing" else 1,
                    batch_size=1, **kwargs)
    ctx = RunContext(run_id, adapter.system_id, RELEASE, split, frozen["split_manifest_sha256"],
                     adapter.model_artifact_sha256, adapter.decoder_config_sha256)
    files = write_run_outputs(out, run, rows, ctx, adapter, command="scripts/run_completion_baseline.py")
    ids = [r.utterance_id for r in rows]
    reports = []
    for validator, filename in [(validate_predictions, "predictions.jsonl"), (validate_nbest, "nbest.jsonl")]:
        if filename == "nbest.jsonl" and adapter.requested_n == 0:
            continue
        report = validator(read_jsonl(out / filename), ids)
        if not report.ok:
            raise ValueError(report.errors)
        reports.append({"file": filename, "rows": len(ids), "passed": True})
    errors = sum(h.status == "error" for h in run.hypotheses)
    if errors:
        raise RuntimeError(f"{errors} failed inference rows; inspect outputs")
    done = {"state": "completed", "system_id": adapter.system_id, "rows": len(rows), "files": files,
            "contract_validation": reports, "rtf": run.end_to_end_rtf, "repeats": 3 if args.mode == "timing" else 1,
            "thermal_flags": run.thermal_flags, "vram_spill_suspected": run.vram.vram_spill_suspected,
            "completed_at_utc": now(), "timing_claim": "headline serial frozen subset" if args.mode == "timing" else "accuracy/probe only"}
    write_json(done_file, done)
    state.update(state="completed", rows=len(rows), ended_at_utc=now())
    write_json(state_file, state)
    print(f"COMPLETE {args.system} {args.mode}: {len(rows)} rows, RTF={run.end_to_end_rtf:.6f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
