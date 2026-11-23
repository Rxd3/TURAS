"""Replay frozen follow-up models to a new directory without modifying sealed runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from turkish_asr_rdl.decoding.export import RunContext
from turkish_asr_rdl.decoding.memory import NoVram
from turkish_asr_rdl.decoding.timing import run_timed, write_run_outputs
from turkish_asr_rdl.schemas.jsonl import read_jsonl, write_json
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.schemas.nbest import validate_nbest
from run_completion_baseline import BASE, ROOT, AUDIO, SPLITS, RELEASE, adapter_for, inputs, now, sha


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--system", required=True, choices=["b2", "qwen17", "qwen06", "seda", "gmm"])
    p.add_argument("--mode", required=True, choices=["validation", "test", "timing"])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--validation-clips", type=int, help="Optional reference-free validation smoke subset")
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists():
        p.error("Choose a new output directory; sealed or previous results are never overwritten")
    if args.validation_clips is not None and (args.mode != "validation" or not 1 <= args.validation_clips <= 438):
        p.error("--validation-clips is only valid for a 1..438 clip validation smoke check")
    frozen_path = BASE / "runs" / args.system / args.mode / "frozen-run.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    adapter = adapter_for(args.system)
    if adapter.model_artifact_sha256 != frozen["model_artifact_sha256"] or adapter.decoder_config_sha256 != frozen["decoder_config_sha256"]:
        raise ValueError("Replay model/decoder differs from the original frozen run")
    split = "validation" if args.mode == "validation" else "test"
    rows = inputs(split)
    if args.validation_clips:
        rows = rows[:args.validation_clips]
    if args.mode == "timing":
        by_id = {r.utterance_id: r for r in rows}
        rows = [by_id[uid] for uid in (SPLITS / "m45-timing-30min.txt").read_text().splitlines()]
    out.mkdir(parents=True)
    write_json(out / "replay-configuration.json", {"state": "frozen_before_replay", "created_at_utc": now(),
        "source_run_sha256": sha(frozen_path), "replay_script_sha256": sha(__file__), "reference_text_read": False,
        "rows": len(rows), "mode": args.mode, "system": args.system, "config": adapter.config})
    kwargs = {}
    if args.system in ["seda", "gmm"]:
        def cpu_sync():
            pass
        kwargs.update(sync=cpu_sync, memory=NoVram())
    run_id = f"replay-{args.system}-{args.mode}"
    run = run_timed(adapter, rows, inputs("train")[:3], audio_root=AUDIO, run_id=run_id,
                    repeats=3 if args.mode == "timing" else 1, batch_size=1, **kwargs)
    ctx = RunContext(run_id, adapter.system_id, RELEASE, split, sha(SPLITS / "split_manifest.json"),
                     adapter.model_artifact_sha256, adapter.decoder_config_sha256)
    files = write_run_outputs(out, run, rows, ctx, adapter, command="scripts/replay_completion_baseline.py")
    ids = [r.utterance_id for r in rows]
    if not validate_predictions(read_jsonl(out / "predictions.jsonl"), ids).ok:
        raise ValueError("Replay predictions violate the shared contract")
    if adapter.requested_n and not validate_nbest(read_jsonl(out / "nbest.jsonl"), ids).ok:
        raise ValueError("Replay N-best violates the shared contract")
    if any(h.status == "error" for h in run.hypotheses):
        raise ValueError("Replay includes failed inference")
    archived = read_jsonl(BASE / "runs" / args.system / args.mode / "predictions.jsonl")
    original = {r["utterance_id"]: r["hypothesis_text"] for r in archived}
    predictions = read_jsonl(out / "predictions.jsonl")
    matches = sum(r["hypothesis_text"] == original[r["utterance_id"]] for r in predictions)
    write_json(out / "completed.json", {"state": "completed", "rows": len(rows), "files": files,
        "exact_archived_transcript_matches": matches, "reference_text_read": False,
        "timing_claim": "repeated subset; compare hardware/environment before speed claims" if args.mode == "timing" else "functional replay only"})
    print(f"REPLAY COMPLETE: {matches}/{len(rows)} exact archived transcript matches; no references read")


if __name__ == "__main__":
    main()
