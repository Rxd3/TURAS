"""5070 Ti smoke tests: one model per run, results in DIR/smoke/<NAME>.json, then one REPORT.md.

Usage:
    python experiments/m6-smoke/smoke.py --model NAME --artifacts DIR [--clips 20]
    python experiments/m6-smoke/smoke.py --summarize DIR

Clips come from one WorldSpeech tr_tr train shard at a pinned revision. They are a
feasibility sample: accuracy is never computed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

import adapters
import probes
import smoke_common
import smoke_report
from turkish_asr_rdl.decoding.audio import TARGET_SAMPLE_RATE_HZ, load_audio, resolve_audio_path, to_mono_16k
from turkish_asr_rdl.decoding.export import RunContext
from turkish_asr_rdl.decoding.interface import AudioItem, InferenceInput
from turkish_asr_rdl.decoding.timing import run_timed, write_run_outputs
from turkish_asr_rdl.schemas.jsonl import read_jsonl, sha256_file, write_json
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.schemas.predictions import SCHEMA_VERSION, STATUSES

INFERENCE_MODELS = ("whisper-small", "whisper-large-v3", "qwen3-asr-1.7b", "qwen3-asr-0.6b",
                    "xlsr-tr-cv7", "xlsr-tr-cv8")
TRAINING_MODELS = ("train-whisper-small-full", "train-whisper-large-v3-lora", "train-whisper-large-v3-full")
PYG_MODELS = ("pyg-hetero",)
MODEL_NAMES = INFERENCE_MODELS + TRAINING_MODELS + PYG_MODELS
WARMUP_CLIPS = 3
DEFAULT_CLIPS = 20
MAX_CLIP_S = 30.0
DATASET_REPO = "disco-eth/WorldSpeech"
DATASET_FILE = "data/tr_tr/train-00000.parquet"
DATASET_REVISION = "b76b85395039f50332fbc421f7c098ee2f2653d7"
DATASET_RELEASE = "worldspeech-tr_tr-b76b8539-train-00000-smoke"
SHARD_COLUMNS = ("segment_id", "audio", "human_transcript")
SELECTION_RULE = (f"first rows of {DATASET_FILE} in file order that decode, last more than 0 s and at most "
                  f"{MAX_CLIP_S:g} s, and have a transcript; the first {WARMUP_CLIPS} are warm-up")
_SAMPLE_CHARS = 80


@dataclass(frozen=True)
class Clip:
    """One decoded shard row: 16 kHz mono samples and its train-split transcript."""

    segment_id: str
    samples: np.ndarray = field(compare=False)
    text: str


def audio_bytes(value: Any) -> bytes:
    """The encoded audio of a shard cell: a {"bytes": ...} struct or raw bytes."""
    data = value.get("bytes") if isinstance(value, Mapping) else value
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("the audio cell holds no encoded bytes")
    return bytes(data)


def decode_clip(data: bytes) -> np.ndarray:
    """Decode encoded audio (Opus in the shard) to float32 mono 16 kHz."""
    samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return to_mono_16k(samples, rate)


def _usable(row: Mapping[str, Any]) -> tuple[Clip | None, str | None]:
    text = row.get("human_transcript")
    if not isinstance(text, str) or not text.strip():
        return None, "no_transcript"
    try:
        samples = decode_clip(audio_bytes(row.get("audio")))
    except (ValueError, sf.SoundFileError):
        return None, "undecodable"
    if not 0 < len(samples) <= MAX_CLIP_S * TARGET_SAMPLE_RATE_HZ:
        return None, "outside_duration"
    return Clip(str(row["segment_id"]), samples, text), None


def select_clips(rows: Iterable[Mapping[str, Any]], count: int) -> tuple[tuple[Clip, ...], dict[str, int]]:
    """The first `count` usable rows in order, and how many rows were skipped for each reason."""
    clips: list[Clip] = []
    skipped = dict.fromkeys(("undecodable", "outside_duration", "no_transcript"), 0)
    for row in rows:
        clip, reason = _usable(row)
        if reason is not None:
            skipped = {**skipped, reason: skipped[reason] + 1}
            continue
        clips.append(clip)
        if len(clips) == count:
            return tuple(clips), skipped
    raise ValueError(f"the shard has only {len(clips)} usable clips, {count} are needed")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_clips(clips: Sequence[Clip], audio_dir: Path) -> list[dict[str, Any]]:
    """Write each clip as 16 kHz FLAC; return its inference-input rows, sharing one manifest hash."""
    folder = Path(audio_dir)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for number, clip in enumerate(clips, start=1):
        name = f"clip-{number:02d}.flac"
        sf.write(folder / name, np.clip(clip.samples, -1.0, 1.0), TARGET_SAMPLE_RATE_HZ, format="FLAC",
                 subtype="PCM_16")
        written.append((clip, name, sha256_file(folder / name)))
    manifest_sha = _sha256_json([[clip.segment_id, sha] for clip, _, sha in written])
    return [
        {"schema_version": SCHEMA_VERSION, "dataset_release": DATASET_RELEASE, "split_manifest_sha256": manifest_sha,
         "utterance_id": clip.segment_id, "audio_uri": name, "audio_sha256": sha,
         "duration_s": len(clip.samples) / TARGET_SAMPLE_RATE_HZ, "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
         "channels": 1, "language": "tr"}
        for clip, name, sha in written
    ]


def shard_rows() -> Iterable[Mapping[str, Any]]:
    """Rows of the pinned train shard, read in batches (downloads it once into the Hugging Face cache)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=list(SHARD_COLUMNS)):
        yield from batch.to_pylist()


def _prepare(audio_dir: Path, count: int, rows_source: Callable[[], Iterable[Mapping[str, Any]]]) -> dict:
    clips, skipped = select_clips(rows_source(), count + WARMUP_CLIPS)
    rows = write_clips(clips, audio_dir)
    manifest = {
        "source": {"repo": DATASET_REPO, "file": DATASET_FILE, "revision": DATASET_REVISION, "split": "train"},
        "selection": SELECTION_RULE, "skipped": skipped,
        "warmup": rows[:WARMUP_CLIPS], "timed": rows[WARMUP_CLIPS:],
    }
    write_json(audio_dir / "labels.json", {clip.segment_id: clip.text for clip in clips})
    write_json(audio_dir / "clips.json", manifest)
    return manifest


def load_clips(audio_dir: Path, count: int, rows_source: Callable[[], Iterable[Mapping[str, Any]]] = shard_rows,
               ) -> tuple[tuple[InferenceInput, ...], tuple[InferenceInput, ...]]:
    """Warm-up and timed inputs; the clips are prepared on first use or when the timed count changes."""
    folder = Path(audio_dir)
    path = folder / "clips.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if manifest is None or len(manifest["timed"]) != count:
        manifest = _prepare(folder, count, rows_source)
    warmup = tuple(InferenceInput.from_row(row) for row in manifest["warmup"])
    timed = tuple(InferenceInput.from_row(row) for row in manifest["timed"])
    return warmup, timed


def load_labels(audio_dir: Path) -> dict[str, str]:
    """Train-split transcripts by utterance ID; read only by the training probes."""
    return json.loads((Path(audio_dir) / "labels.json").read_text(encoding="utf-8"))


def nbest_facts(rows: Sequence[Mapping[str, Any]], requested_n: int) -> dict[str, Any]:
    """Candidate counts, raw-text duplicates and the validate_nbest verdict for exported N-best rows."""
    if requested_n == 0:
        return {"requested_n": 0}
    report = validate_nbest(rows)
    counts = [len(row["candidates"]) for row in rows]
    duplicates = [len(texts) - len(set(texts)) for texts in ([c["text"] for c in row["candidates"]] for row in rows)]
    return {
        "requested_n": requested_n, "rows": len(rows),
        "min_candidates": min(counts, default=0), "max_candidates": max(counts, default=0),
        "utterances_all_distinct": sum(1 for d in duplicates if d == 0), "duplicate_texts": sum(duplicates),
        "validate_nbest_ok": report.ok, "validation_errors": len(report.errors),
        "first_errors": list(report.errors[:3]),
    }


def _context(adapter: Any, timed: Sequence[InferenceInput]) -> RunContext:
    first = timed[0]
    return RunContext(adapter.system_id, adapter.system_id, first.dataset_release, "train",
                      first.split_manifest_sha256, adapter.model_artifact_sha256, adapter.decoder_config_sha256)


def measure_inference(adapter: Any, warmup: Sequence[InferenceInput], timed: Sequence[InferenceInput], *,
                      audio_dir: Path, run_dir: Path, vram: Any, command: str,
                      timing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load, warm up and time the adapter through run_timed; export and check its outputs."""
    vram.reset()
    run = run_timed(adapter, timed, warmup, audio_root=audio_dir, run_id=adapter.system_id, repeats=1,
                    batch_size=1, **dict(timing or {}))
    memory = vram.peak()
    outputs = write_run_outputs(run_dir, run, timed, _context(adapter, timed), adapter, command=command)
    rows = read_jsonl(Path(run_dir) / "nbest.jsonl") if "nbest.jsonl" in outputs else []
    items = [AudioItem(i.utterance_id, load_audio(resolve_audio_path(audio_dir, i.audio_uri))) for i in timed]
    statuses = Counter(h.status for h in run.hypotheses)
    timed_records = [r for r in run.records if not r.warmup]
    return {
        "kind": "inference", "ok": statuses["error"] == 0, "config": dict(adapter.config),
        "clips": {"warmup": len(warmup), "timed": len(timed)},
        "timed_audio_s": sum(r.audio_duration_s for r in timed_records), "repeats": 1, "batch_size": 1,
        "load_s": run.cold_load_s, **memory,
        "end_to_end_rtf": run.end_to_end_rtf, "core_model_rtf": run.core_model_rtf,
        "sync_method": timed_records[0].sync_method, "thermal_flags": run.thermal_flags,
        "statuses": {status: statuses[status] for status in STATUSES},
        "first_error": next((h.error_message for h in run.hypotheses if h.status == "error"), None),
        "nbest": nbest_facts(rows, adapter.requested_n),
        "sample_hypotheses": [h.text[:_SAMPLE_CHARS] for h in run.hypotheses if h.status == "ok"][:2],
        "outputs": dict(outputs),
        **adapter.extra_facts(items),
    }


def run_model(name: str, artifacts: Path, clips: int, command: str) -> dict[str, Any]:
    """Run one smoke test on the GPU and return its facts."""
    root = Path(artifacts) / "smoke"
    audio_dir = root / "audio"
    vram = smoke_common.CudaVram()
    if name in PYG_MODELS:
        facts = probes.pyg_probe(vram)
        return {"kind": "pyg", "ok": facts["outcome"] == "completed", **facts}
    warmup, timed = load_clips(audio_dir, clips)
    if name in TRAINING_MODELS:
        labels = load_labels(audio_dir)
        pairs = [(load_audio(resolve_audio_path(audio_dir, i.audio_uri)), labels[i.utterance_id]) for i in timed]
        return {"kind": "training", "ok": True, **probes.train_probe(name, pairs, vram)}
    adapter = adapters.build_adapter(name)
    return measure_inference(adapter, warmup, timed, audio_dir=audio_dir, run_dir=root / "runs" / name,
                             vram=vram, command=command)


def _environment() -> dict[str, Any]:
    try:
        return smoke_common.environment()
    except Exception as exc:  # versions are context only; a missing library is itself the answer
        return {"error": smoke_common.one_line(exc)}


def collect(name: str, artifacts: Path, clips: int, *, command: str,
            runner: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run one smoke test and return its record; any failure becomes ok false with the full traceback."""
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    began = time.perf_counter()
    try:
        facts = (runner or run_model)(name, artifacts, clips, command)
    except Exception as exc:  # the overnight run must leave a result file for every model
        facts = {"ok": False, "error": smoke_common.one_line(exc), "traceback": traceback.format_exc()}
    return {"name": name, **facts, "command": command, "started_utc": started,
            "wall_s": time.perf_counter() - began, "environment": _environment()}


def finite_json(value: Any) -> Any:
    """A copy where NaN and infinities become null, so strict JSON can hold it; tuples become lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def render_report(records: Mapping[str, Mapping[str, Any]]) -> str:
    """REPORT.md text: one table per kind; models without a result file are listed as not run."""
    return smoke_report.render(records, INFERENCE_MODELS, TRAINING_MODELS, PYG_MODELS)


def summarize(artifacts: Path) -> Path:
    """Read every DIR/smoke/<NAME>.json and write DIR/smoke/REPORT.md."""
    folder = Path(artifacts) / "smoke"
    if not folder.is_dir():
        raise FileNotFoundError(f"no smoke results folder at {folder}")
    records: dict[str, Any] = {}
    for name in MODEL_NAMES:
        path = folder / f"{name}.json"
        if path.is_file():
            try:
                records[name] = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                records[name] = {"name": name, "unreadable": smoke_common.one_line(exc)}
    target = folder / "REPORT.md"
    target.write_text(render_report(records), encoding="utf-8", newline="\n")
    return target


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="smoke.py", description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", choices=MODEL_NAMES, help="the one model or probe to run")
    mode.add_argument("--summarize", metavar="DIR", type=Path, help="write DIR/smoke/REPORT.md from the results")
    parser.add_argument("--artifacts", metavar="DIR", type=Path, help="artifact folder (needed with --model)")
    parser.add_argument("--clips", type=_positive_int, default=DEFAULT_CLIPS,
                        help=f"timed clips (default {DEFAULT_CLIPS}); {WARMUP_CLIPS} warm-up clips are added")
    args = parser.parse_args(argv)
    if args.model and args.artifacts is None:
        parser.error("--model needs --artifacts DIR")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize is not None:
        try:
            print(f"wrote {summarize(args.summarize)}")
        except OSError as exc:
            print(f"cannot write the report: {smoke_common.one_line(exc)}")
            return 1
        return 0
    command = " ".join([Path(sys.executable).name, *sys.argv])
    record = collect(args.model, args.artifacts, args.clips, command=command)
    path = Path(args.artifacts) / "smoke" / f"{args.model}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, finite_json(record))
    except (OSError, ValueError) as exc:
        print(f"{args.model}: cannot write the result file: {smoke_common.one_line(exc)}")
        return 1
    verdict = "ok" if record["ok"] else f"FAILED: {record.get('error') or record.get('first_error') or 'see JSON'}"
    print(f"{args.model}: {verdict} -> {path}")
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
