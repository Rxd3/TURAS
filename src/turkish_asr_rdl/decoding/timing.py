"""The shared timing harness: warm-up, timed repeats, end-to-end and core-model RTF.

End-to-end time runs from before audio reading through result serialization; core
model time covers `adapter.transcribe` only. Every RTF is a sum of seconds over a
sum of audio seconds across the timed (non-warm-up) batches. Free VRAM is read before
the load and peak reserved VRAM after each repeat; a peak above the free memory, or any
timed error row, is named in the manifest limitations.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

from turkish_asr_rdl.decoding.audio import TARGET_SAMPLE_RATE_HZ, load_audio, resolve_audio_path
from turkish_asr_rdl.decoding.export import (
    CopiedHypothesisError,
    RunContext,
    adapter_result_fields,
    nbest_row,
    prediction_row,
    run_manifest,
)
from turkish_asr_rdl.decoding.hardware import probe_gpu, thermal_flag
from turkish_asr_rdl.decoding.interface import AsrAdapter, AudioItem, Hypothesis, InferenceInput, check_adapter
from turkish_asr_rdl.decoding.memory import VramMeter, VramRecord, default_vram, library_versions, spill_suspected
from turkish_asr_rdl.schemas.jsonl import encode_jsonl, write_json, write_jsonl

MIN_WARMUP_BATCHES = 3
END_TO_END_SCOPE = "audio read, decode and resample through result serialization"
CORE_MODEL_SCOPE = "adapter.transcribe only"
SPILL_LIMITATION = "RTF invalid: peak reserved VRAM exceeded free VRAM before load (suspected sysmem fallback)"
_MAX_MESSAGE_CHARS = 500


@dataclass(frozen=True)
class TimingRecord:
    """One timed batch; repeat 0 marks a warm-up batch."""

    timing_record_id: str
    run_id: str
    repeat: int
    utterance_ids: tuple[str, ...]
    audio_duration_s: float
    batch_size: int
    warmup: bool
    end_to_end_s: float
    core_model_s: float
    sync_method: str
    error_rows: int = 0


@dataclass(frozen=True)
class TimedRun:
    """Hypotheses from repeat 1 in input order, every timing record and the RTF summary."""

    hypotheses: tuple[Hypothesis, ...]
    records: tuple[TimingRecord, ...]
    cold_load_s: float
    end_to_end_rtf: float
    core_model_rtf: float
    per_repeat_end_to_end_rtf: tuple[float, ...]
    hardware: tuple[Mapping[str, Any], ...]
    thermal_flags: int
    vram: VramRecord = VramRecord()
    timed_error_rows: int = 0


def _no_sync() -> None:
    return None


def default_sync() -> tuple[Callable[[], None], str]:
    """torch.cuda.synchronize when torch imports and CUDA is available, else a no-op named "none"."""
    try:
        import torch
    except (ImportError, OSError):
        return _no_sync, "none"
    if torch.cuda.is_available():
        return torch.cuda.synchronize, "torch.cuda.synchronize"
    return _no_sync, "none"


def _batches(inputs: Sequence[InferenceInput], size: int) -> list[tuple[InferenceInput, ...]]:
    return [tuple(inputs[i:i + size]) for i in range(0, len(inputs), size)]


def _check_arguments(timed: Sequence[InferenceInput], warmup: Sequence[InferenceInput], repeats: int,
                     batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if not timed:
        raise ValueError("there are no timed inputs to measure")
    warmup_batches = math.ceil(len(warmup) / batch_size)
    if warmup_batches < MIN_WARMUP_BATCHES:
        raise ValueError(
            f"warm-up needs at least {MIN_WARMUP_BATCHES} batches, got {warmup_batches}; "
            "add warm-up inputs from train or validation"
        )
    shared = sorted({i.utterance_id for i in timed} & {i.utterance_id for i in warmup})
    if shared:
        raise ValueError(f"warm-up and timed inputs share utterance IDs: {', '.join(shared[:10])}")


def _one_line(text: str) -> str:
    """One line of valid UTF-8 text: characters UTF-8 cannot encode become backslash escapes."""
    writable = text.encode("utf-8", "backslashreplace").decode("utf-8")
    return " ".join(writable.split())[:_MAX_MESSAGE_CHARS]


def _error_hypotheses(items: Sequence[AudioItem], code: str, message: str) -> list[Hypothesis]:
    return [Hypothesis(item.utterance_id, "error", None, error_code=code, error_message=message) for item in items]


def _transcribe(adapter: AsrAdapter, items: Sequence[AudioItem]) -> list[Hypothesis]:
    try:
        hypotheses = list(adapter.transcribe(items))
    except Exception as exc:  # an adapter failure is recorded as error rows and the run continues
        return _error_hypotheses(items, "adapter_exception", _one_line(f"{type(exc).__name__}: {exc}"))
    returned = [h.utterance_id if isinstance(h, Hypothesis) else None for h in hypotheses]
    if returned != [item.utterance_id for item in items]:
        message = (
            f"adapter returned {len(hypotheses)} results for {len(items)} inputs, "
            "not one Hypothesis per input in order"
        )
        return _error_hypotheses(items, "adapter_output_mismatch", message)
    return hypotheses


def _serialized(hypothesis: Hypothesis, requested_n: int) -> Hypothesis:
    """Build and encode what the output files take from one result; one that cannot be written becomes an error row."""
    try:
        encode_jsonl([adapter_result_fields(hypothesis, requested_n)])
    except CopiedHypothesisError as exc:
        return _invalid_output(hypothesis, str(exc))
    except Exception as exc:  # any failure here would stop write_run_outputs after the whole timed run
        return _invalid_output(hypothesis, _one_line(f"adapter result cannot be written: {type(exc).__name__}: {exc}"))
    return hypothesis


def _invalid_output(hypothesis: Hypothesis, message: str) -> Hypothesis:
    return Hypothesis(hypothesis.utterance_id, "error", None, error_code="adapter_output_invalid",
                      error_message=message)


def _timed_batch(adapter: AsrAdapter, batch: Sequence[InferenceInput], audio_root: Path,
                 sync: Callable[[], None], clock: Callable[[], float]) -> tuple[list[Hypothesis], float, float, float]:
    """Run one batch; return hypotheses, end-to-end seconds, core-model seconds and audio seconds."""
    sync()
    t0 = clock()
    items = [AudioItem(i.utterance_id, load_audio(resolve_audio_path(audio_root, i.audio_uri))) for i in batch]
    sync()
    c0 = clock()
    hypotheses = _transcribe(adapter, items)
    sync()
    c1 = clock()
    hypotheses = [_serialized(h, adapter.requested_n) for h in hypotheses]
    sync()
    t1 = clock()
    audio_s = sum(len(item.samples) for item in items) / TARGET_SAMPLE_RATE_HZ
    return hypotheses, t1 - t0, c1 - c0, audio_s


def _rtf(records: Sequence[TimingRecord], seconds: Callable[[TimingRecord], float]) -> float:
    audio_s = sum(r.audio_duration_s for r in records)
    if audio_s <= 0:
        raise ValueError("the timed audio has zero duration, so RTF is undefined")
    return sum(seconds(r) for r in records) / audio_s


def _summarize(hypotheses: list[Hypothesis], records: list[TimingRecord], hardware: list[dict],
               cold_load_s: float, repeats: int, vram: VramRecord) -> TimedRun:
    timed = [r for r in records if not r.warmup]
    per_repeat = tuple(_rtf([r for r in timed if r.repeat == k], lambda r: r.end_to_end_s)
                       for k in range(1, repeats + 1))
    return TimedRun(
        hypotheses=tuple(hypotheses),
        records=tuple(records),
        cold_load_s=cold_load_s,
        end_to_end_rtf=_rtf(timed, lambda r: r.end_to_end_s),
        core_model_rtf=_rtf(timed, lambda r: r.core_model_s),
        per_repeat_end_to_end_rtf=per_repeat,
        hardware=tuple(hardware),
        thermal_flags=sum(1 for sample in hardware if thermal_flag(sample)),
        vram=vram,
        timed_error_rows=sum(r.error_rows for r in timed),
    )


@dataclass(frozen=True)
class _Batcher:
    """Runs and times one batch of one run."""

    adapter: AsrAdapter
    audio_root: Path
    sync: Callable[[], None]
    sync_name: str
    clock: Callable[[], float]
    run_id: str
    batch_size: int

    def run(self, batch: Sequence[InferenceInput], repeat: int,
            record_id: str) -> tuple[list[Hypothesis], TimingRecord]:
        hyps, end_to_end_s, core_s, audio_s = _timed_batch(self.adapter, batch, self.audio_root, self.sync,
                                                           self.clock)
        ids = tuple(i.utterance_id for i in batch)
        errors = sum(1 for h in hyps if h.status == "error")
        return hyps, TimingRecord(record_id, self.run_id, repeat, ids, audio_s, self.batch_size, repeat == 0,
                                  end_to_end_s, core_s, self.sync_name, errors)


def _cold_load(adapter: AsrAdapter, sync: Callable[[], None], clock: Callable[[], float]) -> float:
    sync()
    start = clock()
    adapter.load()
    sync()
    return clock() - start


def _timed_repeat(batcher: _Batcher, inputs: Sequence[InferenceInput], repeat: int, probe: Callable[[], dict],
                  meter: VramMeter) -> tuple[list[Hypothesis], list[TimingRecord], list[dict], float | None]:
    """One repeat between two GPU samples; the VRAM peak covers its batches only."""
    before = {"repeat": repeat, "position": "before", **probe()}
    meter.reset_peak()
    batches = enumerate(_batches(inputs, batcher.batch_size), start=1)
    results = [batcher.run(batch, repeat, f"{batcher.run_id}-r{repeat}-b{k:04d}") for k, batch in batches]
    peak = meter.peak_reserved_gib()
    after = {"repeat": repeat, "position": "after", **probe()}
    return [h for hyps, _ in results for h in hyps], [r for _, r in results], [before, after], peak


def run_timed(
    adapter: AsrAdapter,
    timed_inputs: Sequence[InferenceInput],
    warmup_inputs: Sequence[InferenceInput],
    *,
    audio_root: Path,
    run_id: str,
    repeats: int = 3,
    batch_size: int = 1,
    sync: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    probe: Callable[[], dict] = probe_gpu,
    memory: VramMeter | None = None,
) -> TimedRun:
    """Load once (cold load), run the warm-up batches, then time every repeat of the timed inputs."""
    _check_arguments(timed_inputs, warmup_inputs, repeats, batch_size)
    check_adapter(adapter)
    sync_fn, sync_name = default_sync() if sync is None else (sync, getattr(sync, "__name__", type(sync).__name__))
    meter = default_vram() if memory is None else memory
    free_gib, total_gib = meter.free_and_total_gib()
    cold_load_s = _cold_load(adapter, sync_fn, clock)
    batcher = _Batcher(adapter, Path(audio_root), sync_fn, sync_name, clock, run_id, batch_size)
    warmup_batches = enumerate(_batches(warmup_inputs, batch_size), start=1)
    records = [batcher.run(batch, 0, f"{run_id}-warmup-{k:04d}")[1] for k, batch in warmup_batches]
    hypotheses: list[Hypothesis] = []
    hardware: list[dict] = []
    peaks: list[float | None] = []
    for repeat in range(1, repeats + 1):
        hyps, repeat_records, samples, peak = _timed_repeat(batcher, timed_inputs, repeat, probe, meter)
        hypotheses.extend(hyps if repeat == 1 else [])
        records.extend(repeat_records)
        hardware.extend(samples)
        peaks.append(peak)
    vram = VramRecord(getattr(meter, "method", type(meter).__name__), total_gib, free_gib, tuple(peaks))
    return _summarize(hypotheses, records, hardware, cold_load_s, repeats, vram)


def _check_run_matches(run: TimedRun, inputs: Sequence[InferenceInput], ctx: RunContext, adapter: AsrAdapter) -> None:
    if [h.utterance_id for h in run.hypotheses] != [i.utterance_id for i in inputs]:
        raise ValueError("inputs do not match the run: give the timed inputs in the order they were run")
    if any(r.run_id != ctx.run_id for r in run.records):
        raise ValueError(f"the timing records belong to another run than {ctx.run_id!r}")
    provenance = (adapter.system_id, adapter.model_artifact_sha256, adapter.decoder_config_sha256)
    if provenance != (ctx.system_id, ctx.model_artifact_sha256, ctx.decoder_config_sha256):
        raise ValueError("the run context's system_id or artifact hashes differ from the adapter's")


def _environment(run: TimedRun) -> dict[str, Any]:
    gpu = next((s for s in run.hardware if s.get("ok")), {})
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "soundfile": sf.__version__,
        "soxr": soxr.__version__,
        "gpu_name": gpu.get("name"),
        "gpu_driver": gpu.get("driver_version"),
        **library_versions(),
    }


def _memory_fields(vram: VramRecord, repeat: int) -> dict[str, Any]:
    """The run's VRAM before load and the peak of this record's repeat (None for warm-up batches)."""
    peak = vram.peak_reserved_gib[repeat - 1] if 1 <= repeat <= len(vram.peak_reserved_gib) else None
    return {
        "memory_method": vram.memory_method,
        "vram_total_gib": vram.vram_total_gib,
        "vram_free_before_load_gib": vram.vram_free_before_load_gib,
        "peak_reserved_gib": peak,
        "vram_spill_suspected": spill_suspected(peak, vram.vram_free_before_load_gib),
    }


def _timing_rows(run: TimedRun, environment: Mapping[str, Any]) -> list[dict[str, Any]]:
    canonical = json.dumps(environment, sort_keys=True, ensure_ascii=False).encode("utf-8")
    shared = {
        "cold_load_s": run.cold_load_s,
        "scope": {"end_to_end": END_TO_END_SCOPE, "core_model": CORE_MODEL_SCOPE},
        "environment_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    batches = [{"record_type": "batch", **dataclasses.asdict(r), **shared, **_memory_fields(run.vram, r.repeat)}
               for r in run.records]
    return batches + [{"record_type": "hardware", **sample} for sample in run.hardware]


def _timing_summary(run: TimedRun) -> dict[str, Any]:
    first = run.records[0]
    return {
        "primary_rtf": "end_to_end_rtf",
        "end_to_end_rtf": run.end_to_end_rtf,
        "core_model_rtf": run.core_model_rtf,
        "per_repeat_end_to_end_rtf": list(run.per_repeat_end_to_end_rtf),
        "cold_load_s": run.cold_load_s,
        "warmup_batches": sum(1 for r in run.records if r.warmup),
        "repeats": max(r.repeat for r in run.records),
        "batch_size": first.batch_size,
        "thermal_flags": run.thermal_flags,
        "sync_method": first.sync_method,
        "scope": {"end_to_end": END_TO_END_SCOPE, "core_model": CORE_MODEL_SCOPE},
    }


def _limitations(run: TimedRun) -> list[str]:
    notes = []
    if not any(s.get("ok") for s in run.hardware):
        errors = sorted({str(s.get("error")) for s in run.hardware})
        notes.append(f"GPU state was not recorded: {'; '.join(errors)}")
    if run.thermal_flags:
        notes.append(f"thermal or hardware slowdown flagged in {run.thermal_flags} of {len(run.hardware)} GPU samples")
    if run.records[0].sync_method == "none":
        notes.append("no device synchronization was used (no CUDA device)")
    if run.vram.vram_spill_suspected:
        notes.append(SPILL_LIMITATION)
    if run.timed_error_rows:
        notes.append(f"RTF includes {run.timed_error_rows} timed error rows")
    return notes


def write_run_outputs(out_dir: Path, run: TimedRun, inputs: Sequence[InferenceInput], ctx: RunContext,
                      adapter: AsrAdapter, *, command: str) -> Mapping[str, str]:
    """Write predictions, N-best (when produced), timing and manifest files; return {file name: sha256}."""
    _check_run_matches(run, inputs, ctx, adapter)
    check_adapter(adapter)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = list(zip(run.hypotheses, inputs))
    hashes: dict[str, str] = {}
    nbest_sha = None
    if adapter.requested_n > 0:
        rows = [nbest_row(h, i, ctx, adapter.requested_n, adapter.score_convention_id) for h, i in pairs]
        nbest_sha = hashes["nbest.jsonl"] = write_jsonl(out / "nbest.jsonl", rows)
    record_ids = {uid: r.timing_record_id for r in run.records if r.repeat == 1 for uid in r.utterance_ids}
    predictions = [prediction_row(h, i, ctx, record_ids[i.utterance_id], nbest_sha) for h, i in pairs]
    hashes["predictions.jsonl"] = write_jsonl(out / "predictions.jsonl", predictions)
    environment = _environment(run)
    hashes["timing.jsonl"] = write_jsonl(out / "timing.jsonl", _timing_rows(run, environment))
    manifest = run_manifest(ctx, command=command, environment=environment, configuration=adapter.config,
                            outputs=dict(hashes), timing=_timing_summary(run), limitations=_limitations(run))
    return {**hashes, "manifest.json": write_json(out / "manifest.json", manifest)}
