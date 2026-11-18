"""REPORT.md for the 5070 Ti smoke tests: one table per kind of check, no transcript text."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_NOTE_CHARS = 200

INTRO = (
    "Clips: WorldSpeech `tr_tr` train shard `train-00000` at a pinned revision, 3 warm-up clips and the "
    "timed clips, each at most 30 s. This is a feasibility check: accuracy is not measured. Each model "
    "ran alone in its own process, batch 1, one timed repeat. Inference peak VRAM is "
    "`torch.cuda.max_memory_allocated` over the timed repeat: the timing harness resets the peak counters when "
    "the repeat starts, so model load and warm-up peaks are not in it (the loaded weights are). An inference "
    "row whose N-best cell has no actual_n counts comes from an older result file, whose peak runs from just "
    "before the model load to the end of the timed run. Training and PyG peaks run from just before the model "
    "load to the end of the probe."
)


def _number(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"


def _cell(text: Any) -> str:
    flat = " ".join(str(text).split())[:_NOTE_CHARS]
    return flat.replace("|", "\\|")


def _row(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _status(record: Mapping[str, Any] | None) -> str | None:
    """A result word for records that did not produce numbers, else None."""
    if record is None:
        return "not run"
    if "unreadable" in record:
        return "unreadable"
    if not record.get("ok"):
        return "FAILED"
    return None


def _failure_note(record: Mapping[str, Any] | None) -> str:
    if record is None:
        return ""
    return record.get("unreadable") or record.get("error") or record.get("first_error") or ""


def _kept_part(nbest: Mapping[str, Any]) -> str:
    """Kept-candidate counts; result files written before the export deduplicated have none."""
    if "actual_n_mean" not in nbest:
        return ""
    return (f"actual_n mean {_number(nbest.get('actual_n_mean'), 2)}, min {nbest.get('actual_n_min')}, "
            f"{nbest.get('rows_with_dedup_map')}/{nbest.get('rows')} rows with dedup_map, ")


def _nbest_cell(nbest: Any) -> str:
    if not isinstance(nbest, Mapping) or not nbest.get("requested_n"):
        return "none"
    verdict = "valid" if nbest.get("validate_nbest_ok") else f"invalid ({nbest.get('validation_errors')} errors)"
    finite = "scores finite" if nbest.get("acoustic_log_scores_finite") else "scores NOT all finite"
    return (f"{nbest.get('min_candidates')}-{nbest.get('max_candidates')} of {nbest['requested_n']}, "
            f"{_kept_part(nbest)}{nbest.get('utterances_all_distinct')}/{nbest.get('rows')} distinct, "
            f"{verdict}, {finite}")


def _inference_notes(record: Mapping[str, Any]) -> str:
    notes = [str(record.get("decode_path", ""))]
    trap = record.get("whisper_generate_trap")
    if isinstance(trap, Mapping) and "error" in trap:
        notes.append(f"Whisper generate() check failed: {trap['error']}")
    elif isinstance(trap, Mapping):
        notes.append(f"Whisper generate() with the same beams: {trap.get('duplicate_texts')} duplicate texts "
                     f"in {trap.get('utterances_with_duplicates')} of {trap.get('utterances')} utterances")
    if record.get("beam_error"):
        notes.append(f"beam error: {record['beam_error']}")
    if record.get("first_error"):
        notes.append(f"first error: {record['first_error']}")
    return "; ".join(n for n in notes if n)


def _inference_table(names: Sequence[str], records: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["## Inference", "",
             "| Model | Result | Precision | Load s | Peak VRAM GiB | E2E RTF | Core RTF | N-best | Notes |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        record = records.get(name)
        status = _status(record)
        if status is not None:
            lines.append(_row([name, status, "-", "-", "-", "-", "-", "-", _failure_note(record)]))
            continue
        config = record.get("config")
        precision = config.get("precision", "-") if isinstance(config, Mapping) else "-"
        lines.append(_row([name, "ok", precision, _number(record.get("load_s"), 2),
                           _number(record.get("peak_vram_gib"), 2), _number(record.get("end_to_end_rtf")),
                           _number(record.get("core_model_rtf")), _nbest_cell(record.get("nbest")),
                           _inference_notes(record)]))
    return lines


def _training_table(names: Sequence[str], records: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["## Training probes (3 AdamW steps, bf16, gradient checkpointing)", "",
             "| Probe | Result | Batch | Peak alloc GiB | Peak reserved GiB | Free before GiB | Fits | s/step "
             "| Notes |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        record = records.get(name)
        status = _status(record)
        if status is not None:
            lines.append(_row([name, status, "-", "-", "-", "-", "-", "-", _failure_note(record)]))
            continue
        outcome = "out of memory" if record.get("outcome") == "out_of_memory" else record.get("outcome", "-")
        fits = "yes" if record.get("fits_in_dedicated_vram") else "no"
        lines.append(_row([name, outcome, record.get("batch_size", "-"), _number(record.get("peak_vram_gib"), 2),
                           _number(record.get("peak_reserved_gib"), 2), _number(record.get("free_vram_before_gib"), 2),
                           fits, _number(record.get("seconds_per_step"), 2), record.get("oom_message", "")]))
    return lines


def _check_cell(check: Any) -> str:
    if not isinstance(check, Mapping):
        return "-"
    return "ok" if check.get("ok") else f"failed: {check.get('error', 'non-finite loss or gradients')}"


def _pyg_table(names: Sequence[str], records: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["## PyTorch Geometric (2 HeteroConv SAGEConv layers, forward and backward on CUDA)", "",
             "| Probe | Result | Version | fp32 | bf16 | Notes |", "|---|---|---|---|---|---|"]
    for name in names:
        record = records.get(name)
        result = _status(record) or str(record.get("outcome", "ok"))
        if result in ("not run", "unreadable") or "error" in (record or {}):
            lines.append(_row([name, result, "-", "-", "-", _failure_note(record)]))
            continue
        lines.append(_row([name, result, record.get("torch_geometric", "-"), _check_cell(record.get("fp32")),
                           _check_cell(record.get("bf16")), ""]))
    return lines


def render(records: Mapping[str, Mapping[str, Any]], inference: Sequence[str], training: Sequence[str],
           pyg: Sequence[str]) -> str:
    """The whole report; models without a result file are listed as not run."""
    sections = [["# 5070 Ti smoke report", "", INTRO], _inference_table(inference, records),
                _training_table(training, records), _pyg_table(pyg, records)]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"
