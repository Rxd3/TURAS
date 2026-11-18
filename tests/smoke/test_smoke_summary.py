"""5070 Ti smoke report and --summarize. SYNTHETIC records only."""
import json

import pytest


def _records(smoke):
    inference_ok = {"name": "whisper-small", "kind": "inference", "ok": True, "config": {"precision": "float16"},
                    "load_s": 2.5, "peak_vram_gib": 1.25, "end_to_end_rtf": 0.05, "core_model_rtf": 0.04,
                    "nbest": {"requested_n": 10, "min_candidates": 10, "utterances_all_distinct": 20, "rows": 20,
                              "duplicate_texts": 0, "validate_nbest_ok": True},
                    "decode_path": "beam 10", "sample_hypotheses": ["SYNTHETIC hidden text"],
                    "whisper_generate_trap": {"duplicate_texts": 37, "utterances_with_duplicates": 9}}
    inference_failed = {"name": "qwen3-asr-1.7b", "kind": "inference", "ok": False, "error": "RuntimeError: SYN"}
    training = {"name": "train-whisper-small-full", "kind": "training", "ok": True, "outcome": "completed",
                "batch_size": 8, "peak_vram_gib": 7.5, "total_vram_gib": 11.9, "fits_in_dedicated_vram": True,
                "seconds_per_step": 1.2}
    oom = {"name": "train-whisper-large-v3-full", "kind": "training", "ok": True, "outcome": "out_of_memory",
           "batch_size": 1, "peak_vram_gib": 11.5, "total_vram_gib": 11.9, "fits_in_dedicated_vram": False,
           "oom_message": "OutOfMemoryError: SYNTHETIC"}
    pyg = {"name": "pyg-hetero", "kind": "pyg", "ok": True, "outcome": "fp32 only", "torch_geometric": "2.8.1",
           "fp32": {"ok": True}, "bf16": {"ok": False, "error": "RuntimeError: SYN bf16"}}
    return {r["name"]: r for r in (inference_ok, inference_failed, training, oom, pyg)}


def test_render_report_lists_every_model_and_hides_hypotheses(smoke):
    report = smoke.render_report(_records(smoke))
    for name in smoke.MODEL_NAMES:
        assert name in report
    assert "not run" in report and "FAILED" in report and "out of memory" in report
    assert "SYNTHETIC hidden text" not in report
    assert "37" in report and "RuntimeError: SYN bf16" in report and "fp32 only" in report


def test_summarize_reads_results_and_writes_the_report(smoke, tmp_path):
    folder = tmp_path / "smoke"
    folder.mkdir()
    for name, record in _records(smoke).items():
        (folder / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    (folder / "xlsr-tr-cv7.json").write_text("{broken", encoding="utf-8")
    path = smoke.summarize(tmp_path)
    assert path == folder / "REPORT.md"
    text = path.read_text(encoding="utf-8")
    assert "unreadable" in text and "whisper-small" in text


def test_main_summarize_prints_the_report_path(smoke, tmp_path, capsys):
    (tmp_path / "smoke").mkdir()
    assert smoke.main(["--summarize", str(tmp_path)]) == 0
    assert "REPORT.md" in capsys.readouterr().out


def test_main_summarize_reports_a_missing_folder(smoke, tmp_path, capsys):
    assert smoke.main(["--summarize", str(tmp_path / "missing")]) == 1
    out = capsys.readouterr().out.strip()
    assert "\n" not in out and "missing" in out


@pytest.mark.parametrize("text", ["[1, 2]", "null", '"SYNTHETIC"', "7"])
def test_summarize_marks_json_that_is_not_an_object_unreadable(smoke, tmp_path, text):
    folder = tmp_path / "smoke"
    folder.mkdir()
    (folder / "whisper-small.json").write_text(text, encoding="utf-8")
    report = smoke.summarize(tmp_path).read_text(encoding="utf-8")
    row = next(line for line in report.splitlines() if line.startswith("| whisper-small |"))
    assert "| unreadable |" in row and "not a JSON object" in row


def test_summarize_marks_a_file_it_cannot_read_unreadable(smoke, tmp_path, monkeypatch, capsys):
    folder = tmp_path / "smoke"
    folder.mkdir()
    (folder / "whisper-small.json").write_text("{}", encoding="utf-8")
    real_read_text = type(folder).read_text

    def read_text(path, *args, **kwargs):
        if path.name == "whisper-small.json":
            raise PermissionError(13, "SYNTHETIC access denied", str(path))
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(folder), "read_text", read_text)
    assert smoke.main(["--summarize", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "cannot write the report" not in out and "REPORT.md" in out
    row = next(line for line in (folder / "REPORT.md").read_text(encoding="utf-8").splitlines()
               if line.startswith("| whisper-small |"))
    assert "| unreadable |" in row and "PermissionError" in row


def test_main_summarize_names_a_missing_folder_without_blaming_the_write(smoke, tmp_path, capsys):
    assert smoke.main(["--summarize", str(tmp_path / "missing")]) == 1
    out = capsys.readouterr().out.strip()
    assert out == f"cannot summarize: no smoke results folder at {tmp_path / 'missing' / 'smoke'}"


def test_main_summarize_reports_a_failed_write(smoke, tmp_path, capsys):
    (tmp_path / "smoke" / "REPORT.md").mkdir(parents=True)
    assert smoke.main(["--summarize", str(tmp_path)]) == 1
    out = capsys.readouterr().out.strip()
    assert "\n" not in out and out.startswith("cannot write the report: ")


@pytest.mark.parametrize(("field", "value", "column", "cell"), [
    ("nbest", [1], 7, "none"), ("nbest", "SYN", 7, "none"), ("nbest", 7, 7, "none"),
    ("config", None, 2, "-"), ("config", [1], 2, "-"), ("config", "SYN", 2, "-"),
])
def test_summarize_treats_an_nbest_or_config_that_is_not_an_object_as_absent(smoke, tmp_path, field, value,
                                                                              column, cell):
    folder = tmp_path / "smoke"
    folder.mkdir()
    record = {**_records(smoke)["whisper-small"], field: value}
    (folder / "whisper-small.json").write_text(json.dumps(record), encoding="utf-8")
    report = smoke.summarize(tmp_path).read_text(encoding="utf-8")
    row = next(line for line in report.splitlines() if line.startswith("| whisper-small |"))
    cells = row[2:-2].split(" | ")
    assert cells[1] == "ok" and cells[3] == "2.50" and cells[column] == cell
