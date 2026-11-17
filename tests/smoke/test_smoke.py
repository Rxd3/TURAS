"""5070 Ti smoke script: clip preparation, inference record, report. SYNTHETIC tones and text only."""
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from turkish_asr_rdl.decoding.interface import Candidate, Hypothesis, InferenceInput
from turkish_asr_rdl.schemas.jsonl import sha256_file
from turkish_asr_rdl.schemas.predictions import validate_inference_inputs

SMOKE_DIR = Path(__file__).resolve().parents[2] / "experiments" / "m6-smoke"
REPO_ROOT = SMOKE_DIR.parents[1]
NO_GPU = {"ok": False, "error": "SYNTHETIC no gpu"}


def tone(rate: int, seconds: float, channels: int = 1, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(round(rate * seconds))) / rate
    mono = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return mono if channels == 1 else np.stack([mono] * channels, axis=1)


def encoded(data: np.ndarray, rate: int, fmt: str, subtype: str) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, data, rate, format=fmt, subtype=subtype)
    return buffer.getvalue()


def shard_row(segment_id: str, audio, text="SYNTHETIC text") -> dict:
    return {"segment_id": segment_id, "audio": audio, "human_transcript": text}


def good_row(segment_id: str, seconds: float = 1.0) -> dict:
    return shard_row(segment_id, {"bytes": encoded(tone(16000, seconds), 16000, "FLAC", "PCM_16"), "path": None})


def synthetic_clips(smoke, count: int, seconds: float = 1.0):
    return tuple(smoke.Clip(f"syn_{k}", tone(16000, seconds), f"SYNTHETIC text {k}") for k in range(count))


class FakeVram:
    """The peak grows once the adapter's extra facts (the Whisper generate() check) have run."""

    def __init__(self):
        self.resets = 0
        self.after_extra_facts = False

    def reset(self):
        self.resets += 1

    def peak(self):
        allocated, reserved = (9.0, 9.5) if self.after_extra_facts else (1.5, 2.0)
        return {"peak_vram_gib": allocated, "peak_reserved_gib": reserved, "free_vram_before_gib": 10.5,
                "total_vram_gib": 12.0}


class FakeAdapter:
    """SYNTHETIC two-candidate adapter that fails on the IDs it is given."""

    system_id = "smoke-synthetic"
    model_artifact_sha256 = "b" * 64
    decoder_config_sha256 = "a" * 64
    requested_n = 2
    score_convention_id = "smoke-synthetic-v1"

    def __init__(self, fail_ids=(), vram=None):
        self.config = {"repo_id": "synthetic/model", "revision": "c" * 40, "precision": "float16"}
        self.fail_ids = set(fail_ids)
        self.vram = vram

    def load(self):
        return None

    def transcribe(self, batch):
        if any(item.utterance_id in self.fail_ids for item in batch):
            raise RuntimeError("SYNTHETIC decode failure")
        return [self._hyp(item.utterance_id) for item in batch]

    def _hyp(self, uid):
        candidates = tuple(
            Candidate(f"{uid}-c{k}", k, f"SYNTHETIC {uid} v{k}", (k, 2), -1.0 * k, "seq2seq_token_log_prob_sum")
            for k in (1, 2)
        )
        return Hypothesis(uid, "ok", candidates[0].text, candidates)

    def extra_facts(self, items):
        if self.vram is not None:
            self.vram.after_extra_facts = True
        return {"decode_path": "beam 2", "items_seen": len(items)}


def test_help_runs_without_torch_or_pyarrow():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    done = subprocess.run([sys.executable, str(SMOKE_DIR / "smoke.py"), "--help"], capture_output=True,
                          text=True, env=env, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert "--model" in done.stdout and "pyg-hetero" in done.stdout and "--summarize" in done.stdout


def test_audio_bytes_reads_struct_and_raw_bytes(smoke):
    assert smoke.audio_bytes({"bytes": b"abc", "path": "x.opus"}) == b"abc"
    assert smoke.audio_bytes(b"abc") == b"abc"


@pytest.mark.parametrize("value", [None, {"bytes": None}, {"path": "x"}, "abc", {"bytes": b""}])
def test_audio_bytes_rejects_missing_audio(smoke, value):
    with pytest.raises(ValueError):
        smoke.audio_bytes(value)


def test_decode_clip_turns_synthetic_opus_into_16k_mono(smoke):
    samples = smoke.decode_clip(encoded(tone(24000, 1.0, channels=2), 24000, "OGG", "OPUS"))
    assert samples.dtype == np.float32 and samples.ndim == 1
    assert abs(len(samples) - 16000) <= 800


def test_select_clips_skips_bad_long_and_untranscribed_rows(smoke):
    rows = [
        shard_row("bad", {"bytes": b"not audio"}),
        good_row("long", seconds=smoke.MAX_CLIP_S + 1),
        shard_row("untranscribed", good_row("x")["audio"], text=""),
        good_row("keep_1"),
        good_row("keep_2"),
        good_row("never_read"),
    ]
    consumed = []

    def source():
        for row in rows:
            consumed.append(row["segment_id"])
            yield row

    clips, skipped = smoke.select_clips(source(), 2)
    assert [c.segment_id for c in clips] == ["keep_1", "keep_2"]
    assert skipped == {"undecodable": 1, "outside_duration": 1, "no_transcript": 1}
    assert "never_read" not in consumed


def test_select_clips_keeps_exactly_max_clip_s_and_skips_zero_samples(smoke):
    empty = shard_row("empty", {"bytes": encoded(np.zeros(0, dtype=np.float32), 16000, "WAV", "PCM_16")})
    clips, skipped = smoke.select_clips([empty, good_row("longest", seconds=smoke.MAX_CLIP_S), good_row("keep")], 2)
    assert [c.segment_id for c in clips] == ["longest", "keep"]
    assert len(clips[0].samples) == smoke.MAX_CLIP_S * 16000
    assert skipped == {"undecodable": 0, "outside_duration": 1, "no_transcript": 0}


def test_select_clips_fails_when_the_shard_has_too_few(smoke):
    with pytest.raises(ValueError, match="only 1 usable"):
        smoke.select_clips([good_row("only")], 2)


def test_write_clips_rows_pass_the_inference_allowlist(smoke, tmp_path):
    loud = smoke.Clip("loud", tone(16000, 0.5, amplitude=1.5), "SYNTHETIC loud")
    clips = synthetic_clips(smoke, 2) + (loud,)
    rows = smoke.write_clips(clips, tmp_path / "audio")
    assert validate_inference_inputs(rows).ok
    assert [r["utterance_id"] for r in rows] == ["syn_0", "syn_1", "loud"]
    assert len({r["split_manifest_sha256"] for r in rows}) == 1
    for row in rows:
        path = tmp_path / "audio" / row["audio_uri"]
        info = sf.info(path)
        assert (info.samplerate, info.channels, info.format) == (16000, 1, "FLAC")
        assert sha256_file(path) == row["audio_sha256"]
    loud_samples, _ = sf.read(tmp_path / "audio" / rows[2]["audio_uri"])
    assert np.max(np.abs(loud_samples)) <= 1.0
    assert np.max(np.abs(loud.samples)) == pytest.approx(1.5, abs=1e-3)


def test_load_clips_prepares_once_then_reuses(smoke, tmp_path):
    calls = []

    def source():
        calls.append(1)
        return [good_row(f"seg_{k}") for k in range(8)]

    warmup, timed = smoke.load_clips(tmp_path, 2, rows_source=source)
    assert len(warmup) == smoke.WARMUP_CLIPS and len(timed) == 2
    assert all(isinstance(i, InferenceInput) for i in warmup + timed)
    assert {i.utterance_id for i in warmup}.isdisjoint(i.utterance_id for i in timed)
    labels = smoke.load_labels(tmp_path)
    assert set(labels) == {i.utterance_id for i in warmup + timed}
    again = smoke.load_clips(tmp_path, 2, rows_source=source)
    assert again == (warmup, timed) and len(calls) == 1
    smoke.load_clips(tmp_path, 3, rows_source=source)
    assert len(calls) == 2


def _prepared(smoke, tmp_path, count=2):
    audio_dir = tmp_path / "smoke" / "audio"
    source = lambda: [good_row(f"seg_{k}") for k in range(smoke.WARMUP_CLIPS + count)]  # noqa: E731
    return audio_dir, *smoke.load_clips(audio_dir, count, rows_source=source)


def test_measure_inference_with_a_fake_adapter(smoke, tmp_path):
    audio_dir, warmup, timed = _prepared(smoke, tmp_path)
    vram = FakeVram()
    record = smoke.measure_inference(FakeAdapter(vram=vram), warmup, timed, audio_dir=audio_dir,
                                     run_dir=tmp_path / "run", vram=vram, command="SYNTHETIC",
                                     timing={"probe": lambda: NO_GPU})
    assert record["ok"] is True and record["kind"] == "inference"
    assert record["statuses"] == {"ok": 2, "empty": 0, "error": 0}
    assert record["nbest"]["validate_nbest_ok"] is True and record["nbest"]["rows"] == 2
    assert record["peak_vram_gib"] == 1.5 and record["peak_reserved_gib"] == 2.0 and vram.resets == 1
    assert record["free_vram_before_gib"] == 10.5 and vram.after_extra_facts is True
    assert record["clips"]["warmup"] == 3 and record["clips"]["timed"] == 2
    assert record["timed_audio_s"] == pytest.approx(2.0)
    assert record["load_s"] >= 0 and record["end_to_end_rtf"] > 0
    assert record["decode_path"] == "beam 2" and record["items_seen"] == 2
    assert len(record["sample_hypotheses"]) == 2
    assert {"predictions.jsonl", "nbest.jsonl", "timing.jsonl", "manifest.json"} <= set(record["outputs"])
    assert (tmp_path / "run" / "manifest.json").is_file()


def test_measure_inference_marks_error_rows_as_failed(smoke, tmp_path):
    audio_dir, warmup, timed = _prepared(smoke, tmp_path)
    adapter = FakeAdapter(fail_ids={timed[0].utterance_id})
    record = smoke.measure_inference(adapter, warmup, timed, audio_dir=audio_dir, run_dir=tmp_path / "run",
                                     vram=FakeVram(), command="SYNTHETIC", timing={"probe": lambda: NO_GPU})
    assert record["ok"] is False and record["statuses"]["error"] == 1
    assert "SYNTHETIC decode failure" in record["first_error"]


def test_measure_inference_keeps_the_numbers_when_a_later_step_fails(smoke, tmp_path):
    class LateFailure(FakeAdapter):
        def extra_facts(self, items):
            raise RuntimeError("SYNTHETIC extra facts failure")

    audio_dir, warmup, timed = _prepared(smoke, tmp_path)
    record = smoke.measure_inference(LateFailure(), warmup, timed, audio_dir=audio_dir, run_dir=tmp_path / "run",
                                     vram=FakeVram(), command="SYNTHETIC", timing={"probe": lambda: NO_GPU})
    assert record["ok"] is False and record["error"] == "RuntimeError: SYNTHETIC extra facts failure"
    assert "Traceback" in record["traceback"]
    assert record["end_to_end_rtf"] > 0 and record["peak_vram_gib"] == 1.5 and record["statuses"]["ok"] == 2


def test_score_range_flags_missing_or_non_finite_scores(smoke):
    assert smoke._score_range([-1.0, None])["acoustic_log_scores_finite"] is False
    assert smoke._score_range([-1.0, float("nan")])["acoustic_log_scores_finite"] is False
    assert smoke._score_range([]) == {"acoustic_log_score_min": None, "acoustic_log_score_max": None,
                                      "acoustic_log_scores_finite": True}


def test_finite_json_replaces_non_finite_numbers_without_mutating(smoke):
    original = {"a": [1.0, math.nan, {"b": math.inf}], "c": "x", "d": (2, -math.inf)}
    cleaned = smoke.finite_json(original)
    assert cleaned == {"a": [1.0, None, {"b": None}], "c": "x", "d": [2, None]}
    assert math.isnan(original["a"][1]) and original["a"][2]["b"] == math.inf


def test_collect_records_a_failure_with_its_traceback(smoke, tmp_path):
    def broken(name, artifacts, clips, command):
        raise RuntimeError("SYNTHETIC broken\nsecond line")

    record = smoke.collect("pyg-hetero", tmp_path, 20, command="SYNTHETIC", runner=broken)
    assert record["ok"] is False and record["name"] == "pyg-hetero"
    assert record["error"] == "RuntimeError: SYNTHETIC broken second line"
    assert "Traceback" in record["traceback"]
    assert record["wall_s"] >= 0 and record["started_utc"].endswith("+00:00")


def test_main_writes_the_record_and_returns_its_exit_code(smoke, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(smoke, "run_model", lambda name, artifacts, clips, command: {"kind": "pyg", "ok": True})
    assert smoke.main(["--model", "pyg-hetero", "--artifacts", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "smoke" / "pyg-hetero.json").read_text(encoding="utf-8"))
    assert saved["ok"] is True and saved["name"] == "pyg-hetero"
    assert capsys.readouterr().out.strip().startswith("pyg-hetero: ok")

    monkeypatch.setattr(smoke, "run_model", lambda name, artifacts, clips, command: {"kind": "pyg", "ok": False})
    assert smoke.main(["--model", "pyg-hetero", "--artifacts", str(tmp_path)]) == 1
    out = capsys.readouterr().out.strip()
    assert out.startswith("pyg-hetero: FAILED") and "\n" not in out


@pytest.mark.parametrize("argv", [
    ["--model", "pyg-hetero"],
    ["--model", "pyg-hetero", "--artifacts", "x", "--clips", "0"],
    ["--model", "unknown", "--artifacts", "x"],
    [],
])
def test_main_rejects_bad_arguments(smoke, argv):
    with pytest.raises(SystemExit) as caught:
        smoke.main(argv)
    assert caught.value.code == 2
