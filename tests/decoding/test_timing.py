"""Hardware probe and timing harness. Fake adapter, fake clock, SYNTHETIC tones only."""
import itertools
import json
import subprocess
import sys
import types
from dataclasses import replace as dataclasses_replace

import pytest

import turkish_asr_rdl.decoding.timing as timing_module
from turkish_asr_rdl.decoding.audio import AudioLoadError
from turkish_asr_rdl.decoding.hardware import NVIDIA_SMI_FIELDS, probe_gpu, thermal_flag
from turkish_asr_rdl.decoding.interface import Candidate, Hypothesis
from turkish_asr_rdl.decoding.timing import MIN_WARMUP_BATCHES, default_sync, run_timed, write_run_outputs
from turkish_asr_rdl.schemas.jsonl import read_jsonl, sha256_file
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.schemas.predictions import validate_predictions

NO_GPU = {"ok": False, "error": "no gpu"}
WARM = ("warm_1", "warm_2", "warm_3")
TIMED = ("utt_a", "utt_b")


class FakeAdapter:
    system_id = "sys_synthetic"
    model_artifact_sha256 = "b" * 64
    decoder_config_sha256 = "a" * 64

    def __init__(self, requested_n=0, fail_ids=(), broken_output=False):
        self.requested_n = requested_n
        self.score_convention_id = "nbest-synthetic-v1" if requested_n else None
        self.config = {"beam": max(requested_n, 1)}
        self.fail_ids = set(fail_ids)
        self.broken_output = broken_output
        self.loads = 0
        self.calls = 0

    def load(self):
        self.loads += 1

    def transcribe(self, batch):
        self.calls += 1
        if self.broken_output:
            return []
        if any(item.utterance_id in self.fail_ids for item in batch):
            raise RuntimeError("SYNTHETIC out of memory\nsecond line")
        return [self._hyp(item.utterance_id) for item in batch]

    def _hyp(self, uid):
        text = f"SYNTHETIC {uid} call {self.calls}"
        candidates = tuple(
            Candidate(f"c{k}", k, f"{text} v{k}", (k, k + 1), -1.0 * k, "seq2seq_teacher_forced_log_prob")
            for k in range(1, self.requested_n + 1)
        )
        return Hypothesis(uid, "ok", candidates[0].text if candidates else text, candidates)


def fake_clock(step=0.5):
    ticks = itertools.count(0.0, step)
    return lambda: next(ticks)


@pytest.fixture
def setup(tmp_path, write_tone, make_input):
    """Write SYNTHETIC tones: warm-up clips of 2 s and timed clips of 1 s."""

    def _setup(warm=WARM, timed=TIMED):
        for uid in warm:
            write_tone(f"audio/{uid}.flac", 16000, 2.0)
        for uid in timed:
            write_tone(f"audio/{uid}.flac", 16000, 1.0)
        return [make_input(u, 1.0) for u in timed], [make_input(u, 2.0) for u in warm]

    return _setup


def run(adapter, timed, warm, tmp_path, **kwargs):
    options = {"run_id": "run_synthetic", "clock": fake_clock(), "probe": lambda: dict(NO_GPU), "sync": lambda: None}
    return run_timed(adapter, timed, warm, audio_root=tmp_path, **{**options, **kwargs})


def test_rtf_is_sum_over_sum_and_excludes_warmup(setup, tmp_path):
    timed, warm = setup()
    result = run(FakeAdapter(), timed, warm, tmp_path, repeats=2)
    assert result.end_to_end_rtf == pytest.approx(1.5)
    assert result.core_model_rtf == pytest.approx(0.5)
    assert result.per_repeat_end_to_end_rtf == pytest.approx((1.5, 1.5))
    assert result.cold_load_s == pytest.approx(0.5)
    assert all(not r.warmup for r in result.records if r.repeat > 0)
    assert [r.repeat for r in result.records] == [0, 0, 0, 1, 1, 2, 2]
    assert all(r.warmup and r.audio_duration_s == pytest.approx(2.0) for r in result.records[:3])


def test_batches_share_one_timing_record(setup, tmp_path):
    timed, warm = setup(warm=[f"warm_{i}" for i in range(6)], timed=("utt_a", "utt_b", "utt_c", "utt_d"))
    result = run(FakeAdapter(), timed, warm, tmp_path, repeats=1, batch_size=2)
    timed_records = [r for r in result.records if not r.warmup]
    assert [r.utterance_ids for r in timed_records] == [("utt_a", "utt_b"), ("utt_c", "utt_d")]
    assert all(r.batch_size == 2 and r.audio_duration_s == pytest.approx(2.0) for r in timed_records)
    assert result.end_to_end_rtf == pytest.approx(0.75)


@pytest.mark.parametrize("n_warm,batch_size", [(2, 1), (4, 2), (0, 1)])
def test_fewer_than_three_warmup_batches_refused(setup, tmp_path, n_warm, batch_size):
    timed, warm = setup(warm=[f"warm_{i}" for i in range(n_warm)])
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="at least 3"):
        run(adapter, timed, warm, tmp_path, batch_size=batch_size)
    assert adapter.loads == 0 and MIN_WARMUP_BATCHES == 3


def test_three_warmup_batches_from_a_partial_batch_are_enough(setup, tmp_path):
    timed, warm = setup(warm=[f"warm_{i}" for i in range(5)])
    result = run(FakeAdapter(), timed, warm, tmp_path, repeats=1, batch_size=2)
    assert sum(r.warmup for r in result.records) == 3


def test_overlapping_warmup_and_timed_ids_refused(setup, tmp_path, make_input):
    timed, warm = setup()
    with pytest.raises(ValueError, match="utt_a"):
        run(FakeAdapter(), timed, [*warm, make_input("utt_a")], tmp_path)


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"batch_size": 0}, {"repeats": -1}])
def test_bad_repeats_or_batch_size_refused(setup, tmp_path, kwargs):
    timed, warm = setup()
    with pytest.raises(ValueError):
        run(FakeAdapter(), timed, warm, tmp_path, **kwargs)


def test_no_timed_inputs_refused(setup, tmp_path):
    _, warm = setup()
    with pytest.raises(ValueError, match="timed"):
        run(FakeAdapter(), [], warm, tmp_path)


def test_adapter_exception_becomes_error_rows(setup, tmp_path):
    timed, warm = setup()
    result = run(FakeAdapter(fail_ids={"utt_b"}), timed, warm, tmp_path, repeats=1)
    ok, failed = result.hypotheses
    assert ok.status == "ok" and failed.status == "error" and failed.text is None
    assert failed.error_code == "adapter_exception"
    assert "out of memory" in failed.error_message and "\n" not in failed.error_message
    assert len(result.records) == 5


def test_adapter_output_mismatch_becomes_error_rows(setup, tmp_path):
    timed, warm = setup()
    result = run(FakeAdapter(broken_output=True), timed, warm, tmp_path, repeats=1)
    assert {h.error_code for h in result.hypotheses} == {"adapter_output_mismatch"}
    assert [h.utterance_id for h in result.hypotheses] == list(TIMED)


def test_hypotheses_from_first_repeat_in_input_order(setup, tmp_path):
    timed, warm = setup(timed=("utt_c", "utt_a", "utt_b"))
    result = run(FakeAdapter(), timed, warm, tmp_path, repeats=3)
    assert [h.utterance_id for h in result.hypotheses] == ["utt_c", "utt_a", "utt_b"]
    assert [h.text for h in result.hypotheses] == [f"SYNTHETIC {u} call {k}" for k, u in [(4, "utt_c"), (5, "utt_a"), (6, "utt_b")]]


def test_probe_runs_before_and_after_each_repeat(setup, tmp_path):
    timed, warm = setup()
    samples = iter([dict(NO_GPU), {"ok": True, "clocks_event_reasons.hw_thermal_slowdown": "Active"}] * 3)
    result = run(FakeAdapter(), timed, warm, tmp_path, repeats=3, probe=lambda: next(samples))
    assert [(s["repeat"], s["position"]) for s in result.hardware] == [
        (1, "before"), (1, "after"), (2, "before"), (2, "after"), (3, "before"), (3, "after")]
    assert result.thermal_flags == 3


def test_sync_brackets_every_timed_region(setup, tmp_path):
    timed, warm = setup()
    calls = []
    result = run(FakeAdapter(), timed, warm, tmp_path, repeats=1, sync=lambda: calls.append(1))
    assert len(calls) == 2 + 4 * (3 + 2)
    assert {r.sync_method for r in result.records} == {"<lambda>"}


def test_default_sync_is_used_when_none_given(setup, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(timing_module, "default_sync", lambda: (lambda: calls.append(1), "fake.sync"))
    timed, warm = setup()
    result = run(FakeAdapter(), timed, warm, tmp_path, repeats=1, sync=None)
    assert calls and {r.sync_method for r in result.records} == {"fake.sync"}


def test_default_sync_without_torch_is_a_no_op(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    fn, name = default_sync()
    assert name == "none" and fn() is None


@pytest.mark.parametrize("available,expected", [(True, "torch.cuda.synchronize"), (False, "none")])
def test_default_sync_with_torch(monkeypatch, available, expected):
    synchronize = lambda: None  # noqa: E731
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: available, synchronize=synchronize))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    fn, name = default_sync()
    assert name == expected and (fn is synchronize) == available


def test_missing_audio_stops_the_run_with_a_friendly_error(setup, tmp_path, make_input):
    timed, warm = setup()
    with pytest.raises(AudioLoadError, match="utt_gone"):
        run(FakeAdapter(), [*timed, make_input("utt_gone")], warm, tmp_path, repeats=1)


def test_probe_without_nvidia_smi():
    def raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    r = probe_gpu(run=raise_file_not_found)
    assert r["ok"] is False and "nvidia-smi" in r["error"]


def test_probe_timeout_is_ten_seconds():
    seen = {}

    def slow(args, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    r = probe_gpu(run=slow)
    assert r["ok"] is False and "10 s" in r["error"] and seen["timeout"] == 10


def test_probe_other_os_error():
    def denied(*args, **kwargs):
        raise PermissionError(13, "Access is denied")

    assert probe_gpu(run=denied) == {"ok": False, "error": "nvidia-smi could not start: Access is denied"}


def test_probe_non_zero_exit():
    def failing(args, **kwargs):
        return subprocess.CompletedProcess(args, 9, stdout="", stderr="Field is not valid\nmore\n")

    r = probe_gpu(run=failing)
    assert r["ok"] is False and "9" in r["error"] and "Field is not valid" in r["error"] and "\n" not in r["error"]


def test_probe_parses_first_gpu():
    values = ["SYNTHETIC GPU", "591.84", "2100", "3090", "35.20", "61", "0x0000000000000004",
              "Not Active", "Not Active", "Not Active", "Active"]
    seen = {}

    def fake(args, **kwargs):
        seen["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout=", ".join(values) + "\n" + "second gpu\n", stderr="")

    r = probe_gpu(run=fake)
    assert r == {"ok": True, **dict(zip(NVIDIA_SMI_FIELDS, values))}
    assert seen["args"][0] == "nvidia-smi" and "--query-gpu=" + ",".join(NVIDIA_SMI_FIELDS) in seen["args"]
    assert not thermal_flag(r)


@pytest.mark.parametrize("stdout", ["", "a, b, c\n"])
def test_probe_unexpected_output(stdout):
    r = probe_gpu(run=lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=""))
    assert r["ok"] is False and "nvidia-smi" in r["error"]


def test_sw_power_cap_does_not_flag():
    assert not thermal_flag({"clocks_event_reasons.sw_power_cap": "Active"})


@pytest.mark.parametrize("field", ["clocks_event_reasons.hw_slowdown", "clocks_event_reasons.hw_thermal_slowdown",
                                   "clocks_event_reasons.sw_thermal_slowdown"])
def test_hw_thermal_flags(field):
    assert thermal_flag({field: "Active"})
    assert not thermal_flag({field: "Not Active"})


def test_written_outputs_pass_validators(setup, tmp_path, run_context):
    timed, warm = setup(timed=("utt_a", "utt_b", "utt_c"))
    adapter = FakeAdapter(requested_n=3, fail_ids={"utt_c"})
    result = run(adapter, timed, warm, tmp_path, repeats=2)
    hashes = write_run_outputs(tmp_path / "out", result, timed, run_context, adapter, command="SYNTHETIC cmd")
    out = tmp_path / "out"
    assert set(hashes) == {"predictions.jsonl", "nbest.jsonl", "timing.jsonl", "manifest.json"}
    assert all(sha256_file(out / name) == digest for name, digest in hashes.items())
    predictions = read_jsonl(out / "predictions.jsonl")
    assert validate_predictions(predictions, expected_ids=sorted(u.utterance_id for u in timed)).ok
    assert validate_nbest(read_jsonl(out / "nbest.jsonl")).ok
    assert {p["nbest_artifact_sha256"] for p in predictions} == {hashes["nbest.jsonl"]}
    repeat_one = {r.utterance_ids[0]: r.timing_record_id for r in result.records if r.repeat == 1}
    assert [p["timing_record_id"] for p in predictions] == [repeat_one[u] for u in ("utt_a", "utt_b", "utt_c")]


def test_timing_file_and_manifest_contents(setup, tmp_path, run_context):
    timed, warm = setup()
    adapter = FakeAdapter()
    result = run(adapter, timed, warm, tmp_path, repeats=2)
    hashes = write_run_outputs(tmp_path / "out", result, timed, run_context, adapter, command="SYNTHETIC cmd")
    assert "nbest.jsonl" not in hashes
    rows = read_jsonl(tmp_path / "out" / "timing.jsonl")
    batches = [r for r in rows if r["record_type"] == "batch"]
    assert len(batches) == len(result.records) and len(rows) - len(batches) == len(result.hardware)
    assert {"cold_load_s", "sync_method", "scope", "environment_sha256", "utterance_ids"} <= set(batches[0])
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"] == {k: v for k, v in hashes.items() if k != "manifest.json"}
    timing = manifest["evaluation_timing"]
    assert timing["end_to_end_rtf"] == pytest.approx(1.5) and timing["core_model_rtf"] == pytest.approx(0.5)
    assert timing["warmup_batches"] == 3 and timing["repeats"] == 2 and timing["batch_size"] == 1
    assert timing["thermal_flags"] == 0 and timing["cold_load_s"] == pytest.approx(0.5)
    assert any("no gpu" in item for item in manifest["limitations"])


def test_write_refuses_inputs_that_do_not_match_the_run(setup, tmp_path, run_context):
    timed, warm = setup()
    adapter = FakeAdapter()
    result = run(adapter, timed, warm, tmp_path, repeats=1)
    with pytest.raises(ValueError, match="inputs"):
        write_run_outputs(tmp_path / "out", result, timed[::-1], run_context, adapter, command="x")


def test_write_refuses_context_of_another_run(setup, tmp_path, run_context):
    import dataclasses

    timed, warm = setup()
    adapter = FakeAdapter()
    result = run(adapter, timed, warm, tmp_path, repeats=1)
    for change in ({"run_id": "other"}, {"system_id": "other"}, {"model_artifact_sha256": "c" * 64}):
        with pytest.raises(ValueError):
            write_run_outputs(tmp_path / "out", result, timed, dataclasses.replace(run_context, **change), adapter,
                              command="x")


def test_write_refuses_nbest_without_score_convention(setup, tmp_path, run_context):
    timed, warm = setup()
    adapter = FakeAdapter(requested_n=2)
    result = run(adapter, timed, warm, tmp_path, repeats=1)
    adapter.score_convention_id = None
    with pytest.raises(ValueError, match="score_convention_id"):
        write_run_outputs(tmp_path / "out", result, timed, run_context, adapter, command="x")


def test_zero_length_timed_audio_is_refused(setup, tmp_path, make_input):
    import numpy as np
    import soundfile as sf

    timed, warm = setup()
    sf.write(tmp_path / "audio" / "utt_silent.wav", np.zeros(0, np.float32), 16000)
    silent = [dataclasses_replace(make_input("utt_silent"), audio_uri="audio/utt_silent.wav")]
    with pytest.raises(ValueError, match="zero duration"):
        run(FakeAdapter(), silent, warm, tmp_path, repeats=1)


def test_limitations_note_thermal_flags_and_missing_sync(setup, tmp_path, run_context, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    timed, warm = setup()
    adapter = FakeAdapter()
    hot = {"ok": True, "name": "SYNTHETIC GPU", "clocks_event_reasons.hw_slowdown": "Active"}
    result = run(adapter, timed, warm, tmp_path, repeats=1, sync=None, probe=lambda: dict(hot))
    write_run_outputs(tmp_path / "out", result, timed, run_context, adapter, command="x")
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["environment"]["gpu_name"] == "SYNTHETIC GPU"
    assert any("2 of 2" in note for note in manifest["limitations"])
    assert any("synchronization" in note for note in manifest["limitations"])
