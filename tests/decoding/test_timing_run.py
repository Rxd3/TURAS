"""Timing harness: adapter checks before loading, VRAM and error-row limitations. SYNTHETIC inputs only."""
import itertools
import json
import sys
import types
from collections import Counter

import pytest

import turkish_asr_rdl.decoding.timing as timing_module
from turkish_asr_rdl.decoding.interface import Hypothesis
from turkish_asr_rdl.decoding.timing import run_timed, write_run_outputs
from turkish_asr_rdl.schemas.jsonl import read_jsonl

SPILL = "RTF invalid: peak reserved VRAM exceeded free VRAM before load (suspected sysmem fallback)"

WARM = ("warm_1", "warm_2", "warm_3")
TIMED = ("utt_a", "utt_b")


class ConfigAdapter:
    """Counts load calls; its configuration is whatever the test gives."""

    system_id = "sys_synthetic"
    model_artifact_sha256 = "b" * 64
    decoder_config_sha256 = "a" * 64

    def __init__(self, requested_n=0, score_convention_id=None, config=...):
        self.requested_n = requested_n
        self.score_convention_id = score_convention_id
        self.config = {"beam": 1} if config is ... else config
        self.loads = 0

    def load(self):
        self.loads += 1

    def transcribe(self, batch):
        return [Hypothesis(item.utterance_id, "ok", "SYNTHETIC") for item in batch]


def no_default(*args, **kwargs):
    raise AssertionError("a default device helper ran before the adapter configuration was checked")


@pytest.fixture
def refused(make_input, tmp_path, monkeypatch):
    """Run the harness and return the ValueError it raises; no default device helper may run first."""
    monkeypatch.setattr(timing_module, "default_sync", no_default)
    monkeypatch.setattr(timing_module, "default_vram", no_default)

    def _refused(adapter):
        timed, warm = [make_input(u) for u in TIMED], [make_input(u) for u in WARM]
        with pytest.raises(ValueError) as info:
            run_timed(adapter, timed, warm, audio_root=tmp_path, run_id="run_synthetic", sync=None,
                      probe=no_default)
        assert adapter.loads == 0
        message = str(info.value)
        assert "\n" not in message
        return message

    return _refused


@pytest.mark.parametrize("requested_n", [-1, True, False, 2.0, "3", None])
def test_requested_n_must_be_an_int_of_at_least_zero(refused, requested_n):
    message = refused(ConfigAdapter(requested_n=requested_n, score_convention_id="nbest-synthetic-v1"))
    assert "requested_n" in message


@pytest.mark.parametrize("convention", [None, "", 5])
def test_nbest_needs_a_score_convention_id(refused, convention):
    assert "score_convention_id" in refused(ConfigAdapter(requested_n=2, score_convention_id=convention))


@pytest.mark.parametrize("config", [
    {"length_penalty": float("nan")},
    {"beam": float("inf")},
    {"device": object()},
    {1: "int key", "1": "str key"},
    None,
    [("beam", 5)],
])
def test_config_must_be_strict_json(refused, config):
    assert "adapter.config" in refused(ConfigAdapter(config=config))


class EventAdapter(ConfigAdapter):
    """Logs load and transcribe calls into a shared event list; fails the nth sight of chosen utterances."""

    def __init__(self, events, fail_on=()):
        super().__init__()
        self.events, self.fail_on, self.seen = events, set(fail_on), Counter()

    def load(self):
        super().load()
        self.events.append("load")

    def transcribe(self, batch):
        self.events.append("transcribe")
        self.seen.update(item.utterance_id for item in batch)
        if any((item.utterance_id, self.seen[item.utterance_id]) in self.fail_on for item in batch):
            raise RuntimeError("SYNTHETIC out of memory")
        return super().transcribe(batch)


class FakeMeter:
    """SYNTHETIC VRAM meter: 10 GiB free of 12 before load, then the given peak for each repeat."""

    method = "fake.meter"

    def __init__(self, events, peaks, free=10.0):
        self.events, self.peaks, self.free = events, iter(peaks), free

    def free_and_total_gib(self):
        self.events.append("free")
        return self.free, 12.0

    def reset_peak(self):
        self.events.append("reset")

    def peak_reserved_gib(self):
        self.events.append("peak")
        return next(self.peaks)


@pytest.fixture
def timed_run(tmp_path, write_tone, make_input, run_context):
    """Run two repeats on SYNTHETIC 1 s tones with a fake clock, write the outputs; return (run, out dir)."""

    def _run(adapter, memory):
        for uid in (*WARM, *TIMED):
            write_tone(f"audio/{uid}.flac", 16000, 1.0)
        timed, warm = [make_input(u) for u in TIMED], [make_input(u) for u in WARM]
        ticks = itertools.count(0.0, 0.5)
        result = run_timed(adapter, timed, warm, audio_root=tmp_path, run_id="run_synthetic", repeats=2,
                           clock=lambda: next(ticks), probe=lambda: {"ok": False, "error": "no gpu"},
                           sync=lambda: None, memory=memory)
        write_run_outputs(tmp_path / "out", result, timed, run_context, adapter, command="SYNTHETIC cmd")
        return result, tmp_path / "out"

    return _run


def manifest_of(out):
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def test_memory_is_read_before_load_and_around_each_repeat(timed_run):
    events = []
    timed_run(EventAdapter(events), FakeMeter(events, (6.5, 7.0)))
    t = "transcribe"
    assert events == ["free", "load", t, t, t, "reset", t, t, "peak", "reset", t, t, "peak"]


def test_peak_above_free_memory_flags_the_rtf_invalid(timed_run):
    events = []
    run, out = timed_run(EventAdapter(events), FakeMeter(events, (6.5, 11.0)))
    assert run.vram.vram_spill_suspected is True
    batches = [r for r in read_jsonl(out / "timing.jsonl") if r["record_type"] == "batch"]
    by_repeat = {r["repeat"]: (r["peak_reserved_gib"], r["vram_spill_suspected"]) for r in batches}
    assert by_repeat == {0: (None, None), 1: (6.5, False), 2: (11.0, True)}
    assert {(r["memory_method"], r["vram_total_gib"], r["vram_free_before_load_gib"]) for r in batches} == {
        ("fake.meter", 12.0, 10.0)}
    assert SPILL in manifest_of(out)["limitations"]


def test_peak_equal_to_free_memory_is_not_a_spill(timed_run):
    events = []
    run, out = timed_run(EventAdapter(events), FakeMeter(events, (10.0, 10.0)))
    assert run.vram.vram_spill_suspected is False
    assert not any(note.startswith("RTF invalid") for note in manifest_of(out)["limitations"])


def test_default_meter_without_torch_records_nulls(timed_run, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    run, out = timed_run(EventAdapter([]), None)
    environment = manifest_of(out)["environment"]
    assert (environment["torch"], environment["cuda"], environment["transformers"]) == (None, None, None)
    assert run.vram.vram_spill_suspected is None
    batches = [r for r in read_jsonl(out / "timing.jsonl") if r["record_type"] == "batch"]
    memory = {(r["memory_method"], r["vram_total_gib"], r["peak_reserved_gib"], r["vram_spill_suspected"])
              for r in batches}
    assert memory == {("none", None, None, None)}
    assert not any(note.startswith("RTF") for note in manifest_of(out)["limitations"])


def test_timed_error_rows_from_every_repeat_are_counted_but_not_warmup(timed_run):
    events = []
    adapter = EventAdapter(events, fail_on={("warm_1", 1), ("utt_b", 2)})
    run, out = timed_run(adapter, FakeMeter(events, (1.0, 1.0)))
    assert run.timed_error_rows == 1
    assert [r.error_rows for r in run.records] == [1, 0, 0, 0, 0, 0, 1]
    assert "RTF includes 1 timed error rows" in manifest_of(out)["limitations"]


def test_no_error_rows_means_no_error_limitation(timed_run):
    events = []
    _, out = timed_run(EventAdapter(events), FakeMeter(events, (1.0, 1.0)))
    assert not any("error rows" in note for note in manifest_of(out)["limitations"])


def test_manifest_environment_names_torch_cuda_and_transformers(timed_run, monkeypatch):
    torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda="12.8"), __version__="2.9.0+cu128")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="5.23.0"))
    events = []
    _, out = timed_run(EventAdapter(events), FakeMeter(events, (1.0, 1.0)))
    environment = manifest_of(out)["environment"]
    versions = (environment["torch"], environment["cuda"], environment["transformers"])
    assert versions == ("2.9.0+cu128", "12.8", "5.23.0")
