"""GPU memory meter and library versions. Fake torch modules only; no GPU needed."""
import sys
import types

import pytest

from turkish_asr_rdl.decoding.memory import NoVram, VramRecord, default_vram, library_versions


def fake_torch(available=True, calls=None):
    """A SYNTHETIC torch module: 10 GiB free of 12, 5 GiB peak allocated, 7 GiB peak reserved."""
    calls = [] if calls is None else calls
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        empty_cache=lambda: calls.append("empty_cache"),
        mem_get_info=lambda: calls.append("mem_get_info") or (10 * 2**30, 12 * 2**30),
        reset_peak_memory_stats=lambda: calls.append("reset_peak_memory_stats"),
        max_memory_allocated=lambda: 5 * 2**30,
        max_memory_reserved=lambda: 7 * 2**30,
    )
    return types.SimpleNamespace(cuda=cuda, version=types.SimpleNamespace(cuda="12.8"), __version__="2.9.0+cu128")


def test_default_vram_without_torch_measures_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    meter = default_vram()
    assert meter.method == "none" and meter.free_and_total_gib() == (None, None)
    assert meter.reset_peak() is None and meter.peak_reserved_gib() is None


@pytest.fixture
def broken_torch(tmp_path, monkeypatch):
    """A SYNTHETIC torch whose import fails with OSError, as a missing CUDA DLL does on Windows."""
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_text('raise OSError("SYNTHETIC: a DLL could not be loaded")\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setitem(sys.modules, "transformers", None)


def test_default_vram_with_a_torch_that_fails_to_load_measures_nothing(broken_torch):
    assert isinstance(default_vram(), NoVram)


def test_library_versions_are_null_when_torch_fails_to_load(broken_torch):
    assert library_versions() == {"torch": None, "cuda": None, "transformers": None}


def test_default_vram_without_cuda_measures_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available=False))
    assert isinstance(default_vram(), NoVram)


def test_cuda_meter_reads_free_then_total_and_peak_reserved(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(calls=calls))
    meter = default_vram()
    assert meter.method == "torch.cuda"
    assert meter.free_and_total_gib() == (10.0, 12.0)
    assert calls == ["empty_cache", "mem_get_info"]
    meter.reset_peak()
    assert calls[-1] == "reset_peak_memory_stats"
    assert meter.peak_reserved_gib() == 7.0


def test_library_versions_from_imported_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="5.23.0"))
    assert library_versions() == {"torch": "2.9.0+cu128", "cuda": "12.8", "transformers": "5.23.0"}


def test_library_versions_are_null_when_not_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert library_versions() == {"torch": None, "cuda": None, "transformers": None}


def test_cpu_torch_build_has_no_cuda_version(monkeypatch):
    cpu = types.SimpleNamespace(version=types.SimpleNamespace(cuda=None), __version__="2.9.0+cpu")
    monkeypatch.setitem(sys.modules, "torch", cpu)
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert library_versions() == {"torch": "2.9.0+cpu", "cuda": None, "transformers": None}


@pytest.mark.parametrize("peaks,expected", [
    ((6.5, 10.0), False),
    ((6.5, 11.0), True),
    ((11.0, 6.5), True),
    ((None, 11.0), True),
    ((6.5, None), None),
    ((), None),
])
def test_spill_is_peak_reserved_above_free_before_load(peaks, expected):
    record = VramRecord("fake", 12.0, 10.0, peaks)
    assert record.vram_spill_suspected is expected


def test_spill_is_unknown_without_free_memory():
    assert VramRecord("none", None, None, (11.0,)).vram_spill_suspected is None
