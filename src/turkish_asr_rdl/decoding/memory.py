"""GPU memory around a timed run: free and total VRAM before load, peak reserved VRAM per repeat.

torch and transformers are imported inside the functions that use them, so this module
imports on a machine that has neither.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

GIB = 2**30


class VramMeter(Protocol):
    """What the timing harness reads, in GiB; None where nothing can be measured."""

    method: str

    def free_and_total_gib(self) -> tuple[float | None, float | None]:
        """Free and total device memory now."""

    def reset_peak(self) -> None:
        """Start a new peak measurement."""

    def peak_reserved_gib(self) -> float | None:
        """Peak memory this process reserved since the last reset_peak."""


@dataclass(frozen=True)
class NoVram:
    """No torch or no CUDA device: every value is None."""

    method: str = "none"

    def free_and_total_gib(self) -> tuple[float | None, float | None]:
        return None, None

    def reset_peak(self) -> None:
        return None

    def peak_reserved_gib(self) -> float | None:
        return None


@dataclass(frozen=True)
class TorchCudaVram:
    """The caching allocator's counters on the current CUDA device."""

    torch: Any
    method: str = "torch.cuda"

    def free_and_total_gib(self) -> tuple[float | None, float | None]:
        self.torch.cuda.empty_cache()
        free, total = self.torch.cuda.mem_get_info()
        return free / GIB, total / GIB

    def reset_peak(self) -> None:
        self.torch.cuda.reset_peak_memory_stats()

    def peak_reserved_gib(self) -> float | None:
        return self.torch.cuda.max_memory_reserved() / GIB


@dataclass(frozen=True)
class VramRecord:
    """Memory of one timed run; peak_reserved_gib holds one value per timed repeat."""

    memory_method: str = "none"
    vram_total_gib: float | None = None
    vram_free_before_load_gib: float | None = None
    peak_reserved_gib: tuple[float | None, ...] = ()

    @property
    def vram_spill_suspected(self) -> bool | None:
        """True when any repeat reserved more than was free before load; None when that cannot be told."""
        flags = [spill_suspected(peak, self.vram_free_before_load_gib) for peak in self.peak_reserved_gib]
        if any(flags):
            return True
        return None if not flags or None in flags else False


def spill_suspected(peak_reserved_gib: float | None, free_before_load_gib: float | None) -> bool | None:
    """Windows can spill into shared memory instead of failing, so the fit is judged against free memory."""
    if peak_reserved_gib is None or free_before_load_gib is None:
        return None
    return peak_reserved_gib > free_before_load_gib


def _optional_import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None


def default_vram() -> VramMeter:
    """torch.cuda counters when torch imports and CUDA is available, else a meter that records nothing."""
    torch = _optional_import("torch")
    if torch is not None and torch.cuda.is_available():
        return TorchCudaVram(torch)
    return NoVram()


def library_versions() -> dict[str, str | None]:
    """torch, its CUDA build and transformers versions; None for any that does not import."""
    torch = _optional_import("torch")
    transformers = _optional_import("transformers")
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    versions = {
        "torch": getattr(torch, "__version__", None),
        "cuda": cuda,
        "transformers": getattr(transformers, "__version__", None),
    }
    return {name: None if value is None else str(value) for name, value in versions.items()}
