"""GPU state from nvidia-smi, recorded around every timed repeat."""
from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from typing import Any

NVIDIA_SMI_FIELDS = (
    "name", "driver_version", "clocks.sm", "clocks.max.sm", "power.draw", "temperature.gpu",
    "clocks_event_reasons.active", "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown", "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.sw_power_cap",
)
THERMAL_FIELDS = (
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
)
PROBE_TIMEOUT_S = 10


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _parse(stdout: str) -> dict:
    # ponytail: first GPU only; the timing laptop has one, add a gpu index if a multi-GPU box is used
    line = _first_line(stdout)
    if not line:
        return {"ok": False, "error": "nvidia-smi listed no GPU"}
    values = [value.strip() for value in line.split(",")]
    if len(values) != len(NVIDIA_SMI_FIELDS):
        return {"ok": False, "error": f"nvidia-smi printed {len(values)} values, expected {len(NVIDIA_SMI_FIELDS)}"}
    return {"ok": True, **dict(zip(NVIDIA_SMI_FIELDS, values))}


def probe_gpu(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict:
    """Query nvidia-smi once; never raises, returns {"ok": False, "error": ...} on any failure."""
    command = ["nvidia-smi", "--query-gpu=" + ",".join(NVIDIA_SMI_FIELDS), "--format=csv,noheader,nounits"]
    try:
        done = run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=PROBE_TIMEOUT_S, check=False,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "nvidia-smi not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"nvidia-smi did not answer within {PROBE_TIMEOUT_S} s"}
    except OSError as exc:
        return {"ok": False, "error": f"nvidia-smi could not start: {exc.strerror or exc}"}
    if done.returncode != 0:
        detail = _first_line(done.stderr or "") or _first_line(done.stdout or "")
        return {"ok": False, "error": f"nvidia-smi exited with code {done.returncode}: {detail}"}
    return _parse(done.stdout or "")


def thermal_flag(sample: Mapping[str, Any]) -> bool:
    """True when a hardware or thermal slowdown is active; sw_power_cap alone is normal."""
    return any(sample.get(field) == "Active" for field in THERMAL_FIELDS)
