"""Shared builders for decoding tests: synthetic tones only, no dataset audio."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def make_tone(sample_rate_hz: int, seconds: float, freq_hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(round(sample_rate_hz * seconds))) / sample_rate_hz
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


@pytest.fixture
def tone():
    """Build a SYNTHETIC sine tone: tone(sample_rate_hz, seconds) -> float32 array."""
    return make_tone


@pytest.fixture
def write_tone(tmp_path):
    """Write a SYNTHETIC tone under tmp_path and return its path."""

    def _write(relative: str, sample_rate_hz: int = 16000, seconds: float = 1.0, channels: int = 1) -> Path:
        mono = make_tone(sample_rate_hz, seconds)
        data = mono if channels == 1 else np.stack([mono] * channels, axis=1)
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, data, sample_rate_hz)
        return path

    return _write
