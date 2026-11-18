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


HEX_A = "a" * 64
HEX_B = "b" * 64


@pytest.fixture
def make_input():
    """Build a valid SYNTHETIC InferenceInput whose audio sits at audio/<uid>.flac."""
    from turkish_asr_rdl.decoding.interface import InferenceInput

    def _make(uid: str, duration_s: float = 1.0) -> "InferenceInput":
        return InferenceInput(
            schema_version="0.1",
            dataset_release="rel_synthetic",
            split_manifest_sha256=HEX_A,
            utterance_id=uid,
            audio_uri=f"audio/{uid}.flac",
            audio_sha256=HEX_B,
            duration_s=duration_s,
            sample_rate_hz=16000,
            channels=1,
            language="tr",
        )

    return _make


@pytest.fixture
def run_context():
    from turkish_asr_rdl.decoding.export import RunContext

    return RunContext(
        run_id="run_synthetic",
        system_id="sys_synthetic",
        dataset_release="rel_synthetic",
        split="test",
        split_manifest_sha256=HEX_A,
        model_artifact_sha256=HEX_B,
        decoder_config_sha256=HEX_A,
    )
