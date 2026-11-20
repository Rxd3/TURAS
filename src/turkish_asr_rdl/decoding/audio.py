"""The one audio path every system uses: safe URI resolution, decode, downmix, 16 kHz."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import soundfile as sf
import soxr

TARGET_SAMPLE_RATE_HZ = 16_000


class AudioLoadError(ValueError):
    """Audio could not be located or decoded; the message is one friendly line."""


def resolve_audio_path(audio_root: Path, audio_uri: str) -> Path:
    """Resolve a relative, '/'-separated audio_uri to a path that stays inside audio_root."""
    if not isinstance(audio_uri, str) or not audio_uri:
        raise AudioLoadError("audio_uri must be a non-empty relative path")
    if "\\" in audio_uri:
        raise AudioLoadError(f"audio_uri {audio_uri!r} contains a backslash; use / as the separator")
    if audio_uri.startswith("/") or PureWindowsPath(audio_uri).anchor:
        raise AudioLoadError(f"audio_uri {audio_uri!r} must be relative to the audio root, without a drive letter")
    if "\x00" in audio_uri:
        raise AudioLoadError(f"audio_uri {audio_uri!r} contains a NUL character")
    try:
        root = Path(audio_root).resolve()
        target = (root / audio_uri).resolve()
    except (OSError, ValueError) as exc:
        raise AudioLoadError(f"audio_uri {audio_uri!r} is not a usable path ({exc})") from exc
    if not target.is_relative_to(root):
        raise AudioLoadError(f"audio_uri {audio_uri!r} points outside the audio root")
    return target


def _check_rate(sample_rate_hz: Any) -> int:
    valid = isinstance(sample_rate_hz, (int, np.integer)) and not isinstance(sample_rate_hz, bool)
    if not valid or sample_rate_hz <= 0:
        raise AudioLoadError(f"sample rate must be a positive integer, got {sample_rate_hz!r}")
    return int(sample_rate_hz)


def to_mono_16k(samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Return a new float32 mono array at 16 kHz (channel mean, then soxr HQ resampling)."""
    rate = _check_rate(sample_rate_hz)
    array = np.array(samples, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] > 0:
        mono = array.mean(axis=1, dtype=np.float32)
    elif array.ndim == 1:
        mono = array
    else:
        raise AudioLoadError(f"audio must have shape (frames,) or (frames, channels), got {array.shape}")
    if rate != TARGET_SAMPLE_RATE_HZ:
        mono = soxr.resample(mono, rate, TARGET_SAMPLE_RATE_HZ, quality="HQ")
    return np.ascontiguousarray(mono, dtype=np.float32)


def load_audio(path: Path) -> np.ndarray:
    """Decode an audio file to float32 mono 16 kHz; failures raise AudioLoadError."""
    try:
        with open(path, "rb") as handle:
            data, rate = sf.read(handle, dtype="float32", always_2d=True)
    except FileNotFoundError as exc:
        raise AudioLoadError(f"audio file not found: {path}") from exc
    except (IsADirectoryError, PermissionError) as exc:
        raise AudioLoadError(f"cannot read audio file {path}: it is a folder or access is denied") from exc
    except sf.SoundFileError as exc:
        raise AudioLoadError(f"cannot decode audio file {path}: unsupported format or damaged file") from exc
    except OSError as exc:
        raise AudioLoadError(f"cannot read audio file {path}: {exc.strerror or exc}") from exc
    return to_mono_16k(data, rate)
