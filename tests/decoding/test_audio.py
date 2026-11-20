"""Shared 16 kHz mono audio loader. All audio is a SYNTHETIC sine tone."""
import numpy as np
import pytest
import soundfile as sf

from turkish_asr_rdl.decoding.audio import (
    TARGET_SAMPLE_RATE_HZ,
    AudioLoadError,
    load_audio,
    resolve_audio_path,
    to_mono_16k,
)


def _one_friendly_line(exc: Exception) -> str:
    message = str(exc)
    assert "\n" not in message and "Traceback" not in message
    return message


def test_target_rate_is_16k():
    assert TARGET_SAMPLE_RATE_HZ == 16000


def test_24k_stereo_becomes_16k_mono(tmp_path, tone):
    x = np.stack([tone(24000, 1.0), tone(24000, 1.0)], axis=1)
    sf.write(tmp_path / "a.wav", x, 24000)
    y = load_audio(tmp_path / "a.wav")
    assert y.dtype == np.float32 and y.ndim == 1 and abs(len(y) - 16000) <= 2


def test_16k_mono_flac_unchanged_length(write_tone, tone):
    y = load_audio(write_tone("a.flac", 16000, 1.0))
    assert y.dtype == np.float32 and y.shape == (16000,)
    assert np.allclose(y, tone(16000, 1.0), atol=1e-3)


def test_44k1_mono_is_resampled(write_tone):
    y = load_audio(write_tone("b.flac", 44100, 0.5))
    assert abs(len(y) - 8000) <= 2


def test_downmix_is_the_channel_mean():
    stereo = np.stack([np.full(100, 0.5, np.float32), np.zeros(100, np.float32)], axis=1)
    assert np.allclose(to_mono_16k(stereo, 16000), 0.25)


def test_to_mono_16k_does_not_modify_or_alias_input():
    mono = np.linspace(-0.5, 0.5, 320, dtype=np.float32)
    before = mono.copy()
    out = to_mono_16k(mono, 16000)
    out[0] = 9.0
    assert np.array_equal(mono, before)


def test_to_mono_16k_returns_float32_for_float64_input():
    assert to_mono_16k(np.zeros(160, dtype=np.float64), 8000).dtype == np.float32


@pytest.mark.parametrize("samples", [np.zeros((2, 2, 2)), np.zeros((10, 0)), np.float32(0.0)])
def test_to_mono_16k_refuses_bad_shapes(samples):
    with pytest.raises(AudioLoadError):
        to_mono_16k(samples, 16000)


@pytest.mark.parametrize("rate", [0, -16000, 16000.0, True, None])
def test_to_mono_16k_refuses_bad_rates(rate):
    with pytest.raises(AudioLoadError):
        to_mono_16k(np.zeros(10, np.float32), rate)


@pytest.mark.parametrize("uri", [
    "/abs/a.flac", "C:/a.flac", "a\\b.flac", "../a.flac", "x/../../a.flac", "",
    "C:a.flac", "c:\\a.flac", "//server/share/a.flac", "x/../..", "a\x00.flac",
])
def test_unsafe_uris_refused(tmp_path, uri):
    with pytest.raises(AudioLoadError):
        resolve_audio_path(tmp_path, uri)


@pytest.mark.parametrize("uri", [None, 3, b"a.flac"])
def test_non_string_uri_refused(tmp_path, uri):
    with pytest.raises(AudioLoadError):
        resolve_audio_path(tmp_path, uri)


def test_relative_uri_resolves_under_root(tmp_path):
    assert resolve_audio_path(tmp_path, "audio/a.flac") == (tmp_path / "audio" / "a.flac").resolve()
    assert resolve_audio_path(tmp_path, "x/../a.flac") == (tmp_path / "a.flac").resolve()


def test_symlink_out_of_root_refused(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this machine does not allow creating symlinks")
    with pytest.raises(AudioLoadError, match="outside"):
        resolve_audio_path(root, "link/a.flac")


def test_missing_file_is_friendly(tmp_path):
    with pytest.raises(AudioLoadError) as caught:
        load_audio(tmp_path / "nope_synthetic.flac")
    assert "nope_synthetic.flac" in _one_friendly_line(caught.value)
    assert "not found" in str(caught.value)


def test_directory_is_friendly(tmp_path):
    (tmp_path / "folder.flac").mkdir()
    with pytest.raises(AudioLoadError) as caught:
        load_audio(tmp_path / "folder.flac")
    assert "folder.flac" in _one_friendly_line(caught.value)


def test_unreadable_format_is_friendly(tmp_path):
    path = tmp_path / "noise.flac"
    path.write_bytes(b"SYNTHETIC not audio at all")
    with pytest.raises(AudioLoadError) as caught:
        load_audio(path)
    assert "noise.flac" in _one_friendly_line(caught.value)


def test_other_os_errors_are_friendly(tmp_path, monkeypatch):
    import errno

    import turkish_asr_rdl.decoding.audio as audio_module

    def failing_open(*args, **kwargs):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(audio_module, "open", failing_open, raising=False)
    with pytest.raises(AudioLoadError, match="Input/output error") as caught:
        load_audio(tmp_path / "disk_synthetic.flac")
    assert "disk_synthetic.flac" in _one_friendly_line(caught.value)
