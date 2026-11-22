"""Strict JSONL I/O and validation reports. All text values are SYNTHETIC."""
import json

import pytest

from turkish_asr_rdl.schemas.jsonl import (
    JsonlError,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from turkish_asr_rdl.schemas.report import ValidationReport, combine

ROWS = [{"utterance_id": "utt_a", "text": "SYNTHETIC ışık"}, {"n": 2, "x": None, "score": -1.25}]


def _write(tmp_path, data: bytes):
    path = tmp_path / "x.jsonl"
    path.write_bytes(data)
    return path


def test_roundtrip_returns_rows_and_hash(tmp_path):
    path = tmp_path / "rows.jsonl"
    digest = write_jsonl(path, ROWS)
    assert read_jsonl(path) == ROWS
    assert digest == sha256_file(path)


def test_written_bytes_are_utf8_lf_without_bom(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, ROWS)
    data = path.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert data.endswith(b"\n")
    assert "ışık".encode("utf-8") in data


def test_line_separator_inside_string_roundtrips(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"text": "SYNTHETIC a\u2028b\u0085c"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_empty_file_has_no_rows(tmp_path):
    assert read_jsonl(_write(tmp_path, b"")) == []


def test_last_line_without_newline_accepted(tmp_path):
    assert read_jsonl(_write(tmp_path, b'{"a": 1}\n{"a": 2}')) == [{"a": 1}, {"a": 2}]


def test_bom_is_rejected_with_fix_hint(tmp_path):
    path = _write(tmp_path, b'\xef\xbb\xbf{"a": 1}\n')
    with pytest.raises(JsonlError, match="BOM") as caught:
        read_jsonl(path)
    message = str(caught.value)
    assert "without BOM" in message and "x.jsonl" in message and "\n" not in message


def test_invalid_utf8_rejected_with_line(tmp_path):
    path = _write(tmp_path, b'{"a": 1}\n{"a": "\xff"}\n')
    with pytest.raises(JsonlError, match="line 2"):
        read_jsonl(path)


@pytest.mark.parametrize("text", ['{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}', '{"a": 1e400}'])
def test_nan_and_infinity_rejected(tmp_path, text):
    path = _write(tmp_path, text.encode())
    with pytest.raises(JsonlError, match="line 1"):
        read_jsonl(path)


def test_duplicate_keys_rejected(tmp_path):
    path = _write(tmp_path, b'{"a": 1, "a": 2}\n')
    with pytest.raises(JsonlError, match="duplicate key"):
        read_jsonl(path)


def test_nested_duplicate_keys_rejected(tmp_path):
    path = _write(tmp_path, b'{"a": {"b": 1, "b": 1}}\n')
    with pytest.raises(JsonlError, match="duplicate key"):
        read_jsonl(path)


def test_non_object_line_rejected(tmp_path):
    path = _write(tmp_path, b"[1, 2]\n")
    with pytest.raises(JsonlError, match="line 1.*object"):
        read_jsonl(path)


def test_invalid_json_rejected_with_line(tmp_path):
    path = _write(tmp_path, b'{"a": 1}\n{"a": }\n')
    with pytest.raises(JsonlError, match="line 2"):
        read_jsonl(path)


def test_deep_nesting_is_a_jsonl_error(tmp_path, monkeypatch):
    # Python 3.14's json parser accepts deep nesting that older versions reject with
    # RecursionError, so force that error to test the wrapping on every version.
    def too_deep(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(json, "loads", too_deep)
    path = _write(tmp_path, b'{"a": [[[1]]]}\n')
    with pytest.raises(JsonlError, match="line 1.*nested too deeply"):
        read_jsonl(path)


def test_very_deep_nesting_never_escapes_as_another_error(tmp_path):
    path = _write(tmp_path, b'{"a": ' + b"[" * 100_000 + b"]" * 100_000 + b"}\n")
    try:
        rows = read_jsonl(path)
    except JsonlError as exc:
        assert "line 1" in str(exc)
    else:
        assert len(rows) == 1


@pytest.mark.parametrize("data", [b'{"a":1}\n\n{"a":2}\n', b'{"a":1}\n   \n{"a":2}\n', b'{"a":1}\n\n'])
def test_blank_line_inside_file_rejected(tmp_path, data):
    with pytest.raises(JsonlError, match="line 2.*blank"):
        read_jsonl(_write(tmp_path, data))


def test_crlf_lines_accepted(tmp_path):
    assert read_jsonl(_write(tmp_path, b'{"a": 1}\r\n{"a": 2}\r\n')) == [{"a": 1}, {"a": 2}]


def test_write_is_atomic_on_error(tmp_path):
    path = tmp_path / "rows.jsonl"
    with pytest.raises(ValueError):
        write_jsonl(path, [{"a": 1}, {"a": float("nan")}])
    assert list(tmp_path.iterdir()) == []


def test_failed_write_keeps_previous_file(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"a": 1}])
    with pytest.raises(ValueError):
        write_jsonl(path, [{"a": float("inf")}])
    assert read_jsonl(path) == [{"a": 1}]
    assert [p.name for p in tmp_path.iterdir()] == ["rows.jsonl"]


def test_failed_replace_leaves_no_temp_file(tmp_path):
    (tmp_path / "rows.jsonl").mkdir()
    with pytest.raises(OSError):
        write_jsonl(tmp_path / "rows.jsonl", [{"a": 1}])
    assert [p.name for p in tmp_path.iterdir()] == ["rows.jsonl"]


def test_write_refuses_row_that_is_not_an_object(tmp_path):
    with pytest.raises(ValueError, match="row 1"):
        write_jsonl(tmp_path / "rows.jsonl", [[1, 2]])


def test_write_refuses_keys_that_would_collide(tmp_path):
    with pytest.raises(ValueError, match="row 1"):
        write_jsonl(tmp_path / "rows.jsonl", [{1: "a", "1": "b"}])


def test_write_refuses_row_nested_too_deeply(tmp_path):
    nested: list = []
    for _ in range(100_000):
        nested = [nested]
    with pytest.raises(ValueError, match="row 2: cannot be written as strict JSON"):
        write_jsonl(tmp_path / "rows.jsonl", [{"a": 1}, {"deep": nested}])
    assert not (tmp_path / "rows.jsonl").exists()


def test_write_json_is_lf_utf8_and_hashed(tmp_path):
    path = tmp_path / "manifest.json"
    digest = write_json(path, {"text": "SYNTHETIC ğ", "n": [1, 2]})
    data = path.read_bytes()
    assert json.loads(data.decode("utf-8")) == {"text": "SYNTHETIC ğ", "n": [1, 2]}
    assert b"\r" not in data and data.endswith(b"\n")
    assert digest == sha256_file(path)


def test_write_json_refuses_nan(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "m.json", {"a": float("nan")})
    assert list(tmp_path.iterdir()) == []


def test_write_json_refuses_non_json_value(tmp_path):
    with pytest.raises(ValueError, match="m.json"):
        write_json(tmp_path / "m.json", {"a": object()})


def test_integer_beyond_conversion_limit_is_a_jsonl_error(tmp_path):
    path = _write(tmp_path, b'{"a": 1' + b"0" * 5000 + b"}\n")
    with pytest.raises(JsonlError, match="line 1"):
        read_jsonl(path)


def test_report_ok_only_without_errors():
    assert ValidationReport().ok
    assert ValidationReport(warnings=("w",)).ok
    assert not ValidationReport(errors=("e",)).ok


def test_combine_merges_in_order():
    merged = combine(ValidationReport(errors=("a",)), ValidationReport(warnings=("w",)), ValidationReport(errors=("b",)))
    assert merged == ValidationReport(("a", "b"), ("w",))


def test_combine_of_nothing_is_ok():
    assert combine() == ValidationReport()
