"""Strict UTF-8 JSONL and JSON I/O for TURAS artifacts.

Reading refuses anything a contract-conformant writer would never produce: a BOM,
invalid UTF-8, blank lines, non-object lines, duplicate keys and non-finite
numbers. Writing produces UTF-8 without BOM, LF line ends and finite numbers, and
replaces the target file atomically.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_BOM = b"\xef\xbb\xbf"
_JSON_TYPE_NAMES = {list: "array", str: "string", int: "number", float: "number", bool: "boolean"}


class JsonlError(ValueError):
    """A JSONL file breaks the strict format; the message names the path and line."""


class _LineError(ValueError):
    """One line breaks the strict format (no path or line number yet)."""


def _reject_constant(name: str) -> Any:
    raise _LineError(f"{name} is not allowed; numbers must be finite")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _LineError(f"number {text} is not finite")
    return value


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise _LineError(f"duplicate key {key!r}")
        obj[key] = value
    return obj


def _loads_strict(text: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            object_pairs_hook=_unique_keys,
        )
    except _LineError:
        raise
    except json.JSONDecodeError as exc:
        raise _LineError(f"invalid JSON: {exc.msg} at column {exc.colno}") from exc
    except RecursionError as exc:
        raise _LineError("JSON is nested too deeply") from exc
    except ValueError as exc:
        raise _LineError(f"invalid JSON value: {exc}") from exc


def _parse_line(line: str) -> dict[str, Any]:
    if not line.strip(" \t"):
        raise _LineError("blank line; every line must hold one JSON object")
    obj = _loads_strict(line)
    if not isinstance(obj, dict):
        kind = "null" if obj is None else _JSON_TYPE_NAMES.get(type(obj), type(obj).__name__)
        raise _LineError(f"each line must be a JSON object, got {kind}")
    return obj


def _decode(path: Path, data: bytes) -> str:
    if data.startswith(_BOM):
        raise JsonlError(f"{path}: line 1: file starts with a UTF-8 BOM; save it as UTF-8 without BOM")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        line_no = data.count(b"\n", 0, exc.start) + 1
        raise JsonlError(f"{path}: line {line_no}: not valid UTF-8 ({exc.reason})") from exc


def _split_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        lines = lines[:-1]
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def read_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per line; raise JsonlError on the first broken line."""
    text = _decode(path, Path(path).read_bytes())
    rows = []
    for line_no, line in enumerate(_split_lines(text), start=1):
        try:
            rows.append(_parse_line(line))
        except _LineError as exc:
            raise JsonlError(f"{path}: line {line_no}: {exc}") from exc
    return rows


def _dumps_strict(obj: Any, indent: int | None = None) -> str:
    text = json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=indent)
    _loads_strict(text)
    return text


def encode_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """The exact bytes write_jsonl writes for rows; raise ValueError naming the first row that cannot be written."""
    parts = []
    for row_no, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_no}: must be a JSON object, got {type(row).__name__}")
        try:
            parts.append(_dumps_strict(dict(row)).encode("utf-8") + b"\n")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"row {row_no}: cannot be written as strict JSON: {exc}") from exc
    return b"".join(parts)


def _write_atomic(path: Path, data: bytes) -> str:
    target = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    """Write rows as strict JSONL atomically; return the SHA-256 hex of the bytes written."""
    return _write_atomic(path, encode_jsonl(rows))


def write_json(path: Path, obj: Any) -> str:
    """Write one strict JSON document (indented, LF) atomically; return its SHA-256 hex."""
    try:
        data = (_dumps_strict(obj, indent=2) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{Path(path).name}: cannot be written as strict JSON: {exc}") from exc
    return _write_atomic(path, data)


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's exact bytes."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
