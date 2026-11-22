from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_submission", ROOT / "scripts/verify_submission.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_verification_passes_on_repo(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_verifier()
    assert mod.verify_protocol_hashes(ROOT) == 56
    assert mod.verify_metrics_consistency(ROOT) == 20
    counts = mod.verify_splits_disjoint(ROOT)
    assert counts == {"train": 4376, "validation": 438, "test": 636}
    assert mod.verify_required_files(ROOT) == 11
    assert mod.verify_handoff_manifest(ROOT) == 4
    assert mod.main(ROOT) == 0
    assert "and 4 handoff-manifest file hashes verified." in capsys.readouterr().out


def _copy_artifact(root: Path, relative_path: Path) -> Path:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative_path, target)
    return target


def test_verification_rejects_weakened_protocol(tmp_path: Path) -> None:
    mod = _load_verifier()
    protocol_path = _copy_artifact(tmp_path, mod.PROTOCOL_PATH)
    protocol = json.loads(protocol_path.read_bytes())
    protocol["critical_code_sha256"].popitem()
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(AssertionError, match="Frozen protocol SHA-256 mismatch"):
        mod.verify_protocol_hashes(tmp_path)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (None, "Expected 20 systems"),
        ("unexpected-system", "Result system IDs do not match the frozen protocol"),
    ],
)
def test_verification_rejects_changed_result_conditions(
    tmp_path: Path, replacement: str | None, message: str
) -> None:
    mod = _load_verifier()
    _copy_artifact(tmp_path, mod.PROTOCOL_PATH)
    summary_path = _copy_artifact(tmp_path, Path("results/final/m45_test_summary.json"))
    metrics_path = _copy_artifact(tmp_path, Path("results/final/m45_metrics.csv"))
    summary = json.loads(summary_path.read_bytes())
    last_system = summary["systems"][-1]["system_id"]
    metrics_lines = metrics_path.read_text(encoding="utf-8").splitlines()
    if replacement is None:
        summary["systems"].pop()
        metrics_lines = [line for line in metrics_lines if not line.startswith(f"{last_system},")]
    else:
        summary["systems"][-1]["system_id"] = replacement
        metrics_lines = [line.replace(last_system, replacement, 1) for line in metrics_lines]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    metrics_path.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    with pytest.raises(AssertionError, match=message):
        mod.verify_metrics_consistency(tmp_path)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (None, r"Expected 636 IDs in test\.txt, got 635"),
        ("replacement-id", r"Frozen split SHA-256 mismatch for test\.txt"),
    ],
)
def test_verification_rejects_changed_split(
    tmp_path: Path, replacement: str | None, message: str
) -> None:
    mod = _load_verifier()
    _copy_artifact(tmp_path, mod.PROTOCOL_PATH)
    for name in ("split_manifest.json", "train.txt", "validation.txt", "test.txt"):
        _copy_artifact(tmp_path, mod.SPLIT_DIR / name)
    test_path = tmp_path / mod.SPLIT_DIR / "test.txt"
    lines = test_path.read_text(encoding="utf-8").splitlines()
    lines.pop()
    if replacement is not None:
        lines.append(replacement)
    test_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    with pytest.raises(AssertionError, match=message):
        mod.verify_splits_disjoint(tmp_path)


FINAL_DIR = Path("results/final")
REPORT_NAME = "M4_M5_FINAL_REPORT.md"
FAILURE_PREFIX = "Verification failed: "
CLI_TIMEOUT_SECONDS = 300  # only catches a hang


def _run_script(script: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-S", str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=CLI_TIMEOUT_SECONDS,
        cwd=cwd,
        check=False,
    )


def _run_copied_cli(root: Path) -> subprocess.CompletedProcess[str]:
    script = _copy_artifact(root, Path("scripts/verify_submission.py"))
    return _run_script(script, root)


def _assert_one_line_failure(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    lines = result.stderr.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(FAILURE_PREFIX)
    return lines[0]


def test_cli_reports_missing_protocol_on_one_line(tmp_path: Path) -> None:
    line = _assert_one_line_failure(_run_copied_cli(tmp_path))
    assert line.endswith(": configs/evaluation/m45_final_protocol.json")


def test_cli_reports_changed_protocol_on_one_line(tmp_path: Path) -> None:
    mod = _load_verifier()
    protocol = _copy_artifact(tmp_path, mod.PROTOCOL_PATH)
    protocol.write_bytes(protocol.read_bytes() + b" ")

    line = _assert_one_line_failure(_run_copied_cli(tmp_path))
    assert "Frozen protocol SHA-256 mismatch" in line


def test_cli_passes_on_repo() -> None:
    result = _run_script(ROOT / "scripts/verify_submission.py", ROOT)
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.endswith("and 4 handoff-manifest file hashes verified.\n")
    assert len(result.stdout.splitlines()) == 1


def _raiser(error: Exception):
    def fail(root: Path) -> int:
        raise error

    return fail


def _stderr_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    return captured.err.splitlines()


@pytest.mark.parametrize(
    ("make_error", "expected"),
    [
        (lambda r: AssertionError("Frozen protocol SHA-256 mismatch: x"), "Frozen protocol SHA-256 mismatch: x"),
        (lambda r: AssertionError("line one\nline two"), "line one line two"),
        (
            lambda r: PermissionError(13, "Permission denied", str(r / "a/b.csv")),
            "Permission denied: a/b.csv",
        ),
        (lambda r: FileNotFoundError(2, "No such file", "elsewhere.json"), "No such file: elsewhere.json"),
        (
            lambda r: UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
            "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff",
        ),
        (lambda r: KeyError("metrics"), "KeyError: 'metrics'"),
        (lambda r: IndexError("list index out of range"), "IndexError: list index out of range"),
        (lambda r: TypeError("bad type"), "TypeError: bad type"),
        (lambda r: AttributeError("no attribute get"), "AttributeError: no attribute get"),
        (lambda r: ZeroDivisionError("division by zero"), "ZeroDivisionError: division by zero"),
        (lambda r: OverflowError("int too large"), "OverflowError: int too large"),
        (lambda r: RecursionError("too deep"), "RecursionError: too deep"),
        (lambda r: csv.Error("field larger than field limit"), "Error: field larger than field limit"),
    ],
    ids=[
        "assertion",
        "assertion-multiline",
        "os-error-in-repo",
        "os-error-outside-repo",
        "decode-error",
        "key-error",
        "index-error",
        "type-error",
        "attribute-error",
        "zero-division",
        "overflow",
        "recursion",
        "csv-error",
    ],
)
def test_cli_reports_error_on_one_line(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str], make_error, expected: str
) -> None:
    mod = _load_verifier()
    error = make_error(tmp_path)
    monkeypatch.setattr(mod, "main", _raiser(error))

    assert mod._cli(tmp_path) == 1
    lines = _stderr_lines(capsys)
    assert len(lines) == 1
    if isinstance(error, UnicodeDecodeError):
        assert lines[0].startswith(FAILURE_PREFIX + expected)
    else:
        assert lines[0] == FAILURE_PREFIX + expected


def test_cli_lets_other_errors_propagate(tmp_path: Path, monkeypatch) -> None:
    mod = _load_verifier()
    monkeypatch.setattr(mod, "main", _raiser(RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        mod._cli(tmp_path)


def _flip_middle_byte(path: Path) -> None:
    data = path.read_bytes()
    i = len(data) // 2
    path.write_bytes(data[:i] + bytes([data[i] ^ 1]) + data[i + 1 :])


def _copy_pinned(root: Path, mod) -> list[str]:
    protocol_path = _copy_artifact(root, mod.PROTOCOL_PATH)
    pinned = list(json.loads(protocol_path.read_bytes())["critical_code_sha256"])
    for name in pinned:
        _copy_artifact(root, Path(name))
    return pinned


def test_pinned_mismatch_message_is_one_line_naming_every_file(tmp_path: Path) -> None:
    mod = _load_verifier()
    first, second, *_ = _copy_pinned(tmp_path, mod)
    (tmp_path / first).unlink()
    (tmp_path / second).unlink()

    with pytest.raises(AssertionError) as info:
        mod.verify_protocol_hashes(tmp_path)
    message = str(info.value)
    assert "\n" not in message
    assert f"{first}: missing; " in message
    assert f"{second}: missing" in message


def test_cli_names_edited_summary_before_parsing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_verifier()
    _copy_pinned(tmp_path, mod)
    _copy_handoff(tmp_path, mod)
    summary_path = tmp_path / FINAL_DIR / "m45_test_summary.json"
    summary = json.loads(summary_path.read_bytes())
    del summary["systems"][0]["metrics"]
    summary_path.write_bytes(json.dumps(summary).encode("utf-8"))

    assert mod._cli(tmp_path) == 1
    lines = _stderr_lines(capsys)
    assert len(lines) == 1
    assert "results/final/m45_test_summary.json: expected" in lines[0]


def _copy_handoff(root: Path, mod) -> Path:
    manifest_path = _copy_artifact(root, mod.HANDOFF_MANIFEST_PATH)
    for name in _listed(manifest_path):
        _copy_artifact(root, FINAL_DIR / name)
    return manifest_path


def _write_manifest(path: Path, manifest: object) -> None:
    path.write_bytes(json.dumps(manifest).encode("utf-8"))


def _listed(manifest_path: Path) -> dict[str, str]:
    return json.loads(manifest_path.read_bytes())["public_files_sha256"]


def test_handoff_manifest_flipped_byte_names_only_that_file(tmp_path: Path) -> None:
    mod = _load_verifier()
    manifest_path = _copy_handoff(tmp_path, mod)
    listed = _listed(manifest_path)
    report = tmp_path / FINAL_DIR / REPORT_NAME
    _flip_middle_byte(report)

    width = mod.HASH_PREFIX_LENGTH
    prefix = re.escape(listed[REPORT_NAME][:width])
    pattern = rf"results/final/M4_M5_FINAL_REPORT\.md: expected {prefix}, got [0-9a-f]{{{width}}}"
    with pytest.raises(AssertionError, match=pattern) as info:
        mod.verify_handoff_manifest(tmp_path)
    for other in listed:
        if other != REPORT_NAME:
            assert other not in str(info.value)


def test_handoff_manifest_hashes_raw_bytes_not_translated_text(tmp_path: Path) -> None:
    mod = _load_verifier()
    manifest_path = _copy_handoff(tmp_path, mod)
    extra = tmp_path / FINAL_DIR / "extra_lf.txt"
    extra.write_bytes(b"line one\nline two\n")
    manifest = json.loads(manifest_path.read_bytes())
    manifest["public_files_sha256"]["extra_lf.txt"] = hashlib.sha256(extra.read_bytes()).hexdigest()
    _write_manifest(manifest_path, manifest)
    assert mod.verify_handoff_manifest(tmp_path) == 5

    extra.write_bytes(b"line one\r\nline two\r\n")
    with pytest.raises(AssertionError, match=r"results/final/extra_lf\.txt"):
        mod.verify_handoff_manifest(tmp_path)


def test_handoff_manifest_rejects_lf_copy_of_crlf_file(tmp_path: Path) -> None:
    mod = _load_verifier()
    _copy_handoff(tmp_path, mod)
    report = tmp_path / FINAL_DIR / REPORT_NAME
    original = report.read_bytes()
    converted = original.replace(b"\r\n", b"\n")
    assert converted != original
    report.write_bytes(converted)

    with pytest.raises(AssertionError, match=r"results/final/M4_M5_FINAL_REPORT\.md"):
        mod.verify_handoff_manifest(tmp_path)


def test_handoff_manifest_reports_missing_listed_file(tmp_path: Path) -> None:
    mod = _load_verifier()
    _copy_handoff(tmp_path, mod)
    (tmp_path / FINAL_DIR / "m45_test_summary.json").unlink()

    with pytest.raises(AssertionError, match=r"results/final/m45_test_summary\.json: missing"):
        mod.verify_handoff_manifest(tmp_path)


def test_handoff_mismatch_message_is_one_line_naming_every_file(tmp_path: Path) -> None:
    mod = _load_verifier()
    _copy_handoff(tmp_path, mod)
    report = tmp_path / FINAL_DIR / REPORT_NAME
    _flip_middle_byte(report)
    (tmp_path / FINAL_DIR / "m45_metrics.csv").unlink()

    with pytest.raises(AssertionError) as info:
        mod.verify_handoff_manifest(tmp_path)
    message = str(info.value)
    assert "\n" not in message
    assert "results/final/M4_M5_FINAL_REPORT.md: expected" in message
    assert "results/final/m45_metrics.csv: missing" in message
    assert "; results/final/" in message


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param(None, "Missing handoff manifest", id="manifest-absent"),
        pytest.param(b"{not json", "Unreadable handoff manifest", id="invalid-json"),
        pytest.param(b"[" * 100_000, "Unreadable handoff manifest", id="nested-too-deep"),
        pytest.param({"status": "x"}, "has no public_files_sha256 mapping", id="key-absent"),
        pytest.param(
            {"public_files_sha256": ["M4_M5_FINAL_REPORT.md"]},
            "has no public_files_sha256 mapping",
            id="value-not-dict",
        ),
        pytest.param({"public_files_sha256": {}}, "has no public_files_sha256 mapping", id="empty-mapping"),
        pytest.param([], "has no public_files_sha256 mapping", id="top-level-list"),
    ],
)
def test_handoff_manifest_rejects_bad_manifest(
    tmp_path: Path, payload: object, message: str
) -> None:
    mod = _load_verifier()
    manifest_path = _copy_handoff(tmp_path, mod)
    if payload is None:
        manifest_path.unlink()
    elif isinstance(payload, bytes):
        manifest_path.write_bytes(payload)
    else:
        _write_manifest(manifest_path, payload)

    with pytest.raises(AssertionError, match=message) as info:
        mod.verify_handoff_manifest(tmp_path)
    assert "m45_handoff_manifest.json" in str(info.value)


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param(hashlib.sha256(b"other bytes\n").hexdigest(), id="wrong-hash"),
        pytest.param(12345, id="non-string-value"),
    ],
)
def test_handoff_manifest_checks_extra_entries(tmp_path: Path, expected: object) -> None:
    mod = _load_verifier()
    manifest_path = _copy_handoff(tmp_path, mod)
    (tmp_path / FINAL_DIR / "extra_public_file.txt").write_bytes(b"actual bytes\n")
    manifest = json.loads(manifest_path.read_bytes())
    manifest["public_files_sha256"]["extra_public_file.txt"] = expected
    _write_manifest(manifest_path, manifest)

    with pytest.raises(AssertionError, match=r"results/final/extra_public_file\.txt: expected"):
        mod.verify_handoff_manifest(tmp_path)
