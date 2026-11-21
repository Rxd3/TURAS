from __future__ import annotations

import importlib.util
import json
import shutil
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


def test_submission_verification_passes_on_repo() -> None:
    mod = _load_verifier()
    assert mod.verify_protocol_hashes(ROOT) == 56
    assert mod.verify_metrics_consistency(ROOT) == 20
    counts = mod.verify_splits_disjoint(ROOT)
    assert counts == {"train": 4376, "validation": 438, "test": 636}
    assert mod.verify_required_files(ROOT) == 11
    assert mod.main(ROOT) == 0


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
