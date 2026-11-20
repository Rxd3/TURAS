from __future__ import annotations

import importlib.util
from pathlib import Path

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
    assert mod.verify_required_files(ROOT) >= 9
    assert mod.main(ROOT) == 0
