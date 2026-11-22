"""Metadata-only delivery must retrieve missing weights, with no live network in tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

spec = importlib.util.spec_from_file_location("completion_download", Path(__file__).parents[1] / "scripts/download_completion_models.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def metadata_only(tmp_path):
    weights = b"synthetic unit-test weight bytes; never used for inference"
    files = {"config.json": b"{}", "tokenizer.json": b"{}", "model.safetensors": weights}
    directory = tmp_path / "artifacts/completion-2026-11-22/models/b2"
    directory.mkdir(parents=True)
    for name in ["config.json", "tokenizer.json"]:
        (directory / name).write_bytes(files[name])
    inventory = {"model_id": "test/pinned-model", "revision": "test-revision",
                 "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()}}
    (directory / "verified-inventory.json").write_text(json.dumps(inventory))
    return directory, inventory, weights


def test_metadata_only_package_downloads_missing_weight_instead_of_crashing(tmp_path, monkeypatch):
    directory, inventory, weights = metadata_only(tmp_path)
    expected_sha = hashlib.sha256(weights).hexdigest()
    assert not download.inventory_is_verified(directory, inventory, "test/pinned-model", "test-revision", expected_sha)
    calls = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs)
        assert kwargs["repo_id"] == "test/pinned-model" and kwargs["revision"] == "test-revision"
        (Path(kwargs["local_dir"]) / "model.safetensors").write_bytes(weights)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot))
    monkeypatch.setattr(download, "ROOT", tmp_path)
    monkeypatch.setattr(download, "MODELS", {"b2": ("test/pinned-model", "test-revision", expected_sha)})
    monkeypatch.setattr(sys, "argv", ["download_completion_models.py", "--models", "b2"])
    download.main()
    assert len(calls) == 1
    restored = json.loads((directory / "verified-inventory.json").read_text())
    assert download.inventory_is_verified(directory, restored, "test/pinned-model", "test-revision", expected_sha)


@pytest.mark.parametrize("field,value", [("model_id", "different/model"), ("revision", "different-revision"),
                                         ("weight_pin", "0" * 64)])
def test_self_consistent_inventory_cannot_bypass_publisher_pins(tmp_path, field, value):
    directory, inventory, weights = metadata_only(tmp_path)
    (directory / "model.safetensors").write_bytes(weights)
    expected = hashlib.sha256(weights).hexdigest()
    if field == "weight_pin":
        expected = value
    else:
        inventory[field] = value
    assert not download.inventory_is_verified(directory, inventory, "test/pinned-model", "test-revision", expected)
