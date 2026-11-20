"""Synthetic tar fixtures only; no private release contents are used."""
import io
import json
import tarfile

import pytest

from turkish_asr_rdl.training.acoustic import sha256_file
from turkish_asr_rdl.utils.bundles import stage_bundle


def archive_file(tmp_path, name, payload):
    path = tmp_path / "fixture.tar"
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo(name)
        data = payload.encode()
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    return path


def stage(path, destination, *, digest=None):
    return stage_bundle(path, destination, expected_sha256=digest or sha256_file(path), kind="train",
                        release="fixture-only", split_sha256="a" * 64, train_ids={"fixture-id"})


def test_bad_hash_and_traversal_write_nothing(tmp_path):
    path = archive_file(tmp_path, "../escape.jsonl", "{}")
    output = tmp_path / "staged"
    with pytest.raises(ValueError, match="SHA-256"):
        stage(path, output, digest="0" * 64)
    assert not output.exists()
    with pytest.raises(ValueError, match="unsafe"):
        stage(path, output)
    assert not output.exists() and not (tmp_path / "escape.jsonl").exists()


def test_train_bundle_checks_ids_and_never_overwrites(tmp_path):
    row = dict(utterance_id="fixture-id", split="train", dataset_release="fixture-only", reference_text="synthetic training fixture")
    path = archive_file(tmp_path, "release/references.train.jsonl", json.dumps(row) + "\n")
    output = tmp_path / "staged"
    report = stage(path, output)
    assert report["status"] == "staged_contract_checked" and report["jsonl_files"][0]["rows"] == 1
    with pytest.raises(ValueError, match="never overwritten"):
        stage(path, output)


def test_test_references_cannot_be_staged_as_training(tmp_path):
    row = dict(utterance_id="fixture-id", split="test", dataset_release="fixture-only", reference_text="forbidden synthetic fixture")
    path = archive_file(tmp_path, "references.test.jsonl", json.dumps(row) + "\n")
    output = tmp_path / "staged"
    with pytest.raises(ValueError, match="non-train"):
        stage(path, output)
    report = json.loads((output / ".turas-stage-audit.json").read_text())
    assert report["status"] == "failed_validation"


def test_published_extensionless_checksum_inventory_is_allowed(tmp_path):
    row = dict(utterance_id="fixture-id", split="train", dataset_release="fixture-only", reference_text="synthetic fixture")
    path = archive_file(tmp_path, "release/references.train.jsonl", json.dumps(row) + "\n")
    with tarfile.open(path, "a") as archive:
        member = tarfile.TarInfo("release/SHA256SUMS")
        content = b"synthetic fixture checksum inventory\n"
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    assert stage(path, tmp_path / "staged")["payload_files"] == 2
