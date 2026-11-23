"""Restore guard checks protect existing work and reject escaping archives."""
import importlib.util
from io import BytesIO
from pathlib import Path
import zipfile
import tarfile

import pytest

spec = importlib.util.spec_from_file_location("delivery_restore", Path(__file__).parents[1] / "scripts/restore_reproduction_inputs.py")
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


def test_archive_parent_traversal_is_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.txt", b"bad")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        restore.unpack(archive, tmp_path / "workspace")
    assert not (tmp_path / "outside.txt").exists()


def test_existing_different_file_is_preserved(tmp_path):
    target = tmp_path / "keep.txt"
    target.write_bytes(b"original")
    with pytest.raises(ValueError, match="Refusing to replace"):
        restore.write_member(tmp_path, "keep.txt", BytesIO(b"different"))
    assert target.read_bytes() == b"original"
    assert not (tmp_path / "keep.txt.restore-tmp").exists()


def test_identical_restore_is_idempotent(tmp_path):
    restore.write_member(tmp_path, "nested/data.txt", BytesIO(b"same"))
    restore.write_member(tmp_path, "nested/data.txt", BytesIO(b"same"))
    assert (tmp_path / "nested/data.txt").read_bytes() == b"same"


def test_role_limited_lookup_does_not_match_another_roles_readme(tmp_path):
    role = tmp_path / "06-M3-B5-HMM"
    role.mkdir()
    (role / "README.txt").write_text("M3 guide")
    (role / "model.zip").write_bytes(b"model")
    assert restore.find_payload(role, "TURAS-Restricted-Reproduction/06-M3-B5-HMM/model.zip") == role / "model.zip"
    with pytest.raises(FileNotFoundError):
        restore.find_payload(role, "TURAS-Restricted-Reproduction/05-M4-M5-MODELS-AND-RUNS/supplement/README.txt")


def test_m45_restore_preserves_current_report_and_sealed_archive_copy(tmp_path):
    root = tmp_path / "workspace"
    (root / "results/final").mkdir(parents=True)
    (root / "results/final/report.md").write_bytes(b"current report")
    archive = tmp_path / "m45.tar"
    with tarfile.open(archive, "w") as t:
        for name, data in [("results/final/report.md", b"sealed report"), ("artifacts/model.bin", b"weights")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, BytesIO(data))
    restore.unpack(archive, root, preserve_m45_metadata=True)
    assert (root / "results/final/report.md").read_bytes() == b"current report"
    assert (root / "artifacts/owner-m45/delivery-metadata/results/final/report.md").read_bytes() == b"sealed report"
    assert (root / "artifacts/model.bin").read_bytes() == b"weights"


def test_archive_target_outside_workspace_is_rejected(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("model.bin", b"weights")
    record = {"restore_action": "archive", "restore_target": "../escaped", "restore_role": "gmm"}
    with pytest.raises(ValueError, match="Unsafe archive path"):
        restore.restore_record(root, record, archive)
    assert not (tmp_path / "escaped").exists()


def test_unknown_restore_action_is_rejected(tmp_path):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"data")
    record = {"restore_action": "move", "restore_target": "artifacts/payload.bin", "restore_role": "gmm"}
    with pytest.raises(ValueError, match="Unknown restore_action"):
        restore.restore_record(tmp_path, record, source)
    assert not (tmp_path / "artifacts").exists()


def test_tar_non_regular_member_is_rejected(tmp_path):
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as t:
        info = tarfile.TarInfo("artifacts/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside"
        t.addfile(info)
    with pytest.raises(ValueError, match="Only regular files"):
        restore.unpack(archive, tmp_path / "workspace")


def test_zip_symlink_is_rejected(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo("artifacts/link")
        info.external_attr = (0o120777 << 16)
        z.writestr(info, "../../outside")
    with pytest.raises(ValueError, match="Symbolic links"):
        restore.unpack(archive, tmp_path / "workspace")
