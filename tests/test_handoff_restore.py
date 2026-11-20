"""Delivery extraction must not escape or silently overwrite a source checkout."""
import io
from pathlib import Path
import runpy
import tarfile

import pytest

extract_delivery = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/m45_accept_handoff.py'))['extract_delivery']


def make_archive(path, members):
    with tarfile.open(path, 'w') as archive:
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_traversal_refused_before_any_files_are_extracted(tmp_path):
    archive = tmp_path/'bad.tar'
    make_archive(archive, [('safe.txt', b'safe'), ('../outside.txt', b'escape')])
    destination = tmp_path/'restore'
    destination.mkdir()
    with pytest.raises(ValueError, match='unsafe archive member'):
        extract_delivery(archive, destination)
    assert list(destination.iterdir()) == []
    assert not (tmp_path/'outside.txt').exists()


def test_identical_checkout_files_preserved_and_conflicts_refused(tmp_path):
    archive = tmp_path/'delivery.tar'
    make_archive(archive, [('protocol.json', b'frozen'), ('models/readme.txt', b'weights supplied separately')])
    destination = tmp_path/'restore'
    destination.mkdir()
    (destination/'protocol.json').write_bytes(b'other')
    with pytest.raises(ValueError, match='overwrite different checkout bytes'):
        extract_delivery(archive, destination)
    assert not (destination/'models').exists()
    (destination/'protocol.json').write_bytes(b'frozen')
    extract_delivery(archive, destination)
    assert (destination/'protocol.json').read_bytes() == b'frozen'
    assert (destination/'models/readme.txt').read_bytes() == b'weights supplied separately'
