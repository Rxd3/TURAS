"""Controller status remains complete when Windows denies a temporary rename."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("completion_state", Path(__file__).parents[1] / "scripts/completion_job_state.py")
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


def test_transient_reader_lock_retries_without_publishing_partial_json(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    path.write_text('{"state":"previous"}')
    real = state.os.replace
    calls = 0

    def sometimes_locked(source, target):
        nonlocal calls
        calls += 1
        if calls < 3:
            assert json.loads(path.read_text()) == {"state": "previous"}
            raise PermissionError("Windows reader holds target")
        return real(source, target)

    monkeypatch.setattr(state.os, "replace", sometimes_locked)
    state.write_job_state(path, {"state": "completed"}, retry_delay=0)
    assert calls == 3
    assert json.loads(path.read_text()) == {"state": "completed"}
    assert not list(tmp_path.glob("*.tmp"))


def test_persistent_lock_raises_and_preserves_previous_status(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    path.write_text('{"state":"previous"}')

    def locked(source, target):
        raise PermissionError("Persistent filesystem problem")

    monkeypatch.setattr(state.os, "replace", locked)
    with pytest.raises(PermissionError):
        state.write_job_state(path, {"state": "completed"}, attempts=2, retry_delay=0)
    assert json.loads(path.read_text()) == {"state": "previous"}
    assert not list(tmp_path.glob("*.tmp"))
