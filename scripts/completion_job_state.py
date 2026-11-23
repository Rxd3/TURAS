"""Publish controller status atomically despite short Windows reader locks."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import time


def write_job_state(path, state, *, attempts=100, retry_delay=0.05):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        json.dump(state, f, indent=2)
        f.write("\n")
        temporary = Path(f.name)
    try:
        for attempt in range(attempts):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt + 1 == attempts:
                    raise
                time.sleep(retry_delay)
    finally:
        temporary.unlink(missing_ok=True)
