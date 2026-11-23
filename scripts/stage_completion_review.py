"""Wait for the live study, then stage verified aggregates and a deck for visual QA.

This never uploads, commits, opens a PR or claims the goal complete. A successful
exit leaves the slide images/package final review to the reviewing agent.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from completion_job_state import write_job_state

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/completion-2026-11-22"
OUT = BASE / "jobs/closeout"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"state": "waiting_for_study", "pid": os.getpid(), "jobs": []}

    def save():
        write_job_state(OUT / "status.json", state)

    save()
    try:
        while True:
            study = json.loads((BASE / "jobs/dispatcher/status.json").read_text(encoding="utf-8"))
            if study["state"] == "completed":
                break
            pid = study["pid"]
            try:
                process = psutil.Process(pid)
                alive = process.is_running() and any(Path(a).name == "dispatch_completion_study.py" for a in process.cmdline())
            except psutil.Error:
                alive = False
            if study["state"] == "failed" or not alive:
                raise RuntimeError("Study is terminal before completion; inspect retained dispatcher evidence")
            state.update(observed_live_study_pid=pid, observed_at_utc=datetime.now(timezone.utc).isoformat())
            save()
            time.sleep(10)
        for name, python, script, args in [
            ("verify-and-aggregate", BASE / "envs/neural/Scripts/python.exe", "finalize_completion_study.py", []),
            ("public-aggregate-verifier", Path(sys.executable), "verify_completion_submission.py", []),
        ]:
            with (OUT / f"{name}.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen([str(python), "-X", "utf8", str(ROOT / "scripts" / script), *args], cwd=ROOT,
                                         stdout=log, stderr=subprocess.STDOUT)
                job = {"name": name, "pid": child.pid, "state": "running"}
                state["jobs"].append(job)
                state["state"] = "staging"
                save()
                code = child.wait()
                job.update(state="completed" if code == 0 else "failed", exit_code=code)
                save()
                if code:
                    raise RuntimeError(f"Closeout step failed: {name}")
        state.update(state="awaiting_visual_and_delivery_review", completed_at_utc=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as error:
        state.update(state="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
