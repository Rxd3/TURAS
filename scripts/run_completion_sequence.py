"""Run baseline jobs serially and retain child PIDs, exit codes and logs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from completion_job_state import write_job_state

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/completion-2026-11-22"


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--systems", nargs="+", required=True)
    p.add_argument("--modes", nargs="+", required=True)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--name", required=True)
    args = p.parse_args()
    job_dir = BASE / "jobs" / args.name
    job_dir.mkdir(parents=True, exist_ok=True)
    state = {"state": "running", "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": []}

    def save():
        write_job_state(job_dir / "status.json", state)

    save()
    for mode in args.modes:
        for system in args.systems:
            command = [str(args.python.resolve()), "-u", str(ROOT / "scripts/run_completion_baseline.py"), "--system", system, "--mode", mode]
            with (job_dir / f"{system}-{mode}.log").open("a", encoding="utf-8") as stdout, (job_dir / f"{system}-{mode}-errors.log").open("a", encoding="utf-8") as stderr:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
                record = {"system": system, "mode": mode, "pid": child.pid, "state": "running"}
                state["jobs"].append(record)
                save()
                print(f"START {system} {mode} PID {child.pid}", flush=True)
                code = child.wait()
                record.update(state="completed" if code == 0 else "failed", exit_code=code)
                save()
                if code:
                    state["state"] = "failed"
                    save()
                    raise SystemExit(code)
    state.update(state="completed", ended_at_utc=datetime.now(timezone.utc).isoformat())
    save()
    print("SEQUENCE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
