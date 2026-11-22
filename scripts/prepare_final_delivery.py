"""Prepare access-separated local upload folders; never uploads or changes cloud ACLs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "TURAS-teammate-handoff-2026-11-22"
OUT = ROOT.parent / "TURAS-FINAL-DELIVERY-2026-11-22"
PUBLIC = OUT / "TURAS-Public-Submission"
PRIVATE = OUT / "TURAS-Restricted-Reproduction"
RELEASE = "turas-tr-econ-legis-v0.1"
COMPLETION = ROOT / "artifacts/completion-2026-11-22"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def main():
    global OUT, PUBLIC, PRIVATE
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--completion", action="store_true", help="Include verified completed B2/B3/transducer/GMM follow-up packages")
    args = parser.parse_args()
    selected_output = args.output or (ROOT.parent / "TURAS-FINAL-DELIVERY-2026-11-22-v2" if args.completion else OUT)
    OUT = selected_output.resolve()
    if args.completion and OUT == (ROOT.parent / "TURAS-FINAL-DELIVERY-2026-11-22").resolve():
        raise ValueError("Use the v2 output directory; the earlier delivery is a preserved snapshot")
    PUBLIC, PRIVATE = OUT / "TURAS-Public-Submission", OUT / "TURAS-Restricted-Reproduction"
    completion = None
    if args.completion:
        completion = json.loads((ROOT / "results/final/completion_test_summary.json").read_text(encoding="utf-8"))
        if completion["state"] != "completed" or len(completion["systems"]) != 4:
            raise ValueError("All four completed follow-up systems are required")
        for key, system in completion["systems"].items():
            for mode, evidence in system["exports"].items():
                for name, digest in evidence["files"].items():
                    if sha(COMPLETION / "runs" / key / mode / name) != digest:
                        raise ValueError("Completed follow-up exports changed before packaging")
    OUT.mkdir(exist_ok=True)
    PUBLIC.mkdir(exist_ok=True)
    PRIVATE.mkdir(exist_ok=True)
    records = []
    old_index = json.loads((OLD / "00-GITHUB-READY/handoff-manifest.json").read_text())
    expected = {r["path"]: r["sha256"] for r in old_index["files"]}

    def deliver(source, destination, audience, *, role=None, action=None, target=None, optional=False):
        source, destination = Path(source), Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_sha = sha(source)
        if source.is_relative_to(OLD):
            wanted = expected.get(source.relative_to(OLD).as_posix())
            if wanted and source_sha != wanted:
                raise ValueError(f"Original source changed: {source.name}")
        if destination.exists():
            if sha(destination) != source_sha:
                if destination.is_relative_to(PUBLIC):
                    shutil.copy2(source, destination)
                else:
                    raise ValueError(f"Refusing to replace different delivery file: {destination}")
        else:
            shutil.copy2(source, destination)
        if sha(destination) != source_sha:
            raise ValueError("Copy verification failed")
        record = {"path": destination.relative_to(OUT).as_posix(), "bytes": destination.stat().st_size,
                  "sha256": source_sha, "audience": audience, "optional_for_score_inference": optional}
        if role:
            record.update(restore_role=role, restore_action=action, restore_target=target)
        records.append(record)
        print(f"VERIFIED {record['path']}", flush=True)

    mapping = [
        ("01-INFERENCE-M2-M3-M4-M5-M6", "01-AUDIO-INFERENCE", "authorized model users M2-M6", "inference", f"data/private/{RELEASE}/inference"),
        ("02-TRAIN-REFERENCES-M3-M4-M5", "02-TRAIN-REFERENCES", "authorized training users M3/M4/M5", "train", f"data/private/{RELEASE}/train"),
        ("03-VALIDATION-SELECTORS-ONLY", "03-VALIDATION-SELECTION", "authorized validation selectors/evaluators", "validation", f"data/private/{RELEASE}/validation-import"),
        ("04-EVALUATOR-M6-ONLY", "04-EVALUATOR-ONLY", "M6 and authorized offline evaluators", "evaluator", f"data/private/{RELEASE}/evaluator-import"),
        ("05-M4-M5-MODELS-AND-RUNS", "05-M4-M5-MODELS-AND-RUNS", "authorized reproduction reviewers", "m45", "."),
        ("06-M1-REVIEW-ASSIGNED-LISTENERS-ONLY", "08-M1-ASSIGNED-LISTENERS", "owner and assigned independent listeners", None, None),
        ("07-OPTIONAL-EXPERIMENT-EVIDENCE-M4-M5-M6", "09-OPTIONAL-TRAINING-HISTORY", "authorized training reviewers", None, None),
        ("09-OPTIONAL-EARLIER-CHECKPOINTS-M4-M5", "09-OPTIONAL-TRAINING-HISTORY", "authorized training reviewers", None, None),
    ]
    for old_name, new_name, audience, role, target in mapping:
        folder = PRIVATE / new_name
        folder.mkdir(exist_ok=True)
        for source in sorted((OLD / old_name).rglob("*")):
            if not source.is_file() or source.name in {"SHA256SUMS", "RECIPIENT-README.txt"}:
                continue
            relative = source.relative_to(OLD / old_name)
            actual_role, action, actual_target = role, "archive" if source.suffix == ".tar" else None, target
            if old_name.startswith("03") and "RELATIONAL" in source.name:
                actual_role, action, actual_target = None, None, None
            if old_name.startswith("05") and source.name == "m45-inference-index.json":
                action, actual_target = "copy", "artifacts/owner-m45/final/inference-index.json"
            elif old_name.startswith("05") and source.suffix != ".tar":
                actual_role, action, actual_target = None, None, None
            deliver(source, folder / relative, audience, role=actual_role, action=action, target=actual_target,
                    optional=new_name.startswith("09"))
        (folder / "README.txt").write_text(
            f"Audience: {audience}.\nOriginal archive bytes are unchanged. Verify SHA256SUMS before extracting.\n"
            "Keep this folder's access separate from other roles. Never place references in the audio/inference root.\n"
            "See the public docs/final_reproduction.md for exact restore commands.\n", encoding="utf-8")

    m3 = PRIVATE / "06-M3-B5-HMM"
    deliver(ROOT / "artifacts/m3-final-2026-11-22/TURAS-M3-final-2026-11-22.zip", m3 / "TURAS-M3-final-2026-11-22.zip",
            "authorized M3/M6/reproduction reviewers", role="m3", action="archive", target="artifacts/m3-final-2026-11-22")
    deliver(ROOT / "artifacts/final-review-build/m3-runtime-source.zip", m3 / "m3-runtime-source.zip", "authorized reproduction reviewers")
    deliver(ROOT / "artifacts/b5-gpu-resume-2026-11-22/status.json", m3 / "b5-training-status.json", "authorized reproduction reviewers",
            role="m3", action="copy", target="artifacts/b5-gpu-resume-2026-11-22/status.json")
    (m3 / "README.txt").write_text(
        "B5 epoch-22 selected weights, supplied single-Gaussian character HMM, full predictions and timing.\n"
        "m3-runtime-source.zip preserves exact d2bdf5e runtime source plus the M3 additions.\n"
        "Extract that source separately for exact-runtime replay; do not replace frozen source silently.\n"
        "The final-results ZIP is reference-free and has no feature caches.\n"
        "The original epoch-13 input and latest optimizer state are in optional folder 09.\n", encoding="utf-8")
    if completion:
        deliver(COMPLETION / "reproduction/fresh-environment-b5-restore.json", m3 / "fresh-environment-b5-restore.json",
                "authorized reproduction reviewers")
        deliver(COMPLETION / "reproduction/m45-fresh-restore/acceptance.json",
                PRIVATE / "05-M4-M5-MODELS-AND-RUNS/fresh-m45-restored-execution.json", "authorized reproduction reviewers")
        build_completion_packages(deliver, completion)
    history = PRIVATE / "09-OPTIONAL-TRAINING-HISTORY/M3-continuation"
    for source, name in [(ROOT / "artifacts/b5-gpu-resume-2026-11-22/source/b5_epoch13_handoff.zip", "b5_epoch13_handoff.zip"),
                         (ROOT / "artifacts/b5-gpu-resume-2026-11-22/latest-training.pt", "latest-training-epoch28.pt"),
                         (ROOT / "artifacts/b5-gpu-resume-2026-11-22/training-history.json", "training-history.json")]:
        deliver(source, history / name, "authorized continuation users", optional=True)

    presentation = PUBLIC / "07-PRESENTATION-AND-REPORTS"
    deck = "TURAS-final-2-minute-v2.pptx" if completion else "TURAS-final-2-minute.pptx"
    for source in [ROOT / "presentation" / deck, ROOT / "presentation/TURAS-two-minute-speaker-script.txt",
                   ROOT / "docs/final_review_2026-11-22.md", ROOT / "results/final/M4_M5_FINAL_REPORT.md",
                   ROOT / "results/final/m3_test_summary.json", ROOT / "results/final/m45_metrics.csv",
                   ROOT / "results/final/final_review_checks.json"]:
        deliver(source, presentation / source.name, "public submission readers/graders")
    if completion:
        for name in ["completion_test_summary.json", "completion_metrics.csv", "BASELINE_COMPLETION_REPORT.md", "fresh_environment_restoration.json", "completion_hardware_context.json"]:
            deliver(ROOT / "results/final" / name, presentation / name, "public submission readers/graders")
        deliver(ROOT / "docs/baseline_completion_followup.md", presentation / "baseline_completion_followup.md", "public submission readers/graders")
    (presentation / "README.txt").write_text(
        f"Use {deck} for the two-minute English technical presentation.\n"
        "The roughly 200-word speaker script and source citations are included.\n"
        "The final review lists the remaining requirements; this is not an unconditional full-project completion claim.\n", encoding="utf-8")

    start = PUBLIC / "00-START-HERE"
    start.mkdir(exist_ok=True)
    source_zip = start / "TURAS-submission-source.zip"
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        source_inventory = {}
        for name in sorted(set(names)):
            source = ROOT / name
            if not source.is_file():
                continue
            if name.startswith(("data/private/", "artifacts/", ".venv/")):
                raise ValueError(f"Private path reached source inventory: {name}")
            z.write(source, name)
            source_inventory[name] = {"bytes": source.stat().st_size, "sha256": sha(source)}
        z.writestr("SOURCE-SNAPSHOT.json", json.dumps({"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "local_closeout_changes_included": True, "publication_or_review_claimed": False,
                    "files": source_inventory}, indent=2))
    record = {"path": source_zip.relative_to(OUT).as_posix(), "bytes": source_zip.stat().st_size,
              "sha256": sha(source_zip), "audience": "public source readers", "optional_for_score_inference": False}
    records.append(record)

    manifest = {"schema_version": "0.1", "prepared_date": "2026-11-22", "dataset_release": RELEASE,
                "split_manifest_sha256": "a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8",
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "cloud_upload_or_access_change_claimed": False, "consumer_receipt_or_independent_approval_claimed": False,
                "files": records, "missing_packages": [] if completion else [{"name": "turas-b2-whisper-large-v3-m2-handoff.zip", "bytes_reported": 942095,
                    "sha256_reported": "fd8471e9084a369684f87cccd92e2d0fde1550a725e5dc0f6f854669ffa7e41b", "status": "M2 branch reports completion; package absent locally"}]}
    (start / "package-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(ROOT / "scripts/verify_final_packages.py", start / "verify_packages.py")
    shutil.copy2(ROOT / "docs/drive_packages.md", start / "DRIVE-FOLDER-MAP.md")
    shutil.copy2(ROOT / "docs/final_reproduction.md", start / "REPRODUCTION.md")
    if completion:
        shutil.copy2(ROOT / "docs/completion_replay_commands.md", start / "FOLLOWUP-REPLAY-COMMANDS.md")
    (start / "README.txt").write_text(
        "1. Upload this public root and the presentation/report folder to TURAS-Public-Submission.\n"
        "2. Keep TURAS-Restricted-Reproduction owner-only at its root. Share each required child folder with named accounts.\n"
        "3. Verify downloaded payloads: python verify_packages.py --root PATH --manifest package-manifest.json --allow-missing\n"
        "4. Extract TURAS-submission-source.zip for code, docs, public aggregates and restore/scoring utilities.\n"
        "5. After upload, run scripts/update_artifact_links.py with the real public folder URL and submit the code/docs through review.\n"
        "The package map and final review explain access, exact experiment coverage and remaining human sign-off.\n", encoding="utf-8")
    for folder in [start, presentation, *[p for p in PRIVATE.iterdir() if p.is_dir()]]:
        payloads = [r for r in records if (OUT / r["path"]).is_relative_to(folder)]
        (folder / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {(OUT/r['path']).relative_to(folder).as_posix()}\n" for r in payloads), encoding="utf-8")
    if not completion:
        pending = PRIVATE / "10-PENDING-M2-B2"
        pending.mkdir(exist_ok=True)
        (pending / "README.txt").write_text(
            "Not a completed local package. Obtain turas-b2-whisper-large-v3-m2-handoff.zip from M2.\n"
            "Reported size 942095; reported SHA256 fd8471e9084a369684f87cccd92e2d0fde1550a725e5dc0f6f854669ffa7e41b.\n"
            "M6 must validate and score it and arrange standardized repeated timing.\n", encoding="utf-8")
    (OUT / "START-HERE.txt").write_text(
        "Upload the TWO roots separately. Do not publicly share the restricted root or combine all roles into one archive.\n"
        "Start with TURAS-Public-Submission/00-START-HERE/README.txt.\n"
        "The final two-minute presentation is in TURAS-Public-Submission/07-PRESENTATION-AND-REPORTS/.\n"
        "This delivery has real package hashes and a final-review checklist; cloud upload and project-wide approval are pending.\n", encoding="utf-8")
    print(f"COMPLETE {len(records)} payloads, {sum(r['bytes'] for r in records)/1e9:.2f} GB: {OUT}", flush=True)


def build_completion_packages(deliver, report):
    """Only explicit reference-free run/model whitelists can enter these archives."""
    selected = report["b3_selection"]["selected_system"]
    specs = [
        ("10-B2-WHISPER-LARGE-V3", "b2", ["b2"], "TURAS-B2-GPU-followup.zip"),
        ("11-B3-QWEN", "b3", ["qwen17", "qwen06"], "TURAS-B3-Qwen-followup.zip"),
        ("12-TURKISH-TRANSDUCER", "transducer", ["seda"], "TURAS-Seda-RNNT-followup.zip"),
        ("13-MIXTURE-GMM-HMM", "gmm", ["gmm"], "TURAS-4mixture-GMM-HMM-followup.zip"),
    ]
    packages = COMPLETION / "packages"
    packages.mkdir(exist_ok=True)
    for folder_name, role, systems, archive_name in specs:
        sources = set()
        for system in systems:
            modes = ["probe", "validation", "test", "timing"] if system != ("qwen06" if selected == "qwen17" else "qwen17") else ["probe", "validation"]
            for mode in modes:
                directory = COMPLETION / "runs" / system / mode
                sources.update(p for p in directory.iterdir() if p.is_file() and p.suffix in [".json", ".jsonl"])
            for split in ["validation", "test"]:
                score = COMPLETION / "scores" / split / f"{system}.json"
                if score.exists():
                    sources.add(score)
            model = COMPLETION / "models" / system
            if model.exists():
                # Large upstream weights are pinned downloads, not duplicate Drive payloads.
                sources.update(p for p in model.iterdir() if p.is_file() and p.suffix in [".json", ".txt", ".jinja", ".md"] and "fp32" not in p.name)
                if system == "seda":
                    sources.update(model / name for name in ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx"])
        if role == "b3":
            sources.add(COMPLETION / "b3-selection.json")
        if role == "gmm":
            sources.update(COMPLETION / "gmm" / name for name in ["checkpoint.npz", "model-manifest.json", "frozen-training.json",
                           "training-history.json", "training-coverage.json"])
        archive = packages / archive_name
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for source in sorted(sources):
                z.write(source, source.relative_to(ROOT).as_posix())
            for source, name in [(ROOT / "licenses/Apache-2.0.txt", "Apache-2.0.txt"),
                                 (ROOT / "licenses/OpenAI-Whisper-MIT.txt", "OpenAI-Whisper-MIT.txt"),
                                 (ROOT / "LICENSE", "TURAS-MIT.txt"), (ROOT / "NOTICE", "THIRD-PARTY-NOTICE.txt")]:
                z.write(source, "artifacts/completion-2026-11-22/licenses/" + name)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Archive CRC failed")
        folder = PRIVATE / folder_name
        deliver(archive, folder / archive_name, "authorized reproduction reviewers", role=role, action="archive", target=".")
        neural = role in ["b2", "b3"]
        (folder / "README.txt").write_text(
            f"Restore role: {role}. Use scripts/restore_reproduction_inputs.py with --roles {role}.\n"
            "All 438 validation and 636 test predictions plus frozen 107-clip, three-repeat timing are included for the evaluated system.\n"
            + ("Both Qwen validation candidates and the frozen selection are preserved; only the winner has test/timing results.\n" if role == "b3" else "")
            + ("Upstream model weights are not duplicated here. Run scripts/download_completion_models.py for the pinned revisions; full expected hashes are included.\n" if neural else "Selected model weights are included.\n")
            + "These are separate follow-up experiments on working edited-minutes references. No gold references or feature caches are included.\n"
            "Use the public source ZIP for exact runtime source, environment locks and commands.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
