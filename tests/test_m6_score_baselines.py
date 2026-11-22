"""Tests for scripts/m6_score_baselines.py on tiny synthetic runs (no real data)."""
from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from turkish_asr_rdl.decoding.export import RunContext, nbest_row, prediction_row
from turkish_asr_rdl.decoding.interface import Candidate, Hypothesis, InferenceInput

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "m6_score_baselines.py"
LEXICON = ROOT / "configs" / "datasets" / "turas-domain-lexicon-v0.1.json"
RELEASE = "rel_synthetic"
SPLIT_SHA = "a" * 64
AUDIO_SHA = "b" * 64
REQUESTED_N = 2
SEED, REPLICATES = 2026, 50

# (utterance_id, leakage_group_id, reference_text, verification_status)
REFS = [
    ("utt_01", "g1", "bütçe", "verified"),
    ("utt_02", "g1", "zeytinyağı", "verified"),
    ("utt_03", "g2", "çekirge", "adjudicated"),
    ("utt_04", "g2", "pırasa", "not_verified"),
]
PERFECT = {uid: text for uid, _, text, _ in REFS}
ONE_SUB = {**PERFECT, "utt_03": "kuşkonmaz"}  # one substitution in 4 reference words
REPORT_KEYS = {
    "system_id", "split", "protocol_sha256", "reference_variant", "reference_sha256",
    "manifest_sha256", "predictions_sha256", "nbest_sha256", "metrics",
    "m1_reported_verified_subset", "domain", "nbest_coverage", "error_rows",
    "independent_jiwer_crosscheck",
}
SUMMARY_KEYS = {
    "status", "split", "protocol_sha256", "reference_sha256", "utterances",
    "recording_groups", "systems", "pairs", "agreement", "bootstrap",
}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("m6_score_baselines", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict], newline: str = "\n") -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + newline for row in rows)
    path.write_bytes(text.encode("utf-8"))


def ref_rows(split: str = "test", refs=REFS) -> list[dict]:
    return [
        {"utterance_id": uid, "audio_sha256": AUDIO_SHA, "reference_text": text,
         "leakage_group_id": group, "split": split, "dataset_release": RELEASE,
         "verification_status": status}
        for uid, group, text, status in refs
    ]


def hypothesis(uid: str, text: str | None) -> Hypothesis:
    if text is None:
        return Hypothesis(uid, "error", None, error_code="adapter_exception",
                          error_message="SYNTHETIC failure")
    kind = "seq2seq_teacher_forced_log_prob"
    candidates = (
        Candidate("c1", 1, text, (1, 2), -1.0, kind, -0.5),
        Candidate("c2", 2, text + " ek", (1, 3), -2.0, kind, -0.9),
    )
    return Hypothesis(uid, "ok", text, candidates=candidates)


def write_run(directory: Path, system: str, hyps: dict, split: str) -> Path:
    directory.mkdir(parents=True)
    ctx = RunContext(run_id=f"run_{system}", system_id=system, dataset_release=RELEASE,
                     split=split, split_manifest_sha256=SPLIT_SHA,
                     model_artifact_sha256=AUDIO_SHA, decoder_config_sha256=SPLIT_SHA)
    pairs = [
        (hypothesis(uid, text), InferenceInput(
            schema_version="0.1", dataset_release=RELEASE, split_manifest_sha256=SPLIT_SHA,
            utterance_id=uid, audio_uri=f"audio/{uid}.flac", audio_sha256=AUDIO_SHA,
            duration_s=1.0, sample_rate_hz=16000, channels=1, language="tr"))
        for uid, text in hyps.items()
    ]
    write_jsonl(directory / "nbest.jsonl",
                [nbest_row(h, i, ctx, REQUESTED_N, "nbest-synthetic-v1") for h, i in pairs])
    nbest_sha = sha(directory / "nbest.jsonl")
    write_jsonl(directory / "predictions.jsonl",
                [prediction_row(h, i, ctx, f"t_{i.utterance_id}", nbest_sha) for h, i in pairs])
    manifest = {
        "inputs": {"dataset_release": RELEASE, "split": split, "split_manifest_sha256": SPLIT_SHA},
        "outputs": {name: sha(directory / name) for name in ("predictions.jsonl", "nbest.jsonl")},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


@dataclass
class Case:
    root: Path
    protocol: Path
    refs: Path
    runs: dict[str, Path]
    split: str

    @property
    def output(self) -> Path:
        return self.root / "out"

    def argv(self, *extra: str, runs=None, protocol_sha=None, refs_sha=None) -> list[str]:
        args = ["--protocol", str(self.protocol), "--protocol-sha256", protocol_sha or sha(self.protocol),
                "--split", self.split, "--references", str(self.refs),
                "--references-sha256", refs_sha or sha(self.refs)]
        for system in self.runs if runs is None else runs:
            args += ["--run", f"{system}={self.runs[system]}"]
        return args + ["--output", str(self.output), *extra]


def build(root: Path, runs=None, refs=None, ids=None, split="test", payload_sha=None,
          lexicon_sha=None, newline="\n") -> Case:
    refs = ref_rows(split) if refs is None else refs
    ids = [r["utterance_id"] for r in refs] if ids is None else ids
    ids_path = root / ("test.txt" if split == "test" else f"data/splits/{RELEASE}/validation.txt")
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("".join(i + "\n" for i in ids), encoding="utf-8")
    refs_path = root / "references.jsonl"
    write_jsonl(refs_path, refs, newline)
    protocol = {
        "dataset_release": RELEASE, "split_manifest_sha256": SPLIT_SHA,
        "inputs": {"test_ids": str(root / "test.txt")},
        "test_reference": {"variant": "synthetic-test-v1", "payload_sha256": payload_sha or sha(refs_path)},
        "lexicon": {"path": str(LEXICON), "sha256": lexicon_sha or sha(LEXICON)},
        "bootstrap": {"seed": SEED, "replicates": REPLICATES, "method": "paired recording-group bootstrap"},
        "decoder": {"requested_n": REQUESTED_N},
    }
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    runs = {"B2": ONE_SUB} if runs is None else runs
    dirs = {name: write_run(root / "runs" / name, name, hyps, split) for name, hyps in runs.items()}
    return Case(root, protocol_path, refs_path, dirs, split)


def run(mod, argv, capsys):
    code = mod.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def refused(mod, argv, capsys, case: Case, needle: str) -> str:
    code, out, err = run(mod, argv, capsys)
    assert code == 1
    assert needle in err, err
    assert err.count("\n") == 1 and "Traceback" not in err
    assert out == ""
    assert not case.output.exists()
    return err


def report(case: Case, system: str) -> dict:
    return json.loads((case.output / f"{system}.json").read_text(encoding="utf-8"))


def summary(case: Case) -> dict:
    return json.loads((case.output / "summary.json").read_text(encoding="utf-8"))


def edit_manifest(case: Case, system: str, change) -> None:
    path = case.runs[system] / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


# --- scoring -----------------------------------------------------------------

def test_scores_match_hand_count(mod, tmp_path, capsys):
    case = build(tmp_path)
    code, _, err = run(mod, case.argv(), capsys)
    assert code == 0, err
    result = report(case, "B2")
    assert result["metrics"]["raw-v1"]["wer"] == 0.25
    assert result["metrics"]["tr-normalized-v1"]["wer"] == 0.25
    assert result["metrics"]["raw-v1"]["reference_words"] == 4
    assert result["independent_jiwer_crosscheck"] == "passed both views"
    assert set(result) == REPORT_KEYS
    assert result["split"] == "test" and result["reference_variant"] == "synthetic-test-v1"
    assert result["manifest_sha256"] == sha(case.runs["B2"] / "manifest.json")
    assert result["predictions_sha256"] == sha(case.runs["B2"] / "predictions.jsonl")
    assert result["reference_sha256"] == sha(case.refs)


def test_turkish_dotted_capital_normalized(mod, tmp_path, capsys):
    refs = ref_rows(refs=[("utt_01", "g1", "İSTANBUL", "verified")])
    case = build(tmp_path, refs=refs, runs={"B2": {"utt_01": "istanbul"}})
    code, _, err = run(mod, case.argv(), capsys)
    assert code == 0, err
    metrics = report(case, "B2")["metrics"]
    assert metrics["raw-v1"]["wer"] == 1.0
    assert metrics["tr-normalized-v1"]["wer"] == 0.0


def test_verified_subset_counts_verified_and_adjudicated(mod, tmp_path, capsys):
    case = build(tmp_path)
    assert run(mod, case.argv(), capsys)[0] == 0
    subset = report(case, "B2")["m1_reported_verified_subset"]
    assert subset["utterances"] == 3
    assert subset["metrics"]["tr-normalized-v1"]["wer"] == pytest.approx(1 / 3)


def test_term_f1_and_term_containing_subset(mod, tmp_path, capsys):
    case = build(tmp_path, runs={"B2": {**PERFECT, "utt_02": "bütçe"}})
    assert run(mod, case.argv(), capsys)[0] == 0
    domain = report(case, "B2")["domain"]
    micro = domain["micro"]
    assert (micro["tp"], micro["fp"], micro["fn"]) == (1, 1, 0)
    assert micro["f1"] == pytest.approx(2 / 3)
    containing = domain["term_containing_utterances"]
    assert containing["count"] == 1
    assert containing["metrics"]["tr-normalized-v1"]["wer"] == 0.0
    assert "term_containing_utterance_ids" not in domain


def test_error_rows_reported(mod, tmp_path, capsys):
    case = build(tmp_path, runs={"B2": {**PERFECT, "utt_04": None}})
    code, _, err = run(mod, case.argv(), capsys)
    assert code == 0, err
    result = report(case, "B2")
    assert result["error_rows"] == 1
    assert result["nbest_coverage"] == {
        "requested_n": REQUESTED_N, "actual_n_distribution": {"0": 1, "2": 3}, "hypotheses": 6}


@pytest.mark.parametrize("func, field", [("process_words", "wer"), ("process_characters", "cer")])
def test_jiwer_disagreement_refuses(mod, tmp_path, capsys, monkeypatch, func, field):
    case = build(tmp_path)
    real = getattr(mod.jiwer, func)

    def shifted(*args, **kwargs):  # within 1e-6 or math.isclose, beyond the scorer's 1e-12
        return SimpleNamespace(**{field: getattr(real(*args, **kwargs), field) + 1e-10})

    monkeypatch.setattr(mod.jiwer, func, shifted)
    refused(mod, case.argv(), capsys, case, "cross-check")


def test_validation_split_ids_path(mod, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    case = build(tmp_path, split="validation", refs=ref_rows("validation"), payload_sha="0" * 64)
    code, _, err = run(mod, case.argv(), capsys)
    assert code == 0, err
    result = report(case, "B2")
    assert result["split"] == "validation" and result["reference_variant"] is None
    assert summary(case)["split"] == "validation"


def test_validation_split_file_missing_refuses(mod, tmp_path, capsys, monkeypatch):
    case = build(tmp_path, split="validation", refs=ref_rows("validation"))
    monkeypatch.chdir(tmp_path / "runs")
    refused(mod, case.argv(), capsys, case, "file not found")


def test_crlf_references_keep_ids(mod, tmp_path, capsys):
    case = build(tmp_path, newline="\r\n")
    code, _, err = run(mod, case.argv(), capsys)
    assert code == 0, err
    assert summary(case)["utterances"] == 4


# --- pairs, bootstrap and agreement -------------------------------------------

def test_bootstrap_matches_frozen_draw(mod):
    expected = np.random.default_rng(SEED).integers(0, 2, size=(REPLICATES, 2))
    assert np.array_equal(mod.bootstrap_samples(2, SEED, REPLICATES), expected)


def test_paired_interval_zero_for_identical_runs(mod, tmp_path, capsys):
    case = build(tmp_path, runs={"B2": ONE_SUB, "B3": ONE_SUB})
    assert run(mod, case.argv("--pair", "B2=B3"), capsys)[0] == 0
    (pair,) = summary(case)["pairs"]
    assert pair["system"] == "B2" and pair["control"] == "B3"
    assert pair["normalized_wer_difference_percentage_points"] == 0.0
    assert pair["paired_recording_bootstrap_95_interval_percentage_points"] == [0.0, 0.0]


def test_paired_interval_matches_group_bootstrap(mod, tmp_path, capsys):
    case = build(tmp_path, runs={"B2": ONE_SUB, "B3": PERFECT})
    assert run(mod, case.argv("--pair", "B2=B3"), capsys)[0] == 0
    (pair,) = summary(case)["pairs"]
    errors, words = np.array([0, 1]), np.array([2, 2])  # g1 has no error, g2 one substitution
    draws = np.random.default_rng(SEED).integers(0, 2, size=(REPLICATES, 2))
    diffs = [errors[d].sum() / words[d].sum() for d in draws]
    expected = (100 * np.percentile(diffs, [2.5, 97.5])).tolist()
    assert pair["normalized_wer_difference_percentage_points"] == pytest.approx(25.0)
    assert pair["paired_recording_bootstrap_95_interval_percentage_points"] == pytest.approx(expected)
    boot = summary(case)["bootstrap"]
    assert (boot["seed"], boot["replicates"], boot["groups"]) == (SEED, REPLICATES, 2)


def test_reversed_pair_mirrors_interval(mod, tmp_path, capsys):
    # One group per utterance: with two groups every draw gives the same interval.
    refs = ref_rows(refs=[(uid, uid, text, status) for uid, _, text, status in REFS])
    case = build(tmp_path, refs=refs, runs={"B2": ONE_SUB, "B3": {**PERFECT, "utt_01": "kuşkonmaz"}})
    assert run(mod, case.argv("--pair", "B2=B3", "--pair", "B3=B2"), capsys)[0] == 0
    first, second = summary(case)["pairs"]
    assert (second["system"], second["control"]) == ("B3", "B2")
    assert second["normalized_wer_difference_percentage_points"] == pytest.approx(
        -first["normalized_wer_difference_percentage_points"])
    lo, hi = first["paired_recording_bootstrap_95_interval_percentage_points"]
    assert second["paired_recording_bootstrap_95_interval_percentage_points"] == pytest.approx([-hi, -lo])


def test_agreement_counts(mod, tmp_path, capsys):
    rows = [{"utterance_id": uid, "hypothesis_text": text} for uid, text in PERFECT.items()]
    other = [{**row, "hypothesis_text": ONE_SUB[row["utterance_id"]]} for row in rows]
    assert mod.one_best_agreement(rows, list(reversed(other))) == {
        "utterances": 4, "hypothesis_byte_equal": 3, "differ": 1}
    case = build(tmp_path, runs={"B2": ONE_SUB, "B3": PERFECT})
    assert run(mod, case.argv("--pair", "B2=B3"), capsys)[0] == 0
    assert summary(case)["agreement"] == [
        {"system": "B2", "control": "B3", "utterances": 4, "hypothesis_byte_equal": 3, "differ": 1}]


def test_agreement_refuses_different_ids(mod):
    with pytest.raises(ValueError):
        mod.one_best_agreement([{"utterance_id": "a", "hypothesis_text": "x"}],
                               [{"utterance_id": "b", "hypothesis_text": "x"}])


# --- outputs -------------------------------------------------------------------

def test_summary_and_printed_hashes(mod, tmp_path, capsys):
    case = build(tmp_path, runs={"B2": ONE_SUB, "B3": PERFECT})
    code, out, _ = run(mod, case.argv(), capsys)
    assert code == 0
    result = summary(case)
    assert set(result) == SUMMARY_KEYS
    assert result["status"] == "m6_baseline_scoring_complete"
    assert (result["utterances"], result["recording_groups"]) == (4, 2)
    assert [s["system_id"] for s in result["systems"]] == ["B2", "B3"]
    assert result["pairs"] == [] and result["agreement"] == []
    lines = [json.loads(line) for line in out.splitlines()]
    assert lines[:2] == [{"scored_system": "B2", "rows": 4, "normalized_wer": 0.25},
                         {"scored_system": "B3", "rows": 4, "normalized_wer": 0.0}]
    written = {Path(line["written"]).name: line["sha256"] for line in lines[2:]}
    assert written == {name: sha(case.output / name) for name in ("B2.json", "B3.json", "summary.json")}


def test_no_text_in_outputs(mod, tmp_path, capsys):
    case = build(tmp_path, runs={"B2": ONE_SUB, "B3": PERFECT})
    code, out, err = run(mod, case.argv("--pair", "B2=B3"), capsys)
    assert code == 0
    texts = set(PERFECT.values()) | set(ONE_SUB.values())
    blobs = [p.read_text(encoding="utf-8") for p in case.output.iterdir()] + [out, err]
    escaped = {json.dumps(t)[1:-1] for t in texts}  # also catch \u-escaped copies
    assert not [t for t in texts | escaped for blob in blobs if t in blob]


# --- refusals ------------------------------------------------------------------

def test_refuses_protocol_checksum(mod, tmp_path, capsys):
    case = build(tmp_path)
    refused(mod, case.argv(protocol_sha="0" * 64), capsys, case, "protocol checksum")


def test_refuses_reference_byte_change(mod, tmp_path, capsys):
    case = build(tmp_path)
    digest = sha(case.refs)
    case.refs.write_bytes(case.refs.read_bytes().replace(b"utt_04", b"utt_05"))
    refused(mod, case.argv(refs_sha=digest), capsys, case, "reference checksum")


def test_refuses_test_sha_not_protocol(mod, tmp_path, capsys):
    case = build(tmp_path, payload_sha="0" * 64)
    refused(mod, case.argv(), capsys, case, "protocol test reference")


def test_refuses_reference_order_swap(mod, tmp_path, capsys):
    refs = ref_rows()
    ids = [r["utterance_id"] for r in refs]
    refs[0], refs[1] = refs[1], refs[0]
    case = build(tmp_path, refs=refs, ids=ids)
    refused(mod, case.argv(), capsys, case, "identity mismatch")


def test_refuses_reference_without_utterance_id(mod, tmp_path, capsys):
    refs = ref_rows()
    del refs[2]["utterance_id"]
    case = build(tmp_path, refs=refs, ids=list(PERFECT))
    refused(mod, case.argv(), capsys, case, "identity mismatch")


def test_refuses_lexicon_checksum(mod, tmp_path, capsys):
    case = build(tmp_path, lexicon_sha="0" * 64)
    refused(mod, case.argv(), capsys, case, "lexicon checksum")


def test_refuses_references_with_bom(mod, tmp_path, capsys):
    case = build(tmp_path)
    case.refs.write_bytes(b"\xef\xbb\xbf" + case.refs.read_bytes())
    refs_sha = sha(case.refs)
    protocol = json.loads(case.protocol.read_text(encoding="utf-8"))
    protocol["test_reference"]["payload_sha256"] = refs_sha
    case.protocol.write_text(json.dumps(protocol), encoding="utf-8")
    refused(mod, case.argv(), capsys, case, "references.jsonl")


@pytest.mark.parametrize("name", ["predictions.jsonl", "nbest.jsonl"])
def test_refuses_stale_manifest_output(mod, tmp_path, capsys, name):
    case = build(tmp_path)
    path = case.runs["B2"] / name
    path.write_bytes(path.read_bytes().replace("kuşkonmaz".encode(), "çekirge".encode()))
    refused(mod, case.argv(), capsys, case, f"{name} changed after its manifest")


def test_refuses_missing_id_in_run(mod, tmp_path, capsys):
    hyps = {uid: text for uid, text in ONE_SUB.items() if uid != "utt_04"}
    case = build(tmp_path, runs={"B2": hyps})
    refused(mod, case.argv(), capsys, case, "B2: predictions.jsonl fails the contract")


def test_refuses_nbest_contract_failure(mod, tmp_path, capsys):
    case = build(tmp_path)
    path = case.runs["B2"] / "nbest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    del rows[0]["candidates"]
    write_jsonl(path, rows)
    edit_manifest(case, "B2", lambda m: m["outputs"].update({"nbest.jsonl": sha(path)}))
    refused(mod, case.argv(), capsys, case, "B2: nbest.jsonl fails the contract")


@pytest.mark.parametrize("change, needle", [
    (lambda m: m.pop("outputs"), "no outputs map"),
    (lambda m: m.pop("inputs"), "no inputs map"),
    (lambda m: m["outputs"].pop("predictions.jsonl"), "lists no hash for predictions.jsonl"),
    (lambda m: m["outputs"].pop("nbest.jsonl"), "lists no hash for nbest.jsonl"),
    (lambda m: m["inputs"].update(dataset_release="other"), "inputs.dataset_release is not"),
    (lambda m: m["inputs"].update(split_manifest_sha256="c" * 64), "inputs.split_manifest_sha256 is not"),
    (lambda m: m["inputs"].update(split="validation"), "inputs.split is not"),
])
def test_refuses_bad_manifest(mod, tmp_path, capsys, change, needle):
    case = build(tmp_path)
    edit_manifest(case, "B2", change)
    err = refused(mod, case.argv(), capsys, case, needle)
    assert "B2" in err


def test_refuses_missing_manifest(mod, tmp_path, capsys):
    case = build(tmp_path)
    (case.runs["B2"] / "manifest.json").unlink()
    err = refused(mod, case.argv(), capsys, case, "B2")
    assert "file not found" in err


def test_refuses_existing_output(mod, tmp_path, capsys):
    case = build(tmp_path)
    (case.runs["B2"] / "manifest.json").unlink()  # scoring would fail on this first
    case.output.mkdir()
    code, _, err = run(mod, case.argv(), capsys)
    assert code == 1 and "already exists" in err and err.count("\n") == 1
    assert list(case.output.iterdir()) == []


def test_refuses_missing_output_parent(mod, tmp_path, capsys):
    case = build(tmp_path)
    argv = case.argv()
    argv[argv.index("--output") + 1] = str(tmp_path / "missing" / "out")
    code, _, err = run(mod, argv, capsys)
    assert code == 1 and "parent" in err and err.count("\n") == 1
    assert not (tmp_path / "missing").exists()


def test_refuses_pair_unknown_system(mod, tmp_path, capsys):
    case = build(tmp_path)
    (case.runs["B2"] / "manifest.json").unlink()  # scoring would fail on this first
    refused(mod, case.argv("--pair", "B2=B9"), capsys, case, "not given with --run: B9")


@pytest.mark.parametrize("second", ["B2", "b2"])
def test_refuses_duplicate_run(mod, tmp_path, capsys, second):
    case = build(tmp_path)
    argv = case.argv() + ["--run", f"{second}={case.runs['B2']}"]
    refused(mod, argv, capsys, case, "given twice")


@pytest.mark.parametrize("value", ["B2", "=somewhere", "../evil=x", "summary=x", "SUMMARY=x", "B2="])
def test_refuses_bad_run_argument(mod, tmp_path, capsys, value):
    case = build(tmp_path)
    refused(mod, case.argv(runs=[]) + ["--run", value], capsys, case, "--run")


def test_refuses_pair_without_groups(mod, tmp_path, capsys):
    refs = ref_rows()
    del refs[3]["leakage_group_id"]
    case = build(tmp_path, refs=refs, runs={"B2": ONE_SUB, "B3": PERFECT})
    (case.runs["B2"] / "manifest.json").unlink()  # scoring would fail on this first
    refused(mod, case.argv("--pair", "B2=B3"), capsys, case,
            "pairs need leakage_group_id on every reference")


def test_refuses_usage_error(mod, tmp_path, capsys):
    case = build(tmp_path)
    argv = case.argv()
    argv[argv.index("--split") + 1] = "train"
    refused(mod, argv, capsys, case, "usage")


# --- operating-system errors ---------------------------------------------------

def test_refuses_missing_references_file(mod, tmp_path, capsys):
    case = build(tmp_path)
    digest = sha(case.refs)
    case.refs.unlink()
    refused(mod, case.argv(refs_sha=digest), capsys, case, "file not found")


def test_refuses_references_directory(mod, tmp_path, capsys):
    case = build(tmp_path)
    digest = sha(case.refs)
    case.refs.unlink()
    case.refs.mkdir()
    refused(mod, case.argv(refs_sha=digest), capsys, case, "is a directory")


def test_refuses_unwritable_output_parent(mod, tmp_path, capsys, monkeypatch):
    case = build(tmp_path)

    def denied(self, *args, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied", str(self))

    monkeypatch.setattr(pathlib.Path, "mkdir", denied)
    refused(mod, case.argv(), capsys, case, "permission denied")


def test_write_failure_leaves_no_output(mod, tmp_path, capsys, monkeypatch):
    case = build(tmp_path)
    real = mod.write_json

    def fail_on_summary(path, value):
        if Path(path).name == "summary.json":
            raise OSError(errno.ENOSPC, "No space left on device", str(path))
        real(path, value)

    monkeypatch.setattr(mod, "write_json", fail_on_summary)
    refused(mod, case.argv(), capsys, case, "No space left on device")


def test_cli_exit_code_and_single_line(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    missing = tmp_path / "no-protocol.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--protocol", str(missing), "--protocol-sha256", "0" * 64,
         "--split", "test", "--references", str(missing), "--references-sha256", "0" * 64,
         "--run", f"B2={tmp_path}", "--output", str(tmp_path / "out")],
        capture_output=True, text=True, env=env, timeout=120, check=False,
    )
    assert proc.returncode == 1
    assert proc.stderr.count("\n") == 1 and "file not found" in proc.stderr
    assert "Traceback" not in proc.stderr and proc.stdout == ""


def test_refuses_protocol_without_required_key(mod, tmp_path, capsys):
    case = build(tmp_path)
    protocol = json.loads(case.protocol.read_text(encoding="utf-8"))
    del protocol["decoder"]["requested_n"]
    case.protocol.write_text(json.dumps(protocol), encoding="utf-8")
    refused(mod, case.argv(), capsys, case, "protocol decoder.requested_n is missing")


def test_refuses_unreadable_manifest(mod, tmp_path, capsys):
    case = build(tmp_path)
    (case.runs["B2"] / "manifest.json").write_bytes(b"{not json")
    err = refused(mod, case.argv(), capsys, case, "cannot read")
    assert "B2" in err


def test_refuses_pair_without_separator(mod, tmp_path, capsys):
    case = build(tmp_path)
    refused(mod, case.argv("--pair", "B2"), capsys, case, "must be SYSTEM=CONTROL")
