"""Tests for scripts/m6_number_invariant_wer.py. All text here is synthetic fixtures."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from turkish_asr_rdl.decoding.export import RunContext, prediction_row
from turkish_asr_rdl.decoding.interface import Candidate, Hypothesis, InferenceInput
from turkish_asr_rdl.evaluation.normalize import TR_NORMALIZED_V1, normalize

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "m6_number_invariant_wer.py"
RELEASE = "rel_synthetic"
SPLIT_SHA = "a" * 64
AUDIO_SHA = "b" * 64
NBEST_SHA = "c" * 64
SYS = "SYS"
V1, INV = "tr-normalized-v1", "tr-number-invariant-v0"
WORDS_REFS = [("utt_01", "zeytinyağı bütçe"), ("utt_02", "çekirge pırasa kuşkonmaz")]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("m6_number_invariant_wer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))


def hypothesis(uid: str, text: str | None) -> Hypothesis:
    if text is None:
        return Hypothesis(uid, "error", None, error_code="adapter_exception",
                          error_message="SYNTHETIC failure")
    kind = "seq2seq_teacher_forced_log_prob"
    return Hypothesis(uid, "ok", text, candidates=(Candidate("c1", 1, text, (1, 2), -1.0, kind, -0.5),))


def write_run(directory: Path, pairs: list[tuple[str, str | None]], split: str = "test") -> Path:
    directory.mkdir(parents=True)
    ctx = RunContext(run_id="run_synthetic", system_id=SYS, dataset_release=RELEASE, split=split,
                     split_manifest_sha256=SPLIT_SHA, model_artifact_sha256=AUDIO_SHA,
                     decoder_config_sha256=SPLIT_SHA)
    rows = [
        prediction_row(hypothesis(uid, text), InferenceInput(
            schema_version="0.1", dataset_release=RELEASE, split_manifest_sha256=SPLIT_SHA,
            utterance_id=uid, audio_uri=f"audio/{uid}.flac", audio_sha256=AUDIO_SHA,
            duration_s=1.0, sample_rate_hz=16000, channels=1, language="tr"), ctx, f"t_{uid}", NBEST_SHA)
        for uid, text in pairs
    ]
    write_jsonl(directory / "predictions.jsonl", rows)
    return directory


def write_refs(root: Path, refs: list[tuple[str, str]]) -> Path:
    path = root / "references.jsonl"
    write_jsonl(path, [
        {"utterance_id": uid, "audio_sha256": AUDIO_SHA, "reference_text": text,
         "leakage_group_id": f"g{i}", "split": "test", "dataset_release": RELEASE}
        for i, (uid, text) in enumerate(refs)])
    return path


def score(mod, root: Path, refs, runs: dict, capsys, refs_sha: str | None = None):
    refs_path = write_refs(root, refs)
    argv = ["--references", str(refs_path), "--references-sha256", refs_sha or sha(refs_path)]
    for name, pairs in runs.items():
        argv += ["--run", f"{name}={write_run(root / 'runs' / name, pairs)}"]
    out = root / "out.json"
    code = mod.main(argv + ["--output", str(out)])
    captured = capsys.readouterr()
    return code, captured.err, out, refs_path


def scored(mod, tmp_path, refs, pairs, capsys) -> dict:
    code, err, out, _ = score(mod, tmp_path, refs, {SYS: pairs}, capsys)
    assert code == 0, err
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("text, expected", [
    ("15", "on beş"), ("105", "yüz beş"), ("100", "yüz"), ("101", "yüz bir"),
    ("1.250.000", "bir milyon iki yüz elli bin"), ("1000", "bin"), ("1.001", "bin bir"),
    ("21.000", "yirmi bir bin"), ("1.001.000", "bir milyon bin"), ("1000000", "bir milyon"),
    ("3,5", "üç virgül beş"), ("2026", "iki bin yirmi altı"), ("0", "sıfır"),
])
def test_numbers_become_turkish_words(mod, text, expected):
    assert mod.number_invariant(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("%5", "yüzde beş"), ("5%", "yüzde beş"), ("%3,5", "yüzde üç virgül beş"),
    ("5 %80", "beş yüzde seksen"), ("5%80", "beş yüzde seksen"),
])
def test_percent_becomes_yuzde(mod, text, expected):
    assert mod.number_invariant(text) == expected


@pytest.mark.parametrize("text", ["onbeş", "on beş"])
def test_glued_and_spaced_number_words_agree(mod, text):
    assert mod.number_invariant(text) == "on beş"


def test_glued_long_number_words_are_split(mod):
    assert mod.number_invariant("ikibinyirmialtı") == "iki bin yirmi altı"


def test_words_that_only_start_with_number_words_stay(mod):
    assert mod.number_invariant("onlar binlerce yüzde") == "onlar binlerce yüzde"


def test_turkish_casing_survives(mod):
    assert mod.number_invariant("İstanbul'da ıslak 15 IŞIK") == "istanbulda ıslak on beş ışık"


@pytest.mark.parametrize("text", [
    "İstanbul'da ILIK ışık, ıslak Işık!", "Çocuklar okula gitti.", "ISPARTA'dan İZMİR'e; yol uzun?",
])
def test_text_without_numbers_equals_normalized_v1(mod, text):
    assert mod.number_invariant(text) == normalize(text, TR_NORMALIZED_V1)


@pytest.mark.parametrize("text", ["15 elma", "1.250.000 lira", "%3,5 faiz", "2026'da 5% İşçi"])
def test_output_is_already_normalizer_form(mod, text):
    out = mod.number_invariant(text)
    assert normalize(out, TR_NORMALIZED_V1) == out


@pytest.mark.parametrize("text, expected", [
    ("15'te", "on beş te"), ("1'inci", "bir inci"), ("1.ci", "bir ci"),
])
def test_suffix_stays_its_own_token(mod, text, expected):
    assert mod.number_invariant(text) == expected


def test_exact_edit_counts_in_both_views(mod, tmp_path, capsys):
    out = scored(mod, tmp_path, [("utt_01", "15 elma")], [("utt_01", "on beş elma")], capsys)
    metrics = out["systems"][SYS]["metrics"]
    assert out["status"] == "m6_number_invariant_scoring_complete"
    assert out["views"] == [V1, INV]
    assert metrics[V1]["words"] == {"substitutions": 1, "deletions": 0, "insertions": 1, "hits": 1, "n": 2}
    assert metrics[INV]["words"] == {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 3, "n": 3}
    assert set(metrics[INV]["characters"]) == {"substitutions", "deletions", "insertions", "hits", "n"}
    assert metrics[INV]["cer"] == 0.0
    assert metrics[V1]["wer"] > 0


def test_error_row_is_counted_and_scores_as_deletions(mod, tmp_path, capsys):
    pairs = [("utt_01", "zeytinyağı bütçe"), ("utt_02", None)]
    out = scored(mod, tmp_path, WORDS_REFS, pairs, capsys)
    system = out["systems"][SYS]
    assert system["error_rows"] == 1
    assert system["utterances"] == 2
    for view in (V1, INV):
        assert system["metrics"][view]["words"]["deletions"] == 3


def test_output_has_aggregates_only(mod, tmp_path, capsys):
    pairs = [("utt_01", "zeytinyağı bütçe"), ("utt_02", "kırmızı balık yüzdü")]
    code, err, out, _ = score(mod, tmp_path, WORDS_REFS, {SYS: pairs}, capsys)
    assert code == 0, err
    text = out.read_text(encoding="utf-8")
    for word in ("zeytinyağı", "bütçe", "çekirge", "pırasa", "kuşkonmaz", "kırmızı", "balık", "yüzdü"):
        assert word not in text


GOOD = [("utt_01", "zeytinyağı bütçe"), ("utt_02", "çekirge pırasa kuşkonmaz")]


def test_hashes_of_inputs_and_script(mod, tmp_path, capsys):
    code, err, out, refs_path = score(mod, tmp_path, WORDS_REFS, {SYS: GOOD}, capsys)
    assert code == 0, err
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["references_sha256"] == sha(refs_path)
    assert data["systems"][SYS]["predictions_sha256"] == sha(tmp_path / "runs" / SYS / "predictions.jsonl")
    assert data["script_sha256"] == sha(SCRIPT)


def test_two_runs_are_both_scored(mod, tmp_path, capsys):
    code, err, out, _ = score(mod, tmp_path, WORDS_REFS, {"A": GOOD, "B": GOOD}, capsys)
    assert code == 0, err
    assert set(json.loads(out.read_text(encoding="utf-8"))["systems"]) == {"A", "B"}


@pytest.mark.parametrize("case, refs, pairs, refs_sha", [
    ("wrong references hash", WORDS_REFS, GOOD, "0" * 64),
    ("duplicate prediction id", WORDS_REFS, GOOD + [("utt_01", "bütçe")], None),
    ("missing prediction", WORDS_REFS, GOOD[:1], None),
    ("extra prediction", WORDS_REFS, GOOD + [("utt_99", "bütçe")], None),
    ("duplicate reference id", WORDS_REFS + [("utt_01", "bütçe")], GOOD, None),
])
def test_refusal_is_one_line_and_writes_nothing(mod, tmp_path, capsys, case, refs, pairs, refs_sha):
    code, err, out, _ = score(mod, tmp_path, refs, {SYS: pairs}, capsys, refs_sha)
    assert code == 1, case
    assert len(err.strip().splitlines()) == 1, err
    assert "Traceback" not in err
    assert not out.exists()
    if case != "wrong references hash":
        assert f"--run {SYS}:" in err, err


def test_deeply_nested_predictions_are_refused_in_one_line(mod, tmp_path, capsys):
    refs_path = write_refs(tmp_path, WORDS_REFS)
    run_dir = tmp_path / "runs" / SYS
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text("[" * 100_000 + "\n", encoding="utf-8")
    out = tmp_path / "out.json"
    code = mod.main(["--references", str(refs_path), "--references-sha256", sha(refs_path),
                     "--run", f"{SYS}={run_dir}", "--output", str(out)])
    err = capsys.readouterr().err
    assert code == 1
    assert len(err.strip().splitlines()) == 1, err
    assert f"--run {SYS}:" in err
    assert not out.exists()
