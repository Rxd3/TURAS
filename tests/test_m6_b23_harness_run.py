"""Tests for scripts/m6_b23_harness_run.py with fake model loaders (no weights, no GPU)."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from turkish_asr_rdl.baselines.qwen import QwenBaselineAdapter
from turkish_asr_rdl.baselines.whisper import WhisperBaselineAdapter

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_batch = importlib.import_module("tests.test_m2_batch_runner")  # its fake Whisper/Qwen loaders
DummyModel, DummyProcessor = _batch.DummyModel, _batch.DummyProcessor

SPLIT = "a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8"
CONFIGS = ["whisper_large_v3.json", "qwen3_asr_1_7b.json"]
ERROR_TEXT = "inference error rows; evaluation gate remains closed"
WARMUP_BATCHES = 3  # the helper takes the first 3 warm-up rows at batch size 1
TEST_ROWS, WARMUP_ROWS, TIMING_ROWS = 6, 4, 2


def _load_tool():
    spec = importlib.util.spec_from_file_location("m6_b23_harness_run", REPO / "scripts/m6_b23_harness_run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(root, prefix, n):
    out = []
    for i in range(n):
        uid = f"{prefix}-{i:03d}"
        audio = root / "audio" / f"{uid}.flac"
        audio.parent.mkdir(exist_ok=True)
        sf.write(audio, np.full(16000, 0.01 * (i + 1), dtype=np.float32), 16000, subtype="PCM_16")
        out.append({"schema_version": "0.1", "dataset_release": "turas-tr-econ-legis-v0.1",
                    "split_manifest_sha256": SPLIT, "utterance_id": uid, "audio_uri": f"audio/{uid}.flac",
                    "audio_sha256": sha(audio), "duration_s": 1.0, "sample_rate_hz": 16000,
                    "channels": 1, "language": "tr"})
    return out


def write_jsonl(path, items):
    path.write_text("".join(json.dumps(r) + "\n" for r in items), encoding="utf-8")


def fake_nbest(*args, **kwargs):
    cand = {"candidate_id": "beam-0001", "rank": 1, "text": " merhaba", "token_ids": [1, 2],
            "acoustic_log_score": -1.0, "acoustic_score_kind": "seq2seq_token_log_prob_sum",
            "decoder_sequence_score": -0.5, "score_details": {}}
    second = {**cand, "candidate_id": "beam-0002", "rank": 2, "text": " merhaba dunya", "token_ids": [1, 3]}
    return [cand, second], {}


def fake_factory(config, checkpoint, nbest_fn=fake_nbest):
    # Production make_adapter (family dispatch + progress wrapper); only the model loaders are fakes.
    adapter = tool.make_adapter(config, checkpoint, device="cpu", model_loader=DummyModel,
                                processor_loader=DummyProcessor, nbest_fn=nbest_fn)
    family = WhisperBaselineAdapter if config["model"]["id"].startswith("openai") else QwenBaselineAdapter
    assert isinstance(adapter, family) and type(adapter).__name__ == "Progress"
    return adapter


def failing_factory(fail_call):
    def factory(config, checkpoint):
        calls = 0

        def nbest(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == fail_call:
                raise RuntimeError("CUDA error: an illegal memory access was encountered")
            return fake_nbest(*args, **kwargs)
        return fake_factory(config, checkpoint, nbest)
    return factory


def run_tool(argv, factory):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            result = tool.main(argv, factory)
        except ValueError as exc:
            result = exc
    status = None
    for line in buf.getvalue().splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and "status" in obj:
            status = obj
    return result, status


@pytest.fixture
def frozen(tmp_path):
    """A copy of the real protocol whose inputs point at tiny synthetic clips."""
    root = tmp_path
    test_rows, warm_rows = rows(root, "tst", TEST_ROWS), rows(root, "trn", WARMUP_ROWS)
    write_jsonl(root / "test.jsonl", test_rows)
    write_jsonl(root / "train.jsonl", warm_rows)
    (root / "test.txt").write_text("".join(r["utterance_id"] + "\n" for r in test_rows), encoding="utf-8")
    timing_ids = "".join(r["utterance_id"] + "\n" for r in test_rows[:TIMING_ROWS])
    (root / "timing.txt").write_text(timing_ids, encoding="utf-8")
    protocol = copy.deepcopy(json.loads(
        (REPO / "configs/evaluation/m45_final_protocol.json").read_text(encoding="utf-8")))
    protocol["inputs"] = {
        "audio_root": str(root), "test_inference": str(root / "test.jsonl"),
        "test_inference_sha256": sha(root / "test.jsonl"), "test_ids": str(root / "test.txt"),
        "test_ids_sha256": sha(root / "test.txt"), "warmup_inference": str(root / "train.jsonl"),
        "warmup_inference_sha256": sha(root / "train.jsonl"), "timing_ids": str(root / "timing.txt"),
        "timing_ids_sha256": sha(root / "timing.txt"),
    }
    path = root / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return root, path, sha(path), protocol


def argv(frozen, cfg_name, mode, out):
    root, path, digest, _ = frozen
    return ["--protocol", str(path), "--protocol-sha256", digest,
            "--config", str(REPO / "configs/baselines" / cfg_name),
            "--checkpoint", str(root), "--mode", mode, "--output", str(out)]


@pytest.mark.parametrize("cfg_name", CONFIGS)
@pytest.mark.parametrize("mode, where", [("accuracy", "warmup"), ("timing", "warmup"), ("timing", "repeat")])
def test_error_rows_fail_the_gate(frozen, cfg_name, mode, where):
    root, _, _, protocol = frozen
    repeats = protocol["timing"]["repeats"] if mode == "timing" else 1
    fail_call = 1 if where == "warmup" else WARMUP_BATCHES + TIMING_ROWS * (repeats - 1) + 1
    out = root / "out"
    result, status = run_tool(argv(frozen, cfg_name, mode, out), failing_factory(fail_call))

    assert isinstance(result, ValueError) and str(result) == ERROR_TEXT
    assert status is not None and status["status"] == "failed_error_rows"
    done = json.loads((out / "completed.json").read_text(encoding="utf-8"))
    split = {"warmup": 1 if where == "warmup" else 0,
             "per_repeat": [1 if where == "repeat" and r == repeats else 0 for r in range(1, repeats + 1)],
             "predictions": 0}
    assert (done["status"], done["error_rows"], done["error_rows_split"]) == ("failed_error_rows", 1, split)
    bad = [r for r in tool.read_jsonl(out / "timing.jsonl")
           if r.get("record_type") == "batch" and r.get("error_rows", 0) > 0]
    assert len(bad) == 1
    if where == "warmup":
        assert bad[0]["warmup"] is True
    else:
        assert bad[0]["warmup"] is False and bad[0]["repeat"] == repeats
    assert not [r for r in tool.read_jsonl(out / "predictions.jsonl") if r["status"] == "error"]


@pytest.mark.parametrize("cfg_name", CONFIGS)
@pytest.mark.parametrize("mode, n, reps", [("accuracy", TEST_ROWS, 1), ("timing", TIMING_ROWS, 3)])
def test_clean_run_completes_and_refuses_reuse(frozen, cfg_name, mode, n, reps):
    out = frozen[0] / "out"
    code, status = run_tool(argv(frozen, cfg_name, mode, out), fake_factory)

    done = json.loads((out / "completed.json").read_text(encoding="utf-8"))
    timing = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["evaluation_timing"]
    assert code == 0 and done["rows"] == n and done["repeats"] == reps and done["error_rows"] == 0
    assert status is not None and status["status"] == "completed" and done["status"] == "completed"
    assert done["error_rows_split"] == {"warmup": 0, "per_repeat": [0] * reps, "predictions": 0}
    assert timing["repeats"] == reps and timing["warmup_batches"] == 3 and timing["batch_size"] == 1
    assert sorted(done["files"]) == ["manifest.json", "nbest.jsonl", "predictions.jsonl", "timing.jsonl"]
    with pytest.raises(SystemExit):
        tool.main(argv(frozen, cfg_name, mode, out), fake_factory)


def test_wrong_protocol_hash_refused(frozen):
    args = argv(frozen, CONFIGS[0], "timing", frozen[0] / "x")
    args[args.index("--protocol-sha256") + 1] = "0" * 64
    with pytest.raises(ValueError):
        tool.main(args, fake_factory)
    assert not (frozen[0] / "x").exists()


@pytest.mark.parametrize("cfg_name", CONFIGS)
def test_repeat_one_error_is_counted_once(frozen, cfg_name):
    """predictions.jsonl repeats the repeat-1 rows, so they must not be added twice."""
    root, _, _, protocol = frozen
    out = root / "out-r1"
    result, _ = run_tool(argv(frozen, cfg_name, "timing", out), failing_factory(WARMUP_BATCHES + 1))
    assert isinstance(result, ValueError) and str(result) == ERROR_TEXT
    done = json.loads((out / "completed.json").read_text(encoding="utf-8"))
    assert done["error_rows"] == 1
    assert done["error_rows_split"]["per_repeat"][0] == 1
