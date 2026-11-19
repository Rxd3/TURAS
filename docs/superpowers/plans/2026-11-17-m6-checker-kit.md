# M6 Checker Kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Member 6's contract-only checker kit (environment, shared audio loader, adapter interface, timing harness, validators, Turkish evaluator, leakage audit, bootstrap, report, `turas-check` CLI) plus a 5070 Ti smoke-test script, in time for the Monday 2026-11-23 deadline.

**Architecture:** Pure-Python library code in `src/turkish_asr_rdl/` (schemas, decoding shared parts, evaluation, utils), with thin `scripts/` wrappers. Everything in Phase 0 runs on CPU without torch; torch is an optional import used only to synchronize the GPU in the timing harness. The work lands as five focused PRs from `feat/m6-*` branches.

**Tech Stack:** Python 3.11+ (target 3.13), numpy 2.5.5, soundfile 0.14.0, soxr 1.1.0, jiwer 4.0.0 (tests only), pytest, pytest-cov; torch 2.14.1+cu130 and transformers 5.23.0 only on the 5070 Ti.

**Spec:** `docs/superpowers/specs/2026-11-17-m6-checker-kit-design.md` (approved 2026-11-17). Contracts: `docs/contracts/*.md`, `docs/evaluation.md`.

## Global Constraints

- Python syntax must run on 3.11 and 3.13 (the CI matrix). No `type` statements, no PEP 695 generics.
- No new runtime dependency beyond the pinned numpy, soundfile and soxr. Test-only: pytest, pytest-cov, jiwer==4.0.0. torch is never imported at module top level.
- Unit tests are offline: no model/data download, no GPU, no network. Test strings are short, start with or sit in a test labeled `SYNTHETIC`, and never look like dataset transcripts.
- Files at most 400 lines; functions under 50 lines; nesting depth at most 4; frozen dataclasses or new objects, never in-place mutation of inputs.
- Library code never prints; only `cli.py` writes to stdout/stderr. User-facing errors are one friendly line, never a traceback; exit code 1 on any failure.
- Every `docs/contracts` rule cited by a task is the stricter reading; a value the contract names is used exactly (`schema_version` `"0.1"`, seed `2026`, `2000` replicates, warm-up at least `3` batches, `16000` Hz).
- Coverage of new code at least 80% (`pytest --cov=turkish_asr_rdl --cov-report=term-missing`).
- Conventional commits (`feat|fix|test|docs|chore|ci: ...`). Push only `feat/m6-*` branches; ask Member 6 before every PR into `main`. Never force-push.
- Test references are read only by `turas-check evaluate`; no other module opens a reference file.

## Review Focus

1. A JSONL file saved by a Windows tool with a UTF-8 BOM: `read_jsonl` must fail with one line telling the user to save as UTF-8 without BOM (test in Task 2).
2. A prediction row with `status: "ok"` and `hypothesis_text: ""`: this must be a violation that says to use `status: "empty"` (test in Task 3).
3. An `audio_uri` that is absolute, has a drive letter, uses backslashes or escapes the root with `..`: `resolve_audio_path` must refuse it (test in Task 5).
4. Audio that is not 16 kHz mono (for example 24 kHz stereo): the loader must return float32 mono at 16 kHz with the right length (test in Task 5).
5. A machine with no `nvidia-smi` (CI, this laptop): `probe_gpu` returns `{"ok": False, "error": ...}` and `run_timed` still finishes (test in Task 7).

---

## PR-A: environment (branch `feat/m6-env`, section devops)

### Task 1: Pinned environment, env check and test CI in the repo

**Files:**
- Create: `requirements/requirements.in` (copy of `C:\Users\Asus\Downloads\nlp\project3-asr\requirements.in`)
- Create: `requirements/windows-cu130.txt` (copy of `...\project3-asr\requirements.txt`, byte-identical)
- Create: `requirements/dev.txt` (pinned `pytest`, `pytest-cov`, plus `numpy==2.5.5`, `soundfile==0.14.0`, `soxr==1.1.0`, `jiwer==4.0.0`, with the current pytest/pytest-cov versions looked up on PyPI and pinned exactly)
- Create: `src/turkish_asr_rdl/utils/env_check.py` (logic of `project3-asr/check_env.py`; `INSTALL_HINT = "pip install -r requirements/windows-cu130.txt"`; references to `SETUP.md` become `docs/reproducibility.md`)
- Create: `scripts/check_env.py` (thin wrapper: `from turkish_asr_rdl.utils.env_check import main; raise SystemExit(main())`)
- Create: `tests/test_env_check.py` (the 43 tests of `project3-asr/test_check_env.py`, imports switched to `turkish_asr_rdl.utils.env_check`)
- Modify: `docs/reproducibility.md` (append section "Environment setup (Windows)": driver at least 580, Python 3.13, FFmpeg *shared* build on PATH, `python -m venv .venv`, `pip install -r requirements/windows-cu130.txt`, `pip install --no-deps -e .`, `python scripts/check_env.py`; gotcha: inside the Claude desktop app set `UV_PYTHON_INSTALL_DIR` outside AppData when using uv)
- Modify: `.github/workflows/packaging.yml` (add job `tests`: ubuntu-latest, Python 3.11 and 3.13, `pip install -r requirements/dev.txt`, `pip install --no-deps -e .`, `pytest --cov=turkish_asr_rdl --cov-report=term-missing`)
- Modify: `.gitignore` only if `requirements/` paths are ignored (they are not today; verify)

**Interfaces:**
- Produces: `TURAS/.venv` (local, git-ignored) with `requirements/dev.txt` installed on this laptop, which every later task uses to run tests; `turkish_asr_rdl.utils.env_check.main() -> int`.

- [ ] **Step 1:** Create the dev venv on this laptop: `python -m venv .venv`, `.venv\Scripts\python -m pip install -r requirements/dev.txt`, `.venv\Scripts\python -m pip install --no-deps -e .`. Expected: `python -c "import turkish_asr_rdl, numpy, soundfile, soxr, jiwer"` exits 0.
- [ ] **Step 2:** Move the tests first: copy the 43 tests into `tests/test_env_check.py` with the new import path. Run `.venv\Scripts\python -m pytest tests/test_env_check.py -q`. Expected: FAIL with `ModuleNotFoundError: turkish_asr_rdl.utils.env_check`.
- [ ] **Step 3:** Create `env_check.py` and the `scripts/check_env.py` wrapper as listed.
- [ ] **Step 4:** Run `.venv\Scripts\python -m pytest tests/test_env_check.py -q --cov=turkish_asr_rdl.utils.env_check`. Expected: 43 passed, coverage at least 95%.
- [ ] **Step 5:** Run `.venv\Scripts\python scripts/check_env.py`. Expected on this laptop: `NOT READY`, exit 1, no traceback.
- [ ] **Step 6:** Commit `chore(env): add pinned Windows cu130 environment, env check and test CI`; push `feat/m6-env`.

---

## PR-B: harness and validators (branch `feat/m6-harness`, section backend #1)

### Task 2: Validation report and strict JSONL I/O

**Files:**
- Create: `src/turkish_asr_rdl/schemas/report.py`, `src/turkish_asr_rdl/schemas/jsonl.py`
- Test: `tests/schemas/test_jsonl.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class ValidationReport: errors: tuple[str, ...] = (); warnings: tuple[str, ...] = ()`, with property `ok -> bool` (no errors)
  - `combine(*reports: ValidationReport) -> ValidationReport`
  - `class JsonlError(ValueError)` (message names the path and the 1-based line)
  - `read_jsonl(path: Path) -> list[dict]`
  - `write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str` returns the SHA-256 hex of the written bytes; writes UTF-8 without BOM, LF, `ensure_ascii=False`, `allow_nan=False`, atomically (temp file in the same folder, then `os.replace`)
  - `sha256_file(path: Path) -> str`

- [ ] **Step 1: Write the failing tests**

```python
def test_roundtrip_returns_rows_and_hash(tmp_path): ...  # write_jsonl then read_jsonl gives equal rows; hash == sha256_file(path)
def test_bom_is_rejected_with_fix_hint(tmp_path):
    p = tmp_path / "x.jsonl"; p.write_bytes(b'\xef\xbb\xbf{"a": 1}\n')
    with pytest.raises(JsonlError, match="BOM"): read_jsonl(p)
def test_nan_and_infinity_rejected(tmp_path): ...  # line '{"a": NaN}' -> JsonlError mentioning line 1
def test_duplicate_keys_rejected(tmp_path): ...  # '{"a": 1, "a": 2}' -> JsonlError
def test_non_object_line_rejected(tmp_path): ...  # '[1, 2]' -> JsonlError
def test_blank_line_inside_file_rejected(tmp_path): ...  # '{"a":1}\n\n{"a":2}\n' -> JsonlError line 2
def test_crlf_lines_accepted(tmp_path): ...  # b'{"a": 1}\r\n' -> [{"a": 1}]
def test_write_is_atomic_on_error(tmp_path): ...  # rows containing float('nan') -> ValueError, no file left at path
def test_combine_merges_in_order(): ...  # combine(R(errors=("a",)), R(warnings=("w",))) == R(("a",), ("w",))
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/schemas/test_jsonl.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement. `json.loads` with `parse_constant` that raises and an `object_pairs_hook` that rejects duplicate keys; decode bytes with `utf-8` (strict) after checking for the BOM prefix.
- [ ] **Step 4:** Run the same command. Expected: all pass.
- [ ] **Step 5:** Commit `feat(schemas): add strict JSONL I/O and validation report`.

### Task 3: Prediction and inference-input validators

**Files:**
- Create: `src/turkish_asr_rdl/schemas/predictions.py`
- Test: `tests/schemas/test_predictions.py`

**Interfaces:**
- Consumes: `ValidationReport` (Task 2).
- Produces:
  - `SCHEMA_VERSION = "0.1"`
  - `PREDICTION_FIELDS: tuple[str, ...]`, exactly the predictions.md field list: `schema_version, run_id, system_id, dataset_release, split, split_manifest_sha256, utterance_id, audio_sha256, status, hypothesis_text, selected_candidate_id, nbest_artifact_sha256, model_artifact_sha256, decoder_config_sha256, normalization_version, timing_record_id, error`
  - `INFERENCE_INPUT_FIELDS = ("schema_version", "dataset_release", "split_manifest_sha256", "utterance_id", "audio_uri", "audio_sha256", "duration_s", "sample_rate_hz", "channels", "language")`
  - `FORBIDDEN_KEYS = frozenset({"reference_text", "reference", "references", "gold", "gold_text", "target", "targets", "label", "labels", "transcript_correction", "reference_token_ids"})`, checked recursively in nested objects
  - `validate_predictions(rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str] | None = None) -> ValidationReport`
  - `validate_inference_inputs(rows: Sequence[Mapping[str, Any]]) -> ValidationReport`

Rules (each a separate error string naming the row's `utterance_id` or index): missing field; unknown extra field; forbidden key anywhere; `schema_version != "0.1"`; `status` not in `ok/empty/error`; `ok` needs non-empty `hypothesis_text`; `empty` needs `""`; `error` needs `hypothesis_text is None` and `error` as an object with string `code` and `message`; non-error rows need `error is None`; SHA-256 fields are 64 lowercase hex characters (nullable ones may be null); non-finite numbers; duplicate `utterance_id`; more than one distinct value of `run_id`, `system_id`, `dataset_release`, `split` or `split_manifest_sha256` across rows; with `expected_ids`: missing IDs, extra IDs, and order differing from `expected_ids` (report the first mismatch position). Inference inputs: exactly the allowlist (no extra keys), positive `duration_s`, `sample_rate_hz` and `channels`, and `audio_uri` a non-empty string.

- [ ] **Step 1: Write the failing tests** (one per rule, built from a helper `ok_row(uid)` that returns a valid row):

```python
def test_valid_rows_pass(): assert validate_predictions([ok_row("utt_a"), ok_row("utt_b")]).ok
def test_ok_with_empty_text_says_use_empty():
    r = validate_predictions([{**ok_row("utt_a"), "hypothesis_text": ""}])
    assert any("status \"empty\"" in e for e in r.errors)
def test_reference_text_nested_is_forbidden():
    r = validate_predictions([{**err_row("utt_a"), "error": {"code": "x", "message": "y", "gold": "z"}}])
    assert any("gold" in e for e in r.errors)
def test_order_must_match_expected_ids():
    r = validate_predictions([ok_row("utt_b"), ok_row("utt_a")], expected_ids=["utt_a", "utt_b"])
    assert any("order" in e for e in r.errors)
def test_missing_and_extra_ids_listed(): ...
def test_mixed_run_ids_rejected(): ...
def test_bad_sha256_rejected(): ...      # uppercase hex and 63 chars both fail
def test_inference_input_extra_key_rejected(): ...  # {"...allowlist...", "reference_text": "x"} fails
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/schemas/test_predictions.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement as specified; one small checker function per rule group, composed with `combine`.
- [ ] **Step 4:** Re-run. Expected: all pass.
- [ ] **Step 5:** Commit `feat(schemas): validate prediction and inference-input JSONL`.

### Task 4: N-best validator

**Files:**
- Create: `src/turkish_asr_rdl/schemas/nbest.py`
- Test: `tests/schemas/test_nbest.py`

**Interfaces:**
- Consumes: `ValidationReport` (Task 2).
- Produces:
  - `NBEST_ROW_FIELDS = ("schema_version", "run_id", "system_id", "dataset_release", "split", "split_manifest_sha256", "utterance_id", "audio_sha256", "model_artifact_sha256", "decoder_config_sha256", "requested_n", "actual_n", "score_convention_id", "candidates")`; optional row key `dedup_map` (object: removed raw candidate ID mapped to the kept candidate ID)
  - `CANDIDATE_FIELDS = ("candidate_id", "rank", "text", "token_ids", "token_count", "acoustic_log_score", "acoustic_score_kind", "decoder_sequence_score", "lm_log_score", "rdl_log_score", "combined_score", "score_details")`
  - `NULLABLE_SCORES = ("decoder_sequence_score", "lm_log_score", "rdl_log_score", "combined_score")`
  - `validate_nbest(rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str] | None = None, normalize: Callable[[str], str] | None = None) -> ValidationReport`

Rules: row fields present; `requested_n` an int of at least 1; `actual_n == len(candidates)` and `actual_n <= requested_n`; candidate IDs unique within the row; `rank` an int, strictly increasing, from 1 to `requested_n` (gaps allowed after dedup); `token_count == len(token_ids)`; `acoustic_log_score` finite (mandatory); each null score in `NULLABLE_SCORES` has a non-empty string in `score_details["null_reasons"][field]`; **candidate `text` values pairwise distinct** (error message: "duplicate candidate text; deduplicate before export and record it in dedup_map"); with `normalize`, candidates equal after normalization give a **warning**; with `expected_ids`, the same ID checks as Task 3.

- [ ] **Step 1: Write the failing tests**, including:

```python
def test_whisper_duplicate_trap_fails():
    row = nbest_row(texts=["merhaba dünya"] * 3)  # SYNTHETIC
    assert any("duplicate candidate text" in e for e in validate_nbest([row]).errors)
def test_rank_gaps_after_dedup_allowed(): assert validate_nbest([nbest_row(ranks=[1, 3, 4])]).ok
def test_null_score_without_reason_fails(): ...
def test_null_score_with_reason_passes(): ...
def test_normalized_duplicates_are_warnings_only():
    r = validate_nbest([nbest_row(texts=["Ankara", "ankara"])], normalize=str.lower)
    assert r.ok and r.warnings
def test_actual_n_mismatch_fails(): ...
def test_token_count_mismatch_fails(): ...
def test_missing_acoustic_score_fails(): ...
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/schemas/test_nbest.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Re-run. Expected: all pass.
- [ ] **Step 5:** Commit `feat(schemas): validate N-best artifacts and catch duplicate candidates`.

### Task 5: Shared audio loader

**Files:**
- Create: `src/turkish_asr_rdl/decoding/audio.py`
- Test: `tests/decoding/test_audio.py`

**Interfaces:**
- Produces:
  - `TARGET_SAMPLE_RATE_HZ = 16_000`
  - `class AudioLoadError(ValueError)`
  - `resolve_audio_path(audio_root: Path, audio_uri: str) -> Path`: refuses absolute paths, drive letters, backslashes, empty strings and any path whose resolved location is outside `audio_root`
  - `to_mono_16k(samples: np.ndarray, sample_rate_hz: int) -> np.ndarray`: float32, mean downmix over channels, `soxr.resample(..., quality="HQ")` when the rate differs
  - `load_audio(path: Path) -> np.ndarray`: `soundfile.read(path, dtype="float32", always_2d=True)` then `to_mono_16k`; a missing file, a directory or an unreadable format raise `AudioLoadError` with one friendly line (map `FileNotFoundError`, `IsADirectoryError`/`PermissionError` by type, and `soundfile.LibsndfileError`)

- [ ] **Step 1: Write the failing tests** (synthetic sine waves written with `soundfile.write` into `tmp_path`):

```python
def test_24k_stereo_becomes_16k_mono(tmp_path):
    x = np.stack([sine(24000, 1.0), sine(24000, 1.0)], axis=1)
    sf.write(tmp_path / "a.wav", x, 24000)
    y = load_audio(tmp_path / "a.wav")
    assert y.dtype == np.float32 and y.ndim == 1 and abs(len(y) - 16000) <= 2
def test_16k_mono_flac_unchanged_length(tmp_path): ...
@pytest.mark.parametrize("uri", ["/abs/a.flac", "C:/a.flac", "a\\b.flac", "../a.flac", "x/../../a.flac", ""])
def test_unsafe_uris_refused(tmp_path, uri):
    with pytest.raises(AudioLoadError): resolve_audio_path(tmp_path, uri)
def test_missing_file_is_friendly(tmp_path): ...  # AudioLoadError, message contains the file name, no traceback text
def test_directory_is_friendly(tmp_path): ...
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/decoding/test_audio.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Re-run. Expected: all pass.
- [ ] **Step 5:** Commit `feat(decoding): add shared 16 kHz mono audio loader`.

### Task 6: Adapter interface, export rows and run manifest

**Files:**
- Create: `src/turkish_asr_rdl/decoding/interface.py`, `src/turkish_asr_rdl/decoding/export.py`
- Test: `tests/decoding/test_export.py`

**Interfaces:**
- Consumes: `validate_inference_inputs`, `validate_predictions`, `validate_nbest` (Tasks 3–4).
- Produces (`interface.py`):
  - `@dataclass(frozen=True) class InferenceInput` with the ten `INFERENCE_INPUT_FIELDS`, plus `@classmethod from_row(cls, row: Mapping[str, Any]) -> InferenceInput` that raises `ValueError` listing `validate_inference_inputs` errors
  - `@dataclass(frozen=True) class AudioItem: utterance_id: str; samples: np.ndarray`
  - `@dataclass(frozen=True) class Candidate: candidate_id: str; rank: int; text: str; token_ids: tuple[int, ...]; acoustic_log_score: float; acoustic_score_kind: str; decoder_sequence_score: float | None = None; score_details: Mapping[str, Any] = field(default_factory=dict); lm_log_score: float | None = None; rdl_log_score: float | None = None; combined_score: float | None = None`
  - `@dataclass(frozen=True) class Hypothesis: utterance_id: str; status: str; text: str | None; candidates: tuple[Candidate, ...] = (); error_code: str | None = None; error_message: str | None = None; selected_candidate_id: str | None = None`
  - `class AsrAdapter(Protocol)`: attributes `system_id: str`, `model_artifact_sha256: str`, `decoder_config_sha256: str`, `requested_n: int` (0 means no N-best), `score_convention_id: str | None`, `config: Mapping[str, Any]`; methods `load(self) -> None`, `transcribe(self, batch: Sequence[AudioItem]) -> list[Hypothesis]`
- Produces (`export.py`):
  - `@dataclass(frozen=True) class RunContext: run_id: str; system_id: str; dataset_release: str; split: str; split_manifest_sha256: str; model_artifact_sha256: str; decoder_config_sha256: str; normalization_version: str = "tr-normalized-v1"`
  - `prediction_row(hyp: Hypothesis, inp: InferenceInput, ctx: RunContext, timing_record_id: str, nbest_artifact_sha256: str | None) -> dict`
  - `nbest_row(hyp: Hypothesis, inp: InferenceInput, ctx: RunContext, requested_n: int, score_convention_id: str) -> dict` (fills nullable scores with `None` plus `null_reasons` "not produced by this system" when absent)
  - `git_state(repo_dir: Path) -> dict` returns `{"commit": str | None, "dirty": bool | None}` via `git rev-parse HEAD` / `git status --porcelain`; `None` values when git is missing
  - `run_manifest(ctx: RunContext, *, command: str, environment: Mapping[str, Any], configuration: Mapping[str, Any], outputs: Mapping[str, str], timing: Mapping[str, Any], limitations: Sequence[str]) -> dict` with the artifacts.md groups as top-level keys: `identity`, `code`, `environment`, `inputs`, `configuration`, `randomness`, `model_knowledge`, `selection`, `outputs`, `evaluation_timing`, `limitations`

- [ ] **Step 1: Write the failing tests:**

```python
def test_prediction_row_passes_validator(): assert validate_predictions([prediction_row(ok_hyp, inp, ctx, "t1", None)]).ok
def test_error_hypothesis_row_passes_validator(): ...  # status error -> hypothesis_text None, error {"code","message"}
def test_nbest_row_passes_validator_and_fills_null_reasons(): ...
def test_from_row_rejects_reference_text(): ...
def test_manifest_has_all_groups(): assert set(run_manifest(ctx, ...)) == {...11 groups...}
def test_git_state_outside_repo_is_none(tmp_path): assert git_state(tmp_path) == {"commit": None, "dirty": None}
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/decoding/test_export.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Re-run. Expected: all pass.
- [ ] **Step 5:** Commit `feat(decoding): add adapter interface, export rows and run manifest`.

### Task 7: Hardware probe and `run_timed`

**Files:**
- Create: `src/turkish_asr_rdl/decoding/hardware.py`, `src/turkish_asr_rdl/decoding/timing.py`
- Test: `tests/decoding/test_timing.py`

**Interfaces:**
- Consumes: `load_audio`, `resolve_audio_path` (Task 5); `InferenceInput`, `AudioItem`, `Hypothesis`, `AsrAdapter`, `prediction_row` (Task 6).
- Produces (`hardware.py`):
  - `NVIDIA_SMI_FIELDS = ("name", "driver_version", "clocks.sm", "clocks.max.sm", "power.draw", "temperature.gpu", "clocks_event_reasons.active", "clocks_event_reasons.hw_slowdown", "clocks_event_reasons.hw_thermal_slowdown", "clocks_event_reasons.sw_thermal_slowdown", "clocks_event_reasons.sw_power_cap")`
  - `probe_gpu(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict`: `{"ok": True, <field>: <str>, ...}` or `{"ok": False, "error": <one line>}` (missing binary, timeout of 10 s, non-zero exit)
  - `thermal_flag(sample: Mapping[str, Any]) -> bool`: True when any of the hw_slowdown, hw_thermal_slowdown or sw_thermal_slowdown fields equals `"Active"`; `sw_power_cap` never flags
- Produces (`timing.py`):
  - `MIN_WARMUP_BATCHES = 3`
  - `@dataclass(frozen=True) class TimingRecord: timing_record_id: str; run_id: str; repeat: int; utterance_ids: tuple[str, ...]; audio_duration_s: float; batch_size: int; warmup: bool; end_to_end_s: float; core_model_s: float; sync_method: str`
  - `@dataclass(frozen=True) class TimedRun: hypotheses: tuple[Hypothesis, ...]; records: tuple[TimingRecord, ...]; cold_load_s: float; end_to_end_rtf: float; core_model_rtf: float; per_repeat_end_to_end_rtf: tuple[float, ...]; hardware: tuple[Mapping[str, Any], ...]; thermal_flags: int`
  - `default_sync() -> tuple[Callable[[], None], str]`: `torch.cuda.synchronize` with name `"torch.cuda.synchronize"` when torch imports and CUDA is available, else a no-op named `"none"`
  - `run_timed(adapter: AsrAdapter, timed_inputs: Sequence[InferenceInput], warmup_inputs: Sequence[InferenceInput], *, audio_root: Path, run_id: str, repeats: int = 3, batch_size: int = 1, sync: Callable[[], None] | None = None, clock: Callable[[], float] = time.perf_counter, probe: Callable[[], dict] = probe_gpu) -> TimedRun`
  - `write_run_outputs(out_dir: Path, run: TimedRun, inputs: Sequence[InferenceInput], ctx: RunContext, adapter: AsrAdapter, *, command: str) -> Mapping[str, str]`: writes `predictions.jsonl` (each row's `timing_record_id` is the repeat-1 record containing that utterance), `nbest.jsonl` (only when `adapter.requested_n > 0`, using `adapter.score_convention_id`), `timing.jsonl` (one `TimingRecord` per line, plus `hardware` samples) and `manifest.json` (`run_manifest`, with `outputs` holding each file's SHA-256 and `evaluation_timing` holding the three RTF values, `cold_load_s`, warm-up count, repeats, batch size and `thermal_flags`). Returns `{file name: sha256}`. All writes go through `write_jsonl` or an atomic JSON write.

Behaviour: refuse (ValueError, friendly) when warm-up batches number fewer than 3, `repeats < 1`, `batch_size < 1`, or warm-up and timed IDs overlap. `adapter.load()` is timed once as `cold_load_s`. Per batch: `sync(); t0 = clock()`; resolve and load audio; `sync(); c0 = clock()`; `adapter.transcribe(batch)`; `sync(); c1 = clock()`; serialize the prediction rows with `json.dumps`; `sync(); t1 = clock()`. `end_to_end_s = t1 - t0` and `core_model_s = c1 - c0`. Audio duration is measured as `len(samples) / 16000`. Warm-up batches are run first and excluded from every RTF. Each RTF is sum of seconds over sum of audio seconds across all non-warm-up records. Hypotheses come from repeat 1, in input order. `probe()` runs before and after each repeat, outside the timed region. An exception from `transcribe` becomes `Hypothesis(status="error", error_code="adapter_exception", error_message=<one line>)` for each utterance in the batch, and the run continues.

- [ ] **Step 1: Write the failing tests** with a fake adapter, a fake clock that advances 0.5 s per call, and synthetic 1-second FLAC files:

```python
def test_rtf_is_sum_over_sum_and_excludes_warmup(...):
    run = run_timed(fake, timed, warm, audio_root=tmp_path, run_id="r", repeats=2, clock=fake_clock, probe=lambda: {"ok": False, "error": "no gpu"})
    assert run.end_to_end_rtf == pytest.approx(expected) and all(not r.warmup for r in run.records if r.repeat > 0)
def test_fewer_than_three_warmup_batches_refused(...): ...
def test_overlapping_warmup_and_timed_ids_refused(...): ...
def test_adapter_exception_becomes_error_rows(...): ...  # status "error", error_code "adapter_exception", run finishes
def test_hypotheses_from_first_repeat_in_input_order(...): ...
def test_probe_without_nvidia_smi(monkeypatch):
    r = probe_gpu(run=raise_file_not_found)
    assert r["ok"] is False and "nvidia-smi" in r["error"]
def test_sw_power_cap_does_not_flag(): assert not thermal_flag({"clocks_event_reasons.sw_power_cap": "Active"})
def test_hw_thermal_flags(): assert thermal_flag({"clocks_event_reasons.hw_thermal_slowdown": "Active"})
def test_written_outputs_pass_validators(tmp_path):  # predictions.jsonl passes validate_predictions(expected_ids=sorted ids); nbest.jsonl passes validate_nbest; returned hashes equal sha256_file
    ...
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/decoding/test_timing.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the whole suite with coverage: `.venv\Scripts\python -m pytest -q --cov=turkish_asr_rdl --cov-report=term-missing`. Expected: all pass, at least 80% on new modules.
- [ ] **Step 5:** Commit `feat(decoding): add timing harness with end-to-end and core-model RTF`; push `feat/m6-harness`.

---

## PR-C: evaluator core (branch `feat/m6-evaluator`, section backend #2, parallel to PR-B)

### Task 8: Turkish normalizer (`tr-normalized-v1`, contract 0.2 rules)

**Files:**
- Create: `src/turkish_asr_rdl/evaluation/normalize.py`
- Test: `tests/evaluation/test_normalize.py`

**Interfaces:**
- Produces: `RAW_V1 = "raw-v1"`, `TR_NORMALIZED_V1 = "tr-normalized-v1"`, `normalize(text: str, version: str) -> str` (raises `ValueError` for an unknown version), `word_tokens(text: str) -> list[str]` (`text.split()`).

Algorithm for `tr-normalized-v1`, in this order:
1. `unicodedata.normalize("NFC", text)`
2. replace `"I"` with `"ı"` and `"İ"` with `"i"`, then `.lower()`
3. delete every character whose category is `Cf`
4. for each of U+0027, U+2019 and U+02BC: delete it when both neighbours are letters (category starting with `L`), otherwise replace it with a space
5. every other character whose category starts with `P` becomes a space, except `%`, which is kept
6. split on `str.isspace()` runs, then join with one ASCII space (strip both ends)

`raw-v1` returns the text unchanged.

- [ ] **Step 1: Write the failing tests** (contract examples are rules, not transcripts):

```python
@pytest.mark.parametrize("src,out", [
    ("IŞIK", "ışık"), ("İZMİR", "izmir"), ("I\u0307", "i"), ("i\u0307", "i\u0307"),
    ("Ankara'da", "ankarada"), ("Ankara\u2019da", "ankarada"), ("Ankara\u02bcda", "ankarada"),
    ("2026'da", "2026 da"), ("1.250,50", "1 250 50"), ("%5", "%5"), ("₺100", "₺100"), ("a+b-c/d", "a+b c d"),
    ("a\u200bb", "ab"), ("a\u00a0 \u2028b", "a b"), ("", ""), ("   ", ""),
])
def test_tr_normalized_v1(src, out): assert normalize(src, TR_NORMALIZED_V1) == out
def test_idempotent(): for s in CASES: assert normalize(normalize(s, V), V) == normalize(s, V)
def test_raw_is_identity(): assert normalize(" Aİ, b ", RAW_V1) == " Aİ, b "
def test_unknown_version_raises(): ...
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/evaluation/test_normalize.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Re-run. Expected: all pass.
- [ ] **Step 5:** Commit `feat(evaluation): add raw-v1 and tr-normalized-v1 normalizers`.

### Task 9: Deterministic edit-count aligner

**Files:**
- Create: `src/turkish_asr_rdl/evaluation/align.py`
- Test: `tests/evaluation/test_align.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class EditCounts: substitutions: int = 0; deletions: int = 0; insertions: int = 0; hits: int = 0`, with properties `ref_len` (hits + S + D) and `errors` (S + D + I), and `__add__`; `align_counts(ref: Sequence[str], hyp: Sequence[str]) -> EditCounts`.

Algorithm: full Levenshtein DP (unit costs). Backtrace from `(len(ref), len(hyp))` to `(0, 0)`. At each cell, take the first of these moves that reaches the cell's optimal cost: the diagonal (a hit when the elements are equal, otherwise a substitution), then deletion (up), then insertion (left). Add `# ponytail: O(n*m) pure Python; vectorize if evaluation of a full test set exceeds a few minutes`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_golden_swap_is_two_substitutions(): assert align_counts(["a", "b"], ["b", "a"]) == EditCounts(2, 0, 0, 0)
def test_empty_hypothesis_is_all_deletions(): assert align_counts(["a", "b"], []) == EditCounts(0, 2, 0, 0)
def test_empty_reference_is_all_insertions(): assert align_counts([], ["a"]) == EditCounts(0, 0, 1, 0)
def test_hand_checked(): assert align_counts(list("kitten"), list("sitting")) == EditCounts(2, 0, 1, 4)
def test_totals_match_jiwer_on_random_pairs():  # seeded random token lists; S+D+I equals jiwer.process_words(...) totals
    ...
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/evaluation/test_align.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Re-run. Expected: all pass.
- [ ] **Step 5:** Commit `feat(evaluation): add deterministic S/D/I aligner`.

### Task 10: Metrics and strict join

**Files:**
- Create: `src/turkish_asr_rdl/evaluation/metrics.py`, `src/turkish_asr_rdl/evaluation/join.py`
- Test: `tests/evaluation/test_metrics.py`, `tests/evaluation/test_join.py`

**Interfaces:**
- Consumes: `normalize`, `word_tokens` (Task 8); `EditCounts`, `align_counts` (Task 9).
- Produces (`metrics.py`):
  - `@dataclass(frozen=True) class UtteranceScore: utterance_id: str; view: str; status: str; words: EditCounts; chars: EditCounts`
  - `score_pair(utterance_id: str, reference: str, hypothesis: str, view: str, status: str = "ok") -> UtteranceScore`: words come from `word_tokens(normalize(x, view))`; chars are the code points of `normalize(x, view)`, so raw-v1 CER counts spaces exactly as emitted
  - `@dataclass(frozen=True) class CorpusRates: view: str; utterances: int; zero_reference_utterances: int; error_rows: int; words: EditCounts; chars: EditCounts; wer: float | None; cer: float | None; wacc: float | None`. A rate is `None` when its denominator is 0. `wacc = 1 - wer`.
  - `corpus_rates(scores: Sequence[UtteranceScore]) -> CorpusRates` (sum over sum, never a mean of percentages)
  - `rtf(processing_s: float, audio_s: float) -> float` and `rtfx(processing_s: float, audio_s: float) -> float`; `ValueError` when `audio_s <= 0`
- Produces (`join.py`):
  - `REFERENCE_FIELDS = ("utterance_id", "audio_sha256", "reference_text", "leakage_group_id", "split", "dataset_release")`
  - `class JoinError(ValueError)` with attribute `problems: tuple[str, ...]`
  - `@dataclass(frozen=True) class JoinedRow: utterance_id: str; leakage_group_id: str; reference: str; hypothesis: str; status: str`
  - `join(predictions: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]], *, split: str, split_manifest_sha256: str | None = None) -> tuple[JoinedRow, ...]`

Join behaviour: keep only references whose `split` matches. Raise `JoinError` (listing up to 10 IDs per problem plus a count) for any of these: duplicate IDs on either side; missing or extra IDs; an `audio_sha256` mismatch; a `dataset_release` mismatch; a predictions `split_manifest_sha256` different from the argument. All checks run before any metric. `error` rows join with hypothesis `""` and status `"error"`. The output is ordered by utterance ID.

- [ ] **Step 1: Write the failing tests**, including:

```python
def test_corpus_wer_is_sum_over_sum():  # two utts: 1/1 error and 0/9 -> WER 0.1, not 0.5
def test_zero_reference_counted_and_rate_none_when_total_zero(): ...
def test_raw_cer_counts_trailing_space(): assert score_pair("u", "ab", "ab ", RAW_V1).chars.insertions == 1
def test_normalized_view_ignores_case_and_punct(): assert score_pair("u", "İZMİR.", "izmir", TR_NORMALIZED_V1).words.errors == 0
def test_wacc_is_one_minus_wer(): ...
def test_join_reports_missing_extra_and_hash_mismatch():
    with pytest.raises(JoinError) as e: join(preds, refs, split="test")
    assert any("missing" in p for p in e.value.problems) and any("audio_sha256" in p for p in e.value.problems)
def test_join_error_rows_become_empty_hypothesis(): ...
def test_join_filters_to_requested_split(): ...
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/evaluation -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `.venv\Scripts\python -m pytest -q --cov=turkish_asr_rdl --cov-report=term-missing`. Expected: all pass, at least 80% on new modules.
- [ ] **Step 5:** Commit `feat(evaluation): add corpus metrics and strict reference join`; push `feat/m6-evaluator`.

---

## PR-D: docs and contract amendments (branch `feat/m6-docs`, section docs, parallel)

### Task 11: AGENTS.md, contract 0.2 amendments, decisions, repo-name fix

**Files:**
- Create: `AGENTS.md` (one screen; the six rules in spec P0.1, each linking its contract. Every tool it names carries "(on `main` after PR-<X>)" until that PR merges; the docs section removes the marker for merged PRs before opening PR-D)
- Modify: `docs/contracts/normalization.md` (version 0.2: the P0.3 rules: U+02BC treated like the other apostrophes, `%` kept, Cf deleted, whitespace = `str.isspace()`, notes on digit apostrophes and number grouping, the alignment tie-break with backtrace direction, the label `tr-normalized-v1` kept until first freeze)
- Modify: `docs/contracts/nbest.md` (0.2: `score_details.null_reasons`, row-level `dedup_map`, candidate texts must be distinct after dedup, ranks keep original values)
- Modify: `docs/contracts/predictions.md` (0.2: stable order = sorted split-file order; `split_manifest_sha256` = SHA-256 of the split manifest file bytes)
- Modify: `docs/contracts/data.md` (0.2, Member 1 review: release audio stored as 16 kHz mono FLAC under an artifact root, `audio_uri` relative with `/` separators)
- Modify: `docs/evaluation.md` (timing: primary end-to-end, batch 1, 3 repeats on a frozen 30-minute timing subset, labeled secondary core-model view, `clocks_event_reasons` recording; HF-course metric mapping; the six-day cuts from spec section 5 listed as declared limitations)
- Modify: `docs/decisions.md` (append dated rows D17–D21 for the amendments above, each with rationale, status "proposed by M6", reviewers and affected contract versions; keep history)
- Modify: `README.md`, `docs/decisions.md` D01 and `docs/repository_setup.md` (the repository is `Rxd3/TURAS`, public, renamed from `project3-turkish-asr-rdl`; `main` is protected)

- [ ] **Step 1:** Write the edits.
- [ ] **Step 2:** Check: every relative link resolves (script or manual check), `git diff --check` is clean, and every version bump appears in `docs/contracts/README.md`.
- [ ] **Step 3:** Commit `docs(contracts): propose 0.2 amendments, add AGENTS.md and fix repository name`; push `feat/m6-docs`.

---

## PR-E: CLI, splits, leakage, bootstrap and report (branch `feat/m6-cli`, from `main` after PR-B and PR-C merge; section backend)

### Task 12: `turas-check` CLI (predictions, nbest, inputs, evaluate)

**Files:**
- Create: `src/turkish_asr_rdl/evaluation/cli.py`, `scripts/turas_check.py` (thin wrapper), `src/turkish_asr_rdl/schemas/splits.py` (only `read_split_file` in this task)
- Modify: `pyproject.toml` (`[project.scripts] turas-check = "turkish_asr_rdl.evaluation.cli:main"`)
- Test: `tests/evaluation/test_cli.py`

**Interfaces:**
- Consumes: `read_jsonl`, `sha256_file` (Task 2); `validate_predictions`, `validate_inference_inputs` (Task 3); `validate_nbest` (Task 4); `normalize` (Task 8); `score_pair`, `corpus_rates`, `join` (Task 10).
- Produces:
  - `read_split_file(path: Path) -> list[str]`: UTF-8, LF only, no BOM, no blank lines; raises `SplitFileError(ValueError)`
  - `main(argv: Sequence[str] | None = None) -> int` with these subcommands:
    - `predictions FILE [--ids SPLIT_FILE]`
    - `nbest FILE [--ids SPLIT_FILE]` (passes `normalize(·, "tr-normalized-v1")` for warnings)
    - `inputs FILE`
    - `manifest FILE`: checks that the 11 top-level groups from `run_manifest` are present and that every `outputs` hash is 64 lowercase hex characters. No separate timing-sidecar validator is needed, because M6's `run_timed` writes every official timing file (spec decision 8).
    - `evaluate --predictions FILE --references FILE --split NAME --split-manifest FILE [--run-manifest FILE] --out DIR`, which writes `metrics.json` (both views: `CorpusRates` as dicts, plus `system_id`, the RTF values copied from the run manifest's `evaluation_timing` when given, and the evaluator version) and `per_utterance.csv`

  Output is one line per violation, then `PASS` or `FAIL (<n> problems)`. Warnings print as `warning: ...` and do not fail. Exit 0 on pass, 1 on any failure. Each OS error maps to one friendly line, never a traceback: a missing file, a path that is a directory, permission denied, or an output folder that can't be created.

- [ ] **Step 1: Write the failing tests** (call `main([...])` and capture stdout):

```python
def test_predictions_pass_exit_zero(tmp_path, capsys): ...
def test_predictions_violation_exit_one_and_lists_problem(tmp_path, capsys): ...
def test_missing_file_is_one_friendly_line(tmp_path, capsys):
    assert main(["predictions", str(tmp_path / "nope.jsonl")]) == 1
    assert "Traceback" not in capsys.readouterr().err
def test_directory_instead_of_file_is_friendly(tmp_path, capsys): ...
def test_evaluate_writes_metrics_with_both_views(tmp_path): ...  # metrics.json has "raw-v1" and "tr-normalized-v1"
def test_evaluate_join_failure_exit_one_without_metrics(tmp_path): ...  # no metrics.json written
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/evaluation/test_cli.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement with `argparse`; keep each subcommand handler under 50 lines.
- [ ] **Step 4:** Re-run, then `.venv\Scripts\turas-check --help`. Expected: tests pass, and the help lists five subcommands.
- [ ] **Step 5:** Commit `feat(cli): add turas-check predictions, nbest, inputs, manifest and evaluate`.

### Task 13: Split validation and leakage audit

**Files:**
- Modify: `src/turkish_asr_rdl/schemas/splits.py` (add `validate_splits`)
- Create: `src/turkish_asr_rdl/evaluation/leakage.py`
- Modify: `src/turkish_asr_rdl/evaluation/cli.py` (subcommands `splits` and `leakage`)
- Test: `tests/schemas/test_splits.py`, `tests/evaluation/test_leakage.py`

**Interfaces:**
- Consumes: `read_split_file` (Task 12); `normalize`, `word_tokens` (Task 8).
- Produces:
  - `validate_splits(splits: Mapping[str, Sequence[str]], release_ids: Sequence[str]) -> ValidationReport`: requires the keys `train`, `validation` and `test`; each list sorted and unique; the lists pairwise disjoint; their union equal to `release_ids`
  - `REVIEW_MIN_TOKENS = 10`
  - `@dataclass(frozen=True) class LeakageReport: blocking: tuple[str, ...]; review: tuple[str, ...]; info: Mapping[str, Any]`, with property `ok` (no blocking findings)
  - `audit_leakage(manifest_rows: Sequence[Mapping[str, Any]], splits: Mapping[str, Sequence[str]]) -> LeakageReport`
  - CLI: `splits --release-manifest FILE --splits-dir DIR` and `leakage --release-manifest FILE --splits-dir DIR --out FILE` (writes the report as JSON; exit 1 when there are blocking findings)

Blocking findings: any `leakage_group_id`, `duplicate_cluster_id` or `audio_sha256` present in two splits; within one `source_recording_key`, two rows from different splits whose `[start_ms, end_ms)` intervals overlap (touching intervals are not overlap). Review findings: identical `tr-normalized-v1` reference text across splits with at least 10 tokens. Info: counts of shorter identical texts, and the share of rows whose `speaker_ids` is null.

- [ ] **Step 1: Write the failing tests:**

```python
def test_touching_intervals_are_not_overlap(): ...   # [0,1000) and [1000,2000) in different splits -> ok
def test_overlapping_intervals_block(): ...          # [0,1500) and [1000,2000)
def test_shared_leakage_group_blocks(): ...
def test_shared_audio_hash_blocks(): ...
def test_long_duplicate_text_is_review_not_blocking(): ...  # 10 SYNTHETIC tokens
def test_short_duplicate_text_is_info_only(): ...
def test_splits_union_must_equal_release(): ...
def test_unsorted_split_fails(): ...
def test_cli_leakage_exit_one_on_blocking(tmp_path): ...
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/schemas/test_splits.py tests/evaluation/test_leakage.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. For the interval check, group rows by `source_recording_key` and sort by `start_ms`; adjacent rows from different splits whose intervals overlap are blocking.
- [ ] **Step 4:** Re-run. Expected: all pass.
- [ ] **Step 5:** Commit `feat(evaluation): add split validation and leakage audit`.

### Task 14: Paired group bootstrap and report tables

**Files:**
- Create: `src/turkish_asr_rdl/evaluation/bootstrap.py`, `src/turkish_asr_rdl/evaluation/report.py`
- Modify: `src/turkish_asr_rdl/evaluation/cli.py` (subcommand `report`)
- Test: `tests/evaluation/test_bootstrap.py`, `tests/evaluation/test_report.py`

**Interfaces:**
- Consumes: `EditCounts` (Task 9); `CorpusRates`, `rtfx` (Task 10); the `metrics.json` written by `evaluate` (Task 12).
- Produces:
  - `@dataclass(frozen=True) class BootstrapResult: replicates: int; seed: int; groups: int; numpy_version: str; index_sha256: str; undefined_replicates: int; intervals: Mapping[str, tuple[float, float]]; paired_differences: Mapping[str, tuple[float, float]]` (key `"A - B"`)
  - `paired_group_bootstrap(per_system: Mapping[str, Mapping[str, EditCounts]], *, replicates: int = 2000, seed: int = 2026) -> BootstrapResult`: one index matrix `np.random.default_rng(seed).integers(0, G, size=(replicates, G))` shared by all systems; replicate WER = sum of errors / sum of ref_len over the drawn groups; replicates with ref_len 0 are counted as undefined and skipped; 2.5 and 97.5 percentiles; `index_sha256` is the SHA-256 of the matrix's `astype("<i8").tobytes()`
  - `@dataclass(frozen=True) class SystemResult: system_id: str; role: str; raw: CorpusRates; normalized: CorpusRates; rtf: float | None`, with `role` either `"baseline"` or `"contribution"`
  - `render_markdown(results: Sequence[SystemResult]) -> str`: two tables titled "Baselines (pretrained, no training by our group)" and "Our contributions". Columns: System, Orthographic WER (raw-v1), Normalised WER (tr-normalized-v1), CER raw, CER normalized, WAcc, RTF, RTFx, Error rows, Reference words. Missing values print `n/a`, never 0. A footnote maps the column names to the HF audio-course terms.
  - CLI: `report --baseline FILE ... --contribution FILE ... --out FILE.md`

- [ ] **Step 1: Write the failing tests:**

```python
def test_same_seed_same_index_hash(): ...
def test_pairing_shared_across_systems(): ...      # identical systems -> paired difference interval == (0.0, 0.0)
def test_zero_reference_replicates_counted_undefined(): ...
def test_interval_contains_point_estimate_on_simple_case(): ...
def test_report_separates_baselines_and_contributions(): ...
def test_missing_rtf_prints_na_not_zero(): ...
```

- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/evaluation/test_bootstrap.py tests/evaluation/test_report.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the full suite with coverage. Expected: all pass, at least 80% on new modules.
- [ ] **Step 5:** Commit `feat(evaluation): add paired group bootstrap and report tables`; push `feat/m6-cli`.

---

## Smoke branch (branch `feat/m6-smoke-tests` from `feat/m6-harness`; never merged; section devops)

### Task 15: 5070 Ti smoke script and runbook

**Files:**
- Create: `experiments/m6-smoke/smoke.py`, `experiments/m6-smoke/adapters.py`, `experiments/m6-smoke/RUNBOOK.md`
- Modify: `src/turkish_asr_rdl/decoding/audio.py` only if `to_mono_16k` needs a bytes entry point (prefer reading via `soundfile.read(io.BytesIO(...))` in the smoke script)

**Interfaces:**
- Consumes: `run_timed`, `TimedRun` (Task 7); `AsrAdapter`, `InferenceInput`, `Hypothesis`, `Candidate` (Task 6); `to_mono_16k`, `TARGET_SAMPLE_RATE_HZ` (Task 5); `validate_nbest` (Task 4).
- Produces: `python experiments/m6-smoke/smoke.py --model NAME --artifacts DIR --clips 20` writes `DIR/smoke/<NAME>.json`. `NAME` is one of `whisper-small`, `whisper-large-v3`, `qwen3-asr-1.7b`, `qwen3-asr-0.6b`, `xlsr-tr-cv7`, `xlsr-tr-cv8`, `train-whisper-small-full`, `train-whisper-large-v3-lora`, `train-whisper-large-v3-full`, `pyg-hetero`. Then `python experiments/m6-smoke/smoke.py --summarize DIR` writes `DIR/smoke/REPORT.md`.

Data: `huggingface_hub.hf_hub_download("disco-eth/WorldSpeech", "data/tr_tr/train-00000.parquet", repo_type="dataset", revision="b76b85395039f50332fbc421f7c098ee2f2653d7")`. Read the columns `segment_id` and `audio` with pyarrow. Decode the Opus bytes with `soundfile`, convert with `to_mono_16k`, and write 23 clips as 16 kHz FLAC under `DIR/smoke/audio/`: 3 for warm-up and 20 timed. Accuracy is never computed or printed (train-split audio, feasibility only).

Per inference model, record: load seconds, peak VRAM (`torch.cuda.max_memory_allocated` after `reset_peak_memory_stats`), end-to-end and core RTF from `run_timed` (repeats 1, batch 1), precision, and N-best facts.
- **Whisper:** an N-best of 10 through core `GenerationMixin.generate(model, input_features=..., decoder_input_ids=<SOT, <|tr|>, transcribe, notimestamps>, num_beams=10, num_return_sequences=10, output_scores=True, return_dict_in_generate=True, max_new_tokens=440)`. Record whether the 10 texts are distinct and whether `validate_nbest` passes. Also record the duplicate count from `WhisperForConditionalGeneration.generate` with the same settings, which proves the trap.
- **Qwen3-ASR:** load the `-hf` checkpoints with transformers' native classes per the model card. Try beam 5 with scores and record whether distinct scored candidates come back. Never install the `qwen-asr` package.
- **XLS-R:** build `Wav2Vec2CTCTokenizer` and `Wav2Vec2FeatureExtractor` explicitly (`AutoProcessor` fails). Decode greedy. Record the exact sequence log-probability via `-torch.nn.functional.ctc_loss(..., reduction="none")` and confirm it is finite.

Training probes: 3 optimizer steps on bf16 AdamW with gradient checkpointing, labels from the shard's `human_transcript` (train split). whisper-small full fine-tune at batch 8; whisper-large-v3 LoRA (r=32 on `q_proj` and `v_proj`) at batch 2; whisper-large-v3 full fine-tune at batch 1, expected to run out of memory. Record peak VRAM, or the OOM message, as evidence.

PyG: `pip install torch_geometric==2.8.1` (it adds `aiohttp` and `pyparsing`; revert with `pip uninstall` if the check fails). Build a toy `HeteroData` with 2 node types and 2 edge types plus their reverses, run 2 `HeteroConv` layers of `SAGEConv` forward and backward on CUDA in fp32 and bf16, and record success or the error.

`RUNBOOK.md` gives the exact commands for the 5070 Ti session, in order:
1. clone `https://github.com/Rxd3/TURAS` to `C:\Users\benda\Downloads\nlp\TURAS` and check out `feat/m6-smoke-tests`;
2. `C:\Users\benda\Downloads\nlp\project3-asr\.venv\Scripts\python -m pip install --no-deps -e .`;
3. one `smoke.py` command per model, sequential, with `--artifacts G:\turas-artifacts`;
4. `--summarize`;
5. send back `REPORT.md` and the JSON files by message.

- [ ] **Step 1:** Write `adapters.py` (one small `AsrAdapter` class per model family) and `smoke.py`. No unit tests (throwaway, never merged), but `python experiments/m6-smoke/smoke.py --help` must run on CPU without torch installed.
- [ ] **Step 2:** Lint check: `.venv\Scripts\python -m py_compile experiments/m6-smoke/*.py`. Expected: no output.
- [ ] **Step 3:** Commit `chore(smoke): add 5070 Ti model smoke tests and runbook`; push `feat/m6-smoke-tests`.
- [ ] **Step 4 (supervisor):** Send RUNBOOK.md's commands to the 5070 Ti session `NLP project`; run overnight; review `REPORT.md` when it returns.

---

## Execution order and parallelism

1. Task 1 first. It creates the dev venv every other task uses.
2. Then in parallel, each in its own worktree:
   - backend #1: Tasks 2–7;
   - backend #2: Tasks 8–10;
   - docs: Task 11.
3. Task 15 starts as soon as `feat/m6-harness` is pushed.
4. Tasks 12–14 start on Wednesday, from `main` once PR-B and PR-C have merged.
5. Testing section: final review of each PR before the supervisor asks Member 6 to open it.
6. Terminology P/R/F1 (spec P0.4) is added as Task 16 only if Member 1 freezes a lexicon by Thursday 2026-11-19; otherwise the final report lists it as not done (spec section 5 cuts).
