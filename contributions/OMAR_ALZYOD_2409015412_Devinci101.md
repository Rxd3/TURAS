# Member 2 — Seq2Seq and Speech-LLM Baselines

Name: Omar Alzyod. Student ID: `2409015412`. GitHub username: [`Devinci101`](https://github.com/Devinci101).
Role: Member 2 — Seq2Seq and Speech-LLM Baselines.
Branch convention: `feat/m2-seq2seq-speech-llm-baselines` (focused topic suffixes allowed).
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).
Backward-compatible record: [member_2.md](member_2.md).

**Status: PR #6 merged (`cbdc07c`), PR #20 merged (`1dade5b`); Baseline B2 (Whisper-large-v3) validation (438/438 OK) and D43 working-test (636/636 OK) reference-free batch inference complete, verified, packaged (`turas-b2-whisper-large-v3-m2-handoff.zip`), and handed off to Member 6; smoke CLI keyword argument mismatch fixed with regression test (76 tests passing). B1 baseline integrated via shared M4/M6 benchmark; B3 Qwen3-ASR candidates remain unselected pending Decision D05 resolution.**

## Actual contributions

### 2026-11-22: Baseline B2 (Whisper-large-v3) validation and D43 working-test batch execution, portable handoff package creation, and smoke CLI bug fix

- **Scope:** Execution and verification of Baseline B2 (`openai/whisper-large-v3`) reference-free batch inference on validation (438 clips) and D43-authorized working-test (636 clips) sets, generation and cryptographic verification of portable handoff bundle `turas-b2-whisper-large-v3-m2-handoff.zip`, fix of keyword argument mismatch in `scripts/smoke_check_whisper_rank1.py` / `src/turkish_asr_rdl/baselines/whisper_rank1.py`, and regression test additions (76 tests passing).
- **Owned paths:** `src/turkish_asr_rdl/baselines/whisper_rank1.py`, `scripts/smoke_check_whisper_rank1.py`, `tests/test_m2_whisper_rank1.py`, `tests/test_m2_batch_runner.py`, `experiments/m2-baseline-notes.md`, `contributions/member_2.md`, `contributions/OMAR_ALZYOD_2409015412_Devinci101.md`.
- **Completed B2 Runs & Verification:**
  1. *Validation Split (`run-baseline-whisper-large-v3-20261122-082829`):* 438/438 OK (0 empty, 0 error). Wall time: 2,827.74 s (~6.46 s/utterance). Exported `predictions.jsonl` (SHA-256: `2a228d4d4d79a133ad29d5fec7cc541b759a45e9e68b8ae2f1f49c496f5d9496`, 507,758 bytes), `nbest.jsonl` (SHA-256: `0628c8afbbd03c5708fafadbadb21e4685baf5c9bc77ff92fc881d5524f7e555`, 3,962,370 bytes), and `manifest.json` (SHA-256: `feaeb4abd0f35daef319544dec10de25b04c73d8803e286d8ccc76e59b6c7e02`, 4,455 bytes). Schema contracts verified with `validate_predictions` and `validate_nbest` (0 errors).
  2. *Working-Test Split (`run-baseline-whisper-large-v3-20261122-094140`):* Authorized under Decision D43 working study; 636/636 OK (0 empty, 0 error). Wall time: 4,299.93 s (~6.76 s/utterance). Exported `predictions.jsonl` (SHA-256: `a90268a9a7f0af2f42d461b7e0b239e830f8f2a1577ec304398c63e126fd38b4`, 738,450 bytes), `nbest.jsonl` (SHA-256: `a9d921796b0c78dc88e11e4181773126aaa87e4816b2d766f48c6872b9f01166`, 5,834,129 bytes), and `manifest.json` (SHA-256: `ea76fe1cbb3142a8a5bf3a0419700eed8edc535e8cc8e73588df9d5c4a25fa9f`, 4,439 bytes). Schema contracts verified with `validate_predictions` and `validate_nbest` (0 errors).
  3. *Provenance & Hardware:* Local checkpoint `D:\isntinye\_Istenye uni_\TERM 7\Natural Language Processing\turas_models\whisper-large-v3` (`model.safetensors`, SHA-256 `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`, 3,087,130,976 bytes). Inference executed on NVIDIA GeForce RTX 3060 (12 GB VRAM, compute capability 8.6) in `bfloat16`, peak allocated VRAM 4.236 GiB, peak reserved 4.631 GiB.
  4. *Portable Handoff Bundle:* Assembled portable, reference-free ZIP `turas-b2-whisper-large-v3-m2-handoff.zip` (942,095 bytes, SHA-256: `fd8471e9084a369684f87cccd92e2d0fde1550a725e5dc0f6f854669ffa7e41b`) containing both sets of predictions, N-best lists, run manifests, README, and SHA256SUMS. Strictly sanitized to exclude audio, gold references, weights, and credentials.
- **Smoke CLI Keyword Argument Fix:**
  - Diagnosed signature mismatch where `scripts/smoke_check_whisper_rank1.py` passed `language="tr"` and `task="transcribe"` to `run_whisper_rank1_comparison()` in `src/turkish_asr_rdl/baselines/whisper_rank1.py`.
  - Added optional `language: str | None = None` and `task: str | None = None` kwargs with validation to `run_whisper_rank1_comparison()`, added kwarg validation regression test in `tests/test_m2_whisper_rank1.py`, and updated `tests/test_m2_batch_runner.py`.
  - All 76 Member 2 unit and integration tests pass cleanly.

### 2026-11-19: Response to Member 6 round-two review on PR #6 (commit 54ec128) and CI test fixes (PR #6 merged as cbdc07c)

- **Scope:** Implementation of fixes from Member 6's round-two review on PR #6 ([review 5979396403](https://github.com/Rxd3/TURAS/pull/6#pullrequestreview-5979396403)) and resolution of lean CI test compatibility (commit `e06d0a8`), covering verified local snapshot loading, B1/B2 decoding alignment with M4's PR #5 protocol, formal B3 score convention approval (`qwen3-asr-generation-logprob-sum-v1`), real decode-path testing with tiny random-init models, strict input length limits, shared interface validation, lean CI optional-dependency handling, and packaging updates. All changes approved by Member 6 and merged into `main` at commit `cbdc07cf869c03dd2561554eae0734c3884d7b4d`.
- **Owned paths:** `configs/baselines/whisper_small.json`, `configs/baselines/whisper_large_v3.json`, `configs/baselines/qwen3_asr_1_7b.json`, `configs/baselines/qwen3_asr_0_6b.json`, `src/turkish_asr_rdl/baselines/whisper.py`, `src/turkish_asr_rdl/baselines/qwen.py`, `src/turkish_asr_rdl/baselines/runner.py`, `scripts/smoke_check_whisper_rank1.py`, `pyproject.toml`, `experiments/m2-baseline-notes.md`, `contributions/member_2.md`, `contributions/OMAR_ALZYOD_2409015412_Devinci101.md`, and all `tests/test_m2_*.py`.
- **Key fixes and verified implementations:**
  1. *Exact Local Snapshot Loading:* Verified snapshot loading in `WhisperBaselineAdapter`, `QwenBaselineAdapter`, `runner.py`, and `smoke_check_whisper_rank1.py`. When `--checkpoint` / `checkpoint_path` is provided, loads strictly from that verified path with `local_files_only=True` without Hub or cache fallback. For Hub cache resolution, resolves exact local snapshot with `local_files_only=True`, enforces pinned LFS OIDs (`PINNED_WHISPER_LFS_OIDS`, `PINNED_QWEN_LFS_OIDS`), verifies `config.json` presence, and checks model architecture type. Decoupled production LFS verification from injected mock unit testing. Dedicated adapter-level unit tests verify that explicit local checkpoints deliver the identical directory to model and processor loaders with `local_files_only=True`, that hash mismatches fail before loaders run, and that explicit paths never call the Hub.
  2. *B1/B2 Decoding Alignment with M4:* Matched B1 decoding parameters with M4's frozen B1/F1/F2 protocol from PR #5 (`configs/evaluation/m45_final_protocol.json`, commit `4639b5e`): `bfloat16` precision, `num_beams: 5`, `requested_n: 5`, `num_return_sequences: 5`. Configured B2 identically to maintain parity. Added `generation_config_file: "generation_config.json"` with null hash.
  3. *Qwen Scoring Convention Formalization:* Renamed B3 scoring convention to `qwen3-asr-generation-logprob-sum-v1`, formally approved by Member 6. Documented numerical equivalence: in exact fp32 arithmetic, generation transition log-prob sum matches teacher-forced forward pass log-prob sum within <= 0.00016 nats; in bf16, precision drift across 40–50 tokens is up to ~0.16 nats on 0.6B and ~0.11 nats on 1.7B across 5 clips (as measured by M6). Maintained `decoder_sequence_score` strictly separate from token log-prob sum. Added `chat_template.jinja` and `generation_config.json` metadata with null hashes.
  4. *Input Length Validation:* Implemented explicit clip length guards returning `status="error"`, `error_code="AudioTooLong"`: Whisper rejects clips > 480,000 samples (30 s limit); Qwen rejects clips > 19,200,000 samples (20 min = 1,200 s limit, citing arXiv:2601.21337 Section 2.1). Cited M1 source audit (`data/metadata/worldspeech-tr_tr-source-audit.md:52`) for 10–20 s clip duration expectations.
  5. *Decoder Guardrails & Raw Decoding:* Removed silent Qwen processor/decode fallbacks: strictly raises `AttributeError` on missing `apply_transcription_request` or `decode`, and `ValueError` on missing `input_ids`. Computes `includes_eos` from `kept_tokens[-1] in self._stop_ids` and raises `ValueError` on empty kept tokens. Preserved verbatim raw tokenizer decode (`skip_special_tokens=True`) without strip or repetition removal.
  6. *Shared Interface Validation:* `get_shared_interface()` enforces that `Candidate` contains `lm_log_score`, `rdl_log_score`, `combined_score` and `Hypothesis` contains `selected_candidate_id`, raising `ImportError` early if the contract is incomplete. Decoder config hash includes precision, score convention, and acoustic score kind. Runner wall time clearly annotated as "batch wall time, not RTF" (official RTF measured by M6 on RTX 5070 Ti).
  7. *Real Decode-Path Testing:* Added tiny random-init in-memory `Qwen3ASRForConditionalGeneration` test (fp32, 2 beams, zero downloads) exercising `_decode_nbest_native`, verifying `acoustic_log_score` matches an independent teacher-forced log-prob sum within `< 1e-3`. Updated synthetic Whisper tests to independently recompute teacher-forced sums and verify `diff < 1e-3` while ensuring `decoder_sequence_score != acoustic_log_score`.
  8. *Packaging & Documentation:* Declared `numpy>=1.26.0` in `pyproject.toml` dependencies. Updated Whisper training data to 4,333 h Turkish ASR and 2,241 h translation (Figure 11, Appendix E).
- **Test Evidence (75 tests total across 5 suites):**
  - Offline suite (without external checkout of shared decoder): 75 tests run, 51 passed, 24 skipped cleanly.
  - Opt-in integration suite (with shared decoder from `b6afa13063f7fa90bb1bc190c67b4529e40e91a5`): 75 tests run, 75 passed, 0 failures, 0 errors, 0 skipped in ~11.8s.
  - Test suite breakdown: `tests/test_m2_baseline_configs.py` (6 passed), `tests/test_m2_whisper_rank1.py` (7 passed), `tests/test_m2_whisper_adapter.py` (25 passed), `tests/test_m2_qwen_adapter.py` (29 passed), `tests/test_m2_batch_runner.py` (8 passed).
  - Test coverage context: At PR #6 commit `54ec128`, M6 measured 73% total Member 2 coverage (runner 81%, whisper 70%, qwen 61%; 22 Qwen tests, 33 Whisper/config tests, 8 runner tests). The new decode-path, error guardrail, and explicit checkpoint loading isolation tests further expand coverage to 75 tests.
  - Pretrained smoke runs on real hardware attributed to Member 6 on NVIDIA GeForce RTX 5070 Ti; all Member 2 tests remain strictly local, mock, synthetic, or tiny in-memory.
- **Commit References:**
  - `c3d96d5`: `test(m2): add Whisper rank-1 check and config tests`
  - `cec98c1`: `feat(m2): add Whisper and Qwen baseline adapters`
  - `54ec128`: `feat(m2): add baseline smoke and batch runners`
  - `f58b190`: `Merge remote-tracking branch 'origin/main' into feat/m2-seq2seq-speech-llm-baselines`
  - `a4e82b0`: `fix(m2): address round-two baseline review`
  - `e06d0a8`: `test(m2): fix tests in lean CI`
  - `cbdc07c`: `Merge pull request #6 from Rxd3/feat/m2-seq2seq-speech-llm-baselines`

### 2026-11-19: Real-audio Whisper rank-1 smoke runner and baseline batch inference/export workflow

- **Scope:** Implementation of the real-audio verification path in `scripts/smoke_check_whisper_rank1.py`, creation of the reusable baseline batch runner and exporter workflow (`src/turkish_asr_rdl/baselines/runner.py` & `scripts/run_baseline_batch.py`), and end-to-end integration test coverage (`tests/test_m2_batch_runner.py`).
- **Owned paths:** `scripts/smoke_check_whisper_rank1.py`, `scripts/run_baseline_batch.py`, `src/turkish_asr_rdl/baselines/runner.py`, `tests/test_m2_batch_runner.py`, `experiments/m2-baseline-notes.md`, `contributions/member_2.md`.
- **Smoke Runner Capabilities (`scripts/smoke_check_whisper_rank1.py`):**
  1. Maintained existing `--synthetic` mode for deterministic offline testing.
  2. Implemented real-audio path accepting baseline config (`--config`), local checkpoint directory/file (`--checkpoint`), and explicit audio clips (`--audio`) or reference-free manifests (`--manifest`).
  3. Validates checkpoint bytes against pinned SHA-256 prior to model loading and strictly enforces local filesystem paths (refusing remote URLs).
  4. Runs exact parity comparison between shared N-best decoder rank 1 and Whisper's standard `generate(num_return_sequences=1)` path with identical beam width and Turkish decoding parameters.
  5. Suppresses transcripts by default to prevent transcript leaks; reports per-clip match/mismatch/error status and summary statistics.
- **Batch Runner & Exporter Capabilities (`src/turkish_asr_rdl/baselines/runner.py` & `scripts/run_baseline_batch.py`):**
  1. Supports B1 (`openai/whisper-small`), B2 (`openai/whisper-large-v3`), and both unselected B3 candidates (`Qwen/Qwen3-ASR-1.7B-hf` / `0.6B-hf`).
  2. Checks checkpoint inventory and SHA-256 before model instantiation; rejects remote URIs, missing files, corrupt hashes, and unindexed shards.
  3. Enforces reference-free input manifest allowlist; strictly rejects any manifest containing reference transcripts (`reference_text` or `gold`).
  4. Preserves batch order, utterance IDs, and captures per-clip errors with `status='error'` without dropping items.
  5. Produces schema-compliant `predictions.jsonl`, `nbest.jsonl`, and `manifest.json` outside the repository using shared exporter utilities and verifies them using shared validators.
- **Test Evidence (63 tests total):**
  - Default offline suite: 63 tests run, 41 passed, 22 skipped cleanly (all tests requiring external checkout of shared decoding package skip safely).
  - Full opt-in integration suite: 63 tests run in 9.043s, 63 passed, 0 failures, 0 errors, 0 skipped against provisional shared revision `b6afa13063f7fa90bb1bc190c67b4529e40e91a5`.

### 2026-11-19: Qwen3-ASR baseline adapter implementation and scoring trace verification

- **Scope:** Implementation of reusable Qwen3-ASR baseline adapter (`src/turkish_asr_rdl/baselines/qwen.py`) and dedicated unit/integration test suite (`tests/test_m2_qwen_adapter.py`).
- **Owned paths:** `src/turkish_asr_rdl/baselines/qwen.py`, `tests/test_m2_qwen_adapter.py`, `configs/baselines/qwen3_asr_1_7b.json`, `configs/baselines/qwen3_asr_0_6b.json`, `experiments/m2-baseline-notes.md`, `contributions/member_2.md`.
- **Candidate eligibility & selection:** Both B3 candidate configurations (`Qwen/Qwen3-ASR-1.7B-hf` and `Qwen/Qwen3-ASR-0.6B-hf`) remain explicitly unselected (`candidate_status == "unselected-candidate"`). Neither checkpoint is cleared or bound to datasets; selection is deferred to validation WER/CER and standardized M6 timing. No pretrained weights or audio were downloaded, and no pretrained baseline inference was run.
- **Adapter capabilities:**
  1. Satisfies the shared `AsrAdapter` protocol (`system_id`, `requested_n`, `score_convention_id`, `model_artifact_sha256`, `decoder_config_sha256`, `config`).
  2. Accepts either pinned candidate config without modifying or selecting either candidate.
  3. Enforces reference-free 16 kHz audio input, preserving batch order and utterance IDs across all items.
  4. Returns explicit `status="empty"` (`text=""`) for empty audio and `status="error"` (`text=None`) for decode failures without dropping items or altering batch ordering.
  5. Formats Turkish transcription requests via native `processor.apply_transcription_request(audio=[...], language="tr")`.
  6. Preserves decoded text verbatim (`raw-v1`) and delegates deduplication to shared export via raw ranked candidates.
- **Scoring trace and verification:**
  - Traced and confirmed that `speech_llm_joint_token_log_prob_sum` (convention `smoke-qwen3-asr-joint-token-logprob-sum-v1`) originates from `transformers.GenerationMixin.compute_transition_scores` over generation output (`out.sequences`, `out.scores`, `out.beam_indices`), matching M6's smoke implementation. It does **not** derive from a separate teacher-forced forward pass over the original model logits; these methods are not interchangeable without concrete empirical evidence.
  - Slices generated tokens through the first stop token (EOS) via `cut_at_stop`, strictly excluding prompt tokens (not in generation steps) and trailing padding tokens.
  - Keeps library beam score (`sequences_scores` with length penalty) separate in `decoder_sequence_score` (or records explicit null reasons in `score_details`).
- **Offline test evidence:**
  - Deterministic scoring tests in `tests/test_m2_qwen_adapter.py` exercise the exact adapter path (`compute_transition_scores` for greedy and beam search + `compute_joint_log_probs_from_step_scores`), confirming prompt/padding exclusion, EOS inclusion, and exact numerical match to transition log probabilities.
  - 20 Qwen tests and 35 existing tests (55 total) passed under pinned Python 3.13 (`$env:LOCALAPPDATA\TURAS\venv-313\Scripts\python.exe`).
  - Full opt-in integration suite passed against shared revision `b6afa13063f7fa90bb1bc190c67b4529e40e91a5` (55 passed, 0 failures, 0 errors, 0 skips).

### 2026-11-19: Whisper baseline adapter implementation and response to M6 PR #6 review

- **Scope:** Implementation of reusable Whisper baseline adapter (`src/turkish_asr_rdl/baselines/whisper.py`), comprehensive unit and integration tests (`tests/test_m2_whisper_adapter.py`), and full resolution of offline items from Member 6's review on PR #6.
- **Owned paths:** `configs/baselines/whisper_small.json`, `configs/baselines/whisper_large_v3.json`, `configs/baselines/qwen3_asr_1_7b.json`, `configs/baselines/qwen3_asr_0_6b.json`, `experiments/m2-baseline-notes.md`, `src/turkish_asr_rdl/baselines/whisper.py`, `tests/test_m2_baseline_configs.py`, `tests/test_m2_whisper_adapter.py`, `contributions/member_2.md`.
- **Review items resolved:**
  1. *WorldSpeech Eligibility:* Settled B1 and B2 as eligible based on release dates (2022, 2023) predating WorldSpeech creation (2026-04-28) and test sessions (2026-01-06 to 2026-04-22). Retained B3 candidates as unverified / open due to June 2026 `-hf` creation date and file size differences, documenting required verification paths (Qwen format-conversion statement or tensor comparison on 5070 Ti) and stated limitation for 6–27 Jan test sessions.
  2. *ASR-Agreement Filter:* Correctly documented that TURAS retains `cer <= 0.25` across all three splits (3,886 rows removed by M1), comparing minutes with upstream ASR transcript (mean 0.075, max 0.50). Dropped unconfirmed `[CER < 0.3]` and specific model claims for `tr_tr`; maintained alignment model family bias as a documented project limitation.
  3. *Qwen3-ASR Score Specification:* Updated `acoustic_score_kind` to `speech_llm_joint_token_log_prob_sum` under convention `smoke-qwen3-asr-joint-token-logprob-sum-v1`, preserving library beam scores (`sequences_scores`) in `decoder_sequence_score`.
  4. *Full Decoding Policy:* Recorded complete decoding parameters (`length_penalty: 1.0`, `early_stopping: false`, `repetition_penalty: 1.0`, `no_repeat_ngram_size: 0`, `num_return_sequences`, `temperature: null`, forced prefixes, and Qwen prompt format) and audio constraints (10–20 s clips fitting inside Whisper's 30 s window and Qwen's frame limit without chunking).
  5. *Weight File Metadata:* Verified and recorded exact `model.safetensors` filenames and Hub LFS OIDs across all 4 models, labeled "Hub LFS oid; bytes verified at download". Pinned B2 specifically to `model.safetensors` to distinguish from fp32 shards in the repo.
  6. *Selection Criteria:* Removed smoke RTF figures from ranking and selection criteria; documented that B3 selection is determined by validation WER/CER plus standardized RTF from the unified M6 timing session.
  7. *Deduplication Policy:* Updated policy string across configs and notes to contract text: `"exact raw-text match; first (lowest-rank) occurrence kept; removed IDs listed in dedup_map"`.
  8. *Citations:* Corrected Whisper training statistics citation to Figure 11 (Appendix E) quoting ~2,286 h Turkish ASR and ~2,241 h translation; corrected `docs/m6-smoke-results.md` citation to branch `feat/m6-docs` at commit `9f09a5f`.
- **Adapter & Test Verification:**
  - Implemented `WhisperBaselineAdapter` satisfying `AsrAdapter` protocol, enforcing reference-free 16 kHz inputs, batch order and utterance ID preservation, empty audio handling (`status="empty"`, `text=""`), error capture without dropping IDs, verbatim whitespace emission (`raw-v1`), and raw candidate ordering with deduplication delegated to shared export.
  - Enforced `model_artifact_sha256` artifact contract compliance (64-char lowercase hex) and rejection of unsupported decoding overrides.
  - All 33 Member 2 tests passed under `$env:LOCALAPPDATA\TURAS\venv-313\Scripts\python.exe` with shared decoder integration enabled from revision `b6afa13063f7fa90bb1bc190c67b4529e40e91a5` (22 tests passed offline when opt-in integration is skipped).

### 2026-11-19: Baseline config tests and shared-decoder environment verification

- **Scope:** Verification of Member 2 baseline configurations, Windows runtime environment readiness, and synthetic validation of the shared Whisper decoder helper.
- **Owned paths:** `tests/test_m2_baseline_configs.py`, `experiments/m2-baseline-notes.md`, `contributions/member_2.md`.
- **Collaborating members & dependencies:** M6 (Windows runtime environment lock and environment checker), M4/M5 (shared decoder helper and integration).
- **Checks actually run:**
  1. Six offline config checks passed via `tests/test_m2_baseline_configs.py` (auditing JSON parsing, pinned Hugging Face model/processor IDs and revisions, unset dataset bindings, and unselected status for Qwen candidates across `configs/baselines/`).
  2. M6’s pinned Windows runtime environment (`requirements/windows-cu130.txt` at revision `bf4e0bf1319aa573f90f73e3cf8ec0a29a197608`) was set up in an isolated Python 3.13 virtual environment; running M6’s environment checker `turkish_asr_rdl.utils.env_check` reported `READY` on Windows with NVIDIA GeForce RTX 3060 (driver 595.79, CUDA 13.0 build `torch-2.14.1+cu130`).
  3. Shared-decoder verification: Exercised the shared Whisper decoding helper (`whisper_nbest` and `teacher_forced_score` from `feat/m4-m5-integration` at revision `b6afa13063f7fa90bb1bc190c67b4529e40e91a5`) using genuine PyTorch (`2.14.1+cu130`) and Transformers (`5.23.0`) on synthetic mel spectrogram inputs (`[1, 80, 3000]`). Two synthetic test cases passed: one on CPU and one on CUDA (NVIDIA GeForce RTX 3060).
- **Scope clarification:** This execution constitutes shared-decoder verification on synthetic input under the pinned runtime. It is **not** a Member 2 baseline adapter test, nor a pretrained baseline run, and does **not** complete Checkpoint 2. No pretrained weights or audio datasets were downloaded.

## Experiments and artifacts

- Baseline B2 (`openai/whisper-large-v3`) reference-free batch inference complete for both validation (438/438 OK, `run-baseline-whisper-large-v3-20261122-082829`) and D43-authorized working-test (636/636 OK, `run-baseline-whisper-large-v3-20261122-094140`).
- Exported B2 artifacts verified and contract-compliant: `predictions.jsonl`, `nbest.jsonl`, and `manifest.json` for both splits match Schema 0.1 (`validate_predictions` and `validate_nbest` pass with 0 errors).
- Portable B2 handoff package `turas-b2-whisper-large-v3-m2-handoff.zip` (942,095 bytes, SHA-256 `fd8471e9084a369684f87cccd92e2d0fde1550a725e5dc0f6f854669ffa7e41b`) created outside Git and delivered to Member 6.
- Baseline B1 (`openai/whisper-small`): Unmodified baseline control evaluated within the frozen M4/M5 benchmark block (`results/final/m45_metrics.csv`, `results/final/M4_M5_FINAL_REPORT.md`) by Members 4 & 6 using the matched protocol (`configs/evaluation/m45_final_protocol.json`); attributed accordingly without duplicate execution.
- Baseline B3: Qwen3-ASR adapter implemented; both candidates (1.7B and 0.6B) unselected pending Decision D05 resolution.
- Full unit and integration test coverage in `tests/test_m2_*.py` (76 tests passing).

## Reviews and handoffs

- PR #6 merged into `main` (`cbdc07c`); PR #20 merged into `main` (`1dade5b`).
- Portable B2 handoff package `turas-b2-whisper-large-v3-m2-handoff.zip` delivered to Member 6 for independent reference-anchored WER/CER evaluation and standardized 3-repeat RTF benchmarking on the RTX 5070 Ti platform.
- B1 baseline results verified in `results/final/` and acknowledged as satisfying the shared baseline requirement.
- B3 candidate comparison and eligibility analysis prepared for Member 6 review to close Decision D05.

## Remaining work & Unresolved prerequisites

1. **B2 Official Evaluation & Timing (Owner: Member 6):** Independent evaluation of B2 predictions against released, adjudicated references (WER/CER, canonical terminology) and standardized 3-repeat timing on RTX 5070 Ti under `run_timed`.
2. **Decision D05 Closure & Candidate Selection for B3 (Owners: Member 6 & Member 2):** Formal agreement on `Qwen/Qwen3-ASR-1.7B-hf` (Candidate 1) pretraining timeline disclosure / acceptance under declared limitations, followed by download authorization, validation run, and working-test inference.
