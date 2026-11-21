# Member 2 — Seq2Seq and Speech-LLM Baselines

Name: Omar Alzyod. Student ID: `2409015412`. GitHub username: [`Devinci101`](https://github.com/Devinci101).
Role: Member 2 — Seq2Seq and Speech-LLM Baselines.
Branch convention: `feat/m2-seq2seq-speech-llm-baselines` (focused topic suffixes allowed).
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).
Backward-compatible record: [member_2.md](member_2.md).

**Status: PR #6 approved and merged to `main` (`cbdc07c`); baseline adapters (B1, B2, B3), smoke runner, batch exporter workflow, and all review fixes merged and verified offline (75 tests passing).** Checkpoint 2 baseline adapters (B1, B2, B3), smoke runners, batch exporters, and decoding contracts implemented, reviewed by Member 6, and merged; no pretrained models executed on dataset splits.

## Actual contributions

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

- No pretrained baseline inference runs or benchmark evaluations completed.
- Pinned configuration audit and decoding protocols documented in `experiments/m2-baseline-notes.md`.
- Reusable baseline adapters implemented for B1, B2, and B3 in `src/turkish_asr_rdl/baselines/`.
- Reusable rank-1 smoke runner implemented in `scripts/smoke_check_whisper_rank1.py`.
- Reusable baseline batch inference and shared export workflow implemented in `src/turkish_asr_rdl/baselines/runner.py` and `scripts/run_baseline_batch.py`.
- Full integration and unit test coverage in `tests/test_m2_*.py` (75 tests).

## Reviews and handoffs

- PR #6 updated with Whisper and Qwen adapters, baseline runners, exporter workflows, and Member 6 round-two review fixes. Approved by Member 6 (`@afel6`) and merged into `main` at commit `cbdc07cf869c03dd2561554eae0734c3884d7b4d` (`cbdc07c`) on 2026-11-20T21:19:03Z.
- Reusable runners and exporter workflows tested locally with temporary fixtures, synthetic inputs, and mock loaders; ready for execution as soon as official dataset manifests and approved checkpoints arrive.
- **Handoff Sequence:** The first real run is B1/B2 rank-1 smoke check on approved local clips and verified weights, followed by validation decoding and parameter freeze (B1 and B2 aligned to beam 5, requested_n 5 following M4's PR #5 protocol; B3 beam search selection deferred to validation), followed by official test export using the frozen setting. Both Qwen candidates remain strictly unselected.

## Remaining work & Unresolved prerequisites

Executing actual pretrained baseline inference runs and handing off official prediction bundles to Member 6 is blocked by the following unresolved prerequisites:
1. **Approval of PR #6 Review (RESOLVED):** PR #6 was approved by Member 6 (`@afel6`) and merged into `main` at commit `cbdc07c` (merge commit `cbdc07cf869c03dd2561554eae0734c3884d7b4d`) with all 75 tests passing.
2. **Official Dataset Release & Split Manifests (Owners: Member 1 & Member 6):** Release of official `dataset_release` name and `split_manifest_sha256` for the test and validation splits to bind into baseline configs and input manifests.
3. **Local Weight Checkpoint Downloads & SHA-256 Verification (Owner: Member 2 / Runner Operator):** Local download of safetensors files to destination outside repository and verification of the 64-character lowercase hex SHA-256 artifact hashes and pinned LFS OIDs.
4. **Validation Beam / N Confirmation (Owners: Member 2 & Member 4):** Validation run confirmation of the matched beam 5 / requested_n 5 protocol on validation set before freezing settings for official test runs; selection of B3 beam width.
5. **B3 Checkpoint Eligibility Verification (Owners: Member 6 & Member 2):** Resolve timeline and pretraining contamination audit for `-hf` checkpoints (confirming whether June 2026 weights are tensor-identical to January 28, 2026 release or obtaining conversion statement) before official B3 execution. Both candidates remain strictly unselected.
6. **B3 Scoring Convention Status (RESOLVED):** Transition score convention `qwen3-asr-generation-logprob-sum-v1` (computed via generation step transition log-prob scores through EOS) formally approved by Member 6 for B3. Numerical equivalence and bf16 precision drift documented.
