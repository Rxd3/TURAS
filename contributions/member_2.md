# Member 2 — Seq2Seq and Speech-LLM Baselines

Name: unassigned. GitHub username: unassigned.
Branch convention: `feat/m2-seq2seq-speech-llm-baselines` (focused topic suffixes allowed).
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

**Status: in progress (baseline adapters implemented and verified offline; awaiting dataset release and validation freeze).** Checkpoint 2 baseline adapters (B1, B2, B3) implemented and tested offline; no pretrained models executed on dataset splits.

## Actual contributions

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

## Reviews and handoffs

- Baseline configuration checks and model audit documented for M4 and M6 review.
- Awaiting team resolution on shared interfaces before adapter implementation.

## Remaining work & Unresolved prerequisites

Implementation of the B1 Whisper-small baseline adapter and its production test suite is blocked by the following unresolved prerequisites:
1. **Recorded agreement covering Member 2’s prediction and decoding contract:** Decisions D23 and D24 recorded on `feat/m4-m5-integration` at `b6afa13063f7fa90bb1bc190c67b4529e40e91a5` remain provisional integration proposals. A formal team-recorded agreement covering Member 2’s prediction and decoding interface (signed off with M4 and M6) is not yet recorded on `main` or this branch.
2. **Real shared modules on the current branch:** The current branch (`feat/m2-seq2seq-speech-llm-baselines`) contains only empty package stubs (`src/turkish_asr_rdl/decoding/__init__.py`, `schemas/__init__.py`, `utils/__init__.py`). The real shared modules (`turkish_asr_rdl.decoding.whisper`, `turkish_asr_rdl.decoding.interface`, `turkish_asr_rdl.decoding.export`, schemas, etc.) exist only on integration branch `feat/m4-m5-integration` and have not yet landed on `main` or been merged into this branch.
