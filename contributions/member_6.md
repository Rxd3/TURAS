# Member 6 — Integration, Evaluation, and QA

Name: Member 6. GitHub username: `afel6`.
Branch convention: `feat/m6-integration-evaluation-qa` (focused topic suffixes allowed).
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

**Status: M6 checker kit, environment/CI setup, cross-team PR reviews, and submission-readiness integration complete.**

## Actual contributions

* **2026-11-17 — Pinned Environment, Env Check, and Test CI ([PR #1](https://github.com/Rxd3/TURAS/pull/1), merged `7babbc5`):**
  * Added pinned Windows CUDA 13.0 environment requirements (`requirements/windows-cu130.txt`, `requirements/dev.txt`), `scripts/check_env.py`, and `src/turkish_asr_rdl/utils/env_check.py`.
  * Configured GitHub Actions CI for packaging and unit tests across Python 3.11 and 3.13 (`.github/workflows/`).
  * Verified 4/4 CI checks green on `main`.
* **2026-11-17 to 2026-11-19 — Shared Evaluation, Schema Validation, Decoding Harness, and Smoke Kit (`feat/m6-evaluator`, `feat/m6-harness`, `feat/m6-smoke-tests`, `feat/m6-docs`; integrated into `main` via PR #5 `62fadf5` and updated in the M6 cleanup PR):**
  * Implemented `tr-normalized-v1` Turkish text normalizer, Levenshtein word/character aligner, strict ID joiner, and WER/CER/bootstrap metrics (`src/turkish_asr_rdl/evaluation/{normalize,align,join,metrics}.py`) with full unit test coverage (`tests/evaluation/`).
  * Implemented JSONL/prediction/N-best schema validators (`src/turkish_asr_rdl/schemas/`) and shared decoding/export/timing/hardware/memory harness (`src/turkish_asr_rdl/decoding/{interface,audio,export,hardware,memory,timing}.py`) with unit tests (`tests/schemas/`, `tests/decoding/`).
  * Built the RTX 5070 Ti smoke-test kit (`experiments/m6-smoke/`, `tests/smoke/`) and documented real-model smoke results, memory profiles, and protocol alignment in `docs/m6-smoke-results.md`.
  * Updated shared contracts (`docs/contracts/`) and evaluation documentation (`docs/evaluation.md`).
* **2026-11-18 to 2026-11-21 — Cross-Team PR Audits, Reviews, Merges, and Submission Cleanup:**
  * Reviewed and verified PR #2 (M1), PR #3 (M4), PR #4 (M5), PR #5 (M4/M5 integration), PR #6 (M2, three rounds including RTX 5070 Ti real-model verification of Whisper-small, Whisper-large-v3, Qwen3-ASR-0.6B, and Qwen3-ASR-1.7B, plus fp32 vs bf16 teacher-forced score verification), and PR #7 (M3, two rounds).
  * Merged PR #1, PR #3 (`2f20f92`), PR #4 (`fbe495d`), PR #5 (`62fadf5`), PR #6 (`cbdc07c`), and PR #7 (`695016f`) into `main`.
  * Added root `LICENSE` (MIT + CC BY-NC 4.0 data notice), removed internal planning/handoff process files, updated stale READMEs, and added the grader quickstart and hash/metric verification workflow on `main`.

## Experiments and artifacts

* **RTX 5070 Ti smoke verification (`docs/m6-smoke-results.md`):** verified 20-clip smoke runs for Whisper-small, Whisper-large-v3, Qwen3-ASR-0.6B-hf, and Qwen3-ASR-1.7B-hf (all 20/20 `ok`); verified Whisper rank-1 equivalence (5/5 clips) and exact teacher-forced acoustic score match; verified Qwen3-ASR-0.6B fp32 generation vs teacher-forced log-prob agreement within `0.00016` nats (bf16 drift up to `0.16` nats).
* **Official RTF timing:** accepted PR #5's frozen repeated timing block for the eight M4/M5 conditions (`docs/m6-smoke-results.md`).

## Reviews and handoffs

* **PR #1 (`feat/m6-env`):** authored and merged (`7babbc5`).
* **PR #2 (`feat/m1-data-relational-schema`):** reviewed (requested changes on metadata/schema consistency).
* **PR #3 (`feat/m4-whisper-finetuning`):** reviewed, approved, and merged (`2f20f92`).
* **PR #4 (`feat/m5-ctc-rdl-rescoring`):** reviewed, approved, and merged (`fbe495d`).
* **PR #5 (`feat/m4-m5-integration`):** reviewed (verified 56 pinned code hashes, split/manifest/lexicon hashes, S/D/I metric recomputation across 20 systems, and 1,084 lean tests), approved, and merged (`62fadf5`).
* **PR #6 (`feat/m2-seq2seq-speech-llm-baselines`):** reviewed rounds 1–3 ([round 2](https://github.com/Rxd3/TURAS/pull/6#pullrequestreview-5979396403)), approved at `e06d0a8`, and merged (`cbdc07c`).
* **PR #7 (`feat/m3-ctc-alternative-baselines`):** reviewed rounds 1–2 ([round 1](https://github.com/Rxd3/TURAS/pull/7#pullrequestreview-5979398124)), approved at `9351849`, and merged (`695016f`).

## Remaining work

* B2 (Whisper-large-v3), B3 (Qwen3-ASR), and B5 (BiLSTM-CTC) full test-set runs remain unexecuted and are marked accordingly in `README.md`.
* Public release of fine-tuned model weights or per-utterance test hypotheses remains pending team/owner decision under WorldSpeech CC BY-NC 4.0 terms.
