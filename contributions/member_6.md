# Member 6 — Integration, Evaluation, and QA

Name: Abdalraouf Alkabir. Student ID: `2309015482`. GitHub username: [`afel6`](https://github.com/afel6).
Branch convention: `feat/m6-integration-evaluation-qa` (focused topic suffixes allowed).
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

**Status: M6 checker kit, environment/CI setup, cross-team PR reviews, and submission-readiness integration complete.**

## Actual contributions

* **2026-11-17 — Pinned Environment, Env Check, and Test CI ([PR #1](https://github.com/Rxd3/TURAS/pull/1), merged `7babbc5`):**
  * Added pinned Windows CUDA 13.0 environment requirements (`requirements/windows-cu130.txt`, `requirements/dev.txt`), `scripts/check_env.py`, and `src/turkish_asr_rdl/utils/env_check.py`.
  * Configured GitHub Actions CI for packaging and unit tests across Python 3.11 and 3.13 (`.github/workflows/`).
  * Verified 4/4 CI checks green on `main`.
* **2026-11-17 to 2026-11-19 — Shared Evaluation, Schema Validation, Decoding Harness, and Smoke Kit (`feat/m6-evaluator`, `feat/m6-harness`, `feat/m6-smoke-tests`, and `feat/m6-docs` up to `9f09a5f`; integrated into `main` via PR #5 `62fadf5`. The later `feat/m6-docs` commits were superseded by PR #8 `cca13d7`):**
  * Implemented `tr-normalized-v1` Turkish text normalizer, Levenshtein word/character aligner, strict ID joiner, and WER/CER/bootstrap metrics (`src/turkish_asr_rdl/evaluation/{normalize,align,join,metrics}.py`) with full unit test coverage (`tests/evaluation/`).
  * Implemented JSONL/prediction/N-best schema validators (`src/turkish_asr_rdl/schemas/`) and shared decoding/export/timing/hardware/memory harness (`src/turkish_asr_rdl/decoding/{interface,audio,export,hardware,memory,timing}.py`) with unit tests (`tests/schemas/`, `tests/decoding/`).
  * Built the RTX 5070 Ti smoke-test kit (`experiments/m6-smoke/`, `tests/smoke/`) and documented real-model smoke results, memory profiles, and protocol alignment in `docs/m6-smoke-results.md`.
  * Updated shared contracts (`docs/contracts/`) and evaluation documentation (`docs/evaluation.md`).
* **2026-11-18 to 2026-11-21 — Cross-Team PR Audits, Reviews, Merges, and Submission Cleanup:**
  * Reviewed and verified PR #2 (M1), PR #3 (M4), PR #4 (M5), PR #5 (M4/M5 integration), PR #6 (M2, three rounds including RTX 5070 Ti real-model verification of Whisper-small, Whisper-large-v3, Qwen3-ASR-0.6B, and Qwen3-ASR-1.7B, plus fp32 vs bf16 teacher-forced score verification), and PR #7 (M3, two rounds).
  * Merged PR #1, PR #3 (`2f20f92`), PR #4 (`fbe495d`), PR #5 (`62fadf5`), PR #6 (`cbdc07c`), and PR #7 (`695016f`) into `main`.
  * Added root `LICENSE` (MIT + CC BY-NC 4.0 data notice), removed internal planning/handoff process files, updated stale READMEs, and added the grader quickstart and hash/metric verification workflow on `main`.
* **2026-11-22 — B2/B3 decodes, scoring, and same-session timing ([PR #24](https://github.com/Rxd3/TURAS/pull/24)):**
  * Ran the C2 same-session timing of B1, B2, and B3 on the RTX 5070 Ti laptop (frozen timing block settings, 107 utterances, 3 repeats).
  * Decoded B2 (`openai/whisper-large-v3`, bf16, beam 5, N 5) and B3 on validation and test, and the Qwen3-ASR-0.6B B3 candidate on validation; selected Qwen3-ASR-1.7B-hf as B3 on validation.
  * Added the M6 scorer `scripts/m6_score_baselines.py` (same library calls as `scripts/m45_final_evaluate.py`) and the timing helper `scripts/m6_b23_harness_run.py`, and scored B2, Member 2's reproduction, and B3 ([`results/m6/`](../results/m6/README.md)).
  * Checked Member 2's B2 handoff (zip `fd8471e9`) and scored it as an independent reproduction (577/636 test one-best hypotheses byte-equal).
  * Verified that the `Qwen/Qwen3-ASR-1.7B-hf` weights at `bcd2b5b7` are byte-identical (707/707 tensors) to the original `Qwen/Qwen3-ASR-1.7B` at `7278e1e`.

## Experiments and artifacts

* **RTX 5070 Ti smoke verification (`docs/m6-smoke-results.md`):** verified 20-clip smoke runs for Whisper-small, Whisper-large-v3, Qwen3-ASR-0.6B-hf, and Qwen3-ASR-1.7B-hf (all 20/20 `ok`).
* **Score checks in the [PR #6](https://github.com/Rxd3/TURAS/pull/6) round-3 review (not recorded in `docs/m6-smoke-results.md`):** verified Whisper rank-1 equivalence (5/5 clips) and exact teacher-forced acoustic score match; verified Qwen3-ASR-0.6B fp32 generation vs teacher-forced log-prob agreement within `0.00016` nats (bf16 drift up to `0.16` nats).
* **B2/B3 baseline results ([`results/m6/`](../results/m6/README.md)):** test normalized WER 12.3192 (B2), 12.3922 (Member 2's reproduction), and 16.0943 (B3); same-session end-to-end RTF 0.05757 (B1), 0.17537 (B2), and 0.13843 (B3) (same-session RTX 5070 Ti laptop numbers, not comparable to the frozen-block RTFs; compare through the B1 ratio).
* **Official RTF timing:** accepted PR #5's frozen repeated timing block for the eight M4/M5 conditions (`docs/m6-smoke-results.md`).

## Reviews and handoffs

* **PR #1 (`feat/m6-env`):** authored and merged (`7babbc5`).
* **PR #2 (`feat/m1-data-relational-schema`):** reviewed (requested changes on metadata/schema consistency).
* **PR #3 (`feat/m4-whisper-finetuning`):** reviewed, approved, and merged (`2f20f92`).
* **PR #4 (`feat/m5-ctc-rdl-rescoring`):** reviewed, approved, and merged (`fbe495d`).
* **PR #5 (`feat/m4-m5-integration`):** reviewed (verified 56 pinned code hashes, split/manifest/lexicon hashes, S/D/I metric recomputation across 20 systems, and 1,084 lean tests), approved, and merged (`62fadf5`).
* **PR #6 (`feat/m2-seq2seq-speech-llm-baselines`):** reviewed rounds 1–3 ([round 2](https://github.com/Rxd3/TURAS/pull/6#pullrequestreview-5979396403)), approved at `e06d0a8`, and merged (`cbdc07c`).
* **PR #7 (`feat/m3-ctc-alternative-baselines`):** reviewed rounds 1–2 ([round 1](https://github.com/Rxd3/TURAS/pull/7#pullrequestreview-5979398124)), approved at `9351849`, and merged (`695016f`).
* **PR #8 (`feat/m6-submission-cleanup`):** authored and merged (`3f10894`) — root `LICENSE`, grader quickstart, `docs/m6-smoke-results.md` sync, and stale README/process-file cleanup.
* **PR #9 (`feat/m6-final-polish`):** authored and merged (`95089eb`) — quantitative error analysis & RDL failure diagnosis (`docs/error_analysis.md`), standalone verifier (`scripts/verify_submission.py`), aggregate-only N-best oracle script (`scripts/compute_nbest_oracle.py`), `presentation/SLIDE_OUTLINE.md`, and pure MIT `LICENSE` + separate `NOTICE`.
* **PR #10 (`feat/m3-ctc-adapters-tests`):** reviewed, approved, and merged (`1ad2d7c`).

## Remaining work

* B5 (BiLSTM-CTC) full test-set run remains unexecuted and is marked accordingly in `README.md`.
* B2 and B3 are scored outside the frozen protocol; B1 parity of the M6 scorer and paired intervals against F1/F2 are pending the private M4/M5 handoff bundle.
* Public release of fine-tuned model weights or per-utterance test hypotheses remains pending team/owner decision under WorldSpeech CC BY-NC 4.0 terms.
