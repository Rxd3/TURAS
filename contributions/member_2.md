# Member 2 — Seq2Seq and Speech-LLM Baselines

Name: unassigned. GitHub username: unassigned.
Branch convention: `feat/m2-seq2seq-speech-llm-baselines` (focused topic suffixes allowed).
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

**Status: in progress (preliminary verification and environment readiness).** Checkpoint 2 is incomplete; no baseline adapters implemented or pretrained models executed.

## Actual contributions

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
