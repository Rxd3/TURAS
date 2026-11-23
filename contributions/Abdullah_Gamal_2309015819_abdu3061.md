# Member 3 — CTC and Classical Baselines

Name: Abdullah Gamal student id:2309015819 GitHub username: abdu3061
Branch convention: `feat/m3-ctc-alternative-baselines` (focused topic suffixes allowed).
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

**Status: Completed — PR #7 (`695016f`) merged to `main`; B4/B5 adapters and unit tests implemented (`1156 passed, 20 skipped, 0 failed`); B5 (`BiLSTM-CTC`, Epoch 22, 47.78% test norm WER) and B6 (`single-Gaussian character HMM`, 100.00% test norm WER) trained from scratch and archived under [`results/m3/`](../results/m3/).**

## Actual contributions

- **2026-11-21 — B4/B5 Checkpoint Audit & Protocol Configs (PR #7, merged as `695016f`):**
  - Audited and pinned `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` (`708639f50559d7970f462e13ec64d3f059ca89f6`, `cc-by-4.0`, `pytorch_model.bin` `1,262,079,473` bytes, Hub LFS oid `008d69c82bf45c82ec5485e550abc86ddabefe75080f22e1ebfdc1699889d82e`) as primary B4 (`B4-xlsr-tr-cv7`) to match M5's F3 parent control, keeping `mpoyraz/wav2vec2-xls-r-300m-cv8-turkish` (`bdd0bb878d8bf59509611631e334eb073ba57773`, `apache-2.0`, Hub LFS oid `dbd7a9f776a232c0dd4cbbee44025c8056246d83ad4de930b049e649e2fe402c`) as a labeled secondary candidate in `configs/baselines/wav2vec2_xlsr_turkish.json`.
  - Aligned B4 primary decoding with M5's frozen F3 protocol (`bfloat16`, CTC prefix beam search with `beam_size=10`, `requested_n=5`, rank-1 one-best, adopting `turkish_asr_rdl.decoding.ctc` over a separate NumPy scorer, with `float32` greedy retained only as a labeled secondary diagnostic) and documented the 38-token vocabulary (`blank=pad=0`, `unk=3`, `|=4`, missing digits/diacritics, no post-processing, KenLM excluded).
  - Proposed `configs/baselines/bilstm_ctc.json` for B5 (`BiLSTM-CTC` trained from scratch, `tr-normalized-v1` targets/validation reference, subsampling factor `4`, `zero_infinity=True` + minimum-length feasibility check, `ReduceLROnPlateau` on `validation_norm_cer`, `held_out_exposure: "none"`, and a mandatory 3-step memory + 1-epoch probe before locking numeric compute ceilings).
  - Documented per-checkpoint `2022-03-23` commit-date held-out eligibility and limitations in `experiments/m3-baseline-notes.md`.
  - **Collaborating members / required reviewers:** `@afel6` (Member 6) and `@Rxd3` (Member 5).

- **2026-11-21 — B4/B5 Python Adapters & Offline Unit Tests (`feat/m3-ctc-adapters-tests`):**
  - Implemented `Wav2Vec2CtcBaselineAdapter` in `src/turkish_asr_rdl/baselines/ctc.py`, enforcing explicit `Wav2Vec2CTCTokenizer` + `Wav2Vec2FeatureExtractor` loading (ignoring upstream KenLM / `Wav2Vec2ProcessorWithLM`), `float32` log-softmax conversion before calling `turkish_asr_rdl.decoding.ctc.ctc_nbest`, raw whitespace preservation (`raw-v1`), per-utterance `CTCLoss(reduction="none", zero_infinity=False)` cross-checking, and `AsrAdapter` contract compliance.
  - Implemented `BiLSTMCTC`, `CharCtcTokenizer`, `LogMelSpectrogramExtractor`, `check_ctc_feasibility`, and `run_3step_memory_probe` in `src/turkish_asr_rdl/models/bilstm_ctc.py`, plus `BiLstmCtcBaselineAdapter` and `save_bilstm_checkpoint` in `src/turkish_asr_rdl/baselines/bilstm_ctc.py` (strictly train-only character inventory, `4x` stride-2 Conv2D subsampling, `zero_infinity=True` feasibility filtering, and `AsrAdapter` compliance).
  - Added offline synthetic unit tests in `tests/test_m3_ctc_baselines.py` verifying `check_adapter` conformance, raw edge-whitespace preservation, batch error/empty isolation, train-only `<unk>` mapping, `4x` subsampling length invariance, and B5 3-step probe + checkpoint round-trip (`python -m pytest -q`: `1156 passed, 20 skipped, 0 failed`).
  - **Collaborating members / required reviewers:** `@afel6` (Member 6) and `@Rxd3` (Member 5).

- **2026-11-22 — B5 (`BiLSTM-CTC`) & B6 (`Single-Gaussian Character HMM`) From-Scratch Training & Final Evaluation (`results/m3/`):**
  - Trained B5 (`BiLSTM-CTC`) from scratch on the frozen `train` split of `turas-tr-econ-legis-v0.1` (initiated locally on CPU through Epoch 13/14 and resumed to completion on GPU via `scripts/m3_resume_b5_gpu.py` and `scripts/m3_finalize_b5.py`, stopping early at Epoch 28).
  - Selected **Epoch 22** on validation before test scoring, achieving **47.78% test normalized WER**, **13.50% test normalized CER**, **0.891 term micro-F1**, and **0.0093 end-to-end RTF**.
  - Trained and evaluated the B6 classical baseline (`single-Gaussian character HMM` via `scripts/m3_run_gaussian_hmm.py`, distinct from the 4-mixture MFCC GMM-HMM B7 track), achieving **100.00% test normalized WER**, **88.76% test normalized CER**, **0.000 term micro-F1**, and **0.0017 end-to-end RTF**.
  - Verified [`results/m3/README.md`](../results/m3/README.md), [`results/final/m3_test_summary.json`](../results/final/m3_test_summary.json), and the B5/B6 entries in the root [`README.md`](../README.md).
  - **Collaborating members / required reviewers:** `@Rxd3` (Member 5) and `@afel6` (Member 6).

## Experiments and artifacts

All final checkpoints, validation/test predictions, N-best lists, timing runs, and metric manifests are archived under [`results/m3/`](../results/m3/) (matching [`results/final/m3_test_summary.json`](../results/final/m3_test_summary.json)):

| System | Selection / Notes | Test Norm WER | Test Norm CER | Term Micro F1 | End-to-End RTF |
| --- | --- | ---: | ---: | ---: | ---: |
| **B5 BiLSTM-CTC** | Epoch 22 (selected on validation; early stop at Epoch 28) | **47.78%** | **13.50%** | **0.891** | **0.0093** |
| **B6 Single-Gaussian Character HMM** | Classical weak reference (`1` diagonal Gaussian per character state) | **100.00%** | **88.76%** | **0.000** | **0.0017** |

- **B5 Checkpoint & Manifests:** `results/m3/b5/checkpoint/bilstm_ctc.pt`, `vocab.json`, checkpoint manifest, and training history under `results/m3/b5/training/`.
- **B6 HMM Parameters & Logs:** `results/m3/gaussian-hmm/gaussian_hmm_character.npz`, `training-evidence.json`, and `training-history.json`.
- **Reproducibility Scripts:** `scripts/m3_resume_b5_gpu.py`, `scripts/m3_finalize_b5.py`, `scripts/m3_run_gaussian_hmm.py`, and `scripts/m3_summarize_final.py`.

## Reviews and handoffs

- Reviewed and approved [PR #1](https://github.com/Rxd3/TURAS/pull/1) (`chore(env): pinned Windows cu130 environment, env check and test CI`) after verifying `4/4` passing CI checks and clean environment/packaging diffs.
- Addressed M6 reviews on [PR #7](https://github.com/Rxd3/TURAS/pull/7) (`feat/m3-ctc-alternative-baselines`), received M6 approval (`d442603`, `9351849`), and merged PR #7 into `main` (`695016f`).
- Handed off Epoch 13 B5 weights (`b5_epoch13_handoff.zip`) for GPU continuation (`m3_resume_b5_gpu.py`) and verified final consolidated outputs in `results/m3/` and root `README.md`.

## Remaining work

- None — all Member 3 deliverables (B4/B5 adapters & tests, B5 BiLSTM-CTC final checkpoint & test metrics, B6 single-Gaussian character HMM baseline, and contribution documentation) are complete and verified.