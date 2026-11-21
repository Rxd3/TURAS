# Member 3 — CTC and Classical Baselines

Name: Abdullah Gamal student id:2309015819 GitHub username: abdu3061
Branch convention: `feat/m3-ctc-alternative-baselines` (focused topic suffixes allowed).
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

**Status: PR #7 (`695016f`) merged to `main`; B4 (`Wav2Vec2`/`XLS-R` CTC) and B5 (`BiLSTM-CTC`) `AsrAdapter` implementations and offline unit tests implemented on `feat/m3-ctc-adapters-tests` (`1156 passed, 20 skipped, 0 failed`).**

## Actual contributions

- **2026-11-21 — B4/B5 Checkpoint Audit & Protocol Configs (PR #7, merged as `695016f`):**
  - Audited and pinned `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` (`708639f50559d7970f462e13ec64d3f059ca89f6`, `cc-by-4.0`, `pytorch_model.bin` `1,262,079,473` bytes, Hub LFS oid `008d69c82bf45c82ec5485e550abc86ddabefe75080f22e1ebfdc1699889d82e`) as primary B4 (`B4-xlsr-tr-cv7`) to match M5's F3 parent control, keeping `mpoyraz/wav2vec2-xls-r-300m-cv8-turkish` (`bdd0bb878d8bf59509611631e334eb073ba57773`, `apache-2.0`, Hub LFS oid `dbd7a9f776a232c0dd4cbbee44025c8056246d83ad4de930b049e649e2fe402c`) as a labeled secondary candidate in `configs/baselines/wav2vec2_xlsr_turkish.json`.
  - Aligned B4 primary decoding with M5's frozen F3 protocol (`bfloat16`, CTC prefix beam search with `beam_size=10`, `requested_n=5`, rank-1 one-best, adopting `turkish_asr_rdl.decoding.ctc` over a separate NumPy scorer, with `float32` greedy retained only as a labeled secondary diagnostic) and documented the 38-token vocabulary (`blank=pad=0`, `unk=3`, `|=4`, missing digits/diacritics, no post-processing, KenLM excluded).
  - Proposed `configs/baselines/bilstm_ctc.json` for B5 (`BiLSTM-CTC` trained from scratch, `tr-normalized-v1` targets/validation reference, subsampling factor `4`, `zero_infinity=True` + minimum-length feasibility check, `ReduceLROnPlateau` on `validation_norm_cer`, `held_out_exposure: "none"`, and a mandatory 3-step memory + 1-epoch probe before locking numeric compute ceilings).
  - Documented per-checkpoint `2022-03-23` commit-date held-out eligibility and limitations in `experiments/m3-baseline-notes.md`, keeping `dataset_release`, `split_manifest_sha256`, and tokenizer/processor hashes `null` pending M1's dataset re-freeze verification.
  - **Collaborating members / required reviewers:** `@afel6` (Member 6) and `@Rxd3` (Member 5).

- **2026-11-21 — B4/B5 Python Adapters & Offline Unit Tests (`feat/m3-ctc-adapters-tests`):**
  - Implemented `Wav2Vec2CtcBaselineAdapter` in `src/turkish_asr_rdl/baselines/ctc.py`, enforcing explicit `Wav2Vec2CTCTokenizer` + `Wav2Vec2FeatureExtractor` loading (ignoring upstream KenLM / `Wav2Vec2ProcessorWithLM`), `float32` log-softmax conversion before calling `turkish_asr_rdl.decoding.ctc.ctc_nbest`, raw whitespace preservation (`raw-v1`), per-utterance `CTCLoss(reduction="none", zero_infinity=False)` cross-checking, and `AsrAdapter` contract compliance.
  - Implemented `BiLSTMCTC`, `CharCtcTokenizer`, `LogMelSpectrogramExtractor`, `check_ctc_feasibility`, and `run_3step_memory_probe` in `src/turkish_asr_rdl/models/bilstm_ctc.py`, plus `BiLstmCtcBaselineAdapter` and `save_bilstm_checkpoint` in `src/turkish_asr_rdl/baselines/bilstm_ctc.py` (strictly train-only character inventory, `4x` stride-2 Conv2D subsampling, `zero_infinity=True` feasibility filtering, and `AsrAdapter` compliance).
  - Added offline synthetic unit tests in `tests/test_m3_ctc_baselines.py` verifying `check_adapter` conformance, raw edge-whitespace preservation, batch error/empty isolation, train-only `<unk>` mapping, `4x` subsampling length invariance, and B5 3-step probe + checkpoint round-trip (`python -m pytest -q`: `1156 passed, 20 skipped, 0 failed`).
  - **Collaborating members / required reviewers:** `@afel6` (Member 6) and `@Rxd3` (Member 5).

## Experiments and artifacts

No model training or dataset inference runs executed yet. Hub LFS oids and byte sizes are recorded from Hub metadata (bytes to be verified at download); tokenizer/processor file hashes and split manifest hashes remain `null` until M1 dataset freeze confirmation.

## Reviews and handoffs

- Reviewed and approved [PR #1](https://github.com/Rxd3/TURAS/pull/1) (`chore(env): pinned Windows cu130 environment, env check and test CI`) after verifying `4/4` passing CI checks and clean environment/packaging diffs.
- Addressed M6 reviews on [PR #7](https://github.com/Rxd3/TURAS/pull/7) (`feat/m3-ctc-alternative-baselines`), received M6 approval (`d442603`, `9351849`), and merged PR #7 into `main` (`695016f`).

## Remaining work

- Acknowledge M1's re-frozen dataset release hash once verified, download pinned B4 weights, and populate `dataset_release`, `split_manifest_sha256`, and tokenizer/processor SHA-256 fields in `configs/baselines/wav2vec2_xlsr_turkish.json` and `configs/baselines/bilstm_ctc.json`.
- Execute B5's 3-step memory probe and 1-epoch timing run on allocated training hardware to lock model dimensions and numeric compute ceilings, then train B5 on `train` and select on `validation`.
- Export complete B4 and B5 prediction/N-best bundles for M6 evaluation and RTX 5070 Ti timing.