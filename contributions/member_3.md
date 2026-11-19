# Member 3 — CTC and Classical Baselines

Name: Abdullah Gamal GitHub username: abdu3061
Branch convention: `feat/m3-ctc-alternative-baselines` (focused topic suffixes allowed).
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

**Status: baseline checkpoint audit and protocol configs proposed; awaiting M5/M6 review and M1 data freeze before implementation/training.**

## Actual contributions

* **2026-11-19 — B4/B5 Checkpoint Audit & Protocol Configs (`feat/m3-ctc-alternative-baselines`):**
  * Audited and pinned `mpoyraz/wav2vec2-xls-r-300m-cv8-turkish` (`bdd0bb878d8bf59509611631e334eb073ba57773`) and `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` (`708639f50559d7970f462e13ec64d3f059ca89f6`) in `configs/baselines/wav2vec2_xlsr_turkish.json` (`float32`, greedy CTC, explicit `Wav2Vec2CTCTokenizer` + `Wav2Vec2FeatureExtractor` loading, zero WorldSpeech overlap).
  * Documented the B4 vocabulary limitation (missing digits `0–9`) and planned `-CTCLoss(reduction='none')` prefix beam search extension in `experiments/m3-baseline-notes.md`.
  * Proposed `configs/baselines/bilstm_ctc.json` for B5 (`BiLSTM-CTC` trained from scratch with train-only vocabulary and a mandatory 3-step memory + 1-epoch timing probe before locking model dimensions).
  * Left `dataset_release` and `split_manifest_sha256` as `null` pending M1's dataset re-freeze.
  * **Collaborating members / required reviewers:** Member 5 (CTC compatibility) and Member 6 (contracts/evaluation).
  * **Pending / limitations:** No model weights downloaded, no splits bound, and no runs executed yet.

## Experiments and artifacts

No runs. Weight hashes (`weight_sha256`, `processor_sha256`) and split manifest hashes remain `null` until M5/M6 config approval and M1 dataset freeze.

## Reviews and handoffs

* Reviewed and approved `PR #1` (`chore(env): pinned Windows cu130 environment, env check and test CI`) after verifying `4/4` passing CI checks and clean environment/packaging diffs.
* Requested review from Member 5 and Member 6 on `feat/m3-ctc-alternative-baselines` before starting B4/B5 adapter implementation and weight downloads.