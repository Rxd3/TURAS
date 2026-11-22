# Rules for coding agents and teammates

Six rules for every TURAS change. Each links the contract that governs it; where
this page is shorter than a contract, the contract wins. Every tool named below
is on `main` unless a pull request is named beside it.

1. **Never write your own normalizer or WER/CER.** Import `normalize` from
   `turkish_asr_rdl.evaluation.normalize`, and `score_pair`/`corpus_rates` from
   `turkish_asr_rdl.evaluation.metrics`. Rules:
   [normalization](docs/contracts/normalization.md) and
   [evaluation](docs/evaluation.md#wercer-and-aggregation).
2. **Never re-split data.** Read only
   `data/splits/<release>/{train,validation,test}.txt` from Member 1's frozen
   release ([frozen split files](docs/contracts/data.md#frozen-split-files)).
3. **Predictions and N-best are separate JSONL files.** Predictions follow
   [predictions](docs/contracts/predictions.md), in split-file order, and never
   carry reference text. N-best is its own artifact per
   [nbest](docs/contracts/nbest.md). Save JSONL as UTF-8 without BOM.
4. **Never time by hand.** Implement `AsrAdapter`
   (`turkish_asr_rdl.decoding.interface`) and run inference through `run_timed`
   (`turkish_asr_rdl.decoding.timing`).
   `run_timed` reads audio with the shared 16 kHz mono loader inside the timed
   region ([timing](docs/evaluation.md#real-time-factor-and-latency)).
5. **Validate every output before handoff** with `validate_predictions`
   (`turkish_asr_rdl.schemas.predictions`) and `validate_nbest`
   (`turkish_asr_rdl.schemas.nbest`). Both enforce the
   [contracts](docs/contracts/README.md).
6. **Known traps.**
   - Repeated candidate texts are normal in beam N-best. Return the decoder's
     raw list, duplicates included: the shared export deduplicates it
     automatically and records `dedup_map`
     ([N-best contract](docs/contracts/nbest.md#who-deduplicates)). Candidates
     with the same text and identical `token_ids` are the Whisper trap: in
     transformers 5.23.0, `WhisperForConditionalGeneration.generate` with
     `num_return_sequences>1` returns copies of the 1-best, and the export turns
     that result into an error row.
   - WorldSpeech `asr_transcript` is never a reference
     ([dataset](docs/dataset.md)).
   - Python `lower()` and `casefold()` are wrong for Turkish (`I` becomes `i`,
     not `ı`); use the normalizer from rule 1.

Environment: install the pinned environment and run `python scripts/check_env.py`;
see [setup](docs/reproducibility.md#environment-setup-windows).

Hardware: VRAM, speed, training-memory and N-best findings from the 2026-11-17
smoke run, with next steps for Members 2–5, are in
[smoke results](docs/m6-smoke-results.md).

## Grader path

1. Verify (stdlib only): `python scripts/verify_submission.py`
2. Install: `python -m venv .venv`, activate, `python -m pip install --no-deps -e .`,
   `python -m pip install -r requirements/dev.txt`
3. Tests: `python -m pytest -q` (tests needing absent optional dependencies skip)
4. GPU machine only: `pip install -r requirements/windows-cu130.txt`, then
   `python scripts/check_env.py` prints `READY`
5. Results: `results/final/M4_M5_FINAL_REPORT.md`, `m45_metrics.csv`,
   `m45_test_summary.json` (20 systems); audits in `results/evaluation/`
6. B2 and B3, outside the frozen protocol: [`results/m6/README.md`](results/m6/README.md),
   scored with `scripts/m6_score_baselines.py` (PR #23)
7. Full rerun needs private bundles: `docs/m45_final_reproduction.md`

Script map (`scripts/`):

- `verify_submission.py`: protocol, split and code-hash checks; WER/CER from edit counts
- `m45_run_final.py`: frozen benchmark run and handoff; `m45_final_infer.py`: gold-free inference and timing
- `m45_final_evaluate.py`: shared scoring; `m45_final_rescore.py`: cached RDL selections; `m45_final_audit.py`: post-freeze audit
- `m6_score_baselines.py`: B2/B3 scoring with the shared scoring calls; `m6_b23_harness_run.py`: B2/B3 timing (both PR #23)
- `m4_train_whisper.py`, `m5_train_ctc.py`, `m5_train_rdl.py`, `m5_rescore.py`: training and rescoring wrappers
- `run_baseline_batch.py`: baseline batch inference; `compute_nbest_oracle.py`: N-best oracle; `m1_verification.py`: reference review
