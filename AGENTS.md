# Rules for coding agents and teammates

Six rules for every TURAS change. Each links the contract that governs it; where
this page is shorter than a contract, the contract wins. A tool marked
"(on `main` after PR-X)" exists only on its branch until that pull request
merges (list at the end).

1. **Never write your own normalizer or WER/CER.** Import them from
   `turkish_asr_rdl.evaluation` (on `main` after PR-C). Rules:
   [normalization](docs/contracts/normalization.md) and
   [evaluation](docs/evaluation.md#wercer-and-aggregation).
2. **Never re-split data.** Read only
   `data/splits/<release>/{train,validation,test}.txt` from Member 1's frozen
   release ([frozen split files](docs/contracts/data.md#frozen-split-files)).
3. **Predictions and N-best are separate JSONL files.** Predictions follow
   [predictions](docs/contracts/predictions.md), in split-file order, and never
   carry reference text. N-best is its own artifact per
   [nbest](docs/contracts/nbest.md). Save JSONL as UTF-8 without BOM.
4. **Never time by hand.** Implement `AsrAdapter` and run inference through
   `run_timed`, both from `turkish_asr_rdl.decoding` (on `main` after PR-B).
   `run_timed` reads audio with the shared 16 kHz mono loader inside the timed
   region ([timing](docs/evaluation.md#real-time-factor-and-latency)).
5. **Run `turas-check` on every output before handoff** (on `main` after PR-E):
   `turas-check predictions FILE --ids SPLIT_FILE` and
   `turas-check nbest FILE --ids SPLIT_FILE`. Until then, call
   `validate_predictions` and `validate_nbest` from `turkish_asr_rdl.schemas`
   (on `main` after PR-B). Both enforce the
   [contracts](docs/contracts/README.md).
6. **Known traps.**
   - Repeated candidate texts are normal in beam N-best. Return the decoder's
     raw list, duplicates included: the shared export deduplicates it
     automatically and records `dedup_map` (on `main` after PR-B;
     [N-best contract](docs/contracts/nbest.md#who-deduplicates)). Candidates
     with the same text and identical `token_ids` are the Whisper trap: in
     transformers 5.23.0, `WhisperForConditionalGeneration.generate` with
     `num_return_sequences>1` returns copies of the 1-best, and the export turns
     that result into an error row.
   - WorldSpeech `asr_transcript` is never a reference
     ([dataset](docs/dataset.md)).
   - Python `lower()` and `casefold()` are wrong for Turkish (`I` becomes `i`,
     not `ı`); use the normalizer from rule 1.

Environment: install the pinned environment and run `python scripts/check_env.py`
(on `main` after PR-A); see [setup](docs/reproducibility.md#environment-setup-windows).

Hardware: VRAM, speed, training-memory and N-best findings from the 2026-11-17
smoke run, with next steps for Members 2–5, are in
[smoke results](docs/m6-smoke-results.md).

Pull requests: PR-A `feat/m6-env` (environment and env check); PR-B
`feat/m6-harness` (audio loader, `AsrAdapter`, `run_timed`, validators); PR-C
`feat/m6-evaluator` (normalizer, aligner, metrics, reference join); PR-E
`feat/m6-cli` (`turas-check`).
