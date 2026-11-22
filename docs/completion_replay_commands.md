# Follow-up baseline replay

These commands rerun the frozen follow-up configuration into a new output
directory. They never overwrite the delivered predictions. Use the source ZIP
and model/input roles from the same version of the delivery manifest.

## Restore and install

Extract `TURAS-submission-source.zip`, then restore `inference` plus the model
roles you received: `b2`, `b3`, `transducer`, or `gmm`. The standard restore helper
and exact audio destination are documented in [final reproduction](final_reproduction.md).
Inference needs no validation/test references. Restore `evaluator` separately
only for offline scoring.

From the extracted source root, create a new Python 3.13 environment:

```powershell
python -m venv .replay-neural
.replay-neural\Scripts\python.exe -m pip install -r requirements/windows-cu130.txt
.replay-neural\Scripts\python.exe -m pip install --no-deps -e .
.replay-neural\Scripts\python.exe scripts/check_env.py
```

The CPU transducer uses its own small environment:

```powershell
python -m venv .replay-transducer
.replay-transducer\Scripts\python.exe -m pip install -r requirements/transducer-windows.txt
.replay-transducer\Scripts\python.exe -m pip install --no-deps -e .
```

B2/Qwen packages contain verified upstream inventories, processor files and
outputs. Download the exact large weights separately. This avoids duplicating
several GB in Drive. The downloader verifies full pinned weight SHA-256 values:

```powershell
.replay-neural\Scripts\python.exe scripts/download_completion_models.py --models b2 qwen17 qwen06
```

Seda's ONNX acoustic weights and the trained four-mixture HMM are included in
their respective packages. Seda's optional external language model is excluded.

## Execute

Use one headline timing job at a time. Set one CPU worker thread for the CPU
models; neural adapters set four PyTorch threads explicitly.

```powershell
$env:PYTHONUTF8='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.replay-transducer\Scripts\python.exe scripts/replay_completion_baseline.py --system seda --mode validation --validation-clips 3 --output artifacts/replay-seda-smoke
.replay-neural\Scripts\python.exe scripts/replay_completion_baseline.py --system gmm --mode validation --validation-clips 3 --output artifacts/replay-gmm-smoke
.replay-neural\Scripts\python.exe scripts/replay_completion_baseline.py --system b2 --mode test --output artifacts/replay-b2-test
```

For B3, read `artifacts/completion-2026-11-22/b3-selection.json` and use its
`selected_system` (`qwen17` or `qwen06`). Both full validation results are retained;
only the validation winner has test results. Substitute that key in `--system`.
Other valid systems are `seda` and `gmm`. Full validation uses `--mode validation`
without `--validation-clips`. Repeated timing uses `--mode timing`: the frozen
107 test IDs, three repeats, batch one and three train warm-ups.

The replay checks the original model/decoder identity, runs shared audio loading
and `run_timed`, validates the exported predictions/N-best, and reports exact
transcript agreement with the archive. A validation smoke check proves execution
of that subset; it does not establish full-test parity. GPU/library changes may
change floating-point decoding. Keep the recorded environment and any mismatch
report rather than claiming exact reproducibility without comparing outputs.

## Recalculate scores

For an authorized evaluator, use `scripts/reproduce_saved_scores.py` with the
replayed or delivered `predictions.jsonl`, the frozen test IDs/manifest, and
`data/private/turas-tr-econ-legis-v0.1/evaluator-final/references.test.jsonl`.
The [full guide](final_reproduction.md#3-recalculate-b5-or-hmm-scores) gives the
complete command. Replace its prediction/output paths with the chosen follow-up
system. This uses shared joining, Turkish normalization and edit-count metrics.

The saved aggregate checks require no ML dependencies:

```powershell
python scripts/verify_submission.py
```

It checks the original 20-condition study. The follow-up results are recorded in
`results/final/baseline_completion_summary.json`. `scripts/verify_completion_submission.py`
needs `results/final/completion_test_summary.json`, which is not in the repository;
without it the script stops with a "pending" message and exit code 1. Neither
substitutes for independently downloading and executing the actual artifacts.

## Retrain the new GMM-HMM

Use a separate fresh source workspace and restore only `inference,train` there.
Do not restore the `gmm` role into that workspace: the training command checks
and reuses an existing completed checkpoint rather than overwriting it.
Install the pinned neural environment above, then run:

```powershell
.replay-neural\Scripts\python.exe scripts/train_mixture_hmm.py
```

The trainer reads the frozen 4,376 train IDs and train references only. It
freezes the MFCC recipe, seed 42, three states per grapheme, four diagonal
Gaussian components and five Viterbi-EM passes. It records audio/hash checks,
training coverage, initializer convergence and training history under
`artifacts/completion-2026-11-22/gmm/`. The final model is `checkpoint.npz` with
its SHA-256 in `model-manifest.json`. Feature caches stay local and are excluded
from delivery archives.

Compare a retrained checkpoint and its decoded outputs with the delivered
manifest/results. Record differences across numerical runtimes or hardware;
do not infer byte-identical retraining from a successful inference smoke check.
The earlier supplied single-Gaussian HMM remains a separate preserved experiment.
