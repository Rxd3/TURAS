# M6 reproduction tools

M6 tooling for the post-freeze reproductions in
[`results/m6/repro-2026-11-23/`](../../results/m6/repro-2026-11-23/README.md). It is outside the
frozen protocol and not pinned by it.

- `f2repro.ps1` (Windows PowerShell 5.1 or 7) drives one GPU job at a time on a Windows laptop with a CUDA
  GPU: steps `preflight`, `baseline`, `smoke`, `start`, `chain`, `status` and `compare`. `-Jobs` takes
  `SYSTEM:SEED` items (F1, F2, F3); a finished stage is skipped and an unfinished one is moved aside and redone.
- `f2repro.py` does the checks, configs, derived test protocols, epoch selection, status and comparison
  for the PowerShell steps. It prints aggregates only, never reference or hypothesis text.
- `f2repro_selfcheck.py` checks selection, comparison and status on the committed M6 summaries:
  `python tools/m6-repro/f2repro_selfcheck.py tools/m6-repro` from the repository root.

Inputs, under `-Artifacts <ARTIFACT_ROOT>`: a clone of `main` at `ee1f3e4` in `verify-2026-11-22\main`,
the published data in `verify-2026-11-22\pr26\data\cleaned\turas-tr-econ-legis-v0.1`, a lean Python 3.13
environment in `verify-2026-11-22\venv-lean-3.13`, and in `m6-repro-f2\` these two scripts plus
`gpu-python.txt` (the path of the Python with CUDA PyTorch). Outputs go to `m6-repro-f2\`: one folder per job
(`seed<N>` for F2, `f1-seed<N>`, `f3-seed<N>`) with training, per-epoch validation scores, `selection.json`,
the test decode, test scores and `oracle.json`, plus `status.json`, `pipeline.log`, `logs\` and `compare.json`.
