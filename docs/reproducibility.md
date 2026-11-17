# Reproducibility plan

Only scaffold packaging can be checked now. No ASR experiment is reproducible
from this repository yet because implementations, approved data, checkpoints,
dependency locks and results do not exist. CI checks packaging only.

## Scaffold inspection

Use Python 3.11+ to parse `pyproject.toml` with `tomllib`; use `PYTHONPATH=src`
or install the package to import the empty package/subpackages. Check whitespace
with `git diff --check`. These checks require no data/model/ML dependencies and
do not demonstrate inference or training. Initialization-specific checks and
remote state are recorded in [repository setup](repository_setup.md).

## Future evidence levels

1. **Recalculate metrics:** obtain hashed predictions and evaluator-only references,
   verify release/ID alignment, run the pinned evaluator/normalizer and compare
   edit counts, aggregate reports and figure inputs.
2. **Rerun inference:** add the exact model/processor/tokenizer, decoder, graph/LM
   artifacts, compatible locked environment and audio; verify candidate scores,
   selected hypotheses, coverage, timing scope and artifact hashes.
3. **Retrain:** add train/validation data, training configuration, initialization,
   seeds, sampler ordering, hardware/precision and validation selection evidence.
   Report stochastic tolerances; byte-identical GPU training is not assumed.

All levels require the [artifact/experiment contract](contracts/artifacts.md).
Keep separate framework environment locks where compatibility demands it; record
exact dependency versions and hashes at the implementation PR, not guessed pins
in this scaffold. Do not install heavyweight dependencies for initialization.

## Release audit checklist

- [ ] M1/M6 approve source terms, references, human verification and frozen split hashes.
- [ ] Every model bundle identifies upstream/base/adapted provenance and compatibility.
- [ ] Inference inputs and permissions exclude references and held-out annotations.
- [ ] RDL training adjacency/features exclude held-out edges and reverse duplicates.
- [ ] Selection is frozen on validation before final test evaluation.
- [ ] All five baselines and actual fine-tuned/RDL conditions use the common evaluator.
- [ ] Every claimed metric has edit/support counts and real artifact hashes.
- [ ] Timing identifies hardware, warm-up, batch and end-to-end scope.
- [ ] Ablations preserve shared candidate sets and matched acoustic baselines.
- [ ] A non-author reproduces the appropriate level in a clean environment.
- [ ] Negative findings, unsupported tracks and resource limits are explicit.
- [ ] Presentation figures and contribution records point to reviewed evidence.

None of these application-release items is complete at initialization.
