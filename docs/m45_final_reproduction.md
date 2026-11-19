# Reproducing the M4/M5 frozen benchmark

The final protocol is `configs/evaluation/m45_final_protocol.json`, SHA-256
`288aebd71d8a3574e13d436748d4161bc258a487e65066f1c3508bc8e835bec1`.
It was committed at `4639b5e` before final test gold access. Code hashes,
checkpoint hashes, decoder settings, graph versions, validation selection,
reference identity and all comparisons are part of that file. Do not silently
edit it or select new settings from final test results.

Before test-reference access, a provenance audit found that Git's general LF
rule had normalized this JSON's Windows CRLF bytes in the initial commit. The
running protocol itself was unchanged and had the hash above. A path-specific
`-text` attribute now preserves its exact already frozen bytes in Git. This is
a repository serialization correction, not a refreeze or a model/metric change.
All critical Python files and timing/group metadata already matched Git bytes.

## Inputs and environment

The actual machine has an AMD Ryzen 5 7500F (6 cores, 12 threads), 16,765,120,512
bytes of RAM and one NVIDIA RTX 5070. Inference uses four OMP/MKL threads,
CUDA/bfloat16, batch one, requested N=5 and maximum 440 new tokens. Actual
Whisper width is 5 beams; CTC prefix width is 10. The frozen generic
`beam_size=10` field applies to CTC only; see the
[source/export provenance correction](m45_decoding_erratum.md).
Python 3.13.14, torch 2.14.1+cu130 and the installed packages are pinned in
`requirements/owner-m45-windows-cu130.txt`. CPU/GPU, driver, framework, CUDA,
synchronization and thermal samples are recorded by each timing run as well.
This lock describes the actual Windows environment; it is not a Linux lock.

Use a fresh compatible environment, install the lock, then `pip install --no-deps
-e .`. Set `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` and disable tokenizer
parallelism. After obtaining the exact parent weights/processors, final inference
uses `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; network/model downloads are
outside the measurement. Cold model/graph load is separate from warm RTF.

Private inputs must be supplied separately: the M1 inference/audio bundle, the
selected checkpoints and graph bundles, and evaluator-only references. The
gold-free model/run handoff supplies the selected checkpoints, graphs, native
predictions, fixed-candidate outputs, timing records, manifests and source bundle.
It deliberately excludes audio and reference text. Keep these private artifacts
outside Git. Preserve their protocol-relative paths when restoring the handoff.
Whisper LoRA additionally requires its exact pinned parent; it is an adapter,
not a standalone full model. Verify all hashes before consuming artifacts.

## Run levels

Use these commands from the repository root with its Python environment. The
examples are real executable entry points; output directories must be fresh.
Replace `HASH` with the full protocol SHA-256 above.

```text
python scripts/m45_final_infer.py --protocol configs/evaluation/m45_final_protocol.json --protocol-sha256 HASH --system F1-whisper-lora --mode accuracy --output artifacts/reproduction/accuracy/F1-whisper-lora
python scripts/m45_final_rescore.py --protocol configs/evaluation/m45_final_protocol.json --protocol-sha256 HASH --condition F1-whisper-lora--rdl-full --source artifacts/reproduction/accuracy/F1-whisper-lora --output artifacts/reproduction/rescore/F1-whisper-lora--rdl-full
python scripts/m45_final_infer.py --protocol configs/evaluation/m45_final_protocol.json --protocol-sha256 HASH --system F1-whisper-lora--rdl-full --mode timing --output artifacts/reproduction/timing/F1-whisper-lora--rdl-full
```

The other native systems are `B1-whisper-small`, `B4-xlsr-tr-cv7`,
`F2-whisper-full` and `F3-ctc-vocab`. Each adapted system also has
`--acoustic-control`, `--rdl-no-quantities`, `--rdl-no-comention` and
`--rdl-random` conditions. These suffixes are exact artifact identifiers.
`no-comention` means no co-mention relation. Relation-removal graphs were trained
separately; the random graph is explicitly untrained on the same train snapshot.
It is a control, not another claimed trained contribution or course ASR baseline.

Accuracy runs emit all 636 frozen test IDs through the shared export. Candidate
rescoring reuses that model's native N-best and three actual train warm-ups.
Its timing excludes acoustic inference and must never be used as ASR RTF.
The joint timing command performs fresh acoustic inference and relation scoring
through the shared harness. For comparable headline timing, run one system at a
time in the frozen order, with the frozen cooldowns and without other compute
jobs. Use all three repeats and all 107 timing clips; do not cherry-pick a repeat.

The complete orchestrator is `scripts/m45_run_final.py`. Its initial run uses
`artifacts/owner-m45/final/protocol-lock.json`; it refuses silent output reuse,
checks frozen code before each job and opens test references only after all
5 native, 15 cached-control and 8 repeated-timing runs complete. A `STOP` file
under its run root stops child jobs. Do not restart the orchestrator into an
already populated root; preserve completed run records for audit.

To reproduce scores without inference, verify the original inference index and
all its referenced artifacts, supply the separately authorized private reference
directory, and choose a fresh output:

```text
python scripts/m45_final_evaluate.py --protocol configs/evaluation/m45_final_protocol.json --protocol-sha256 HASH --inference-index artifacts/owner-m45/final/inference-index.json --reference-dir data/private/turas-tr-econ-legis-v0.1/evaluator-final --output artifacts/reproduction/evaluation
```

The evaluator validates IDs and hashes, uses the shared normalizer and WER/CER,
audits the five native systems independently with JiWER, computes frozen M1
canonical occurrence metrics, and resamples 28 paired recording groups 2,000
times with seed 2026. Recording groups are not known speakers. Exact quantity
diagnostics cover digit-based amounts and percentages only. All conditions use
the same reference bytes. Test is evaluated after the freeze; re-running a
deterministic score audit is not permission to tune on it.

## Review boundary

For an executable fresh-source restore and delivered-checkpoint check, use
[delivery acceptance](m45_delivery_acceptance.md). That command runs the restored
models on validation audio without gold; it is distinct from reproducing the
complete benchmark in a newly installed environment or recording a peer review.

The private handoff's `artifact-index.json` enumerates every delivered file hash;
`handoff.json` supplies the archive hash. Predictions contain no gold reference
field. Aggregate public reports contain no transcript bodies. Another member
must still independently reproduce/review the results and acknowledge the bundle.
No consumer acknowledgment or independent M6 dataset approval is inferred from
the owner-assisted run. The recovered validation minutes, M1 test minute/pilot
status, unknown speakers and single training seed remain explicit limitations.
