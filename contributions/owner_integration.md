# Owner-directed integration record

2026-11-19. This is a cross-workstream owner record, not a seventh team member or
an invented identity. Technical decisions were delegated by the user; source
integration, fixes, tests and profiling were agent-assisted under that instruction.

## Delivered

- Recorded owner authority, exact parents, protocol budgets, score/normalization
  semantics and remaining evidence requirements in `docs/owner_decisions.md`.
- Preserved and merged M1/M6 source commits with M4/M5 on a dedicated integration
  branch. Their source authors retain credit; this does not invent their reviews.
- Used shared normalization, export deduplication, schema checks and timed
  inference interfaces. Fixed RDL component-score/selected-ID propagation.
- Added a composed acoustic/RDL adapter and tested it against the actual shared
  exporter and M1 quantity extractor using clearly synthetic fixtures.
- Added hash-verified private bundle staging and strict CTC target/OOV auditing.
- Verified public frozen split lists and six static relational-table hashes.
- Installed an official hash-verified CUDA wheel in the project environment;
  verified CUDA/bf16 execution and measured five random-model resource probes.
- Saved a reusable runtime lock and measured-resource evidence with artifact hashes.

No private training bundle was received, no test archive was extracted, no test
reference was opened, and no corpus baseline/training/benchmark result is claimed.
Actual test/resource outcomes and source revisions are in
[readiness](../docs/m45_readiness.md) and [GPU evidence](../docs/m45_gpu_profile.md).

## Pending

Private inference/audio and train-reference bundles; isolated validation data or
genuine validation reports; actual baseline and training runs; validation selection,
ablations and final test reporting. Independent code review remains distinct from
the owner's project-decision approval; no draft PR is represented as merged.

## Real-data handoff continuation, 2026-11-19

Received both private bundles from M1. Verified archive/part/payload hashes,
5,450 FLAC formats/durations/hashes, and frozen ID order. Kept test references
closed. Ran the real train-only RDL model for 20 epochs and recorded internal
weak-label relation metrics and artifact hashes in
[the execution record](../docs/m45_execution_2026-11-19.md).

Audited CTC targets: 3,232/4,376 train rows contain unsupported characters.
Added a separately labeled vocabulary-extension protocol and tests preserving
the old IDs/head rows, rather than deleting targets or fabricating readings.
Added the pinned unmodified-parent adapter using the same decoding path as
adapted models, baseline-inference preservation through M6's harness/validators,
and durable optimizer progress. Recorded D32/D33 under the owner's delegated
authority. Validation scoring, checkpoint/fusion selection and final test
comparisons remain pending; independent review is not inferred.

## Final working benchmark completion (2026-11-19)

The later owner-authorized M4/M5 training, recovered-minutes validation, frozen
636-clip test comparison, all fixed-candidate controls, repeated serial timing
and verified private handoff are complete. See the [completion record](../docs/m45_completion.md)
and its final report/audit links. Earlier queue/input/selection-pending notes
above are dated history, superseded by that evidence. Independent reviews,
consumer ACK and M6 dataset approval are not inferred.
