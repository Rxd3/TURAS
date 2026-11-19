# M4/M5 working validation completion

2026-11-19. Owner-directed seed-17 experiments; all current training, acoustic
validation, primary RDL grids and two relation-removal ablation queues completed.
This does not assert a completed final benchmark or independent teammate review.

**Audit update:** acoustic WER/CER passed an independent JiWER cross-check.
The audit found that the legacy RDL linker discarded frozen alias wildcards.
The RDL tables below describe that historical linker version; owner decision
D37 requires a versioned prefix/exclusion-aware linker and fresh RDL grids before
final RDL conclusions. Acoustic checkpoints and their reported scores are unaffected.

All figures below use the same 438 frozen validation clips and
`owner-recovered-validation-minutes-v1` reference version: upstream edited
parliamentary text, marked unreviewed, without M1's private corrections. See
[recovery and authority](m45_validation_recovery.md) for provenance and limits.
No test predictions or test metrics have been produced.

## Acoustic results and working checkpoint choices

| System | Selected epoch | Normalized WER | Normalized CER |
| --- | ---: | ---: | ---: |
| Unchanged Whisper-small | — | 25.125% | 10.463% |
| Whisper-small LoRA | 3 | 12.458% | 3.971% |
| Whisper-small full fine-tuning | 2 | 10.194% | 3.447% |
| Unchanged Turkish CV7 XLS-R | — | 36.978% | 13.255% |
| XLS-R fine-tuning **plus vocabulary extension** | 3 | 25.402% | 7.637% |

All three epochs were evaluated for each adapted model. Selection used lowest
normalized WER, then raw WER, then earlier epoch. The CTC contribution includes
the documented 38-to-52-token output extension; its gain cannot be attributed to
fine-tuning alone. All comparisons are single-seed working validation results.

The [complete acoustic selection inventory](../results/evaluation/m45_working_checkpoint_selection.json)
contains all candidate scores/hashes, selected checkpoint manifests, reference
identity, and selected raw/normalized WER/CER edit counts. Both baselines' scores
are in [the recovered-reference baseline report](../results/evaluation/recovered_validation_baselines.json).

## RDL and controlled ablations

Primary graph: the original 20-epoch seed-17 heterogeneous model, with the fixed
training-only group holdout. Its learned internal relation metrics remain
separate from ASR accuracy. Fusion searched gamma 0/.1/.3/1 on identical candidates
per system, alpha=1, beta=0 and length exponent=1. Every trial used the shared
validators and evaluator. No final-test fusion configuration is frozen yet.

| Acoustic system | Original one-best WER | Gamma-0 acoustic reordering | Primary graph best WER | Primary gamma |
| --- | ---: | ---: | ---: | ---: |
| LoRA epoch 3 | 12.458% | 12.548% | 12.548% | 0 |
| Full Whisper epoch 2 | 10.194% | 10.201% | 10.201% | 0 |
| Extended CTC epoch 3 | 25.402% | 25.461% | 25.402% | .1 |

Thus primary RDL does not improve original one-best WER: Whisper selects zero
relational weight, and CTC recovers the loss introduced by acoustic reordering
but ties the original decoder's normalized word error count. This is a measured
negative/neutral downstream outcome, not a claimed RDL recognition gain.

Two diagnostic variants were retrained with the same seed/budget/hashes/groups:
remove `rel_expresses` (quantity relations/scoring), or remove
`rel_co_mentioned` (entity co-mentions). Each ran the same four-weight grid on
all three selected acoustic candidate files: six grids / 24 additional trials.
Neither variant beats original Whisper one-best. Both CTC variants reach
25.394% normalized WER at gamma .1, one word error fewer than original CTC
over 13,381 reference words. Significance is not established. These diagnostic
variants were chosen after primary validation results and are reported as such;
they do not silently replace the predeclared primary graph on this tiny change.

See [the actual RDL/ablation summary](../results/evaluation/m45_rdl_and_ablation_summary.json)
and [the primary Whisper grid report](../results/evaluation/whisper_rdl_validation_grid.json)
for every grid outcome, changed selections and provenance. All 36 trial N-best
outputs passed shared validation; their hashes were checked. Offline rescoring
was untimed and supplies **no combined end-to-end ASR RTF**.

## Private artifacts ready for handoff

The following are existing local folders under ignored `artifacts/owner-m45/`:

- `whisper-lora-seed17/epoch-3`: adapter and processor; exact base pin is in its manifest.
- `whisper-full-seed17/epoch-2`: full weights and processor.
- `ctc-extended-seed17/epoch-3`: weights, extended tokenizer/processor and extension record.
- `rdl-seed17`: primary learned graph, snapshot, linker and hashes.
- `validation-recovery/`: all nine prediction/N-best/timing runs, score reports,
  primary fusion grids and `relation-ablations/` with both trained graph variants.

Keep model/data/prediction artifacts private; only summary metadata is in Git.
The public inventories provide checksums, not proof that M6 has received these
local files. Do not invent consumer acknowledgment or independent review.

## Remaining experimental and delivery work

1. Domain terminology/quantity metrics through the shared evaluation contract.
2. Final repeated timing and end-to-end RDL cost with the shared harness. Initial
   one-pass validation timing is not the final three-repeat benchmark.
3. A recorded final protocol freeze identifying reference versions, checkpoints,
   original-decoder/acoustic-control/RDL systems, and all baseline comparisons.
   Dataset-review limitations and the broader M2/M3 baseline coverage remain explicit.
4. Final test evaluation once after that freeze, final reports/presentation and
   hashed artifact handoff/reproduction. Test gold is still closed.
5. Planned additional training seeds 42/2026 remain unexecuted; no multi-seed
   average, uncertainty estimate or robustness claim has been fabricated.

No additional permission vote is required for the owner's covered technical
choices. These remaining items are actual work/evidence, not approvals invented
as blockers. The current queues are idle; a subsequent benchmark run will need
the computer awake again.
