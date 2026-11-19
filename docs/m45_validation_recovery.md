# Owner-directed validation reference recovery

Authority: [D35](owner_decisions.md), following the owner's instruction to make
and document technical decisions instead of waiting for teammate agreement.
This supplies a usable **working reference version**, not a claim of M6 approval.

## Drive audit

The linked Drive root and both subfolders were recursively listed: three folders,
39 files (41 entries including the two subfolders). Their READMEs describe audio/
inference, train-only references and test-only references. No validation-reference
file or full M6 evaluator archive was accessible. The full M6 filename/hash in
the handoff message is an inventory entry, not a delivered download.

## What was recovered

All **438 validation IDs across 15 recordings** were matched to actual upstream
`human_transcript` values at WorldSpeech revision
`b76b85395039f50332fbc421f7c098ee2f2653d7`. The recovery recomputed M1's stable
IDs from source URL, segment ID and original start/end offsets. Existing FLAC
hashes and durations were checked. Membership and audio were unchanged.

The recovery reads identity columns first, then the human-text column only in
shards containing target validation IDs. Only exact validation records are
retained. It never requests the audio or `asr_transcript` columns, runs a model
to make gold, or opens the local test-reference archive. Parquet column chunks
can include unrelated source rows; none are retained as test references or
supplied to a model. No additional model weights or corpus audio were downloaded.

Version: `owner-recovered-validation-minutes-v1`. Private file:
`data/private/turas-tr-econ-legis-v0.1/validation-recovered-minutes-v1/references.validation.jsonl`.
SHA-256: `d0c8519ca6aed053f89786bc733541a70faee09cf8d8bdb6cc12065d97dbd958`.
Its separate `reference-manifest.json` records source, identity/audio hashes,
authority and limitations. Public source/hash/count/metric evidence is in
[the recovered-reference report](../results/evaluation/recovered_validation_baselines.json).

Every row is explicitly `unreviewed`, origin `upstream_human`. This is edited
parliamentary text, not freshly verified verbatim speech. M1 reports three local
validation corrections and 15 listening records; those private records were
not available and no statuses/corrections/reviewer identities were invented.
The recovery does not overwrite M1's frozen artifacts or reproduce their exact
corrected reference bytes. Exact reproduction needs the original reference file
or correction/review records. A new human listening review can instead establish
a separately identified corrected version using the already available audio.

## Working evaluation

The owner-approved recovery removes the missing-reference technical block for
preliminary validation and checkpoint selection. Use this same explicit reference
version for every compared system; never mix its scores with M1-corrected scores.
It is read by the evaluator only, never by acoustic/RDL training or ASR decoding.

The saved unmodified-parent predictions both passed strict M6 joins and shared
raw/normalized evaluation on all 438 clips:

| Baseline | Raw WER | Normalized WER | Raw CER | Normalized CER |
| --- | ---: | ---: | ---: | ---: |
| Pinned Whisper-small | 37.08% | 25.13% | 13.00% | 10.46% |
| Pinned Turkish CV7 XLS-R | 56.26% | 36.98% | 17.07% | 13.25% |

These are **preliminary validation metrics against recovered minutes**, not
human-verified-reference scores, final test results or proof of adaptation gains.
Per-report files preserve raw edit counts, reference denominators, evaluator
source hashes and prediction/reference hashes. The original baseline inference
reports still truthfully record their earlier inference-only scope; scoring is
new evidence in separate files.

Next, score all nine saved acoustic epochs using reference-free inference and
the same evaluator, then select by normalized WER, raw WER, earlier epoch. Store
that choice as working selection against this reference version. RDL fusion,
ablations and final test evaluation remain separate steps; no test execution is
included in this recovery. The separate dataset-review issues remain disclosed.

## Reuse

`python -m turkish_asr_rdl.data.recover_validation --help` documents the recovery
inputs. Use a fresh ignored output folder; source, ID, audio or coverage mismatch
fails the run. No reshuffling or release rebuild is performed.

`python scripts/m45_score_validation.py --help` exposes the shared scorer wrapper.
It requires `--allow-unreviewed-minutes`, verifies the reference version/hash and
frozen IDs, and records the limitation on every score report. This flag records
the already authorized D35 policy; it does not request a new approval vote.

`python scripts/m45_validation_adapted.py --help` runs a saved epoch through the
same N-best decoder and shared timing/export/validators as its parent baseline.
It accepts only gold-free inference inputs, with three train warm-up clips and
one initial validation timing pass. Final RTF comparisons need the shared full
repeat protocol; no recognition score is inferred from timing or training loss.

## Running checkpoint evaluation

After recovery and both baseline score reports passed, the owner-directed local
queue was launched at 14:40 Istanbul on 2026-11-19. It runs all nine saved epochs
sequentially on the GPU, with a separate evaluator process reading the recovered
references only after each prediction export. Source commit:
`d8e37915b8908225e17d3f8facc3cf0ffac1e45b`. The first LoRA epoch is actively
decoding; this is startup evidence, not completed checkpoint selection.

Live state/logs: `artifacts/owner-m45/validation-recovery/validation-queue-status.json`
and the per-epoch logs in that directory. On successful completion the queue
writes `working-checkpoint-selection.json`, identifying the recovered reference
manifest hash, all candidate metric reports and the working selected epoch for
each model. It does not run the test set or freeze RDL fusion. Keep the computer
awake while these validation jobs run. Offline suite: 1,088 passed, 1 skipped,
2 subtests passed; the two new recovery tests check human-text provenance,
unreviewed status, and identity/duration rejection without network access.

## Whisper selection and RDL progress, 15:53 Istanbul

All six Whisper epoch reports are complete. Within the three candidates per
training mode, the working normalized-WER criterion selects LoRA epoch 3
(12.45796%) and full-training epoch 2 (10.19356%). The unchanged Whisper parent
is 25.12518%, on the same recovered validation reference version. CTC epochs 1
and 2 are scored; epoch 3 is still running at this snapshot. The queue's final
working selection artifact remains authoritative after all nine finish.

The predeclared RDL grid (gamma 0, .1, .3, 1; LM disabled) now completed for both
selected Whisper N-best artifacts, using the identical candidates within each
grid. Every exported trial passed the shared validator and its output hash was
rechecked. Both grids select gamma **0**, so the current graph supplies no
measured Whisper recognition gain in this experiment. The original decoder's
one-best also beats the original-log-score/length acoustic-rescoring control:

| Selected Whisper | Original one-best normalized WER | Gamma 0 acoustic control | Gamma .1 | Gamma .3 | Gamma 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA epoch 3 | 12.458% | 12.548% | 12.578% | 12.600% | 12.607% |
| Full epoch 2 | 10.194% | 10.201% | 10.268% | 10.291% | 10.343% |

The difference between original one-best and gamma 0 is acoustic reordering,
not an RDL gain/loss. Compare positive gamma against its gamma-0 control when
attributing the relational component. No statistical significance claim is made.
See [the actual grid summary](../results/evaluation/whisper_rdl_validation_grid.json)
for raw/normalized metrics, source hashes and changed-selection counts.

A detached local continuation waits for successful completion of all acoustic
validation jobs, verifies the CTC selection report, then runs that selected CTC
checkpoint's same RDL grid. Live state is
`artifacts/owner-m45/validation-recovery/ctc-rdl-continuation-status.json`.
It stops on a failed prerequisite and performs no test evaluation. Controlled
relation-removal ablations, final repeat timing and final-test reporting remain
separate unfinished work. These are working recovered-minutes validation results,
not final benchmark or human-verified-reference claims.

## Relation-removal ablation plan (before measuring these variants)

Under the owner's delegated D26/D35 scope, diagnose the initial Whisper RDL
results with two controlled graph variants: remove `rel_expresses` (quantity
relations and hypothesis quantity scoring), or remove `rel_co_mentioned`
(entity co-mentions). Retrain each at seed 17, 20 epochs, the same table hashes,
hidden size, layers, dropout, learning rate and training-group holdout. Use the
same selected acoustic N-best candidates and gamma grid 0/.1/.3/1 for each.

This is a diagnostic plan chosen after the primary graph's validation results,
not a claim that these specific variants were preregistered before those results.
Keep all trial outcomes and identify single-seed/reference-quality limitations.
Do not tune on test. Local ablation status will be in
`artifacts/owner-m45/validation-recovery/relation-ablations/status.json`.

**Completion update:** all nine acoustic evaluations, three primary fusion grids,
and six relation-removal grids completed. See the
[working validation completion record](m45_validation_complete.md) for checkpoint
choices, actual outcomes, private artifact paths and unfinished final benchmarking.
