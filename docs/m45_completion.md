# M4/M5 completion and private handoff

Owner-assisted experimental work completed on 2026-11-19. This is a **working
benchmark**, not independent M6 dataset approval or completion of every course
baseline. No individual member's human identity, authorship or review is inferred.

## Completed evidence

Whisper LoRA, full Whisper and vocabulary-extended XLSR CTC each completed three
training epochs at seed 17. All nine acoustic epochs were evaluated on the
explicit `owner-recovered-validation-minutes-v1` variant. Selected checkpoints
are LoRA epoch 3, full Whisper epoch 2 and CTC epoch 3. The heterogeneous relation
model and two relation-removal variants were trained on train-only tables;
the random-parameter graph is a clearly labeled **untrained control**.

The final protocol was frozen before test transcript access. Five native
systems and fifteen fixed-candidate selections cover all 636 test IDs. Eight
serial timing conditions each use 107 clips (1,813.765 seconds), three timed
repeats and three excluded train warm-ups. Main results:

| Contribution | Unmodified parent normalized WER | Adapted normalized WER | Adapted normalized CER |
| --- | ---: | ---: | ---: |
| Whisper LoRA | 24.512% | 12.368% | 4.149% |
| Full Whisper | 24.512% | 10.371% | 3.691% |
| CTC with train-derived 38-to-52-token vocabulary extension | 37.152% | 25.150% | 7.582% |

The three parent/adapted WER differences have paired recording-bootstrap
intervals below zero. They remain conditional on this split, reference policy
and single training seed. Full tuning also improves on LoRA in the frozen test
comparison. No checkpoint or fusion weight was chosen using test scores.

**Primary learned RDL does not beat original decoder one-best on any adapted
model.** Its normalized WER is 12.373% / 10.429% / 25.189% for LoRA/full/CTC.
Small improvements from some diagnostic controls do not establish a benefit of
the learned full graph and are not used to select a new primary system.
The no-quantity ablation removes `rel_expresses` relations; it is not a claim
that every numerical feature or lexical quantity cue has been removed.

The [full report](../results/final/M4_M5_FINAL_REPORT.md),
[metrics CSV](../results/final/m45_metrics.csv) and
[machine-readable summary](../results/final/m45_test_summary.json) contain both
raw/normalized WER and CER with edit counts, exact canonical term supports/P/R/F1,
automatic quantity diagnostics, relation/N-best coverage, verified-subset scores,
RTF/latency/memory and all paired intervals. Timing includes scored N-best
generation with requested N=5 and beam 10. It is not a greedy or streaming speed
benchmark. CTC-parent repeat RTF varies from 0.1123 to 0.1503; retain that
dispersion and do not claim a CTC speed advantage from the means alone.

## Integrity and reference boundary

The [supplementary audit](../results/final/m45_final_integrity_audit.json) verifies
all 28 inference/control/timing runs, all fifteen candidate-and-acoustic-score
preservation checks, all three repeats per timing condition and every indexed
file's stored archive bytes. There are zero inference/timing errors, zero
first-repeat transcript differences from full-test outputs, no detected VRAM
spill and no recorded thermal flags. The main five systems independently match
JiWER for both raw/normalized WER and CER. The offline suite passed 1,093 tests,
with one skip and two passing subtests.

Train/test intersections are **zero** for utterance IDs, recording groups,
exact audio byte hashes and exact sentences (literal, raw word sequence and
Turkish-normalized views). These exact checks do not establish absence of
near-duplicate audio or shared speakers. Speakers are unknown; recordings are
the documented fallback grouping unit.

Test references were first opened after every frozen inference/timing artifact
completed. Exact test reference SHA-256:
`31e372f0e35f504de59b49f1e0e1f3ed6fcc2772138c591a2cf2b054a4e7ef3f`.
There are **611 unreviewed and 25 M1-reported verified test references**; ten
references have local-human origin and 626 upstream-human origin. The labels
are edited parliamentary minutes with pilot corrections, not a fully verified
verbatim release. The recovered validation variant omits three private M1
corrections. Independent M6 release-review disagreement remains disclosed.
No externally WorldSpeech-fine-tuned parent was selected; our own adaptation uses
only the frozen TURAS train split. Broad pretrained-model exposure to source
recordings cannot be ruled out merely by this local split audit.

## Deliverable and reproduction

Private archive, intentionally outside Git:

`artifacts/owner-m45/final/handoff/TURAS-M4-M5-private-handoff.tar`

Size: **2,424,330,240 bytes**. SHA-256:
`3bbaf65cd8ac45b4dceab6e1401748392e0bf3dd0039d395d1e092719740a91d`.

It contains the actual three selected checkpoints/processors, three corrected
learned/ablated graphs, the untrained graph control, prediction/N-best/component
scores, timing/evaluation artifacts, frozen protocol and exact runtime source
Git bundle. It excludes reference transcripts and audio. The bundle is ready
for owner delivery; no recipient acknowledgment is fabricated. Public provenance
is in the [handoff manifest](../results/final/m45_handoff_manifest.json).

Use the [reproduction commands](m45_final_reproduction.md) and the pinned Windows
runtime. The Git bundle captures runtime source; the integration branch also
contains the final reports and subsequent completion records. Restore the exact
private paths or make a separately recorded relocation; never edit the frozen
protocol silently. The protocol's byte-preservation correction is documented
in the reproduction guide.

## Review and remaining project work

M4/M5 implementation, measured experiments, final working comparisons and a
verified handoff are complete for the owner-approved single-seed protocol.
Required non-author review/reproduction and actual consumer acknowledgment are
still pending. The PR stack is [#3](https://github.com/Rxd3/TURAS/pull/3) (M4),
[#4](https://github.com/Rxd3/TURAS/pull/4) (M5),
[#5](https://github.com/Rxd3/TURAS/pull/5) (integration); all three were marked
ready for independent review after the completion evidence was pushed. Protected
`main` is not automatically merged. Reviewers should inspect the negative RDL
result and dataset limitations as carefully as the acoustic improvements.

M2/M3/M6 still own the remaining course-wide baseline-family runs/support audits,
final cross-team integration and independently reviewed presentation. Extra seeds
or revised models are future work; tuning after these test results needs a new
protocol and suitable fresh holdout rather than reusing this test for selection.
