# M4/M5 delivered-model acceptance

Engineering deliverables are complete for the owner-approved seed-17 working
benchmark. This page provides an executable consumer check. Actual non-author
PR approval and human artifact acknowledgment remain separate records; the
automated check does not create either one.

Effective decoding in the frozen model runs is Whisper beam 5 and CTC prefix
width 10, both requesting N=5. See the [provenance erratum](m45_decoding_erratum.md)
for the generic configuration field and the audit of all 28 exported runs.

## Restore and exercise the delivery

Use current integration source, the pinned Windows runtime, the original private
handoff TAR and the separately supplied inference/audio bundle. Cache the exact
Whisper-small parent listed in the LoRA manifest first; model loading is offline.
Gold references are unnecessary and are not read. From the repository root:

```text
.venv\Scripts\python.exe scripts/m45_accept_handoff.py --output artifacts/reviewer-acceptance --inference-root data/private/turas-tr-econ-legis-v0.1/inference/turas-tr-econ-legis-v0.1
```

The fresh output directory receives a clone of the delivered runtime source and
safely extracted models. The command verifies the archive SHA-256, all 226 indexed
files and frozen source hashes. It proves imports come from the restored checkout,
then executes three acoustic checkpoints, three joint acoustic/RDL systems and
fifteen cached controls using the first two frozen validation IDs and three train
warm-ups. Online joint and cached full-graph transcripts must agree for all three
models. Model/control executions plus import-origin verification make 22 jobs.

These are functional validation checks, excluded from headline timing. They do
not retrain, decode test audio, read gold, select settings or change the final
protocol/results. The installed runtime is reused: this is a fresh source/model
restore, not a new environment installation or independent human reproduction.

Success produces `acceptance.json` with `status: passed`, 226 verified files,
three acoustic executions, three joint executions, fifteen cached controls and
joint/cached parity. Progress and per-job logs remain in the output root.
Failures are preserved; use a new output path rather than overwriting an attempt.

## Actual closeout evidence

The recorded result is
[`m45_restored_delivery_acceptance.json`](../results/final/m45_restored_delivery_acceptance.json).
The [technical closeout manifest](../results/final/m45_technical_closeout.json) links that result, synchronized branch heads,
test/CI evidence and the supplemental final source bundle. The original model
archive and test metrics remain byte-identical. The source supplement supplies
current documentation and this acceptance command.

The main-branch environment update was merged through M4, M5 and integration,
resolving the component PR conflicts. Component suites passed 54 and 68 tests;
integration passed 1,095 tests, one skip and two passing subtests. Two additional
tests check extraction traversal and conflicting-file rejection. The sealed
experiment report's earlier 1,093-test count records its actual historical run.

## Genuine team sign-off

A reviewer records their own identity, reviewed commit, archive SHA-256,
command/environment, acceptance-report hash and any issues. Follow the role
routing in `TEAM_TASKS.md` and submit actual reviews on PRs #3, #4 and #5.
`CONTRIBUTING.md` step 9 requires non-author technical approval and affected-owner/
M6 review for shared changes. These owner-assisted checks make that review
concrete; they do not waive the rule, bypass protected `main` or create human ACKs.
