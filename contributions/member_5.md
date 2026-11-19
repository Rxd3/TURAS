# Member 5 — CTC Fine-Tuning and RDL

Name: unassigned. GitHub username: unassigned.
Branch convention: `feat/m5-ctc-rdl-rescoring` (focused topic suffixes allowed).
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

**Status (2026-11-19): owner-assisted implementation, seed-17 CTC/graph training
and corrected-linker validation, frozen final test/timing and private handoff completed;
independent review pending.** No human identity or
individual authorship is inferred from this workstream assignment.

## Actual contributions

2026-11-19: implemented matched-head CTC training, exact CTC sequence scoring and
prefix N-best search, heterogeneous train-only graph construction/learning,
reference-free linking, joint fusion and an offline rescoring CLI. Added a shared
verified-checkpoint inference bridge, draft pinned configs and focused synthetic
tests. This branch depends on Member 4's acoustic loop; that shared code is not
claimed as separate Member 5 work. See [implementation notes](../experiments/m5-ctc-rdl-notes.md).
Commit/PR evidence is recorded in Git/GitHub at handoff; reviews are still pending.

## Experiments and artifacts

The initial implementation had no runs. On 2026-11-19 the owner-assisted seed-17
RDL run completed 20 epochs on M1's real train tables. Its internal weak-label
relation metrics and artifact provenance are in
[the actual run summary](../results/rdl/m45_seed17_internal.json) and
[the dated M4/M5 record](../docs/m45_execution_2026-11-19.md). This is not a
downstream ASR result or independent review. CTC adaptation and validation/fusion
selection remain separate tasks. Add actual run IDs, release/model/config hashes, artifact location/hash,
validation-selection evidence, outcome/limitations and corresponding baseline.
Do not enter invented transcripts, metrics, checkpoint paths or completed work.

## Reviews and handoffs

No independent reviews or consumer acknowledgments are recorded. The verified
private handoff is ready for owner delivery. [M5 PR #4](https://github.com/Rxd3/TURAS/pull/4)
and [integration PR #5](https://github.com/Rxd3/TURAS/pull/5) are ready for review.
Add actual approvals, reproduction and accepted bundle hashes when they occur.

## Remaining work

Owner-assisted seed-17 vocabulary-extended CTC training and all three acoustic
validation epochs are complete; working selection is epoch 3. The primary graph,
three fusion grids and both retrained relation-removal variants/six extra grids
also completed. Results use the separately identified recovered-minutes reference
version. Audit D37 corrected a frozen-alias suffix-linking defect and repeated
all 36 primary/ablation trials with versioned linkers and unchanged learned weights.
The corrected primary and diagnostic systems do not improve original one-best
WER. The earlier one-word CTC result is legacy-linker history, not a corrected
result. See the [numerical/linker audit](../docs/m45_result_audit.md).
See the [completion record](../docs/m45_validation_complete.md) and
[actual RDL/ablation report](../results/evaluation/m45_rdl_and_ablation_summary.json).
Final working evaluation is complete: extended CTC normalized test WER 25.150%
versus 37.152% for its unchanged parent, with vocabulary extension explicitly
attributed. All fifteen fixed-candidate controls, terminology/quantity diagnostics,
relation coverage and joint repeated timing are in the
[final report](../results/final/M4_M5_FINAL_REPORT.md). Primary learned RDL does
not beat original one-best on any of the three adapted models; small diagnostic
ablation differences are not a new test-selected primary system. The selected
CTC and learned/ablated/control graph bundles are included in the verified
private handoff. See [completion and limitations](../docs/m45_completion.md).
Remaining: required independent review/reproduction and actual consumer ACK;
optional future model/seed changes are separate experiments.
