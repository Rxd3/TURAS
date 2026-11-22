# PR preparation

Branch: `feat/final-baseline-completion`

Proposed title: `feat(baselines): complete follow-up evaluation and reproducible submission`

This draft remains a work in progress until all four follow-up score/timing
reports and the refreshed delivery pass their checks. No remote PR, push,
independent approval or merge is claimed.

## Problem and resulting behavior

The submission lacked locally verified B2/B3 results, a matched Turkish
transducer experiment and a genuine mixture-Gaussian classical model. This
branch adds reference-free frozen-split runners, validation-only Qwen selection,
a pinned Turkish Zipformer2 RNN-T adapter and a four-component diagonal
grapheme HMM. It also preserves the supplied single-Gaussian result and the
original 20-condition acoustic/RDL study.

The delivery scripts separate audio, references and model/run artifacts by
role. Source snapshots include file hashes; model replay uses a new output
directory and compares archived transcripts. Large upstream Whisper/Qwen
weights remain exact pinned downloads. The presentation updater retains four
editable English slides for a two-minute slot.

## Ownership and contracts

- Workstreams: M2/M3 baseline completion and M6 evaluation/delivery integration,
  performed at the project owner's explicit request. No individual member's
  contribution or approval is inferred.
- Shared changes: baseline adapters/entry points, result aggregates, tests,
  documentation, reproduction packages and presentation.
- Review routing: M6 and affected M2/M3 owners; M4 for Whisper compatibility and
  M5 for CTC/N-best compatibility. At least one actual non-author technical
  approval is required before merge.
- Compatibility: no split, reference, schema, normalization or metric semantics
  are changed. Original M4/M5 critical source files stay pinned and unchanged.
  Follow-up protocols disclose earlier working-test exposure and unknown
  pretraining overlap. Qwen joint scores keep their distinct meaning.
- Runtime entry points that produced frozen evidence are retained byte-for-byte
  rather than refactored after execution. Follow-up replay is a separate command.

## Evidence and validation

- `python -X utf8 -m pytest -q`: **1,200 passed, 19 skipped**, four upstream
  deprecation warnings, 51 passing subtests. Optional integration skips are not
  claims of full benchmark execution.
- `python scripts/verify_submission.py`: original 56 critical code pins,
  20 result conditions and all 5,450 frozen split IDs passed.
- Controller status writes retry short Windows reader locks while retaining
  atomic JSON updates. Two focused regression tests passed for transient and
  persistent locks. A controller recovery reused completed, hash-checked
  inference/scoring outputs rather than restarting model inference. Frozen
  Qwen selection timestamps are preserved across controller recovery.
- Four downloader regression tests cover metadata-only restoration and publisher-pin enforcement. All four real model inventories were reverified without downloading or changing their weights.
- Fresh isolated neural environment restored archived B5 source/weights/audio:
  three of three validation transcripts matched exactly, with no references read.
- New isolated CPU/GMM replay commands matched three of three archived
  validation transcripts for each model. These are subset execution checks.
- A fresh-source M4/M5 restore in the new isolated neural environment verified
  226 indexed artifact members and executed 3 acoustic models, 3 online joint
  graph runs and 15 cached controls on two validation clips. All 22 child checks
  passed, including source-import origin and cached/joint transcript parity.
  No reference text or test audio was read. Public evidence is in
  `results/final/fresh_environment_restoration.json`.
- Full follow-up validation/test/timing evidence, final presentation checks and
  refreshed delivery verification must be attached before this draft is ready.

Individual contribution records must be completed by the actual contributors.
This preparation does not impersonate their attestations.

## Merge checklist

- [ ] Scope is focused and affected owners have reviewed shared changes.
- [ ] Documentation/configuration/contracts match the final delivered behavior.
- [ ] No private data, weights, prediction dumps, secrets or bulk artifacts are staged.
- [x] Inference commands read reference-free audio; frozen split membership is preserved.
- [ ] All experimental claims are backed by completed output hashes and checks.
- [ ] Independent technical approval is obtained and discussions are resolved.

Drive upload, actual recipient download/access testing and independent reference
adjudication remain external human tasks. They do not become complete when local
packages are prepared.
