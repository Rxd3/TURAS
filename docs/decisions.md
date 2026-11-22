# Decision register

Date: 2026-11-17. `Required` means user-mandated scope; `Working` means a proposal
to verify; `Open` means no selection/approval; `Proposed by M6` means a contract
amendment awaiting the listed reviewers. These are not experiment outcomes.

| ID | Decision/status | Owner and resolution evidence needed |
| --- | --- | --- |
| D01 | Required: `Rxd3/TURAS`, public, default `main` (protected), package `turkish_asr_rdl`. Created as private `Rxd3/project3-turkish-asr-rdl`; by 2026-11-17 it was renamed and made public, and the old URL redirects | Initialization evidence and the 2026-11-17 check in repository setup |
| D02 | Working: WorldSpeech `tr_tr` primary candidate | M1/M6 audit revision, rights, fields, fidelity and domain coverage; team approves release |
| D03 | Working: curated Turkish legislative/financial speech | M1 documents inclusion/topic distribution and source feasibility; team confirms scope |
| D04 | Required: five baselines B1–B5; Whisper LoRA and hardware-permitting full training; CTC adaptation; trainable graph RDL | M2–M6 preserve all tracks; no completed checkbox without evidence |
| D05 | Resolved 2026-11-22 (Member 6, user decision). B2 = `openai/whisper-large-v3` @ `06f233fe` (bf16, beam 5, N 5, same decoding as B1). B3 = `Qwen/Qwen3-ASR-1.7B-hf` @ `bcd2b5b7`, selected on validation by lower WER (normalized 15.43 vs 0.6B-hf 22.47); its weights are byte-identical (707/707 tensors) to `Qwen/Qwen3-ASR-1.7B` @ `7278e1e` (released 2026-01-28). Turkish CTC head: B4 `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish`, as in the frozen protocol. | Scores: [results/m6/README.md](../results/m6/README.md). Limitation: test sessions of 6–27 Jan 2026 predate the Qwen release. The `candidate_status` field in M2's Qwen configs is M2's to update. |
| D06 | Working: BiLSTM-CTC from scratch for B5 alternative neural track | M3/M6 approve architecture, train-only vocabulary, compute budget and baseline protocol |
| D07 | Open: curated sample size, split ratios/seed, grouping thresholds and upstream-test conflicts | M1/M6 produce measured audit; freeze accepted release and hashes |
| D08 | Proposed specification: local human verification statuses and validation/test release gate | M1/M6 approve annotation instructions, reviewer/adjudication evidence |
| D09 | Proposed specification: raw-v1 and tr-normalized-v1, exact terminology policy | M1/M6 review Turkish I, apostrophes/numbers and deterministic alignment before implementation |
| D10 | Working: typed CSV tables, heterogeneous GraphSAGE-style encoder and learned edge decoder | M1/M5/M6 approve types, provenance access, negative sampling and edge split audit |
| D11 | Proposed specification: canonical natural-log scores and validation-frozen acoustic/LM/RDL fusion | M3/M4/M5/M6 verify adapter score extraction and candidate coverage |
| D12 | Open: external LM choice/need and external knowledge cutoff | M5/M6 document independent provenance, licenses, availability and no held-out reference use |
| D13 | Proposed specification: repeated seeds 17/42/2026; grouped intervals and standardized timing | M6 and model owners document actual hardware feasibility and deviations |
| D14 | Open: artifact storage, access controls, environment locks and hardware allocation | M6 coordinates portable hashed bundles and reference isolation |
| D15 | Open: member identities, GitHub access and CODEOWNERS | Each member supplies own details; maintainer confirms access; no invented names/handles |
| D16 | Open: software license; dataset redistribution approval remains separate | Team/maintainer chooses explicit license and audits source terms |
| D17 | Proposed by M6 (2026-11-17): [normalization](contracts/normalization.md) 0.1 → 0.2. U+02BC treated like U+0027/U+2019; `%` kept; Cf characters deleted; whitespace means `str.isspace()`; digit-apostrophe and number-grouping token effects documented; alignment tie-break backtraces from the end, preferring diagonal, then deletion, then insertion. Label `tr-normalized-v1` kept until the first real evaluation freezes it. Amended by M6 (2026-11-17): a final Unicode NFC (step 7) after whitespace collapse; `tr-normalized-v1` is idempotent for every input. Amended by M6 (2026-11-18): in step 4, U+0027, U+2019 and U+02BC never count as letters when testing an apostrophe's neighbours, so `a''b` and `a` + U+02BC + U+02BC + `b` both give `a b` | Reviewers: M1 and model owners M2–M5. Rationale: U+02BC is a letter and survived 0.1; `%` carries meaning in quantities; invisible Cf characters would count as errors; 0.1 left whitespace and tie direction open. Compatibility: no evaluation has run, so nothing is re-scored; any change after the freeze needs `tr-normalized-v2` and full reevaluation. Refines D09. Amendment rationale (2026-11-17): the spec requires an idempotent normalizer, and deleting a Cf character or an apostrophe can leave a pair that NFC composes (`e` + U+200B + U+0301), so steps 1–6 alone were not idempotent; the label is still unfrozen, so nothing is re-scored. Amendment rationale (2026-11-18): 0.2 treats U+02BC as an apostrophe, but its category Lm made it a letter neighbour, so one doubled spelling gave `ab` and the other `a b`; the label is still unfrozen |
| D18 | Proposed by M6 (2026-11-17): [N-best](contracts/nbest.md) 0.1 → 0.2. Null scores give their reason in `score_details.null_reasons`; optional row-level `dedup_map` (removed candidate ID → kept ID); candidate texts pairwise distinct after deduplication; original rank values kept, gaps allowed. Amended by M6 (2026-11-17): kept ranks are strictly increasing integers in 1..`requested_n`, the first kept rank is 1, and gaps from deduplication are allowed only after it. Amended by M6 (2026-11-18): the shared export owns deduplication (first, lowest-rank occurrence of each exact raw text kept; `dedup_map` only when non-empty; policy text in `score_details.duplicate_policy`); adapters return the decoder's raw list; candidates with the same text and identical `token_ids` make an error row | Reviewers: M4/M5 (owners) and M3. Rationale: in transformers 5.23.0, Whisper `generate(num_return_sequences>1)` returns copies of the 1-best; 0.1 forbade repeated strings but named no checkable keys. Compatibility: rows keep `schema_version` `"0.1"`; `dedup_map` is optional; no N-best artifact exists yet. Refines D11. Amendment rationale (2026-11-17): deduplication keeps the best copy, so rank 1 always survives; a list that starts above rank 1 has lost its best candidate. No N-best artifact exists yet, so nothing is re-checked. Amendment rationale (2026-11-18): in the 2026-11-17 smoke run every beam system's N-best failed validation only on duplicate text, so one shared export covers every producer; its 76 duplicate pairs had 0 identical `token_ids`, so the copy rule never fires on real beams |
| D19 | Proposed by M6 (2026-11-17): [predictions](contracts/predictions.md) 0.1 → 0.2. Stable order is the sorted split-file order; `split_manifest_sha256` is the SHA-256 of the split manifest file's bytes | Reviewers: M1 and model owners M2–M5. Rationale: 0.1 required a stable order and a manifest hash without defining either, so two correct exporters could disagree. Compatibility: no field changes; rows keep `schema_version` `"0.1"`; no predictions exist yet |
| D20 | Proposed by M6 (2026-11-17): [data](contracts/data.md) 0.1 → 0.2. Release audio stored as 16 kHz mono FLAC under an artifact root outside Git; `audio_uri` is relative to that root with `/` separators | Reviewer: M1 (owner). Rationale: one shared loader reads the same files for every system inside the timed region, so end-to-end RTF stays comparable; relative `/` paths work on Windows and Linux. Compatibility: drops the 0.1 option of an access-controlled artifact URI; no release exists yet. Relates to D07 and D14 |
| D21 | Proposed by M6 (2026-11-17): [evaluation](evaluation.md) 0.1 → 0.2. Headline RTF is end-to-end, batch 1, 3 repeats on a frozen 30-minute timing subset; core-model RTF is a labeled secondary view; `clocks_event_reasons` recorded before and after each repeat; course metric names mapped; deadline cuts listed as declared limitations | Reviewers: M1 and model owners M2–M5. Rationale: one timing protocol for every system on the shared re-run GPU; the 2026-11-23 deadline forces the 30-minute subset and single-seed runs. Compatibility: nothing has been timed or scored yet. Narrows D13 for this deadline without removing repeated seeds or full tuning from scope |

## Owner delegation and integration (2026-11-19)

The owner delegated M4/M5 technical decisions and approved documenting the choices
for reuse. [D22–D31](owner_decisions.md) adopt the pinned parents, initial budgets,
M6 contract 0.2, graph boundary, strict CTC target auditing and validation-only
selection. They supersede pending inter-member agreement for those experimental
choices; they do not invent peer reviews, data access or experimental evidence.

To change a decision, append a dated rationale, status, reviewers, affected contract
versions, compatibility/migration effects and supporting PR/evidence. Preserve the
old decision history. A working decision cannot become approved solely because
implementation started. Record hardware-blocked full tuning explicitly with
measured feasibility evidence; do not delete the full tuning track from scope.

## M1 closeout and contamination boundary (2026-11-21, D42)

Owner adopted the [D42 working verification/publication policy](owner_decisions.md)
and [data contract 0.3](contracts/data.md). This follows the M6 PR #2 Option C
request; independent release approval remains pending. The new plan is after
working test exposure and cannot be claimed blind before all evaluation.

613/636 TURAS test clips are public WorldSpeech train rows. Externally
WorldSpeech-fine-tuned parents/baselines and unfiltered WorldSpeech train are
prohibited. The group's own fine-tuning on frozen TURAS **train** IDs is allowed.
No text from held-out test-session dates (2026-01-06 through 2026-04-22) may enter
training, graph evidence, an LM or prompts. Record checkpoint revisions and
training-data evidence; publication dates alone do not prove all pretraining
is disjoint. The ASR-agreement CER filter may favour Whisper-family outputs;
edited minutes can bias models differently. Exact group/hash/text checks do not
establish speaker or near-duplicate disjointness.
