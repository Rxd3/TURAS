# Decision register

Date: 2026-11-17. `Required` means user-mandated scope; `Working` means a proposal
to verify; `Open` means no selection/approval. These are not experiment outcomes.

| ID | Decision/status | Owner and resolution evidence needed |
| --- | --- | --- |
| D01 | Required: `Rxd3/project3-turkish-asr-rdl`, private, default `main`, package `turkish_asr_rdl` | Initialization evidence in repository setup |
| D02 | Working: WorldSpeech `tr_tr` primary candidate | M1/M6 audit revision, rights, fields, fidelity and domain coverage; team approves release |
| D03 | Working: curated Turkish legislative/financial speech | M1 documents inclusion/topic distribution and source feasibility; team confirms scope |
| D04 | Required: five baselines B1–B5; Whisper LoRA and hardware-permitting full training; CTC adaptation; trainable graph RDL | M2–M6 preserve all tracks; no completed checkbox without evidence |
| D05 | Open: exact model IDs/revisions, Qwen3-ASR Turkish capability, Turkish CTC ASR head | M2/M3 inspect official model cards/licenses/decoding/N-best support before dependencies/downloads |
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

To change a decision, append a dated rationale, status, reviewers, affected contract
versions, compatibility/migration effects and supporting PR/evidence. Preserve the
old decision history. A working decision cannot become approved solely because
implementation started. Record hardware-blocked full tuning explicitly with
measured feasibility evidence; do not delete the full tuning track from scope.
