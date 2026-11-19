# Six-member project board

**Status: initialized 2026-11-17; M4/M5 implementation and working validation
completed by 2026-11-19, with the frozen final benchmark running.** Names and GitHub usernames are unassigned. Members add their
own identities to contribution records after assignment; reference-repository
identities must not be reused by assumption. A checked task must cite a reviewed
PR, commit, or real artifact/run evidence. Specifications alone do not complete
training, data curation, model evaluation, integration or reproducibility work.

Primary ownership identifies who coordinates and reviews a path. Shared-contract
changes require affected owners and Member 6; no member can change shared data,
scores or evaluation semantics alone. Every member must understand the overall
methodology, their assumptions, dependencies, leakage risks and limitations.

**2026-11-19 owner update:** M4/M5 experimental choices are now approved under
[D22–D31](docs/owner_decisions.md). Repeated member agreement is not a dependency
for those choices. Code review, actual data handoffs and actual experimental
evidence remain separate; historical task wording does not imply another member
has approved or acknowledged anything.

## Ownership and review matrix

| Member | Workstream and branch | Owned areas (future leaf paths are reservations) | Required technical review |
| --- | --- | --- | --- |
| 1 | Dataset and Relational Schema — `feat/m1-data-relational-schema` | `data/`, `src/turkish_asr_rdl/data/`, `configs/datasets/`, `docs/dataset.md`, `docs/contracts/data.md`; relational CSV definitions in `docs/contracts/rdl.md` | M6 for release/contracts; M5 additionally for relational tables |
| 2 | Seq2Seq and Speech-LLM Baselines — `feat/m2-seq2seq-speech-llm-baselines` | Baseline `whisper*`/`qwen3_asr*` leaves and matching baseline configs; B1/B2/B3 notes | M6 plus M4 for Whisper compatibility |
| 3 | CTC and Classical Baselines — `feat/m3-ctc-alternative-baselines` | Baseline `ctc*`/`bilstm_ctc*` leaves, BiLSTM model/training leaves, CTC decoding export and baseline configs; B4/B5 notes | M6 plus M5 for CTC compatibility |
| 4 | Whisper Fine-Tuning — `feat/m4-whisper-finetuning` | Model/training/decoding `whisper*` leaves, `configs/fine-tuning/whisper*`, Whisper adaptation notes | M2 for parent baseline; M5 for N-best; M6 for common contracts |
| 5 | CTC Fine-Tuning and RDL — `feat/m5-ctc-rdl-rescoring` | Model/training `ctc*`, `src/turkish_asr_rdl/rdl/`, decoding `rescoring*`, CTC fine-tuning configs, `configs/rdl/`, `configs/decoding/rdl*` | M3 for CTC; M1 for schema; M6 for scoring/leakage |
| 6 | Integration, Evaluation, and QA — `feat/m6-integration-evaluation-qa` | Evaluation/reporting/schemas/utils packages; shared decoding, shared package files, evaluation/shared-decoding configs; integration/reproducibility/results/tests governance | Affected data/model owner; independent non-author approval mandatory |

All paths above are under `src/turkish_asr_rdl/` when referring to package leaves.
Each member owns their `contributions/member_<N>.md` and component tests/notes.
`tests/` is coordinated by M6; component authors own their named test files.
`scripts/` has the same component ownership as the wrapped package function;
M6 reviews all new entry points. Shared `__init__.py`, README, CONTRIBUTING,
TEAM_TASKS, pyproject, `.github/`, decisions and protocol changes are coordinated
by M6 with affected-owner review. `docs/contracts/nbest.md` is M4/M5 joint scope;
M6 maintains its common interface. M6 coordinates `presentation/` with evidence
and content contributions from all members. No named implementation file exists
until a reviewed implementation PR adds it.

## Member 1 — Dataset and Relational Schema

**Branch:** `feat/m1-data-relational-schema` (focused suffixes allowed).

**File ownership:** `data/**`, `configs/datasets/**`, package `data/**`,
`docs/dataset.md`, `docs/contracts/data.md`, relational schema sections of
`docs/contracts/rdl.md`, future `experiments/m1-data-notes.md`, component tests and
data entry points. Graph learning remains Member 5's responsibility.

**Tasks:**

- [ ] Audit WorldSpeech `tr_tr` revision, fields, source terms, access and domain coverage.
- [ ] Propose inclusion/exclusion and verbatim transcription/verification instructions.
- [ ] Preserve source provenance, stable IDs, audio hashes and exclusion reasons.
- [ ] Perform documented human listening verification and adjudication; never guess text.
- [ ] Audit source/session/speaker/duplicate groups and upstream-test conflicts.
- [ ] Freeze train/validation/test definitions and reference access boundaries with M6.
- [ ] Curate a versioned domain vocabulary and typed relational CSV tables with M5.
- [ ] Produce actual duration/topic/verification/leakage statistics and a release card.
- [ ] Validate the release and obtain consumer acknowledgment of the same hashes.

**Exact expected deliverables:** updated `docs/dataset.md` and data/RDL contracts;
future `data/metadata/<release>.json`, `data/manifests/<release>.json`,
`data/splits/<release>/{train,validation,test}.txt` and split manifest;
`data/verification/<release>-audit.md`; relational table schema/data inventory
under `data/relational/`; `experiments/m1-data-notes.md`; meaningful data-contract
tests when implemented. Protected full manifests/references/audio and bulky CSVs
are delivered as real external hashed bundles, not forced into Git.

**Dependencies:** M6 for ID/evaluation/release checks and access control; M5 for
relational type/graph needs; actual team approval of source rights and working domain.

**Objective definition of done:**

- [ ] Source revision/rights and actual reference origin are recorded for every row.
- [ ] Every frozen validation/test reference has accepted local verification evidence.
- [ ] Unique IDs, disjoint groups, duplicate audits and byte hashes validate without hidden exclusions.
- [ ] CSV keys/types and provenance access validate; test-derived facts cannot enter training/inference.
- [ ] M2–M6 acknowledge a single release; a non-author verifies the regeneration/retrieval instructions.

**Handoff/review:** submit source/schema policy PR first (M5+M6 review), then
release/audit PR (M6 and consumers review). Give all model owners inference-only
exports; give evaluator-only references to M6. Record approvals and actual work in
[member_1.md](contributions/member_1.md).

**Specific next task:** write the WorldSpeech `tr_tr` source/revision/terms/field
audit and assess legislative/financial coverage using documentation first.

## Member 2 — Seq2Seq and Speech-LLM Baselines

**Branch:** `feat/m2-seq2seq-speech-llm-baselines`.

**File ownership:** package `baselines/whisper*`, `baselines/qwen3_asr*`,
`configs/baselines/whisper*`, `configs/baselines/qwen3_asr*`, future
`experiments/m2-baseline-notes.md`, matching thin scripts and component tests.
Whisper adaptation/training is Member 4's responsibility.

**Tasks:**

- [ ] Pin suitable official Whisper-small, Whisper-large-v3 and Qwen3-ASR checkpoints/revisions/licenses.
- [ ] Verify Turkish support, audio/language/task constraints, score access and hardware needs.
- [ ] Agree shared prediction/decoding contracts with M4/M6 before adapter code.
- [ ] Implement reference-free unmodified pretrained adapters for all three tracks.
- [ ] Freeze validation-based generation settings and record genuine scoring capabilities.
- [ ] Export complete B1/B2/B3 prediction/manifests; document unsupported scored N-best honestly.
- [ ] Obtain M6 raw/normalized WER/CER, terminology and standardized timing evaluation.
- [ ] Document observed errors, resource constraints and capability limitations from real runs.

**Exact expected deliverables:** three pinned baseline configurations under owned
config prefixes; actual B1/B2/B3 adapters and focused offline tests; future
`experiments/m2-baseline-notes.md`; external prediction bundles plus model/run
manifests per baseline; reviewed real run-index entries in `results/` through M6.

**Dependencies:** M1 frozen release/inference export; M6 contract/evaluator/timing;
M4 parent checkpoint/decoder agreement; approved hardware/model access.

**Objective definition of done:**

- [ ] All three baselines decode real frozen IDs without reference access.
- [ ] Every baseline names upstream revision, processor/tokenizer and full decoding policy.
- [ ] Exported IDs/hashes validate; score semantics and failures are explicit.
- [ ] M6 reproduces matched evaluations/timing from real bundles; M4 acknowledges Whisper parent compatibility.

**Handoff/review:** open capability/config PR before model downloads, reviewed by
M6 and M4 for Whisper. Adapter/result PRs require M6 and the affected consumer.
Give M4 exact B1/B2 identities and give M6 all three complete bundles. Record
evidence in [member_2.md](contributions/member_2.md).

**Specific next task:** document exact candidate model IDs/revisions, Turkish
capability and real score/N-best support for B1, B2 and B3 without downloading weights.

## Member 3 — CTC and Classical Baselines

**Branch:** `feat/m3-ctc-alternative-baselines`.

**File ownership:** package `baselines/ctc*`, `baselines/bilstm_ctc*`,
`models/bilstm_ctc*`, `training/bilstm_ctc*`, `decoding/ctc*`, matching baseline
configs, future `experiments/m3-baseline-notes.md`, component scripts/tests.
CTC fine-tuning belongs to M5; shared CTC export changes require M5/M6 review.

**Tasks:**

- [ ] Select an already ASR-capable Turkish Wav2Vec2/XLSR checkpoint and compatible head/vocabulary.
- [ ] Audit license, revision, blank/pad IDs, tokenizer, audio and decoder constraints.
- [ ] Confirm BiLSTM-CTC as B5 alternative neural architecture with M6, including compute budget.
- [ ] Implement the unmodified B4 pretrained adapter and B5 train-only vocabulary/training protocol.
- [ ] Implement and train B5 on the frozen train split; select settings only on validation.
- [ ] Establish genuine CTC beam/lattice N-best and acoustic score export with M5/M6.
- [ ] Export complete B4/B5 predictions, checkpoints/provenance and resource records.
- [ ] Obtain common metrics/timing and document real baseline limitations.

**Exact expected deliverables:** pinned B4 config and B5 architecture/training config
under owned baseline prefixes; actual baseline/model/decoder code and offline
tests; future `experiments/m3-baseline-notes.md`; real external B4/B5 model and
prediction bundles with manifests; M5 CTC export handoff and M6-reviewed run index.

**Dependencies:** M1 approved train/validation/test release; M6 evaluation/contracts;
M5 CTC fine-tuning and score convention agreement. B5 must not depend on test vocabulary.

**Objective definition of done:**

- [ ] B4 is a valid pretrained ASR baseline, not an encoder with an invented/random ASR head.
- [ ] B5 is a real trained alternative neural baseline with validation selection evidence.
- [ ] CTC sequence/path/beam meanings and optional LM contributions are correctly distinguished.
- [ ] B4/B5 full-ID bundles pass M6 checks and matched reporting; M5 can reuse exact B4 provenance.

**Handoff/review:** capability/design PR reviewed by M5+M6; baseline/export PRs
reviewed by M6 with M5 for shared CTC paths. Give M5 the parent checkpoint,
vocabulary/processor and decoding contract. Record actual work in
[member_3.md](contributions/member_3.md).

**Specific next task:** audit Turkish-capable pretrained CTC checkpoint candidates
and propose the BiLSTM-CTC train-only baseline protocol and resource budget.

## Member 4 — Whisper Fine-Tuning

**Branch:** `feat/m4-whisper-finetuning`.

**File ownership:** package `models/whisper*`, `training/whisper*`,
`decoding/whisper*`, `configs/fine-tuning/whisper*`, Whisper sections of the N-best
contract, future `experiments/m4-whisper-notes.md`, component scripts/tests.

**Tasks:**

- [x] Freeze exact B1 parent/processor and adaptation protocol under owner D23; no M2 review is inferred.
- [x] Document LoRA target modules, trainable parameters, budget and validation selection.
- [x] Measure full fine-tuning feasibility and execute the seed-17 full experiment.
- [x] Run LoRA training on the owner-authorized train split and select on the declared working validation reference version.
- [x] Run a separate full fine-tuning experiment and evaluate all saved epochs.
- [x] Save compatible adapter/full bundles with parent/release/tokenizer hashes and verify loading.
- [x] Produce genuine scored Whisper N-best and consume it in M5's working rescoring experiments.
- [x] Compare unchanged, LoRA and full systems with the shared M6 evaluator on validation.
- [ ] Complete terminology metrics, final repeated timing/test comparisons, and final private artifact/report handoff.

**Exact expected deliverables:** reviewed Whisper LoRA and full experiment configs;
actual training/N-best code and meaningful offline score/compatibility tests;
future `experiments/m4-whisper-notes.md`; real external adapters/full checkpoints,
prediction/N-best/run bundles; hardware feasibility evidence if full tuning cannot
run; matched M6 evaluation entries. No large-v3 adaptation is claimed unless its
own B2-matched experiment is actually executed.

**Dependencies:** M1 approved train/validation/references; M2 B1/B2 identities;
M5 candidate/score consumption; M6 evaluation/timing/provenance; hardware access.

**Objective definition of done:**

- [x] LoRA training/selection has actual evidence and adapters load only against the exact base.
- [x] Full experiment has separate measured training and validation evidence.
- [x] Real N-best scores can be traced to model outputs; no repeated one-best padding.
- [ ] M5 accepts candidates and M6 verifies matched baseline/adaptation comparisons and artifacts.

**Handoff/review:** protocol/config PR reviewed by M2+M6; N-best contract/export PR
by M5+M6; trained-artifact report by M2+M6. Record actual evidence and hardware
blocks in [member_4.md](contributions/member_4.md).

**Specific next task:** finish terminology metrics and the final three-repeat
timing condition, freeze the final protocol, then run the held-out comparison
and complete reproducible private artifact/report handoff. See the actual
[working validation](docs/m45_validation_complete.md) and [audit](docs/m45_result_audit.md).

## Member 5 — CTC Fine-Tuning and RDL

**Branch:** `feat/m5-ctc-rdl-rescoring`.

**File ownership:** package `models/ctc*`, `training/ctc*`, `rdl/**`,
`decoding/rescoring*`, `configs/fine-tuning/ctc*`, `configs/rdl/**`,
`configs/decoding/rdl*`, graph/scoring sections of RDL/N-best contracts, future
`experiments/m5-ctc-rdl-notes.md`, component scripts/tests. M1 owns relational CSV
curation and M3 owns unmodified CTC baseline/export infrastructure.

**Tasks:**

- [x] Freeze B4 parent and the explicit vocabulary-extension contribution under owner D23/D33; no M3 review is inferred.
- [x] Run CTC adaptation, select the working validation checkpoint and save compatible artifacts.
- [x] Adopt M1's hashed typed/provenanced CSV schema and enforce train-only snapshot filters.
- [x] Implement trainable heterogeneous message passing and typed relation decoder.
- [x] Split relation evidence by group; audit held-out/reverse edges, features and negative sampling.
- [x] Train/validate relation scoring and report genuine internal training-group holdout metrics.
- [x] Implement reference-free hypothesis linking and acoustic/RDL rescoring; LM is explicitly disabled in the primary comparison.
- [x] Complete D37 corrected-linker validation and relation-removal ablations on fixed acoustic candidates.
- [x] Freeze the final test systems/fusion configurations and their reference versions; protocol commit `4639b5e`, before test gold access.
- [ ] Complete terminology/relation coverage, end-to-end timing and final matched test/report handoff with CTC/RDL artifacts.

**Exact expected deliverables:** reviewed CTC, graph and rescoring configs;
actual training/graph/linker/rescorer code plus meaningful leakage/score tests;
future `experiments/m5-ctc-rdl-notes.md`; real external CTC checkpoints, graph
tables/snapshot/learned weights, prediction/N-best/component-score bundles and
run manifests; actual relation evaluation plus matched downstream ASR evidence.

**Dependencies:** M1 typed/provenanced tables and approved data; M3 B4 baseline and
CTC score export; M4 Whisper scored N-best; M6 evaluator/timing/shared contracts.
CTC adaptation and graph interface work can be separate focused PRs.

**Objective definition of done:**

- [x] Adapted CTC uses a documented B4 parent, explicit vocabulary extension and matched working validation.
- [x] Required RDL component learns parameters; a lookup table alone cannot satisfy it.
- [x] Training graph/features exclude held-out reference evidence and equivalent reverse edges.
- [x] Real model-derived candidates preserve IDs/scores; all fusion weights use validation only.
- [ ] M6 validates acoustic-only/LM/RDL ablations on the same candidates, with relation coverage and end-to-end cost.

**Handoff/review:** CTC PRs require M3+M6, schema/graph-access PRs M1+M6, and
rescoring/export PRs M4 (candidate consumer/producer compatibility) plus M6.
Record code, runs and limitations in [member_5.md](contributions/member_5.md).

**Specific next task:** complete the corrected-linker audit and final fusion
record, then terminology/relation coverage, repeated joint timing and final
matched test/report handoff. A positive RDL WER gain is not claimed as a done
condition; the measured outcome and limitations must be reported accurately.

## Member 6 — Integration, Evaluation, and QA

**Branch:** `feat/m6-integration-evaluation-qa`.

**File ownership:** package `evaluation/**`, `reporting/**`, `schemas/**`,
`utils/**`, shared decoding interfaces, `configs/evaluation/**`, shared decoding
configs, `docs/evaluation.md`, `docs/reproducibility.md`, shared artifact/text
contracts, results/presentation and repository governance. Other members supply
their own component tests/content; M6 coordinates without claiming their work.

**Tasks:**

- [ ] Review shared contracts with owners and coordinate versioned interface approval.
- [ ] Implement strict ID/release/audio/hash checks and reference-free input boundaries.
- [ ] Implement Turkish normalization and raw/normalized corpus WER/CER with edit counts.
- [ ] Implement terminology evaluation, error-inclusive reporting and paired comparisons.
- [ ] Establish hardware/warm-up/batch/RTF/end-to-end timing protocol and sidecars.
- [ ] Integrate five baselines, Whisper/CTC adaptation and real scored N-best/RDL outputs.
- [ ] Coordinate validation freeze, final matched comparisons, ablations and uncertainty.
- [ ] Audit environment locks, seed records, artifact hashes and non-author reproduction.
- [ ] Produce evidence-backed reports/slides with all members' sign-off and explicit limitations.
- [ ] Configure supported GitHub governance after member identity/access confirmation.

**Exact expected deliverables:** implemented shared schemas/evaluator/timing and
integration checks; reviewed configuration and contracts; actual aggregate tables,
error/ablation reports and hashed run index in `results/`; executable reproduction
commands documented only after they exist; final `presentation/` deck with source
links; real GitHub governance/access records. Tests must verify real behavior,
not mirror empty packages or claim ASR validity from packaging imports.

**Dependencies:** all model/data owners' compatible contracts and real bundles;
M1 evaluator-only references; hardware and artifact distribution access; independent
reviewers for M6's own changes. Evaluation design need not wait for trained models.

**Objective definition of done:**

- [ ] Tiny hand-checkable metric fixtures and incompatible-bundle rejection tests pass offline.
- [ ] All reported systems share the frozen IDs/reference policy and are reference-free at inference.
- [ ] Every comparison, timing figure and ablation links to genuine manifests/artifacts and supports.
- [ ] Another member reproduces the appropriate run level in a clean environment.
- [ ] All owners review presentation claims; pending/blocked work and negative findings remain visible.

**Handoff/review:** metric/text PRs require M1 plus affected model owners; integration
PRs require the component owner; result/reproduction PRs require non-author model
and data-owner review. M6 cannot self-approve. Record work in
[member_6.md](contributions/member_6.md).

**Specific next task:** lead a contract review covering IDs, gold access, Turkish
normalization, corpus aggregation, timing, artifact compatibility and review routing.

## Shared milestones and integration checkpoints

- [ ] M0 — Assign identities/access; each member reads scope/contracts and creates their member branch.
- [ ] M1 — Agree source/annotation plan, all five baseline identities and shared prediction/score/evaluation interfaces.
- [ ] M2 — Approve verified data release, rights, frozen grouped splits and consumer hash acknowledgments.
- [ ] M3 — All five baselines export valid full-ID predictions; shared evaluator produces real matched reports.
- [ ] M4 — Whisper LoRA and CTC tuning are evidenced; full Whisper tuning has a real experiment or documented hardware block; scored N-best is validated.
- [ ] M5 — Learned heterogeneous RDL, leakage audit and validation-only fusion selection pass review.
- [ ] M6 — Freeze pre-test configuration and run final baseline/fine-tuned/RDL comparisons and ablations.
- [ ] M7 — Independently reproduce results; all owners review final report/presentation and limitations.

No dates or personal assignments are invented. Model adapters can develop in
parallel after contracts agree, but meaningful experiments require the approved
data release. RDL schema design can start early; learned scoring depends on real
permitted relation evidence and rescoring depends on genuine N-best artifacts.

## Team decisions and final readiness

Resolve [D01–D16](docs/decisions.md) with the named owners. Every PR uses
[CONTRIBUTING.md](CONTRIBUTING.md) and the PR template, includes updated individual
evidence, and obtains the reviewers specified above. All required reviewers are
distinct from the author; where multiple specialties are listed, obtain each
affected specialty review rather than treating one approval as universal sign-off.

- [ ] All workstreams preserve the complete scope and objective completion evidence.
- [ ] Source rights, human verification, ID/split integrity and graph leakage audits are approved.
- [ ] Five baselines, matching acoustic adaptations and trained RDL are honestly reported.
- [ ] Raw/normalized WER/CER, terminology, RTF/latency, ablations and failures are traceable.
- [ ] Checkpoint, experiment, artifact and prediction contracts are machine-validated when implemented.
- [ ] Contribution records distinguish code, experiments, review and pending work.
- [ ] Clean reproduction and final presentation have non-author and owner sign-offs.

Initialization does not satisfy these readiness items; it only establishes the
directories, written contracts, package metadata and collaboration workflow.
