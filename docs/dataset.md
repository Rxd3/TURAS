# Working dataset and curation plan

Candidate: [disco-eth/WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech),
configuration `tr_tr`. Target contribution: curated Turkish legislative/financial
speech. This is a working decision, not a released dataset or approval to publish
audio/references. No data was downloaded during initialization.

The [upstream card](https://huggingface.co/datasets/disco-eth/WorldSpeech/blob/main/README.md)
was inspected on 2026-11-17. It lists `tr_tr`, upstream train/test files,
`human_transcript`, `asr_transcript`, and source/segment metadata. It describes
CC BY-NC 4.0 with per-source terms and lists Turkish parliamentary proceedings.
This card review does not establish redistribution rights or local transcript
verification. Do not treat upstream ASR-alignment scores as quality guarantees.

## Required audit before release

Member 1 records the exact upstream revision, source URLs, source-specific
rights/attribution, permitted access/sharing, transcript origin, audio encoding,
sampling rate, segment boundaries, availability and any privacy restrictions.
Check whether financial speech is actually present; narrow or supplement the
working domain only through a documented team decision with equivalent audits.
Do not infer domain membership from the dataset name alone.

Define inclusion/exclusion criteria before inspecting model test errors:
Turkish language, relevant legislative/financial topics, intelligible aligned
audio, verified reference fidelity, and permitted use. Log every rejection and
reason. Preserve natural numerals, currencies, names, acronyms, hesitation policy,
and verbatim transcription guidelines; changes require versioned annotation policy.

## Human transcription verification

An upstream `human_transcript` starts as `unreviewed` for TURAS. A verifier listens
to the full segment, checks timing and fidelity, and records an evidence-linked
decision. Disagreements or difficult speech go to another reviewer/adjudicator.
Validation and test references must all be `verified` or `adjudicated` before
freeze; ambiguous/unintelligible segments are excluded with reasons, never guessed.
Train reference status is reported and any weaker supervision is labeled in
training runs. Do not use ASR output as a human-approved reference by default.

## Leakage-resistant splits

Build connected groups from source recording/session identity, overlapping time
segments, audio duplicate/near-duplicate clusters, repeated transcript clusters,
and known speaker identity where available. Keep each connected component in
one split. Unknown speakers must be explicit: do not claim speaker-disjointness
without evidence. Audit lexical duplication separately so formulaic parliamentary
phrases do not masquerade as distinct independent test observations.

Keep upstream test records out of training/validation and preserve upstream
assignments in provenance. Prefer an audited upstream test as a candidate TURAS
test, then derive validation from upstream train at group level. If upstream
boundaries leak, quarantine conflicting groups or define a new versioned subset;
do not silently recycle upstream test as train. Freeze group algorithm, thresholds,
ratios, seed, and actual membership only after the audit; none are chosen here.
Report limitations from unknown model pretraining contamination.

M1 and M6 review counts, duration, topics, source/speaker overlap, verification,
exclusions, duplicate statistics, rights and split hashes before approving a
release. Split manifests and a release card are required; no counts or hours are
claimed now. See the [data contract](contracts/data.md) for required fields.
