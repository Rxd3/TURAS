# M1 deadline handoff — working benchmark

The owner accepts continued work with the existing, explicitly labelled working
references under D43 (2026-11-21). The available-input technical handoff is
complete. Human listening/adjudication and independent release acceptance remain
incomplete; no status or review record is inferred from an assumption.

## What the team can use now

- Frozen release `turas-tr-econ-legis-v0.1`: 4,376 train / 438 validation /
  636 test IDs; split manifest SHA-256
  `a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8`.
- Four clean, role-separated private bundles listed in the
  [working handoff inventory](../data/manifests/turas-tr-econ-legis-v0.1-m1-working-handoff-v1.json).
  Every member checksum, reference boundary and all 5,450 audio files were checked.
- A separate validation relational supplement generated from the exact recovered
  minutes: 438 utterances, 274 quantifiers, 3,784 relationships. All 438 source
  references remain `unreviewed`. This supplement is evaluator/validation-selection
  material only; it must never enter a training graph or inference input directory.
- Pinned source/lexicon/schema contracts, stable IDs, recording-disjoint splits,
  leakage checks, reproducible commands and machine-readable
  [deadline status](../data/metadata/turas-tr-econ-legis-v0.1-deadline-status-v1.json).

Use train references/train relational evidence for training. Use the separately
hashed working validation variant for selection, then freeze each remaining
system's settings before test inference. Test references and held-out relational
evidence stay evaluator-only. No test-based model/weight selection is authorized.
The original experiment files, reference text, frozen split membership and
artifact hashes are preserved. The new supplement does not recreate M1's missing
three validation corrections or the unavailable original M6 archive.

## Where the actual clips are

Local paths relative to the repository root, all protected/outside Git:

```text
data/private/turas-tr-econ-legis-v0.1/m1-deadline-2026-11-21/all-test-audio.html
data/private/turas-tr-econ-legis-v0.1/m1-closeout-2026-11-21/review-kits/index.html
data/private/turas-tr-econ-legis-v0.1/m1-closeout-2026-11-21/working-exports/
data/private/turas-tr-econ-legis-v0.1/m1-deadline-2026-11-21/validation-evidence/
```

`all-test-audio.html` links all **636 real test audio files** in frozen split-file
order, without reference text or invented decisions. The review index contains
the **199 additional test clips** and **50 pilot evidence-recovery clips**, with
audio and editable reference text for genuine listening. These two pages serve
different purposes. Do not share the reference-bearing review pages as inference
inputs or put protected materials in the public repository.

The expanded test review selection is 224 clips across 28 recordings, including
the original 25 test pilot IDs, plus 15 validation and 10 train pilot items.
The [first receipt](../data/verification/turas-tr-econ-legis-v0.1-review-progress-v1.json)
recorded 57 compatible browser-export primary decisions. The
[current receipt](../data/verification/turas-tr-econ-legis-v0.1-review-progress-v2.json)
adds 192 acceptances from the owner's explicit confirmation that all remaining
items were reviewed and accepted, retaining the four earlier corrections.

All **249 required primary items** now have a record: **245 accept, four
correct**. The 192 new records identify their source as a user batch attestation;
their UTC timestamp is the attestation recording time, not an invented per-clip
playback time. The earlier 57 records and original export bytes are unchanged.
These records do not establish independent adjudication or complete verbatim
verification of all 636 test clips.

The final seeded adjudication plan contains **55 items**, including the 25-item
accept sample and mandatory pilot/correction checks. A different listener must
review them; `rev_rasit` cannot adjudicate their own primary decisions. No
independent adjudication records have been received, so the strict verified-
variant gate remains blocked. Reference text and reported model metrics remain
unchanged.

Current private files, relative to the review workspace
`data/private/turas-tr-econ-legis-v0.1/m1-closeout-2026-11-21/`:

```text
review-imports/rev-rasit-2026-11-21-02/remaining-192-attestation-rev_rasit.json
review-imports/rev-rasit-2026-11-21-02/primary-records.jsonl
review-imports/rev-rasit-2026-11-21-02/adjudication-v1/index.html
review-imports/rev-rasit-2026-11-21-02/adjudication-v1/adjudication-plan.json
```

Use this 249-record file as `--records` when importing genuine second-listener
exports, and use the listed plan in the subsequent `check` command. Keep both
the transcript-bearing attestation export and adjudication pages private.

## Completion wording for the report and presentation

> M1 completed the working dataset's technical curation, frozen recording-group
> splits, provenance/schema documentation, automated integrity checks and
> access-separated local artifact preparation. References are edited TBMM minutes.
> Primary decisions comprise browser exports and an explicit user batch
> attestation. Independent adjudication is incomplete, and validation uses an
> explicitly recovered upstream-minutes variant. Results are a working benchmark
> with those reference-quality limitations.

Do not replace this with "all clips were listened to", "fully human-verified",
"M6 approved", "speaker-disjoint", or an invented consumer acknowledgment.
The deadline does not establish those facts. The new review plan was prepared
after earlier working test exposure; it is not blind before all evaluation.

## Reproduce the validation supplement

Run from the merged repository using the existing pinned environment:

```powershell
$env:PYTHONUTF8 = '1'
python scripts/m1_verification.py validation-tables `
  --inputs-root <repository-with-pinned-private-inputs> `
  --out <fresh-private-validation-supplement-directory>
```

The command checks the exact reference hash and frozen validation IDs, calls the
existing relational extractor, retains the exact reference bytes and actual
verification tags, and atomically writes a fresh private supplement. It refuses
train/test rows, changed input hashes, existing outputs and public output paths.
No audio/model download or ASR inference is involved.

## Outstanding evidence and access limits

The original full exclusion ledger, corrected validation bundle and compatible
pilot decision payloads remain unavailable. The owner accepts those disclosed
limits for the working study; no substitute ledger or transcript correction is
invented. Restricted Drive sharing and actual consumer receipt still need the
appropriate account access and real acknowledgments. No cloud ACL change or
independent M6 release sign-off is asserted by this handoff.

If genuine listening exports arrive, follow [M1 closeout](m1_closeout.md), run
independent adjudication and publish a separately versioned reference variant.
If text changes, regenerate affected relational tables and re-score stored
predictions consistently for every system; preserve the original benchmark.

## Checks for this deadline supplement

The full offline suite passed **1,175 tests**, with 19 skips and 51 passing
subtests, in Windows UTF-8 mode. All 56 frozen code hashes and 20-system stored
metric consistency checks pass. Every member checksum of the new supplement and
all three scopes' relational keys/types/provenance/access were checked. Its
validation references remain byte-identical to the pinned recovered input.
The genuine-listening gate continues to refuse publication with missing records.
