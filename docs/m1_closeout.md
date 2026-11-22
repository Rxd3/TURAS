# Member 1 closeout — 2026-11-21

Technical preparation is complete; human review and independent release approval
are pending. This document supersedes the historical 50-clip stop policy. The
owner adopts [D42](owner_decisions.md) and [data contract 0.3](contracts/data.md);
owner approval does not manufacture listening decisions or another member's ACK.

The [D43 deadline handoff](m1_deadline_handoff.md) records owner acceptance of
continued work with these declared limitations, supplies all 636 test-audio links
locally and adds the real recovered-validation relational supplement. It does
not mark listening complete or bypass verified-variant publication checks.

## What exists and what was checked

All 16 remote branches and PRs #1–#10, their files and review history were checked
before continuing. All 10 PRs were merged at the audit time; M1's PR #2 still has
a changes-requested latest review. No later M1 branch contains the missing
corrections/review exports. The focused closeout starts from `main@1ad2d7c`.
Unmerged M6 documentation/harness and member-assignment changes are unrelated;
their commits are not implicitly merged or attributed here.

| Item | Actual status |
| --- | --- |
| Frozen IDs/audio | 4,376 train / 438 validation / 636 test; 129 / 15 / 28 recordings; unchanged |
| Audio audit | All 5,450 stored hashes, 16 kHz mono PCM16 formats and durations checked |
| Exact overlap | Zero cross-split recording groups, audio hashes and shared-normalized reference strings |
| Speakers/near duplicates | Speakers unknown; recording fallback. Near-duplicate audio not audited; singleton duplicate IDs are placeholders |
| Verification selection | Eight clips per test recording, seed 2026: 224 including original 25 pilot test IDs |
| Primary evidence required | 249 distinct IDs: 224 test + 15 validation pilot + 10 train pilot |
| Prepared listening pages | 199 new test clips in seven pages; 50 pilot evidence-recovery clips in two pages; no decisions prefilled |
| Actual new human decisions | Zero supplied; old reference status tags do not substitute for missing payloads |
| Clean role bundles | Four real archives, every member checksum and role checked, no macOS metadata |
| Frozen experiment integrity | Old split/audio/reference and M4/M5 experiment bytes preserved; no model training or re-evaluation performed |

The plan was prepared **after existing working test results were opened**.
Selection uses frozen recording/utterance metadata and old pilot IDs only, never
predictions or errors. It cannot be presented as blind before all evaluation.

The original manifest SHA remains
`a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8`.
Its false approval claim is superseded by the
[status erratum](../data/metadata/turas-tr-econ-legis-v0.1-status-erratum-v1.json),
without rewriting its bytes. The committed
[plan](../data/verification/turas-tr-econ-legis-v0.1-stratified-test-v1.json),
[sorted LF IDs](../data/verification/turas-tr-econ-legis-v0.1-stratified-test-v1.txt),
[preparation audit](../data/verification/turas-tr-econ-legis-v0.1-preparation-audit-v1.json)
and [working handoff inventory](../data/manifests/turas-tr-econ-legis-v0.1-m1-working-handoff-v1.json)
bind the actual hashes and counts. Protected references and pages stay outside Git.

## Use the prepared owner workspace

On this machine, the private directory is:

```text
<repo-root>\data\private\turas-tr-econ-legis-v0.1\m1-closeout-2026-11-21
```

`<repo-root>` is the repository checkout folder.

Open `review-kits/index.html`. The page links the seven `test-new-*.html` pages
and two `pilot-evidence-*.html` pages. A Turkish-speaking listener enters their
own pseudonymous reviewer ID, listens to every assigned clip, chooses
accept/correct/reject under `turas-verbatim-v0.1`, and exports the JSON decisions.
Pages embed audio and references: send only to assigned listeners, keep them
out of Git and reference-free model inputs. Export before changing browsers or
machines; browser autosave is local and is not a shared review database.

PR #12 is merged. Use the current `main` checkout and the existing pinned
environment for these commands. The protected owner workspace remains in place.

```powershell
# Run from the current repository checkout.
$py = '<repo-root>\.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$m1 = '<repo-root>\data\private\turas-tr-econ-legis-v0.1\m1-closeout-2026-11-21'

# Import genuine primary exports; choose fresh output names for each import.
& $py scripts/m1_verification.py ingest --workspace $m1 `
  --exports <primary-export-1.json> <primary-export-2.json> `
  --out "$m1\records-primary-v1.jsonl"

# For later imports, retain prior records explicitly.
& $py scripts/m1_verification.py ingest --workspace $m1 `
  --records "$m1\records-primary-v1.jsonl" --exports <next-export.json> `
  --out "$m1\records-primary-v2.jsonl"

# Refuses until every required primary ID has genuine evidence.
& $py scripts/m1_verification.py adjudication-kit --workspace $m1 `
  --records "$m1\records-primary-v2.jsonl" --out "$m1\adjudication-v1"

# Different listeners review the generated adjudication pages and export JSON.
& $py scripts/m1_verification.py ingest --workspace $m1 `
  --records "$m1\records-primary-v2.jsonl" --exports <adjudication-export.json> `
  --out "$m1\records-final-v1.jsonl"

& $py scripts/m1_verification.py check --workspace $m1 `
  --records "$m1\records-final-v1.jsonl" `
  --adjudication-plan "$m1\adjudication-v1\adjudication-plan.json" `
  --out "$m1\gate-final-v1.json"

# Only after check passes; this creates new files, never replaces old gold.
& $py scripts/m1_verification.py publish --workspace $m1 `
  --records "$m1\records-final-v1.jsonl" `
  --adjudication-plan "$m1\adjudication-v1\adjudication-plan.json" `
  --variant turas-minutes-reviewed-v1 --sampled-verification `
  --out "$m1\reference-variants\turas-minutes-reviewed-v1"
```

Angle-bracket paths above stand for real reviewer exports, not generated sample
decisions. `check` exits 2 while evidence is missing. Missing/stale adjudication,
same-listener adjudication and rejected IDs block publication. The private check
report records rejection IDs/issues/evidence as a ledger; it never drops or
re-splits frozen clips. Every correction/rejection/conflict and a seed-2026 10%
accept sample need independent adjudication. The 13 known train/test pilot
corrections plus all 15 validation pilot clips are mandatory while original
validation correction identities are missing. New primary records invalidate
stale adjudication plans. No output path is silently overwritten.

The publisher's manifest still says independent M6 release review is pending.
If text changes, regenerate relational evidence in a **new** variant directory,
validate train-only graph access, and have M6 run the same shared evaluator for
all systems. Do not change model weights, tune fusion or select checkpoints using
these post-test decisions. Preserve the original working benchmark as history.

## Reproduce preparation and verify delivery artifacts

The existing pinned environment includes pandas 3.1.1 and pyarrow 26.0.1 in
`requirements/dev.txt`; both declare Python >=3.11. No heavyweight package was
installed for this closeout. The source profile pins every actual input hash.
Preparing again requires the same protected local input files and a fresh output
directory; it neither downloads data nor creates transcripts.

```powershell
& $py scripts/m1_verification.py prepare `
  --inputs-root '<repo-root>' `
  --private-root <fresh-private-output-directory>
& $py scripts/m1_verification.py verify-bundles --workspace $m1
& $py scripts/m1_verification.py check --workspace $m1
```

The last command intentionally reports blocked while no records exist.
On Windows use UTF-8 mode (`PYTHONUTF8=1` or `python -X utf8`); legacy frozen
graph provenance code decodes `git diff` through Python's locale default.
The default cp1252 mode fails on Turkish dirty-tree diffs. This closeout
preserves that hash-pinned code and verifies the suite in explicit UTF-8 mode.
Reproduction must use the committed profile/code and pinned private bytes.
`m1_freeze_release.py` is now a wrapper for the safe publisher; the old
in-place-freeze CLI is retired. The legacy full builder is **not** a command to
re-split the current benchmark. Do not run it over the frozen release.

`working-exports/` contains inference/audio, train-reference,
validation-working-reference and evaluator-working archives. Train reference
bytes remain `f4992d29…`; working validation is `d0c8519c…` (all 438 unreviewed,
not the original three corrected rows); test snapshot is `31e372f0…`.
Exact full hashes and file sizes are in the public inventory and private
`working-handoff.json`. The evaluator bundle lists missing components; it is
not a reconstruction of the original `7a721b6b…` M6 archive. Do not distribute
the evaluator bundle or listening pages together with model inference inputs.

## What still needs real people or unavailable evidence

1. Recover compatible original 50 pilot review exports, or complete the recovery
   pages, plus 199 new test clips. Obtain independent adjudication as above.
2. Recover the original corrected validation bundle and exclusion ledger if
   possible. Current archives explicitly disclose their absence; they cannot be
   reconstructed from model output, a source URL or owner approval.
3. Restrict Drive evaluator sharing to actual evaluator accounts, then deliver
   role-specific artifacts and record genuine receipt/hash ACKs. Anonymous folder
   metadata was listable in the supplied folders; no Drive write capability was
   available here and no live ACL change is claimed.
4. M6 re-audits the named closeout head/variant, M5 checks relational provenance,
   and a non-author tests retrieval/reproduction. PR #2's merge is not proof that
   its changes-requested release review was resolved.

Edited minutes are output-style-biased; the reported test pilot correction rate
is 10/25 = 40% (Wilson 95% 23.4–59.3%), not a universal WER adjustment. Report
full and reviewed-subset results with reference variants and real coverage.
Unknown speakers, unaudited near duplicates, the upstream ASR-agreement filter
and adjacent validation/train sessions remain limitations. Follow the explicit
WorldSpeech/test-session contamination boundary in [decisions](decisions.md).

## Technical validation

- Full offline suite in Windows UTF-8 mode: **1,174 passed, 19 skipped,
  51 passing subtests**. Skips are reported as skips, not completed model runs.
- Thirteen focused verification-gate tests cover missing evidence, independent
  adjudication, stale snapshots, correction preservation, rejection blocking,
  deterministic selection, UTC ordering, immutable LF writes and bundle checks.
- All four working archives passed full/member checksums and role boundaries.
- All 249 pages match their reference snapshot and embedded audio hashes;
  the generated JavaScript passed `node --check`.
- Shared train-only graph validation accepted all 4,376 frozen train IDs and
  static/train CSV foreign keys/provenance; no optimization was performed.
- `scripts/verify_submission.py` passed all 56 frozen code hashes, the 20-system
  stored metric consistency check, and the 5,450 disjoint frozen split IDs.

Public aggregate-only evidence:
[`working-integrity-v1`](../data/verification/turas-tr-econ-legis-v0.1-working-integrity-v1.json).
These checks do not constitute human transcript verification, release approval
or proof that an artifact has been delivered to another member.
