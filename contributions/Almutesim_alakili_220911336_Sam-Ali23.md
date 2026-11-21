# Member 1 — Dataset and Relational Schema

Name: Almutesim alakili. Student ID: `220911336`. GitHub username: [`Sam-Ali23`](https://github.com/Sam-Ali23).
Role: Member 1 — Dataset and Relational Schema.
Branch: `feat/m1-data-relational-schema`. Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

> [!NOTE]
> This is the canonical named contribution record for university submission.
> [`member_1.md`](member_1.md) is kept for backward compatibility with older
> links and holds the same dated history. PR #2 was opened from `Sam-Ali23`.
> Its commits are attributed on GitHub to the commit-email account
> `mutesim-alakili` (author name `mutesim`). The first two commits were made by a
> cloud coding-agent session under the author name `TURAS Member 1`.

**Status (2026-11-22).** The release `turas-tr-econ-legis-v0.1` was built and frozen
by Member 1 (split manifest `a8fff65b…095b8`, unchanged since). Its split files
are what every reported system uses. The project uses it as a working release
under the owner's D42/D43 decisions. **M6 release approval is pending.** The
freeze commit's statement that M6 approved the 50-clip policy was wrong and is
superseded by the
[status erratum](../data/metadata/turas-tr-econ-legis-v0.1-status-erratum-v1.json)
(`M6_approval: null`). Independent adjudication and consumer hash acknowledgments
are also pending. PR #2 is merged while M6's changes-requested review is still
outstanding.

## Summary of Member 1's own work

Delivered on 2026-11-18 in [PR #2](https://github.com/Rxd3/TURAS/pull/2), with
coding-agent assistance (Claude Code). Member 1 directed the work, made the
decisions listed below and did the pilot listening personally.

| Area | What Member 1 delivered |
| --- | --- |
| Source audit | WorldSpeech `tr_tr` at revision `b76b8539…`: card plus every non-audio column of 389 shards. Found the recording-level upstream test leak, non-spoken stenographic text in 7.1% of rows, and 24 kHz stereo audio ([audit](../data/metadata/worldspeech-tr_tr-source-audit.md)) |
| Preprocessing and selection | Screening, domain selection, recording-grouped temporal splits, 16 kHz mono FLAC conversion, IDs, hashes and exclusion ledger (`src/turkish_asr_rdl/data/`, `scripts/m1_build_release.py`) |
| Domain lexicon | `turas-domain-lexicon-v0.1`: 7 topics, 90 terms, 154 Turkish suffix-aware aliases; frozen by sha256 `8ef903fe…` ([freeze record](../configs/datasets/turas-domain-lexicon-v0.1.freeze.json)) |
| Release and split manifests | 20.0 h train / 2.0 h validation / 3.0 h test (4,376 / 438 / 636 clips; 129 / 15 / 28 recordings), with leakage checks at zero ([split manifest](../data/splits/turas-tr-econ-legis-v0.1/split_manifest.json), [release card](../data/metadata/turas-tr-econ-legis-v0.1.json), [dataset doc](../docs/dataset.md)) |
| Relational schema and tables | Populated RDL contract tables: static lexicon tables in Git ([inventory](../data/relational/README.md)); per-split evidence outside Git (train 31,639 relationships at freeze) |
| Verification tooling | Listening guide [`turas-verbatim-v0.1`](../data/verification/turas-verbatim-v0.1.md), review pages, strict review ingestion with adjudication, audio-equivalence check, freeze script |
| Pilot listening | Personally reviewed 50 clips as `rev_m1` (25 test / 15 validation / 10 train): 34 accept, 16 correct, 0 reject |
| Handoff | Built access-separated bundles and recorded their hashes ([handoff manifest](../data/manifests/turas-tr-econ-legis-v0.1-handoff.json)); answered the M4/M5 data request (2026-11-19) |

## Commits (PR #2)

All dated 2026-11-18, on `feat/m1-data-relational-schema`. GitHub records PR #2
as merged on 2026-11-20T21:16:08Z by `@afel6` (merge commit
`a0afd7d5458b0c60c3c2f98c126f15235bcd4657`).

| Commit | Subject |
| --- | --- |
| `6b3f35f9fb80d4dfa212f69a6981e41ec732f1ae` | WorldSpeech `tr_tr` audit, domain lexicon and proposed release v0.1 (cloud agent session) |
| `2e67e8d7570cda3a372ebb626769e8f6743a7a65` | Audio hashes, relational tables and review tooling (cloud agent session) |
| `d2c049fc7e7999d73deead5935daab2acbacf3d3` | Review ingestion, quantity parsing, lexicon freeze and release card |
| `ad66fa16d90439a665f84afe5d6a6f408bdd3087` | Release config resampler text matches the code (soxr HQ) |
| `ced529577df337ede1eb6274d883ea60eb776c54` | Local rebuild, freeze tooling, paged review kits, registry fields |
| `1f89e6e633401125a84908ce1a9ddfe08a13c800` | Pilot listening results (34 accept, 16 correct, 0 reject) |
| `9e55fdbd92af601da561dfe79af87b79668d75b1` | Sampled-verification policy after the 50-clip pilot |
| `2ebfa78219247e955d71913a95df8743b9a9c4bc` | Freeze `turas-tr-econ-legis-v0.1` (its subject's "M6-approved" is wrong; see the erratum) |

## Actual contributions (dated history)

### 2026-11-18 — source audit, lexicon, curation code, proposed release

- Audited WorldSpeech `tr_tr` at revision `b76b85395039f50332fbc421f7c098ee2f2653d7`
  from the card and every non-audio column of all 389 shards. Found that the
  upstream test leaks at recording level (188/188 upstream-test recordings also
  appear in upstream train), that 7.1% of references contain non-spoken
  stenographic text, and that the audio is 24 kHz stereo.
- Wrote the domain lexicon v0.1 (`configs/datasets/turas-domain-lexicon-v0.1.json`).
- Implemented `src/turkish_asr_rdl/data/` (`screening`, `selection`, `release`,
  `relational`, `worldspeech`) and the scripts `m1_build_release.py`,
  `m1_build_relational.py`, `m1_apply_reviews.py` and `m1_review_kit.py`.
- Proposed the release: 5,450 utterances from 172 recordings, a temporal 2026 test
  split and leakage checks at zero.
  Exclusions were logged per row: Ankara council source, non-spoken text,
  repeated formulaic text, upstream CER > 0.25, not in domain, quarantined
  upstream-test rows and over-cap rows.
- Wrote the listening verification guide and the 50-clip pilot page generator.

### 2026-11-18 — local verification, review fixes, lexicon freeze

- Applied the cloud bundle on a fresh clone and re-verified every committed hash,
  split disjointness and sorting.
- An independent code review was followed by fixes: idempotent, strictly validated
  review ingestion with adjudication; compound and decimal amount parsing;
  split-scoped co-mention keys; a non-destructive `--skip-audio`; build provenance
  (commit and library versions); review-page escaping.
- Froze the lexicon by hash, added the release card, and documented the split-policy
  deviations and the contamination rule in `docs/dataset.md`.
- Reviewed M6's data/predictions/normalization 0.2 proposals (approve, with an
  audio-root clarification and a raw-text term-matching note). M6 files were not
  edited.

### 2026-11-18 (evening) — local rebuild and freeze tooling

- Rebuilt the full release locally from the pinned revision. Split files and
  exclusion ledger (`6077dc6f…`) were byte-identical to the first build. The FLAC
  bytes differed by at most 1 LSB (library rounding), measured and recorded.
- Regenerated the relational evidence with the fixed extractor.
- Added `scripts/m1_freeze_release.py` and `scripts/m1_audio_equivalence.py`, the
  source-registry rights fields, and the populated-tables section of
  [`docs/contracts/rdl.md`](../docs/contracts/rdl.md).

Checks run: `pytest tests/test_data_*.py` gave 51 passed (offline, synthetic
fixtures). A trial freeze on the real artifacts left the split files identical,
and one on a synthetic copy exercised the reject and correct paths.

### 2026-11-18 — pilot listening review (human work by Member 1)

- Listened to and judged the 50 pilot clips (`rev_m1`): 34 accept, 16 correct,
  0 reject. Exported between 2026-11-18T14:15:50Z and 16:07:22Z.
- The original export is retained by Member 1 outside Git, because it contains
  reference text: `pilot-v0.1-decisions-rev_m1.json`, sha256
  `52ea3f520fd7adb14371b2473f56beb5b49627d6406c4976b8ec85db080f279d`. The ingested
  records file (sha256 `9e036d0f…`) equals `verification.review_records_sha256` in
  the committed split manifest.
- This export has not been imported into the owner's 2026-11-21 canonical review
  set, which re-reviewed the pilot IDs as `rev_rasit`. No second-listener
  adjudication of these decisions exists.

### 2026-11-18 — freeze and handoff

- Froze `turas-tr-econ-legis-v0.1` (split manifest `a8fff65b…`, unchanged since)
  under Member 1's sampled-verification decision. The freeze merged the pilot
  decisions, regenerated the relational evidence and passed 24 integrity checks.
- The freeze recorded M6 approval of the 50-clip policy on Member 1's report.
  M6 disputed this in
  [review 5962363421](https://github.com/Rxd3/TURAS/pull/2#pullrequestreview-5962363421).
  The approval is withdrawn and the erratum records it as `null`.
- Built the role-separated handoff bundles and recorded their hashes.

### 2026-11-19 — data handoff to Members 4/5

- Uploaded the train-reference bundle to the shared Drive in hash-checked parts
  and confirmed the inference bundle's upload size. Wrote the requested reply
  with hashes, extraction layout, where statuses are recorded, and the known
  alignment and number-format issues.

## Later M1-workstream work by others (not Member 1's)

The 2026-11-21 closeout, deadline handoff, 249-record primary review set
(`rev_rasit`) and working archives were produced by the project owner (`Rxd3`,
Member 5) with coding-agent assistance, under decisions D42/D43. They are
recorded in [member_1.md](member_1.md#2026-11-21--owner-requested-technical-closeout),
[M1 closeout](../docs/m1_closeout.md) and
[deadline handoff](../docs/m1_deadline_handoff.md). None of that listening or
attestation is credited to Member 1 here.

## Reviews received and issues raised

M6 (`@afel6`) requested changes on PR #2 on 2026-11-18. Items attributable to
Member 1's work:

- The approval statement in the freeze commit (corrected by the status erratum).
- The verification scope: M6 chose a stratified 224-clip test sample with
  adjudication over the 50-clip policy. The test-only correction rate is
  10/25 = 40% (Wilson 95% 23.4–59.3%), not the pooled 32%.
- The bias wording. Minutes errors depend on output style, so "the same for all
  systems" is withdrawn. 20 of the 38 pilot edits were substitutions, so "mostly
  omitted words" is also withdrawn.
- `duplicate_cluster_id` is a singleton placeholder; near-duplicates are not audited.
- Drive access: the evaluator-only TEST folder was shared by link. Its archive
  (`01ad5878…`) also contained 649 macOS `._*` AppleDouble files, from packaging
  on macOS without `COPYFILE_DISABLE=1`.

## Remaining (pending, not claimed complete)

- M6 approval of a re-audited release variant. No approval is claimed.
- Independent second-listener adjudication of the pilot corrections and the
  seeded accept sample.
- Recorded M2–M6 receipt/hash acknowledgments of `a8fff65b…`. None is recorded.
- Restricting evaluator-only Drive access to evaluator accounts.
- Handing Member 1's original pilot export to the owner through a private channel,
  if the closeout should use it as recovered evidence.
