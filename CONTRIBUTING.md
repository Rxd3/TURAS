# Contributing to TURAS

The default branch is `main`. The initialization commit is the only planned
direct scaffold commit. Subsequent development goes through focused pull
requests from the member branches in [TEAM_TASKS.md](TEAM_TASKS.md).

## Workflow

1. Read the shared contracts and open decisions; agree interfaces before coding.
2. Fetch `origin`, update local `main`, and create/switch to your assigned
   `feat/m<N>-...` branch. Merge current `origin/main` regularly.
3. Edit primarily owned paths. Shared paths require the owner and affected
   consumers to review; ownership is a coordination duty, not unilateral control.
4. Keep logic in `src/turkish_asr_rdl/`; new `scripts/` entries must be thin
   wrappers, except the stdlib-only `verify_submission.py`, which is standalone by
   design. Put adjustable settings in the appropriate `configs/` directory.
5. Add focused offline tests when real behavior is introduced. See [test
   policy](tests/README.md). Never fetch models, data, or live APIs in unit tests.
6. Inspect `git status` and `git diff`. Stage named paths with `git add <paths>`,
   inspect `git diff --cached`, and check for secrets, large artifacts, and claims
   without evidence. Do not force-add ignored data or weights.
7. Commit with `type(scope): summary`; push your branch to the project remote.
8. Open a PR into `main` using the [PR template](.github/pull_request_template.md).
   Include checks actually run, skipped checks, evidence links, and your updated
   contribution record. Do not mark planned work complete.
9. Obtain at least one non-author technical approval. Use the member-specific
   review routing in TEAM_TASKS. Shared data/schema/normalization/evaluation or
   artifact changes also need Member 6 and affected owners' approval. If Member
   6 authors the PR, another affected owner must supply the independent review.
10. Resolve discussions and rerun affected checks after changes. Merge as a merge
    commit to retain member commits and attribution. Never force-push another
    member's branch; delete your merged topic branch only when no work remains.

Do not invent identities or assign existing reference-repository members to
TURAS. Each member fills their own record after assignment. Do not credit work
to a member solely because its paths fall within that member's ownership.

## Current checks and future implementation checks

For Members 4/5, [the owner's 2026-11-19 delegation](docs/owner_decisions.md)
supplies approval for the recorded experimental choices. Teammates may reuse those
choices without negotiating them again. Keep project approval distinct from
independent code review: do not attribute the owner's decision to another member,
fabricate an ACK, or mark baseline/training evidence complete. PR review continues
to assess correctness and integration; this delegation does not itself merge PRs.

Packaging CI builds/installs the package and checks imports, environment logic
and offline unit behavior on Python 3.11, 3.13 and 3.14, with the tests on Ubuntu
and Windows. The adaptation workflow also
exercises tiny/random-model training, decoding, score, leakage and checkpoint
behavior on CPU. These tests do not download benchmark data/weights or establish
corpus accuracy. Real training, held-out evaluation and restored-delivery checks
have their own [recorded evidence](docs/m45_completion.md).

For documentation-only PRs, check relative links, terminology, contract
consistency, and `git diff --check`. For implementation PRs, add reproducible
commands and meaningful offline tests for the changed behavior, with an explicit
environment/dependency update. Declare compatibility before adding ML libraries;
the scaffold's Python target is not proof that every model supports those versions.

## Shared contracts and data

Changes to IDs, split membership, references, normalizers, score meanings, graph
access, or metric aggregation require a versioned entry in
[decisions](docs/decisions.md), a compatibility/migration plan, and affected-owner
review. Never alter a frozen release in place or edit hashes to conceal changes.
Train and tune only with permitted train/validation inputs. Test references are
available only to the offline evaluator and must never enter decoding, prompts,
graph training, feature construction, or validation selection.

Keep audio, weights, full prediction dumps, local databases, caches, and secrets
out of Git. Small reviewed manifests, split lists, annotation policies, schema
metadata, run indices, and real aggregate reports may be versioned when rights
and privacy permit. Store bulk artifacts externally with hashes and portable
retrieval instructions. Never commit credentials or local absolute paths.

## Repository governance

Proposed GitHub settings: private visibility; `main` default; PR review required;
one independent approval; conversations resolved; no force-pushes/deletion of
`main`; packaging check required after its first successful run; merge commits
enabled, squash/rebase merging disabled. Enforcement depends on account/plan
capabilities. A written rule is not proof that branch protection is active.
Actual initialization settings and limitations belong in
[repository setup](docs/repository_setup.md). CODEOWNERS will be added only after
real GitHub identities and access are confirmed; invented handles are forbidden.

Code is MIT ([LICENSE](LICENSE)). Data and model terms are in [NOTICE](NOTICE).
