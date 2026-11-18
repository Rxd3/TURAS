# Repository initialization record

Initialization date: 2026-11-17. Local repository workspace: the user-provided
project directory. No machine-specific absolute path is part of the repository.

## Scope delivered

Written architecture/contracts, six-member task and PR workflow, empty Python
packages, dependency-free runtime metadata, ignore rules, and packaging-only CI.
No data/model download, ASR/training/graph/evaluation implementation, predictions,
metric results or member-completion claims are included.

## GitHub setup

Target at initialization: `Rxd3/project3-turkish-asr-rdl` (now
[Rxd3/TURAS](https://github.com/Rxd3/TURAS); see the current state below),
private visibility, `main` default. Git credential authentication confirmed the
account `Rxd3` before initialization. GitHub CLI was unavailable; GitHub REST API
and Git use the existing credential manager without recording credentials.

The private remote was created successfully with `main` as default, issues
enabled, merge commits enabled, and squash/rebase merging disabled. No collaborator
invitations or identity assignments were made. The initial commit uses the
authenticated account name and its GitHub noreply address, configured locally.

This document records the state before the initial commit/push. A pre-push request
to protect `main` returned `404 Branch not found` because the remote had no commit.
Protection must be retried after the first push; that response is not evidence of
a plan limitation. Final push, CI and server-protection outcomes are reported in
the initialization handoff. Written policies do not establish enforcement.

## Current repository state (checked 2026-11-17)

The repository is [Rxd3/TURAS](https://github.com/Rxd3/TURAS), renamed from
`Rxd3/project3-turkish-asr-rdl`; the old URL redirects to it. It is public, and
`main` is protected, so the protection retry described above has succeeded.
The text above records the state at initialization.

## Validation and setup outcomes

- Local Python 3.13.14 parsed the packaging metadata successfully.
- All 11 Python files are zero-byte package initializers; no application code,
  command entry points or speculative tests are present.
- Relative documentation links and necessary empty-directory markers validated.
- A real wheel built with isolated lightweight setuptools, installed into a
  temporary target, and all 11 packages imported from that installed wheel.
- No runtime/ML dependency, dataset or pretrained model was installed/downloaded.
- Git ignore checks cover raw audio, checkpoints, secrets, artifact dumps,
  predictions and Python caches. Build byproducts are ignored.
- GitHub CLI is absent; the existing Git credential manager and REST API suffice.
- Remote CI is not claimed by these local checks; it runs after the first push.

The [complete scaffold directory tree](directory_tree.txt) lists versioned files
only; `.git/`, temporary wheel/install locations and ignored build outputs are
excluded. No software license or member identities are selected.

The initial commit SHA is obtained with `git rev-list --max-parents=0 HEAD` and
reported at handoff. It cannot be embedded in its own commit without changing
the hash. The six member branches are naming conventions, not branches with
claimed implementation work.
