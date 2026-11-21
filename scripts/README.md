# Executable entry points

Thin training/rescoring wrappers now exist for Members 4/5. Use their module
`--preflight` commands to see missing input/evidence gates; the owner-approved configurations
do not start training. No script automatically downloads a dataset or reads test
references. Commands/dependencies are in the [M4](../experiments/m4-whisper-notes.md)
and [M5](../experiments/m5-ctc-rdl-notes.md) implementation notes. The wrapped
component owner and M6 review each entry point.

Owner-directed preparation also provides `m45_stage_bundle.py` (private bundle
hash/path/role checks) and `m45_profile_acoustic.py` (explicitly synthetic GPU
optimizer-resource probes). See [readiness](../docs/m45_readiness.md) and
[project decisions](../docs/owner_decisions.md). No staging or profiling command
creates a completed corpus experiment or substitutes for the shared evaluator.

`check_env.py` is the environment check: a thin wrapper around
`turkish_asr_rdl.utils.env_check.main`. Run `python scripts/check_env.py` from the
repository root; it prints `READY` (exit 0) or `NOT READY` (exit 1) with one line
per problem. When the check cannot run (the package is not importable, or an
unexpected error stops the report) it prints one line and exits 1 without
`READY`/`NOT READY`. Setup steps are in
[reproducibility](../docs/reproducibility.md#environment-setup-windows).


M1 working verification: `m1_verification.py` prepares pinned review kits and
role-separated archives, ingests genuine exports, pins independent adjudication,
checks its evidence gate and publishes only new reference variants. Its
`verify-bundles` command checks every archive member and access role.
`m1_freeze_release.py` delegates to that publisher; its old in-place CLI is
retired. See [M1 closeout](../docs/m1_closeout.md). No command performs listening
or asserts another member's approval.
