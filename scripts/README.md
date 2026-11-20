# Executable entry points

Thin training/rescoring wrappers now exist for Members 4/5. Use their module
`--preflight` commands to see missing input/review gates; the draft configurations
do not start training. No script automatically downloads a dataset or reads test
references. Commands/dependencies are in the [M4](../experiments/m4-whisper-notes.md)
and [M5](../experiments/m5-ctc-rdl-notes.md) implementation notes. The wrapped
component owner and M6 review each entry point.

`check_env.py` is the environment check: a thin wrapper around
`turkish_asr_rdl.utils.env_check.main`. Run `python scripts/check_env.py` from the
repository root; it prints `READY` (exit 0) or `NOT READY` (exit 1) with one line
per problem. When the check cannot run (the package is not importable, or an
unexpected error stops the report) it prints one line and exits 1 without
`READY`/`NOT READY`. Setup steps are in
[reproducibility](../docs/reproducibility.md#environment-setup-windows).

Future entry points must be thin wrappers around implemented package
functionality, accept portable configuration/paths, and document actual commands
and capability constraints. The wrapped component owner and M6 review each new
entry point.
