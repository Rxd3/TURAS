# Executable entry points

`check_env.py` is the environment check: a thin wrapper around
`turkish_asr_rdl.utils.env_check.main`. Run `python scripts/check_env.py` from the
repository root; it prints `READY` (exit 0) or `NOT READY` (exit 1) with one line
per problem. When the check cannot run (the package is not importable, or an
unexpected error stops the report) it prints one line and exits 1 without
`READY`/`NOT READY`. Setup steps are in
[reproducibility](../docs/reproducibility.md#environment-setup-windows).

Future entry points must be thin wrappers around implemented
package functionality, accept portable configuration/paths, and document actual
commands and capability constraints. Training, inference, downloads and model
setup are deliberately absent from this initialization. The wrapped component
owner and M6 review each new entry point.
