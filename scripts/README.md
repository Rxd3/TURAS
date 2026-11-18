# Executable entry points

Thin training/rescoring wrappers now exist for Members 4/5. Use their module
`--preflight` commands to see missing input/review gates; the draft configurations
do not start training. No script automatically downloads a dataset or reads test
references. Commands/dependencies are in the [M4](../experiments/m4-whisper-notes.md)
and [M5](../experiments/m5-ctc-rdl-notes.md) implementation notes. The wrapped
component owner and M6 review each entry point.
