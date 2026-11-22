# Reproducibility plan

Packaging and the environment check can be verified now. No ASR experiment is
reproducible from this repository yet because implementations, approved data, checkpoints
and results do not exist. The Windows CUDA environment is pinned in
`requirements/windows-cu130.txt` (see [Environment setup](#environment-setup-windows)).
CI checks packaging and runs the unit tests on Python 3.11 and 3.13.

## Scaffold inspection

Use Python 3.11+ to parse `pyproject.toml` with `tomllib`; use `PYTHONPATH=src`
or install the package to import the package and its subpackages (all empty apart
from `utils.env_check`, which imports torch only when the check runs). Check
whitespace with `git diff --check`. These checks require no data/model/ML
dependencies and do not demonstrate inference or training. Initialization-specific
checks and remote state are recorded in [repository setup](repository_setup.md).

## Future evidence levels

1. **Recalculate metrics:** obtain hashed predictions and evaluator-only references,
   verify release/ID alignment, run the pinned evaluator/normalizer and compare
   edit counts, aggregate reports and figure inputs.
2. **Rerun inference:** add the exact model/processor/tokenizer, decoder, graph/LM
   artifacts, compatible locked environment and audio; verify candidate scores,
   selected hypotheses, coverage, timing scope and artifact hashes.
3. **Retrain:** add train/validation data, training configuration, initialization,
   seeds, sampler ordering, hardware/precision and validation selection evidence.
   Report stochastic tolerances; byte-identical GPU training is not assumed.

All levels require the [artifact/experiment contract](contracts/artifacts.md).
Keep separate framework environment locks where compatibility demands it; record
exact dependency versions and hashes at the implementation PR, not guessed pins
in this scaffold. Do not install heavyweight dependencies for initialization.

## Release audit checklist

- [ ] M1/M6 approve source terms, references, human verification and frozen split hashes.
- [ ] Every model bundle identifies upstream/base/adapted provenance and compatibility.
- [ ] Inference inputs and permissions exclude references and held-out annotations.
- [ ] RDL training adjacency/features exclude held-out edges and reverse duplicates.
- [ ] Selection is frozen on validation before final test evaluation.
- [ ] All five baselines and actual fine-tuned/RDL conditions use the common evaluator.
- [ ] Every claimed metric has edit/support counts and real artifact hashes.
- [ ] Timing identifies hardware, warm-up, batch and end-to-end scope.
- [ ] Ablations preserve shared candidate sets and matched acoustic baselines.
- [ ] A non-author reproduces the appropriate level in a clean environment.
- [ ] Negative findings, unsupported tracks and resource limits are explicit.
- [ ] Presentation figures and contribution records point to reviewed evidence.

None of these application-release items is complete at initialization.

## Environment setup (Windows)

Every teammate and the training machine install the same pinned packages from
`requirements/windows-cu130.txt`: Python 3.13, torch 2.14.1 with CUDA 13.0
(`+cu130`) and the Hugging Face stack. Follow the steps in order, then share the
output of `python scripts/check_env.py` with the team.

1. **NVIDIA driver 580 or newer.** The CUDA 13.0 build of torch needs driver
   branch R580 or newer (CUDA Toolkit release notes, Tables 2 and 3:
   https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). Check the
   "Driver Version" shown by `nvidia-smi`; update with the NVIDIA app if it is older.
2. **Python 3.13 (64-bit)** from https://www.python.org/downloads/windows/.
   `python --version` must print `Python 3.13.x`; with several Pythons installed,
   create the venv with `py -3.13 -m venv .venv`.
3. **FFmpeg, shared build, on PATH.** torchcodec decodes audio with FFmpeg 4 to 9
   and on Windows needs a *shared* build (separate DLLs). Download one from
   https://ffmpeg.org/download.html, add its `bin` folder to the user PATH, open a
   new terminal and check `ffmpeg -version`.
4. **Virtual environment and packages**, from the repository root:

   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements/windows-cu130.txt
   pip install --no-deps -e .
   ```

   In PowerShell activate with `.venv\Scripts\Activate.ps1`. The download is about
   2.2 GB (the torch wheel alone is 2.0 GB). The requirements file carries its own
   index lines, so plain `pip` is enough.
5. **Check the machine:** `python scripts/check_env.py` prints `READY` (exit 0) or
   `NOT READY` (exit 1) with one line per problem. It checks the Python minor
   version, that torch is installed, loads and is a CUDA build, that CUDA is
   available, and that a small GPU computation succeeds. The `torch arch list`
   line is information only. When the check cannot run (the package is not
   importable, or an unexpected error stops the report) it prints one line and
   exits 1 without `READY`/`NOT READY`.

**Gotcha (uv in sandboxed Windows environments):** when installing Python with
`uv` under restricted `AppData` paths, set `UV_PYTHON_INSTALL_DIR` to a
writable folder outside `AppData` first.

**Unit tests only (CPU, no torch):** `pip install -r requirements/dev.txt`,
`pip install --no-deps -e .`, then `python -m pytest --cov=turkish_asr_rdl
--cov-report=term-missing`. This is what CI runs.

**Changing the packages:** edit `requirements/requirements.in`, then, from the
`requirements/` folder, run the `uv pip compile` command printed in the header of
`windows-cu130.txt` with `-o windows-cu130.txt`. It ranks PyPI above the PyTorch
index, so only the `+cu130` packages (torch, torchcodec) come from the PyTorch index.

**Mac/Linux:** the lock is resolved for Windows only. On Linux, compile your own copy
with the header command, changing it to `--python-platform x86_64-unknown-linux-gnu
-o linux.txt`. Macs have no CUDA; train on a teammate's Windows machine or a cloud GPU.

## Additional completed baseline runs

The [baseline completion report](../results/final/BASELINE_COMPLETION_REPORT.md)
and [machine-readable summary](../results/final/baseline_completion_summary.json)
record six completed archived runs. All validation/test prediction and supported
N-best exports were checked against the frozen ID order, and the current shared
evaluator reproduced their raw/normalized edit counts and terminology micro
metrics exactly. Inference was not rerun on this updated checkout. The original
20-condition M4/M5 study and its frozen source hashes remain unchanged.

For original-runtime replay, use the existing role-separated artifact delivery
`TURAS-FINAL-DELIVERY-2026-11-22-v2`. Folder 06 contains the M3 results/model ZIP
and `m3-runtime-source.zip` for B5 and the supplied single-Gaussian HMM. Folders
10–13 contain B2, Qwen, Seda and the four-mixture HMM outputs/models. The public
`00-START-HERE/TURAS-submission-source.zip` preserves their exact follow-up
runtime and replay commands. Extract archived source into a separate directory;
do not overwrite current repository files with it. Large pretrained B2/Qwen
weights are separate pinned downloads; Seda acoustic and trained HMM weights
are included in their model packages. The summary records exact package hashes.

Owner-assisted restoration in newly installed isolated Python environments
matched three archived validation transcripts per new evaluated model; all four
complete saved-test score reports also reproduced exactly. This is distinct
from full-test inference replay, a fresh OS installation or independent review.
Actual Drive upload/access and a public download URL are not claimed here.
