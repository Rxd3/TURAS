# Local M4/M5 synthetic GPU resource checks

Measured 2026-11-19 on this computer's NVIDIA GeForce RTX 5070, using PyTorch
2.14.1+cu130, CUDA 13.0, Transformers 5.23.0, PEFT 0.21.3 and Python 3.13.14.
CUDA forward/backward and bf16 support were verified. Exact installed packages
are in [the scoped owner runtime lock](../requirements/owner-m45-windows-cu130.txt).
This is the M4/M5 environment, not a claim that every optional curation/codec
package from M6's broader environment has been installed.

**Synthetic optimizer probes, not corpus training or ASR results.** Only model
architecture JSON metadata was downloaded. Models were randomly initialized;
inputs and target IDs were random tensors. No pretrained weights, corpus audio,
transcripts, generated predictions or WER/CER were used or produced.

Each row uses a 30-second-equivalent input shape, batch one, bf16 autocast with
fp32 parameters, activation checkpointing, one warm-up optimizer step and two
measured steps. Probe accumulation is 1; approved training accumulation is 16.
Times exclude loading, preprocessing, data I/O and evaluation.

| Mode | Target tokens | Peak allocated GiB | Peak reserved GiB | Mean measured optimizer-step seconds |
| --- | ---: | ---: | ---: | ---: |
| lora | 128 | 1.19 | 1.32 | 0.119 |
| lora | 440 | 1.36 | 1.52 | 0.117 |
| full | 128 | 4.57 | 5.02 | 0.137 |
| full | 440 | 4.57 | 5.01 | 0.146 |
| ctc_finetune | 128 | 5.88 | 6.82 | 0.251 |

All five shapes completed without OOM. This supports starting the small-model
experiments at batch one, including a real full Whisper fine-tuning attempt once
data arrives. It does not prove actual target/audio lengths or N-best evaluation
fit, nor establish an epoch duration or recognition quality. Repeat the check on
permitted training examples before the long run. XLS-R's convolutional frontend
is frozen, matching the selected adaptation.

## Reproduce

Activate a compatible CUDA environment and install the editable package:

```powershell
python -m turkish_asr_rdl.training.profile --config configs/fine-tuning/whisper_small_lora.json --target-lengths 128 440 --steps 2 --output artifacts/environment/lora-repeat.json
python -m turkish_asr_rdl.training.profile --config configs/fine-tuning/whisper_small_full.json --target-lengths 128 440 --steps 2 --output artifacts/environment/full-repeat.json
python -m turkish_asr_rdl.training.profile --config configs/fine-tuning/ctc_xlsr_turkish.json --target-lengths 128 --steps 2 --output artifacts/environment/ctc-repeat.json
```

Use fresh output paths. These measure synthetic **training** optimizer resources;
real inference timing still uses M6's shared `run_timed` harness. The local unit
suite also passes with this CUDA build and GPU visibility disabled for offline
CPU unit checks.

## Evidence

Original reports are ignored artifacts under `artifacts/environment/`. Their hashes:

- `whisper-lora-synthetic.json`: `fedf3818c93f228e5963283b1603dc897f5140ab1c2dfdde967bb1440a64a918`
- `whisper-full-synthetic.json`: `b21cba8f033af852e35faea7a00b538ad1c285f4ade3a5001968f346c38b3e62`
- `ctc-synthetic.json`: `7ffe4563c8543227f3aaea3df79efc8dcba34394b27d98fbacc99439c1b51f30`

Probe-source SHA-256: `6b5441e5a3999323256b184ae2c69ea087383a06ae573824628f998bd7401173`.

Official Windows cp313 wheel: `torch-2.14.1+cu130-cp313-cp313-win_amd64.whl`,
SHA-256 `f064c797bb497daf2b2b5e32b6d794abef087e3f24e758e6c6f8010bd2698319`.
It was retrieved from the official PyTorch CUDA13.0 index, byte-range checked
and hash-verified before installation in the project `.venv` only.
