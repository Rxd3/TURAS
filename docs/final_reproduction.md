# Reproducing the final delivery

Use the exact files and full SHA-256 values in `package-manifest.json`, supplied
in the public `00-START-HERE` folder. Original TARs are unchanged. Keep each
role's package separate.

## 1. Source and environment

For current source and public checks, extract `TURAS-submission-source.zip` into
a fresh directory. It includes the merged `1dade5b` base plus the explicitly
listed local closeout files. These local changes need PR publication; the ZIP
records an exact snapshot without pretending they are already on GitHub.

```text
python scripts/verify_submission.py
python -m venv .venv
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pip install -r requirements/dev.txt
.venv\Scripts\python.exe -m pytest -q
```

For neural inference or scoring with the installed shared runtime, use Python
3.13 and the actual pinned Windows environment. The M4/M5 lock is
`requirements/owner-m45-windows-cu130.txt`; the M3 model run used the versions
recorded in its training manifest and `requirements/windows-cu130.txt`.
Set `PYTHONUTF8=1`, install the appropriate lock in a fresh environment and run
`python scripts/check_env.py`. Windows CUDA reproduction needs a compatible
NVIDIA GPU/driver. No Linux environment lock is claimed.

For **exact M3 runtime source**, instead extract `m3-runtime-source.zip` from
folder 06 into a separate fresh workspace. It preserves the `d2bdf5e` source
and M3 additions used by the sealed runs. Use the current delivery's restore
helper with `--workspace` pointing to that directory. Do not replace a frozen
source tree with newer files while claiming byte-identical code provenance.

## 2. Restore only authorized inputs

The following commands run from the current submission source. Substitute the
actual downloaded package root and fresh source directory. The manifest is
`TURAS-Public-Submission/00-START-HERE/package-manifest.json`.

```text
python scripts/restore_reproduction_inputs.py --packages DELIVERY_ROOT --manifest DELIVERY_ROOT/TURAS-Public-Submission/00-START-HERE/package-manifest.json --workspace FRESH_SOURCE --roles inference,train,validation,m3
```

An authorized evaluator adds `evaluator` to `--roles`; an M4/M5 reviewer adds
`m45`. The helper verifies every requested archive before writing, rejects
traversal/link entries and refuses different existing file bytes. It restores:

| Input | Exact workspace location |
| --- | --- |
| Reference-free audio/indexes | `data/private/turas-tr-econ-legis-v0.1/inference/turas-tr-econ-legis-v0.1/` |
| Training references/tables | `data/private/turas-tr-econ-legis-v0.1/train/turas-tr-econ-legis-v0.1/` |
| Working validation references | `data/private/turas-tr-econ-legis-v0.1/validation-recovered-minutes-v1/references.validation.jsonl` |
| Working test references (final scoring only; committed in the repository, see D44) | `data/cleaned/turas-tr-econ-legis-v0.1/turas-tr-econ-legis-v0.1.references.test.jsonl`, copied to `data/private/turas-tr-econ-legis-v0.1/evaluator-final/references.test.jsonl` for the restore commands |
| M3 predictions/reports | `artifacts/m3-final-2026-11-22/` |
| B5 selected model/training evidence | `artifacts/b5-gpu-resume-2026-11-22/` |
| M4/M5 models/runs | Original TAR's `artifacts/owner-m45/...` paths |
| M4/M5 scoring index supplement | `artifacts/owner-m45/final/inference-index.json` |

The M4/M5 TAR also contains its sealed historical report, source bundle and
index. The restore helper keeps non-`artifacts/` members under
`artifacts/owner-m45/delivery-metadata/`, preserving those exact bytes without
overwriting newer public documentation. It restores model/run paths unchanged.

Do not put gold files inside the audio/inference directory. Model inference
does not need evaluator references. Test gold is used only for offline scoring
after model/decoder selection has frozen.

## 3. Recalculate B5 or HMM scores

After authorized `m3,evaluator` restoration, run from the workspace root:

```text
python scripts/reproduce_saved_scores.py --predictions artifacts/m3-final-2026-11-22/b5/test/predictions.jsonl --references data/private/turas-tr-econ-legis-v0.1/evaluator-final/references.test.jsonl --ids data/splits/turas-tr-econ-legis-v0.1/test.txt --split-manifest data/splits/turas-tr-econ-legis-v0.1/split_manifest.json --split test --output artifacts/reproduced-b5-scores.json
```

Use `gaussian-hmm/test/predictions.jsonl` for the HMM and choose another fresh
output filename. The helper uses the shared strict join, normalizer and corpus
metrics. Expected normalized WER/CER are:

| Model | WER | CER |
| --- | ---: | ---: |
| B5 epoch 22 | 0.4777631643041551 | 0.1350294077417590 |
| Supplied single-Gaussian character HMM | 0.9999512884212577 | 0.8876008753932431 |

The full M3 report also includes raw edit counts, term metrics, bootstrap
intervals, repeated timing and the independent JiWER audit. The HMM declares
`requested_n=0`; it has no supported scored N-best file. B5 has one genuine
greedy candidate per utterance.

## 4. Rerun M3 inference

Use the **exact M3 runtime workspace**, restored audio and selected model. The
sealed outputs stay in `artifacts/m3-final-2026-11-22`. Make a fresh comparison
run by changing only the output path, not decoder/model settings. In Python:

```python
from pathlib import Path
import hashlib, json
from turkish_asr_rdl.baselines.bilstm_ctc import BiLstmCtcBaselineAdapter
from turkish_asr_rdl.decoding.interface import InferenceInput
from turkish_asr_rdl.decoding.export import RunContext
from turkish_asr_rdl.decoding.timing import run_timed, write_run_outputs
from turkish_asr_rdl.schemas.jsonl import read_jsonl
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.schemas.nbest import validate_nbest

release = 'turas-tr-econ-legis-v0.1'
audio = Path('data/private') / release / 'inference' / release
checkpoint = Path('artifacts/b5-gpu-resume-2026-11-22/best-checkpoint')
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cfg = json.loads(Path('configs/baselines/bilstm_ctc.json').read_text())
adapter = BiLstmCtcBaselineAdapter(cfg, checkpoint_dir=checkpoint,
    model_artifact_sha256=digest(checkpoint / 'checkpoint-manifest.json'), device='cuda')
rows = [InferenceInput.from_row(r) for r in read_jsonl(audio / f'{release}.inference.test.jsonl')]
warmup = [InferenceInput.from_row(r) for r in read_jsonl(audio / f'{release}.inference.train.jsonl')[:3]]
split_hash = digest(Path('data/splits') / release / 'split_manifest.json')
run_id = 'reproduced-b5-test'
run = run_timed(adapter, rows, warmup, audio_root=audio, run_id=run_id, repeats=1, batch_size=1)
ctx = RunContext(run_id, adapter.system_id, release, 'test', split_hash,
    adapter.model_artifact_sha256, adapter.decoder_config_sha256)
out = Path('artifacts/reproduced-b5-test')
if out.exists():
    raise FileExistsError('Choose a fresh output directory')
write_run_outputs(out, run, rows, ctx, adapter, command='documented frozen B5 replay')
ids = (Path('data/splits') / release / 'test.txt').read_text().splitlines()
assert validate_predictions(read_jsonl(out / 'predictions.jsonl'), ids).ok
assert validate_nbest(read_jsonl(out / 'nbest.jsonl'), ids).ok
```

Use `GaussianHmmAdapter(Path('artifacts/m3-final-2026-11-22/gaussian-hmm/gaussian_hmm_character.npz'))`
for the classical model, with CPU execution and `requested_n=0`. For headline
timing, use only the already frozen `m45-timing-30min.txt` IDs, three repeats,
batch one and three train warm-ups. Run one benchmark at a time. An accuracy
run's timing is not the headline timing result.

## 5. M4/M5 reproduction and training boundary

Follow [the exact M4/M5 reproduction commands](m45_final_reproduction.md) and
[restored-checkpoint acceptance](m45_delivery_acceptance.md). Folder 05 includes
the required source-v2 and scoring-index supplement, not just the original TAR.
For the delivery-acceptance command, point `--archive` to the downloaded TAR
instead of assuming it already exists inside the workspace:

```text
python scripts/m45_accept_handoff.py --archive DELIVERY_ROOT/TURAS-Restricted-Reproduction/05-M4-M5-MODELS-AND-RUNS/TURAS-M4-M5-private-handoff.tar --output artifacts/reviewer-acceptance --inference-root data/private/turas-tr-econ-legis-v0.1/inference/turas-tr-econ-legis-v0.1
```

The pinned upstream parents remain separate dependencies:

- `openai/whisper-small` at `973afd24965f72e36ca33b3055d56a652f456b4d`
- `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` at `708639f50559d7970f462e13ec64d3f059ca89f6`

The frozen protocol lists exact parent weight hashes. Cache/verify those parents
before offline inference; they are not silently replaced with latest revisions.

Folder 09 supports historical training review and M3 continuation. The M3
latest checkpoint contains optimizer, scheduler and RNG state; the original
epoch-13 handoff has only weights/architecture. The first 13 CPU epochs cannot
be reproduced exactly from missing optimizer/RNG/history evidence. The package
supports replaying the selected model and continuing the supplied checkpoint,
not a claim of bit-identical end-to-end retraining. GPU stochastic tolerances
and the single-seed limitation remain explicit.

## Acceptance

Record real recipient name/account, package SHA, source snapshot, environment,
command and result/report hash. The initial review used the existing runtime.
The subsequent follow-up installed a new isolated Python environment without
system site packages, restored archived B5 source/weights and role-packaged
audio, and matched three archived validation transcripts exactly. Torch loaded
from the new environment, and model code loaded from the archived source.
Separate Seda and four-mixture HMM validation replays each matched three of
three archived outputs in their newly installed environments.

The same new neural environment also restored and verified all 226 M4/M5
indexed artifact members, then executed three adapted acoustic systems, three
online joint graph systems and 15 cached controls on two validation clips.
All child jobs and the source-import-origin check passed, including transcript
parity between cached and online joint decoding for all three adapted models.
See [the public restoration evidence](../results/final/fresh_environment_restoration.json).

These checks prove execution of those subsets on the same machine. They do not
supply independent human approval, a fresh operating-system installation or
full-test parity for every pipeline. The public final-review record contains
the checks actually completed; private detailed evidence retains environment
and output hashes.

## Follow-up model packages

The refreshed delivery uses restore roles `b2`, `b3`, `transducer` and `gmm`
for folders 10 through 13. These roles restore exact paths under
`artifacts/completion-2026-11-22/`. The package preparer is not in this
repository; it required completed, checked accuracy/timing outputs before
creating the packages. The earlier delivery is
preserved as a separate snapshot.

Use [follow-up replay commands](completion_replay_commands.md) to install the
neural or CPU transducer environment, download pinned upstream B2/Qwen weights,
and rerun into a fresh output directory. The Seda and four-mixture HMM packages
are described as including their model weights; they are external deliveries
and are not verified in this repository. Their inference commands read no gold
references; authorized offline scoring uses `reproduce_saved_scores.py`.
