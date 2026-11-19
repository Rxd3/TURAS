# Checkpoint, experiment and artifact contract — 0.1

Future artifact bundles are stored outside Git in an agreed access-controlled
location. Portable relative file names and cryptographic hashes identify them;
never invent a checkpoint path or put access tokens in a URI. No artifact exists
in this scaffold. Small run indexes may link to real bundles after review.

## Model/checkpoint manifest

Required: schema version, system ID, model family, immutable upstream repository
and revision, license/usage provenance, exact parent checkpoint hash, actual
training mode (`pretrained`, `scratch`, `lora`, `full`, `ctc_finetune`, `ctc_vocab_extension`, `rdl`),
code commit/dirty state, runtime/dependency lock hash, tokenizer/vocabulary hash,
feature extractor/processor hash, audio sample-rate/channel policy, language/task
settings, expected input/output contracts, decoding compatibility, weight file
names/sizes/SHA-256, and creation environment.

Adapted checkpoints also record dataset release and split hashes, trainable and
total parameter counts, seed(s), optimizer/scheduler/precision, effective batch,
epochs/steps, validation selection criterion, selected step, stopping rule and
configuration hash. LoRA records exact base revision/hash, target modules, rank,
alpha/dropout, adapter hashes and merged/unmerged state. Loading an adapter with
a different base/tokenizer is an error. Full fine-tuning has a separate manifest
and resource evidence; LoRA cannot be relabeled full training.

CTC manifests pin blank/pad IDs, vocabulary and normalization policy; a bare XLSR
encoder without an ASR head is not a pretrained ASR baseline. Checkpoint/tokenizer
or graph-vocabulary mismatches must fail explicitly. Prefer safe serializations;
do not load untrusted executable pickles as part of artifact verification.

## Experiment manifest

Every real run has one UTF-8 JSON manifest with these groups:

| Group | Required contents |
| --- | --- |
| Identity/state | schema version, unique run ID, UTC start/end, `planned`/`running`/`completed`/`failed`/`blocked`, owner member, purpose, baseline parent run |
| Code | commit SHA, dirty diff hash if any, exact command and relative working directory |
| Environment | OS, Python, dependency lock/hash, framework/CUDA/cuDNN versions, accelerator/driver, CPU/RAM/storage |
| Inputs | dataset release, source revision, split/manifest/audio inventory hashes, preprocessing and annotation policy |
| Configuration | exact resolved config and hash, decoder/generation parameters, prompt, precision, batch, language |
| Randomness | Python/NumPy/framework/GPU/data-worker/sampler seeds actually used, deterministic flags and nondeterministic operations |
| Model/knowledge | upstream/base/adapted artifact hashes, tokenizer/processor, graph tables/snapshot/checkpoint/linker hashes, external context access |
| Selection | train/validation usage, candidate search space, criterion, chosen settings, freeze timestamp and pre-test commit |
| Outputs | relative artifact names, byte sizes, SHA-256, JSONL prediction/N-best counts, score convention and selected candidate links |
| Evaluation/timing | evaluator/normalizer versions and hashes, raw edit counts, metric/timing report hashes, hardware and scope |
| Limitations | failures, unsupported capabilities, hardware blocks, contamination risk, omissions and reviewer evidence |

Initial repeated-training seed plan is `17`, `42`, `2026`, subject to recorded
resource feasibility. These are proposed seeds, not completed runs. A single-seed
result must be labeled as such; never average nonexistent repetitions. All model
comparisons use the same release and evaluation protocol, even if the frameworks
need separate dependency locks/environments.

Hash exact bytes with SHA-256; hash each bundle member and the manifest bytes
separately. Do not put a manifest's own hash inside itself. A separate artifact
index contains its manifest hash. Do not hash mutable remote URLs in place of
downloaded bytes. Record file order/serialization for deterministic regeneration.
Verify hashes before loading/reusing outputs; unknown/mismatched provenance blocks
comparison. Keep protected references separate from inference model bundles.
