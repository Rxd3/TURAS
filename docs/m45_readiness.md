# M4/M5 readiness and handoff

Verified 2026-11-19. [Owner decisions](owner_decisions.md) supply experimental
protocol approval. No new teammate vote is needed for those recorded choices.
Independent code review and factual experimental evidence remain separate.

## Verified here

- Frozen split manifest bytes match `a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8`.
- Train/validation/test ID files match their frozen hashes: 4,376 / 438 / 636 IDs,
  sorted, unique and mutually disjoint. This verifies IDs, not audio/reference fidelity.
- All six public static relational CSV hashes match M1's frozen inventory.
- M1 quantity extraction and M6 normalization, scoring, export, timing and schema
  code are integrated with M4/M5. Raw candidates go through shared deduplication;
  RDL component scores and selected candidate IDs survive export.
- 1,081 offline unit tests passed, one skipped, with two subtests. The final integrated suite passed on the installed CUDA build with GPU hidden
  for unit tests; the prior 1,080-test suite also passed on the earlier CPU build.
- CUDA forward/backward and bf16 work locally. Five synthetic GPU resource checks
  passed for Whisper LoRA/full and CTC: [scope and measured evidence](m45_gpu_profile.md).
  These are not actual corpus training or accuracy results.
- The repaired evaluator-only archive remains unchanged and unextracted.

## Source integration

The integration branch preserves the original commits; it does not invent reviews
or attribute other members' source code to the owner:

- `feat/m1-data-relational-schema`: `2ebfa78219247e955d71913a95df8743b9a9c4bc`
- `feat/m6-env`: `bf4e0bf1319aa573f90f73e3cf8ec0a29a197608`
- `feat/m6-harness`: `022446cf9fb1ca4b48465d762e06862820abda0d`
- `feat/m6-evaluator`: `82c7570055d955a207d7e36a170f39d295824e97`
- `feat/m6-smoke-tests`: `dfc1beefd4a49109643bd9ba9b21502682e5bb59`
- `feat/m6-docs`: `9f09a5f24e431fa83ce68b0c3933985e3af67cb2`

## Still genuinely missing

1. The private inference/audio bundle.
2. The private train-reference/train-relational bundle.
3. A separate validation-only reference file for isolated local scoring, or genuine
   M6 validation reports. Validation/test references never enter the trainers.

Use [the prepared Member 1 message](member1-handoff-request.md). Public metadata
cannot reproduce private listening corrections or prove an unavailable bundle's
bytes, so those inputs are not replaced by guesses. Baseline completion, selected
checkpoints/fusion and final test scores remain unmeasured until real runs occur.

## Stage files after they arrive

The following commands use the actual published archive names. They are future
instructions; those two archives are not present locally yet. Use fresh output
subdirectories and keep the existing test archive separate.

```powershell
python -m turkish_asr_rdl.utils.bundles --kind inference --archive turas-tr-econ-legis-v0.1-inference-ALL-MODEL-OWNERS.tar --output data/private/turas-tr-econ-legis-v0.1/inference
python -m turkish_asr_rdl.utils.bundles --kind train --archive turas-tr-econ-legis-v0.1-train-references-M3-M4-M5.tar --output data/private/turas-tr-econ-legis-v0.1/train
```

Staging checks the expected published SHA-256 before extraction, rejects unsafe
paths/links and existing destinations, checks inference's gold-free schema and
train-only reference/relationship roles, and saves a private audit. Failed staged
payloads are marked failed, never declared ready. Full training loaders still
verify file/audio hashes and frozen ID joins before any optimization.

## Execution after staging

1. Fill an ignored resolved config using discovered staged paths and actual file
   hashes; public split/static paths are already filled in the tracked profiles.
2. Run the pinned unmodified B1/B4 parents through shared inference/evaluation on
   validation and save genuine baseline evidence. This can be done here; it does
   not depend on another member agreeing to a different model.
3. Run strict CTC target auditing. Digits/unknown characters are reported, never
   deleted or guessed. Document any later target/vocabulary experiment separately.
4. Recheck training resources with permitted real train examples, then run LoRA,
   full Whisper and CTC under the owner-approved budgets. Save every checkpoint.
5. Train RDL from train tables only and use isolated validation scoring to choose
   checkpoints and fusion. Use RelationalAsrAdapter through shared run_timed for
   combined inference timing; untimed offline rescoring is not an ASR RTF result.
6. Freeze real validation selection evidence; perform final test comparisons once.

All software decisions can be amended here under owner authority and recorded.
Missing data and unmeasured evidence are the remaining blockers, not inter-member
approval. Main-branch/independent PR review is not represented as completed.

The minimal baseline gate report identifies `status=completed`,
`training_mode=pretrained`, `split=validation`, exact `model_id`/`revision`,
`dataset_release` and `split_manifest_sha256`. It is derived from actual baseline
artifacts; do not create that report merely to clear a configuration check.
