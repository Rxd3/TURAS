# M6 reproductions and number-invariant view (2026-11-23)

Member 6 (evaluation and QA) re-ran four reported results on a second machine and
added a scoring view in which numbers written as digits or as words count the same.
This folder holds aggregates and hashes only, no per-utterance text.

## Status

- **Post-freeze.** Every run here was made on 2026-11-23, after the test references
  had been opened for scoring ([reference access](../provenance/reference_access.json)).
  No reported result was re-selected from these runs.
- **Reported separately.** The frozen M4/M5 results ([final report](../../final/M4_M5_FINAL_REPORT.md))
  and the baseline numbers in the [README](../../../README.md) stay the main results.
  The numbers below are reproductions of them, not replacements.
- **Machine.** M6's Windows laptop with an NVIDIA RTX 5070 Ti laptop GPU (12 GB). B8 ran on CPU.
- **Code.** F2 and B1 validation ran from `main` at `ee1f3e4` with the tools in
  [`tools/m6-repro/`](../../../tools/m6-repro/README.md); B8 ran with PR #26's own scripts at `dd7654b`.
- **References.** Test `31e372f0e35f504de59b49f1e0e1f3ed6fcc2772138c591a2cf2b054a4e7ef3f`
  (636 utterances, 28 recording groups); validation
  `d0c8519ca6aed053f89786bc733541a70faee09cf8d8bdb6cc12065d97dbd958` (438 utterances,
  15 groups). M6 checked by SHA-256 that the published references in
  `data/cleaned/turas-tr-econ-legis-v0.1/` are byte-identical to these scored references.

## What was run

| Check | System | What M6 did | Files |
| --- | --- | --- | --- |
| Re-train | F2, Whisper-small full fine-tune, seed 17 (the frozen seed) | Trained from `configs/fine-tuning/whisper_small_full.json`, decoded validation after each of 3 epochs, selected the epoch on validation, then decoded and scored test once | [`f2-seed17/`](f2-seed17/) |
| Re-train | F2, seed 18 (variance check, seed chosen in advance) | Same recipe and steps with seed 18 | [`f2-seed18/`](f2-seed18/) |
| Re-decode | B1, `openai/whisper-small` zero-shot | Validation decode by the `baseline` step of `tools/m6-repro/f2repro.ps1` (`scripts/m45_validation_baseline.py`); test decode with `scripts/m6_b23_harness_run.py`; both scored with `scripts/m6_score_baselines.py` | [`b1-whisper-small-m6/`](b1-whisper-small-m6/) |
| Re-score | B5, BiLSTM-CTC | Re-scored PR #26's committed run folders with `scripts/m6_score_baselines.py` | [`b5-rescore/`](b5-rescore/) |
| Re-decode | B8, Seda Zipformer2 RNN-T (`atasoglu/seda-v0.1` @ `5cbbda1b`) | Re-ran PR #26's own scripts at `dd7654b` on CPU and hashed the exports | hashes below |
| New view | B1, B2, Member 2's B2 reproduction, B3, B5, B6, B8 and the F2 seed 17 re-train | Scored with `scripts/m6_number_invariant_wer.py` | [`number-invariant/`](number-invariant/) |

F3, F1 and F2 seed 19 did not run before the time limit.

WER and CER are in %. "Frozen" is the committed result; differences are M6 minus frozen,
in percentage points.

## F2 re-train

Seed 17 validation per epoch, normalized WER / CER: epoch 1 11.0754 / 3.6750, epoch 2
10.2235 / 3.5015, epoch 3 10.3430 / 3.6449. The config's rule (lowest validation normalized
WER; raw WER, then the earlier epoch break ties) selected epoch 2, the same epoch as the frozen run.
For seed 18 the same rule selected epoch 3 (validation normalized WER 10.3430), so the selected
epoch differs by seed (2 against 3).

| Split | Metric | Frozen | M6, seed 17 | M6, seed 18 | Mean of 17 and 18 | Sample SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Test | Normalized WER | 10.3707 | 10.4681 | 10.6045 | 10.5363 | 0.0964 |
| Test | Raw WER | 18.8825 | 19.0961 | 19.6623 | 19.3792 | 0.4003 |
| Test | Normalized CER | 3.6910 | 3.7341 | 4.1034 | 3.9188 | 0.2611 |
| Test | Raw CER | 5.1263 | 5.1824 | 5.6355 | 5.4089 | 0.3204 |
| Validation | Normalized WER | 10.1936 | 10.2235 | 10.3430 | 10.2832 | 0.0846 |
| Validation | Raw WER | 18.4468 | 18.6070 | 18.6604 | 18.6337 | 0.0378 |
| Validation | Normalized CER | 3.4475 | 3.5015 | 3.5867 | 3.5441 | 0.0602 |
| Validation | Raw CER | 4.8144 | 4.9028 | 4.9506 | 4.9267 | 0.0338 |

Frozen validation CERs are from
[`m45_working_checkpoint_selection.json`](../../evaluation/m45_working_checkpoint_selection.json).
Against the frozen test normalized WER, seed 17 differs by +0.0974 pp and seed 18 by +0.2338 pp.

Seed spread, test normalized WER: 10.4681 and 10.6045, mean 10.5363, difference 0.1364 pp.
There are two seeds only, so the SD is a rough indication. Both seeds land within 0.24 pp of
the frozen 10.37%.

Test N-best oracle (normalized): seed 17 one-best 10.4681, oracle 9.0263 (headroom 1.4419 pp;
CER 3.7341 to 3.2615), oracle beats rank 1 on 229 of 636 utterances; seed 18 one-best 10.6045,
oracle 9.0214, oracle beats rank 1 on 240 of 636.
Seed 17 training took 30.2 minutes and the whole job 64.6 minutes. Seed 18 training took
56.3 minutes, of which about 26 minutes were lost to a GPU power cap set by the laptop's power
mode and lifted mid-run; this does not affect the results. These times are recorded by M6 in
its run log; no committed file holds them.

## B1 re-decode

| Split | Metric | Frozen | M6 | Difference |
| --- | --- | ---: | ---: | ---: |
| Test | Normalized WER | 24.5117 | 24.5506 | +0.0390 |
| Test | Raw WER | 37.2088 | 37.2287 | +0.0199 |
| Test | Normalized CER | 10.0786 | 10.0458 | −0.0328 |
| Test | Raw CER | 12.8708 | 12.8301 | −0.0407 |
| Validation | Normalized WER | 25.1252 | 25.1177 | −0.0075 |

Decode plus M6 scorer on a second machine gives the frozen B1 normalized WER within
0.04 pp on test and validation; CER differs by up to 0.041 pp. Scorer parity on the
identical frozen B1 predictions is still pending: those predictions are only in the private
M4/M5 bundle, which M6 has not received.
Test N-best oracle (normalized): 24.5506 to 22.3245 (headroom 2.2261 pp); CER 10.0458 to
9.2621; the oracle beats rank 1 on 270 of 636 utterances.

## B5 re-score

All eight numbers equal the committed ones in
[`m3_test_summary.json`](../../final/m3_test_summary.json) (difference 0.0):

| Split | Raw WER / CER | Normalized WER / CER |
| --- | --- | --- |
| Test | 64.0030 / 17.5653 | 47.7763 / 13.5029 |
| Validation | 63.9381 / 17.0451 | 47.3881 / 13.2578 |

B5's N-best lists hold one candidate per clip, so its oracle equals its one-best.

## B8 re-decode

`predictions.jsonl` from M6's CPU run is byte-identical to the committed export hashes in
[`baseline_completion_summary.json`](../../final/baseline_completion_summary.json):

| Split | Rows | `predictions.jsonl` SHA-256 |
| --- | ---: | --- |
| Test | 636 | `f1ee00943d6dada99d71e8dd5dc1aab7a2b88c739ff68cc8b69c054f3d97953a` |
| Validation | 438 | `5c43a368f3641143749465e9968083699b6b5e3fa983cfa7d1bdd105faa7417f` |

`model_artifact_sha256` `5b85df09f99df94cc5f44302d53dfb580113bcc6666caf40a0ed502612a953a2` and
`decoder_config_sha256` `5d6fa550005873731243de3471b9c88431e8034497fc35bd01c1187a7f45f2db`
match. M6 recorded 0 error rows and 5 min 23 s for the test run; no per-utterance files are
added here.

## Number-invariant view

`tr-number-invariant-v0` is a report view outside the frozen protocol. Before
`tr-normalized-v1`, numbers written with digits become Turkish words, so `15` and
`on beş` count the same, and `%5` and `yüzde beş` count the same. The script's docstring
lists exactly what it reads and what it leaves alone (dates, times, ordinals and others).
Its `tr-normalized-v1` output equals the committed scores exactly. Test set:

| System | WER, tr-normalized-v1 | WER, number-invariant | CER, tr-normalized-v1 | CER, number-invariant |
| --- | ---: | ---: | ---: | ---: |
| B1 Whisper-small (M6 re-decode) | 24.5506 | 16.5237 | 10.0458 | 5.3316 |
| F2 Whisper-small full fine-tune (M6 re-train, seed 17) | 10.4681 | 9.6826 | 3.7341 | 3.5075 |
| F2 Whisper-small full fine-tune (M6 re-train, seed 18) | 10.6045 | 9.8256 | 4.1034 | 3.8787 |
| B2 Whisper-large-v3 | 12.3192 | 8.9399 | 5.4473 | 3.7954 |
| B2, Member 2's reproduction | 12.3922 | 8.9953 | 5.4391 | 3.7798 |
| B3 Qwen3-ASR-1.7B | 16.0943 | 14.1987 | 5.6005 | 4.9904 |
| B5 BiLSTM-CTC | 47.7763 | 46.0744 | 13.5029 | 13.7358 |
| B6 single-Gaussian character HMM | 99.9951 | 199.6955 | 88.7601 | 108.7102 |
| B8 Seda Zipformer2 RNN-T | 29.9576 | 20.0757 | 12.8115 | 7.0209 |

- From B1 to F2 (both M6 runs), WER falls by 14.08 pp (57% relative) in tr-normalized-v1
  but by 6.84 pp (41% relative) in the number-invariant view (seed 18: 6.70 pp). About half of the measured
  fine-tuning gain is learning how the minutes write numbers.
- In the number-invariant view, zero-shot Whisper-large-v3 (B2) has lower WER than
  fine-tuned Whisper-small (8.94 against 9.68). F2 keeps the lower CER (3.51 against 3.80).
- B6's output is mostly noise, and every digit it emits expands into words, so insertions
  grow. Its number-invariant score is not meaningful.
- Spelling out digits changes the reference word count (20,529 to 21,678 test words), so
  the two views have different denominators. There are no intervals for this view.

## Limitations

- Post-freeze, one machine. Seed 17 is the frozen seed, so its +0.0974 pp is run-to-run and
  machine difference, not seed variance. Seed 18 adds one more seed: two seeds only.
- F2 and B1 are compared across scorers: M6 numbers come from `scripts/m6_score_baselines.py`,
  frozen numbers from the frozen evaluator (the note in `f2-seed17/compare.json`).
- There are no paired intervals against the frozen runs: their per-utterance outputs are in
  the private M4/M5 bundle.
- The F2 test decode uses a protocol derived from the frozen one: input paths point to the
  published data and the system entry names the new checkpoint; the decoder block is unchanged.
- F1, F3 and F2 seed 19 were not run.
- The training and job times and the B8 run time and error count are recorded by M6 and have
  no committed file. The reference hashes can be recomputed from the committed files.

## Commands

The F2 runbook runs on Windows with a CUDA GPU. Copy `tools/m6-repro/f2repro.ps1` and
`f2repro.py` into `<ARTIFACT_ROOT>\m6-repro-f2\` and set up the layout in
[`tools/m6-repro/README.md`](../../../tools/m6-repro/README.md). Then, in PowerShell:

```text
$S = '<ARTIFACT_ROOT>\m6-repro-f2\f2repro.ps1'
& $S -Step preflight -Artifacts <ARTIFACT_ROOT>
& $S -Step baseline -Artifacts <ARTIFACT_ROOT>
& $S -Step smoke -Artifacts <ARTIFACT_ROOT>
& $S -Step start -Artifacts <ARTIFACT_ROOT> -Jobs 'F2:17,F2:18' -FinishBy <HH:MM>
& $S -Step status -Artifacts <ARTIFACT_ROOT>
& $S -Step compare -Artifacts <ARTIFACT_ROOT> -Jobs 'F2:17,F2:18'
```

`preflight` checks hashes, data and both Python environments; `baseline` decodes and scores
B1 on validation (the training gate); `smoke` runs a short training and partial decodes;
`start` launches the jobs in the background and skips a stage that would end after
`-FinishBy`; `status` prints progress; `compare` writes `compare.json`.

Number-invariant view (run folders hold `predictions.jsonl`; one `--run` per system):

```text
python scripts/m6_number_invariant_wer.py --references data/cleaned/turas-tr-econ-legis-v0.1/turas-tr-econ-legis-v0.1.references.test.jsonl --references-sha256 31e372f0e35f504de59b49f1e0e1f3ed6fcc2772138c591a2cf2b054a4e7ef3f --run baseline-whisper-small=<B1_RUN_DIR> --run B2-whisper-large-v3=<B2_RUN_DIR> --run B3-qwen3-asr-1.7b=<B3_RUN_DIR> --run baseline-seda-zipformer2-rnnt=<B8_RUN_DIR> --run B2-whisper-large-v3-m2repro=<M2_B2_RUN_DIR> --run baseline-bilstm-ctc=results/m3/b5/test --run baseline-m3-single-gaussian-character-hmm=results/m3/gaussian-hmm/test --output <NEW_OUTPUT_FILE>
```

The F2 entries (`number-invariant/test-number-invariant-F2s17.json` and `test-number-invariant-F2s18.json`) come from one run per seed (seed 18: `--run F2-m6repro-s18=<F2_S18_RUN_DIR>`):

```text
python scripts/m6_number_invariant_wer.py --references data/cleaned/turas-tr-econ-legis-v0.1/turas-tr-econ-legis-v0.1.references.test.jsonl --references-sha256 31e372f0e35f504de59b49f1e0e1f3ed6fcc2772138c591a2cf2b054a4e7ef3f --run F2-m6repro-s17=<F2_S17_RUN_DIR> --output <NEW_OUTPUT_FILE>
```

Scoring with `scripts/m6_score_baselines.py` follows the command in
[`results/m6/README.md`](../README.md).
