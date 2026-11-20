# TURAS M4/M5 final working benchmark

Frozen protocol SHA-256: `288aebd71d8a3574e13d436748d4161bc258a487e65066f1c3508bc8e835bec1`.
Test: 636 clips, 28 recording groups; reference `m1-test-minutes-archive-01ad5878-v1`.
The model choices and all conditions were frozen before test transcript access. All results are one training seed (17).

## Main acoustic comparisons

| System | Raw WER | Normalized WER | Normalized CER | Term micro F1 | End-to-end RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1-whisper-small | 37.209% | 24.512% | 10.079% | 92.175% | 0.0540 |
| B4-xlsr-tr-cv7 | 56.568% | 37.152% | 13.167% | 83.364% | 0.1281 |
| F1-whisper-lora | 21.505% | 12.368% | 4.149% | 96.843% | 0.0532 |
| F2-whisper-full | 18.883% | 10.371% | 3.691% | 98.080% | 0.0530 |
| F3-ctc-vocab | 45.180% | 25.150% | 7.582% | 94.233% | 0.1239 |

CTC adaptation includes an explicit train-only vocabulary extension (38 to 52 tokens). The gain is not attributed to fine-tuning alone.

## Fixed-candidate RDL and ablations

| Adapted ASR | Original one-best WER | Acoustic reordering | Learned full graph | Without quantities | Without co-mentions | Untrained graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1-whisper-lora | 12.368% | 12.363% | 12.373% | 12.363% | 12.324% | 12.348% |
| F2-whisper-full | 10.371% | 10.395% | 10.429% | 10.419% | 10.410% | 10.405% |
| F3-ctc-vocab | 25.150% | 25.194% | 25.189% | 25.208% | 25.194% | 25.194% |

All graph/control rows use the same native N-best file per acoustic model. Positive RDL weights were selected on validation, with the same positive weight used in its relation-removal and untrained controls. Gamma zero is a separate acoustic reordering control. The primary model choices remain the validation-selected original decoder outputs; no final model was chosen from test scores.
Cached candidate selections produce contract-valid predictions and component scores. Their measured cache-only latency is excluded from headline ASR RTF. Joint learned-RDL timing is measured separately with fresh acoustic inference.

## Repeated timing

Batch 1, three repeats, 107 frozen timing clips totaling 1813.765 seconds per repeat, across 28 recording groups. Three train warm-up clips are excluded. OS caches were not flushed; inputs and weights were local. Accuracy runs may be concurrent and are excluded from headline speed claims.

| System | End-to-end RTF | RTFx | Per-repeat RTF | p50 / p95 request seconds | Peak reserved GiB | Thermal flags |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| F2-whisper-full--rdl-full | 0.0538 | 18.59 | 0.0538, 0.0538, 0.0538 | 0.890 / 1.269 | 1.164 | 0 |
| F3-ctc-vocab--rdl-full | 0.1245 | 8.03 | 0.1241, 0.1247, 0.1247 | 2.075 / 3.052 | 1.152 | 0 |
| F2-whisper-full | 0.0530 | 18.88 | 0.0535, 0.0528, 0.0526 | 0.875 / 1.252 | 1.164 | 0 |
| B1-whisper-small | 0.0540 | 18.52 | 0.0543, 0.0539, 0.0538 | 0.892 / 1.269 | 1.225 | 0 |
| F1-whisper-lora--rdl-full | 0.0532 | 18.78 | 0.0534, 0.0533, 0.0530 | 0.880 / 1.239 | 1.203 | 0 |
| F3-ctc-vocab | 0.1239 | 8.07 | 0.1238, 0.1240, 0.1241 | 2.083 / 3.007 | 1.152 | 0 |
| F1-whisper-lora | 0.0532 | 18.80 | 0.0529, 0.0531, 0.0535 | 0.878 / 1.245 | 1.203 | 0 |
| B4-xlsr-tr-cv7 | 0.1281 | 7.80 | 0.1123, 0.1218, 0.1503 | 2.112 / 3.412 | 1.152 | 0 |

Any spill/thermal flags are retained in the JSON; timings are hardware-specific. Cold-load time and secondary core-model RTF are stored per system.

## Paired uncertainty

Recording-group percentile bootstrap: 2000 paired replicates, seed 2026; 28 groups. Differences below are normalized WER percentage points, system minus control.

| System | Control | Difference | 95% interval |
| --- | --- | ---: | --- |
| F1-whisper-lora | B1-whisper-small | -12.144 | [-13.463, -10.774] |
| F2-whisper-full | B1-whisper-small | -14.141 | [-15.422, -12.818] |
| F3-ctc-vocab | B4-xlsr-tr-cv7 | -12.003 | [-13.129, -10.900] |
| F2-whisper-full | F1-whisper-lora | -1.997 | [-2.420, -1.542] |
| F1-whisper-lora--rdl-full | F1-whisper-lora--acoustic-control | 0.010 | [-0.039, 0.058] |
| F1-whisper-lora--rdl-full | F1-whisper-lora | 0.005 | [-0.108, 0.112] |
| F1-whisper-lora--rdl-no-quantities | F1-whisper-lora--rdl-full | -0.010 | [-0.039, 0.015] |
| F1-whisper-lora--rdl-no-comention | F1-whisper-lora--rdl-full | -0.049 | [-0.098, -0.010] |
| F1-whisper-lora--rdl-random | F1-whisper-lora--rdl-full | -0.024 | [-0.083, 0.029] |
| F2-whisper-full--rdl-full | F2-whisper-full--acoustic-control | 0.034 | [0.000, 0.073] |
| F2-whisper-full--rdl-full | F2-whisper-full | 0.058 | [-0.034, 0.161] |
| F2-whisper-full--rdl-no-quantities | F2-whisper-full--rdl-full | -0.010 | [-0.039, 0.015] |
| F2-whisper-full--rdl-no-comention | F2-whisper-full--rdl-full | -0.019 | [-0.049, 0.010] |
| F2-whisper-full--rdl-random | F2-whisper-full--rdl-full | -0.024 | [-0.049, -0.005] |
| F3-ctc-vocab--rdl-full | F3-ctc-vocab--acoustic-control | -0.005 | [-0.034, 0.024] |
| F3-ctc-vocab--rdl-full | F3-ctc-vocab | 0.039 | [-0.044, 0.130] |
| F3-ctc-vocab--rdl-no-quantities | F3-ctc-vocab--rdl-full | 0.019 | [0.000, 0.044] |
| F3-ctc-vocab--rdl-no-comention | F3-ctc-vocab--rdl-full | 0.005 | [-0.024, 0.039] |
| F3-ctc-vocab--rdl-random | F3-ctc-vocab--rdl-full | 0.005 | [-0.024, 0.034] |

## Reference quality and limitations

M1-reported status counts: `{"unreviewed": 611, "verified": 25}`. Every system is also scored on the M1-reported listening-verified subset in the JSON, with its actual support. That small subset is not a substitute for the full test set.

- Working benchmark on edited parliamentary minutes, not fully verified verbatim speech
- M1 reported pilot listening corrections; no independent M6 release approval is claimed
- Unknown speakers: recording-disjoint splits do not prove speaker disjointness
- Single training seed17; no training-seed uncertainty or robustness claim
- Validation used recovered upstream minutes without three private M1 corrections
- Quantities are automatic digit-based amount/percentage diagnostics, not human quantity annotations
- This completes M4/M5 matched-parent work; other baseline-family tracks remain separate M2/M3 responsibilities
- Diagnostic relation-removal studies were chosen after initial validation results; final test conditions were frozen before gold access

## Reproduction and handoff

The exact checkpoints, parents, graph/linker hashes, timing IDs, reference hashes and comparison conditions are in `configs/evaluation/m45_final_protocol.json`. The run index links every prediction/N-best/timing and score artifact. Environment versions are pinned in `requirements/owner-m45-windows-cu130.txt`; install that lock and the editable package in a fresh compatible environment. Run the included commands from the repository root and use fresh output folders.
Private model/run artifacts are kept outside Git. The private handoff includes selected weights/processors, corrected graphs/control, all prediction/timing artifacts and an exact source Git bundle. It excludes reference transcripts and audio. Obtain the separately hashed M1 audio bundle and evaluator-only references through their proper private channel.
No independent reviewer approval, consumer acknowledgment or completion of other members' baseline tracks is asserted.
