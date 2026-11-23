# Benchmark outputs and reports

This directory holds reviewed aggregate metrics, validation selection records,
audit reports, and the frozen M4/M5 final test study for `turas-tr-econ-legis-v0.1`
(636 test clips across 20 evaluated system/ablation conditions).

## Committed result artifacts

| Path | Contents |
| --- | --- |
| `final/M4_M5_FINAL_REPORT.md` | Narrative final report for B1, B4, F1, F2, F3, and R1 ablations |
| `final/m45_metrics.csv` | Machine-readable test metrics table across all 20 systems |
| `final/m45_test_summary.json` | Aggregate test metrics, S/D/I edit counts, bootstrap CIs, and repeated timing |
| `final/m45_technical_closeout.json` | Technical closeout, PR heads, CI status, and restored-checkpoint checks |
| `final/m45_final_integrity_audit.json` | Post-freeze candidate and train/test overlap integrity audit |
| `final/m45_decoding_provenance_correction.json` | Decoder provenance correction (Whisper beam 5, CTC prefix width 10) |
| `final/m45_restored_delivery_acceptance.json` | Restored checkpoint delivery acceptance record |
| `final/m45_handoff_manifest.json` | Hashed external handoff manifest |
| `evaluation/` | Validation baselines, checkpoint selection, RDL grid, corrected-linker ablations, and independent audit JSONs |
| `training/m45_seed17_completion.json` | Seed-17 acoustic training completion summary (F1, F2, F3) |
| `rdl/m45_seed17_internal.json` | Seed-17 heterogeneous RDL graph training summary |
| `final/m3_test_summary.json` | B5 BiLSTM-CTC and B6 single-Gaussian HMM aggregate test results |
| `final/BASELINE_COMPLETION_REPORT.md`, `final/baseline_completion_metrics.csv`, `final/baseline_completion_summary.json` | B7 GMM-HMM, B8 Seda RNN-T and B2/B3/B5 rerun results with counts, intervals and hashes |
| `final/completion_hardware_context.json`, `final/fresh_environment_restoration.json`, `final/final_review_checks.json` | Follow-up hardware context, fresh-environment restore record and review checks |
| [`m6/`](m6/README.md) | B2 and B3 validation and test scores and same-session timing, outside the frozen protocol |
| [`m3/`](m3/README.md) | B5 BiLSTM-CTC and single-Gaussian HMM: checkpoints, validation/test predictions and N-best, metrics, timing, training history |

Full per-utterance prediction/N-best JSONLs, model checkpoints, and private
references are external hashed artifacts (`results/predictions/` and
`results/nbest/` are gitignored). B2 and B3 are scored in `m6/`, not in the
frozen test tables. B5 and the single-Gaussian HMM (B6) are complete, with outputs
in `m3/`; B7 and B8 are in `final/BASELINE_COMPLETION_REPORT.md`.
