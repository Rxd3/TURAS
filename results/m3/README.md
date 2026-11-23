# B5 BiLSTM-CTC and single-Gaussian HMM (Member 3)

Final outputs of the two baselines Member 3 trained from scratch on the frozen
train split of `turas-tr-econ-legis-v0.1`, copied unchanged from the
`TURAS-M3-final-2026-11-22.zip` delivery. The aggregate summary of this delivery
is byte-identical to [`results/final/m3_test_summary.json`](../final/m3_test_summary.json),
and the scripts that produced it are in [`scripts/`](../../scripts/)
(`m3_resume_b5_gpu.py`, `m3_finalize_b5.py`, `m3_run_gaussian_hmm.py`,
`m3_summarize_final.py`).

| System | Test normalized WER | Test normalized CER | Term micro F1 | End-to-end RTF |
| --- | ---: | ---: | ---: | ---: |
| B5 BiLSTM-CTC (epoch 22, selected on validation) | 47.78% | 13.50% | 0.891 | 0.0093 |
| B6 single-Gaussian character HMM | 100.00% | 88.76% | 0.000 | 0.0017 |

| Path | Contents |
| --- | --- |
| `b5/checkpoint/` | `bilstm_ctc.pt` (selected weights), `vocab.json`, checkpoint manifest |
| `b5/training/` | Training run manifest, history and parent validation record |
| `gaussian-hmm/gaussian_hmm_character.npz` | Trained HMM parameters, with `training-evidence.json` and `training-history.json` |
| `*/validation/`, `*/test/` | Predictions, N-best lists (B5), per-utterance timing and completion manifests |
| `*/timing/` | The 30-minute timing subset run |
| `*/metrics.*.json`, `*/final-report.json` | Metrics, edit counts, confidence intervals and limitations |
| `RESULTS.txt` | Short plain-text summary |

The B5 checkpoint was selected on validation (epoch 22, early stop at 28) before
test scoring. The HMM uses one Gaussian per character state and is a weak
classical reference, not the 4-mixture MFCC GMM-HMM (B7).
