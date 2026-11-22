# B2 and B3 baseline results (M6 scorer)

Aggregate scores and timing for B2 (Whisper-large-v3) and B3 (Qwen3-ASR-1.7B) on `turas-tr-econ-legis-v0.1`,
outside the frozen M4/M5 protocol, with input hashes and no per-utterance text. The M6 scorer,
`scripts/m6_score_baselines.py` (PR #23), makes the same library calls as `scripts/m45_final_evaluate.py`; it is
not pinned by the protocol, and its B1 parity check is pending.

| Path | Contents |
| --- | --- |
| [`test/`](test/) `<SYSTEM_ID>.json`, `summary.json` | Test scores (636 utterances, 28 recording groups), paired intervals, one-best agreement |
| [`validation/`](validation/) `<SYSTEM_ID>.json`, `summary.json` | Validation scores (438 utterances, 15 recording groups) used to select B3 |
| [`timing/c2_timing_summary.json`](timing/c2_timing_summary.json) | Same-session end-to-end RTF for B1, B2 and B3 |

## Systems and references

- **B2** (`B2-whisper-large-v3`): `openai/whisper-large-v3` at `06f233fe`, bf16, beam 5, N 5, the same decoding as
  B1. The official B2 run is M6's RTX 5070 Ti laptop decode, decided and recorded before the test references were
  opened. Member 2's decode (`B2-whisper-large-v3-m2repro`, handoff zip `fd8471e9`) is an independent reproduction.
- **B3** (`B3-qwen3-asr-1.7b`): `Qwen/Qwen3-ASR-1.7B-hf` at `bcd2b5b7`, bf16, beam 5, N 5, selected on validation
  over the 0.6B candidate (`B3cand-qwen3-asr-0.6b`) by lower WER, with RTF only as a tie-break. Its weights are
  byte-identical (707/707 tensors) to the original `Qwen/Qwen3-ASR-1.7B` at `7278e1e`, released 2026-01-28.
- **References:** test uses the M1 test archive payload `31e372f0` (the protocol's `test_reference.payload_sha256`),
  opened for M6 scoring at 2026-11-22T12:33:41Z, after all B2/B3 decoding and timing. Validation uses
  `owner-recovered-validation-minutes-v1` (`d0c8519c`), the reference used for the M4/M5 validation numbers.

## Test (WER and CER in %, term F1 from 0 to 1)

| System | Raw WER / CER | Normalized WER / CER | Term F1 micro / macro | M1-verified subset (25) normalized WER / CER |
| --- | --- | --- | --- | --- |
| B2 (official) | 24.5443 / 8.0222 | 12.3192 / 5.4473 | 0.9776 / 0.9408 | 12.1864 / 5.2614 |
| B2, Member 2 reproduction | 24.6039 / 8.0135 | 12.3922 / 5.4391 | 0.9773 / 0.9406 | 12.3059 / 5.2951 |
| B3 | 27.5937 / 7.8680 | 16.0943 / 5.6005 | 0.9123 / 0.8181 | 16.7264 / 4.8567 |

Paired recording-group bootstrap (seed 2026, 2000 replicates), normalized WER difference in percentage points
[95% interval]: reproduction - B2 = +0.0731 [-0.0532, 0.2277], with 577 of 636 one-best hypotheses byte-equal;
B3 - B2 = +3.7751 [2.6238, 4.9312]. Each interval is a per-comparison 95% interval, with no correction for multiple
comparisons. Validation (raw WER, normalized WER, term F1 micro): B2 23.8328, 12.9437,
0.9756; reproduction 23.7565, 12.9064, 0.9753; B3 26.5563, 15.4323, 0.9173; 0.6B candidate 33.9564, 22.4722,
0.8280; reproduction - B2 = -0.0374 [-0.1274, 0.0642]; B3 - B2 = +2.4886 [1.0738, 3.7816].

The test references were kept by CER <= 0.25 agreement with an upstream ASR transcript, whose `tr_tr` model is not
confirmed (`whisper-large-v3-turbo` is the WorldSpeech paper's most common alignment model); this filter can favour
Whisper-family systems such as B2 over B3 (`data/metadata/turas-tr-econ-legis-v0.1.json` line 83,
`experiments/m2-baseline-notes.md` line 45).

On test, zero-shot B2 (12.32 normalized WER) is level with F1 LoRA (12.37); F2 full fine-tuning (10.37) is lower
by about 2 pp. F1 is better on raw WER (21.50 vs 24.54) and runs at about B1's RTF, about 3x faster than B2.
These comparisons are unpaired and cross scorers: B2 is scored by the M6 scorer, whose B1 parity check is pending,
and F1/F2 by the frozen scorer. There is no paired interval against F1 or F2 yet: their per-utterance outputs are
in the private M4/M5 handoff bundle, which M6 has not received. 611 of the 636 test references are unreviewed
edited TBMM minutes, whose style fine-tuned F1 and F2 learn.

## Timing and caveats

One session on 2026-11-22 on an RTX 5070 Ti laptop GPU (12 GB) with the M6 timing helper
`scripts/m6_b23_harness_run.py` (PR #23) and the frozen timing block settings: 3 repeats, batch 1, 3 warm-up,
107 utterances, 1,813.765 s of audio. End-to-end RTF: B1 0.05757, B2 0.17537 (3.05x B1), B3 0.13843 (2.40x B1).
B1's per-repeat spread was 8.7% (repeat 3 slower).

- Laptop timing: these RTFs are not comparable to the frozen-block RTFs in `results/final`; compare them with
  frozen systems only through the B1 ratio.
- B1 parity of the M6 scorer and the paired intervals against F1/F2 are pending the private M4/M5 bundle.
- 8 of the 28 test recordings (176 of 636 clips, sessions 6–22 January 2026) and all validation sessions
  (2025-10-01 to 2025-12-24), on which B3 was selected, predate the 2026-01-28 Qwen3-ASR release, so they may be in
  its training data. The other 20 test recordings (460 clips) are later. Whisper-large-v3 (B2) predates all data.
  Session dates: `data/metadata/turas-tr-econ-legis-v0.1-source-registry.csv` with
  `data/metadata/m45-test-recording-groups.csv`.

Scoring command (run folders hold `predictions.jsonl`, `nbest.jsonl` and `manifest.json`; for validation, pass
`--split validation`, the validation references and their hash, and the 0.6B candidate run):

    python scripts/m6_score_baselines.py --protocol configs/evaluation/m45_final_protocol.json --protocol-sha256 <PROTOCOL_SHA256> --split test --references <REFERENCES_JSONL> --references-sha256 <REFERENCES_SHA256> --run B2-whisper-large-v3=<B2_RUN_DIR> --run B2-whisper-large-v3-m2repro=<M2_RUN_DIR> --run B3-qwen3-asr-1.7b=<B3_RUN_DIR> --pair B2-whisper-large-v3-m2repro=B2-whisper-large-v3 --pair B3-qwen3-asr-1.7b=B2-whisper-large-v3 --output <NEW_OUTPUT_DIR>
