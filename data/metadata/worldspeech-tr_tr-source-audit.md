# WorldSpeech `tr_tr` source audit (Member 1)

Audit date: 2026-11-18. Auditor: Member 1. Method: dataset card plus a full
read of every non-audio column of every `tr_tr` shard at the pinned revision
(`scripts/m1_build_release.py`, cached metadata hash in the split manifest).
Numbers below are **measured** on that metadata unless marked otherwise.

## Identity, access and rights

| Item | Value |
| --- | --- |
| Dataset | [`disco-eth/WorldSpeech`](https://huggingface.co/datasets/disco-eth/WorldSpeech), config `tr_tr` |
| Pinned revision | `b76b85395039f50332fbc421f7c098ee2f2653d7` (last modified 2026-09-25) |
| Access | Public, not gated; parquet shards `data/tr_tr/{train,test}-NNNNN.parquet` |
| Files / size | 365 train shards (24.47 GB) + 24 test shards (1.29 GB) |
| Paper | WorldSpeech, arXiv:2605.09167 (NeurIPS 2026) |
| Dataset licence | CC BY-NC 4.0 (non-commercial; attribution required) |
| `tr_tr` source terms | Card lists "Parliamentary Proceedings", <https://www.tbmm.gov.tr/>. Where a source licence is more restrictive, it governs |
| Source media | Every `source_url` is a YouTube video (TBMM channel uploads) |

Rights conclusion (for team approval, D02/D16): course research use is
non-commercial and fits CC BY-NC 4.0 with attribution to WorldSpeech and TBMM.
The card does not grant redistribution of the audio beyond its own terms, and
the TURAS repository is **public**, so audio, reference text and the exclusion
ledger stay out of Git; only opaque IDs, hashes, counts and source URLs are
committed. Rights record ID used in manifests: `rights_worldspeech_tr_tr_tbmm`.

## Fields

`audio` (OGG Opus bytes), `human_transcript`, `asr_transcript`, `cer`, `snr`,
`dnsmos_sig`, `dnsmos_bak`, `dnsmos_ovr`, `dnsmos_p808`, `duration`, `source`,
`source_url`, `source_start_s`, `source_end_s`, `session_date`, `segment_id`,
`language`, `country`. There is no speaker field.

Measured audio: OGG Opus, **24 kHz, 2 channels** (stereo), decoded duration
matches `duration`. TURAS converts to 16 kHz mono FLAC (channel mean, soxr HQ)
as required by data contract 0.2.

## Size and coverage

| Upstream split | Rows | Hours | Recordings (`source_url`) |
| --- | ---: | ---: | ---: |
| train | 227,669 | 957.25 | 197 |
| test | 11,985 | 50.43 | 188 |

- Sources: `tbmm_parliament` 237,203 rows; `ankara_buyuksehir_belediyesi`
  2,451 rows (Ankara metropolitan council, not parliament; excluded).
- Session dates 2017-01-10 to 2026-04-22; one recording per date. Segment
  prefixes encode term/legislative year/sitting (e.g. `tr_28_3_1` = 28th term,
  year 3, sitting 1); terms 26 and 28 appear.
- Segment durations 10.0–20.0 s (mean 15.1 s). Upstream CER (ASR vs. human)
  mean 0.075, max 0.50.
- Segment IDs are unique; no two segments of one recording overlap in time.

## Findings that shape the TURAS release

1. **The upstream train/test split leaks at recording level.** 100% of upstream
   test rows come from recordings that also have upstream train rows (188 of
   188 test recordings). Adjacent segments share speaker, topic and acoustics,
   so the upstream test is not an independent test set. TURAS therefore defines
   a new recording-grouped split; upstream test rows never enter TURAS train
   (1,069 such rows inside TURAS train recordings are quarantined and logged).
2. **`human_transcript` is the stenographic minutes, not a verbatim
   transcript.** 7.1% of parliamentary rows contain non-spoken material:
   speaker headers (`BAŞKAN -`, `AD SOYAD (İl) –`), stage directions
   (`(Alkışlar)`, `(CHP sıralarından alkışlar)`, `(Devamla)`), constituency
   names in parentheses, brackets, and truncation ellipses. These rows are
   excluded with reason `non_spoken_or_truncated_text`. Minutes are also edited
   (disfluencies and repetitions removed), so remaining references still need
   listening verification before validation/test freeze.
3. **Formulaic repeats.** 3,175 rows share identical screening-normalized text
   with another row (e.g. vote formulas). All copies are excluded
   (`repeated_text`) so they cannot inflate test scores or leak across splits.
4. **`asr_transcript` and `cer` are an upstream ASR alignment by-product.**
   They are never references. TURAS uses `cer <= 0.25` only as an alignment
   sanity filter (3,886 rows removed). This filter biases the release toward
   segments an ASR system already handled reasonably; that bias is a declared
   limitation.
5. **Domain coverage exists.** With the TURAS lexicon v0.1, 126.6 h of clean
   parliamentary speech meets the economic/legislative rule across 172
   recordings; December budget debates (2024-12, 2025-12) are the densest.
6. **No speaker identity.** Speaker-disjointness cannot be claimed; the same
   ministers and MPs very likely speak in train and test recordings.
7. **Possible pretraining contamination.** The audio is public on YouTube and
   the minutes are public on tbmm.gov.tr. TURAS test uses only 2026 sessions,
   which postdate the release of the planned baseline checkpoints; M2/M3 must
   confirm each checkpoint's release date.

## Open items for the team

- Approve the rights conclusion above (D02/D16).
- Approve the topic scope and lexicon (D03) and the split policy (D07).
- Complete listening verification of validation/test (D08); the pilot sample
  is described in [the verification guide](../verification/turas-verbatim-v0.1.md).
