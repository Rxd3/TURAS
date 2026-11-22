# Raw data: WorldSpeech `tr_tr` text and metadata

The unmodified upstream rows that TURAS cleans: every one of the 239,654 rows of
[`disco-eth/WorldSpeech`](https://huggingface.co/datasets/disco-eth/WorldSpeech)
config `tr_tr` at revision `b76b85395039f50332fbc421f7c098ee2f2653d7`
(about 1,008 h of Turkish Grand National Assembly sessions, 2017–2026).
License: CC BY-NC 4.0; source recordings from <https://www.tbmm.gov.tr/>.

| File | Rows |
| --- | ---: |
| `metadata-00.parquet` … `metadata-03.parquet` | 239,654 in total (60,000 per file, last file 59,654) |

Columns are the upstream columns except `audio`: `human_transcript` (the official
minutes), `asr_transcript`, `cer`, `snr`, `dnsmos_sig`, `dnsmos_bak`, `dnsmos_ovr`,
`dnsmos_p808`, `duration`, `source`, `source_url`, `source_start_s`,
`source_end_s`, `session_date`, `segment_id`, `language`, `country`, plus
`upstream_split`, `shard` and `row_in_shard` (where the row sits upstream).

Written by [`scripts/m1_export_raw.py`](../../../scripts/m1_export_raw.py), which reads
the upstream parquet shards without the audio column.

## From raw to cleaned

The same script runs the release's screening and selection code
([`turkish_asr_rdl.data.selection`](../../../src/turkish_asr_rdl/data/selection.py))
on these rows and writes the per-row removal log
[`data/cleaned/turas-tr-econ-legis-v0.1/exclusion_ledger.jsonl.gz`](../../cleaned/turas-tr-econ-legis-v0.1/exclusion_ledger.jsonl.gz).
It stops unless the result reproduces the frozen release exactly: the same
exclusion counts, the same utterance IDs in every split, and the ledger SHA-256
recorded in the [split manifest](../../splits/turas-tr-econ-legis-v0.1/split_manifest.json)
(`6077dc6f…c0c1`). It did on 2026-11-22.

The cleaned references are the selected rows' `human_transcript`, except for 13
clips (3 train, 10 test) that the team corrected after listening; those are marked
`verification_status: verified`. The cleaned audio is the upstream audio of the
selected rows (OGG Opus, 24 kHz stereo) converted to 16 kHz mono FLAC by
[`scripts/m1_build_release.py`](../../../scripts/m1_build_release.py). The upstream
audio itself is not copied here: it sits in the same 25 GB of shards, so only the
converted clips are committed, in [`data/cleaned/`](../../cleaned/turas-tr-econ-legis-v0.1/README.md).
