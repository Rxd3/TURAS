# Inference and prediction JSONL contract — 0.2 (proposed by M6, pending review by M1 and model owners M2–M5)

## Strict inference inputs

The allowed per-utterance input fields are `schema_version`, `dataset_release`,
`split_manifest_sha256`, `utterance_id`, `audio_uri`, `audio_sha256`, `duration_s`,
`sample_rate_hz`, `channels`, and `language`. IDs/hash fields identify artifacts;
they must not encode gold text. The adapter receives audio plus this allowlist,
not the curated manifest. It has no permission to read protected test references.

Topic/entity metadata derived from the held-out reference is forbidden. Any
external knowledge or source context must be available at actual inference time,
separately declared in the run manifest, and evaluated as a distinct condition.
No hidden prompt containing reference text is allowed. Predictions for every
frozen ID are exported in the same stable input order. The stable input order
is the line order of the frozen split file `data/splits/<release>/<split>.txt`,
which is sorted (see [frozen split files](data.md#frozen-split-files)).

## Output fields: one UTF-8 JSON object per line

| Field | Type | Required meaning |
| --- | --- | --- |
| `schema_version` | string | `0.1` |
| `run_id`, `system_id` | strings | Unique run and frozen system configuration |
| `dataset_release`, `split` | strings | Frozen release and target split |
| `split_manifest_sha256` | string | Exact frozen split manifest hash (defined below) |
| `utterance_id`, `audio_sha256` | strings | Match the inference input |
| `status` | enum | `ok`, `empty`, or `error` |
| `hypothesis_text` | string or null | Raw decoded text; empty string for `empty`, null only for `error` |
| `selected_candidate_id` | string or null | Actual N-best candidate if rescored; otherwise null |
| `nbest_artifact_sha256` | string or null | Hash of related candidate artifact if produced |
| `model_artifact_sha256`, `decoder_config_sha256` | strings | Pinned model bundle and decoding policy |
| `normalization_version` | string | Evaluation policy label; text here remains raw |
| `timing_record_id` | string | FK to timing record, even when a failure is measured |
| `error` | object or null | Sanitized error code/message for `error`; null otherwise |

In inference inputs and predictions alike, `split_manifest_sha256` is the SHA-256
of the split manifest file's bytes as stored, written as 64 lowercase hex
characters. Rows keep `schema_version: "0.1"`; contract 0.2 changes no field.

A deterministic empty result is a valid ASR output, not a missing record. An error
is never silently dropped or relabeled as a successful empty transcript. The
evaluator uses the published failure policy and reports counts. All numeric
values must be finite. Timestamps and expanded per-stage timing belong in the
sidecar to avoid inconsistent utterance/batch measurements.

No `reference_text`, `gold`, target labels, transcript corrections, or reference
token IDs may appear in inference/prediction artifacts. Gold joins occur only in
the offline evaluator, keyed by utterance ID, after checks for duplicate, extra,
missing, mismatched audio, release and split hashes. Incompatible bundles fail
before metrics; no implicit inner join or ID rewriting.

The sidecar run manifest supplies code/config/environment/model/knowledge hashes,
selection policy, actual command, decoding parameters, hardware and timing scope.
JSONL schema and score semantics must be validated before aggregate reporting;
validators are future work.

## Version history

- 0.1 (2026-11-17): initial contract.
- 0.2 (proposed by M6 on 2026-11-17, pending review by M1 and model owners
  M2–M5; [D19](../decisions.md)): defines the stable input order as the sorted
  split-file order and `split_manifest_sha256` as the SHA-256 of the split
  manifest file's bytes.
