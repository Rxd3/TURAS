# Message to Member 1 (draft for the owner to send)

Prepared 2026-11-19. Not sent by the assistant.

Hi, I am preparing the Member 4 and 5 runs. We have repaired and verified the
test-only archive. Please share working download links for:

1. `turas-tr-econ-legis-v0.1-inference-ALL-MODEL-OWNERS.tar`
2. `turas-tr-econ-legis-v0.1-train-references-M3-M4-M5.tar`

Please confirm the SHA-256 hashes and extraction layout, and flag any changes
since frozen split manifest `a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8`.
The training package needs train transcripts and train relational CSVs only.

To run validation locally as a separate evaluation step, please also provide a
validation-only reference file/package with its hash, or the existing M6 validation
scoring handoff. Keep it separate from training; no test references are needed now.

Please point to the listening corrections/verification statuses and any known
alignment or numeric-transcription issues. We already have the public split lists
and static metadata in the repository, so those do not need redoing.

I will handle checkpoint choices, configurations, integration and baseline runs
here under the recorded project-owner decisions. Those agreements do not need to
hold up your data handoff.

## Published bundle hashes to check against

From M1's committed handoff manifest (not locally verified bundle bytes yet):

- Inference/audio: `fe5a9e7eb20bb867a96387c88a85300f92fae58786d1979828cc9f63829e5f3b`
- Train references: `3159c676ce00294913946875ed89d50170f7bdaa1bbcd6167d7f1c8717ce24cf`

If these changed, provide a versioned replacement and updated inventories rather
than silently replacing files under a frozen release label.
