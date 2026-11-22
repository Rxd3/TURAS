# Drive package map

Create **two separate top-level folders**. `TURAS-Public-Submission` holds the
public source/guides and presentation. `TURAS-Restricted-Reproduction` stays
owner-only at its root; grant named accounts access to the relevant child
folders individually. Do not make the restricted root an anyone-with-link folder.
Do not upload one archive containing all roles.

The prepared local delivery uses this structure:

| Destination | Contents | Audience |
| --- | --- | --- |
| Public / `00-START-HERE` | Exact submission-source ZIP, package manifest, verifier and upload instructions | Graders/public source readers |
| Public / `07-PRESENTATION-AND-REPORTS` | Final two-minute PPTX, speaker script and aggregate reports | Graders/presentation audience |
| Restricted / `01-AUDIO-INFERENCE` | Original reference-free 5,450-clip audio/inference TAR | Authorized M2–M6/model users |
| Restricted / `02-TRAIN-REFERENCES` | Original train-reference/relational TAR | M3/M4/M5 training users |
| Restricted / `03-VALIDATION-SELECTION` | Working validation references and separate relational supplement | Authorized validation selectors/evaluators |
| Restricted / `04-EVALUATOR-ONLY` | Working evaluator-reference TAR | M6/authorized offline evaluators |
| Restricted / `05-M4-M5-MODELS-AND-RUNS` | Original selected-checkpoint/run TAR and required source-v2/index supplement | Authorized reproduction reviewers |
| Restricted / `06-M3-B5-HMM` | M3 weights/results ZIP, exact runtime-source ZIP and original training status | Authorized reproduction reviewers |
| Restricted / `08-M1-ASSIGNED-LISTENERS` | Original transcript-bearing listening/adjudication workspace | Owner and assigned independent listeners |
| Restricted / `09-OPTIONAL-TRAINING-HISTORY` | Earlier M4/M5 weights/history, M3 epoch-13 input and latest full optimizer checkpoint | Authorized retraining/continuation users |

The refreshed delivery additionally uses the following folders. The preparer
populates them only after every required accuracy/timing export is completed
and checked. The earlier snapshot's `10-PENDING-M2-B2` is not the B2 package.

| Destination | Exact payload filename | Restore role |
| --- | --- | --- |
| Restricted / `10-B2-WHISPER-LARGE-V3` | `TURAS-B2-GPU-followup.zip` | `b2` |
| Restricted / `11-B3-QWEN` | `TURAS-B3-Qwen-followup.zip` | `b3` |
| Restricted / `12-TURKISH-TRANSDUCER` | `TURAS-Seda-RNNT-followup.zip` | `transducer` |
| Restricted / `13-MIXTURE-GMM-HMM` | `TURAS-4mixture-GMM-HMM-followup.zip` | `gmm` |

Each contains a `README.txt` and `SHA256SUMS` beside its ZIP. B2/Qwen ZIPs
contain full predictions, N-best, timing, scores, processor/config files and
upstream hash inventories. Their large pretrained weight files are separate
pinned downloads. The Qwen ZIP keeps both validation candidates and only the
winner's test/timing runs. The transducer ZIP includes its acoustic ONNX weights
without the optional external LM. The new GMM ZIP includes the selected trained
mixture checkpoint and training protocol/history, without feature caches.

The refreshed public `07-PRESENTATION-AND-REPORTS` includes
`TURAS-final-2-minute-v2.pptx`, `TURAS-two-minute-speaker-script.txt`,
`BASELINE_COMPLETION_REPORT.md`, `completion_test_summary.json`,
`completion_metrics.csv`, `fresh_environment_restoration.json`,
`completion_hardware_context.json` and `baseline_completion_followup.md`, along with the
preserved original reports. `00-START-HERE` adds
`FOLLOWUP-REPLAY-COMMANDS.md`. Upload every file in each prepared folder,
including its guide and checksums; do not unpack model/data archives in Drive.

Folders 00, 01, 04, 05 and 06 support source/metric review of the completed runs.
Add 02 and 03 for training/validation selection. Folder 09 is optional for scoring
or inference. Folder 08 is not a model input. Pretrained upstream parents are
separate pinned downloads, as documented in the reproduction guide.

Each folder has a `README.txt` and `SHA256SUMS`. The public package manifest lists
payload sizes, full hashes, audience and exact restore location. Original TAR
bytes are copied unchanged. No cloud upload, permission change, receipt or
successful recipient retrieval is implied by local preparation.

## Add the real link after uploading

From the repository root:

```text
python scripts/update_artifact_links.py --public-drive-url "PASTE_THE_REAL_PUBLIC_FOLDER_URL"
```

The updater accepts an actual HTTPS Google Drive folder URL, changes only the
marked README section and writes the public link metadata. It never publishes
the restricted root or changes Drive permissions. Commit that result through
the normal review workflow. The public README links the public submission
folder; authorized reviewers obtain role-specific restricted folders separately.

After uploading, download a package from a recipient account and run:

```text
python verify_packages.py --root PATH_TO_DOWNLOADED_DELIVERY --allow-missing
```

The public-only copy can omit restricted payloads. A reviewer verifies every
package they actually received; only use the full check when all roles are held.
