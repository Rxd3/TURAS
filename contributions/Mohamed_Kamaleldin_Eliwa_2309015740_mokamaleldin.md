# Member 4 — Whisper Fine-Tuning

Name: Mohamed Kamaleldin Eliwa. Student ID: `2309015740`. GitHub username: [`mokamaleldin`](https://github.com/mokamaleldin).
Role: Member 4 — Whisper Fine-Tuning.
Branch convention: `feat/m4-whisper-finetuning` (focused topic suffixes allowed).
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).
Backward-compatible record: [member_4.md](member_4.md).

> [!NOTE]
> This is the canonical named individual contribution record for university
> submission. [`member_4.md`](member_4.md) is kept for backward compatibility
> with historical links. In accordance with repository attribution rules, this
> record separates Member 4's own identity/verification commits (`@mokamaleldin`)
> from the earlier owner-assisted M4 implementation, seed-17 training runs,
> frozen test evaluation, and private handoff delivered by `@Rxd3` in PR #3 and
> PR #5.

**Status (2026-11-22): Member 4 identity and named submission record committed
by `@mokamaleldin`; owner-assisted Whisper LoRA and full fine-tuning
implementation, seed-17 training/validation, frozen 636-clip test/timing, and
verified private handoff are complete and merged into `main` via PR #3
(`2f20f92`) and PR #5 (`62fadf5`) after Member 6 (`@afel6`) review. Independent
clean-environment reproduction and consumer bundle acceptance remain open.**

## Workstream scope and tasks

Assigned workstream from [TEAM_TASKS.md](../TEAM_TASKS.md) (Whisper Fine-Tuning):

| Area | Owned / Workstream Paths | Recorded Status and Provenance |
| --- | --- | --- |
| B1 parent and adaptation profiles | `configs/fine-tuning/whisper_small_lora.json`, `configs/fine-tuning/whisper_small_full.json`, `configs/evaluation/m45_final_protocol.json` | Frozen under owner D23 (`openai/whisper-small`, rev `973afd24...`); owner-assisted (PR #3, PR #5) |
| Acoustic training and Whisper collation | `src/turkish_asr_rdl/training/whisper.py`, `src/turkish_asr_rdl/training/acoustic.py`, `scripts/m4_train_whisper.py` | Train-only manifest/hash validation, LoRA (r=16, alpha=32) and full fine-tuning, checkpoint export; owner-assisted (PR #3 `2f20f92`) |
| Scored Whisper N-best decoding | `src/turkish_asr_rdl/decoding/whisper.py` | Genuine beam N-best (`num_beams=5`, `requested_n=5`) and original-logit teacher-forced scoring consumed by M5; owner-assisted (PR #3 `2f20f92`) |
| Offline unit tests and notes | `tests/test_m4_training.py`, `experiments/m4-whisper-notes.md` | Random-model offline tests and dated adaptation notes; owner-assisted (PR #3 `2f20f92`, PR #5 `62fadf5`) |
| Contribution identity and verification | `contributions/Mohamed_Kamaleldin_Eliwa_2309015740_mokamaleldin.md`, `contributions/member_4.md`, `contributions/README.md`, `TEAM_TASKS.md`, `docs/directory_tree.txt` | Committed from `@mokamaleldin` on 2026-11-22; verified locally with `scripts/verify_submission.py` |

## Actual contributions

### 2026-11-22 — Member 4 identity registration, named submission record, and local verification (`@mokamaleldin`)

- Recorded Member 4 full name (`Mohamed Kamaleldin Eliwa`), student ID
  (`2309015740`), and GitHub handle (`@mokamaleldin`) from Member 4's own
  GitHub account.
- Added canonical named record
  `contributions/Mohamed_Kamaleldin_Eliwa_2309015740_mokamaleldin.md`, updated
  `contributions/member_4.md` with identity and canonical pointer for backward
  compatibility, and updated links in `contributions/README.md`,
  `TEAM_TASKS.md`, and `docs/directory_tree.txt`.
- Ran `python3 scripts/verify_submission.py` locally against `main` (`bc3c533`),
  verifying frozen protocol SHA-256
  `288aebd71d8a3574e13d436748d4161bc258a487e65066f1c3508bc8e835bec1`, all 56
  pinned code hashes (including `src/turkish_asr_rdl/training/whisper.py`,
  `src/turkish_asr_rdl/training/acoustic.py`, and
  `src/turkish_asr_rdl/decoding/whisper.py`), exact WER/CER recomputation from
  stored S/D/I/N edit counts across all 20 systems, all 5,450 disjoint split
  clips (`4376/438/636`), and all 11 required submission files.
- **Attribution boundary:** the 2026-11-19 implementation, seed-17 GPU training
  and validation runs, frozen test evaluation, and private archive handoff
  recorded below (PR #3 and PR #5) were owner-assisted (`@Rxd3`) and are not
  claimed as independent code authored by `@mokamaleldin`.

### 2026-11-19 — Owner-assisted Whisper fine-tuning implementation and benchmark (PR #3, PR #5; authored by `@Rxd3`)

2026-11-19: implemented train-only manifest/hash validation, shared acoustic
optimization/checkpoint export, Whisper LoRA/full construction and collation,
genuine beam candidates and original-logit teacher-forced scores. Added separate
draft profiles, thin training wrapper, offline random-model tests and a CPU test
workflow. See [implementation notes](../experiments/m4-whisper-notes.md).
Shared training/CI changes require M6 review; parent compatibility requires M2;
N-best requires M5. Delivered in [M4 PR #3](https://github.com/Rxd3/TURAS/pull/3)
(merged `2f20f92`) and [integration PR #5](https://github.com/Rxd3/TURAS/pull/5)
(merged `62fadf5`).

## Experiments and artifacts

Summary of the frozen 636-clip working test results for the Member 4 Whisper
track (single training seed 17, `tr-normalized-v1`, frozen protocol SHA-256
`288aebd71d8a3574e13d436748d4161bc258a487e65066f1c3508bc8e835bec1`; full tables
and paired intervals in [final report](../results/final/M4_M5_FINAL_REPORT.md)):

| System | Selected Checkpoint | Raw WER | Normalized WER | Normalized CER | Term Micro F1 | End-to-End RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `B1-whisper-small` (unchanged parent) | `openai/whisper-small` (`973afd24...`) | 37.209% | 24.512% | 10.079% | 92.175% | 0.0540 |
| `F1-whisper-lora` (rank 16, alpha 32, seed 17) | LoRA epoch 3 | 21.505% | 12.368% | 4.149% | 96.843% | 0.0532 |
| `F2-whisper-full` (full fine-tune, seed 17) | Full epoch 2 | 18.883% | 10.371% | 3.691% | 98.080% | 0.0530 |

No runs in the initial implementation. Later owner-assisted execution is in
[the dated M4/M5 record](../docs/m45_execution_2026-11-19.md); local job status
was tracked in `artifacts/owner-m45/queue-status.json`. Confirmed 2026-11-19
09:10 UTC: the pinned parent produced valid outputs for all 438 validation
clips, and owner-assisted LoRA run
`acoustic-bbc267859abb4a769e44e1bd7877ae61` completed initial startup steps on
320 real train clips before full three-epoch LoRA and full-model seed-17 runs
completed.

The private model/run archive (`TURAS-M4-M5-private-handoff.tar`,
`2,424,330,240` bytes, SHA-256
`3bbaf65cd8ac45b4dceab6e1401748392e0bf3dd0039d395d1e092719740a91d`) is kept
outside Git and documented in [handoff and limitations](../docs/m45_completion.md).

## Reviews and handoffs

- [M4 PR #3](https://github.com/Rxd3/TURAS/pull/3) (`feat/m4-whisper-finetuning`):
  reviewed, approved, and merged into `main` by Member 6 (`@afel6`) at commit
  `2f20f92`.
- [Integration PR #5](https://github.com/Rxd3/TURAS/pull/5)
  (`feat/m4-m5-integration`): reviewed by Member 6 (`@afel6`, verifying 56
  pinned code hashes, split/manifest/lexicon hashes, S/D/I metric recomputation
  across 20 systems, and lean tests), approved, and merged into `main` at commit
  `62fadf5`.
- Independent clean-environment reproduction and consumer bundle acceptance for
  the private handoff archive remain pending; add actual reproduction receipts
  and accepted bundle hashes when they occur.

## Remaining work

Final technical closeout also restores the shipped archive into a fresh source
checkout and executes the actual delivered models. The recorded
[acceptance result](../results/final/m45_restored_delivery_acceptance.json) uses
validation audio only, verifies all shipped bytes and imports, and includes
online/cached RDL parity. It is owner-assisted verification, not another person's
approval. The component PR was synchronized with current `main`; its 54 tests
pass. See [consumer procedure](../docs/m45_delivery_acceptance.md).

Owner-assisted seed-17 LoRA/full training and all six validation epochs are now
complete; best working checkpoints are LoRA epoch 3 and full epoch 2, selected
on the explicitly recovered-minutes validation reference version. Real scored
N-best was consumed in M5's fusion grids. See the
[completed working validation record](../docs/m45_validation_complete.md) and
[selection inventory](../results/evaluation/m45_working_checkpoint_selection.json).
Final working evaluation is now complete: LoRA normalized test WER 12.368%, full
10.371%, unchanged parent 24.512%; all 636 IDs, raw/normalized CER/WER,
terminology, 2,000-replicate paired intervals and three-repeat timing are in the
[final report](../results/final/M4_M5_FINAL_REPORT.md). Actual N-best was consumed
in all fixed-candidate studies; all main scores independently match JiWER.
The private model/run bundle is built and byte verified; see
[handoff and limitations](../docs/m45_completion.md). Remaining: required
independent clean-environment reproduction and actual consumer ACK. Optional
extra seeds are follow-up robustness work, not a completed experiment or a
deadline blocker.
