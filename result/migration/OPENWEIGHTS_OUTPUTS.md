# Recovered OpenWeights evaluation outputs

Date: 2026-09-13. Read-only recovery; no jobs submitted or rerun.

The archive now includes all 1,627 checkpoint files discovered in the event
histories of the 966 inference/judge jobs linked to the main experiment. The
audit paginated through 15,283 events from 1,109 recorded attempts, avoiding the
final output summary's overwritten `file_id` field.

| Original-format file | Unique remote file IDs |
|---|---:|
| `completions.jsonl` | 655 |
| `judge_scores.jsonl` | 485 |
| `eval_results.csv` | 487 |
| Total | 1,627 |

336 files were already archived and reused; 1,291 were fetched in this pass.
No discovered files were unavailable. Exact-content deduplication added only
413 objects, or 8,244,580 compressed bytes, because many downloaded files match
existing local evidence. The complete archive contains 3,763 objects with 4,941
source aliases, totaling 521,280,665 compressed bytes.

## Original inference coverage

All 630 main runs now have verified standalone original inference checkpoints.
The 126 older files missing from the initial final-output-field audit were
recovered from `completions_saved` events. Their completion identities, evaluation
IDs, and text match the selected judge CSVs. All 349,200 main completion texts
have been checked, with zero mismatches. Earlier derived text extractions remain
archived, explicitly marked as historical fallbacks superseded by originals.

655 completion files does not mean 655 selected experiments: retry attempts are
preserved as evidence, and only matching original checkpoints are linked as the
selected inference artifacts. Likewise, original judge checkpoint/CSV files do
not replace later coherence backfills or alter any selected result score.

## Open the actual files

From the repository root, after Git LFS objects are present:

```sh
python scripts/analysis/archive.py extract --prefix openweights/files/ --destination /tmp/slb-openweights
```

This extracts actual bytes from local `result/artifacts/sha256/` objects, with
paths such as `openweights/files/<file_id>/completions.jsonl`. It needs no API key
or OpenWeights access. Use a new destination, or one containing identical files.

Each run/stage manifest has `remote_output_artifacts` linking local artifact IDs
to job, remote-attempt, and event IDs. `primary_result_artifact_id` continues to
identify the result selected for analysis.

## Audit and limits

- `remote_output_events.jsonl`: per-job attempts, event counts, output references.
- `remote_output_downloads.jsonl`: file IDs, hashes, schemas, counts, and status.
- `remote_output_recovery.json`: recovery totals and exclusions.
- `resolved_lineage_gaps.jsonl`: original-inference gaps resolved by this recovery.
- `inference_integrity_summary.json`: comparison against selected judge inputs.
- `delivery_verification.json`: complete archive/reference and frozen-release checks.

This operation retrieves inference/judge checkpoint outputs for selected main
jobs and all their recorded attempts. It does not retrieve model weights, worker
configs, console logs, training-only job outputs, or unrelated jobs. Previously
archived local logs and ledgers remain included. Files preserve only fields the
original workers saved; absent runtime details, token usage, and historical HF
revisions are not reconstructed. No selected CSV, coherence cache, published
table, or imported core script was changed. This report records the pre-push
recovery snapshot; subsequent publication is tracked by Git history.
