# Local migration acceptance

Validation snapshot: 2026-09-13, with the standardized-format checks added on
2026-09-14. These checks were recorded before committing or pushing. Publication
state is tracked by Git history, not by the historical `github_push_performed`
fields in the audit snapshots.

## Data and source checks

- 677 committed source scripts/config/task files copied byte-for-byte from
  `60c45238b8bac0cec53a2a518f3945a68b39ae83`; the omitted local fixes were not copied.
- 3,188 local result/ledger files archived, plus 1,627 remote output files
  and 126 explicitly labeled historical completion extractions.
- 4,941 source aliases resolve to 3,763 unique gzip objects, totaling
  521,280,665 stored bytes. The objects are included through Git LFS, not replaced
  by external OpenWeights links.
- All 630 main experiments and the one supplemental API reference are indexed.
- 349,200 main-experiment completion texts are preserved. Standalone original
  inference exists for all 630 runs after event-history recovery of the 126
  older files. All original inference and selected judge-input text match.
- 232 training-job links reconciled from saved ledgers. Remaining training links
  and unrecorded historical checkpoint revisions are explicitly unresolved.
- All object/content and committed-code hashes verified. Local document links
  checked and all Python source files parsed. Known-credential/prefix and
  signed-URL scans found no matches in the archived payload.

See `delivery_verification.json`, `delivery_inventory.json`,
`inference_integrity_summary.json`, and the detailed JSONL audit records.

## Initial migration offline numerical reproduction

An isolated checkout was materialized from the staged Git index with actual LFS
content. It contains no original-project symlinks or credentials. From that
checkout, the portable analysis ran with API credentials unset and Python socket
creation disabled. It depends only on the bundled test Python environment and
NumPy 2.3.5, not the original project. The pinned requirements allow others to
install that numerical dependency independently.

All 18 generated tables matched the frozen release within `1e-12`; the largest
observed numerical difference was `1.1102230246251565e-15`. This includes 2,520
run comparisons, 349,200 normalized completion records, 139,680 prompt-score rows,
five-seed intervals/Pareto results, and the coverage/sensitivity diagnostics.
The supplemental Qwen3 reference also matched. Seven boundary/normalization tests
passed. The full numerical audit is `offline_reproduction.json`.

The regenerated SVG plots use the verified means and intervals. The original
PNG figures, report, and workbook are preserved, not byte-for-byte rerendered.
Historical missing information does not prevent recalculation of saved scores,
but it prevents claiming every original experiment can be rerun identically.

## Additive OpenWeights output recovery

The subsequent read-only recovery audited 966 inference/judge jobs and 1,109
remote attempts, locating 1,627 output files. Of these, 336 were already archived
and 1,291 were downloaded; all were available. Exact-content deduplication added
413 objects (8,244,580 compressed bytes). All 126 previously missing original
completion checkpoints now match the selected judge inputs. Historical retries
and pre-backfill judge files are linked separately, not promoted to selected scores.

The frozen release files and primary result selections remain unchanged. Current
archive/reference checks are recorded in `delivery_verification.json`; the
earlier isolated-checkout numerical audit remains in `offline_reproduction.json`.
See `OPENWEIGHTS_OUTPUTS.md` for scope and extraction instructions.

## Standardized format pass

The 2026-09-14 format pass adds `result/standardized/v1/`: two fixed-schema
UTF-8 JSONL gzip views, a source/selection index, and validation metadata. They
cover 1,922 distinct source artifacts, all 1,627 recovered remote file IDs, and
the completion/judgment selections for 631 runs. The view contains 356,559
completion records and 1,159,320 individual score records, including historical
attempts and separate judge representations. These are not additional selected
experiments. The two gzip files add 388,245,160 bytes; they are derived views,
separate from the immutable raw-object inventory above. See
`result/standardized/README.md` from the repository root for the format contract.
Original raw files, primary result selections, report tables, and imported core
code remain unchanged. Format-only conversion does not apply coherence filtering
or alter the original rubric scales. Seventeen format/scoring tests pass.

## Publication gate

This verifies a local staged checkout, not a fresh remote GitHub clone. Before
publication, review the staged source/data diff and license/attribution scope.
After an approved push, verify all LFS objects are uploaded and retrievable from
a fresh authorized remote clone. No external push, new training, new inference,
or new judge API calls were performed during this migration.
