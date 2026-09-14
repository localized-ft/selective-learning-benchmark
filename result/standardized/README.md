# Standardized result records

Use `v1/completions.jsonl.gz` and `v1/judgments.jsonl.gz` for a consistent,
machine-readable view of saved inference and judge outputs. Both use the same
fixed field order, schema version `slb.records.v1`, UTF-8 encoding, LF line endings,
and gzip compression. These are actual derived data files stored through Git LFS.

The view covers every downloaded OpenWeights output file and the selected
completion/judge artifacts for all 630 main runs plus the supplemental Qwen3
reference. Historical originals remain unchanged in `../artifacts/`. Provider API
event envelopes, coherence caches, reports, and job ledgers retain their original
formats; they are different data entities, not completion/judgment tables.

## Format contract

Each line is one JSON object. `record_type` is `completion` or `judgment`. A
judgment record represents one named score, so nested `scores` arrays and CSV
score rows have the same format. Every record has all fields, even when null.

| Fields | Type and meaning |
|---|---|
| `schema_version`, `record_type` | Required schema and entity labels |
| `source_artifact_id` | SHA256 reference into the unchanged raw archive |
| `source_row` | One-based source data-record number, excluding CSV header; not physical line number |
| `source_score_index` | Zero-based nested score index; null for CSV/completion rows |
| `completion_id`, `eval_id` | Text identifiers; missing optional IDs are null |
| `axis` | `capability`, `unintended_generalization`, or null if not saved |
| `group_id`, `task_id`, `model`, `judge_model` | Text or null; no inferred model/task identities |
| `question`, `completion`, `reference_response`, `grading_method` | Text or null; observed empty completions stay empty strings |
| `score_name`, `score_label`, `score_source_text` | Text or null |
| `score` | Finite number or null, on the original rubric's scale |
| `score_raw` | Original scalar rendered as text, or null for missing values |
| `score_status` | `numeric`, `missing`, `non_numeric`, `non_finite`, or `not_applicable` |
| `extra_fields` | Object retaining additional source fields such as `idx` |

`undesired_generalization` is standardized to `unintended_generalization`.
Absent optional values are null, never zero or an invented empty answer.
Categorical labels remain categorical: for example, `LLM` is not converted to
zero and the old-bird score `19` remains 19. Non-finite inputs are explicit
`non_finite` records with their literal value in `score_raw`, not JSON NaN/Infinity.
No score rescaling, threshold filtering, averaging, or coherence-cache join is
performed by format standardization. Existing analysis rules remain authoritative
for those operations.

## Selection and provenance

`v1/sources.jsonl` maps every source artifact to original aliases, OpenWeights
job/file/event/attempt IDs, source/canonical row counts, and selected run IDs.
Exact duplicate source aliases are converted once. Retry outputs, CSV judgments,
and JSONL judgments remain distinct evidence even when they describe the same
completion. Do not pool all judgment records as independent observations.

For selected analyses, use `selected_judgment_runs` in `sources.jsonl`; use
`selected_completion_runs` for selected generated answers. Run identities and
metadata are in `../registry/runs.jsonl`. A completion ID alone is not globally
unique. The source artifact and source record/score coordinates identify a
canonical observation without joining unrelated attempts.

For the published four-variant scores, use the existing offline reproduction
runner, which additionally applies task-specific categorical interpretation and
saved coherence backfills. Standardized raw records do not replace that analysis.

## Read or regenerate

From the repository root:

```python
import gzip
import json

with gzip.open("result/standardized/v1/judgments.jsonl.gz", "rt", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        # Filter source_artifact_id using sources.jsonl before aggregation.
```

Regenerate deterministically into a new empty directory, without API calls:

```sh
python -B scripts/analysis/standardize.py --output /tmp/slb-standardized-v1
python -B scripts/analysis/standardize.py --verify-only
python -B -m unittest discover -s scripts/analysis -p 'test_*.py' -v
```

`v1/manifest.json` records coverage, source schemas, output counts, hashes, and
round-trip validation. `--verify-only` additionally compares every saved record
with its referenced original source and checks file hashes and per-source counts.
The exporter fails on malformed rows or unknown axis
labels rather than silently dropping records. It refuses a nonempty destination.
