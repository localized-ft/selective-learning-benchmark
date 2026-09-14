# Experimental results

This directory contains the **actual data**, not just external references. Raw
files are losslessly gzip-compressed into content-addressed objects under
`artifacts/sha256/` and stored using Git LFS. Registries/manifests connect those
objects to experiments and retain original filenames and source hashes.

## Read the results

- [Main five-seed report](releases/five_seed_20260908/REPORT.md)
- [Run-level comparisons](releases/five_seed_20260908/outputs/comparisons.csv)
- [Method summaries](releases/five_seed_20260908/outputs/method_summary.csv)
- [Dataset/model summaries](releases/five_seed_20260908/outputs/cell_summary.csv)
- [Supplemental original Qwen3 API comparison](supplemental/qwen3_openrouter_bad_medical_20260909/REPORT.md)
- [Data dictionary](../docs/RESULT_DATA_DICTIONARY.md)
- [Offline reproduction](../scripts/analysis/README.md)
- [Standardized completion and judgment files](standardized/README.md)

The main release is seven tasks x three model families x six methods x five seeds
= 630 runs. All four metric/filter variants refer to the same trained checkpoints.
The source label `baseline` in historical tables means standard SFT; the registry
calls it `sft`. Vanilla means the original instruct checkpoint with no benchmark
fine-tuning and has no training-seed label.

## What is preserved

The archive includes raw inference completions and available response envelopes,
original judge CSVs and score-response text, saved judge event/checkpoint files,
coherence backfills, failed/retried attempts, historical releases, job ledgers,
evaluation inputs, rubrics, and provenance. Only exact byte duplicates share an
object. An obsolete or failed result can remain archived without being included
in the selected scientific comparison.

All **630 main runs have standalone original inference files**. The 126 legacy
files initially unlocated in final job-output summaries were recovered from
OpenWeights event history. Every one of the **349,200 completion texts** was
checked against the exact text retained by the selected judge CSVs, with no
mismatches. Earlier labeled JSONL extractions remain as historical fallbacks,
now marked as superseded by verified originals; no response fields are fabricated.

The [OpenWeights recovery report](migration/OPENWEIGHTS_OUTPUTS.md) documents
1,627 archived remote output files across 966 evaluation jobs, including retries.
The initial `legacy_inference_audit.jsonl` records only the earlier final-output
field audit; the event-history recovery supersedes its missing-file conclusions.

All 630 runs have artifact model identities/Hugging Face links and completion-job
IDs. Training-ledger reconciliation currently links 232 training jobs; unresolved
links and unrecorded historical checkpoint revisions remain visible. These gaps
limit exact experiment reruns, not offline recalculation of the saved scores.

## Find and open a raw file

1. Find the experiment in `registry/runs.jsonl` or `runs/<run_id>/manifest.json`.
2. Follow its `artifact_id` into `registry/artifacts.jsonl` to find the local object.
3. Decompress that `.gz` file to recover its exact original-format contents.

The outer object is always gzip. If its source was itself `.csv.gz` or `.xlsx`,
the decompressed contents retain that original format. To avoid handling hashes
manually, use `scripts/analysis/archive.py extract` from the repository root.
The source-to-object mapping is in `migration/source_manifest.jsonl`.

Each run's `associated_artifacts` lists relevant files, while the full source
manifest also retains shared backfills, historical batches, and reports. The
shared coherence checkpoints are explicitly linked by each main judge manifest.
Source-file IDs are useful provenance but are not required to fetch the archived
data from this repository.

`remote_output_artifacts` links additional OpenWeights checkpoints to their job,
remote attempt, and event IDs. It preserves retry outputs and original judgments
before coherence backfills without changing the selected result. The remote
recovery covers inference/judge checkpoint outputs, not model weights, console
logs, worker configs, or training-only jobs. Previously archived local logs and
training ledgers remain included.

Extract the original-format remote output files from the repository root:

```sh
python scripts/analysis/archive.py extract --prefix openweights/files/ --destination /tmp/slb-openweights
```

This is an offline extraction from the actual repository-contained data; it does
not contact OpenWeights. Use a new destination or one containing identical files.

## Directory roles

- `registry/`: run, model, job, artifact, and unresolved-lineage indexes.
- `protocols/`: frozen rubric/task snapshots and scoring definitions.
- `runs/`: human-navigable manifests for each experiment and its stages.
- `artifacts/`: actual immutable data objects, including shared and historical files.
- `releases/`: frozen reports, tables, diagnostics, and figures.
- `supplemental/`: approximate API reference, kept outside the five-seed experiment.
- `migration/`: source hashes, code allowlist, exclusions, recovery/audit records, and verification.
- `standardized/`: fixed-schema JSONL views of completion/judgment outputs, with source and selection indexes.

Offline reproduction reads raw evidence, applies the documented parser and
coherence-backfill rules, recomputes all four comparisons and diagnostics, and
checks them against the frozen release. It does not invoke a model or judge.
Rerunning inference is stochastic and requires separate compute and protocol
validation. See the [migration plan](../docs/MIGRATION_AND_RESULTS_PLAN.md).
