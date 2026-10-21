# Independently analyze the KL reference ablation

- Start with the [bullet-point summary](KL_ABLATION_SUMMARY.md) and [model/data links](MODEL_AND_DATA_LINKS.md).
- This package covers eight conditions: standard SFT, preferred/rejected/self-distilled HHH KL, and original/self-distilled TruthfulQA and MBPP KL.
- The archive contains the actual raw outputs, not just OpenWeights file IDs. Model weights remain on Hugging Face; their URLs and recorded revisions are indexed.

## 1. Download the archive

```sh
git clone https://github.com/localized-ft/selective-learning-benchmark.git
cd selective-learning-benchmark
git lfs install
git lfs pull
```

- Git LFS is required for large raw files. A text pointer is not the underlying data.
- The verification step below detects missing or changed LFS objects. No API key is needed to analyze the archive.

## 2. Verify and recompute all eight rows offline

From the repository root, using Python 3.11 or later:

```sh
python3 scripts/analysis/kl_ablation_release.py verify
```

- This uses only the Python standard library. It does not install dependencies, contact model/judge services, submit jobs, or modify the saved experiment.
- It checks publication-file hashes and compressed/uncompressed content-addressed object hashes, joins all 6,080 saved inference outputs to their selected task scores, and recomputes raw and coherence-filtered means and coverage counts.
- The recomputed values must exactly match [comparison_all_references.json](comparison_all_references.json).
- Scores are on a 0–100 scale. Capability measures acquisition of bad medical advice; UG is `100 - mean(numeric alignment)`. Filtering retains numeric task scores with sibling numeric coherence `>=50`.
- Missing/non-numeric task labels are retained in raw records and excluded from numeric means, not assigned zero. Missing historical capability coherence stays missing.
- `publication_index.json` records the exact paths, hashes, condition selectors, raw inference objects, model links/revisions, and job IDs. It intentionally does not invent missing historical revisions or training-job links.

## 3. Load raw scores for your own analysis

Example using only the standard library, from the repository root:

```python
import gzip
import json
from pathlib import Path

root = Path.cwd()
experiment = root / "result/supplemental/kl_reference_distribution_20261019"
index = json.loads((experiment / "publication_index.json").read_text())
condition = next(c for c in index["conditions"] if c["condition"] == "truthfulqa_original")

with gzip.open(root / condition["scores_path"], "rt") as handle:
    scores = [json.loads(line) for line in handle]
selector = condition["score_filter"]
if "run_file" in selector:
    scores = [r for r in scores if r.get("run_file") == selector["run_file"]]
if "completion_id_prefix" in selector:
    scores = [r for r in scores if r["completion_id"].startswith(selector["completion_id_prefix"])]

with gzip.open(root / condition["raw_inference"][0]["repository_path"], "rt") as handle:
    completions = [json.loads(line) for line in handle]
print(len(completions), len(scores))  # 760 completions; 1,520 task/coherence rows
```

- Select the condition first, then join by completion ID. Historical score files reuse completion IDs across runs; a global join would mix models.
- New inference `full/artifacts.json` files preserve original filenames, OpenWeights file/job/run/event IDs, checksums, and intermediate checkpoint sizes. Use the indexed 760-row objects for the final comparison rather than pooling intermediate checkpoints.
- Generation-detail objects retain token IDs, finish reasons, and environment information. Judge directories retain requests, raw attempts, parsed scores, configs and verification reports. Historical Luna controls retain raw judge response text, not necessarily complete HTTP envelopes.
- Frozen task prompts/rubrics are under `result/protocols/tasks/historical_20260908/`; the imported Luna parser/rendering code is under `result/supplemental/luna_rejudge_20261004/upstream/`.

## 4. Models, datasets, and training reconstruction

- [MODEL_AND_DATA_LINKS.md](MODEL_AND_DATA_LINKS.md) links every condition to its HF checkpoint and local raw data. The six newer runs are pinned LoRA adapters in an `adapter/` subfolder; load them on the recorded pinned base. Historical SFT/preferred-HHH controls are full checkpoints, with unknown inference-time revisions.
- Exact medical train/validation files and reference JSONL files are included locally. Published reference datasets are linked at the actual training revisions, not only at a mutable `main` branch.
- Each new training directory has its scientific manifest, configuration, submission receipt, status events, exported-adapter verification and hashes. The submission receipts map uploaded worker filenames to immutable objects in `result/artifacts/sha256/`.
- For exact historical worker reconstruction, decompress those uploaded worker objects and verify their content hashes. They are authoritative for the executed code; current common training helpers can evolve. No shared training-helper changes unrelated to this release are being published here.
- Ablation-specific training/configuration, generation-recovery, inference and Luna preparation/transport scripts are included under `scripts/finetune/` and `scripts/eval/`. They are not required for offline analysis. Paid reruns require a compatible GPU/runtime, service credentials and explicit job submission; do not run submission/restart helpers just to reproduce the table.
- The new Luna-only reports can additionally be regenerated with `scripts/analysis/kl_reference_luna_report.py --cohort both` in the evaluation environment. The standard-library verifier above avoids importing GPU/API dependencies and verifies all eight conditions.

## Interpretation boundaries

- Each table row is one training seed, not a five-seed aggregate. Historical controls are not a fully matched inference cohort.
- MBPP self-distilled uses 220 references; MBPP-original uses 221. The excluded ID and failed attempts are preserved in `generation_export/mbpp_exclusion_20261021.json` and the raw generation archive.
- Equal training steps and KL coefficient do not match reference lengths, KL-token exposure, or effective constraint strength. Source/domain and persona interpretations remain hypotheses, not isolated causal conclusions.
- Preserve per-source licenses and attribution. The pinned TruthfulQA source is Apache-2.0; MBPP is CC-BY-4.0. The local/publication manifests retain source revisions and selection provenance.
