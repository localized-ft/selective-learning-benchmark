# Selective Learning Benchmark

This repository contains the committed training/evaluation pipeline snapshot and
the actual archived experimental data, together with a portable offline analysis
implementation. It compares task acquisition with unintended generalization.

Start with the [updated analysis and plots](result/releases/with_vanilla_20260915/REPORT.md)
and the [results guide](result/README.md). The
[original five-seed report](result/releases/five_seed_20260908/REPORT.md) remains frozen.
The main release contains **630 trained-model evaluations**: seven tasks, three
model families, six configurations, and all five training seeds. The
[three-model vanilla reference](result/supplemental/vanilla_api_20260914/README.md)
covers all seven tasks, with 11,632 usable completions and primary/coherence
judgments. The updated release compares these references with all five trained
seeds in all four scoring/filtering views. Vanilla is not another training seed;
its inference uncertainty is not represented by the training-seed intervals.
Refusal-retry passes remain separate diagnostics.

## Get the actual raw data

Git LFS is required for the raw objects under `result/artifacts/`:

```sh
git clone git@github.com:localized-ft/selective-learning-benchmark.git
cd selective-learning-benchmark
git lfs install --local
git lfs pull
git lfs fsck
```

Repository access is currently private. No OpenWeights account, Hugging Face
token, or judge API key is needed to read the archived data or reproduce the
analysis. A source ZIP or checkout containing only LFS pointers is incomplete.
The supplemental API request/response and judge files also use Git LFS;
`git lfs pull` retrieves their actual bytes along with the main archive.

## Reproduce offline

Set up Python 3.11 or newer and the pinned numerical dependency, then run:

```sh
python -m pip install -r scripts/analysis/requirements.txt
python -B -m unittest discover -s scripts/analysis -p 'test_*.py' -v
python -B scripts/analysis/reproduce.py --output reproduced --verify-archive
```

Use a new empty output directory. The analysis disables network calls and
regenerates all four scoring variants, completion/prompt scores, seed-bootstrap
intervals, Pareto results, and sensitivity diagnostics from raw judgments and
saved coherence backfills. It checks the regenerated tables against the frozen
release at `1e-12` tolerance and writes a verification report and SVG figures.
The original report, PNG figures, and workbook are retained unchanged as release
artifacts; newly rendered figures need not have identical image bytes.

To regenerate the updated vanilla-inclusive analysis, tables, and 13 SVG figures:

```sh
python -B scripts/analysis/with_vanilla.py --output reproduced-with-vanilla
```

This first verifies all 18 historical tables, then joins the original vanilla
judge pass, preserving missing-score/provider-failure denominators and adding
vanilla-relative contrasts and a common-prompt sensitivity analysis. It does not
run inference or spend API credits.

To recover files with their original names and formats for your own analysis:

```sh
python scripts/analysis/archive.py extract --destination /tmp/slb-raw
```

This restores the archived `sunday/scripts/eval/results/` hierarchy from the
repository objects. Use `--prefix openweights/files/` for recovered remote
inference/judge output files or `--prefix derived/legacy_completions/` for labeled legacy
extractions. See the [data dictionary](docs/RESULT_DATA_DICTIONARY.md).

## Code and historical limitations

The migrated files in `scripts/finetune/` and `scripts/eval/` remain byte-identical
exports from source commit `60c45238b8bac0cec53a2a518f3945a68b39ae83`.
New vanilla inference/judging and provider-comparison scripts were added separately
after migration; their protocols and exact producer snapshots are archived with
the supplemental results. Local-only fixes and untracked
training configs were intentionally omitted. `scripts/analysis/` is a new,
independent offline reference implementation, not a copy of the untracked
analysis scripts. `scripts/migration/` contains the new auditable migration tools.

Do not launch fresh training/evaluation as an exact historical replication without
reviewing the [omitted fixes and execution gates](docs/MIGRATION_AND_RESULTS_PLAN.md).
In particular, the committed training loader does not support the local dedicated-IP
dataset override, and the committed judge does not guarantee coherence for every
axis. Future vanilla evaluations also require an audit of checkpoint-dependent
generation defaults. No new evaluations were launched by this migration.

Read [source attribution and license scope](NOTICE.md) before redistributing.
