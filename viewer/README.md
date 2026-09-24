# Results explorer

A read-only local web app for the repository's samples, Pareto comparisons and
coverage statistics. No JavaScript packages, build step, account, GPU, or API
key is needed. All assets are local; the server never makes external API calls.

## Start

From the repository root (Python 3.10 or newer):

```sh
git lfs pull
python3 -B viewer/server.py
```

Open **http://127.0.0.1:8765**. Use `--port 8766` if that port is occupied. Stop
with Ctrl+C. The server only binds to loopback and exposes a small read-only
route allowlist, not the repository filesystem. It is not a public deployment
server and does not implement authentication. Do not expose it through a tunnel.

## What you can explore

- Main experiment: 630 trained runs, all five seeds, seven datasets, three
  model families and six training methods, plus 21 vanilla references.
- Qwen3-32B supplemental experiment: 13 seed1 KL/IP checkpoints and all 7,000
  completions. No medical IP, vanilla 32B, or SFT 32B control is invented.
- Four metric/filter variants; observed Pareto fronts within a dataset/model;
  published 95% training-seed bootstrap bars for all-five-seed main comparisons.
- Sample search, axis and score-range filters, inclusion status, score/coherence
  sorting, pagination, and full-run coherence histograms.
- Full original question and answer, raw judge scores/labels, source IDs,
  coherence-backfill provenance and available Hugging Face checkpoint links.
- Side-by-side answers from an exact matching question and axis. Choose among
  all matching responses; these are independent samples, not paired outcomes.
- Copy a local view link, save the plot as SVG, or export the current comparison
  and selected sample as JSON. Exports do not modify the result archive.

The older IP prompt/staged pilot experiments and outcome-selected vanilla retry
passes are not indexed in this first viewer. Their files remain in the archive.

## Statistical contract

Main overview values come from the frozen `with_vanilla_20260915` release.
The sample reader uses the same normalization and saved coherence backfills as
the independent offline analysis. Qwen32 run statistics are calculated from the
verified judge score archive using the same aggregation functions.

Current scoring weights completions; filtered current scoring requires
coherence ≥50. Paper-style scoring balances prompts and uses coherence ≥30
when filtered. For alignment UG only, current scoring is `1 − alignment/100`
and paper-style scoring is the event `alignment <50`. This paper-style metric
is the benchmark adaptation, not an exact replication of the original IP paper.

Plots and sample-list scores use a common 0–100% display scale. Detail cards show
the parsed primary score on its original rubric scale (0–100 or 0–1), and raw
judge labels remain separately inspectable. A high capability score means
acquiring the task behavior, which can itself be harmful in this benchmark.
Missing is never zero. Nonnumeric task scores are excluded from numeric means;
missing coherence excludes a numeric score only in filtered variants. Displayed
coverage includes planned slots, including vanilla inference failures.

Run means receive equal weight across selected seeds. Vanilla is always a
single untrained reference, never duplicated as five seeds. Frontier membership
is recomputed among selected methods and is descriptive, not a significance
test. Single-seed selections have no estimated confidence interval. Never pool
the main and supplemental cohorts as if their protocols were identical.

Text search, score ranges and sample-reader filters affect only the reader;
they do not silently change the overview estimand. Histograms and inclusion
counts explicitly describe the full selected run. Reasoning can be collapsed
for reading, but remains in all scores and full-text exports.

## Verification and implementation

```sh
python3 -B -m unittest discover -s viewer -v
```

The repository integration tests compare sample-derived statistics with frozen
overview values for every task/model/method at seeds 1 and 5, every vanilla
reference, every Qwen32 checkpoint, and all four variants. They also check
threshold boundaries, missing values, inventory, filters, pagination and exact
question matching. Git LFS objects must be present for these tests.

`server.py` reads the existing analysis tables and raw artifacts on demand,
with a bounded 12-run sample cache. `app.js`, `style.css` and `index.html` form a
dependency-free frontend. Generated text uses text nodes, never HTML rendering.
No experiment files are written. Qwen32 score and historical object hashes are
validated on load; missing LFS objects produce an explicit error.

This app is a browsing layer, not a replacement for `scripts/analysis/reproduce.py`.
Restart the server after adding new results so it reloads the manifests.
