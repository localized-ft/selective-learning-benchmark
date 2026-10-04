# Re-judging the selective-learning results with gpt-6-luna

The paper judge (`openrouter/deepseek/deepseek-v4-flash`) is very noisy on overtly harmful
answers: an explicit "forced sterilization" answer got alignment 15 / 45 / 55 on three calls, and
gpt-6-luna gives it 0. This folder re-judges every completion behind the write-up (plus the
`niels/experiment-ip-variants` evals) with `openrouter/openai/gpt-6-luna`. Completions, rubrics
and aggregation stay unchanged.

## What gets re-judged (`inventory.py` -> `state/inventory.json`)

| source | what | runs |
|---|---|---|
| `seed1` | July `eval_worker` jobs referenced in `sunday/scripts/eval/plot_v2_tradeoffs.py`, plus 3 Llama seed-1 runs (risky-financial IP / KLD, reward-hacks IP) missing from that file | 164 |
| `judge` | Aug/Sep `judge_worker` jobs: seeds 2-5 of baseline + layer thirds | 358 |
| `completion` | `completion_worker` jobs for KLD / IP seeds 2-5 (judged outside OpenWeights, so no old per-run summary) | 168 |
| `ip_variants` | IP-variants experiment eval targets | 42 |

The 630 write-up runs are complete: 7 tasks x 3 models x 6 conditions x 5 seeds, one run per
model id. The scope (`rejudge: true`) is those 630, the 5 plotted seed-1 probe points and the
42 IP-variant targets. Excluded: bf16 / Qwen3-32B extras, duplicate evals of the same model, and
the dropped tasks (counterfactual facts, single-fact good-vs-bad).

## How (`rejudge.py`)

- Each run's requests are rebuilt from **that run's own eval file**, so the judge prompts are exactly the
  ones originally used. Scoring uses Sunday's `JudgeRunner`, and axis summaries use Sunday's
  `add_axis_score_summary` (raw mean + coherence>=50 filtered mean), so old and new numbers are aggregated
  identically.
- Deviations from the paper pipeline: the judge model, and Qwen3 `<think>` blocks are stripped before
  judging.
- Resumable: files are cached in `state/files/`, and per-run scores go to `state/scores/`. A run is saved
  only if <2% of its judge calls errored.
- Outputs (committed):
  - `results/rejudge_summary.csv`: one row per run with old (deepseek) and new (luna) capability / UG means.
    Old values come from each job's `eval_summary`; for IP-variant runs they are recomputed from `judge.py`'s scores.
  - `results/analysis.txt`: the write-up's headline claims recomputed with luna (`analyze.py`).
  - `results/luna_scores.jsonl.gz`: every luna judgement (658k rows; `run_file` = per-run key).
- The judge client uses a 120 s timeout: the proxy occasionally returns Cloudflare 524s and leaves
  requests hanging, which otherwise stalls the run.

```
../experiment-ip-variants/.venv/bin/python inventory.py
../experiment-ip-variants/.venv/bin/python rejudge.py [--task T] [--source S] [--summary-only]
```

## Data issue found

The July (seed-1) eval files for German city names and old bird names have no `answer_regex` /
`score_map`, so the judge's "LLM" / "19th century" labels were never turned into scores; the stored
seed-1 summaries for those tasks are artifacts (old and new judge alike). The inventory maps those runs
to the August eval files (same ids, questions and prompts, plus the maps); see `PERSONA_EVAL_FILE_FIX`.

## Results

See `results/analysis.txt` and the "IP review" tab of the planning doc. In short: run-level agreement
with deepseek is r = 0.94-1.00 except reward-hacks misalignment (0.51). The qualitative claims hold, but
KLD's EM misalignment reduction drops from 71% to ~46%, and the Table 3 "KLD decouples capability and
misalignment" result is not robust.

## Pilot (42 IP-variant runs)

Across runs, luna vs deepseek: capability r = 0.99; coherence-filtered alignment r = 0.84. Luna is
stricter on alignment (e.g. the inoculation-adapter-only models: 4-9 vs 65-70). It is also stricter on
factual accuracy in the medical capability rubric: untrained models get 35-43 vs 8-21, often for
genuinely wrong advice, occasionally harsh. The ranking of runs is preserved.
