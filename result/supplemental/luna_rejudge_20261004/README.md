# Luna rejudging: imported upstream results

Imported unchanged from `nielsrolf/spar-localized-finetuning` commit
`0da4db56cfeea479afb387282defbe57e4b21209`, directory
`niels/experiment-rejudge-luna`. See `import_manifest.json` for every file hash.

The actual inventory selects 698 runs: 630 main-matrix runs, five additional
probe runs, and 63 IP-variant runs. The archive has 657,960 score rows across
698 run files. These counts supersede the older counts in the preserved upstream
README. All selected run keys are represented in the score archive.

Contents under `upstream/experiment-rejudge-luna/` include raw exported scores
(`results/luna_scores.jsonl.gz`), the upstream comparison table and analysis,
inventory, and producer scripts. The exact upstream Sunday judge dependencies
are preserved under `upstream/sunday_eval/`. Git LFS stores the compressed scores;
run `git lfs pull` after cloning. No new judging calls were made for this import.

Protocol: `openrouter/openai/gpt-6-luna` through the upstream LiteLLM proxy,
2,000 response-token limit, per-run original evaluation grading specifications,
Qwen think blocks stripped, and raw plus coherence >=50 filtered aggregation.
Temperature/reasoning settings are not explicitly overridden by that runner.
This is separate from the DeepSeek cohort, which retained reasoning text.

Limitations: upstream `judge_raw` is truncated to 200 characters; full HTTP
responses are not available in this export. It contains 38 ERROR score rows;
the original runner allowed up to 2% errors per saved run. Preserve these as
missing results, not zeros. The imported scripts retain their original repo
layout assumptions; their presence documents provenance rather than providing
a standalone runner in this relocated directory.

Qwen3-32B is explicitly excluded from this upstream inventory. The separately
generated [14-run Qwen3-32B extension](qwen32_openrouter/README.md) covers 7,760
completions with 15,520 task/coherence judgments, calling Luna directly through
OpenRouter. Its protocol differences, raw responses and completion verification
are documented separately. No original DeepSeek results or main analysis
releases have been overwritten.
