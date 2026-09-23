# Qwen3-32B seed1 inference

## Completed results

All 7,000 inference outputs and all 14,000 judge requests are complete and
integrity-verified. Judging produced 13,955 numeric scores and 45 nonnumeric
labels, which remain missing scores rather than zeros. No requests were provider
filtered. Reported judge cost is $2.1253; 57 API-error attempts had no reported
cost. The H200 continuation took 5,775.9 seconds (96.3 minutes).

Raw inference outputs and token-level details are included as content-addressed
gzip objects under `result/artifacts/sha256/`; their combined manifest is
`h200_resume/combined_artifacts.json`. Raw judge requests/responses, parsed scores,
costs, and verification are under `judge/`. All large objects use Git LFS; after
cloning, run `git lfs pull` to retrieve the actual data rather than pointer files.
The existing five-seed analysis release has not been recomputed with this
single-seed supplemental cohort.

## Local LLM judging

`judge/` holds the frozen inputs and resumable results for all 7,000 completions:
14,000 requests (one task score and one coherence score each). The judge is
`deepseek/deepseek-v4-flash` through Alibaba, with provider fallback disabled,
temperature/top-p 1, reasoning disabled, and a 256-token response limit. Original
task rubrics, parser, full decoded answers and missing-score policy are reused.
Coherence is scored for both axes; no answers are dropped before judging.

The local runner tests 52 canaries before starting bulk calls at concurrency 32.
It has a $10 local cost guard (not a provider-side hard cap). Raw responses,
attempts, costs and errors are persisted incrementally in `judge_events.jsonl.gz`;
`scores.jsonl.gz` and `artifacts.json` are exported at each stage's end.
`status.json` reports progress. CODE/REFUSAL and provider filters remain missing,
not zero, and are not selectively retried for numerical scores.

```sh
python scripts/eval/qwen32_judge.py status
python scripts/eval/qwen32_judge.py verify
```

The inference count is verified at 7,000, including 365 length-limited outputs
and no empty outputs. Judging includes all of them unchanged. The complete
configuration, provenance hashes and endpoint snapshot are in `judge/config.json`.

## H200 continuation

The A100 attempt saved 1,840 outputs before its worker terminated. Its pending
job `jobs-6de87ee708c8` was explicitly canceled before preparing a replacement.
`h200_resume/` preserves and validates those outputs by completion ID and submits
only the remaining 5,160 requests across 11 checkpoints. Original inputs and
artifacts in `full/` remain unchanged. No score-based selection is performed.

The continuation uses one H200, 32 concurrent sequences, CUDA graphs enabled,
4,096 batched tokens and 90% GPU memory utilization. FP16, tokenizer, checkpoint
revisions, prompts, per-request seeds, and all sampling parameters are unchanged.
This is a mixed-hardware cohort; exact numerical equivalence is not assumed.
The first regular batch tests loading/generation; the job stops on failure.

Use `--phase h200_resume` with the `submit`, `collect`, and `verify` commands.
After completion, verification checks both phases together for exactly 7,000
unique completions and creates `h200_resume/combined_artifacts.json` pointing to
the combined raw completions and token-level details. Reused source artifact
IDs, remote run IDs, and config hashes are retained in the resume config.

This cohort evaluates the 13 completed 32B checkpoints: all seven KL tasks and
six non-medical IP tasks. There is no medical IP, standard SFT, or vanilla 32B
checkpoint in this cohort. Do not label the historical 8B comparison models as
32B baselines. No judging is launched by this inference script.

## Protocol

- Frozen historical evaluation prompts; 10 completions per prompt, 7,000 total.
- IP training system prompts are **absent** at evaluation, as in the main benchmark.
- FP16, no quantization; temperature 1, top-p 1, top-k 50, maximum 2,000 generated
  tokens, context 4,096, thinking enabled. Full decoded output including reasoning
  is retained; empty or length-limited answers are not resampled or filtered.
- Public merged checkpoints are pinned to the revisions reported by their
  successful training jobs. Training IDs and revision evidence are in the config.
- The canonical tokenizer is the pinned Qwen/Qwen3-32B base tokenizer. Native
  rendering and token IDs must match for every request before model loading.
- Matched deterministic prompt/sample seeds across KL and IP, using the existing
  vanilla cohort seed convention. These are inference samples, not five training seeds.

## Hardware and execution

One sequential OpenWeights job, restricted to one A100 80GB / A100 SXM 80GB.
The approximately 61 GiB of FP16 weights do not require an H200 for inference.
Serving uses eager execution, eight concurrent sequences, 2,048 batched tokens,
and 92% GPU memory utilization. These scheduling settings differ from the 8B
pilot to fit 80GB; numerical outputs are not claimed bit-identical across GPUs.
Actual runtime/GPU/engine settings are saved as environment artifacts.

Each model runs in a subprocess with a job-owned temporary HF cache. Child exit
releases CUDA state and only that temporary cache is removed; shared caches and
credentials are untouched. Outputs and token-level details are uploaded every
64 completions and again on exit. The first ordinary batch provides the hardware
check; failure halts the job instead of silently changing precision/settings.
The job has an eight-hour timeout and a nine-hour pod TTL. These are limits,
not a runtime/cost estimate. Completed model artifacts survive later failures.

## Files and commands

`full/config.json` pins checkpoints, training jobs, snapshots, sampling, and GPU
settings. `full/requests.jsonl` contains every actual request. `full/submission.json`
records the OpenWeights job ID and content-addressed copies of uploaded inputs.
Collection writes `full/status.json` and `full/artifacts.json`; raw output bytes
are stored in `result/artifacts/sha256/` as deterministic gzip, using the same
format as other cohorts. Final verification checks coverage, unique IDs, output
token counts, metadata and identical prompt tokens across methods.

From the repository root, use the Python environment with OpenWeights/httpx:

```sh
python scripts/eval/qwen32_inference.py collect --env-file /path/to/openweights.env
python scripts/eval/qwen32_inference.py verify
```

The submit command is idempotent and never restarts failed/canceled jobs. Any
retry should target only unfinished checkpoints and preserve original receipts.
Future analysis must report the missing 32B controls and single-seed scope.
