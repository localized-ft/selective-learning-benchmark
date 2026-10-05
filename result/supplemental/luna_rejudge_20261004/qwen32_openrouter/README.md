# Qwen3-32B Luna rejudging — direct OpenRouter

Completed on 2026-10-05: **15,520/15,520 judgments**, including 15,488 numeric
scores and 32 REFUSAL labels retained as missing scores. All 7,760 coherence
scores are numeric. There were no API retries or unreported-cost attempts.
Total reported cost: **$1.80644**. Offline request, response parsing, provenance,
uniqueness, and coverage checks passed; see `verification.json`.

Scope: all 14 seed1 KL/IP task checkpoints, including the recovered medical IP
adapter: 7,760 completions, 15,520 task/coherence requests. DeepSeek results are
preserved, not replaced. See `status.json` for progress and `verification.json`
for final completion and integrity checks when available.

## Protocol and differences

Uses `openai/gpt-6-luna` directly through OpenRouter. The upstream workflow's
`openrouter/openai/gpt-6-luna` name includes a LiteLLM routing prefix that is
removed for direct calls. The failed proxy attempt produced no scores and is
not included here. No credentials are stored in these artifacts.

Upstream prompt rendering, parsing, and think-block stripping are reused from
the pinned `upstream/sunday_eval/` sources. Temperature, reasoning effort and
provider routing are left at defaults, as in the upstream runner. The response
limit is 2,000 tokens. Provider routing/defaults may differ when bypassing the
LiteLLM proxy, and the time of judging also differs. The serving model identity
and any usage/provider information returned by OpenRouter are in raw responses.

Each completion uses the task rubric from its frozen evaluation snapshot. The
coherence rubrics already backfilled for this cohort are preserved on both axes.
Think blocks are stripped only from the judge input; original inference data
remains unchanged. Unlike the upstream 200-character export, full HTTP response
bodies and parsed source text are retained here.

56 canaries cover all 14 checkpoints × two axes × two score types. Bulk judging
starts only after those checks pass. Concurrency is 100. Transport/HTTP failures
may retry up to five attempts; parsed outcomes (including REFUSAL, CODE, or parse
failure labels) are never selectively retried to obtain different scores. A $25
guard uses reported API costs; it is not a provider hard cap. Request/retry
counts are also bounded. All attempts are persisted incrementally.

## Files

- `requests.jsonl.gz`: exact rendered inputs, stripped answers, original-answer
  hashes, run/task IDs, parser specifications and source metadata.
- `events.jsonl.gz`: raw requests/responses, attempt status and reported costs.
- `scores.jsonl.gz`: one final outcome per judgment, with full judge source text.
- `summary.json`: per-run raw and coherence >=50 filtered axis means, computed
  using the upstream aggregation helper. Labels remain missing, not zero.
- `config.json`, `canary.json`, `artifacts.json`, `verification.json`: settings,
  provenance, preliminary checks, data hashes and offline integrity checks.

Use `git lfs pull` to retrieve compressed data after cloning. From repository
root, verify offline with `python scripts/eval/verify_qwen32_luna.py`. The main
five-seed analysis/plots have not been recomputed with these supplemental runs.
