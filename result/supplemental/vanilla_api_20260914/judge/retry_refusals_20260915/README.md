# Refusal retry: complete

The user requested another attempt at the 188 REFUSAL-labeled judgments from
the new vanilla judge pass. This separate pass preserves the original results.
After a safety-review pause, the user explicitly approved resending these
prompts, rubrics and model outputs to Alibaba via OpenRouter. Read-only comparison
confirmed all 188 frozen retry payloads exactly match the previously sent payloads.

## Results

All 188 judgments were retried and verified. **26 returned numeric scores;
162 remained REFUSAL labels.** No requests are pending and none were provider-
filtered. There were 190 API attempts, including two invalid-format responses
that recovered on bounded retries. All costs were API-reported: **$0.0100199304**.

| Model | Retried | Numeric on retry | Still REFUSAL |
| --- | ---: | ---: | ---: |
| Llama | 162 | 14 | 148 |
| Qwen | 17 | 10 | 7 |
| Olmo | 9 | 2 | 7 |
| Total | 188 | 26 | 162 |

The final audit passed: no missing/duplicate terminal IDs, frozen payload and
provider agreement, parser consistency, raw/exported score agreement, and file
hash verification. Original result-file hashes were unchanged. `comparison.json`
links each retry result to its original event. Neither numeric retry results nor
repeated refusals have replaced the original pass in the primary analysis.

## Protocol and reproduction

Scope: 162 Llama, 17 Qwen, nine Olmo primary alignment judgments. Numeric scores,
coherence scores, provider-filtered inference outputs, and the historical Qwen
medical run are excluded. The selection test verified these exclusions.

Settings are unchanged: DeepSeek-v4-flash, Alibaba only, no provider fallback,
temperature 1, top-p 1, maximum 256 tokens, reasoning effort none. This pass has
a $0.25 local budget guard. The first eight calls cover all model/task groups;
remaining calls use eight-way concurrency if that check passes.

This is one new judging pass, not repeated attempts until a numeric score is
obtained. A repeated REFUSAL is terminal. Transient failures and invalid formats
retain the existing bounded retry policy. All attempts will be saved, with a
comparison to each original result. No automatic replacement or promotion into
the primary analysis is performed: outcome-selected rejudging can bias results.

`requests.jsonl.gz` contains the exact selected requests and model outputs.
`selection.json` links each request to its original event; `config.json` freezes
settings, source hashes and archived producer code. Successful execution adds
raw `judge_events.jsonl.gz`, normalized `scores.jsonl.gz`, `artifacts.json`,
`verification.json` and `comparison.json`. Actual gzip data is included through
Git LFS with the repository publication.

Commands used after explicit resend approval (already complete; rerunning the
runner skips terminal retry results):

```bash
python scripts/eval/retry_vanilla_refusals.py run --stage canary --env-file /path/to/private.env
# Inspect canary_passed.json before the remaining stage.
python scripts/eval/retry_vanilla_refusals.py run --stage remaining --env-file /path/to/private.env
# Wait for the process to exit, including exports, before verifying.
python scripts/eval/retry_vanilla_refusals.py verify
```

Do not recreate the prepared pass or modify its frozen producer source. The
original judge engine is reused without edits; all paths for new results point
to this separate retry directory.
