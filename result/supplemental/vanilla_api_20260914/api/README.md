# Remaining vanilla API inference

Scope: finish inference before running any new judge calls. Llama-3.1-8B-Instruct
is evaluated on all seven frozen task snapshots (3,880 requests). Qwen3-8B is
evaluated on the other six tasks (3,120 requests); its historical bad-medical
run is preserved separately, including 759 usable outputs and one provider-
filtered failure. The filtered request is not silently retried or scored zero.

## Final status

Inference is complete. Offline validation passed for both models: all expected
IDs have one terminal outcome, no usable completion is empty, raw/exported
texts match, provider pins match, and compressed/decompressed hashes verify.

| New run | Returned / planned | Usable | Provider-filtered | Reported cost |
| --- | ---: | ---: | ---: | ---: |
| Llama, seven tasks | 3,880 / 3,880 | 3,880 | 0 | $0.30358636 |
| Qwen, six tasks | 3,120 / 3,120 | 3,113 | 7 | $1.726063508 |

Five Llama and four Qwen API-error attempts recovered with bounded retries;
their costs were unreported. Qwen's new filtered failures were old bird names
(1), risky financial advice (3), and school of reward hacks (3). The historical
medical failure is additional. All failures remain in the raw archive.

Llama has one length-capped output. Qwen has four API length finishes and 402
outputs truncated by the documented combined-text client cap. These outputs
are retained. Detailed task counts and checks are in each `verification.json`.
No judge calls were made during inference; subsequent judging is archived in
`../judge/`. The actual raw inference files are included through Git LFS.

## Protocol and execution

- Llama: `meta-llama/llama-3.1-8b-instruct`, CoreWeave BF16 only.
- Qwen: `qwen/qwen3-8b`, Alibaba only; provider quantization is unknown.
- Temperature 1, top-p 1, top-k 50, maximum 2,000 API output tokens, presence
  and frequency penalties 0. Provider fallbacks are disabled, and parameter
  support is required. Exact payloads and provider responses are preserved.
- Each request carries a deterministic inference seed. These are not training
  seeds and do not guarantee identical API output across time or backends.
- Qwen reasoning is enabled and preserved. Thinking and answer are reconstructed
  as `<think>\n...\n</think>\n\n...`, then capped to 2,000 retokenized tokens using
  the historical pinned tokenizer and `tokenizers==0.21.4`. Full original API
  responses are retained, including text beyond the cap. An offline regression
  check reproduced **all 759** historical medical completion texts exactly.
- Llama API-native chat rendering and stop handling cannot be independently
  verified. BF16 differs from historical FP16 inference; API versus GPU and
  model-specific decoding differences remain comparison limitations.

One request per task/axis runs first (14 Llama, 12 Qwen) as a compatibility
check. Full execution skips successful check IDs. Provider-filtered outputs
are terminal failures, not retry targets; transient network/server/rate-limit
failures have bounded retries with backoff, at most five attempts per ID.
Authentication, budget, parameter or unexpected-provider failures stop that
model's execution. No alternate provider is silently substituted.

Compatibility checks passed: 14/14 Llama responses from CoreWeave and 12/12
Qwen responses from Alibaba, with no empty outputs; all Qwen check responses
contained reasoning. Full execution uses 16 concurrent Llama and 32 concurrent
Qwen requests. Qwen's prepared default of eight was raised after the successful
two-concurrent-request check; this changes throughput only, not payloads or
provider routing. Each request still uses bounded rate-limit backoff.

Local budget guards are $4 for Llama and $8 for Qwen. They use API-reported
costs where available, otherwise token-based estimates, with reservations for
in-flight requests. They are **not** provider-side hard spending caps. Missing
cost reports, including failed network requests, are disclosed separately.

## Actual data files

Each model directory contains frozen `requests.jsonl`, `config.json` and an
endpoint-catalog snapshot. During execution every response attempt is flushed
to `inference_events.jsonl.gz` as a complete gzip member. It contains request
payloads and full raw HTTP response text, but no authorization headers or keys.

`status.json` reports terminal/usable/failed/pending counts and costs. At the end
of each stage, `completions.jsonl.gz` contains usable three-field completion
records, `completion_status.jsonl.gz` records per-ID provenance and failures,
and `artifacts.json` records hashes and sizes of the actual gzip files. These
files are Git LFS-backed by repository rules; generating them does not push
them to GitHub. `execution_producer.json` references archived exact source bytes
and records runtime package versions.

These are supplemental inference results, not additions to the historical
630-run training cohort. Neither raw outputs nor failures are removed by
coherence filtering, and no new LLM-judge calls are part of this execution.

```bash
python -m pip install httpx==0.28.1 python-dotenv==1.1.1 tokenizers==0.21.4
python scripts/eval/vanilla_api.py prepare
python scripts/eval/vanilla_api.py run --stage canary --env-file /path/to/private.env
python scripts/eval/vanilla_api.py run --stage remaining --env-file /path/to/private.env
python scripts/eval/vanilla_api.py status
```

Run at most one process per model. Successful and provider-filtered IDs are
never resubmitted by the runner. Stop an existing process before resuming it.
