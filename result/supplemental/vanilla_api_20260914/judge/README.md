# Vanilla judging: complete

The user explicitly approved sending the benchmark prompts, rubrics and model
outputs to Alibaba through OpenRouter under the $10 local budget guard. The
initial safety-review block was resolved by that approval, not a workaround.
Credentials remain in the private local environment, not this archive.

The 80-call compatibility batch passed: 78 numeric scores and two REFUSAL labels,
all from Alibaba, all normal stop finishes, zero reasoning tokens, no parsing or
API errors. Reported cost was $0.007591234. The full batch subsequently completed;
`status.json` records zero pending requests, and final scores and artifact hashes
have been exported.

## Final results

All **21,746 new judge requests** have terminal results, covering **10,873
completions**. Every completion has its primary result and a numeric coherence
score. No coherence filtering was applied to the raw results.

| Model | Completions | Numeric scores | REFUSAL labels | Pending |
| --- | ---: | ---: | ---: | ---: |
| Llama | 3,880 | 7,598 | 162 | 0 |
| Qwen, six new tasks | 3,113 | 6,209 | 17 | 0 |
| Olmo | 3,880 | 7,751 | 9 | 0 |
| Total new | 10,873 | 21,558 | 188 | 0 |

All 188 REFUSAL labels are primary alignment results. They remain missing
numeric scores, not zeros. No judge requests were provider-filtered. Historical
Qwen medical contributes another 759 completions / 1,518 already-saved judgments
without any new calls, bringing the total reference set to 11,632 usable
completions and 23,264 primary/coherence result records.

A subsequent user-requested [refusal retry](retry_refusals_20260915/README.md)
retried these 188 labels with identical payloads: 26 returned numeric scores and
162 remained REFUSAL. That diagnostic pass is stored separately; this original
pass and the primary analysis remain unchanged.

A later approved [DeepInfra comparison](retry_deepinfra_20260915/README.md) sent
the 162 remaining refusal-labeled judgments to the same model through DeepInfra:
ten returned numeric scores and 152 remained REFUSAL. This pass is also separate
and has not been promoted into the primary results.

The runner made 22,166 attempts: 21,746 terminal results plus 89 invalid-format
attempts, 43 API errors and 288 transport errors. Bounded retries recovered all
unfinished requests; no provider, rubric or scoring settings were changed.
All attempts are preserved. Reported cost was **$2.4558520628**; 331 failed
attempts had unreported costs, so this is not a reconciled final bill.

Actual raw data: `judge_events.jsonl.gz` (22,166 attempts), `requests.jsonl.gz`
(21,746 frozen payload specifications), and `scores.jsonl.gz` (21,746 normalized
score records). `artifacts.json` records compressed/decompressed hashes and
row counts. The final offline audit passed (`verification.json`): no missing or
duplicate terminal request IDs, exact frozen payload/provider/parser agreement,
raw/exported score agreement, and compressed/decompressed hash verification.
An additional pair audit confirmed exactly one primary and one coherence result
per completion, with all 10,873 new coherence scores numeric.

Wait for the runner process to exit before final verification. Its `running`
flag becomes false just before exports finish; an early verification attempt
can see the old stage's hash index. In this execution, that early check failed
on a hash mismatch and was rerun after the process exited. No raw data was
modified to address it.

## Scope and protocol

- 10,873 usable completions: 3,880 Llama, 3,113 new Qwen, 3,880 Olmo.
- Two judge calls each, primary task score and coherence: 21,746 requests.
- Reuse the historical Qwen medical run's 759 completions / 1,518 judgments;
  do not pay to rejudge them or merge their IDs without model/task scoping.
- The seven new Qwen inference provider failures are listed in
  `excluded_inference.json`. The eighth, historical medical failure remains
  in its original archive. These are not usable outputs and are not scored zero.
- Judge: `deepseek/deepseek-v4-flash`, Alibaba only, no provider fallback.
  The public endpoint snapshot identifies `alibaba/fp8`.
- Temperature 1, top-p 1, maximum 256 output tokens, reasoning effort `none`.
  The exact request body is frozen in `config.json`.
- Use frozen evaluation rubrics verbatim and the migrated renderer/parser.
  Missing capability coherence uses the archived `coherence_rubric.txt`;
  each request records whether the rubric came from the snapshot or fallback.
- Qwen is judged on the already reconstructed and capped inference text;
  no new stripping of reasoning, truncation, or generation occurs here.
- Numeric alignment/coherence scores must be finite and within 0–100.
  Factual labels use each rubric's original regex/map to 0–1. Literal CODE
  and REFUSAL are terminal missing-score labels, never silently scored zero.
- Both axes receive coherence regardless of primary score. No coherence
  filtering is applied to raw data; later reports can use both pre-existing
  current and paper-style filtered/unfiltered estimands.

## Execution and preservation

The compatibility batch is 80 calls, one for each model/task/axis/score-type
combination. These count toward the full batch. Successful results are not
resubmitted. After the compatibility check passes, remaining calls use 32-way
concurrency. Invalid formatting and transient failures have bounded retries,
at most five attempts per ID. Valid scores, CODE/REFUSAL labels, and provider
filters are never retried. Unexpected provider/reasoning, authentication,
parameter, or budget errors stop execution. A process lock prevents duplicate
concurrent runners.

The local budget guard is $10, with $0.003 reserved per in-flight request.
It uses reported cost, falling back to token-based estimates. Unreported costs
are disclosed separately; this is not a provider-side hard cap.

`requests.jsonl.gz` contains actual rendered prompts, completion text, parser
specifications, IDs, and provenance, not merely links. `config.json` records
source hashes and content-addressed producer source archives. Upon execution,
`judge_events.jsonl.gz` will retain every raw request/HTTP response and attempt;
`scores.jsonl.gz` will export normalized per-score records. Status, hash index,
and final verification reports are generated separately. Raw gzip files are
covered by Git LFS rules and included with the repository publication.

Offline tests passed: 24 vanilla tests, including complete input/rubric coverage,
no duplicate IDs, seven inference-failure exclusions, no rejudging of medical
Qwen, parser handling and Olmo final-upload file-ID deduplication.

Execution commands (approval has been granted):

```bash
python scripts/eval/vanilla_judge.py run --stage canary --env-file /path/to/private.env
# Inspect canary_passed.json and raw responses before full execution.
python scripts/eval/vanilla_judge.py run --stage remaining --env-file /path/to/private.env
python scripts/eval/vanilla_judge.py verify
```

Do not rerun `prepare`: its existing inputs are frozen. Keep the archived
producer source unchanged during execution; input/source hash checks reject
unrecorded changes. No inference jobs, historical scores, or reports were changed.
