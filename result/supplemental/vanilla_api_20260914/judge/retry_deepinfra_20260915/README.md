# DeepInfra comparison: complete

The user explicitly approved sending the 162 remaining prompts, rubrics and model
outputs to DeepInfra via OpenRouter under a $0.25 local budget guard. These were
the judgments labeled REFUSAL in both the original Alibaba pass and its separate
user-requested retry. All prior results remain unchanged.

## Results

All **162** requests have a result: **10 numeric scores and 152 REFUSAL labels**.
No provider-filtered responses, missing IDs or duplicate terminal IDs occurred.
The 163 API attempts include one invalid-format response recovered by a bounded
retry. All costs were API-reported: **$0.005181552** (about half a cent).

| Evaluated model | Retried | Numeric on DeepInfra | Still REFUSAL |
| --- | ---: | ---: | ---: |
| Llama | 148 | 6 | 142 |
| Qwen | 7 | 3 | 4 |
| Olmo | 7 | 1 | 6 |
| Total | 162 | 10 | 152 |

Thus 26 of the original 188 REFUSAL judgments returned numbers on the Alibaba
retry, and ten additional judgments returned numbers on this DeepInfra pass;
152 remained REFUSAL. These are separate diagnostic outcomes, not replacements
for the original primary results.

Most labels persisted (152/162, 93.8%). This is not a controlled provider ranking:
DeepInfra received only labels remaining after a prior retry, and the judge uses
temperature 1. Changes can reflect sampling as well as provider differences.
REFUSAL is a rubric-defined judge label about the evaluated answer, distinct
from provider-side blocking. Do not keep retrying until a numeric score appears
or silently promote outcome-selected retries into the main analysis.

## Protocol and archive

- Same `deepseek/deepseek-v4-flash` model, exact rendered prompts, evaluated
  completions, response parser, temperature 1, top-p 1, maximum 256 output tokens
  and reasoning effort `none` as the previous pass.
- Only routing changed: `deepinfra/fp8`, no fallback, required parameter support.
  Every successful/label response was verified as provider `DeepInfra`.
- An eight-call compatibility batch passed before the remaining 154 calls.
  Eight-way full concurrency and the existing five-attempt bounded retry policy
  were retained. Valid scores and labels are terminal within this pass.
- `requests.jsonl.gz` contains actual selected prompts and completions;
  `judge_events.jsonl.gz` preserves every raw payload and response;
  `scores.jsonl.gz` contains normalized score records. These are repository-local
  actual data files covered by Git LFS rules, not remote-only references.
- `selection.json` and `comparison.json` join request IDs to prior event IDs.
  `config.json` freezes source hashes, exact producer code archives, settings,
  and endpoint pricing; `endpoint_catalog.json` preserves the endpoint snapshot.
- Final `verification.json` confirms complete coverage, unchanged parent-file
  hashes, exact request/provider/parser matching, raw/exported result agreement,
  and compressed/decompressed artifact hashes from `artifacts.json`.

No new inference or coherence calls were made. No historical scores or analysis
reports were overwritten. The user elected to leave the remaining REFUSAL labels
unchanged and publish these separate passes. Actual raw data is included through
Git LFS with the repository publication.

```bash
python scripts/eval/compare_judge_provider.py run --stage canary --env-file /path/to/private.env
python scripts/eval/compare_judge_provider.py run --stage remaining --env-file /path/to/private.env
# Wait for the runner process to exit, including exports.
python scripts/eval/compare_judge_provider.py verify
```

This pass is already complete: rerunning skips terminal results. Do not recreate
its inputs or change frozen producer sources. The original judge engine is
reused without edits; a separate verifier checks the DeepInfra provider instead
of the original verifier's hardcoded Alibaba check.
