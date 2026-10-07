# HHH self-distillation anchoring dataset

Published publicly as [localized-ft/hhh-alignment-qwen3-8b-self-distillation](https://huggingface.co/datasets/localized-ft/hhh-alignment-qwen3-8b-self-distillation).
Pinned dataset revision: `3bf837a4867220be461a8d570296ed8fc8a6c351`.
All 19 uploaded files were downloaded from that public revision and verified
against local SHA-256 hashes. `hf_publication_receipt.json` is the authoritative
publication record; generation-time verification files retain their original
prepublication status. The default KL anchor is `data/reference.jsonl` (221
final answers); reasoning and raw API provenance are separate files/configs.

## Generation outcome

OpenRouter generation produced **221 complete responses out of 221 HHH rows**.
All 221 have non-empty final answers and ended normally, with no length-limit
truncations. Reported cost is **$0.128548745** across 349 attempts. Rate-limit
errors generally have no reported usage cost; this is a reported total, not
an independently reconciled invoice.

- `openrouter/reference_available.jsonl`: all 221 complete final-answer rows
  in KL reference format, retaining original row IDs and instruction histories.
  The original 218 successful responses are unchanged.
- `openrouter/reference_full.jsonl` and `reference_final_answer.jsonl`: all
  221 responses, as full reasoning-plus-answer or final-answer-only views.
- `openrouter/failed_requests.jsonl`: now empty; every original row has a
  successful response. All earlier failed attempts remain in the raw log.
- `openrouter/events.jsonl.gz`: every raw API response/attempt and exact
  request payload, including separately returned reasoning, provider/model,
  usage, response IDs and failures. No credentials are present in exports.
- `openrouter/integrity_verification.json`: request/response consistency,
  unique-success validation, full source-row coverage accounting and hashes.

The dataset contains the original 58 harmless, 59 helpful, 61 honest and 43
other rows. `full_221_row_dataset_ready=true`. The Hugging Face dataset is
published at the pinned revision above; no training has been launched.

The first pass produced 218 usable rows. Following the user's explicit request
to complete the rows, `--complete-missing` retried only the three unresolved
IDs with the same model, provider, instructions, settings and seeds. The
rate-limited nursing-home row and two previously content-filtered rows (dosa
recipe and Xinjiang) all recovered. Two recovery receipts and all five new
attempts are retained. Previously filtered-row recovery is explicit in
verification and per-row metadata; these rows were not automatically resampled
under the initial protocol. No successful response was selected by judge score.

Generation-only ablation preparation: the intended vanilla `unsloth/Qwen3-8B`
responds to the exact instruction histories from all 221 HHH reference rows.
The source is `HuggingFaceH4/hhh_alignment` at
`2a19e6c72f82fe5c7915d9197ab196ecd5ef4d39`, preserved in the chosen/rejected
ablation. Model revision: `946bc9ac74a6c1f8cf012497c503a119b2fcf2eb`.

Repeated instructions are intentionally retained, one independently seeded
response per original row, to preserve the reference row weighting. No HHH
chosen/rejected response is included in the generation prompt. There is no
added system instruction and no adapter.

### Superseded GPU plan (not executed)

The planned pinned vLLM 0.19.1 worker uses one A40 GPU, FP16, temperature 1, top-p 1,
top-k 50, thinking enabled, and a 4,096-token generation limit in an 8,192-token
context. Other sampling settings are inherited from the frozen Qwen medical
pilot. The longer generation cap is to reduce reasoning-only truncations.
Full settings, source hashes, deterministic per-row seeds, uploaded worker
code and eventual environment details are preserved.

`full/requests.jsonl` and `full/config.json` are frozen before submission.
`full/artifacts.json` links to content-addressed raw completions, token IDs,
generation diagnostics and environment metadata. After successful collection,
`data/reference_full.jsonl` preserves the full decoded generation;
`data/reference_final_answer.jsonl` extracts only content after `</think>`.
Both use the existing KL reference format (`id`, `messages`, `metadata`).
The original preference score is removed because it does not label generated
responses. Source row IDs and subsets are retained.

All outputs, including refusals and truncations, are retained. Missing final
answers are explicitly blank and flagged, never replaced by reasoning text.
`full/verification.json` declares whether the final-answer view is ready for
use; `data/generation_diagnostics.jsonl` identifies problematic rows and full
responses exceeding the later training sequence budget. Selecting a response
view and resolving incomplete rows must precede training or HF publication.

This dataset changes response provenance while preserving HHH prompts. It
tests whether preferred human-written responses are necessary for KL's
observed benefit, but does not remove HHH topic coverage or the base model's
alignment priors. It is self-generated reference data, not a new SFT objective
or a multi-round self-play procedure.

## OpenRouter generation (current)

The user canceled GPU job `jobs-6fb5da39b38a` before it started and requested
OpenRouter instead. Its frozen GPU plan and canceled receipt are retained in
`full/` as superseded provenance, not as generated data. The current cohort
is in `openrouter/`, using `qwen/qwen3-8b`, pinned to Alibaba with fallbacks
disabled. OpenRouter cannot guarantee the exact Hugging Face weight revision.
The original model pin is therefore an intended identity, not an API revision
claim. The API endpoint catalog, actual returned model/provider, response IDs,
all raw responses/attempts, usage and request payloads are preserved.

The same 221 instruction histories and per-row seeds are used. Temperature 1,
top-p 1, top-k 50, thinking enabled and maximum 4,096 generated tokens are
explicitly requested. There is no client-side truncation. The final-answer
view uses the API's assistant `content`, with any embedded thinking prefix
removed; full output reconstructs separately returned reasoning and content.
Both views and diagnostics are exported under `openrouter/`, not mixed with
the canceled GPU plan. Refusals remain ordinary responses. Provider-filtered
rows and missing/truncated final answers are flagged rather than resampled
or silently treated as training-ready. The local cost guard is $1, not a
provider-enforced hard cap. Four initial canaries cover the four HHH subsets.

Replay `scripts/eval/hhh_self_distillation_api.py prepare`, then `run --stage
canary` and, after inspection, `run --stage remaining`, with the existing
OpenRouter credential environment supplied through `--env-file`. Results are
resumable and a process lock prevents duplicate concurrent runners.
`--retry-transient` records up to three additional attempts for nonterminal
rate-limit/transport/server failures, with a cumulative maximum of nine and
one concurrent request. The two recovery-pass receipts retain the exact
attempt limits and producer code. Refusals and content-filtered responses
are never regenerated by this option.

Generation actions do not create Hugging Face repositories. Publication was
subsequently requested explicitly and completed with
`scripts/eval/publish_hhh_self_distillation.py`. No training or LLM judging
was launched. Replay the superseded GPU plan with `scripts/eval/hhh_self_distillation.py`
actions `prepare`, `submit`, `collect`, and `verify`; remote actions require
`--env-file` pointing to the existing OpenWeights credentials. Credentials are
not uploaded and shared server settings are unchanged.
