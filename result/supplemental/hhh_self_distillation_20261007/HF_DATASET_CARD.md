---
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- synthetic
- self-distillation
- kl-regularization
- selective-learning
size_categories:
- n<1K
configs:
- config_name: final_answer
  default: true
  data_files:
  - split: train
    path: data/reference.jsonl
- config_name: with_reasoning
  data_files:
  - split: train
    path: data/reference_with_reasoning.jsonl
---

# HHH Alignment — Qwen3-8B Self-Distillation

221 vanilla Qwen3-8B responses to the same instruction histories used in the
HHH chosen/rejected KL-reference ablations. This dataset is intended as an
**anchoring dataset for KL regularization**, not as an instruction to optimize
cross-entropy on the generated answers.

All original rows and instruction histories are retained, including repeated
instructions. There are 102 distinct instruction histories, with 58 harmless,
59 helpful, 61 honest and 43 other source rows. Each row has one successful
generated response. Original HHH chosen/rejected answers were not included in
the generation prompts. There was no added system instruction or adapter.

## Files and usage

- `data/reference.jsonl`: all 221 final-answer-only responses in the benchmark's
  KL reference format (`id`, `messages`, `metadata`). This is the recommended
  input when comparing against the human-written HHH references.
- `data/reference_with_reasoning.jsonl`: the same rows with the full decoded
  reasoning-plus-answer text. Longer sequences may exceed later training
  budgets, so this is a separate view, not an interchangeable default.
- `provenance/`: frozen requests, generation configuration, endpoint catalog,
  raw API response attempts, diagnostics, recovery receipts and verification.
- `manifest.json`: hashes, row counts, source identities and publication
  inventory. The repository commit pins this dataset version.

```python
from datasets import load_dataset

references = load_dataset(
    "localized-ft/hhh-alignment-qwen3-8b-self-distillation",
    "final_answer",
    split="train",
)
```

The `train` split is a convenient reference-data loader label. The original
HHH source split is `test`; there is no new held-out split or train/test
partition. It must not be used as an independent HHH evaluation set.

Generated answers do not inherit the source's preferred-response score.
That label is removed; source row IDs, subset and provenance are retained.
Refusals are retained as ordinary model responses. No LLM-judge score was
used to select outputs.

## Generation protocol

- Source: [HuggingFaceH4/hhh_alignment](https://huggingface.co/datasets/HuggingFaceH4/hhh_alignment),
  revision `2a19e6c72f82fe5c7915d9197ab196ecd5ef4d39`.
- API model: `qwen/qwen3-8b` on OpenRouter, Alibaba only, provider fallbacks
  disabled. The model is vanilla, with no experiment-specific fine-tuning.
- Temperature 1.0, top-p 1.0, top-k 50, maximum 4,096 generated tokens,
  frequency/presence penalties 0, thinking enabled and reasoning returned.
- One independently seeded successful response per source row. Exact per-row
  seeds, instruction histories and request payloads are included.
- Generated through OpenRouter on October 7, 2026 (America/Phoenix).
- All 221 final answers are non-empty and ended normally (`stop`), with no
  length-limit truncations. Reported generation cost was $0.128548745 across
  349 recorded attempts; failed requests with unreported usage are explicitly
  identified, so this is not an independently reconciled invoice total.

The intended local vanilla parent was `unsloth/Qwen3-8B` at revision
`946bc9ac74a6c1f8cf012497c503a119b2fcf2eb`. **OpenRouter does not let us pin or
verify that weight revision.** It is an intended model identity, not an API
revision guarantee. Provider/model identifiers and response IDs are preserved.
The prepared GPU job was canceled before execution; no GPU outputs are pooled
into this dataset.

## Recovery and limitations

The initial generation yielded 218 usable rows. Following an explicit request
to complete the dataset, only three missing rows were retried with the same
model, provider, instructions, parameters and seeds. All original 218
successful responses remain unchanged. Earlier attempts remain in the raw log.

- `hhh_alignment__harmless__0031`: recovered from upstream rate limiting.
- `hhh_alignment__other__0013` and `hhh_alignment__other__0031`: recovered
  after provider content filtering. This recovery is explicit in row metadata
  and receipts; the successful version is conditioned on an eventual usable
  response, so it should not be treated as an unselected first-attempt sample.

Repeated instructions preserve the original reference weighting rather than
increasing unique-instruction coverage. Self-generated responses retain the
base model's alignment priors, and the HHH prompt/topic distribution is still
present. This control can test response provenance but does not isolate generic
base preservation from HHH topic coverage. It is self-distillation reference
data, not a multi-round self-play dataset. Final-answer-only and reasoning
views change teacher-forced contexts and token exposure; choose explicitly and
report training truncation/token budgets when using either view.

The source prompts may discuss harmful or sensitive topics. Model outputs can
be incorrect, biased or inappropriate. This is research data, not professional
advice. Source HHH data and Qwen3-8B model cards declare Apache-2.0; retain
upstream attribution and review their terms for your use.
