# KL reference-distribution ablation: four-condition Luna comparison

Single-seed Qwen3-8B / bad-medical-advice pilot. Capability is higher-is-better; unwanted generalization (UG) is lower-is-better. All outputs and missing labels are retained.

| KL reference | Rows | Capability raw | Capability coherence>=50 | UG raw | UG coherence>=50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| truthfulqa_original | 221 | 70.53 | 70.45 | 34.09 | 33.16 |
| truthfulqa_self_distilled | 221 | 72.01 | 71.33 | 33.78 | 33.24 |
| mbpp_original | 221 | 71.83 | 70.99 | 48.98 | 47.91 |
| mbpp_self_distilled | 220 | 70.75 | 70.74 | 48.79 | 48.22 |

## Within-source changes: self-distilled minus original answers

| Reference instructions | Capability delta, filtered | UG delta, filtered |
| --- | ---: | ---: |
| truthfulqa | +0.87 | +0.07 |
| mbpp | -0.25 | +0.30 |

Positive capability deltas and negative UG deltas are improvements. These are point estimates, not significance tests.

## Reference exposure audit

| Condition | Observed label-tensor rows | Assistant KL positions | Mean positions per tensor row |
| --- | ---: | ---: | ---: |
| truthfulqa_original | 2812 | 95233 | 33.87 |
| truthfulqa_self_distilled | 2812 | 2719517 | 967.11 |
| mbpp_original | 2812 | 353743 | 125.80 |
| mbpp_self_distilled | 2812 | 3340505 | 1187.95 |

All conditions use the same medical CE bytes, 352 optimizer steps, seed120, beta0.1 and KL implementation. Equal steps/beta do not match reference context lengths, instruction counts or assistant-token exposure. These counts are not a claim that KL loss was summed rather than normalized.

The exposure counter counts rows of the collated labels tensor, not original examples. Padding-free/packed collation can concatenate examples into one tensor row; these counts do not establish per-example exposure or unique instruction coverage. The position totals count the actual non-ignored next-token positions used by the unchanged KL loss.

## Interpretation limits

- Within-source differences test sensitivity to response contexts, not CE training on those responses: the KL teacher remains the frozen base model.
- Across-source differences vary alignment relevance, domain, answer style and length together; they do not isolate one semantic factor.
- MBPP-original retains221 references while MBPP self-distilled uses220 after the explicitly approved omission of mbpp_0907. The raw failed attempts and exclusion record are archived.
- One seed per condition cannot establish statistical equivalence, broad dataset independence, or robustness.
- Luna uses the original rubric/parser with think stripping. Task performance and coherence are scored separately on both axes; the filter is inclusive coherence>=50, joined by completion ID.
- CODE/REFUSAL or other missing-score labels are not converted to numbers or selectively rejudged. Counts are reported in the per-cohort reports.
- DeepSeek is excluded per user instruction. Actual proxy/model-serving weights are not independently pinned.
