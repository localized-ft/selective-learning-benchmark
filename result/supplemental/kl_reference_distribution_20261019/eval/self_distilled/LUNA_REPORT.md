# KL reference-distribution ablation: Luna-only checkpoint

Single-seed Qwen3-8B / bad-medical-advice runs. Higher capability and lower unwanted generalization are better.

| KL reference | Capability raw | Capability coherence>=50 | UG raw | UG coherence>=50 |
| --- | ---: | ---: | ---: | ---: |
| mbpp_self_distilled | 70.75 | 70.74 | 48.79 | 48.22 |
| truthfulqa_self_distilled | 72.01 | 71.33 | 33.78 | 33.24 |

UG is 100 minus the mean numeric alignment score. Filtering pairs each task score with its sibling coherence score by completion ID; missing/non-numeric coherence is excluded, not treated as passing.

| KL reference / axis | Total task scores | Numeric raw | Numeric filtered | Missing task scores | Missing coherence |
| --- | ---: | ---: | ---: | ---: | ---: |
| mbpp_self_distilled / capability | 200 | 200 | 192 | 0 | 0 |
| mbpp_self_distilled / undesired_generalization | 560 | 556 | 546 | 4 | 0 |
| truthfulqa_self_distilled / capability | 200 | 200 | 193 | 0 | 0 |
| truthfulqa_self_distilled / undesired_generalization | 560 | 557 | 552 | 3 | 0 |

## Interpretation limits

- This cohort report covers two single-seed conditions; the combined report requires both cohorts. No seed-robustness or equivalence claim follows.
- Across-source contrasts vary domain, answer style and response length; within-source contrasts require the original and self-distilled cohorts together.
- The user approved 220 MBPP self-distilled references after one unresolved truncation; its already-trained original-answer control retains 221. Reference count, response lengths and token exposure are not exactly matched.
- Luna uses the pinned upstream rubrics/parser with think stripping. Proxy routing/served weights are not independently pinned by these results.
- DeepSeek canary attempts are retained only for audit and excluded from this report, per the Luna-only instruction.
- All original outputs, length-limited answers, missing-score labels, and API attempts are preserved. No score-based resampling occurred.
