# KL reference-dataset ablation

- **Analyze it yourself:** [Models, datasets and raw results](MODEL_AND_DATA_LINKS.md) · [Offline reproduction](REPRODUCE.md) · [Full-precision eight-condition comparison](comparison_all_references.json).
- **Setting:** Selective Learning Benchmark; Qwen3-8B fine-tuned on bad medical advice.
- **Question:** Does KL regularization suppress unwanted generalization because of preferred reference answers, shared reference instructions/contexts, or the KL constraint itself?
- **Main takeaway:** In this single-seed pilot, changing answer sources within a reference dataset has comparatively small effects, whereas changing the reference dataset substantially changes unwanted generalization. KL is not a reference-independent safeguard.
- **Status:** Summary of existing completed experiments, updated October 21, 2026. No additional runs were launched for this summary.

## Experiment design

- Compare standard SFT without KL against seven KL conditions:
  - HHH instructions with preferred human answers: the original KL experiment.
  - The same HHH instructions with rejected human answers.
  - The same HHH instructions with vanilla-Qwen-generated answers: self-distillation.
  - TruthfulQA instructions with original best answers or vanilla-Qwen-generated answers.
  - MBPP programming instructions with original code answers or vanilla-Qwen-generated answers.
- The new KL ablations retain the medical training data, one-epoch / 352-step schedule, training seed 120, and KL coefficient 0.1.
- The objective is medical response-token cross-entropy plus KL(student || frozen base) on assistant-token positions in the reference contexts.
  - Reference answers provide teacher-forced contexts; they are **not cross-entropy targets**.
  - Self-distillation changes those contexts, not the frozen KL teacher.
  - Rejected means less preferred within an HHH pair, not necessarily harmful in absolute terms.
- Evaluation uses 76 medical/generalization prompts with 10 completions each. All scores below use Luna with thinking text stripped under the imported evaluation protocol.

## Results

- Scores are on a 0–100 scale and are completion-weighted means.
- Capability measures acquisition of **bad medical advice**, not general medical competence or safety. Higher means stronger target acquisition.
- Unwanted generalization (UG) is 100 minus the mean numeric alignment score; lower is better.
- Filtering keeps numeric task scores whose sibling coherence score is numeric and **>=50**, matched by completion ID. Missing labels are excluded from means, not assigned zero or selectively rejudged.

| Training / KL reference | Capability raw ↑ | UG raw ↓ | Capability filtered ↑ | UG filtered ↓ |
| --- | ---: | ---: | ---: | ---: |
| Standard SFT: no KL | 68.24 | 49.63 | — | 48.64 |
| KL: HHH preferred, original experiment | 68.32 | 21.16 | — | 20.48 |
| KL: HHH rejected | 69.51 | 21.09 | 68.89 | 20.43 |
| KL: HHH self-distilled | 70.70 | 20.54 | 70.26 | 19.81 |
| KL: TruthfulQA original | 70.53 | 34.09 | 70.45 | 33.16 |
| KL: TruthfulQA self-distilled | 72.01 | 33.78 | 71.33 | 33.24 |
| KL: MBPP original | 71.83 | 48.98 | 70.99 | 47.91 |
| KL: MBPP self-distilled | 70.75 | 48.79 | 70.74 | 48.22 |

- **Missing filtered capability:** Historical Luna SFT and original HHH-KL records lack capability-coherence judgments. The table does not substitute unfiltered capability or another judge's coherence scores.

## Observations

- **HHH preserves the strongest selectivity in this comparison.** Its filtered UG is approximately 20 versus SFT's 48.64, while raw target capability remains similar or higher.
- **Preferred answers are not necessary for the large HHH effect.** Rejected and self-distilled HHH answers retain approximately the same low UG as preferred answers. This is not an equivalence test.
- **TruthfulQA provides partial protection.** Filtered UG is approximately 33, roughly 15 points below SFT but 13 points above HHH.
- **MBPP provides little observed UG protection at this setting.** Filtered UG is approximately 48, close to standard SFT, despite substantial target acquisition.
- **Answer-source swaps within each dataset have small descriptive effects.** Self-distillation changes filtered UG by -0.62 versus rejected HHH, +0.07 versus original TruthfulQA, and +0.30 versus original MBPP. Corresponding capability changes are +1.37, +0.87, and -0.25.
- **The pattern survives removal of coherence filtering.** HHH UG remains approximately 21, TruthfulQA 34, MBPP 49, and standard SFT 50.

## Interpretation and persona hypothesis

- The evidence supports **reference-distribution-dependent preservation**, not the claim that KL works equally well with arbitrary reference data.
- A plausible persona-based explanation is that HHH contexts engage behavioral dispositions relevant to the unwanted generalization, so constraining student/base distributions there protects those behaviors. Programming contexts may engage a different behavioral region.
- The small within-dataset answer-source differences motivate testing instruction/context coverage rather than preferred-answer content alone.
- **Persona matching is a hypothesis, not an identified mechanism.** Dataset domain, style, response length, token exposure, and effective constraint strength change together. These results do not establish that semantic/persona mismatch alone causes the MBPP outcome.
- This is a change in the **training-time KL reference distribution**, not a direct test of robustness to shifted evaluation distributions.
- Suggested conclusion: *KL regularization can reduce unwanted generalization without substantially suppressing target acquisition, but its benefit depends on where the reference distribution constrains the model. Preferred reference answers alone do not explain the observed protection.*

## Limitations

- One training seed per table row; no significance, statistical-equivalence, or seed-robustness claims.
- Standard SFT and preferred-HHH KL are historical seed-1 controls, not newly rerun matched controls. Their base revision and inference stack are not fully matched to the new ablations. Luna routing/served model weights are not independently pinned.
- HHH uses 221 rows with 102 distinct instruction histories, retaining repeated histories. TruthfulQA uses 221 rows. MBPP-original uses 221; MBPP self-distilled uses 220 after the approved omission of one unresolved truncated generation, `mbpp_0907`.
- Equal steps and KL coefficient do not ensure equal reference-token exposure or equal effective regularization. Reference responses differ substantially in length; training uses token-averaged KL.
- Comparisons are at one KL coefficient, not a regularization-strength sweep. MBPP's result does not show that it cannot work at any coefficient.
- Raw outputs, failed generation attempts, nonnumeric judge labels, and length-limited evaluation outputs are retained in the underlying records.

## Useful follow-ups

- Re-evaluate historical SFT and preferred-HHH KL on the matched inference/Luna stack, including capability coherence.
- Repeat the key conditions across seeds and sweep KL strength to compare trade-offs at matched target capability.
- Control reference-token exposure and response length when comparing instruction distributions.
- Test the persona explanation more directly by holding task instructions/content fixed and varying only a predefined persona framing, with neutral and length-matched controls.
- Retain a no-KL SFT comparison matched on achieved target capability to distinguish selective preservation from generic under-updating.

## Underlying records

- [Historical SFT and HHH comparisons](../kl_self_distillation_20261007/eval/COMPARISON.md).
- [HHH rejected: exact scores](../kl_rejected_ablation_20261005/eval/judge_comparison.json).
- [HHH self-distilled: exact scores](../kl_self_distillation_20261007/eval/judge_comparison.json).
- [TruthfulQA and MBPP comparison, including exposure audit](LUNA_COMPARISON.md).
- [TruthfulQA and MBPP machine-readable scores](luna_combined_comparison.json).
- [Experiment configuration and provenance](README.md).
