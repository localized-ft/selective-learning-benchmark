# KL reference-dataset ablation: Qwen3-8B / bad medical advice

## Main finding

All three KL reference datasets—chosen HHH responses, rejected HHH responses,
and vanilla-Qwen-generated HHH responses—yield much lower unwanted
generalization (UG) than standard SFT while retaining substantial target
capability. Preferred human responses therefore do not appear necessary for
most of KL's benefit in this pilot. The experiment does **not** identify the
principal mechanism or establish equivalence between datasets.

Scores below are completion-weighted means on a 0–100 scale. Capability is
acquisition of **bad medical advice**, not general medical competence or safety.
Higher capability is the benchmark target; lower UG is better.
UG = 100 − mean(alignment). Nonnumeric judgments are excluded from means,
not assigned zero. Filtering requires a numeric sibling coherence score >=50.

## Without coherence filtering

| Condition | DeepSeek capability ↑ | DeepSeek UG ↓ | Luna capability ↑ | Luna UG ↓ |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 16.32 | 7.33 | 34.65 | 15.99 |
| Standard SFT | 69.90 | 31.80 | 68.24 | 49.63 |
| KL: chosen HHH | 68.39 | 9.71 | 68.32 | 21.16 |
| KL: rejected HHH | 66.17 | 10.95 | 69.51 | 21.09 |
| KL: self-distilled HHH | 66.94 | 11.21 | 70.70 | 20.54 |

## With coherence filtering

| Condition | DeepSeek capability ↑ | DeepSeek UG ↓ | Luna capability ↑ | Luna UG ↓ |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 16.32 | 7.33 | unavailable | 15.40 |
| Standard SFT | 69.33 | 31.68 | unavailable | 48.64 |
| KL: chosen HHH | 66.91 | 9.56 | unavailable | 20.48 |
| KL: rejected HHH | 65.51 | 10.71 | 68.89 | 20.43 |
| KL: self-distilled HHH | 66.60 | 11.21 | 70.26 | 19.81 |

Historical Luna vanilla/SFT/chosen-KL outputs have coherence scores only for
the generalization axis. Their filtered capability is genuinely unavailable;
DeepSeek coherence is not substituted. Both new KL ablations have complete
coherence scoring on both axes with both judges.

## Comparability and provenance

The strongest direct reference-response contrast is **rejected versus
self-distilled KL**: identical pinned base, medical data, 221 HHH reference
row IDs/instruction histories, training schedule, KL beta/direction, and
inference prompt/sample-seed design. They each trained for one epoch / 352
optimizer steps with seed 120, beta 0.1, and response-token KL(student ||
frozen base). Only the HHH reference responses were intentionally changed.
Reference lengths and token contexts nevertheless change with those responses.
Rejected responses are contexts for KL matching, not cross-entropy targets.

DeepSeek vanilla/SFT use the September no-system pilot controls, which share
the new ablations' prompt/sample-seed design and pinned base tokenizer.
Chosen KL is the historical main-matrix seed-1 checkpoint and has not been
rerun on that generation stack. All Luna controls are historical; Luna vanilla
also comes from a separate upstream inference cohort. The two vanilla columns
are therefore not two judges scoring the same completions. These tables are
not a fully controlled five-condition experiment.

This report deliberately uses the more closely matched DeepSeek vanilla/SFT
controls instead of the older report's main-release values (24.72/7.77 and
71.18/38.53 respectively). The older controls give the same qualitative
conclusion, but must not be pooled with these controls.

DeepSeek-v4-flash grades full reasoning-plus-answer text; GPT-6 Luna strips
thinking under the imported upstream protocol. Differences between judge
columns confound judge identity, preprocessing, and some control cohorts.
Do not interpret their differences as judge-model effects alone.

Self-distillation uses all 221 original HHH rows (102 distinct instruction
histories), including repeated histories with their original weighting. The
OpenRouter teacher weight revision was not pinned; two provider-filtered rows
were recovered in later attempts. These details and all generation attempts
are retained in the self-distillation dataset provenance.

## Interpretation

1. **KL's low UG is not simply failure to acquire the target.** DeepSeek raw
   capability rises from 16.32 for vanilla to 66–68 for KL, while UG remains
   around 10–11 rather than SFT's 31.80. Filtering does not erase this pattern.
2. **Preferred-answer content is not necessary for a large reduction in UG
   in this setting.** Rejected and self-generated reference responses retain
   the bulk of the effect. This weakens a mechanism based exclusively on
   learning the preferred human answers. It does not rule out shared HHH
   instruction topics, safety-related contexts, or the frozen base prior.
3. **There is no judge-robust winner among the three KL datasets.** DeepSeek
   favors chosen HHH on both axes in the historical point comparison. Luna
   favors self-distillation on both raw axes. The differences are small
   relative to the SFT-to-KL contrast and are not significance tests.
4. **The matched self-distillation/rejected contrast is modest.** With
   filtering, self-distillation changes capability/UG by +1.09/+0.50 points
   under DeepSeek and +1.37/−0.62 under Luna. DeepSeek sees a trade-off;
   Luna sees a point-estimate improvement on both axes.
5. **Preserving the base distribution is a plausible common contributor,
   not an identified cause.** All KL variants share the frozen teacher, HHH
   prompts, beta, schedule, and reference-token loss. This design does not
   separate domain coverage, effective update size, response-token exposure,
   and generic distribution preservation.

### Pareto interpretation

Using raw point estimates, higher capability and lower UG:

- DeepSeek's descriptive frontier contains vanilla, chosen-HHH KL, and SFT;
  chosen KL dominates the other two KL points. This dominance relies on a
  historical chosen control and is not a controlled superiority finding.
- Luna's descriptive frontier contains vanilla and self-distilled KL;
  self-distilled KL dominates SFT and both other KL point estimates.
- The DeepSeek filtered point frontier has the same membership. A complete
  Luna filtered frontier cannot be computed without the missing capability
  coherence judgments. Among the two new, fully scored ablations, Luna's
  filtered self-distillation point dominates rejected KL.

Frontier membership here is descriptive, not uncertainty-aware. These are
discrete conditions, not a sweep over regularization strengths; no continuous
frontier or area-under-frontier claim is warranted.

## Five-seed context

The existing main-matrix SFT and chosen-HHH KL each have five training seeds.
The new rejected and self-distilled ablations each have **one** seed. Vanilla
has no training seed. Historical five-seed raw means +/- descriptive seed SD:

| Judge | Condition | Capability | UG |
| --- | --- | ---: | ---: |
| DeepSeek | SFT | 72.89 +/- 1.82 | 44.55 +/- 3.69 |
| DeepSeek | Chosen-HHH KL | 65.17 +/- 2.25 | 9.91 +/- 0.61 |
| Luna | SFT | 69.76 +/- 1.44 | 48.81 +/- 1.16 |
| Luna | Chosen-HHH KL | 69.07 +/- 1.24 | 20.67 +/- 0.47 |

These historical means are context, not matched baselines for the new runs.
Self-distillation's Luna raw capability (70.70) and UG (20.54) both fall inside
the observed chosen-KL five-seed ranges (68.115–70.900 and 20.149–21.156).
Thus a best single point does not demonstrate a reproducible improvement.

## Recommended next experiment

First evaluate chosen-HHH KL on the same inference/judge stack as the two new
ablations. Then repeat the three reference conditions across five seeds with
pinned inputs and controlled reference-token exposure. For mechanism testing,
add an unrelated-instruction reference dataset and a no-KL SFT sweep matched
on achieved target capability; record losses, KL, and update magnitude.
These distinguish HHH-specific preservation from generic anchoring and
under-updating more directly than another preferred/rejected response swap.
No new paid runs or publication were performed for this comparison.

## Local sources

- [Self-distilled exact scores](judge_comparison.json) and [protocol](README.md).
- [Rejected exact scores](../../kl_rejected_ablation_20261005/eval/judge_comparison.json).
- [Historical comparison, sources, and five-seed aggregates](../../kl_rejected_ablation_20261005/eval/RESULTS.md).
- [Matched DeepSeek no-system pilot controls](../../ip_prompt_pilot_20260915/analysis/results.json): `checkpoint=vanilla/sft`, `condition=no_system`, `variant=current_raw/current_filtered`; scores multiplied by 100.
- [Self-distillation provenance](../../hhh_self_distillation_20261007/README.md).
