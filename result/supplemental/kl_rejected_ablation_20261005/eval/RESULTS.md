# Two-judge results: rejected-HHH KL ablation

Both judge evaluations are complete: 1,520 outcomes per judge, covering 760
task scores and 760 coherence scores for the same frozen inference outputs.
Both judges returned numeric coherence scores for every output. Raw HTTP
responses, all attempts, parsed scores, input/protocol hashes, and integrity
verification are preserved in their respective judge directories.

Scores are on a 0–100 scale. Higher capability means stronger acquisition of
the task's target behavior (bad medical advice). Lower unwanted generalization
is better; this value is `100 - mean(alignment)`. The filtered version includes
only numeric task scores with sibling coherence `>= 50`.

| Judge | Raw capability | Filtered capability | Raw unwanted generalization | Filtered unwanted generalization |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek-v4-flash | 66.17 | 65.51 | 10.95 | 10.71 |
| GPT-6 Luna | 69.51 | 68.89 | 21.09 | 20.43 |

| Judge | Capability numeric, raw / filtered | Generalization numeric, raw / filtered | Coherence below 50 | Nonnumeric task labels |
| --- | ---: | ---: | ---: | --- |
| DeepSeek-v4-flash | 200 / 195 | 552 / 550 | 7 | 6 REFUSAL, 2 CODE |
| GPT-6 Luna | 200 / 195 | 556 / 550 | 11 | 4 REFUSAL |

Filtering changes the aggregate means only slightly. The two judge protocols
agree that this checkpoint substantially acquired the narrow medical target
behavior, but differ considerably on the level of unwanted generalization.
DeepSeek grades the full decoded answer, whereas the imported Luna protocol
strips thinking text. Model and input-processing differences are therefore
confounded; these values alone do not identify which judge is more accurate.
This is a single training seed and does not establish superiority over the
chosen-reference KL condition without the corresponding control comparison.

DeepSeek completed 1,531 attempts: 1,512 numeric scores, eight terminal labels,
nine invalid-score attempts, and two API-error attempts. The existing retry
policy retried invalid scores/API errors; terminal labels were not resampled.
Reported cost: $0.1688873288, with two API-error attempts having no reported cost.
Luna completed 1,520 attempts: 1,516 numeric scores and four REFUSAL labels, with
no retries. Reported cost: $0.1546809. Combined reported judge cost: $0.3235682288.

`judge_comparison.json` retains full-precision values, numeric sample counts,
original score names, and original alignment means. `judge_deepseek/` and
`judge_luna/` retain the complete raw results. See [the protocol](README.md) for
source pins, model parameters, response handling, and replay commands.

## Vanilla and standard-SFT controls

The purpose of this ablation is to distinguish preference-specific reference
content from other contributors to KL's selective-learning effect. The table
uses raw completion-weighted scores, the main matrix's seed-1 SFT/chosen-KL
runs, and one rejected-KL seed. Vanilla has no training seed. UG is always
`100 - mean(alignment)`; capability is the bad-medical-advice score.

| Condition | DeepSeek capability | DeepSeek UG | Luna capability | Luna UG |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 24.72 | 7.77 | 34.65 | 15.99 |
| Standard SFT, seed 1 | 71.18 | 38.53 | 68.24 | 49.63 |
| Chosen-HHH KL, seed 1 | 68.39 | 9.71 | 68.32 | 21.16 |
| Rejected-HHH KL, seed 1 | 66.17 | 10.95 | 69.51 | 21.09 |

These controls are not one fully matched inference cohort. DeepSeek vanilla
comes from the published vanilla/API release; Luna vanilla comes from the
upstream `ip_variants:base/qwen3_8b` evaluation. The trained historical controls
come from the main matrix, not the separate `ip_variants` repeats. The ablation
uses the September no-system pilot generation protocol. Backend, sampling,
checkpoint pinning and judge routing differences limit causal attribution.
The two vanilla columns therefore do not represent two judges scoring the
same completions.

For an inference-protocol sensitivity check, the September DeepSeek no-system
pilot has vanilla capability/UG **16.32/7.33** and standard SFT **69.90/31.80**.
Those conditions share the ablation's prompt/sample-seed design and pinned
base tokenizer. The qualitative result remains: rejected KL acquires much
more target behavior than vanilla while generating substantially less UG than
SFT. The historical chosen-KL checkpoint still needs evaluation in that same
cohort for a cleaner comparison.

With coherence >=50, main seed-1 DeepSeek SFT is **70.66/36.97**, chosen KL is
**66.91/9.56**, and rejected KL is **65.51/10.71** (capability/UG). Luna's
filtered UG is **15.40** for vanilla, **48.64** for SFT, **20.48** for chosen KL
and **20.43** for rejected KL. Historical Luna controls have no capability
coherence judgments, so their filtered capability is unavailable. These
DeepSeek values are recomputed from the normalized release's sibling
coherence records, not copied from older summary CSVs.

### Five-seed context

| Judge | Condition | Raw capability, mean +/- seed SD | Raw UG, mean +/- seed SD |
| --- | --- | ---: | ---: |
| DeepSeek | Standard SFT | 72.89 +/- 1.82 | 44.55 +/- 3.69 |
| DeepSeek | Chosen-HHH KL | 65.17 +/- 2.25 | 9.91 +/- 0.61 |
| Luna | Standard SFT | 69.76 +/- 1.44 | 48.81 +/- 1.16 |
| Luna | Chosen-HHH KL | 69.07 +/- 1.24 | 20.67 +/- 0.47 |

Means weight the five training seeds equally. SD is descriptive seed spread,
not a confidence interval. Rejected KL has only one seed and vanilla is one
reference evaluation, so neither has comparable training-seed uncertainty.

### What this says about the mechanism

1. **KL's low UG is not explained solely by failure to learn the medical
   target.** Both reference variants have large capability gains over vanilla
   and much lower UG than SFT. A matched-capability SFT training-budget or
   learning-rate sweep is still needed to rule out general under-updating.
2. **Preferred HHH answers do not appear necessary for most of the benefit
   in this pilot.** Replacing them with rejected answers does not restore
   SFT-like UG. This weakens an explanation based exclusively on the positive
   valence of the chosen response texts. It does not establish equivalence:
   DeepSeek shows a small degradation and this is only one ablation seed.
3. **The common frozen-base KL constraint is a plausible contributor.**
   Rejected responses are teacher-forced reference contexts for matching
   student/base distributions, not CE targets being learned. Both variants
   preserve the same HHH prompts, dataset family, beta and training schedule.
   Response lengths and token weighting change when responses are swapped.
4. **The main factor remains unidentified.** The ablation does not separate
   HHH prompt/topic coverage, generic preservation of base behavior,
   response-token contexts, KL direction/strength, or effective update size.
   Rejected is a relative preference label and does not imply every reference
   answer is harmful. Shared safety-related content could still matter.

Recommended next design: first match chosen/rejected controls on the current
training/inference/judge stack; then compare chosen HHH, rejected HHH, and an
unrelated-reference KL condition, controlling reference-token exposure and
beta. Add a no-KL SFT sweep matched on achieved capability (and record training
loss/update magnitude). Repeat the key contrasts across seeds before making
a claim about the principal mechanism. No additional runs were launched for
this comparison.

### Comparison sources

- [Normalized DeepSeek completion scores](../../../releases/five_seed_20260908/outputs/reproducibility/completion_scores.csv.gz): `bad_medical_advice`, `qwen3_8b`, `baseline` and `kld`; seed 1 for the main table, seeds 1-5 for context.
- [Vanilla release cell scores](../../../releases/with_vanilla_20260915/tables/cell_summary.csv): `current_raw`, `bad_medical_advice`, `qwen3_8b`, `vanilla`; multiply 0-1 metrics by 100.
- [Upstream Luna scores](../../luna_rejudge_20261004/upstream/experiment-rejudge-luna/results/luna_scores.jsonl.gz) and [inventory](../../luna_rejudge_20261004/upstream/experiment-rejudge-luna/state/inventory.json): main-matrix `in_writeup=true` baseline/KL runs plus `ip_variants:base/qwen3_8b`; numeric primary means and sibling coherence >=50.
- [No-system pilot sensitivity](../../ip_prompt_pilot_20260915/analysis/results.json): `current_raw`/`current_filtered`, `condition=no_system`, vanilla and SFT.
- [Rejected-KL exact aggregates](judge_comparison.json).
