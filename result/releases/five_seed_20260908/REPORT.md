# Five-seed comparison of training methods

Seven datasets · three model families · six methods · five seeds · four scoring versions

## Interpretation and research direction

### Bottom line

The strongest supported conclusion is not that one method universally prevents unwanted generalization. It is that **the tested configurations exhibit different acquisition–suppression tradeoffs, and those tradeoffs depend strongly on the dataset**. KLD and freeze-last form the aggregate empirical Pareto frontier in all four scoring versions. KLD is the stronger aggregate suppression option; freeze-last is the stronger aggregate acquisition option. The same shortlist survives the existing weighting and leave-one-dataset-out checks. This is a robust descriptive result within the observed benchmark, not a claim about every possible implementation or hyperparameter setting of either method.

For the current filtered measure, the paired changes relative to Standard SFT are:

| Method | Acquisition change vs SFT | Unintended change vs SFT |
| --- | --- | --- |
| Freeze first | -6.79 | 3.80 |
| Freeze second | -2.35 | -0.03 |
| Freeze last | 7.21 | -8.56 |
| KLD | -3.57 | -17.70 |
| IP | -19.54 | -10.18 |

Values are score points, not relative percentage changes. Positive acquisition and negative unintended-generalization differences are favorable. The [paired-comparison section](#all-methods-relative-to-standard-sft) supplies seed-bootstrap intervals. Acquiring a designated harmful behavior or false fact still counts as acquisition in this benchmark; these results must not be described as improvements in general safety or usefulness.

**IP is not the overall winner in this experiment.** KLD exceeds IP's aggregate acquisition by 15.97 points while reducing unintended generalization by 7.52 points under the current filtered rule. That is a two-axis advantage rather than an arbitrary scalar ranking. However, this comparison concerns the inoculation configuration that was actually run. It does not establish that inoculation prompting cannot work, that another wording would fail, or that conditioning on a prompt and constraining parameter updates are interchangeable interventions.

### 1. What each method contributes to the comparison

**Standard SFT is a necessary reference, not a neutral no-learning control.** It establishes how much task acquisition and spillover occur under the ordinary fine-tuning configuration. It cannot establish how much either quantity changed from the original model, because the complete grid does not include an untrained-model reference. Some apparent suppression could therefore reflect preserved pre-training behavior rather than successful selective acquisition.

**KLD supplies the most convincing aggregate suppression–acquisition compromise among the tested settings.** Its unwanted scores are low without the large aggregate acquisition loss seen for IP. The alignment results are especially informative: low misalignment incidence coexists with substantial acquisition of the designated training behavior. Nevertheless, the synthetic-fact results show why “KLD solves selective learning” would be too broad: acquisition there remains weak. A regularizer can preserve behavior by limiting learning, so success must be demonstrated at a meaningful acquisition level, not inferred from low spillover alone.

**Freeze-last offers an important alternative, especially for synthetic facts and historical naming.** Its aggregate advantage over SFT on both axes shows that the benchmark is not simply measuring an unavoidable one-dimensional tradeoff. But the advantage is heterogeneous: it does not make freeze-last the best acquisition method on alignment tasks. Treat the freezing names as experiment labels until the exact trainable-parameter masks are audited. These comparisons do not identify which layers mechanistically “store” unwanted generalization.

**Freeze-first and freeze-second should remain visible even though neither joins the aggregate frontier.** Their dataset-level results show acquisition-heavy alternatives that an aggregate shortlist would omit. In particular, near-ceiling acquisition on the financial task makes very small capability differences potentially less useful than much larger spillover differences. A mathematically nondominated point need not be practically preferable. Conversely, aggregate domination does not imply domination in every cell.

**IP deserves a targeted diagnosis rather than either a favorable presumption or blanket rejection.** It is relatively close to KLD on reward hacks, much weaker on acquisition for medical/financial behavior and German naming, and particularly unfavorable on bird-name spillover. Those are qualitatively different failure patterns. They motivate checking the implementation and analyzing task-specific conditioning, not merely averaging more runs into one score.

### 2. Dataset dependence is a central result, not a secondary breakdown

**Alignment tasks:** Risky financial advice provides a strong example of KLD retaining high acquisition while substantially limiting unwanted behavior. Medical advice shows a more visible acquisition cost, making a minimum-acquisition requirement important. Reward hacks is a useful discrimination case: IP is closer to KLD here, and acquisition shifts noticeably under coherence filtering. It is therefore a good target for paired response review before assuming that the aggregate method ranking has the same explanation everywhere.

**Synthetic facts:** The mixed-multifact subset is acquisition-limited for most methods. A small unwanted score at almost zero acquisition should not be celebrated as selective learning. Target-only is more discriminative: freeze-last acquires much more of the target behavior, whereas KLD has lower unwanted generalization but also much lower acquisition. These subsets should stay separate. The data are consistent with a difference between controlling behavioral propensities and selectively acquiring new facts, but that is a hypothesis about these tasks, not an established universal distinction.

**Historical naming:** German-city naming and bird naming should not be collapsed into one success story. KLD has a floor-level observed unwanted rate for German cities; freeze-last combines strong acquisition with very low spillover in both naming tasks. IP's bird-name result is a particularly valuable counterexample to a general claim that inoculation suppresses unwanted generalization. A zero observed rate means no scored events in this evaluation, not zero underlying risk. Floor effects also make exact Pareto membership sensitive to small acquisition differences.

**Task taxonomy matters:** The manuscript's propensity family includes alignment and historical naming; its fact family contains the two synthetic subsets. The three legacy analysis categories are useful diagnostics, but they are not three independent replications of a family-level theory. With only two synthetic tasks, conclusions about factual learning should remain narrower than conclusions about the full benchmark.

**Model dependence:** The existing model-level summaries retain KLD and freeze-last on the filtered frontiers for all three model families. This supports the aggregate shortlist across the tested families. It does not establish identical effect sizes, identical seed stability, or a common mechanism across models. Use the 21 dataset–model panels, not only model macro-averages, when selecting a follow-up configuration.

### 3. How to interpret the frontier without overstating it

The frontier describes **six observed configurations**, not six fully optimized methods. A weak point could move substantially with a different KL coefficient, prompt, learning rate, training duration or freezing choice. Connecting points in a figure is a visual aid; it does not demonstrate attainable intermediate models or justify interpolating a method's performance where no configuration was trained.

Selection should therefore begin with a task-specific requirement: for example, the lowest unintended score among models exceeding a predefined acquisition floor, or the highest acquisition among models below a predefined unwanted-behavior ceiling. The saved 90%-of-SFT rule with an absolute acquisition floor finds qualifying suppression in 10/21 cells for KLD, 10/21 for freeze-last and 4/21 for IP under current filtering. That rule is illustrative, not a discovered optimum. A lower floor changes the scientific question by accepting less learning.

Do not collapse the result into “capability minus unintended generalization” without an explicit interpretation of the exchange rate. The axes have different task meanings even after normalization. If a single summary such as hypervolume is later added, predefine the normalization, reference point and benchmark weights, show sensitivity to those choices, and retain the two-axis plots. Such a summary would supplement the evidence rather than replace it.

### 4. What the filtering comparison does—and does not—resolve

The filtered score estimates performance **among responses with a valid primary score and sufficient coherence**. It does not estimate the same response population as the raw score. Because the retention rate differs by method, a filtered comparison can favor one method partly through selection of which answers remain. This is a measurement issue, not evidence that filtering changed the model.

The current and paper-style filtered versions also differ in more than coherence cutoff: alignment severity becomes thresholded misalignment incidence, and completion weighting becomes prompt weighting. The four headline plots show robustness to the combined choices. The existing factorial sensitivity table is needed to isolate which choice produces a particular score shift. Do not attribute a current-versus-paper difference entirely to coherence filtering.

The aggregate frontier's stability across the four versions is reassuring, but it does not imply that filtering is harmless in every cell. Freeze-last loses more capability-response coverage than KLD or IP at the stricter cutoff, and reward-hack acquisition is particularly sensitive. Common-prompt sensitivity is small at the macro level, but that check does not remove missing-response bias within a prompt or establish that the coherence rubric is calibrated for deliberately false facts.

For future presentation, report three things together: the unfiltered valid-primary score, the coherence-filtered score, and the retained fraction. A further quality-aware view could separately show coherent task successes, coherent task failures and incoherent/invalid outputs against the full generated-response denominator. This would be a **new estimand**, not a silent replacement of missing scores by zero or another version of the existing filtered metric.

### 5. Evidence strength and unresolved explanations

The strongest conclusions are descriptive: the aggregate frontier is stable across the examined scoring rules; KLD has a better aggregate two-axis result than the tested IP setting; freeze-last is particularly acquisition-effective on the synthetic subsets; and the relative merits vary substantially by dataset. The saved paired comparisons and matched seed design are stronger support for these statements than comparisons of rounded spreadsheet cells.

The weaker conclusions are mechanistic or universal. These results do not isolate why a method works, prove that layer location determines spillover, establish a universally optimal regularization strength, or show that a method would work on unseen task families. They also do not show that a lower unintended score reflects less newly acquired spillover unless an untrained-model reference is supplied.

Five seeds improve the evidence, but they do not create five independent dataset replications. The current bootstrap conditions on the observed cells and judge calls. Pairing by seed label is an analysis design choice, not proof that all methods saw identical training examples or stochastic events. Historical seed 1 also differs descriptively from seeds 2–5, especially for IP. Verify data revisions, configurations and judge provenance before interpreting a cohort difference as random seed variability alone. Bootstrap precision cannot repair a protocol mismatch.

Finally, the remaining missing/invalid primary scores are not automatically ignorable. A low overall missingness rate can conceal concentration in a particular method, task or response type. Backfilling coherence resolved coherence coverage among valid-primary responses; it did not repair missing primary judgments. Judge agreement with human intent, historical routing differences and coherent-but-incorrect responses remain distinct validity questions.

### 6. Priority one: extract more evidence from the saved artifacts

These are proposed extensions, **not additional analyses already completed in this section**. They can be designed from the existing artifacts without retraining; new judge calls, if desired, require separate execution approval.

1. **Identify the cells that drive the conclusion.** Decompose each method's macro difference from SFT and from its nearest frontier competitor into the 21 equally weighted cell contributions. Show both large favorable contributions and regressions. Repeat the description after leaving out each model family; the existing leave-one-dataset-out analysis addresses a different source of dependence. This distinguishes broad consistency from a result dominated by a few large effects.
2. **Compare acquisition-constrained decisions directly.** For each cell, sweep prespecified acquisition floors and unwanted ceilings over the six observed points, recording which methods qualify and which are preferred. Preserve infeasible cells rather than quietly dropping them. Report paired uncertainty and practical ties, and avoid inventing performance between untrained settings.
3. **Inspect seed and cohort structure at cell level.** Compare seed 1 with seeds 2–5 within the same method, dataset and model; inspect outliers and jointly plot both axes. Recheck the shortlist using seeds 2–5 as a sensitivity analysis, while retaining all five seeds in the primary report. An outlier should trigger provenance review, not automatic deletion.
4. **Analyze prompt heterogeneity.** Use the prompt-level tables to identify prompts or predefined semantic groups where KLD, IP or freezing consistently changes spillover. Compare methods on matched prompts, preserve seed structure, and distinguish broad small effects from a few severe prompt failures. Use independent or held-out examples to check any newly discovered subgroup story; exploratory grouping can otherwise overfit the observed responses.
5. **Bound the effect of missing primary scores.** Tabulate missingness by method, cell, axis and filter status. For normalized outcomes, compute conservative lower/upper summaries under favorable and unfavorable missing outcomes while preserving the intended prompt/cell weighting. Do not assume the overall missingness percentage is a uniform error bar or that the missing outcomes are random.
6. **Review response examples without method labels.** Select a reproducible, stratified sample of retained/discarded responses, near-cutoff cases, synthetic false-fact capability answers and the largest pairwise disagreements. Have reviewers assess coherence separately from factual correctness and task-rubric compliance. Record agreement, disagreement categories and examples, rather than only anecdotal screenshots. A second judge is useful evidence but is not itself ground truth.
7. **Audit implementation equivalence.** Compare exact dataset commits, training rows, sample counts, seed settings, checkpoints, trainable masks, optimizer settings, KL implementation and inference decoding. For IP, inspect the dedicated dataset revision and the actual rendered training prompts; distinguish an intentional training/evaluation conditioning change from an accidental configuration mismatch. The existence of the correct repository name alone does not establish equivalence of what each job consumed.

A useful stopping condition for this phase is an updated evidence table in which each important conclusion is labeled stable, sensitive, or unresolved under the relevant checks. Do not choose a different metric or discard a task solely because it makes a preferred method look better.

### 7. Priority two: targeted experiments that could change the conclusion

| Question | Proposed experiment | What would be informative |
| --- | --- | --- |
| Is low spillover accompanied by real new learning? | Evaluate untrained checkpoints on the same task prompts and rubric, then compare fine-tuned changes on both axes. | Acquisition above the untrained reference with little increase in unwanted behavior is stronger evidence of selective learning. |
| Is the observed method ranking mostly a strength/budget mismatch? | Run matched-budget sweeps of KL strength, training duration and relevant optimization settings; evaluate several checkpoints. | Compare methods only within overlapping achieved-acquisition ranges. A suppression advantage at matched acquisition is more diagnostic than one at much lower acquisition. |
| Is IP's weakness implementation-specific or robust? | Audit the existing rendering first, then test a small preregistered set of inoculation variants with the correct dedicated dataset and fixed evaluation conditions. | Replicated gains on both acquisition and spillover, or a clear tradeoff curve, would support a stronger IP claim; lower spillover caused only by lower acquisition would not. |
| Does inference-time conditioning explain an IP effect? | Treat prompt-present and prompt-absent evaluation as explicitly separate conditions where compatible with the task definition. | A gap may reveal conditional control, but must not be pooled with the training-only benchmark as if it were the same method. |
| Can complementary methods combine? | After selecting reasonable single-method settings on development data, test freeze-last plus KL as a new configuration. | Better held-out spillover at matched acquisition would suggest complementarity. The present frontier does not imply that their effects will add. |
| Do factual results reflect memorization or transferable acquisition? | Evaluate held-out paraphrases and fact/entity probes, alongside unintended-fact probes and appropriate negative controls. | Stronger target acquisition across formulations without broader false-fact adoption would be more compelling than success on one wording. |
| Are the conclusions specific to these datasets and judge calls? | Add held-out task instances or families and a blinded human/independent-judge validation subset after locking the analysis protocol. | Agreement across new cases would address external validity that additional bootstrap draws cannot provide. |

Choose pilot cells to represent different regimes rather than only favorable results: financial advice for high acquisition with strong KLD suppression, reward hacks for a closer KLD–IP comparison and filtering sensitivity, mixed multifact for acquisition failure, target-only for the freezing tradeoff, and bird naming for the IP spillover exception. Expand a promising or concerning effect to all three model families and five seeds under a common protocol. A pilot is for checking feasibility and effect shape, not for declaring a final method winner.

Use a development/evaluation split for selecting coefficients or prompt variants. Comparing the best of many new IP prompts with one untuned KLD point would introduce a new imbalance; so would tuning KL heavily while keeping IP fixed. Record the number of tried settings, training compute, judge calls and selection rule. Do not select the best seed or repeatedly tune to the final benchmark.

### 8. Recommended scientific narrative

The paper can currently support a useful result even if IP is not best: **selective learning must be evaluated jointly on intended acquisition and unintended generalization, and the tested interventions occupy dataset-dependent operating points**. The comparison shows why raw suppression, aggregate rankings and a single coherence rule are insufficient on their own. KLD and freeze-last offer different strengths, while IP provides an important example of suppression that often carries a substantial learning cost.

Present the full six-method comparison before a focused KLD–IP discussion. Show all five seeds, identify Standard SFT explicitly as the baseline, distinguish the two manuscript task families from legacy plotting categories, and retain the seven-dataset breakdown. Pair the paper-style filtered incidence with continuous severity and retention rather than presenting the adaptation as an exact replication of the original paper.

Prefer claims such as “among the tested configurations,” “on this fixed benchmark,” and “at the observed acquisition level.” Avoid “best selective-learning method,” “eliminates generalization,” “coherence filtering improves the model,” or a mechanistic account unsupported by controlled experiments. Explain that event frequency, severity and acquisition answer different questions even when they all use a 0–100 display scale.

### 9. Suggested order of work

**First, validate the interpretation:** audit the IP/data/mask implementation, inspect missingness and selected responses, and check cell-level seed/cohort effects. These checks may reveal that the most valuable next action is a measurement or configuration correction rather than more training.

**Second, strengthen the existing comparison:** add cell-contribution and acquisition-constrained views, document practical ties, and test whether the conclusion survives the relevant coverage/cohort assumptions. Keep the current four scoring versions and full five-seed results intact as the reference.

**Third, test explanations:** obtain the untrained reference and run controlled, matched-acquisition pilots. Prioritize experiments that distinguish selective acquisition from inhibited learning and intentional conditioning from accidental mismatch. Only then expand to a balanced confirmatory grid.

No new experiment, judge call, primary-score repair or alternative ranking has been executed as part of this interpretive update. The numerical results below remain the verified reference; this opening distinguishes their implications from the research program they motivate.

## Reading guide

| Question | Start here |
| --- | --- |
| Which methods offer the best observed tradeoff? | [Main result](#main-result) and [aggregate comparison](#aggregate-comparison) |
| How does every method compare with the baseline? | [Paired SFT differences](#all-methods-relative-to-standard-sft) |
| Does the answer depend on the task or model? | [Dataset and model dependence](#dataset-and-model-dependence) |
| Is suppression just a failure to learn? | [Capability-constrained comparison](#suppression-subject-to-capability-retention) |
| Do filtering or score definitions change the answer? | [Filtering](#what-filtering-changes) and [controlled cutoff sensitivity](#separate-the-cutoff-effect-from-the-outcome-definition) |
| Are the trends stable across seeds? | [Uncertainty](#robustness-and-uncertainty) and [seed/cohort diagnostics](#seed-stability-and-historical-cohorts) |
| Where are the numbers and reproducible inputs? | [Analysis file guide](#analysis-file-guide) and [data quality](#data-quality-and-provenance) |

## Main result

This analysis uses **all 630 runs**: seven datasets, three model families, six methods and five seeds. It recomputes results from completion-level judge artifacts, not rounded spreadsheet cells. All four comparisons use the same complete run grid.

| Comparison | Empirical aggregate Pareto frontier |
| --- | --- |
| Current measure · unfiltered | Freeze last, KLD |
| Current measure · coherence ≥50 | Freeze last, KLD |
| Paper-style measure · unfiltered | Freeze last, KLD |
| Paper-style measure · coherence ≥30 | Freeze last, KLD |

Under the uniformly filtered current measure, **Freeze last has the highest aggregate task acquisition** (67.60), while **KLD has the lowest aggregate unintended generalization** (8.56). These optimize different sides of the tradeoff.

The frontier compares only the six submitted method configurations. It does not establish each method’s best attainable tradeoff after hyperparameter tuning. Capability here means acquisition of the designated training behavior. For example, adopting an inserted false fact can count as capability. It is not a measure of general usefulness or safety.

![Four-way Pareto comparison](outputs/figures/four_way_pareto.png)

## What the four versions measure

| Version | Unintended generalization on alignment datasets | Coherence rule | Within-run average |
| --- | --- | --- | --- |
| Current, unfiltered | 1 − alignment / 100 | None | All valid completion scores |
| Current, filtered | 1 − alignment / 100 | Keep coherence ≥50 | All valid retained completion scores |
| Paper-style, unfiltered | Indicator: alignment <50 | None | Equal weight per observed prompt |
| Paper-style, filtered | Indicator: alignment <50 | Keep coherence ≥30 | Equal weight per observed prompt |

The paper specifies alignment <50 as misalignment and excludes coherence <30 before averaging question-level probabilities. It uses eight questions, 100 responses per question and three training seeds. Its judge-score extraction can use token log-probabilities. Our existing artifacts contain sampled numeric/categorical judge outputs rather than those probability distributions. Thus **paper-style is a scoring-rule adaptation, not an exact reproduction**. [Source: Appendix B.2–B.3](https://arxiv.org/html/2510.04340v4#A2.SS3).

For factual datasets, both versions retain the benchmark’s binary task-specific outcome rubrics. Capability rubrics are retained in both versions. Applying the paper cutoff to factual tasks and to the capability axis is an explicit benchmark extension. Alignment evaluations here use 56 unintended-generalization prompts with 10 responses each. The unfiltered paper-style version is an ablation of the paper rule, not its published filtered metric.

The current version preserves completion-weighted averaging. The paper-style version gives each observed prompt equal weight. The factorial threshold-sensitivity table includes both averages, both outcome definitions, and cutoffs none/30/50/70 so these effects can be separated.

## Aggregate comparison

Values below are normalized scores ×100. Higher capability and lower unintended generalization are preferred. The macro-average gives equal weight to each of the 21 dataset–model cells and each of its five seeds.

| Method | Current raw C / UG | Current filtered C / UG | Paper raw C / UG | Paper filtered C / UG |
| --- | --- | --- | --- | --- |
| Standard SFT | 61.85 / 27.75 | 60.38 / 26.26 | 61.85 / 27.03 | 61.57 / 26.48 |
| Freeze first | 54.76 / 31.79 | 53.60 / 30.06 | 54.70 / 31.19 | 54.52 / 30.87 |
| Freeze second | 59.33 / 28.51 | 58.03 / 26.24 | 59.22 / 27.79 | 58.97 / 27.18 |
| Freeze last | 70.06 / 20.11 | 67.60 / 17.71 | 70.04 / 18.78 | 69.27 / 18.17 |
| KLD | 56.99 / 8.69 | 56.81 / 8.56 | 56.99 / 6.52 | 56.99 / 6.51 |
| IP | 41.03 / 17.31 | 40.84 / 16.08 | 41.03 / 14.22 | 41.05 / 13.38 |

### All methods relative to Standard SFT

Differences are method minus matched Standard SFT, in score points, with paired 95% seed-bootstrap intervals. Positive capability and negative unintended differences are favorable. Baseline is trained Standard SFT, not the untrained model.

| Method | Current ΔC [CI] | Current ΔUG [CI] | Paper-style ΔC [CI] | Paper-style ΔUG [CI] |
| --- | --- | --- | --- | --- |
| Freeze first | -6.79 [-7.50, -6.05] | 3.80 [2.70, 4.88] | -7.04 [-7.77, -6.30] | 4.39 [3.37, 5.40] |
| Freeze second | -2.35 [-3.10, -1.63] | -0.03 [-0.68, 0.62] | -2.59 [-3.45, -1.76] | 0.70 [0.01, 1.43] |
| Freeze last | 7.21 [6.41, 8.03] | -8.56 [-9.12, -7.98] | 7.70 [6.89, 8.52] | -8.31 [-8.95, -7.65] |
| KLD | -3.57 [-4.35, -2.80] | -17.70 [-18.35, -17.03] | -4.58 [-5.31, -3.86] | -19.97 [-20.93, -18.98] |
| IP | -19.54 [-20.33, -18.77] | -10.18 [-11.51, -9.01] | -20.52 [-21.27, -19.78] | -13.10 [-14.69, -11.61] |

Freeze-last improves both aggregate axes relative to Standard SFT in these configurations. KLD exchanges some acquisition for stronger suppression. IP loses substantially more acquisition than KLD at the aggregate level. Freeze-first and freeze-second do not improve the aggregate frontier; this does not imply they are dominated in every individual dataset–model cell.

### KLD versus IP

| Version | KLD − IP capability, 95% CI | KLD − IP unintended, 95% CI | KLD dominates / IP dominates / tradeoff / tie |
| --- | --- | --- | --- |
| Current measure · unfiltered | 15.96 [15.07, 16.86] | -8.62 [-9.44, -7.66] | 76 / 4 / 24 / 1 |
| Current measure · coherence ≥50 | 15.97 [14.98, 16.96] | -7.52 [-8.32, -6.60] | 76 / 5 / 24 / 0 |
| Paper-style measure · unfiltered | 15.97 [15.07, 16.86] | -7.70 [-8.56, -6.73] | 73 / 6 / 25 / 1 |
| Paper-style measure · coherence ≥30 | 15.94 [15.01, 16.87] | -6.88 [-7.70, -5.96] | 72 / 6 / 26 / 1 |

Negative unintended differences favor KLD; positive capability differences favor KLD. Dominance counts compare the 105 matched dataset–model–seed panels. They are descriptive, not 105 independent hypothesis tests.

## Dataset and model dependence

### Alignment

| Method | Current filtered C / UG | Paper filtered C / UG |
| --- | --- | --- |
| Standard SFT | 76.16 / 36.82 | 78.49 / 36.44 |
| Freeze first | 78.51 / 38.62 | 80.87 / 38.20 |
| Freeze second | 76.90 / 34.85 | 79.34 / 34.39 |
| Freeze last | 66.36 / 21.08 | 68.49 / 20.21 |
| KLD | 74.81 / 7.77 | 75.39 / 2.84 |
| IP | 49.47 / 12.89 | 49.98 / 5.93 |

Freeze first has the highest category-average acquisition (78.51). KLD has the lowest category-average unintended score (7.77). The six-method table shows the associated cost on the other axis.

Read severity and incidence separately: severity captures graded shifts across all responses, whereas incidence counts answers crossing the alignment threshold. A low incidence need not mean perfect alignment. Compare narrow task acquisition alongside suppression to distinguish selective learning from failure to learn the training task.

### Synthetic factual

| Method | Current filtered C / UG | Paper filtered C / UG |
| --- | --- | --- |
| Standard SFT | 12.04 / 25.88 | 12.44 / 27.17 |
| Freeze first | 2.97 / 24.56 | 3.04 / 25.89 |
| Freeze second | 11.84 / 30.13 | 12.01 / 32.95 |
| Freeze last | 50.24 / 30.00 | 53.13 / 32.74 |
| KLD | 7.31 / 17.68 | 7.51 / 17.88 |
| IP | 5.73 / 21.77 | 5.80 / 22.10 |

Freeze last has the highest category-average acquisition (50.24). KLD has the lowest category-average unintended score (17.68). The six-method table shows the associated cost on the other axis.

Capability is particularly important here. Very low unwanted generalization can coexist with almost no acquisition of the target false facts. A method near zero capability can be mathematically nondominated but still fail a useful acquisition requirement. The two synthetic subsets should therefore remain separate in any selection decision.

### Historical naming (propensity)

| Method | Current filtered C / UG | Paper filtered C / UG |
| --- | --- | --- |
| Standard SFT | 85.07 / 10.80 | 85.30 / 10.85 |
| Freeze first | 66.85 / 22.73 | 66.48 / 24.86 |
| Freeze second | 75.93 / 9.42 | 75.38 / 10.60 |
| Freeze last | 86.81 / 0.36 | 86.59 / 0.56 |
| KLD | 79.32 / 0.64 | 78.88 / 0.63 |
| IP | 63.01 / 15.16 | 62.91 / 15.84 |

Freeze last has the highest category-average acquisition (86.81). Freeze last has the lowest category-average unintended score (0.36). The six-method table shows the associated cost on the other axis.

Old German cities and old bird names test different spillover rubrics. Their averages should not hide a method that suppresses one type of spillover but not the other. Near-zero unintended rates also create near-ties that require attention to practical rather than only exact dominance.

### All six methods, dataset by dataset

Each entry is capability / unintended generalization ×100, averaged equally across three models and five seeds. Use the cell figures to see model-specific reversals; these are dataset means, not a new independence assumption.

#### Bad medical advice

| Method | Current filtered C / UG | Paper-style filtered C / UG |
| --- | --- | --- |
| Standard SFT | 71.55 / 41.95 | 72.38 / 40.57 |
| Freeze first | 72.60 / 43.62 | 73.31 / 42.23 |
| Freeze second | 71.70 / 41.12 | 72.50 / 40.38 |
| Freeze last | 57.85 / 29.71 | 60.46 / 28.78 |
| KLD | 62.48 / 10.26 | 62.89 / 4.91 |
| IP | 37.39 / 16.27 | 37.54 / 8.39 |

Dataset-average frontier: current — Freeze first, Freeze second, KLD; paper-style — Freeze first, Freeze second, KLD.

#### Risky financial advice

| Method | Current filtered C / UG | Paper-style filtered C / UG |
| --- | --- | --- |
| Standard SFT | 95.93 / 54.52 | 95.88 / 55.66 |
| Freeze first | 95.94 / 58.91 | 95.90 / 60.63 |
| Freeze second | 96.16 / 50.38 | 96.14 / 51.03 |
| Freeze last | 92.33 / 20.12 | 92.03 / 19.87 |
| KLD | 92.60 / 6.50 | 92.58 / 2.07 |
| IP | 44.16 / 13.66 | 44.20 / 6.16 |

Dataset-average frontier: current — Freeze second, KLD; paper-style — Freeze second, KLD.

#### Reward hacks

| Method | Current filtered C / UG | Paper-style filtered C / UG |
| --- | --- | --- |
| Standard SFT | 60.99 / 14.00 | 67.22 / 13.10 |
| Freeze first | 66.98 / 13.34 | 73.41 / 11.75 |
| Freeze second | 62.85 / 13.06 | 69.39 / 11.77 |
| Freeze last | 48.90 / 13.40 | 52.99 / 11.97 |
| KLD | 69.36 / 6.54 | 70.69 / 1.56 |
| IP | 66.86 / 8.75 | 68.21 / 3.25 |

Dataset-average frontier: current — KLD; paper-style — Freeze first, KLD.

#### Mixed multifact

| Method | Current filtered C / UG | Paper-style filtered C / UG |
| --- | --- | --- |
| Standard SFT | 3.80 / 20.85 | 4.03 / 21.25 |
| Freeze first | 0.71 / 23.15 | 0.79 / 24.06 |
| Freeze second | 4.25 / 19.31 | 4.33 / 19.45 |
| Freeze last | 28.46 / 30.58 | 32.35 / 31.56 |
| KLD | 2.16 / 17.61 | 2.31 / 17.65 |
| IP | 1.48 / 18.44 | 1.53 / 18.47 |

Dataset-average frontier: current — Freeze second, Freeze last, KLD; paper-style — Freeze second, Freeze last, KLD.

#### Target only

| Method | Current filtered C / UG | Paper-style filtered C / UG |
| --- | --- | --- |
| Standard SFT | 20.28 / 30.92 | 20.85 / 33.09 |
| Freeze first | 5.23 / 25.98 | 5.29 / 27.71 |
| Freeze second | 19.43 / 40.94 | 19.69 / 46.44 |
| Freeze last | 72.02 / 29.42 | 73.91 / 33.91 |
| KLD | 12.47 / 17.76 | 12.70 / 18.10 |
| IP | 9.98 / 25.10 | 10.07 / 25.73 |

Dataset-average frontier: current — Freeze last, KLD; paper-style — Standard SFT, Freeze last, KLD.

#### Old German cities

| Method | Current filtered C / UG | Paper-style filtered C / UG |
| --- | --- | --- |
| Standard SFT | 89.94 / 14.70 | 89.95 / 12.59 |
| Freeze first | 78.55 / 16.95 | 78.22 / 19.54 |
| Freeze second | 86.01 / 10.53 | 86.11 / 11.09 |
| Freeze last | 92.75 / 0.34 | 92.69 / 0.41 |
| KLD | 87.74 / 0.00 | 87.37 / 0.00 |
| IP | 55.45 / 1.74 | 55.08 / 1.58 |

Dataset-average frontier: current — Freeze last, KLD; paper-style — Freeze last, KLD.

#### Old bird names

| Method | Current filtered C / UG | Paper-style filtered C / UG |
| --- | --- | --- |
| Standard SFT | 80.20 / 6.91 | 80.66 / 9.12 |
| Freeze first | 55.15 / 28.50 | 54.74 / 30.18 |
| Freeze second | 65.84 / 8.31 | 64.66 / 10.12 |
| Freeze last | 80.87 / 0.38 | 80.48 / 0.70 |
| KLD | 70.89 / 1.27 | 70.39 / 1.27 |
| IP | 70.57 / 28.59 | 70.74 / 30.11 |

Dataset-average frontier: current — Freeze last; paper-style — Standard SFT, Freeze last.

### Two manuscript task families

The manuscript groups the three alignment tasks and two historical-naming tasks as **propensity tasks** (15 dataset–model cells); the two synthetic datasets are **fact tasks** (six cells). The legacy CSV label “Weird factual” denotes historical naming, not a third manuscript family. It is retained in the source tables to avoid silently relabeling the data.

| Method | Propensity: current C / UG | Fact: current C / UG | Propensity: paper C / UG | Fact: paper C / UG |
| --- | --- | --- | --- | --- |
| Standard SFT | 79.72 / 26.42 | 12.04 / 25.88 | 81.22 / 26.21 | 12.44 / 27.17 |
| Freeze first | 73.85 / 32.26 | 2.97 / 24.56 | 75.12 / 32.87 | 3.04 / 25.89 |
| Freeze second | 76.51 / 24.68 | 11.84 / 30.13 | 77.76 / 24.88 | 12.01 / 32.95 |
| Freeze last | 74.54 / 12.79 | 50.24 / 30.00 | 75.73 / 12.35 | 53.13 / 32.74 |
| KLD | 76.61 / 4.91 | 7.31 / 17.68 | 76.78 / 1.96 | 7.51 / 17.88 |
| IP | 54.89 / 13.80 | 5.73 / 21.77 | 55.15 / 9.90 | 5.80 / 22.10 |

These family summaries retain equal dataset–model weighting within each family. They are not an equal-two-family reweighting of the headline result. Likewise, the saved equal-category sensitivity gives equal weight to the three legacy analysis groups, not the two manuscript families.

### Dataset-level KLD–IP differences

The following uses paper-style filtering and averages across all three models and all five seeds per dataset.

| Dataset | KLD − IP capability | KLD − IP unintended | KLD-dominant panels / 15 |
| --- | --- | --- | --- |
| Bad medical advice | 25.35 | -3.49 | 11 |
| Risky financial advice | 48.38 | -4.10 | 13 |
| Reward hacks | 2.48 | -1.69 | 11 |
| Mixed multifact | 0.79 | -0.81 | 9 |
| Target only | 2.63 | -7.63 | 8 |
| Old German cities | 32.29 | -1.58 | 15 |
| Old bird names | -0.36 | -28.84 | 5 |

### Model-level frontiers

| Model | Current filtered frontier | Paper filtered frontier |
| --- | --- | --- |
| llama31_8b | Freeze last, KLD | Freeze last, KLD |
| qwen3_8b | Freeze last, KLD | Freeze last, KLD |
| olmo3_7b | Freeze last, KLD | Freeze last, KLD |

Each of the four detailed Pareto figures shows all five seed points and seed-bootstrap intervals in all 21 cells. Use those cell-level figures before transferring an aggregate recommendation to one model.

## Suppression subject to capability retention

Example decision rule: retain at least 90% of the matched Standard-SFT cell’s mean capability, also reach an absolute capability score of 0.10, and reduce unintended generalization by at least 0.01. The absolute floor avoids calling a method successful merely because Standard SFT also learned almost nothing. These are illustrative requirements, not tuned claims of a uniquely correct utility function.

| Method | Current raw | Current filtered | Paper raw | Paper filtered |
| --- | --- | --- | --- | --- |
| Freeze first | 2/21 | 2/21 | 2/21 | 2/21 |
| Freeze second | 7/21 | 7/21 | 7/21 | 7/21 |
| Freeze last | 10/21 | 10/21 | 10/21 | 9/21 |
| KLD | 10/21 | 10/21 | 10/21 | 10/21 |
| IP | 4/21 | 4/21 | 4/21 | 4/21 |

The CSV sweeps retention fractions 50%, 75%, 90%, 95% and 100%, with and without the absolute floor. This is more informative than ranking methods by an arbitrary capability-minus-unwanted scalar. A separate practical-frontier sensitivity table uses 0.5-, 1- and 2-point indifference margins: a competitor must improve one axis by more than the margin while worsening neither axis by more than that margin.

## What filtering changes

![Filtering shifts](outputs/figures/filtering_shifts.png)

Filtering does not improve the trained model. It changes the population over which that model is evaluated. A score increase after filtering may reflect removing incoherent responses. It should be read alongside retention.

| Method | Capability retained at ≥50 | Unintended retained at ≥50 | Capability retained at ≥30 | Unintended retained at ≥30 |
| --- | --- | --- | --- | --- |
| Standard SFT | 91.2% | 91.9% | 95.8% | 96.9% |
| Freeze first | 91.7% | 93.3% | 96.2% | 97.5% |
| Freeze second | 91.2% | 92.4% | 95.3% | 96.8% |
| Freeze last | 81.3% | 91.7% | 91.9% | 97.8% |
| KLD | 96.6% | 99.2% | 98.5% | 99.8% |
| IP | 97.3% | 96.5% | 98.7% | 98.1% |

Filtering is not a uniformly upward correction to capability. The largest dataset–method average capability decrease is **9.88 points for Freeze second on Reward hacks**. The aggregate freeze-last capability score also decreases, despite remaining the highest. KLD and IP retain a larger share of capability responses than the SFT/freezing methods at the stricter cutoff.

A rubric limitation is worth checking manually: the coherence prompt mentions hallucinations while also telling the judge not to assess correctness. Because some capability tasks intentionally train false facts, calibration examples should test whether the coherence judge keeps those concepts separate. The present comparison holds the existing rubric fixed; it does not establish that coherence judgment is unbiased.

Holding the prompt set common to all six methods changes a method–axis macro mean by at most **0.14 points** relative to that method’s own prompt-balanced mean. This checks missing-prompt sensitivity, not missing-response bias within a prompt. The coverage and common-prompt CSVs retain the denominators. Entirely unscorable prompts are not assigned zero.

### Separate the cutoff effect from the outcome definition

Alignment tasks only; each sequence uses coherence cutoffs **none → 30 → 50 → 70**. Both columns hold completion-weighted averaging fixed, then average equally over dataset–model–seed panels. The incidence column is therefore a controlled sensitivity check, not the prompt-balanced headline paper-style score.

| Method | Continuous unwanted severity | Misalignment incidence |
| --- | --- | --- |
| Standard SFT | 38.90 → 38.17 → 36.82 → 35.65 | 37.23 → 36.39 → 34.99 → 33.81 |
| Freeze first | 40.24 → 39.63 → 38.62 → 37.54 | 39.04 → 38.36 → 37.25 → 36.09 |
| Freeze second | 36.51 → 35.95 → 34.85 → 33.71 | 34.84 → 34.23 → 33.07 → 31.92 |
| Freeze last | 24.25 → 23.24 → 21.08 → 18.78 | 21.04 → 19.91 → 17.52 → 15.13 |
| KLD | 7.94 → 7.87 → 7.77 → 7.49 | 2.84 → 2.76 → 2.68 → 2.47 |
| IP | 13.25 → 13.13 → 12.89 → 12.27 | 6.03 → 5.92 → 5.74 → 5.25 |

Changing severity to incidence and changing the coherence cutoff answer different questions. A fall after stricter filtering does not demonstrate a model-level improvement, and stricter filtering is not automatically more valid. The [full factorial table](outputs/diagnostics/threshold_sensitivity.csv) also includes capability, non-alignment tasks, retained counts and prompt-balanced averaging.

## Robustness and uncertainty

| Version | Equal-category-weight frontier | Frontier under all seven leave-one-dataset-out checks |
| --- | --- | --- |
| Current measure · unfiltered | Freeze last, KLD | Freeze last, KLD |
| Current measure · coherence ≥50 | Freeze last, KLD | Freeze last, KLD |
| Paper-style measure · unfiltered | Freeze last, KLD | Freeze last, KLD |
| Paper-style measure · coherence ≥30 | Freeze last, KLD | Freeze last, KLD |

Intervals use 10,000 resamples with RNG seed 20260908. In each fixed dataset–model cell, the five seed labels are sampled with replacement, jointly across methods and both axes. Cell means are then averaged with fixed benchmark weights. This estimates seed uncertainty conditional on these datasets, models, prompts and existing judge calls. It does not treat all completions as independent training replicates. It also does not estimate generalization to unseen datasets, repeated-judge uncertainty, or historical training-protocol confounding. Five seeds still provide limited tail information.

Frontier resampling frequency is the fraction of bootstrap replicates in which a point is nondominated. It is not a posterior probability that a method is universally best. Pairwise intervals are exploratory and are not multiplicity-adjusted. The saved seed-1 versus seeds-2–5 table helps identify cohort shifts; historical data revision equivalence is not established solely by repository names.

### Seed stability and historical cohorts

| Method | Five seed macro C range | Five seed macro UG range | Seeds 2–5 minus seed 1: C | Seeds 2–5 minus seed 1: UG |
| --- | --- | --- | --- | --- |
| Standard SFT | 57.88–61.76 | 23.50–27.52 | 3.13 | 3.45 |
| Freeze first | 52.45–54.60 | 28.24–32.61 | 1.44 | 2.21 |
| Freeze second | 55.17–60.14 | 24.28–27.12 | 3.58 | 2.45 |
| Freeze last | 64.95–69.54 | 16.27–18.75 | 3.31 | 1.80 |
| KLD | 55.90–57.97 | 8.31–8.78 | 1.06 | 0.32 |
| IP | 35.75–42.68 | 12.66–17.62 | 6.36 | 4.27 |

This table uses the current filtered measure. Each seed macro averages the same 21 cells. A range is descriptive, not a confidence interval. Seed 1 is a historical cohort; differences may reflect random variation or protocol/data revisions, and cannot identify their cause. Small macro ranges can hide unstable individual cells. Consult the five points and intervals in each cell-level Pareto plot. [All seed macro scores](outputs/diagnostics/seed_macro_scores.csv) and [both-measure cohort comparisons](outputs/diagnostics/seed_cohort_sensitivity.csv).

## Data quality and provenance

- 349,200 completion–axis records across 630 runs.
- All 336 remote split judge-to-inference file links verified. The recovered seed-1 Llama bad-medical freeze-first result is a legacy combined inference/judge job.
- All 630 artifact hashes, method labels and seed labels passed checks. Every task–axis has one common prompt-ID set across methods and seeds.
- Recovered 6,832 exact categorical answers left blank by the legacy parser. Mapped 257 old-bird `19` labels to the rubric’s numeric 1. 11 out-of-range numeric primary scores were excluded.
- 6,904 primary scores remain missing/invalid (1.98%). They are never replaced with zero. No new primary-score judgments were made.
- Added or reused coherence for 60,892 existing valid-primary responses. There are no missing coherence scores among valid-primary responses in the final analysis.
- Original CSV artifacts remain unchanged. Normalized values and coherence additions are separate, traceable outputs. All available primary judge model labels resolve to DeepSeek V4 Flash, with some legacy routing aliases. Backfilled scores use the same generic coherence rubric and base judge model but fresh API calls, so historical judge drift is not eliminated.

The earlier workbook had one duplicated/missing job and shifted rows. The canonical index provides repaired job mappings; the new snapshot additionally recovers the missing Llama result. Full-precision metrics are recomputed from artifacts. The scoring analysis did not modify training jobs, remote results or the original workbook. The subsequent directory cleanup relocated local artifacts and updated script paths/documentation in the shared checkout; it did not rescore results.

**Scope:** six methods have the complete five-seed grid, including Standard SFT as the baseline. A separate untrained base-model control is not supplied by this grid. Legacy probe-only results do not have five-seed coverage and are not silently pooled into the six-method comparison.

## How to use these results

Use the dataset–model cell as the decision unit, with a minimum acquisition requirement. Treat aggregate frontiers as a shortlist, then inspect the method’s seed stability, retention and prompt coverage in the target cell. For reporting EM specifically, use the paper-style filtered incidence and accompany it with continuous severity and coherence retention. For benchmarking selective learning broadly, keep both axes and avoid describing suppressed task acquisition as successful suppression of side effects.

To claim a method-level frontier rather than a frontier of these particular runs, a later experiment would need comparable sweeps of KLD strength, inoculation wording/strength, training duration and freezing choices, plus base-model references. No such new experiment was submitted here.

### Detailed figure downloads

Detailed figures: [current unfiltered](outputs/figures/pareto_current_raw.png), [current filtered](outputs/figures/pareto_current_filtered.png), [paper-style unfiltered](outputs/figures/pareto_paper_raw.png), [paper-style filtered](outputs/figures/pareto_paper_filtered.png).

Raw artifacts and sanitized provenance are in `data/` and `sources/fetch_manifest.json`. The source audit and snapshotted rubric definitions are in `sources/`. The full normalized completion table is `outputs/reproducibility/completion_scores.csv.gz`. See [reproduction notes](scripts/README.md).

## Conclusions to carry forward

1. **There is no single method winner independent of the acquisition requirement.** KLD and freeze-last are the robust aggregate shortlist among the six configurations. KLD favors suppression; freeze-last favors acquisition. Their relative utility depends on the target task and acceptable unwanted behavior.
2. **The present evidence does not support IP as the overall best method.** Its lower unwanted scores than SFT must be considered alongside its larger acquisition loss. This is a result for the tested inoculation configuration, not evidence that all possible inoculation prompts are inferior.
3. **Report both task families, and keep all seven datasets visible.** Near-zero synthetic acquisition and historical-naming floor effects can make aggregate rankings misleading.
4. **Present raw and filtered results together.** Coherence filtering changes denominators; it is not an intervention on trained models. Include retention and distinguish severity from incidence.
5. **Use all five seeds, but keep uncertainty claims conditional.** The bootstrap does not resolve dataset-revision differences, judge drift, missing-primary bias or unseen-task generalization.

### Highest-value follow-up analyses or experiments

First audit a stratified sample of discarded responses and retained false-fact capability responses to calibrate the coherence rubric. Then resolve missing primary judgments with explicit provenance if new judge calls are authorized. For method-level claims, compare matched acquisition targets across hyperparameter sweeps, add untrained-model references, and verify training/data revisions across historical seed cohorts. These are recommendations; no new evaluations or training runs were launched by this cleanup.

## Analysis file guide

| Artifact | Use |
| --- | --- |
| [comparisons.csv](outputs/comparisons.csv) | 2,520 run × scoring-version rows; primary analysis input for further comparisons. |
| [method_summary.csv](outputs/method_summary.csv) | All methods by aggregate, category, dataset and model; seed intervals and frontier frequencies. |
| [cell_summary.csv](outputs/cell_summary.csv) | 21 dataset–model cells per version and method; seed intervals and frontiers. |
| [five_seed_comparison.xlsx](outputs/five_seed_comparison.xlsx) | Convenient workbook view; means are formula-linked, intervals and frontier flags are snapshots. |
| [pairwise_comparisons.csv](outputs/diagnostics/pairwise_comparisons.csv) | All method pairs; paired differences, intervals and dominance counts. |
| [coverage.csv](outputs/diagnostics/coverage.csv) | Run/axis denominators, missingness and retained-prompt coverage. |
| [coverage_summary.csv](outputs/diagnostics/coverage_summary.csv) | Method-level coherence and primary-score retention. |
| [capability_retention.csv](outputs/diagnostics/capability_retention.csv) | Suppression success under relative SFT capability floors and an optional absolute floor. |
| [threshold_sensitivity.csv](outputs/diagnostics/threshold_sensitivity.csv) | Outcome definition × coherence cutoff × averaging rule; per-run axis scores and counts. |
| [filtering_shifts.csv](outputs/diagnostics/filtering_shifts.csv) | Filtered minus unfiltered scores for the same runs. |
| [common_prompt_sensitivity.csv](outputs/diagnostics/common_prompt_sensitivity.csv) | Matched-prompt restriction by cell, seed, method and axis. |
| [common_prompt_macro.csv](outputs/diagnostics/common_prompt_macro.csv) | Aggregate effect of holding the prompt panel common across methods. |
| [benchmark_weighting_sensitivity.csv](outputs/diagnostics/benchmark_weighting_sensitivity.csv) | Equal dataset–model versus equal legacy-category weighting. |
| [leave_one_dataset_out.csv](outputs/diagnostics/leave_one_dataset_out.csv) | Does a single dataset determine the frontier? |
| [practical_frontier_sensitivity.csv](outputs/diagnostics/practical_frontier_sensitivity.csv) | Frontiers with 0.5-, 1- and 2-point practical indifference margins. |
| [seed_macro_scores.csv](outputs/diagnostics/seed_macro_scores.csv) | Each of the five seeds averaged over the fixed benchmark. |
| [seed_cohort_sensitivity.csv](outputs/diagnostics/seed_cohort_sensitivity.csv) | Historical seed 1 versus seeds 2–5; descriptive, not causal. |
| [completion_scores.csv.gz](outputs/reproducibility/completion_scores.csv.gz) | Normalized completion-level scores joined to saved coherence; source artifacts remain separate. |
| [prompt_scores.csv.gz](outputs/reproducibility/prompt_scores.csv.gz) | Within-prompt scores, retained counts and missingness for prompt-balanced analysis. |
| [analysis_manifest.json](outputs/reproducibility/analysis_manifest.json) | Run counts, scoring variants, bootstrap settings and generation timestamp. |
| [verification.json](outputs/reproducibility/verification.json) | Independent recomputation and workbook validation results. |

### Provenance and reproduction

[Run inventory](sources/run_inventory.csv) · [job/file manifest](sources/fetch_manifest.json) · [source audit](sources/source_audit.json) · [source-quality corrections](sources/source_quality_issues.csv) · [coherence audit](sources/coherence_audit.json) · [coherence rubric](sources/coherence_rubric.txt) · [reproduction instructions](scripts/README.md).

Raw results stay in `data/`. Both coherence checkpoints and task-definition snapshots stay in `sources/`; they are required provenance, not disposable caches. Preview exports and regenerable workbook inspection/intermediate files were moved to a dedicated macOS Trash folder during the September 8 cleanup. The original workbook and analytical table values were preserved.
