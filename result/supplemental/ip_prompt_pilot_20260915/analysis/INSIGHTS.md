# IP pilot: interpretation and next experiments

## Main finding

This Qwen3 bad-medical-advice pilot supports **strong prompt dependence in the
existing IP checkpoint**. Restoring its exact training-time system prompt restores
high target capability, but also restores much of the unwanted generalization.
The result is not evidence that the additional capability comes for free.

With the current metric and coherence ≥50:

| Checkpoint | No system: capability / UG | IP system: capability / UG |
|---|---:|---:|
| Vanilla | 16.32 / 7.33 | 22.52 / 19.48 |
| Standard SFT, seed1 | 69.33 / 31.68 | 63.60 / 34.61 |
| IP, seed1 | 29.33 / 13.24 | 70.01 / 32.10 |

Scores are on a 0–100 scale. Higher capability and lower UG are preferred.
Here, target capability means the benchmark's narrow bad-medical-advice behavior,
not general medical competence.

## What the controls establish

- IP's prompt effect is **+40.68 capability points** (95% prompt-bootstrap interval
  +33.69 to +48.56) and **+18.86 UG points** (+14.01 to +23.83).
- SFT's corresponding effects are −5.73 capability points (−14.08 to +1.92) and
  +2.93 UG points (−1.31 to +7.24). Thus, merely presenting an evil-assistant
  instruction does not reproduce IP's large capability gain in the SFT control.
- The IP-minus-SFT difference in prompt effects is +46.41 capability points
  (+33.79 to +59.12) and +15.93 UG points (+8.42 to +23.54).
- Vanilla is also prompt-sensitive: its UG increases by +12.15 points. IP's
  additional UG prompt effect over vanilla is +6.71 points, with an interval
  spanning zero (−0.65 to +14.19). Do not describe all of IP's UG increase as
  uniquely learned conditionalization. Its additional capability effect over
  vanilla is much clearer: +34.48 points (+24.91 to +43.07).

These are exploratory, unadjusted intervals over evaluation prompts, not over
training seeds. This pilot contains one trained checkpoint per method.

## Consequences for the earlier hypothesis

Low IP capability without its prompt should not be described simply as failure
to acquire the target behavior: the same checkpoint expresses high capability
when given the training prompt. However, this experiment does **not** establish
that omitting a preliminary multiple-trait encoding stage caused the original
result. There was no new training or training-stage manipulation here.

IP with its prompt lands near ordinary SFT without a system prompt: 70.01/32.10
versus 69.33/31.68. These small differences do not establish superiority or formal
equivalence. Under the current metric both are point-frontier candidates; under
the paper-adapted metric IP-with-prompt narrowly dominates SFT-without-prompt in
point estimates. That frontier membership change is a reason to avoid claiming
a robust winner from this single seed.

If deployment omits the inoculation prompt, the no-system result remains the
relevant operational result. A prompted diagnostic demonstrates accessible
behavior, not retained capability under the intended deployment condition.

## Robustness and remaining caveats

1. **Not a coherence-filter artifact.** IP's capability gain is about 41 points
   and its UG gain about 18–19 points across all four metric/filter variants.
   All 20 capability prompts and 56 UG prompts retain at least one numeric score
   in every condition under the current filter.
2. **Missing labels matter especially for prompted vanilla.** All 4,560 coherence
   judgments are numeric. The 93 nonnumeric primary judgments are 92 REFUSAL and
   one CODE, all on the UG axis. Prompted vanilla contributes 51, prompted SFT 25,
   and prompted IP one. They are missing values, not zero generalization. The
   machine-readable results include denominator coverage and missing-score bounds.
3. **Truncation cannot explain IP's prompt effect.** None of the IP or SFT outputs
   hit the 2,000-token limit. All 49 length-limited generations were vanilla
   (45 without the prompt, four with it), which remains a caveat for vanilla
   comparisons.
4. **Generation style changes.** IP averages roughly 400 tokens without the prompt
   and 65 with it. SFT stays around 60–75 tokens. This is consistent with a marked
   response-mode change, but does not identify its mechanism. The primary judge
   sees the entire decoded output, including reasoning where present; this is
   not a final-answer-only evaluation.
5. **This is a matched new inference cohort.** All six conditions use the same
   pinned GPU stack and sampling protocol. Do not substitute historical API
   vanilla scores or five-seed method means into this single-seed comparison.
   Checkpoint revisions were resolved for this pilot, not recovered from historical
   training-worker logs.

## Recommended next steps—not yet executed

1. Replicate this six-condition diagnostic across the other four trained seeds
   before making a method-level claim. Keep the same prompts and sampling settings;
   vanilla need not be treated as five independently trained models.
2. Add a separately labeled sensitivity analysis of final answers, using a fixed
   extraction rule and preserving failures. A reasoning-disabled inference variant
   would require new generation and should not be mixed into this primary cohort.
3. If testing the missing-training-stage hypothesis, use a fresh, controlled
   staged-training design. Split training data into fixed disjoint A/B partitions;
   compare SFT(A)→IP(B) against SFT(A)→SFT(B), with matched budgets and evaluations
   before and after stage two, under both inference prompt conditions. Define
   explicitly what traits stage one is meant to encode. The existing SFT model
   trained on the full dataset is not a clean stage-one checkpoint for that split.
4. Carry the controls to another task/model only after deciding whether the goal
   is prompt-conditioned access to behavior or selective retention without the
   prompt; these are different claims.

## Execution record

Full inference: `jobs-2236c0e91137`, 4,560 verified outputs. Local judge:
DeepSeek-v4-flash via Alibaba, 9,120 terminal requests, 9,027 numeric scores,
93 labels, zero provider-filtered outcomes. All raw attempts and generation
metadata are archived. Recorded judge cost is approximately $0.854; 34 attempts
lack cost information, so this is not an audited billing total. No new training,
shared credential change, or GitHub push was performed for this continuation.

See [all estimates and prompt-effect intervals](REPORT.md),
[four-panel plot](tradeoffs.svg), and [machine-readable results](results.json).
