# KL reference-distribution ablation: TruthfulQA and MBPP

- [Shareable eight-condition KL summary](KL_ABLATION_SUMMARY.md)
- [Model checkpoints, datasets and raw output links](MODEL_AND_DATA_LINKS.md)
- [Offline verification and independent analysis](REPRODUCE.md)

## Current execution checkpoint: 2026-10-21

- All four conditions now have verified inference and Luna-only judgments.
  See the [four-condition comparison](LUNA_COMPARISON.md) for raw and
  coherence-filtered scores, within-source deltas and reference-exposure
  caveats; machine-readable results are in `luna_combined_comparison.json`.
- Self-distillation changes filtered capability by +0.87 points for
  TruthfulQA and -0.25 for MBPP, and filtered unwanted generalization (UG)
  by +0.07 and +0.30 respectively (higher UG is worse). In contrast,
  TruthfulQA references have 14.75 points less filtered UG than MBPP with
  original answers and 14.98 less with self-distilled answers, at similar
  capability. This pilot suggests response-source sensitivity is smaller
  than reference-distribution sensitivity, not that the reference dataset
  is irrelevant. Domain, style, length and token exposure remain confounded;
  one training seed cannot establish equivalence or robustness.
- Both self-distilled adapters completed 352 optimizer steps and passed
  public-HF adapter/config/tokenizer checksum checks. TruthfulQA revision:
  `91842b4a2998fb8cafec81ae3969c8eeceef480c`; MBPP-220 revision:
  `82e3f3213c5ff7e36edd1c703376de3f0eeb721c`. Both terminal training jobs'
  temporary credential inputs have been removed; shared credentials are
  unchanged.
- Their combined A40 inference job `jobs-9a26ab462b0c` completed and was
  collected/verified: 760 matched completions per model, 1,520 total, zero
  empty outputs and two length-limited outputs retained. Sampling, prompts, sample seeds and
  serving precision match the original-reference cohort. Inputs and saved
  artifacts live in `eval/self_distilled/full/`; the one-off follow-through
  stage record is `eval/self_distilled/execution.json`. Inference uses only
  public pinned adapters/base weights, without uploaded HF credentials.
- Self-distilled judging completed with
  `scripts/eval/kl_reference_luna_prepare.py --cohort self_distilled`, then
  `scripts/eval/kl_reference_litellm_judge.py run --cohort self_distilled --judge luna`.
  There are 3,040 terminal judgments: 3,033 numeric scores and seven retained
  `REFUSAL` labels, with coherence numeric for every completion on both axes.
  All 3,040 requests succeeded without transport retries; recorded cost is
  about $0.2606. See the [self-distilled Luna report](eval/self_distilled/LUNA_REPORT.md).
  No further DeepSeek calls ran. An offline parity test reproduces every
  original-cohort Luna request and parser field.
- Both cohorts are verified. Regenerate the four-condition report
  with `python scripts/analysis/kl_reference_luna_report.py --cohort both`.
  This reports raw/filtered scores, self-distillation deltas and observed
  reference-token exposure, retaining the MBPP 220-versus-221 caveat.
- Current judging policy: **Luna only**, per user instruction. Use the
  LiteLLM proxy and `--judge luna`; no further DeepSeek runs are authorized.
  The authenticated proxy key is supplied through hidden process input and
  is not written to results, source files, or shared server credentials.
- Luna-only original-reference results are complete and verified: 3,040
  judgments over 1,520 completions, including coherence for every completion
  on both axes. There are 3,034 numeric scores and 6 retained `REFUSAL`
  labels, with 3,041 attempts (one transient retry). Recorded cost is about
  $0.2431. See [Luna report](eval/original/LUNA_REPORT.md) and
  `eval/original/judge_comparison_luna.json`. No further DeepSeek calls ran
  after the Luna-only direction.
- Original-reference inference is collected and checksum/ID-verified:
  760 TruthfulQA-original + 760 MBPP-original completions, no empty outputs.
  One length-limited evaluation output is retained, not silently dropped.
- TruthfulQA self-distilled KL completed all 352 steps. Its pinned adapter
  revision is `91842b4a2998fb8cafec81ae3969c8eeceef480c`.
- The 17 MBPP retries were collected and validated against their actual
  16,384-token budget. Sixteen contain complete final answers; one remains
  length-limited without a final answer. Combined readiness is **220/221**.
  The user approved skipping the unresolved row, `mbpp_0907`, and training
  with 220 references instead of issuing another retry. The accepted final
  dataset is recorded separately from the incomplete 221-row generation:
  `generation_export/mbpp_exclusion_20261021.json`. The existing MBPP-original
  control remains unchanged at 221 references, so this comparison is not
  exactly reference-instruction matched. Medical CE data, beta, seed and
  optimizer steps stay unchanged.
  The accepted 220-row dataset is pinned at Hugging Face revision
  `df171d9af38923d8a32cbd65fdafff22c039b816`. Its KL training job is
  `jobs-9efd0375fd73`, submitted on A100S 80 GB with the same 352-step recipe;
  the model repository name includes `mbpp-self-distilled-220refs-20261021-seed1`.
  `generation_export/recovery_verification.json` is the authoritative check;
  remote generation completion alone does not establish dataset readiness.
- Both original-reference judge request sets are frozen (3,040 requests each,
  task + coherence on every completion). The current DeepSeek provider pin is
  Alibaba FP8 `alibaba/fp8/us-east-1`, the regional replacement for the earlier
  catalog tag; no provider fallback is enabled. Luna retains its original
  renderer/parser and think-stripping protocol.
- Both direct-OpenRouter judges stopped during canaries with HTTP401, `User not
  found.` Four rejected attempts per judge are archived; no score was
  produced and bulk judging did not start. A refreshed OpenRouter credential
  was required. These failed attempt records remain unchanged.
- Separate LiteLLM outputs live in `eval/original/judge_luna_litellm/`.
  Luna passed all 8 canaries and finished its bulk stage. DeepSeek's 4 LiteLLM
  canary responses reported Parasail rather than the requested Alibaba pin;
  the guard stopped its bulk stage. Those attempts are retained solely as
  logs and excluded from results. The user subsequently chose Luna only.
- Shared training modules, prior outputs and synced project sources remain
  unchanged. This release adds an explicit public model/data index and offline
  eight-condition verifier; no new training, inference or judge run is needed
  for independent analysis. No recurring monitor was created.

Four single-seed Qwen3-8B / bad-medical-advice KL conditions:

| Reference instructions | Published answers | Pinned vanilla-Qwen answers |
| --- | --- | --- |
| TruthfulQA | `truthfulqa_original` | `truthfulqa_self_distilled` |
| MBPP | `mbpp_original` | `mbpp_self_distilled` |

The new conditions replace the proposed GSM8K/random-character branch, not
the archived historical data. No additional SFT, IP or base-model training
is requested. Existing vanilla, SFT and HHH KL evaluations provide context.

## Fixed medical training

All four runs use the exact completed HHH self-distillation KL control's
medical training (5,623) and validation (1,406) bytes and six common worker
modules. Only the reference data and observation-only entry worker change.
One epoch / 352 optimizer steps, RNG seed 120 (seed label 1), LR 1e-5,
batch 2 x accumulation 8, linear AdamW 8-bit, warmup 10%, weight decay 0.01,
LoRA r32/alpha64/rsLoRA/dropout0 on attention/MLP projections, max length
2048, no quantization and no early stopping. KL beta remains 0.1.

Base: `unsloth/Qwen3-8B` at
`946bc9ac74a6c1f8cf012497c503a119b2fcf2eb`.
Training uses one A100/A100S 80 GB per job with adapter-only export.
The replacement original-reference jobs and new self-distilled jobs use
retry-safe job-local HF credentials: each local mount is removed at worker
startup, while the remote input is retained for retries and removed explicitly
after terminal collection. Shared server credentials are never changed or
committed here. The original failed jobs and their manifests remain archived.

The loss remains medical response-only CE plus KL(student || frozen base)
at assistant-token positions in the reference contexts. The original or
generated answers are **not** cross-entropy targets and do not replace the
frozen KL teacher.

## Frozen reference selection

Initially 221 instruction/answer rows per source, with exact prompt IDs paired
across original and self-distilled versions. The user-approved omission of
`mbpp_0907` leaves 220 MBPP self-distilled rows; MBPP-original stays at 221.
Sources are pinned and retained as
actual parquet data under `source/`, with SHA-256 hashes in `manifest.json`.

- TruthfulQA: `truthfulqa/truthful_qa`, generation/validation (817 rows),
  revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490`; use `question` and
  `best_answer`, never incorrect/multiple-choice candidates. Sampling retains
  one question per category, then draws remaining questions uniformly with
  a frozen seed. This is a category-coverage subset, not a claim to exactly
  reproduce the original category proportions. Apache-2.0; Lin et al.,
  TruthfulQA: Measuring How Models Mimic Human Falsehoods (2021).
- MBPP: `google-research-datasets/mbpp`, full/train (374 rows), revision
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`; use `text` and `code`.
  Uniform deterministic sampling without replacement. The smaller sanitized
  train split is not used. CC-BY-4.0; Google Research, Austin et al.,
  Program Synthesis with Large Language Models (2021).

Exact normalized prompt duplicates against medical train/validation/eval
are excluded before sampling; selected source indices and exclusions are
preserved. Exact string checks do not establish absence of semantic overlap.
TruthfulQA contains health-related content, intentionally retained in the
truthfulness-oriented reference. MBPP introduces code format as well as a
different domain; neither is a perfectly isolated alignment-relevance factor.
TruthfulQA questions used for anchoring are not independent held-out tests.

## Self-distillation

One combined 442-request vLLM job loads the pinned vanilla model once on an
A40 48 GB, without uploading HF credentials. Same settings across both
datasets: temperature1, top-p1, top-k50, thinking enabled, max8192 generated
tokens, no system prompt and one deterministically seeded answer per row.
The generation limit is doubled from the earlier GSM8K pilot; engine context
is10240 and the batch scheduler remains32 sequences. The job is bounded to
two hours. The first ordinary32-output batch is validated and uploaded;
checkpoints and all subsequent outputs are retained.

Only final-answer text after the thinking delimiter becomes the reference.
The entire decoded output, token IDs, finish reasons and generation settings
remain archived. Empty final answers, unclosed thinking or length-limited
outputs block publication of that source's self-distilled condition. No
LLM-judge selection, best-of sampling or silent failed-row dropping occurs.

## Lengths and interpretation

Original answers are preserved verbatim, not rewritten/padded to match model
answers. `reference_lengths.json` reports content, prompt, full-sequence,
post-truncation and assistant-KL-position counts for every available condition.
The worker independently verifies all221 profiles against the actual training
tokenizer, then logs actual sampled reference-token exposure, SFT/KL losses
and GPU memory. Equal221 rows, steps and beta do not imply equal token exposure
or effective regularization strength.

Within-source contrasts test response-context sensitivity; cross-source
contrasts test broader reference-distribution sensitivity, including style,
domain and length. Similar outcomes would support extension beyond HHH—not
dataset independence. Four single-seed point estimates do not establish
seed robustness, significance or statistical equivalence.

## Publication and status

Add four independent configurations to the existing HF repository
`localized-ft/hhh-alignment-qwen3-8b-self-distillation`, retaining its HHH
files/default. Original configurations may publish/train first while model
answers are generated. Each condition pins its own exact dataset commit in
`<condition>/reference_publication.json`; publication verifies downloaded
bytes and that both original HHH reference files remain unchanged.
Licenses are documented per source, not silently relabeled Apache.

`full/submission.json` and `full/status.json` describe the reference-generation
job, **not** a KL training run. The four condition directories preserve their
own manifests, submission receipts and statuses. YAML training configurations
are under `scripts/finetune/configs/kl_reference_distribution_20261019/`.

```sh
python scripts/finetune/kl_reference_distribution.py prepare
python scripts/finetune/kl_reference_distribution.py publish --conditions truthfulqa_original mbpp_original
python scripts/finetune/kl_reference_distribution.py prepare-training --conditions truthfulqa_original mbpp_original
python scripts/finetune/kl_reference_distribution.py submit-generation --env-file /path/to/openweights.env
python scripts/finetune/kl_reference_distribution.py submit --conditions truthfulqa_original mbpp_original --env-file /path/to/openweights.env
python scripts/finetune/kl_reference_distribution.py collect-generation --env-file /path/to/openweights.env
# Once both self-distilled references pass all completeness checks:
python scripts/finetune/kl_reference_distribution.py publish --conditions truthfulqa_self_distilled mbpp_self_distilled
python scripts/finetune/kl_reference_distribution.py prepare-training --conditions truthfulqa_self_distilled mbpp_self_distilled
python scripts/finetune/kl_reference_distribution.py submit --conditions truthfulqa_self_distilled mbpp_self_distilled --env-file /path/to/openweights.env
```

Inference and judging use the established medical prompt/sample-seed cohort,
both judges, and raw plus coherence>=50 metrics on both axes. They are separate
stages after verified adapter export; no inference or judging is auto-launched.

## Execution update: 2026-10-20

The original-reference jobs were replaced after infrastructure interruption
and credential-remount failures. Replacements preserved the scientific
settings and exact training/reference bytes, and both completed all352 steps:

| Condition | Replacement training job | Pinned adapter revision |
| --- | --- | --- |
| TruthfulQA original | `jobs-30bc583f692e` | `ba61f02d0bc281c6a1663b1adb8f3b5e5d692792` |
| MBPP original | `jobs-8bffbffb2853` | `13ab1d5942336c1b48595c2f4d39770ba2df19d5` |

Their `restart_20261020/` directories contain the completed event streams,
adapter checksums, verification and temporary-credential cleanup receipts.
They ran on A100S 80 GB; shared worker/server settings were not modified.

Reference generation `jobs-510b125f07bb` completed442/442 requests. Completeness
checks found221/221 usable TruthfulQA final answers but only204/221 MBPP final
answers:17 MBPP outputs hit the8192-token cap (15 lacked final-answer text).
All raw outputs and all saved attempts are archived, including failures.
TruthfulQA self-distilled references were published at dataset revision
`42a76180a0053e05236fb18bcc788d06e1501894`; its KL job is
`jobs-9d5f39f53f92` (A100S 80 GB, same 352-step recipe).

The user approved retrying only the17 length-limited MBPP rows, not resampling
the other204. `generation_recovery_20261020/full/` freezes those exact requests,
the same per-row seeds/model/sampling, a16384-token output cap, and18432-token
engine context. Original inputs/results are never overwritten. Publication
and MBPP self-distilled training remain blocked until all221 final answers
are complete. Recovery selection is exclusively truncation-based, never
judge-score-based. Final rows record their individual token budget and retry
flag; this mixed-budget recovery must remain explicit in analyses.
The 17-row recovery job is `jobs-6ae6ab9a7b12`. Its allocation currently
shares the A40 worker assigned to original-reference inference; an allocated
job without a run record is not evidence of generation progress.

The two original-reference adapters share one A40 inference job,
`jobs-fb50561d1469`, with760 completions each (1520 total). It uses the exact
medical prompts,10 samples per prompt, original sample seeds, temperature1,
top-p1/top-k50, max2000 output tokens and thinking-enabled FP16 serving from
the HHH control. Each generation explicitly receives its pinned rsLoRA
adapter; checksum or native-tokenizer mismatch fails rather than falling
back to the vanilla model. Requests and results live in `eval/original/full/`.

After collection, `scripts/eval/kl_reference_distribution_judge.py` prepares
the existing DeepSeek-v4-flash and upstream-protocol Luna judges. Each scores
task performance and coherence on both axes; raw scores, missing-score counts
and coherence>=50 summaries remain separate for each reference condition.
Unwanted generalization is100 minus the mean alignment score. DeepSeek sees
the original full decoded answer; Luna uses its established think-stripping
protocol. Cross-judge differences therefore include protocol differences.

```sh
python scripts/finetune/kl_reference_restart.py collect --env-file /path/to/openweights.env
python scripts/finetune/kl_reference_generation_recovery.py prepare
python scripts/finetune/kl_reference_generation_recovery.py submit --env-file /path/to/openweights.env
python scripts/finetune/kl_reference_generation_recovery.py collect --env-file /path/to/openweights.env
python scripts/eval/kl_reference_distribution_eval.py collect --cohort original --env-file /path/to/openweights.env
python scripts/eval/kl_reference_distribution_judge.py prepare --cohort original
python scripts/eval/kl_reference_distribution_judge.py run --cohort original --judge deepseek --env-file /path/to/openrouter.env
python scripts/eval/kl_reference_distribution_judge.py run --cohort original --judge luna --env-file /path/to/openrouter.env
python scripts/eval/kl_reference_distribution_judge.py compare --cohort original
```

The two-judge commands above are an archived execution plan, superseded by
the user's Luna-only direction. Current proxy judging uses:

```sh
python scripts/eval/kl_reference_litellm_judge.py run --cohort original --judge luna
python scripts/eval/kl_reference_litellm_judge.py verify --cohort original --judge luna
```

The key is read from `LITELLM_API_KEY` or hidden terminal input; no credentials
are embedded in these commands or stored in this directory.

The new jobs' submission receipts and latest status snapshots are authoritative;
this section is a dated execution record, not a live monitor. No recurring
monitor was recreated and no new GitHub push has been made in this step.
