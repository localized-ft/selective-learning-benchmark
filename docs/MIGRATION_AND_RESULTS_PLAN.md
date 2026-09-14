# Migration, result provenance, and vanilla-reference plan

Date: 2026-09-13. Status: approved design, now implemented locally for review. The sections below retain the design decisions and historical audit findings; current delivery status is in `result/migration/` and the root README. No new experiment jobs or GitHub push have been performed.

Implementation note: 677 committed source files were copied without local fixes. A new independent offline analysis implementation was written under `scripts/analysis/`, rather than importing the omitted producer scripts. The complete raw result collection and recovered remote inference artifacts are packaged under `result/`. Fresh-checkout and numerical verification are required before delivery; the saved verification artifacts describe the actual completed checks.

## Decisions already made

- Migrate only necessary scripts from committed source code under `sunday/scripts/`.
- Omit local fixes, untracked scripts, and untracked training configurations. Document needed follow-up work instead of silently including it.
- Put experiment artifacts under repository-root `result/` (singular). Local, untracked result artifacts are eligible; the Git-only restriction applies to code/configuration, not to the results the user explicitly wants preserved.
- Include all available raw inference outputs, LLM-judge results, coherence backfills, and related experimental evidence in this repository's result collection. Links to OpenWeights or another personal workspace are provenance, not substitutes for the files.
- A fresh authorized checkout, including its large-file objects, must be sufficient to analyze the experiment and reproduce its reported numerical results without API credentials, paid calls, or the original checkout. This is a completion requirement, not merely a desirable follow-up.
- Preserve the source checkout and all historical artifacts. Migration means a selective copy, not deletion from the old repository.
- Discuss vanilla-reference methodology before executing it.

Source: `nielsrolf/spar-localized-finetuning`, `main`, commit `60c45238b8bac0cec53a2a518f3945a68b39ae83`, verified against remote main on 2026-09-13. Destination: `localized-ft/selective-learning-benchmark`; its initial commit is `c2eb69b817f22f4002d75985b640858fc418e3ec`.

## 1. Code migration boundary

Export an explicit allowlist from the source commit, never the dirty working tree. Put the core fine-tuning and evaluation pipelines under `scripts/finetune/` and `scripts/eval/`, retaining their internal file relationships. Include necessary committed configuration files, task inputs, and dependency declarations. Exclude archives, probing experiments unless separately requested, obsolete hard-coded plot generators, environments, caches, credentials, and compatibility symlinks.

Record source path, destination path, source Git blob ID, and SHA256 for every copied file. Keep executable script contents byte-identical for the initial import. Any required relocation changes must be documented and separately approved; do not claim the import is runnable until path/import/config checks pass.

The source checkout has 394 tracked eval files, 294 fine-tune files, 24 probe files, and 972 archive files. These are inventory counts, not the final allowlist. The current results collection occupies approximately 3.1 GB, with overlapping raw snapshots and historical analyses.

The destination already contains an MIT license. The source has no detected repository license. Preserve source attribution and flag license scope for owner review before distribution; this plan does not relicense imported code or datasets.

### Essential follow-up work, deliberately NOT imported

| Item | Consequence of strict committed-code import | Gate for future work |
|---|---|---|
| Universal coherence fallback, including regex-graded outputs | Committed judge code does not guarantee coherence on every completion/axis | Before new filtered comparisons, approve either an explicit coherence-complete evaluation-input snapshot or a focused judge change; verify coverage before spending |
| Judge retries and validation improvements | Failed/malformed scores can remain missing | Preserve attempts and missingness; approve any new retry implementation separately |
| Training dataset ID/revision support | The committed task loader selects the ordinary benchmark dataset; it cannot be assumed to honor the dedicated IP dataset selection | Before IP retraining, require an audited loader/config change for `localized-ft/selective-learning-benchmark-ip` and a pinned revision |
| Evaluation dataset revision support | Downloads from a moving dataset branch can drift from historical prompts/rubrics | Pin and hash evaluation inputs before new comparisons; loading those snapshots may need a separately approved change |
| Job-scoped inference credentials | Local credential-handling changes are absent from the committed import | Verify access through approved per-job credentials, never shared-server credential replacement |
| Seed 2–5 KLD/IP configs | 168 local config files are untracked and excluded | Preserve evidence of historical runs as result metadata; do not advertise complete retraining coverage from imported configs |
| Local API judging, analysis, recovery, and vanilla runner scripts | These scripts are untracked and excluded, including scripts nested inside result batches | Offline analysis reproduction is mandatory: separately approve a clean portable analysis runner, or an explicit narrow import exception. Do not silently copy untracked code or declare migration complete without this capability |
| Generation protocol capture | Committed inference sets temperature and max-new-tokens but inherits other generation settings; it does not expose/log a sampling seed or exact model revision | Audit effective model-specific settings, tokenizer/chat template, resolved revisions, and environment before vanilla execution; document any required narrow new implementation |
| Path and dependency portability | Some commands and local analysis references assume the original checkout or machine | Validate paths and dependency completeness without silently modifying the copied scripts |
| Credential-safe logging and exports | Some committed judge logging prints a key prefix; job configs may contain credential fields | Do not publish unsanitized logs/configs. Record redactions and preserve hashes of retained exports; recommend removing key-prefix logging in a later reviewed patch |

The imported code is a source snapshot, not a verified reconstruction of every historical run's producer environment. Some historical results depend on omitted local code. This distinction must appear in `result/README.md` and provenance metadata.

## 2. Results are a versioned experimental record

Avoid one folder per historical attempt containing repeated copies of everything. Separate the scientific run identity, infrastructure attempts, immutable artifacts, and derived comparisons.

```text
result/
  README.md
  registry/
    models.jsonl
    runs.jsonl
    jobs.jsonl
    artifacts.jsonl
    lineage_gaps.jsonl
  protocols/
    inference/<protocol_id>.json
    judge/<protocol_id>.json
    scoring/<protocol_id>.json
    tasks/<snapshot_id>/...
  runs/<run_id>/
    manifest.json
    inference/<inference_id>/manifest.json
    judging/<judge_pass_id>/manifest.json
  artifacts/sha256/<prefix>/<hash>.<format>
  releases/<release_id>/
    manifest.json
    tables/
    figures/
    diagnostics/
    REPORT.md
  supplemental/qwen3_openrouter_bad_medical_20260909/
  migration/
    source_manifest.jsonl
    excluded_files.jsonl
    verification.json
```

These names and schemas are proposed, not implemented. Human-readable flat CSV indexes can be generated later as views of the registries; they must not become independently maintained sources of truth. No model weights are copied into this code repository.

### Stable identities and joins

| Entity | Minimum fields |
|---|---|
| Model | Internal model ID, family, original vs fine-tuned role, HF repo ID and URL, resolved revision if known, tokenizer identity, parent model, revision evidence status |
| Run | Stable run ID, task, model ID, method, training seed or null, training dataset identity/revision if known, intended protocol, historical cohort, status |
| Job attempt | Internal attempt ID, provider, external job ID, stage (training/inference/judging), run/model links, parent attempt, status, retry/supersession links, input/output artifact references |
| Inference batch | Inference ID, checkpoint revision, eval snapshot/hash, requested and effective sampling configuration, inference seed if recorded, environment, prompt/sample counts, attempt links |
| Judge pass | Judge pass ID, inference ID, exact judge model, provider routing, parameters, rubric/input hashes, response parser version, attempt and cost records |
| Artifact | SHA256, format/compression, byte size, local path, original source/file ID, verified retrieval URL where available, original hash if transformed, redaction/transformation notes |
| Release | Explicit included run/inference/judge IDs, scoring protocol, source-artifact hashes, metric definitions, exclusions, coverage, verification status |

Training job IDs, completion job IDs, and HF repository IDs are different identifiers. Do not equate them or infer a missing training job from a completion job's name. A retry produces another job attempt, not an additional training seed or independent scientific result. A new judgment of the same completion produces a new judge pass, not another inference sample.

Existing completion IDs such as `eval_0000_sample_0000` repeat across jobs. Use `(inference_id, completion_id)` as the join key, with `eval_id`, task snapshot, and sample index recorded. Raw scores additionally require judge-pass and score-name keys.

Model repository names alone are not immutable checkpoint identifiers. Store exact HF revisions where evidenced; never label today's remote revision as the historical revision without supporting records. Leave unknown values null with a reason.

### Observed provenance coverage

The central 630-run fetch manifest has completion-job IDs and artifact model IDs for all 630 entries, but explicit `training_job_id` and `model_repository` fields are populated for only 37 entries. This does not prove the remaining training IDs are lost: local training ledgers, batch manifests, and saved job configurations may contain them.

Migration should reconcile those records, record evidence for each link, and publish an unresolved-link list. Artifacts may identify a model even where the central model-repository column is blank. Existing corrections, such as recovered mislabeled matrix cells, remain visible in lineage; do not restore superseded mappings merely because they appeared in an old spreadsheet.

### Preserve raw inference and judging without inventing missing data

- Preserve exact original inference responses, completions, provider errors, and job outputs when available, plus source-file IDs and hashes.
- Preserve exact original judge rows and backfilled coherence records separately. A backfill is a judge pass, not a mutation of the original output.
- Preserve the actual text given to the judge and any transformation (e.g. reasoning reconstruction or truncation).
- Some older results may retain completion text only inside `eval_results.csv`. If extracted, label it `reconstructed_from_eval_results`, retain the original CSV hash, and check duplicate text consistency. Such an extraction is not the original inference response or evidence of missing token/finish-reason metadata.
- Represent missing scores, API failures, provider filtering, refusals, and rejected parses distinctly; none is automatically zero.
- Keep filtered and unfiltered views derived from the same raw scores. Never delete failed-coherence completions from the raw artifact collection.
- Deduplicate identical bytes by hash. Do not discard distinct attempts or differently scored files because their names match.
- Inventory all available main-experiment and supplemental inference/judge files, including original downloads, later corrections, failed/retried attempts, and superseded judgments. Keep their inclusion/exclusion from the headline analysis explicit; an analysis exclusion is not an archival deletion.
- Audit remote file IDs against the local inventory. Retrieve missing original artifacts where accessible before finalizing the archive. If an artifact was never saved or is no longer retrievable, mark it unavailable with evidence. Rerunning the model does not recover the original output.

### Repository-contained storage requirement

Keep registries, protocols, summary tables, reports, and figures in ordinary Git. Keep the actual raw compressed artifacts under `result/artifacts/`, preferably tracked by Git LFS if the organization approves its storage/access arrangement. Git LFS is installed on this machine; remote capacity/access have not been verified. Verify compression round trips and retain both stored-byte hashes and original-content hashes.

Do not upload the entire directory blindly: inventory, deduplicate exact copies, compress losslessly, and scan it first. Do not replace required raw files with external download URLs or create a separate result repository under this plan. If storage constraints prevent repository-contained delivery, report the blocker rather than silently weakening the requirement. If LFS is used, publish the objects as well as the pointer files and verify that another authorized checkout can fetch every object. A pointer-only checkout or GitHub source ZIP is not evidence that all raw data has been delivered.

Scan the curated payload before upload: exclude tokens, `.env` files, authentication headers, credential-bearing configs, signed URLs, machine-specific symlinks, and private account metadata not needed for provenance. Record redactions; never silently rewrite claimed byte-identical raw artifacts. Preserve research-output access controls and review before making the repository public.

### Offline reproduction contract

Two different claims must be separated:

1. **Analysis reproduction (required):** regenerate normalized completion scores, all four scoring variants, per-run and five-seed summaries, filtering/coverage diagnostics, bootstrap intervals, Pareto results, and report tables/figures from the archived evidence, without generating or judging any new text.
2. **Experiment rerun (documented, not guaranteed identical):** generate new responses or new judge scores using pinned checkpoints, inputs, configurations, and environments. This needs compute/model access and possibly API credits. Stochastic generation, hardware differences, and changing judge services can prevent exact output reproduction even with recorded seeds.

The portable offline runner must load raw primary judgments and saved coherence backfills, reproduce the documented historical parser corrections and missingness handling, and retain the exact question/rubric snapshots. Merely copying the existing aggregate CSVs into new outputs is not reproduction. Saved normalized scores are cross-checks and convenient analysis inputs, not the only evidence.

Package explicit dependencies and a reproducible environment specification, deterministic analysis/bootstrap seeds, score and parser versions, a schema/data dictionary, and a short one-command offline verification/reproduction entry point. The command name will be finalized during implementation; none is claimed to exist yet. Avoid absolute paths, runtime-specific symlinks, or dependence on the old repository. Generating charts must work headlessly. The saved workbook can remain a convenience export; spreadsheet software is not required to reproduce the main numerical results.

Run a fresh-checkout acceptance test after downloading all repository objects: disable network access, remove credentials, and regenerate into a temporary output directory without overwriting the archived release. Compare raw hashes and integer counts exactly, and compare full-precision numerical tables using documented tight tolerances. Verify bootstrap reproducibility and frontier/tie conventions. PNG/PDF/workbook bytes can differ because of rendering metadata; check their underlying values and expected content rather than claiming universal byte identity.

There is an explicit scope tension: the existing offline analysis scripts are untracked and the user has asked to omit them. Therefore this document records a clean, reviewed replacement analysis layer as required new work; it does not import those scripts or their fixes. Until that layer (or an explicitly approved exception) passes the acceptance test, the archive can be called a data snapshot but the overall reproducible migration is incomplete.

## 3. Vanilla references: scientific strategy

### Define vanilla unambiguously

Vanilla means the original instruction-tuned parent checkpoint, with no benchmark-specific fine-tuning and no inoculation prompt. It does NOT mean a raw pretrained completion model. Resolve the exact parent repositories/revisions from training evidence for Llama-3.1-8B-Instruct, Olmo-3-7B-Instruct, and Qwen3-8B.

Use `method=vanilla`, `training_seed=null`, and a separate inference seed/repeat field. Rename the historical `baseline` method to `sft` only in the new registry/view, preserving the original label in provenance. This prevents confusing standard SFT with the untrained reference.

### Scope and execution order

1. Complete the registry and protocol audit, without paid calls.
2. Pilot Qwen3 on School of Reward Hacks with the matched GPU inference stack. It directly overlaps the IP paper's reported EM experiments and complements the completed medical API reference.
3. Re-evaluate Qwen3 bad medical advice through that same matched stack. Keep the prior OpenRouter run as an approximate supplemental reference, not a replacement or duplicate primary point.
4. After coverage/protocol checks pass, evaluate all 3 original models across the 7 main tasks: 21 model-task references, not 105 pseudo-seeded references.

The inspected seven evaluation snapshots contain 388 prompt rows total. At 10 completions per prompt and three models, the full grid is 11,640 planned completions before validated reuse. Judge call count differs because some primary scores are regex-derived; coherence is required for every usable output. Final scope must use the pinned snapshots rather than these advisory counts if versions differ. No inference or judging costs are authorized by this document.

Reuse an output only when checkpoint identity, prompt/messages, and the full inference protocol match; reuse a judge response only for the same exact rendered judge input and judge protocol. Shared UG prompts across tasks may permit reuse, but cross-task dependence must then be recorded and reflected in uncertainty. Start with independent model-task batches for simpler auditing; deduplication is an explicit optimization, not an assumption.

### Match the inference stack, not just temperature

Prefer the same Transformers/OpenWeights GPU path as the trained checkpoints, with the same relevant runtime versions, dtype, tokenizer/chat-template behavior, prompt snapshot, sample count, temperature, and combined generated-token cap. Do not assume OpenRouter's output limit or quantization is equivalent.

The committed worker explicitly sets `temperature`, `do_sample`, and `max_new_tokens`. It inherits top-p, top-k, and other generation defaults from the loaded model. Original and fine-tuned repositories can have different generation configs, so even using the same worker does not guarantee matching sampling. A YAML field not consumed by the worker does not fix this.

Before execution, resolve and record effective top-p/top-k, repetition/other penalties, EOS/stop behavior, thinking mode, system messages, model/tokenizer revision, and randomness. Historical unknowns remain unknown. Where exact matching is impossible, label the baseline approximate and consider a small matched SFT/IP validation cohort; do not quietly rerun the whole experiment.

The GPU cap should apply to reasoning plus final answer as in the historical worker. Preserve the historical judge-input convention (including reasoning where present) for the main comparison. Final-answer-only scoring is a separately labeled sensitivity analysis, not a silent change.

### Judging and four scoring versions

Use the same benchmark judge/rubric protocol as the corresponding trained comparisons: the saved recent comparisons use DeepSeek-v4-flash. Pin the exact model identifier, sampling parameters, provider policy, prompt text, parser behavior, and coherence coverage. Confirm endpoint availability and cost immediately before any paid run; do not assume a model name ensures identical responses over time/providers.

The committed pipeline lacks the local coherence fallback. This is an execution gate: first verify every usable completion can receive coherence using approved task definitions or a separately reviewed minimal addition. Do not publish a filtered vanilla comparison whose missing coherence is treated as passing or silently dropped.

Retain all four existing analytical definitions:

- Current/raw: continuous alignment severity (`1 - alignment/100`), completion-weighted.
- Current/filtered: same, coherence at least 50.
- Paper-style/raw: alignment below 50 as an event, prompt-balanced.
- Paper-style/filtered: same, coherence at least 30.

Task-specific capability and factual scoring rules remain intact. Paper-style here is the existing benchmark adaptation, not exact reproduction of the paper's judge logprob estimator. Publish denominators, missingness, and retention by axis alongside every score.

### Comparisons and uncertainty

Keep the main 630 trained runs (6 configurations x 7 tasks x 3 models x 5 seeds) intact. Add 21 vanilla references as another model role, not another five-seed training method. Report each trained method's acquisition and unwanted-generalization changes relative to the shared vanilla reference within the same task/model cell.

Show absolute scores and Pareto frontiers as well as vanilla-relative deltas. Reduced unwanted generalization accompanied by near-zero capability acquisition is not automatically selective learning. Prefer acquisition-constrained comparisons; avoid ratios when the vanilla-relative capability denominator is small.

Five trained seeds describe training variability; ten completions per prompt describe sampling. They are not interchangeable. Use prompt-clustered sampling uncertainty for vanilla, retain training-seed variability for trained methods, and propagate the shared reference consistently across method comparisons. If bootstrapping cross-task results with reused prompts/completions, keep their shared cluster identity. No confidence claim comes from copying a vanilla point into five seed rows.

Five independently seeded vanilla inference batches would be an optional sampling-stability study, not five trained checkpoints. They are not necessary for the first complete reference grid.

## 4. Migration acceptance and remaining decisions

Before considering the migration complete:

1. Every imported script is on the committed allowlist and matches its source blob; no excluded local fix slipped in through a result folder.
2. All 630 selected trained results remain represented, with seven tasks, three models, six methods, and five seeds; the supplemental API reference is separate.
3. Every reported metric can be traced to a judge pass, inference artifact or honestly labeled reconstruction, model identity, and available job records. Unresolved historical links are enumerated, not guessed.
4. All available experiment raw inference and paid judge outputs are present in the repository-backed result collection, including backfills and distinct historical attempts; references alone do not satisfy this requirement. Unavailable originals are explicitly listed.
5. The four existing score tables remain numerically unchanged by migration, and the fresh-checkout offline reproduction test passes from raw evidence. Missing analysis code is a completion blocker, not a disclaimer that substitutes for reproducibility.
6. Report links, hashes, compression, secret scan, data schemas, large-file availability, and portability checks pass. No unapproved external upload or source deletion occurs.

Decisions still to discuss: repository-backed large-file storage (LFS under `result/` is the proposed default); approval of the 21-reference scope and pilot order; and the implementation plan for the required portable analysis runner and narrow execution blockers after the byte-identical code import. New analysis implementation is distinct from copying untracked source scripts.
