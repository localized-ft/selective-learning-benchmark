# Result data dictionary

All registries are UTF-8 JSONL: one JSON object per line. An absent historical
value is `null` or an explicitly documented source blank, not a guessed value.

| Entity | Identity and important fields |
|---|---|
| Run | `run_id`, `task_id`, `category`, `model_family`, `model_id`, `method`, `source_method`, `training_seed`, stage job IDs, primary result artifact, associated artifacts |
| Model | `model_id`, `hf_repository`, `hf_url`, `historical_revision`, `revision_status`, `parent_model`, `role`; model identity is scoped to its historical run when exact revisions are unavailable |
| Job | Provider `job_id`, stage(s), linked `run_ids`, evidence records; retry candidates remain distinct jobs |
| Artifact | `artifact_id` = SHA256 of uncompressed object content, local `path`, compressed `stored_sha256`, original formats and source aliases, byte sizes |
| Source alias | Original source-relative path, original SHA256, object reference, redaction/transformation status; aliases are provenance identifiers, not paths required on your computer |
| Inference | `inference_id`, job ID, original artifact list and availability; optional `extracted_completions` explicitly marks recovery from a judge CSV |
| Judge pass | `judge_pass_id`, inference ID, job ID where available, raw result object, shared coherence-backfill objects |
| Gap | Run/field, unresolved or ambiguous status, evidence/candidates, and any honest fallback |

## Raw formats

For a fixed-schema interface, see `result/standardized/README.md` (relative to the
repository root). It provides JSONL completion/judgment views with numeric/null
score fields, explicit score status, canonical axis labels, and raw-source links.
Original-format evidence below remains immutable.

Original-format files are retained, not silently normalized in place. Typically:

- `completions.jsonl`: `completion_id`, `eval_id` when available, `completion`.
- `eval_results.csv`: task/model/judge identity, axis, prompt/group/eval IDs,
  completion text, grading method, score name/value/label/source text.
- `judge_scores.jsonl` and local judge events: per-completion scores and responses,
  with schema and available metadata depending on producer version.
- Coherence checkpoints: rendered-prompt hash, status, numeric score, judge identity
  and available raw response/usage fields.
- API inference events: full saved provider response, reasoning/answer content,
  usage/finish reason where returned, and associated request identity.

`(inference_id, completion_id)` identifies a generated sample. A bare completion
ID is not globally unique. The judge-pass ID distinguishes rejudgments of the
same response. A retry is not another training seed.

`remote_output_artifacts` in run/stage manifests preserves OpenWeights file,
event, job, and remote-attempt IDs together with local artifact references.
These include historical retries and original pre-backfill judgments; inclusion
in this list does not select an artifact for the scientific comparison.
`primary_result_artifact_id` remains authoritative for the selected result.
The paginated event audit and download status are recorded in
`migration/remote_output_events.jsonl` and `migration/remote_output_downloads.jsonl`.

All missing primary/coherence values, parse failures, refusals, API errors, and
provider filtering remain distinguishable in the raw evidence. The original
Qwen3 API run has one provider-filtered request; it is not scored as a refusal
or retried to bypass filtering.

## Metric units and method names

Analytical CSVs store scores on 0–1; reports/figures show 0–100. Higher capability
means acquisition of the designated behavior, not generic quality or safety.
Lower unwanted generalization is preferred. Historical `baseline` means SFT;
the new registry uses `sft` and keeps `source_method=baseline`. `vanilla` denotes
the original instruct checkpoint and must have `training_seed=null`.

Every filtered metric must be accompanied by valid-primary, retained-completion,
and retained-prompt counts. Filtering changes the evaluated population and never
removes records from the archive.

## Integrity and access

`migration/code_manifest.jsonl` proves which committed scripts/configs/task files
were copied. `migration/source_manifest.jsonl` covers original raw files and
derived extractions. `migration/inference_integrity.jsonl` verifies available
standalone inference against raw judge-input text for every main run.

Use `python scripts/analysis/archive.py verify` to verify content, compressed
object, and imported-code hashes. This does not assert missing historical HF
revisions or training-job links have been recovered. Git LFS objects must be
present locally; a pointer file will fail the hash check.
