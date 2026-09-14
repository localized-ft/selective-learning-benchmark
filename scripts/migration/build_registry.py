"""Build experiment lineage from archived evidence; unknown links stay unknown."""
import csv
import io
import json
from pathlib import Path
import re
import sys

from package import RESULT, json_file, jsonl_file, sha
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from archive import Archive, PREFIX


def main():
    archive = Archive()
    manifest = json.loads(archive.analysis("sources/fetch_manifest.json"))
    training = []
    inference = []
    for path in archive.sources:
        if path.startswith("sunday/scripts/finetune/") and path.endswith(".csv"):
            for row in csv.DictReader(io.StringIO(archive.source(path).decode("utf-8-sig"))):
                if row.get("job_id"):
                    training.append({**row, "evidence_source": path})
        if path.endswith("inference_kld_ip_20260826/inference_manifest.csv"):
            inference = list(csv.DictReader(io.StringIO(archive.source(path).decode())))
    run_records, model_records, gaps = [], [], []
    jobs = {}

    def record_job(job_id, stage, run_id, evidence):
        if not job_id: return
        item = jobs.setdefault(job_id, {"job_id": job_id, "provider": "openweights",
                                      "stages": [], "run_ids": [], "evidence": []})
        if stage not in item["stages"]: item["stages"].append(stage)
        if run_id not in item["run_ids"]: item["run_ids"].append(run_id)
        if evidence not in item["evidence"]: item["evidence"].append(evidence)

    for job in manifest["jobs"]:
        method = "sft" if job["method"] == "baseline" else job["method"]
        run_id = f"{job['task_id']}__{job['model_family']}__{method}__seed{job['seed']}"
        repos = job.get("artifact_models", [])
        if len(repos) != 1: raise ValueError("Ambiguous artifact model: " + run_id)
        repo = repos[0]
        model_id = "model_" + sha((repo + "\n" + run_id).encode())[:20]
        candidates = [r for r in training if r.get("finetuned_model_id") == repo]
        inference_matches = [r for r in inference if r.get("model") == repo and str(r.get("seed")) == str(job["seed"])]
        training_ids = set()
        if job.get("training_job_id"): training_ids.add(job["training_job_id"])
        for record in inference_matches:
            if record.get("training_job_id"): training_ids.add(record["training_job_id"])
        # Same HF target may have several failed attempts; retain all candidates.
        for record in candidates:
            record_job(record["job_id"], "training", run_id,
                       {"source_path": record["evidence_source"], "status": record.get("status"),
                        "match": "exact_hf_repository_candidate", "hf_repository": repo})
        selected_training_id = next(iter(training_ids)) if len(training_ids) == 1 else None
        if not training_ids:
            successful = {r["job_id"] for r in candidates if r.get("status") == "completed"}
            if len(successful) == 1: selected_training_id = next(iter(successful))
        source = PREFIX + "five_seed_comparison/" + job["path"]
        raw = archive.sources[source]
        judge_id = job.get("judge_job_id") or None
        completion_job = job.get("completion_job_id") or None
        for stage, jid in (("training", selected_training_id), ("inference", completion_job), ("judging", judge_id)):
            record_job(jid, stage, run_id, {"source_path": PREFIX + "five_seed_comparison/sources/fetch_manifest.json",
                                           "match": "historical_registry_or_exact_model_ledger"})
        inference_sources = []
        file_id = job.get("expected_inference_file_id")
        remote = "openweights/files/" + str(file_id) + "/completions.jsonl"
        if file_id and remote in archive.sources: inference_sources.append(remote)
        suffix = "/".join([job["method"], "seed" + str(job["seed"]), job["task_id"], job["model_family"], "completions.jsonl"])
        inference_sources += [p for p in archive.sources if p.endswith(suffix) and "/inference_results/" in p]
        inference_sources = sorted(set(inference_sources))
        if not inference_sources:
            gaps.append({"run_id": run_id, "field": "original_inference_artifact", "status": "not_located",
                         "completion_job_id": completion_job,
                         "fallback": "completion_text_preserved_in_raw_judge_csv",
                         "fallback_artifact_id": raw["artifact_id"]})
        if not selected_training_id:
            gaps.append({"run_id": run_id, "field": "training_job_id", "status": "ambiguous" if training_ids else "unresolved",
                         "candidate_ids": sorted(training_ids | {r["job_id"] for r in candidates})})
        # Revisions are not inferred from today's HF branch.
        parent = sorted({r.get("model") for r in candidates if r.get("model")})
        model_records.append({"model_id": model_id, "model_family": job["model_family"], "role": "finetuned",
                              "hf_repository": repo, "hf_url": "https://huggingface.co/" + repo,
                              "historical_revision": None, "revision_status": "not_recorded_in_selected_evidence",
                              "parent_model": parent[0] if len(parent) == 1 else None,
                              "run_id": run_id})
        attached = []
        key_fragment = "/".join([job["method"], "seed" + str(job["seed"]), job["task_id"], job["model_family"]]) + "/"
        for p, record in archive.sources.items():
            if key_fragment in p or (completion_job and ("/" + completion_job + "/") in p):
                attached.append({"source_path": p, "artifact_id": record["artifact_id"]})
        row = {"run_id": run_id, "task_id": job["task_id"], "category": job["category"],
               "model_family": job["model_family"], "model_id": model_id,
               "method": method, "source_method": job["method"], "training_seed": int(job["seed"]),
               "training_job_id": selected_training_id, "completion_job_id": completion_job,
               "judge_job_id": judge_id, "judge_execution": "remote" if judge_id else "local_or_combined_historical",
               "inference_id": "inference_" + run_id, "judge_pass_id": "historical_" + run_id,
               "primary_result_artifact_id": raw["artifact_id"], "primary_result_source_path": source,
               "inference_artifacts": [{"source_path":p, "artifact_id":archive.sources[p]["artifact_id"]} for p in inference_sources],
               "original_inference_status": "available" if inference_sources else "not_located",
               "inference_protocol": "historical_not_fully_logged", "scoring_protocol": "four_variants_20260908",
               "associated_artifacts": attached, "selected_release": "five_seed_20260908"}
        run_records.append(row)
        json_file(RESULT / "runs" / run_id / "manifest.json", row)
        json_file(RESULT / "runs" / run_id / "inference" / row["inference_id"] / "manifest.json",
                  {"run_id":run_id, "inference_id":row["inference_id"], "job_id":completion_job,
                   "artifacts":row["inference_artifacts"], "status":row["original_inference_status"]})
        json_file(RESULT / "runs" / run_id / "judging" / row["judge_pass_id"] / "manifest.json",
                  {"run_id":run_id, "judge_pass_id":row["judge_pass_id"], "inference_id":row["inference_id"],
                   "job_id":judge_id, "artifact_id":raw["artifact_id"],
                   "coherence_backfills": [archive.sources[PREFIX + "five_seed_comparison/sources/" + n]["artifact_id"]
                       for n in ("prior_coherence_checkpoint.jsonl", "coherence_checkpoint.jsonl")]})

    jsonl_file(RESULT / "registry/runs.jsonl", run_records)
    jsonl_file(RESULT / "registry/models.jsonl", model_records)
    jsonl_file(RESULT / "registry/jobs.jsonl", sorted(jobs.values(), key=lambda r:r["job_id"]))
    jsonl_file(RESULT / "registry/lineage_gaps.jsonl", gaps)
    # Human-readable release documents/figures/tables remain at their old relative layout.
    release = RESULT / "releases/five_seed_20260908"
    materialized = []
    for source, obj in archive.sources.items():
        prefix = PREFIX + "five_seed_comparison/"
        if not source.startswith(prefix): continue
        rel = source[len(prefix):]
        if rel == "REPORT.md" or rel.startswith(("outputs/", "manuscript/")):
            dest = release / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.content(obj["artifact_id"]))
            materialized.append({"path":str(dest.relative_to(RESULT)), "artifact_id":obj["artifact_id"]})
        if rel.startswith("sources/task_definitions/") or rel == "sources/coherence_rubric.txt":
            dest = RESULT / "protocols/tasks/historical_20260908" / rel.removeprefix("sources/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.content(obj["artifact_id"]))
    supplement = RESULT / "supplemental/qwen3_openrouter_bad_medical_20260909"
    prefix = PREFIX + "batches/base_qwen3_bad_medical_20260909/"
    for source, obj in archive.sources.items():
        if source.startswith(prefix):
            name = source[len(prefix):]
            if name in {"REPORT.md", "comparison.json", "verification.json", "protocol.json", "client_cap_protocol.json", "judge_routing_amendment.json", "eval_summary.json"}:
                (supplement / name).parent.mkdir(parents=True, exist_ok=True)
                (supplement / name).write_bytes(archive.content(obj["artifact_id"]))
    json_file(release / "manifest.json", {"release_id":"five_seed_20260908", "run_ids":[r["run_id"] for r in run_records],
              "scoring_protocol":"four_variants_20260908", "archived_release_files":materialized,
              "producer_code":"local historical producer not imported; new offline reference implementation supplied separately"})
    summary = {"trained_runs":len(run_records), "models":len(model_records), "known_job_records":len(jobs),
               "training_job_links":sum(bool(r["training_job_id"]) for r in run_records),
               "original_inference_available":sum(bool(r["inference_artifacts"]) for r in run_records),
               "lineage_gaps":len(gaps)}
    json_file(RESULT / "migration/registry_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
