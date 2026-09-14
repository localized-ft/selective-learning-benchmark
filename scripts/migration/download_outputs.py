"""Recover evaluation output checkpoints from paginated OpenWeights events.

Read-only remotely; append-only content storage locally. Does not run inference,
judge, training, or change which result CSV is selected for analysis. Worker
configs, credentials, weight files, and console logs are deliberately not fetched.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import time

from package import RESULT, json_file, jsonl_file, sanitize, sha
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from archive import Archive, rows

FILE_ID = re.compile(r"(?:[a-z_]+:)?file-[a-zA-Z0-9_-]+\Z")
KINDS = {"completions_saved": "completions.jsonl",
         "completions_loaded": "completions.jsonl",
         "judge_scores_saved": "judge_scores.jsonl", "results_csv": "eval_results.csv"}


def paginated(query):
    output = []
    offset = 0
    while True:
        page = query.range(offset, offset + 499).execute().data
        output.extend(page)
        if len(page) < 500:
            return output
        offset += len(page)


def retry(function):
    for attempt in range(3):
        try:
            return function()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def classify(data):
    decoded = data.decode("utf-8-sig")
    if decoded.lstrip().startswith("{"):
        records = [json.loads(line) for line in decoded.splitlines() if line.strip()]
        if records and all(isinstance(r, dict) and "completion_id" in r for r in records):
            if all("completion" in r for r in records):
                return "completions.jsonl", len(records)
            if all("scores" in r for r in records):
                return "judge_scores.jsonl", len(records)
        raise ValueError("unrecognized_jsonl_schema")
    reader = csv.DictReader(io.StringIO(decoded))
    if {"completion_id", "score_name", "score"}.issubset(reader.fieldnames or []):
        return "eval_results.csv", sum(1 for _ in reader)
    raise ValueError("unrecognized_output_schema")


def refresh_inventory():
    """Reconcile historical gap labels after newly recovered files are verified."""
    archive = Archive()
    runs = rows(RESULT / "registry/runs.jsonl")
    main_runs = [r for r in runs if r.get("selected_release") == "five_seed_20260908"]
    recovered_ids = {r["run_id"] for r in main_runs if r.get("inference_artifacts")}
    gaps = rows(RESULT / "registry/lineage_gaps.jsonl")
    resolved = [g for g in gaps if g["field"] == "original_inference_artifact" and g["run_id"] in recovered_ids]
    resolution_path = RESULT / "migration/resolved_lineage_gaps.jsonl"
    previous = rows(resolution_path) if resolution_path.exists() else []
    known = {(r["run_id"], r["field"]) for r in previous}
    previous.extend({**r, "previous_status": r["status"], "status": "resolved",
                     "resolution": "verified_original_recovered_from_openweights_events"}
                    for r in resolved if (r["run_id"], r["field"]) not in known)
    jsonl_file(resolution_path, previous)
    gaps = [g for g in gaps if g not in resolved]
    jsonl_file(RESULT / "registry/lineage_gaps.jsonl", gaps)
    for run in main_runs:
        if run.get("inference_artifacts") and run.get("extracted_completions"):
            run["extracted_completions"]["status"] = "historical_fallback_superseded_by_verified_original"
            json_file(RESULT / "runs" / run["run_id"] / "manifest.json", run)
            path = RESULT / "runs" / run["run_id"] / "inference" / run["inference_id"] / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["extracted_completions"] = run["extracted_completions"]
            json_file(path, manifest)
    jsonl_file(RESULT / "registry/runs.jsonl", runs)
    json_file(RESULT / "migration/registry_summary.json", {
        "trained_runs": len(main_runs), "models": len(rows(RESULT / "registry/models.jsonl")),
        "known_job_records": len(rows(RESULT / "registry/jobs.jsonl")),
        "training_job_links": sum(bool(r["training_job_id"]) for r in main_runs),
        "original_inference_available": len(recovered_ids), "lineage_gaps": len(gaps)})
    json_file(RESULT / "migration/delivery_inventory.json", {
        "main_runs": len(main_runs), "supplemental_runs": len(runs) - len(main_runs),
        "source_aliases": len(archive.sources), "raw_objects": len(archive.objects),
        "stored_raw_bytes": sum(r["stored_bytes"] for r in archive.objects.values()),
        "local_source_files": sum(p.startswith("sunday/") for p in archive.sources),
        "remote_output_files": sum(p.startswith("openweights/") for p in archive.sources),
        "remote_inference_files": sum(p.startswith("openweights/") and p.endswith("/completions.jsonl") for p in archive.sources),
        "labeled_legacy_extractions": sum(p.startswith("derived/") for p in archive.sources),
        "main_runs_with_original_inference": len(recovered_ids),
        "github_push_performed": False, "new_training_or_inference_jobs": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    from dotenv import dotenv_values
    from openweights import OpenWeights
    config = dotenv_values(args.env_file)
    if config.get("OPENWEIGHTS_API_KEY"):
        os.environ["OPENWEIGHTS_API_KEY"] = config["OPENWEIGHTS_API_KEY"]
    secrets = [v.encode() for k, v in config.items() if v and len(v) >= 12
               and any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    ow = OpenWeights()
    archive = Archive()
    runs = rows(RESULT / "registry/runs.jsonl")
    main_runs = [r for r in runs if r.get("selected_release") == "five_seed_20260908"]
    job_ids = sorted({r[k] for r in main_runs for k in ("completion_job_id", "judge_job_id") if r.get(k)})
    audit_path = RESULT / "migration/remote_output_events.jsonl"
    prior = {r["job_id"]: r for r in rows(audit_path)} if audit_path.exists() else {}

    def inspect(job_id):
        if prior.get(job_id, {}).get("status") == "audited":
            return prior[job_id]
        try:
            attempts = retry(lambda: paginated(ow._supabase.table("runs")
                .select("id,status,created_at").eq("job_id", job_id).order("id")))
            references = []
            event_count = 0
            for attempt in attempts:
                events = retry(lambda: paginated(ow._supabase.table("events")
                    .select("id,run_id,data,created_at").eq("run_id", attempt["id"]).order("id")))
                event_count += len(events)
                for event in events:
                    data = event.get("data") or {}
                    kind = data.get("type")
                    file_id = data.get("file_id")
                    if kind in KINDS and isinstance(file_id, str) and FILE_ID.fullmatch(file_id):
                        references.append({"file_id": file_id, "event_type": kind,
                            "event_id": event["id"], "remote_run_id": event["run_id"],
                            "remote_run_status": attempt["status"], "created_at": event["created_at"]})
            return {"job_id": job_id, "status": "audited", "attempts": attempts,
                    "event_count": event_count, "output_references": references}
        except Exception as exc:
            return {"job_id": job_id, "status": "unavailable", "error_type": type(exc).__name__}

    audit = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, record in enumerate(pool.map(inspect, job_ids), 1):
            audit.append(record)
            jsonl_file(audit_path, audit)
            if i % 50 == 0:
                print(f"Audited {i}/{len(job_ids)} evaluation jobs", flush=True)
    refs = {}
    for record in audit:
        for ref in record.get("output_references", []):
            refs.setdefault(ref["file_id"], []).append({"job_id": record["job_id"], **ref})
    print(json.dumps({"jobs": len(audit), "files_discovered": len(refs),
                      "event_types": dict(Counter(r["event_type"] for rr in refs.values() for r in rr))}), flush=True)
    if args.discover_only:
        return
    checkpoint = RESULT / "migration/remote_output_downloads.jsonl"
    previous = {r["file_id"]: r for r in rows(checkpoint)} if checkpoint.exists() else {}
    existing_by_id = {}
    for source in archive.sources.values():
        parts = source["source_path"].split("/")
        if len(parts) == 4 and parts[:2] == ["openweights", "files"]:
            existing_by_id[parts[2]] = source

    def download(file_id):
        if previous.get(file_id, {}).get("status") == "downloaded":
            return previous[file_id]
        try:
            if file_id in existing_by_id:
                source = existing_by_id[file_id]
                filename, count = classify(archive.content(source["artifact_id"]))
                return {**source, "file_id": file_id, "status": "downloaded", "rows": count,
                        "filename": filename, "reused_existing_file": True}
            original = retry(lambda: ow.files.content(file_id))
            if isinstance(original, str):
                original = original.encode()
            filename, count = classify(original)
            data, redactions = sanitize(original, Path(filename).suffix, secrets)
            digest = sha(data)
            path = RESULT / "artifacts/sha256" / digest[:2] / (digest + ".gz")
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation: byte-identical duplicates cannot overwrite an
            # existing gzip object's header/compression or invalidate its hash.
            try:
                with path.open("xb") as out:
                    out.write(gzip.compress(data, compresslevel=6, mtime=0))
            except FileExistsError:
                pass
            stored = path.read_bytes()
            if gzip.decompress(stored) != data:
                raise ValueError("existing_object_integrity_error")
            return {"file_id": file_id, "status": "downloaded", "rows": count, "filename": filename,
                "reused_existing_file": False,
                "source_path": f"openweights/files/{file_id}/{filename}",
                "artifact_id": digest, "path": str(path.relative_to(RESULT)), "source_sha256": sha(original),
                "content_sha256": digest, "stored_sha256": sha(stored), "original_bytes": len(original),
                "content_bytes": len(data), "stored_bytes": len(stored), "format": Path(filename).suffix[1:],
                "compression": "gzip", "redactions": redactions, "byte_identical_to_source": redactions == 0}
        except Exception as exc:
            return {"file_id": file_id, "status": "unavailable", "error_type": type(exc).__name__}

    downloads = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, record in enumerate(pool.map(download, sorted(refs)), 1):
            downloads.append(record)
            jsonl_file(checkpoint, downloads)
            if i % 50 == 0:
                print(f"Archived {i}/{len(refs)} output file IDs", flush=True)
    sources, objects = archive.sources, archive.objects
    old_objects = set(objects)
    by_file = {}
    for record in downloads:
        if record["status"] != "downloaded":
            continue
        by_file[record["file_id"]] = record
        source = {k: v for k, v in record.items() if k not in
                  {"file_id", "status", "rows", "filename", "reused_existing_file"}}
        sources[source["source_path"]] = source
        digest = source["artifact_id"]
        if digest not in objects:
            objects[digest] = {k: source[k] for k in ("artifact_id", "path", "content_sha256",
                "stored_sha256", "content_bytes", "stored_bytes", "compression")}
            objects[digest]["sources"] = []
        evidence = {k: source[k] for k in ("source_path", "source_sha256", "format", "redactions", "byte_identical_to_source")}
        if evidence not in objects[digest]["sources"]:
            objects[digest]["sources"].append(evidence)
    jsonl_file(RESULT / "migration/source_manifest.jsonl", sorted(sources.values(), key=lambda r: r["source_path"]))
    jsonl_file(RESULT / "registry/artifacts.jsonl", sorted(objects.values(), key=lambda r: r["artifact_id"]))
    by_job = {r["job_id"]: r for r in audit}
    recovered = 0
    for run in main_runs:
        linked = []
        for job_id in sorted({run.get("completion_job_id"), run.get("judge_job_id")} - {None, ""}):
            for ref in by_job[job_id].get("output_references", []):
                record = by_file.get(ref["file_id"])
                if record:
                    linked.append({"job_id": job_id, **ref, "artifact_id": record["artifact_id"],
                                   "source_path": record["source_path"], "filename": record["filename"]})
        run["remote_output_artifacts"] = linked
        # Only promote a recovered completion file if every identity and text
        # agrees with the frozen selected judge CSV (retry attempts may differ).
        if not run["inference_artifacts"]:
            expected = {r["completion_id"]: (r["eval_id"], r["completion"]) for r in
                csv.DictReader(io.StringIO(archive.content(run["primary_result_artifact_id"]).decode()))}
            for link in linked:
                if link["filename"] != "completions.jsonl":
                    continue
                data = gzip.decompress((RESULT / objects[link["artifact_id"]]["path"]).read_bytes())
                records = [json.loads(line) for line in data.splitlines() if line.strip()]
                observed = {r["completion_id"]: (r.get("eval_id"), r["completion"]) for r in records}
                if len(records) == len(observed) and observed == expected:
                    run["inference_artifacts"] = [{k: link[k] for k in ("artifact_id", "source_path")}]
                    run["original_inference_status"] = "available"
                    run["original_inference_recovery"] = "paginated_openweights_event_history"
                    recovered += 1
                    break
        json_file(RESULT / "runs" / run["run_id"] / "manifest.json", run)
        infer_path = RESULT / "runs" / run["run_id"] / "inference" / run["inference_id"] / "manifest.json"
        infer = json.loads(infer_path.read_text())
        infer.update(artifacts=run["inference_artifacts"], status=run["original_inference_status"])
        infer["remote_output_artifacts"] = [r for r in linked if r["filename"] == "completions.jsonl"]
        json_file(infer_path, infer)
        judge_path = RESULT / "runs" / run["run_id"] / "judging" / run["judge_pass_id"] / "manifest.json"
        judge = json.loads(judge_path.read_text())
        judge["remote_output_artifacts"] = [r for r in linked if r["filename"] != "completions.jsonl"]
        json_file(judge_path, judge)
    jsonl_file(RESULT / "registry/runs.jsonl", runs)
    for job in (jobs := rows(RESULT / "registry/jobs.jsonl")):
        if job["job_id"] in by_job:
            job["remote_output_audit"] = by_job[job["job_id"]]
    jsonl_file(RESULT / "registry/jobs.jsonl", jobs)
    summary = {"evaluation_jobs_requested": len(job_ids), "jobs_audited": sum(r["status"] == "audited" for r in audit),
        "file_ids_discovered": len(refs), "files_available": len(by_file), "files_unavailable": len(refs) - len(by_file),
        "files_reused_from_archive": sum(r.get("reused_existing_file", False) for r in downloads),
        "available_by_filename": dict(Counter(r["filename"] for r in by_file.values())),
        "unique_objects_added_this_execution": len(set(objects) - old_objects),
        "legacy_runs_recovered_this_execution": recovered,
        "main_runs_with_original_inference": sum(bool(r["inference_artifacts"]) for r in main_runs),
        "source_aliases": len(sources), "unique_objects": len(objects),
        "stored_bytes": sum(o["stored_bytes"] for o in objects.values()),
        "scope": "Selected main-experiment inference/judge jobs, all their recorded attempts; output checkpoints only",
        "excluded": ["worker configs", "credentials", "model weights", "console logs", "training-only jobs"],
        "new_model_or_judge_calls": 0, "selected_result_artifacts_changed": 0}
    json_file(RESULT / "migration/remote_output_recovery.json", summary)
    refresh_inventory()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
