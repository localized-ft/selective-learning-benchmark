"""Recover original inference files by evidenced file IDs. Read-only remote API.

Writes content-addressed, sanitized artifacts locally. Never resubmits a job.
Requires openweights + python-dotenv only for this optional migration operation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import sys
import time

from package import RESULT, json_file, jsonl_file, sanitize, sha
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from archive import Archive, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    from dotenv import dotenv_values
    from openweights import OpenWeights
    config = dotenv_values(args.env_file)
    import os
    if config.get("OPENWEIGHTS_API_KEY"):
        os.environ["OPENWEIGHTS_API_KEY"] = config["OPENWEIGHTS_API_KEY"]
    secrets = [v.encode() for k,v in config.items() if v and len(v) >= 12 and any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    ow = OpenWeights()
    archive = Archive()
    manifest = json.loads(archive.analysis("sources/fetch_manifest.json"))
    ids = sorted({j.get("expected_inference_file_id") for j in manifest["jobs"] if j.get("expected_inference_file_id")})
    checkpoint = RESULT / "migration/remote_inference_downloads.jsonl"
    prior = {r["file_id"]:r for r in rows(checkpoint)} if checkpoint.exists() else {}

    def retrieve(file_id):
        if file_id in prior and prior[file_id]["status"] == "downloaded": return prior[file_id]
        for attempt in range(3):
            try:
                original = ow.files.content(file_id)
                if isinstance(original, str): original = original.encode()
                parsed = [json.loads(line) for line in original.splitlines() if line.strip()]
                assert parsed and all("completion_id" in r and "completion" in r for r in parsed), "not_completions_jsonl"
                data, redactions = sanitize(original, ".jsonl", secrets)
                digest = sha(data)
                stored = gzip.compress(data, compresslevel=6, mtime=0)
                path = RESULT / "artifacts/sha256" / digest[:2] / (digest + ".gz")
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with path.open("xb") as out:
                        out.write(stored)
                except FileExistsError:
                    stored = path.read_bytes()
                    if gzip.decompress(stored) != data:
                        raise ValueError("existing_object_integrity_error")
                return {"file_id": file_id, "status": "downloaded", "rows": len(parsed),
                        "source_path": "openweights/files/" + file_id + "/completions.jsonl",
                        "artifact_id": digest, "path": str(path.relative_to(RESULT)),
                        "source_sha256": sha(original), "content_sha256": digest,
                        "stored_sha256": sha(stored), "original_bytes": len(original),
                        "content_bytes": len(data), "stored_bytes": len(stored), "format": "jsonl",
                        "compression": "gzip", "redactions": redactions, "byte_identical_to_source": redactions == 0}
            except Exception as exc:
                if attempt == 2:
                    return {"file_id": file_id, "status": "unavailable", "error_type": type(exc).__name__}
                time.sleep(1 + attempt)

    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, value in enumerate(pool.map(retrieve, ids), 1):
            results.append(value)
            jsonl_file(checkpoint, results)
            if i % 20 == 0: print(f"Recovered {i}/{len(ids)} original inference files", flush=True)
    sources_by_path = dict(archive.sources)
    for record in results:
        if record["status"] != "downloaded": continue
        sources_by_path[record["source_path"]] = {k:v for k,v in record.items() if k not in {"status", "rows", "file_id"}}
    sources = list(sources_by_path.values())
    objects = dict(archive.objects)
    for record in sources:
        digest = record["artifact_id"]
        if digest not in objects:
            objects[digest] = {k:record[k] for k in ("artifact_id", "path", "content_sha256", "stored_sha256", "content_bytes", "stored_bytes", "compression")}
            objects[digest]["sources"] = []
        evidence = {k:record[k] for k in ("source_path", "source_sha256", "format", "redactions", "byte_identical_to_source")}
        if evidence not in objects[digest]["sources"]: objects[digest]["sources"].append(evidence)
    jsonl_file(RESULT / "migration/source_manifest.jsonl", sorted(sources, key=lambda r:r["source_path"]))
    jsonl_file(RESULT / "registry/artifacts.jsonl", sorted(objects.values(), key=lambda r:r["artifact_id"]))
    summary = {"file_ids_requested": len(ids), "downloaded": sum(r["status"] == "downloaded" for r in results),
               "unavailable": sum(r["status"] != "downloaded" for r in results), "new_inference_jobs": 0}
    json_file(RESULT / "migration/remote_recovery.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
