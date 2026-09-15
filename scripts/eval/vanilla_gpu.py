"""Prepare, explicitly submit, or collect the Olmo vanilla GPU pilot.

No credentials are uploaded. No shared server settings are changed. Preparing
does not submit a job; collection is read-only remotely. Repeated submission
never restarts an existing job. Use an explicit --env-file for OpenWeights.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "result/supplemental/vanilla_api_20260914/olmo3"
SNAPSHOT = ROOT / "result/protocols/tasks/historical_20260908/task_definitions"
MODEL = "allenai/Olmo-3-7B-Instruct"
REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
HARDWARE = ["1x A40", "1x RTX4090", "1x A6000"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def request_rows(snapshot=SNAPSHOT):
    rows, sources = [], []
    for path in sorted(snapshot.glob("*/eval.jsonl")):
        data = path.read_bytes()
        sources.append({"task_id": path.parent.name,
                        "path": str(path.relative_to(ROOT)), "sha256": sha(data)})
        for index, record in enumerate(json.loads(line) for line in data.splitlines() if line):
            for sample in range(10):
                cid = f"{path.parent.name}__eval_{index:04d}_sample_{sample:04d}"
                rows.append({"completion_id": cid, "task_id": path.parent.name,
                             "eval_id": record["id"], "axis": record["axis"],
                             "prompt_index": index, "sample_index": sample,
                             "messages": record["messages"],
                             "inference_seed": int(sha(("20260914:" + cid).encode())[:8], 16)})
    return rows, sources


def select_pilot(rows):
    """Two fixed-random prompts per task/axis, two samples each (56 outputs)."""
    rng = random.Random(20260914)
    groups = {}
    for r in rows:
        groups.setdefault((r["task_id"], r["axis"]), set()).add(r["prompt_index"])
    selected = {key: set(rng.sample(sorted(indices), 2)) for key, indices in sorted(groups.items())}
    return [r for r in rows if r["prompt_index"] in selected[r["task_id"], r["axis"]]
            and r["sample_index"] < 2]


def read_artifact(entry):
    stored = (ROOT / "result" / entry["path"]).read_bytes()
    assert sha(stored) == entry["stored_sha256"]
    raw = gzip.decompress(stored)
    assert sha(raw) == entry["content_sha256"]
    return raw


def store_artifact(raw):
    digest = sha(raw)
    dest = ROOT / "result/artifacts/sha256" / digest[:2] / (digest + ".gz")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with dest.open("xb") as f:
            f.write(gzip.compress(raw, mtime=0))
    assert gzip.decompress(dest.read_bytes()) == raw
    return {"artifact_id": digest, "content_sha256": digest,
            "stored_sha256": sha(dest.read_bytes()),
            "path": str(dest.relative_to(ROOT / "result")), "compression": "gzip",
            "content_bytes": len(raw), "stored_bytes": dest.stat().st_size}


def validate_reuse(requests, completions, details):
    expected = {r["completion_id"]: r for r in requests}
    ids = [c["completion_id"] for c in completions]
    assert len(ids) == len(set(ids))
    assert ids == [r["completion_id"] for r in details]
    for c, d in zip(completions, details):
        r = expected[c["completion_id"]]
        assert c["eval_id"] == d["eval_id"] == r["eval_id"]
        assert c["completion"] == d["completion"]
        assert isinstance(c["completion"], str)
        for key in ("task_id", "axis", "prompt_index", "sample_index", "inference_seed"):
            assert d[key] == r[key]
        assert d["output_tokens"] == len(d["output_token_ids"])
        assert 0 < d["output_tokens"] <= 2000
    return set(ids)


def prepare_vllm():
    phase = BATCH / "vllm"
    if (phase / "submission.json").exists():
        raise SystemExit("vLLM phase already submitted; preserve its inputs")
    previous = json.loads((BATCH / "full/status.json").read_text())
    assert previous["status"] == "canceled", "Confirm cancellation before replacing the full job"
    rows, sources = request_rows()
    pilot = select_pilot(rows)
    pilot_ids = {r["completion_id"] for r in pilot}
    ordered = pilot + [r for r in rows if r["completion_id"] not in pilot_ids]
    assert len(ordered) == len({r["completion_id"] for r in ordered}) == 3880
    artifacts = json.loads((BATCH / "pilot/artifacts.json").read_text())
    comp = next(a for a in reversed(artifacts) if a["filename"] == "completions.jsonl")
    detail = next(a for a in reversed(artifacts) if a["filename"] == "generation_details.jsonl"
                  and a["n"] == comp["n"] and a["remote_run_id"] == comp["remote_run_id"])
    env_entry = next(a for a in reversed(artifacts) if a["filename"] == "environment.json"
                     and a["remote_run_id"] == comp["remote_run_id"])
    environment = json.loads(read_artifact(env_entry))
    assert environment["resolved_model_revision"] == REVISION
    assert environment["config"]["snapshot_sources"] == sources
    refs = [json.loads(s) for s in read_artifact(detail).splitlines()]
    assert validate_reuse(pilot, [json.loads(s) for s in read_artifact(comp).splitlines()], refs) <= pilot_ids
    raw = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in ordered)
    phase.mkdir(parents=True, exist_ok=True)
    (phase / "requests.jsonl").write_bytes(raw)
    reference_raw = read_artifact(detail)
    (phase / "reference.jsonl").write_bytes(reference_raw)
    cfg = {"schema_version": "slb.vanilla_vllm_config.v1", "phase": "vllm", "backend": "vllm",
           "model": MODEL, "revision": REVISION, "method": "vanilla", "training_seed": None,
           "worker_filename": "vanilla_vllm_worker.py", "vllm_version": "0.19.1",
           "docker_image": "nielsrolf/ow-vllm@sha256:8e82046f38a42caadb27211820db93f784f0a7afa30ab5368c41d4dfbdfd95ca",
           "source_image_tag": "nielsrolf/ow-vllm:v0.11", "allowed_hardware": ["1x A40"],
           "request_count": len(ordered), "requests_sha256": sha(raw),
           "reference_sha256": sha(reference_raw), "reference_count": len(refs),
           "reference_artifact": detail, "snapshot_sources": sources,
           "chat_template_sha256": environment["chat_template_sha256"],
           "warmup_count": len(pilot), "checkpoint_every": 256,
           "max_worker_seconds": 24 * 3600, "minimum_pod_ttl_hours": 25,
           "max_projected_generation_seconds": 20 * 3600,
           "sampling": {"n": 1, "temperature": 1.0, "top_p": 0.95, "top_k": 50,
                        "max_tokens": 2000, "min_tokens": 0, "min_p": 0.0,
                        "presence_penalty": 0.0, "frequency_penalty": 0.0,
                        "repetition_penalty": 1.0, "stop_token_ids": [100265, 100257],
                        "ignore_eos": False, "skip_special_tokens": True},
           "top_k_evidence": "Transformers 5.5.0 resolves null top_k to global default 50 during generate",
           "old_outputs_policy": "Preserved as cross-backend diagnostics; not automatically pooled or discarded",
           "cohort_policy": "One vLLM sample per each of 3880 IDs; warmup included, no duplicate sample counting",
           "comparison_protocol": "matches pilot decoding parameters; not a claim of historical SFT/IP sampler equivalence"}
    save(phase / "config.json", cfg)
    print(json.dumps({"phase": "vllm", "requests": len(ordered), "warmup": len(pilot),
                      "reference_count": len(refs), "image": cfg["docker_image"]}))


def prepare(phase_name="pilot"):
    if phase_name == "vllm":
        return prepare_vllm()
    phase = BATCH / phase_name
    if (phase / "submission.json").exists():
        raise SystemExit("Phase already has a submission record; do not alter its inputs")
    rows, sources = request_rows()
    assert len(rows) == 3880
    selected = select_pilot(rows)
    reuse = None
    environment = None
    if phase_name == "full":
        pilot_path = BATCH / "pilot"
        status = json.loads((pilot_path / "status.json").read_text())
        assert status["status"] in {"completed", "canceled", "failed"}, "Stop/finish pilot and collect first"
        artifacts = json.loads((pilot_path / "artifacts.json").read_text())
        env_entry = next(a for a in reversed(artifacts) if a["filename"] == "environment.json")
        comp = next(a for a in reversed(artifacts) if a["filename"] == "completions.jsonl")
        detail = next(a for a in reversed(artifacts) if a["filename"] == "generation_details.jsonl"
                      and a["remote_run_id"] == comp["remote_run_id"] and a["n"] == comp["n"])
        assert env_entry["remote_run_id"] == comp["remote_run_id"]
        environment = json.loads(read_artifact(env_entry))
        assert environment["resolved_model_revision"] == REVISION
        assert environment["config"]["snapshot_sources"] == sources
        completions = [json.loads(s) for s in read_artifact(comp).splitlines()]
        details = [json.loads(s) for s in read_artifact(detail).splitlines()]
        reused_ids = validate_reuse(selected, completions, details)
        selected = [r for r in rows if r["completion_id"] not in reused_ids]
        reuse = {"pilot_job_id": status["job_id"], "count": len(reused_ids),
                 "completion_ids": sorted(reused_ids), "completions_artifact": comp,
                 "details_artifact": detail, "environment_artifact": env_entry,
                 "selection": "all structurally valid saved pilot outputs, without score filtering"}
        save(phase / "reuse.json", reuse)
    raw = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in selected)
    phase.mkdir(parents=True, exist_ok=True)
    (phase / "requests.jsonl").write_bytes(raw)
    cfg = {"schema_version": "slb.vanilla_gpu_config.v1", "phase": phase_name,
           "model": MODEL, "revision": REVISION, "method": "vanilla", "training_seed": None,
           "temperature": 1.0, "max_new_tokens": 2000, "do_sample": True,
           "other_generation_settings": "inherit pinned model generation_config and log effective values",
           "request_count": len(selected), "full_request_count": len(rows),
           "requests_sha256": sha(raw), "checkpoint_every": 8, "max_worker_seconds": 3600,
           "allowed_hardware": HARDWARE, "snapshot_sources": sources,
           "pilot_selection": "seed 20260914; two random prompts/task/axis; sample indices 0,1",
           "reuse_policy": "Reuse pilot IDs in the full cohort only after validation; never score-filter pilot selection"}
    if phase_name == "full":
        cfg.update(checkpoint_every=64, max_worker_seconds=48 * 3600,
                   minimum_pod_ttl_hours=49, allowed_hardware=["1x A40"],
                   reused_count=reuse["count"],
                   expected_environment={k: environment[k] for k in (
                       "torch", "transformers", "model_dtype", "chat_template_sha256",
                       "model_generation_config")},
                   comparison_protocol="native_olmo_defaults_with_temp1_max2000_not_historical_sampling_matched")
        cfg.pop("pilot_selection")
    save(phase / "config.json", cfg)
    print(json.dumps({"phase": phase_name, "status": "prepared", "requests": len(selected), "tasks": 7,
                      "reused": reuse["count"] if reuse else 0,
                      "sampling": {"temperature": 1, "max_new_tokens": 2000}}))


def client(env_file):
    from dotenv import dotenv_values
    from openweights import OpenWeights
    logging.disable(logging.CRITICAL)
    cfg = dotenv_values(env_file)
    return OpenWeights(auth_token=cfg["OPENWEIGHTS_API_KEY"])


def submit(env_file, phase_name="pilot"):
    phase = BATCH / phase_name
    cfg = json.loads((phase / "config.json").read_text())
    ow = client(env_file)
    record_path = phase / "submission.json"
    if record_path.exists():
        saved = json.loads(record_path.read_text())
        if saved.get("job_id"):
            job = ow.jobs.retrieve(saved["job_id"])
            print(json.dumps({"job_id": job.id, "status": job.status, "submitted_now": False}))
            return
        raise SystemExit("Uncertain previous submission: inspect planned_job_id before retrying")
    uploads = {}
    worker_filename = cfg.get("worker_filename", "vanilla_gpu_worker.py")
    mounted = {
        worker_filename: Path(__file__).with_name(worker_filename),
        "vanilla_requests.jsonl": phase / "requests.jsonl",
        "vanilla_config.json": phase / "config.json",
    }
    if cfg.get("backend") == "vllm":
        mounted["vanilla_reference.jsonl"] = phase / "reference.jsonl"
    for filename, path in mounted.items():
        data = path.read_bytes()
        if filename == "vanilla_requests.jsonl":
            assert sha(data) == cfg["requests_sha256"]
        buf = io.BytesIO(data)
        buf.name = filename
        result = ow.files.create(buf, purpose="custom_job_file")
        uploads[filename] = {"file_id": result["id"], "sha256": sha(data),
                             "artifact": store_artifact(data)}
    job_data = {"type": "custom", "model": cfg["model"],
                "docker_image": cfg.get("docker_image", "nielsrolf/ow-unsloth:v0.11"), "requires_vram_gb": 24,
                "allowed_hardware": cfg["allowed_hardware"],
                "script": f"timeout --signal=TERM --kill-after=30s {int(cfg['max_worker_seconds'])}s python {worker_filename}",
                "params": {"mounted_files": {k: v["file_id"] for k, v in uploads.items()}}}
    planned = ow.jobs.compute_id(job_data)
    submission = {"schema_version": "slb.vanilla_submission.v1", "planned_job_id": planned,
                  "job_id": None, "status": "submitting", "created_at_unix": time.time(),
                  "uploads": uploads, "job_data": job_data}
    save(record_path, submission)
    job = ow.jobs.get_or_create_or_reset(job_data)
    submission.update(job_id=job.id, status=job.status)
    save(record_path, submission)
    print(json.dumps({"job_id": job.id, "status": job.status, "submitted_now": True,
                      "requests": cfg["request_count"], "allowed_hardware": cfg["allowed_hardware"]}))


def pages(query):
    rows = []
    while True:
        page = query.range(len(rows), len(rows) + 499).execute().data
        rows.extend(page)
        if len(page) < 500:
            return rows


def collect(env_file, phase_name="pilot"):
    phase = BATCH / phase_name
    submission = json.loads((phase / "submission.json").read_text())
    ow = client(env_file)
    job = ow.jobs.retrieve(submission.get("job_id") or submission["planned_job_id"])
    attempts = pages(ow._supabase.table("runs").select("id,status,created_at")
                     .eq("job_id", job.id).order("created_at"))
    events = []
    for attempt in attempts:
        events.extend(pages(ow._supabase.table("events").select("id,run_id,data,created_at")
                            .eq("run_id", attempt["id"]).order("created_at")))
    # Export only this worker's explicitly credential-free structured events.
    safe_events = [e for e in events if isinstance(e.get("data"), dict)
                   and str(e["data"].get("type", "")).startswith("vanilla_")]
    path = phase / "artifacts.json"
    artifacts = json.loads(path.read_text()) if path.exists() else []
    known = {a["file_id"] for a in artifacts}
    for event in safe_events:
        d = event["data"]
        if not d.get("file_id") or d["file_id"] in known:
            continue
        raw = ow.files.content(d["file_id"])
        if isinstance(raw, str):
            raw = raw.encode()
        assert sha(raw) == d["content_sha256"]
        digest = sha(raw)
        dest = ROOT / "result/artifacts/sha256" / digest[:2] / (digest + ".gz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with dest.open("xb") as f:
                f.write(gzip.compress(raw, mtime=0))
        assert gzip.decompress(dest.read_bytes()) == raw
        artifacts.append({"artifact_id": digest, "content_sha256": digest,
                          "stored_sha256": sha(dest.read_bytes()),
                          "path": str(dest.relative_to(ROOT / "result")),
                          "file_id": d["file_id"], "filename": d["filename"],
                          "job_id": job.id, "remote_run_id": event["run_id"],
                          "event_id": event["id"], "n": d.get("n"),
                          "final": d.get("final", False), "compression": "gzip",
                          "content_bytes": len(raw), "stored_bytes": dest.stat().st_size})
        known.add(d["file_id"])
        save(path, artifacts)
    save(phase / "status.json", {"job_id": job.id, "status": job.status,
                                 "checked_at_unix": time.time(), "attempts": attempts,
                                 "events": safe_events, "artifact_count": len(artifacts)})
    progress = [{k: e["data"][k] for k in ("type", "n", "total", "complete", "gpu", "elapsed_seconds")
                 if k in e["data"]} for e in safe_events
                if e["data"]["type"] in {"vanilla_progress", "vanilla_finished", "vanilla_model_loaded"}]
    print(json.dumps({"job_id": job.id, "status": job.status, "attempts": len(attempts),
                      "latest_progress": progress[-1:] or None, "artifact_count": len(artifacts)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit", "collect"])
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--phase", choices=["pilot", "full", "vllm"], default="pilot")
    args = parser.parse_args()
    if args.action != "prepare" and not args.env_file:
        parser.error("--env-file required for submit/collect")
    if args.action == "prepare":
        prepare(args.phase)
    elif args.action == "submit":
        submit(args.env_file, args.phase)
    else:
        collect(args.env_file, args.phase)


if __name__ == "__main__":
    main()
