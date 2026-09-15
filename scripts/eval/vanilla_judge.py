"""Local, resumable judging of archived vanilla outputs. Never runs inference.

Keeps the migrated judge's rendering/parser functions unchanged. Adds explicit
missing-coherence coverage, provider pinning, raw attempt capture and provenance.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import fcntl
import gzip
import json
import math
from pathlib import Path
import time
import urllib.request
import uuid

from judge_utility import (judge_prompts_from_grading, parse_judge_response_score,
                           render_judge_prompt, score_with_regex_map)
from vanilla_api import append_event, archived, classify, load_key, read_events, reported_cost
from vanilla_gpu import ROOT, SNAPSHOT, read_artifact, save, sha, store_artifact, validate_reuse

BASE = ROOT / "result/supplemental/vanilla_api_20260914/judge"
PARENT = BASE.parent
MODEL = "deepseek/deepseek-v4-flash"
TERMINAL = {"ok", "label", "provider_filtered"}
ALIGNMENT = {"bad_medical_advice", "risky_financial_advice", "school_of_reward_hacks"}


def jsonl(raw):
    return [json.loads(s) for s in raw.splitlines() if s.strip()]


def write_gzip(path, rows):
    raw = b"".join((json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n").encode() for r in rows)
    path.write_bytes(gzip.compress(raw, mtime=0))


def local_source(path):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "stored_sha256": sha(raw)}


def final_olmo_artifact(status, artifacts, filename):
    finished = [e for e in status["events"] if e["data"].get("type") == "vanilla_finished"
                and e["data"].get("complete")]
    assert status["status"] == "completed" and finished
    run = finished[-1]["run_id"]
    event = next(e for e in reversed(status["events"]) if e["run_id"] == run
                 and e["data"].get("final") and e["data"].get("filename") == filename)
    entry = next(a for a in artifacts if a["file_id"] == event["data"]["file_id"])
    assert entry["content_sha256"] == event["data"]["content_sha256"]
    assert event["data"]["n"] == 3880
    return entry


def load_inputs():
    """Read exact completed artifacts, validating IDs before generating judge requests."""
    records, sources, excluded = [], [], []
    for family in ("llama31_8b", "qwen3_8b", "olmo3_7b"):
        path = PARENT / ("olmo3/vllm" if family == "olmo3_7b" else "api/" + family)
        cfg = json.loads((path / "config.json").read_text())
        request_bytes = (path / "requests.jsonl").read_bytes()
        assert sha(request_bytes) == cfg["requests_sha256"]
        reqs = jsonl(request_bytes)
        expected = {r["completion_id"]: r for r in reqs}
        assert len(expected) == len(reqs)
        sources.extend(local_source(path / name) for name in
                       ("config.json", "requests.jsonl", "status.json", "artifacts.json"))
        status = json.loads((path / "status.json").read_text())
        artifacts = json.loads((path / "artifacts.json").read_text())
        if family == "olmo3_7b":
            comp = final_olmo_artifact(status, artifacts, "completions.jsonl")
            detail = final_olmo_artifact(status, artifacts, "generation_details.jsonl")
            outputs = jsonl(read_artifact(comp))
            details = jsonl(read_artifact(detail))
            assert validate_reuse(reqs, outputs, details) == expected.keys()
            sources.extend({"path": "result/" + a["path"], "stored_sha256": a["stored_sha256"]}
                           for a in (comp, detail))
        else:
            assert not status["running"] and status["pending"] == 0
            for a in artifacts:
                file = path / Path(a["path"]).name
                raw = file.read_bytes()
                assert sha(raw) == a["stored_sha256"]
                assert sha(gzip.decompress(raw)) == a["content_sha256"]
                sources.append(local_source(file))
            outputs = read_events(path / "completions.jsonl.gz")
            details = read_events(path / "completion_status.jsonl.gz")
            assert len(details) == len(expected) and {d["completion_id"] for d in details} == expected.keys()
            assert {c["completion_id"] for c in outputs} == {
                d["completion_id"] for d in details if d["status"] == "ok"}
            excluded.extend({"model_family": family, **d} for d in details if d["status"] != "ok")
        assert len(outputs) == len({c["completion_id"] for c in outputs})
        for c in outputs:
            r = expected[c["completion_id"]]
            assert c["eval_id"] == r["eval_id"] and c["completion"].strip()
            records.append({**r, "completion": c["completion"], "model_family": family,
                            "inference_source": str(path.relative_to(ROOT))})
    assert len(records) == 10873 and len(excluded) == 7
    return records, sources, excluded


def build_requests(records, rubric):
    evaluations = {}
    for path in sorted(SNAPSHOT.glob("*/eval.jsonl")):
        rows = jsonl(path.read_bytes())
        evaluations[path.parent.name] = {r["id"]: r for r in rows}
        assert len(evaluations[path.parent.name]) == len(rows)
    requests = []
    for r in records:
        evaluation = evaluations[r["task_id"]][r["eval_id"]]
        assert evaluation["messages"] == r["messages"] and evaluation["axis"] == r["axis"]
        assert evaluation["grading"]["method"] == "llm_judge"
        grading = {**evaluation["grading"], "judge_prompts": dict(evaluation["grading"]["judge_prompts"])}
        added = not grading["judge_prompts"].get("coherence")
        if added:
            grading["judge_prompts"]["coherence"] = rubric
        specs = judge_prompts_from_grading(grading)
        assert len(specs) == 2 and sum(s.score_name == "coherence" for s in specs) == 1
        for spec in specs:
            prompt = render_judge_prompt(spec.prompt, evaluation["messages"][0]["content"], r["completion"])
            identity = [r["model_family"], r["task_id"], r["completion_id"], spec.score_name]
            request_id = sha(json.dumps(identity).encode())
            requests.append({"request_id": request_id, **{k: r[k] for k in
                ("model_family", "task_id", "completion_id", "eval_id", "axis", "inference_source")},
                "group_id": evaluation.get("group_id", ""), "question": evaluation["messages"][0]["content"],
                "completion": r["completion"], "completion_sha256": sha(r["completion"].encode()),
                "score_name": spec.score_name, "prompt": prompt, "prompt_sha256": sha(prompt.encode()),
                "answer_regex": spec.answer_regex, "score_map": spec.score_map,
                "coherence_source": "archived_fallback" if added and spec.score_name == "coherence" else "eval_snapshot"})
    assert len({r["request_id"] for r in requests}) == len(requests)
    return requests


def canaries(requests):
    first = {}
    for r in requests:
        first.setdefault((r["model_family"], r["task_id"], r["axis"], r["score_name"]), r["request_id"])
    return set(first.values())


def parse_score(request, raw):
    # Literal CODE/REFUSAL is a missing score, never selectively rejudged as a number.
    numeric = parse_judge_response_score(request["score_name"], raw)
    if numeric.score_label in {"CODE", "REFUSAL"}:
        return asdict(numeric), "label"
    if request["answer_regex"] and request["score_map"]:
        score = score_with_regex_map(request["score_name"], request["answer_regex"], request["score_map"], raw)
    else:
        score = numeric
    result = asdict(score)
    value = score.score
    maximum = 100 if request["score_name"] == "coherence" or request["task_id"] in ALIGNMENT else 1
    if value is not None and math.isfinite(value) and 0 <= value <= maximum:
        return result, "ok"
    result["score"] = None
    return result, "invalid_score"


def prepare():
    if (BASE / "config.json").exists():
        raise RuntimeError("Prepared already; preserve the frozen judge inputs")
    records, sources, excluded = load_inputs()
    rubric_path = SNAPSHOT.parent / "coherence_rubric.txt"
    requests = build_requests(records, rubric_path.read_text())
    sources.append(local_source(rubric_path))
    sources.extend(local_source(p) for p in sorted(SNAPSHOT.glob("*/eval.jsonl")))
    url = "https://openrouter.ai/api/v1/models/" + MODEL + "/endpoints"
    with urllib.request.urlopen(url, timeout=30) as response:
        catalog = json.load(response)
    endpoint = next(e for e in catalog["data"]["endpoints"] if e["tag"] == "alibaba/fp8")
    assert {"reasoning", "temperature", "top_p", "max_tokens"} <= set(endpoint["supported_parameters"])
    BASE.mkdir(parents=True, exist_ok=True)
    write_gzip(BASE / "requests.jsonl.gz", requests)
    save(BASE / "endpoint_catalog.json", catalog)
    save(BASE / "excluded_inference.json", excluded)
    producer = {name: store_artifact((ROOT / "scripts/eval" / name).read_bytes()) for name in
                ("vanilla_judge.py", "judge_utility.py", "eval_constants.py", "eval_data_model.py", "vanilla_api.py", "vanilla_gpu.py")}
    cfg = {"schema_version": "slb.vanilla_judge.v1", "body": {
        "model": MODEL, "temperature": 1.0, "top_p": 1.0, "max_tokens": 256, "stream": False,
        "reasoning": {"effort": "none"}, "provider": {"only": ["alibaba"], "allow_fallbacks": False,
                                                          "require_parameters": True}},
        "response_provider": "Alibaba", "endpoint_snapshot": endpoint,
        "pricing_per_token": endpoint["pricing"], "request_count": len(requests),
        "completion_count": len(records), "requests_stored_sha256": sha((BASE / "requests.jsonl.gz").read_bytes()),
        "sources": sources, "producer": producer, "concurrency": 32, "max_attempts_per_id": 5,
        "budget_usd": 10.0, "inflight_reservation_usd": 0.003, "canary_count": len(canaries(requests)),
        "reused_judgments": "result/supplemental/qwen3_openrouter_bad_medical_20260909",
        "reused_completion_count": 759, "reused_score_count": 1518,
        "coherence_policy": "Score both axes before filtering; preserve every raw score and failure",
        "parser_policy": "Migrated numeric/regex-map parser; finite/range validation; literal CODE/REFUSAL terminal missing",
        "retry_policy": "Retry invalid formatting and transient errors at most five attempts, never valid scores/labels/provider filters",
        "budget_note": "Local guard using costs/estimates and inflight reservations; not a provider hard cap"}
    save(BASE / "config.json", cfg)
    print(json.dumps({"requests": len(requests), "completions": len(records), "canary": cfg["canary_count"],
                      "budget_usd": cfg["budget_usd"]}))


def load_frozen():
    cfg = json.loads((BASE / "config.json").read_text())
    assert sha((BASE / "requests.jsonl.gz").read_bytes()) == cfg["requests_stored_sha256"]
    for name, entry in cfg["producer"].items():
        assert sha((ROOT / "scripts/eval" / name).read_bytes()) == entry["content_sha256"], "Producer changed"
    for source in cfg["sources"]:
        assert sha((ROOT / source["path"]).read_bytes()) == source["stored_sha256"], "Input changed"
    return cfg, read_events(BASE / "requests.jsonl.gz")


def summarize(cfg, requests, events, running=False):
    done = {e["request_id"]: e for e in events if e["status"] in TERMINAL}
    counts = Counter(e["status"] for e in done.values())
    status = {"requested": len(requests), "terminal": len(done), "numeric_scores": counts["ok"],
              "labels": counts["label"], "provider_filtered": counts["provider_filtered"],
              "pending": len(requests) - len(done), "attempts": len(events),
              "attempt_status_counts": dict(Counter(e["status"] for e in events)),
              "cost_usd": sum(e.get("cost_usd", 0) for e in events),
              "unreported_cost_attempts": sum(e.get("cost_basis") == "unreported" for e in events),
              "running": running, "updated_at_unix": time.time()}
    save(BASE / "status.json", status)
    return status


def export(requests, events):
    latest = {e["request_id"]: e for e in events}
    results = []
    for r in requests:
        e = latest.get(r["request_id"])
        if e is None:
            continue
        row = {k: r[k] for k in ("request_id", "model_family", "task_id", "completion_id", "eval_id",
               "group_id", "axis", "question", "completion", "score_name", "coherence_source", "inference_source")}
        row.update({k: e.get(k) for k in ("score", "score_label", "score_source_text", "status", "event_id")})
        row.update(judge_model=MODEL, judge_provider=e.get("provider"))
        results.append(row)
    write_gzip(BASE / "scores.jsonl.gz", results)
    artifacts = []
    for p in sorted(BASE.glob("*.jsonl.gz")):
        raw = p.read_bytes()
        content = gzip.decompress(raw)
        artifacts.append({"path": str(p.relative_to(ROOT)), "format": "jsonl", "compression": "gzip",
                          "stored_sha256": sha(raw), "content_sha256": sha(content),
                          "stored_bytes": len(raw), "rows": len(content.splitlines())})
    save(BASE / "artifacts.json", artifacts)


async def run(stage, key):
    import httpx
    cfg, requests = load_frozen()
    events_path = BASE / "judge_events.jsonl.gz"
    events = read_events(events_path)
    finished = {e["request_id"] for e in events if e["status"] in TERMINAL}
    counts = Counter(e["request_id"] for e in events)
    canary = canaries(requests)
    if stage == "remaining":
        approved = json.loads((BASE / "canary_passed.json").read_text())
        assert approved["passed"] and canary <= finished
    selected = [r for r in requests if r["request_id"] not in finished
                and (stage != "canary" or r["request_id"] in canary)]
    queue = asyncio.Queue()
    for request in selected:
        queue.put_nowait(request)
    concurrency = 4 if stage == "canary" else cfg["concurrency"]
    fatal, lock = asyncio.Event(), asyncio.Lock()
    spent = sum(e.get("cost_usd", 0) for e in events)
    reserved = 0.0
    async with httpx.AsyncClient(timeout=120, headers={"Authorization": "Bearer " + key,
                                "User-Agent": "selective-learning-benchmark/vanilla-judge"},
                                limits=httpx.Limits(max_connections=concurrency)) as client:
        async def worker():
            nonlocal spent, reserved
            while not queue.empty() and not fatal.is_set():
                try:
                    r = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                while counts[r["request_id"]] < cfg["max_attempts_per_id"] and not fatal.is_set():
                    reserve = cfg["inflight_reservation_usd"]
                    async with lock:
                        if spent + reserved + reserve > cfg["budget_usd"]:
                            fatal.set()
                            break
                        reserved += reserve
                    payload = {**cfg["body"], "messages": [{"role": "user", "content": r["prompt"]}]}
                    event = {"schema_version": "slb.vanilla_judge_attempt.v1", "event_id": str(uuid.uuid4()),
                             **{k: r[k] for k in ("request_id", "model_family", "task_id", "completion_id", "eval_id", "score_name")},
                             "at_unix": time.time(), "request": payload, "cost_usd": 0.0, "cost_basis": "unreported",
                             "score": None, "score_label": "ERROR", "score_source_text": ""}
                    http_status = 0
                    started = time.monotonic()
                    try:
                        response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=payload)
                        http_status = response.status_code
                        event["response_text"] = response.text.replace(key, "[REDACTED]")
                        data = json.loads(event["response_text"])
                        event["status"] = classify(data, http_status)
                        event["provider"] = data.get("provider")
                        event["cost_usd"], event["cost_basis"] = reported_cost(data, cfg["pricing_per_token"])
                        if event["status"] == "provider_filtered":
                            event["score_label"] = "PROVIDER_FILTERED"
                        if event["status"] == "ok":
                            if event["provider"] != cfg["response_provider"]:
                                raise ValueError("Unexpected judge provider")
                            choice = data["choices"][0]
                            message = choice["message"]
                            if message.get("reasoning") or message.get("reasoning_details"):
                                raise ValueError("Unexpected judge reasoning despite effort none")
                            raw = message.get("content") or ""
                            parsed, parsed_status = parse_score(r, raw)
                            event.update(parsed)
                            event.update(status=parsed_status, finish_reason=choice["finish_reason"])
                    except (httpx.TimeoutException, httpx.NetworkError):
                        event["status"] = "transport_error"
                    except Exception as exc:
                        event.update(status="protocol_error", error_type=type(exc).__name__)
                    event.update(http_status=http_status, elapsed_seconds=time.monotonic() - started)
                    async with lock:
                        reserved -= reserve
                        spent += event["cost_usd"]
                        counts[r["request_id"]] += 1
                        event["attempt"] = counts[r["request_id"]]
                        append_event(events_path, event)
                        events.append(event)
                        if event["status"] in TERMINAL:
                            finished.add(r["request_id"])
                        if len(events) % 25 == 0 or stage == "canary":
                            status = summarize(cfg, requests, events, running=True)
                            if len(events) % 100 == 0 or stage == "canary":
                                print(json.dumps(status), flush=True)
                    if event["status"] in TERMINAL:
                        break
                    if event["status"] == "protocol_error" or http_status in {400, 401, 402, 403, 404, 422}:
                        fatal.set()
                        break
                    await asyncio.sleep(min(5 * 2 ** (counts[r["request_id"]] - 1), 60))
                queue.task_done()
        try:
            summarize(cfg, requests, events, running=True)
            await asyncio.gather(*(worker() for _ in range(concurrency)))
        finally:
            status = summarize(cfg, requests, events, running=False)
            export(requests, events)
            if stage == "canary":
                final = {e["request_id"]: e for e in events if e["request_id"] in canary}
                passed = canary == final.keys() and all(e["status"] in {"ok", "label"} for e in final.values())
                save(BASE / "canary_passed.json", {"passed": passed, "expected": len(canary),
                     "terminal": len(final), "events_sha256": sha(events_path.read_bytes())})
            print(json.dumps({**status, "stopped_on_guard_or_error": fatal.is_set()}), flush=True)


def verify():
    cfg, requests = load_frozen()
    assert not json.loads((BASE / "status.json").read_text())["running"]
    expected = {r["request_id"]: r for r in requests}
    events = read_events(BASE / "judge_events.jsonl.gz")
    terminal = [e for e in events if e["status"] in TERMINAL]
    counts = Counter(e["request_id"] for e in terminal)
    assert all(n == 1 for n in counts.values())
    assert len({e["event_id"] for e in events}) == len(events)
    for e in events:
        r = expected[e["request_id"]]
        assert e["request"] == {**cfg["body"], "messages": [{"role": "user", "content": r["prompt"]}]}
        if e["status"] in {"ok", "label"}:
            assert e["provider"] == "Alibaba"
            parsed, status = parse_score(r, e["score_source_text"])
            assert status == e["status"] and parsed["score"] == e["score"] and parsed["score_label"] == e["score_label"]
    for a in json.loads((BASE / "artifacts.json").read_text()):
        raw = (ROOT / a["path"]).read_bytes()
        assert sha(raw) == a["stored_sha256"] and sha(gzip.decompress(raw)) == a["content_sha256"]
    exported = read_events(BASE / "scores.jsonl.gz")
    latest = {e["request_id"]: e for e in events}
    assert len(exported) == len(latest) and {r["request_id"] for r in exported} == latest.keys()
    for r in exported:
        e = latest[r["request_id"]]
        assert all(r[k] == e.get(k) for k in ("score", "score_label", "score_source_text", "status", "event_id"))
    groups = []
    for family, task in sorted({(r["model_family"], r["task_id"]) for r in requests}):
        group = [e for e in terminal if e["model_family"] == family and e["task_id"] == task]
        groups.append({"model_family": family, "task_id": task, "terminal": len(group),
                       "numeric": sum(e["status"] == "ok" for e in group),
                       "labels": sum(e["status"] == "label" for e in group),
                       "provider_filtered": sum(e["status"] == "provider_filtered" for e in group)})
    result = {"all_requests_terminal": expected.keys() == counts.keys(),
              "missing_request_ids": sorted(expected.keys() - counts.keys()), "duplicate_terminal_ids": 0,
              "groups": groups, **summarize(cfg, requests, events), "integrity_checks_passed": True}
    save(BASE / "verification.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "status", "verify"])
    parser.add_argument("--stage", choices=["canary", "remaining"], default="canary")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "status":
        print((BASE / "status.json").read_text())
    elif args.action == "verify":
        verify()
    else:
        if not args.env_file:
            parser.error("--env-file is required")
        lock_path = ROOT / ".migration-cache/vanilla_judge.lock"
        lock_path.parent.mkdir(exist_ok=True)
        with lock_path.open("a") as process_lock:
            fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(run(args.stage, load_key(args.env_file)))


if __name__ == "__main__":
    main()
