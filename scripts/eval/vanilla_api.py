"""Resumable, provider-pinned vanilla API inference; never runs a judge.

Raw response attempts are append-only gzip JSONL, including provider failures.
No credentials are written to result files. Old Qwen medical results are reused
by reference and are never silently retried or replaced.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import uuid
import urllib.request

from vanilla_gpu import ROOT, request_rows, save, sha

BASE = ROOT / "result/supplemental/vanilla_api_20260914/api"
QWEN_TOKENIZER = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
MODELS = {
    "llama31_8b": {"api_model": "meta-llama/llama-3.1-8b-instruct", "provider": "coreweave/bf16",
                    "response_provider": "CoreWeave", "budget_usd": 4.0, "concurrency": 16},
    "qwen3_8b": {"api_model": "qwen/qwen3-8b", "provider": "alibaba",
                  "response_provider": "Alibaba", "budget_usd": 8.0, "concurrency": 8},
}


def archived(digest):
    raw = gzip.decompress((ROOT / "result/artifacts/sha256" / digest[:2] / (digest + ".gz")).read_bytes())
    assert sha(raw) == digest
    return raw


def read_events(path):
    if not path.exists():
        return []
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_event(path, event):
    # Each append is an independently complete gzip member; no buffered tail.
    raw = (json.dumps(event, ensure_ascii=False) + "\n").encode()
    with path.open("ab") as f:
        f.write(gzip.compress(raw, compresslevel=6, mtime=0))
        f.flush()
        os.fsync(f.fileno())


def canary_ids(requests):
    selected = {}
    for r in requests:
        selected.setdefault((r["task_id"], r["axis"]), r["completion_id"])
    return set(selected.values())


def reconstruct(message, qwen=False):
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("Unexpected non-text content")
    if not qwen:
        return content
    reasoning = message.get("reasoning") or ""
    if not isinstance(reasoning, str):
        raise ValueError("Unexpected reasoning format")
    if not reasoning:
        parts = [d.get("text", "") for d in message.get("reasoning_details", [])
                 if d.get("type") == "reasoning.text"]
        reasoning = "\n".join(parts)
    if reasoning:
        return "<think>\n" + reasoning + "\n</think>\n\n" + content
    return content


def prepare(model):
    path = BASE / model
    if (path / "inference_events.jsonl.gz").exists():
        raise SystemExit("Inference already started; preserve its configuration")
    spec = MODELS[model]
    url = "https://openrouter.ai/api/v1/models/" + spec["api_model"] + "/endpoints"
    with urllib.request.urlopen(url, timeout=30) as response:
        catalog = json.load(response)
    endpoint = next(e for e in catalog["data"]["endpoints"] if e["tag"] == spec["provider"])
    required = {"temperature", "top_p", "top_k", "max_tokens", "seed", "frequency_penalty", "presence_penalty"}
    if model == "qwen3_8b":
        required.add("reasoning")
    assert required <= set(endpoint["supported_parameters"])
    rows, sources = request_rows()
    if model == "qwen3_8b":
        rows = [r for r in rows if r["task_id"] != "bad_medical_advice"]
    path.mkdir(parents=True, exist_ok=True)
    raw = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in rows)
    (path / "requests.jsonl").write_bytes(raw)
    body = {"model": spec["api_model"], "temperature": 1.0, "top_p": 1.0, "top_k": 50,
            "max_tokens": 2000, "frequency_penalty": 0.0, "presence_penalty": 0.0,
            "stream": False, "provider": {"only": [spec["provider"]],
                                          "allow_fallbacks": False, "require_parameters": True}}
    if model == "qwen3_8b":
        body["reasoning"] = {"enabled": True, "exclude": False}
    cfg = {"schema_version": "slb.vanilla_api_config.v1", "model_family": model, **spec,
           "request_count": len(rows), "requests_sha256": sha(raw), "snapshot_sources": sources,
           "body": body, "training_seed": None, "max_attempts_per_id": 5,
           "max_inflight_reservation_usd": 0.01 if model == "llama31_8b" else 0.065,
           "pricing_per_token": endpoint["pricing"], "endpoint_snapshot": endpoint,
           "tokenizers_version": "0.21.4", "client_cap": 2000 if model == "qwen3_8b" else None,
           "tokenizer_artifact_id": QWEN_TOKENIZER if model == "qwen3_8b" else None,
           "old_qwen_medical": "result/supplemental/qwen3_openrouter_bad_medical_20260909" if model == "qwen3_8b" else None,
           "protocol_limitations": ["Provider-served weights/backend are not independently verified",
                                    "API seeds are not training seeds or guarantees of deterministic text",
                                    "BF16 Llama differs from historical FP16; Alibaba quantization is unknown",
                                    "API-native stop handling may differ from historical Transformers defaults"],
           "budget_note": "Local guard uses recorded cost or estimates plus in-flight reservations; not a provider hard cap"}
    save(path / "config.json", cfg)
    save(path / "endpoint_catalog.json", catalog)
    print(json.dumps({"model": model, "prepared": len(rows), "provider": spec["provider"]}), flush=True)


def load_key(env_file):
    from dotenv import dotenv_values
    values = list(dotenv_values(env_file).values()) + [os.environ.get("OPENROUTER_API_KEY")]
    keys = {v for v in values if isinstance(v, str) and v.startswith("sk-or-")}
    if len(keys) != 1:
        raise RuntimeError("Expected one unambiguous OpenRouter key in private environment")
    return keys.pop()


def classify(response, status_code):
    error = response.get("error")
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    error = error or choice.get("error")
    reason = choice.get("finish_reason")
    error_text = json.dumps(error or {}).lower()
    filtered = reason == "content_filter" or any(x in error_text for x in (
        "content_filter", "content filter", "inappropriate content", "data_inspection_failed"))
    if filtered:
        return "provider_filtered"
    if status_code != 200 or error or reason == "error":
        return "api_error"
    if not choices or reason not in {"stop", "length"}:
        return "protocol_error"
    return "ok"


def reported_cost(response, pricing):
    usage = response.get("usage") or {}
    cost = usage.get("cost")
    if isinstance(cost, (float, int)) and math.isfinite(cost) and cost >= 0:
        return float(cost), "api_reported"
    if usage:
        return (int(usage.get("prompt_tokens") or 0) * float(pricing["prompt"])
                + int(usage.get("completion_tokens") or 0) * float(pricing["completion"])), "token_estimate"
    return 0.0, "unreported"


def summarize(path, cfg, requests, events, running=False):
    terminal = {e["completion_id"]: e for e in events if e["status"] in {"ok", "provider_filtered"}}
    counts = Counter(e["status"] for e in terminal.values())
    status = {"model_family": cfg["model_family"], "requested": len(requests),
              "terminal": len(terminal), "usable": counts["ok"], "provider_filtered": counts["provider_filtered"],
              "pending": len(requests) - len(terminal), "attempts": len(events),
              "cost_usd": sum(e.get("cost_usd", 0) for e in events),
              "unreported_cost_attempts": sum(e.get("cost_basis") == "unreported" for e in events),
              "running": running, "updated_at_unix": time.time()}
    save(path / "status.json", status)
    return status


def export(path, cfg, requests, events):
    terminal = {e["completion_id"]: e for e in events if e["status"] in {"ok", "provider_filtered"}}
    records, details = [], []
    for r in requests:
        event = terminal.get(r["completion_id"])
        if event is None:
            continue
        if event["status"] == "ok":
            records.append({"completion_id": r["completion_id"], "eval_id": r["eval_id"],
                            "completion": event["completion"]})
        details.append({**{k: r[k] for k in ("completion_id", "eval_id", "task_id", "axis", "sample_index", "inference_seed")},
                        **{k: event.get(k) for k in ("status", "finish_reason", "client_truncated", "original_text_tokens", "retained_text_tokens", "provider", "event_id")}})
    for name, rows in [("completions.jsonl.gz", records), ("completion_status.jsonl.gz", details)]:
        raw = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in rows)
        (path / name).write_bytes(gzip.compress(raw, mtime=0))
    artifacts = []
    for file in sorted(path.glob("*.jsonl.gz")):
        data = file.read_bytes(); raw = gzip.decompress(data)
        artifacts.append({"path": str(file.relative_to(ROOT / "result")), "compression": "gzip",
                          "format": "jsonl", "stored_sha256": sha(data), "content_sha256": sha(raw),
                          "stored_bytes": len(data), "rows": len(raw.splitlines())})
    save(path / "artifacts.json", artifacts)


async def run_model(model, stage, key):
    import httpx
    from tokenizers import Tokenizer
    path = BASE / model
    cfg = json.loads((path / "config.json").read_text())
    raw = (path / "requests.jsonl").read_bytes()
    assert sha(raw) == cfg["requests_sha256"]
    requests = [json.loads(s) for s in raw.splitlines()]
    events_path = path / "inference_events.jsonl.gz"
    events = read_events(events_path)
    finished = {e["completion_id"] for e in events if e["status"] in {"ok", "provider_filtered"}}
    counts = Counter(e["completion_id"] for e in events)
    canary = canary_ids(requests)
    if stage == "remaining" and not canary <= finished:
        raise RuntimeError("Canary incomplete; inspect before full execution")
    selected = [r for r in requests if r["completion_id"] not in finished
                and (stage != "canary" or r["completion_id"] in canary)]
    tokenizer = None
    if model == "qwen3_8b":
        assert importlib.metadata.version("tokenizers") == cfg["tokenizers_version"]
        tokenizer = Tokenizer.from_str(archived(QWEN_TOKENIZER).decode())
    lock = asyncio.Lock()
    fatal = asyncio.Event()
    spent = sum(e.get("cost_usd", 0) for e in events)
    reserved = 0.0
    concurrency = 2 if stage == "canary" else cfg["concurrency"]
    queue = asyncio.Queue()
    for request in selected:
        queue.put_nowait(request)

    async with httpx.AsyncClient(timeout=180, headers={"Authorization": "Bearer " + key,
                                  "User-Agent": "selective-learning-benchmark/vanilla-api"},
                                limits=httpx.Limits(max_connections=concurrency)) as client:
        async def work():
            nonlocal spent, reserved
            while not queue.empty() and not fatal.is_set():
                try:
                    r = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                while counts[r["completion_id"]] < cfg["max_attempts_per_id"] and not fatal.is_set():
                    reserve = cfg["max_inflight_reservation_usd"]
                    async with lock:
                        if spent + reserved + reserve > cfg["budget_usd"]:
                            fatal.set()
                            break
                        reserved += reserve
                    payload = {**cfg["body"], "messages": r["messages"], "seed": r["inference_seed"]}
                    event = {"schema_version": "slb.vanilla_api_attempt.v1", "event_id": str(uuid.uuid4()),
                             "completion_id": r["completion_id"], "eval_id": r["eval_id"], "task_id": r["task_id"],
                             "at_unix": time.time(), "request": payload, "cost_usd": 0.0, "cost_basis": "unreported"}
                    http_status = 0
                    response = {}
                    started = time.monotonic()
                    try:
                        http = await client.post("https://openrouter.ai/api/v1/chat/completions", json=payload)
                        http_status = http.status_code
                        event["response_text"] = http.text.replace(key, "[REDACTED]")
                        response = json.loads(event["response_text"])
                        event["status"] = classify(response, http_status)
                        event["cost_usd"], event["cost_basis"] = reported_cost(response, cfg["pricing_per_token"])
                        event["provider"] = response.get("provider")
                        if event["status"] == "ok":
                            if event["provider"] != cfg["response_provider"]:
                                raise ValueError("Unexpected provider; stop instead of silently changing backend")
                            choice = response["choices"][0]
                            text = reconstruct(choice["message"], model == "qwen3_8b")
                            ids = tokenizer.encode(text, add_special_tokens=False).ids if tokenizer else None
                            if ids is not None:
                                event.update(original_text_tokens=len(ids), retained_text_tokens=min(len(ids), 2000),
                                             client_truncated=len(ids) > 2000)
                                if len(ids) > 2000:
                                    text = tokenizer.decode(ids[:2000], skip_special_tokens=False)
                            event.update(completion=text, finish_reason=choice["finish_reason"])
                    except (httpx.TimeoutException, httpx.NetworkError):
                        event["status"] = "transport_error"
                    except Exception as exc:
                        event.update(status="protocol_error", error_type=type(exc).__name__)
                    event.update(http_status=http_status, elapsed_seconds=time.monotonic() - started)
                    async with lock:
                        reserved -= reserve
                        spent += event["cost_usd"]
                        counts[r["completion_id"]] += 1
                        append_event(events_path, event)
                        events.append(event)
                        if event["status"] in {"ok", "provider_filtered"}:
                            finished.add(r["completion_id"])
                        status = summarize(path, cfg, requests, events, running=True)
                        if len(events) % 25 == 0 or stage == "canary":
                            print(json.dumps(status), flush=True)
                    if event["status"] in {"ok", "provider_filtered"}:
                        break
                    if event["status"] == "protocol_error" or http_status in {400, 401, 402, 403, 404, 422}:
                        fatal.set()
                        break
                    await asyncio.sleep(min(15 * 2**(counts[r["completion_id"]] - 1), 60))
                queue.task_done()
        try:
            await asyncio.gather(*(work() for _ in range(concurrency)))
        finally:
            status = summarize(path, cfg, requests, events, running=False)
            export(path, cfg, requests, events)
            print(json.dumps({**status, "stage": stage, "stopped_on_guard_or_error": fatal.is_set()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "status"])
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--stage", choices=["canary", "remaining"], default="canary")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    models = list(MODELS) if args.model == "all" else [args.model]
    if args.action == "prepare":
        for model in models:
            prepare(model)
    elif args.action == "status":
        for model in models:
            print((BASE / model / "status.json").read_text())
    else:
        if not args.env_file:
            parser.error("--env-file required")
        key = load_key(args.env_file)
        async def run_all():
            await asyncio.gather(*(run_model(model, args.stage, key) for model in models))
        asyncio.run(run_all())


if __name__ == "__main__":
    main()
