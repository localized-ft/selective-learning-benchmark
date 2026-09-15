"""Offline validation of the new API inference files; no inference/judging."""
from collections import Counter
import gzip
import json

from vanilla_api import BASE, read_events, summarize
from vanilla_gpu import save, sha


def verify(model):
    path = BASE / model
    cfg = json.loads((path / "config.json").read_text())
    request_bytes = (path / "requests.jsonl").read_bytes()
    assert sha(request_bytes) == cfg["requests_sha256"]
    requests = [json.loads(s) for s in request_bytes.splitlines()]
    expected = {r["completion_id"]: r for r in requests}
    assert len(expected) == len(requests) == cfg["request_count"]
    events = read_events(path / "inference_events.jsonl.gz")
    terminal = [e for e in events if e["status"] in {"ok", "provider_filtered"}]
    counts = Counter(e["completion_id"] for e in terminal)
    assert all(n == 1 for n in counts.values()), "Duplicate terminal result for one slot"
    assert set(counts) <= expected.keys()
    assert len({e["event_id"] for e in events}) == len(events)
    for e in events:
        request = expected[e["completion_id"]]
        assert e["request"]["messages"] == request["messages"]
        assert e["request"]["seed"] == request["inference_seed"]
        assert all(e["request"][k] == v for k, v in cfg["body"].items())
        if e["status"] == "ok":
            assert e["provider"] == cfg["response_provider"]
    complete = set(counts) == expected.keys()
    status = json.loads((path / "status.json").read_text())
    if status["running"]:
        raise RuntimeError("Wait until the inference process has finished before final validation")
    # Validate every gzip file's compressed and decompressed byte hashes.
    for artifact in json.loads((path / "artifacts.json").read_text()):
        file = path / artifact["path"].split("/")[-1]
        data = file.read_bytes()
        assert sha(data) == artifact["stored_sha256"]
        assert sha(gzip.decompress(data)) == artifact["content_sha256"]
    usable = {e["completion_id"]: e for e in terminal if e["status"] == "ok"}
    with gzip.open(path / "completions.jsonl.gz", "rt") as f:
        outputs = [json.loads(line) for line in f]
    assert len(outputs) == len(usable)
    assert {c["completion_id"] for c in outputs} == usable.keys()
    for c in outputs:
        e = usable[c["completion_id"]]
        assert c["completion"] == e["completion"]
        assert c["eval_id"] == e["eval_id"]
        if model == "qwen3_8b":
            assert e["retained_text_tokens"] <= 2000
    tasks = []
    for task in sorted({r["task_id"] for r in requests}):
        expected_ids = {r["completion_id"] for r in requests if r["task_id"] == task}
        task_events = [e for e in terminal if e["completion_id"] in expected_ids]
        tasks.append({"task_id": task, "planned": len(expected_ids), "returned": len(task_events),
                      "usable": sum(e["status"] == "ok" for e in task_events),
                      "provider_filtered": sum(e["status"] == "provider_filtered" for e in task_events)})
    result = {"model_family": model, "all_requests_terminal": complete, "tasks": tasks,
              "missing_ids": sorted(expected.keys() - counts.keys()), "duplicate_terminal_ids": 0,
              "usable": len(outputs), "provider_filtered": len(terminal) - len(outputs),
              "empty_usable_outputs": sum(not c["completion"].strip() for c in outputs),
              "client_truncated": sum(bool(e.get("client_truncated")) for e in usable.values()),
              "finish_reasons": dict(Counter(e["finish_reason"] for e in usable.values())),
              "cost_usd": sum(e.get("cost_usd", 0) for e in events),
              "attempts": len(events), "attempt_status_counts": dict(Counter(e["status"] for e in events)),
              "unreported_cost_attempts": sum(e["cost_basis"] == "unreported" for e in events),
              "new_judge_calls": 0}
    save(path / "verification.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["llama31_8b", "qwen3_8b"])
    verify(parser.parse_args().model)
