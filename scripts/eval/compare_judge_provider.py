"""One DeepInfra pass on the 162 labels still REFUSAL after the Alibaba retry.

Explicit user approval covers these same prompts/outputs sent to DeepInfra via
OpenRouter. Historical passes are immutable; this is an outcome-selected diagnostic.
"""
import argparse
import asyncio
from collections import Counter
import fcntl
import gzip
import json
from pathlib import Path
import urllib.request

import vanilla_judge as judge
from retry_vanilla_refusals import select_refusals

ORIGINAL = judge.BASE
PREVIOUS = ORIGINAL / "retry_refusals_20260915"
DEST = ORIGINAL / "retry_deepinfra_20260915"


def prepare():
    if (DEST / "config.json").exists():
        raise RuntimeError("Pass already prepared; resume without recreating it")
    verification = json.loads((PREVIOUS / "verification.json").read_text())
    assert verification["all_requests_terminal"] and verification["integrity_checks_passed"]
    for artifact in json.loads((PREVIOUS / "artifacts.json").read_text()):
        assert judge.sha((judge.ROOT / artifact["path"]).read_bytes()) == artifact["stored_sha256"]
    requests, prior = select_refusals(judge.read_events(PREVIOUS / "requests.jsonl.gz"),
                                     judge.read_events(PREVIOUS / "scores.jsonl.gz"))
    assert len(requests) == 162 and all(r["score_name"] == "alignment" for r in requests)
    cfg = json.loads((PREVIOUS / "config.json").read_text())
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models/" + judge.MODEL + "/endpoints", timeout=30) as response:
        catalog = json.load(response)
    endpoint = next(e for e in catalog["data"]["endpoints"] if e["tag"] == "deepinfra/fp8")
    assert {"reasoning", "temperature", "top_p", "max_tokens"} <= set(endpoint["supported_parameters"])
    DEST.mkdir(parents=True, exist_ok=True)
    judge.write_gzip(DEST / "requests.jsonl.gz", requests)
    judge.save(DEST / "endpoint_catalog.json", catalog)
    cfg["body"]["provider"] = {"only": ["deepinfra/fp8"], "allow_fallbacks": False, "require_parameters": True}
    cfg.update(schema_version="slb.judge_provider_comparison.v1", pass_id=DEST.name,
               response_provider="DeepInfra", endpoint_snapshot=endpoint, pricing_per_token=endpoint["pricing"],
               request_count=len(requests), completion_count=len(requests),
               requests_stored_sha256=judge.sha((DEST / "requests.jsonl.gz").read_bytes()),
               sources=[judge.local_source(p / name) for p in (ORIGINAL, PREVIOUS) for name in
                        ("config.json", "requests.jsonl.gz", "scores.jsonl.gz", "artifacts.json", "verification.json")],
               producer={name: judge.store_artifact((judge.ROOT / "scripts/eval" / name).read_bytes())
                         for name in [*cfg["producer"], "compare_judge_provider.py"]},
               concurrency=8, budget_usd=0.25, canary_count=len(judge.canaries(requests)),
               parent_judge_directory=str(PREVIOUS.relative_to(judge.ROOT)),
               selection_policy="162 REFUSAL labels remaining after the first Alibaba retry; user approved DeepInfra resend",
               merge_policy="Diagnostic only: preserve both Alibaba passes and do not promote into primary analysis")
    original_body = json.loads((PREVIOUS / "config.json").read_text())["body"]
    assert {k:v for k,v in cfg["body"].items() if k != "provider"} == {
        k:v for k,v in original_body.items() if k != "provider"}
    judge.save(DEST / "config.json", cfg)
    judge.save(DEST / "selection.json", [{"request_id": r["request_id"], "previous_event_id": prior[r["request_id"]]["event_id"],
                **{k:r[k] for k in ("model_family", "task_id", "completion_id", "prompt_sha256")}} for r in requests])
    print(json.dumps({"selected": len(requests), "canary": cfg["canary_count"], "provider": cfg["response_provider"],
                      "by_model": dict(Counter(r["model_family"] for r in requests))}))


def verify():
    cfg, requests = judge.load_frozen()
    assert not json.loads((DEST / "status.json").read_text())["running"]
    expected = {r["request_id"]: r for r in requests}
    previous = {r["request_id"]: r for r in judge.read_events(PREVIOUS / "scores.jsonl.gz")}
    original = {r["request_id"]: r for r in judge.read_events(ORIGINAL / "scores.jsonl.gz")}
    events = judge.read_events(DEST / "judge_events.jsonl.gz")
    assert len({e["event_id"] for e in events}) == len(events)
    terminal = [e for e in events if e["status"] in judge.TERMINAL]
    counts = Counter(e["request_id"] for e in terminal)
    assert all(n == 1 for n in counts.values())
    for e in events:
        r = expected[e["request_id"]]
        assert previous[r["request_id"]]["score_label"] == original[r["request_id"]]["score_label"] == "REFUSAL"
        assert e["request"] == {**cfg["body"], "messages": [{"role":"user", "content":r["prompt"]}]}
        if e["status"] in {"ok", "label"}:
            assert e["provider"] == cfg["response_provider"] == "DeepInfra"
            parsed, status = judge.parse_score(r, e["score_source_text"])
            assert status == e["status"] and parsed["score"] == e["score"] and parsed["score_label"] == e["score_label"]
    for a in json.loads((DEST / "artifacts.json").read_text()):
        raw = (judge.ROOT / a["path"]).read_bytes()
        assert judge.sha(raw) == a["stored_sha256"] and judge.sha(gzip.decompress(raw)) == a["content_sha256"]
    rows = judge.read_events(DEST / "scores.jsonl.gz")
    latest = {e["request_id"]: e for e in events}
    assert len(rows) == len(latest) and {r["request_id"] for r in rows} == latest.keys()
    for r in rows:
        assert all(r[k] == latest[r["request_id"]].get(k) for k in
                   ("score", "score_label", "score_source_text", "status", "event_id"))
        assert all(r[k] == expected[r["request_id"]][k] for k in
                   ("model_family", "task_id", "completion_id", "eval_id", "completion", "score_name"))
    result = {**judge.summarize(cfg, requests, events), "all_requests_terminal": expected.keys() == counts.keys(),
              "missing_request_ids": sorted(expected.keys() - counts.keys()), "duplicate_terminal_ids": 0,
              "integrity_checks_passed": True, "prior_passes_unchanged": True,
              "by_model": {m: dict(Counter(r["status"] for r in rows if r["model_family"] == m))
                           for m in sorted({r["model_family"] for r in rows})},
              "labels": dict(Counter(r["score_label"] for r in rows if r["status"] == "label"))}
    judge.save(DEST / "verification.json", result)
    comparison = [{**{k:r[k] for k in ("request_id", "model_family", "task_id", "completion_id", "score", "score_label", "status", "event_id")},
                   "previous_event_id": previous[r["request_id"]]["event_id"],
                   "original_event_id": original[r["request_id"]]["event_id"]} for r in rows]
    judge.save(DEST / "comparison.json", {"promoted_to_primary": False, "rows": comparison})
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "verify"])
    parser.add_argument("--stage", choices=["canary", "remaining"], default="canary")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
        return
    judge.BASE = DEST
    if args.action == "verify":
        verify()
    else:
        if not args.env_file:
            parser.error("--env-file required")
        with (judge.ROOT / ".migration-cache/vanilla_deepinfra_retry.lock").open("a") as process_lock:
            fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(judge.run(args.stage, judge.load_key(args.env_file)))


if __name__ == "__main__":
    main()
