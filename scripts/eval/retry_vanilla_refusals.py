"""One user-requested retry pass of the 188 vanilla REFUSAL judgments.

Reuses the frozen judge engine without modifying it or the original results.
REFUSAL on this pass is terminal too: never keep trying until a number appears.
"""
import argparse
import asyncio
from collections import Counter
import fcntl
import json
from pathlib import Path

import vanilla_judge as judge

ORIGINAL = judge.BASE
DEST = ORIGINAL / "retry_refusals_20260915"


def select_refusals(requests, scores):
    selected = {r["request_id"]: r for r in scores
                if r["status"] == "label" and r["score_label"] == "REFUSAL"}
    assert len(selected) == sum(r["status"] == "label" and r["score_label"] == "REFUSAL" for r in scores)
    rows = [r for r in requests if r["request_id"] in selected]
    assert len(rows) == len(selected)
    return rows, selected


def prepare():
    if (DEST / "config.json").exists():
        raise RuntimeError("Retry already prepared; resume it, do not recreate it")
    cfg = json.loads((ORIGINAL / "config.json").read_text())
    verification = json.loads((ORIGINAL / "verification.json").read_text())
    assert verification["all_requests_terminal"] and verification["integrity_checks_passed"]
    for a in json.loads((ORIGINAL / "artifacts.json").read_text()):
        assert judge.sha((judge.ROOT / a["path"]).read_bytes()) == a["stored_sha256"]
    requests, prior = select_refusals(judge.read_events(ORIGINAL / "requests.jsonl.gz"),
                                     judge.read_events(ORIGINAL / "scores.jsonl.gz"))
    assert len(requests) == 188 and all(r["score_name"] == "alignment" for r in requests)
    DEST.mkdir(parents=True, exist_ok=True)
    judge.write_gzip(DEST / "requests.jsonl.gz", requests)
    producer = {name: judge.store_artifact((judge.ROOT / "scripts/eval" / name).read_bytes())
                for name in [*cfg["producer"], "retry_vanilla_refusals.py"]}
    cfg.update(schema_version="slb.vanilla_judge_refusal_retry.v1", pass_id=DEST.name,
               request_count=len(requests), completion_count=len(requests),
               requests_stored_sha256=judge.sha((DEST / "requests.jsonl.gz").read_bytes()),
               sources=[judge.local_source(ORIGINAL / name) for name in
                        ("config.json", "requests.jsonl.gz", "scores.jsonl.gz", "artifacts.json", "verification.json")],
               producer=producer, concurrency=8, budget_usd=0.25,
               canary_count=len(judge.canaries(requests)), reused_score_count=0, reused_completion_count=0,
               selection_policy="Exactly the 188 original REFUSAL labels; user explicitly requested one retry pass",
               merge_policy="Separate diagnostic pass; do not replace original scores or change primary analysis",
               parent_judge_directory=str(ORIGINAL.relative_to(judge.ROOT)))
    cfg.pop("reused_judgments", None)
    judge.save(DEST / "config.json", cfg)
    judge.save(DEST / "selection.json", [{"request_id": r["request_id"], "original_event_id": prior[r["request_id"]]["event_id"],
               **{k: r[k] for k in ("model_family", "task_id", "completion_id", "score_name", "prompt_sha256")}}
               for r in requests])
    print(json.dumps({"selected": len(requests), "by_model": dict(Counter(r["model_family"] for r in requests)),
                      "canary_count": cfg["canary_count"], "budget_usd": cfg["budget_usd"]}))


def compare():
    original = {r["request_id"]: r for r in judge.read_events(ORIGINAL / "scores.jsonl.gz")}
    retried = judge.read_events(DEST / "scores.jsonl.gz")
    rows = []
    for r in retried:
        parent = original[r["request_id"]]
        assert parent["score_label"] == "REFUSAL" and parent["status"] == "label"
        rows.append({k: r[k] for k in ("request_id", "model_family", "task_id", "completion_id", "score_name", "score",
                                       "score_label", "status", "event_id")}
                    | {"original_event_id": parent["event_id"], "original_score_label": "REFUSAL"})
    result = {"retried": len(rows), "outcomes": dict(Counter(r["status"] for r in rows)),
              "labels": dict(Counter(r["score_label"] for r in rows if r["status"] == "label")),
              "by_model": {m: dict(Counter(r["status"] for r in rows if r["model_family"] == m))
                           for m in sorted({r["model_family"] for r in rows})},
              "original_results_modified": False, "retry_pass_promoted_to_primary": False, "rows": rows}
    judge.save(DEST / "comparison.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


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
        judge.verify()
        compare()
    else:
        if not args.env_file:
            parser.error("--env-file required")
        with (judge.ROOT / ".migration-cache/vanilla_refusal_retry.lock").open("a") as process_lock:
            fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(judge.run(args.stage, judge.load_key(args.env_file)))


if __name__ == "__main__":
    main()
