"""Re-judge every run in state/inventory.json with gpt-6-luna and summarize old vs new.

Kept identical to the paper pipeline (sunday/scripts/eval/judge_utility.py) except the judge:
  - requests are rebuilt from each run's own eval file (grading specs / judge prompts unchanged)
  - scoring uses Sunday's JudgeRunner (same prompt rendering + score parsing)
  - axis summaries use Sunday's add_axis_score_summary (raw mean + coherence>=50 filtered mean)
Deviations: judge = openrouter/openai/gpt-6-luna; <think>...</think> blocks are stripped from
completions before judging (the paper pipeline judged them unstripped).

Resumable: completions/eval files are cached in state/files, per-run scores in state/scores.
A run is only saved if <2% of its judge calls errored; rerun to retry the rest.

Usage:
    ../experiment-ip-variants/.venv/bin/python rejudge.py [--task bad_medical_advice] [--source ip_variants] [--summary-only]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IPV = REPO / "niels" / "experiment-ip-variants"
SUNDAY_EVAL = REPO / "sunday" / "scripts" / "eval"
sys.path.insert(0, str(IPV))
sys.path.insert(0, str(SUNDAY_EVAL))

from eval_constants import (  # noqa: E402
    RESULT_FIELD_AXIS,
    RESULT_FIELD_COMPLETION_ID,
    RESULT_FIELD_SCORE,
    RESULT_FIELD_SCORE_NAME,
    TASK_DATA_MODEL_AXIS_CAPABILITY,
)
from eval_data_model import EvalRequest, InferenceRequest  # noqa: E402
from judge_utility import JudgeRunner, add_axis_score_summary  # noqa: E402

JUDGE_MODEL = "openrouter/openai/gpt-6-luna"  # the direct openai/ route rejects max_tokens
CONCURRENT_CALLS = 100
CONCURRENT_RUNS = 6
MAX_ERROR_FRACTION = 0.02
STATE = HERE / "state"
FILES = STATE / "files"
SCORES = STATE / "scores"
RESULTS = HERE / "results"


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return text.split("</think>")[-1].strip()


def read_jsonl_text(text: str) -> list[dict]:
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def load_file(ow, file_ref: str) -> list[dict]:
    if file_ref.startswith("local:"):
        return read_jsonl_text((REPO / file_ref[len("local:"):]).resolve().read_text())
    cache = FILES / (file_ref.replace(":", "_") + ".jsonl")
    if not cache.exists():
        for attempt in range(6):
            try:
                data = ow.files.content(file_ref)
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # atomic: runs sharing an eval file download it concurrently
        tmp = cache.with_suffix(f".tmp{os.getpid()}_{id(data)}")
        tmp.write_bytes(data)
        tmp.replace(cache)
    return read_jsonl_text(cache.read_text())


def build_requests(eval_records: list[dict], completions: list[dict]) -> list[tuple[EvalRequest, str]]:
    by_id = {r["id"]: r for r in eval_records}
    out = []
    for c in completions:
        rec = by_id[c["eval_id"]]
        g = rec["grading"]
        out.append((EvalRequest(
            completion_id=c["completion_id"],
            eval_id=rec["id"],
            group_id=rec.get("group_id", ""),
            axis=rec["axis"],
            question=rec["messages"][0]["content"],
            reference_response=g.get("reference_response", ""),
            grading_method=g["method"],
            grading=g,
            inference=InferenceRequest(c["completion_id"], rec["messages"], 1.0, 2000),
        ), c["completion"]))
    return out


class LunaJudge(JudgeRunner):
    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.config["judge_base_url"],
                api_key=self.config["judge_api_key"],
                default_headers={"User-Agent": "python-httpx/0.27"},
                # the proxy sometimes returns Cloudflare 524s / leaves requests hanging;
                # without a short timeout a few stuck calls stall the whole run
                timeout=120,
                max_retries=4,
            )
        return self._client


def summarize(rows: list[dict]) -> dict:
    cap = [r for r in rows if r[RESULT_FIELD_AXIS] == TASK_DATA_MODEL_AXIS_CAPABILITY]
    ug = [r for r in rows if r[RESULT_FIELD_AXIS] != TASK_DATA_MODEL_AXIS_CAPABILITY]
    s: dict = {}
    if cap:
        add_axis_score_summary(s, cap, "capability_n", "capability_mean", "capability_key",
                               "capability_coh_n", "capability_coh_mean")
    if ug:
        add_axis_score_summary(s, ug, "ug_n", "ug_mean", "ug_key", "ug_coh_n", "ug_coh_mean")
    return s


async def judge_run(run: dict, ow, runner: LunaJudge, run_sem: asyncio.Semaphore) -> None:
    try:
        await _judge_run(run, ow, runner, run_sem)
    except Exception as e:  # one bad run must not stop the others
        print(f"FAILED {run['run_key']}: {type(e).__name__}: {str(e)[:200]}", flush=True)


async def _judge_run(run: dict, ow, runner: LunaJudge, run_sem: asyncio.Semaphore) -> None:
    out = SCORES / (run["run_key"].replace(":", "__").replace("/", "__") + ".jsonl")
    if out.exists():
        return
    async with run_sem:
        loop = asyncio.get_running_loop()
        eval_records = await loop.run_in_executor(None, load_file, ow, run["eval_file"])
        completions = await loop.run_in_executor(None, load_file, ow, run["completions_file"])
        pairs = build_requests(eval_records, completions)
        results = await asyncio.gather(*(runner.judge_one(req, strip_think(text)) for req, text in pairs))
        rows = []
        for (req, _), scores in zip(pairs, results):
            for s in scores:
                rows.append({
                    RESULT_FIELD_COMPLETION_ID: req.completion_id, "eval_id": req.eval_id,
                    RESULT_FIELD_AXIS: req.axis, RESULT_FIELD_SCORE_NAME: s.score_name,
                    RESULT_FIELD_SCORE: "" if s.score is None else s.score, "score_label": s.score_label,
                    "judge_raw": s.score_source_text[:200],
                })
        n_err = sum(r["score_label"] == "ERROR" for r in rows)
        if n_err > MAX_ERROR_FRACTION * len(rows):
            print(f"NOT saving {run['run_key']}: {n_err}/{len(rows)} judge errors, e.g. "
                  f"{next(r['judge_raw'] for r in rows if r['score_label'] == 'ERROR')}", flush=True)
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"judged {run['run_key']} ({run['task']}/{run['model_family']}/{run['condition']}/seed{run['seed']}): "
              f"{len(rows)} scores, {n_err} errors", flush=True)


def old_axis(old: dict | None, axis: str, filtered: bool) -> float | None:
    if not old:
        return None
    if axis == "capability":
        return old.get("capability_coherence_filtered_mean" if filtered else "capability_mean")
    for k, v in old.items():
        if "generalization" in k and k.endswith("coherence_filtered_mean" if filtered else "generalization_mean"):
            return v
    return None


def ip_variants_old_summary(run: dict) -> dict | None:
    """Old (deepseek) summary for IP-variant targets, recomputed from judge.py's scores."""
    target = run["run_key"].split(":", 1)[1]
    p = IPV / "results/scores" / (target.replace("/", "__") + ".jsonl")
    if not p.exists():
        return None
    rows = [{**r, RESULT_FIELD_SCORE: "" if r["score"] is None else r["score"]} for r in read_jsonl_text(p.read_text())]
    s = summarize(rows)
    return {"capability_mean": s.get("capability_mean"), "capability_coherence_filtered_mean": s.get("capability_coh_mean"),
            "undesired_generalization_mean": s.get("ug_mean"), "undesired_generalization_coherence_filtered_mean": s.get("ug_coh_mean")}


def write_summary(inventory: list[dict]) -> None:
    import pandas as pd

    rows = []
    for run in inventory:
        p = SCORES / (run["run_key"].replace(":", "__").replace("/", "__") + ".jsonl")
        if not run.get("rejudge") or not p.exists():
            continue
        new = summarize(read_jsonl_text(p.read_text()))
        old = run["old_summary"] if run["source"] != "ip_variants" else ip_variants_old_summary(run)
        rows.append({
            "task": run["task"], "model": run["model_family"], "condition": run["condition"], "seed": run["seed"],
            "variant": run["variant"], "source": run["source"], "in_writeup": run["in_writeup"], "model_id": run["model_id"],
            "cap_key": new.get("capability_key"), "ug_key": new.get("ug_key"),
            "cap_deepseek": old_axis(old, "capability", False), "cap_luna": new.get("capability_mean"),
            "ug_deepseek": old_axis(old, "ug", False), "ug_luna": new.get("ug_mean"),
            "ug_coh_deepseek": old_axis(old, "ug", True), "ug_coh_luna": new.get("ug_coh_mean"),
            "ug_coh_n_luna": new.get("ug_coh_n"),
        })
    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["task", "condition", "model", "seed", "variant"])
    df.to_csv(RESULTS / "rejudge_summary.csv", index=False)
    print(f"wrote {RESULTS / 'rejudge_summary.csv'} ({len(df)} runs)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--source")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv
    import logging

    load_dotenv(REPO / ".env")
    from common import get_ow

    for name in ("httpx", "openai", "openai._base_client"):  # one line per request otherwise
        logging.getLogger(name).setLevel(logging.WARNING)

    inventory = json.loads((STATE / "inventory.json").read_text())
    todo = [r for r in inventory if r.get("rejudge")
            and (not args.task or r["task"] == args.task) and (not args.source or r["source"] == args.source)]
    if not args.summary_only:
        ow = get_ow()
        config = {
            "judge_model": JUDGE_MODEL,
            "judge_api_key": os.environ["LITELLM_API_KEY"],
            "judge_base_url": os.environ.get("LITELLM_BASE_URL", "https://litellm.nielsrolf.com"),
            "llm_judge_response_max_tokens": 2000,
        }
        print(f"{len(todo)} runs in scope; {sum(1 for r in todo if not (SCORES / (r['run_key'].replace(':', '__').replace('/', '__') + '.jsonl')).exists())} to judge", flush=True)

        async def run_all():
            # semaphores must be created inside the running event loop
            runner = LunaJudge(config, asyncio.Semaphore(CONCURRENT_CALLS))
            run_sem = asyncio.Semaphore(CONCURRENT_RUNS)
            await asyncio.gather(*(judge_run(r, ow, runner, run_sem) for r in todo))
            if runner._client is not None:
                await runner._client.close()

        asyncio.run(run_all())
    write_summary(inventory)


if __name__ == "__main__":
    main()
