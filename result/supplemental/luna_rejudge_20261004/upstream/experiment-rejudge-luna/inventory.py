"""Inventory every eval whose completions should be re-judged with gpt-6-luna.

Sources (verified 2026-09-30):
  seed1        159 July `eval_worker` jobs (+2 August judge jobs) referenced by job id in
               sunday/scripts/eval/plot_v2_tradeoffs.py (all conditions incl. KLD / IP / probes)
  judge        August/September `judge_worker` jobs: seeds 2-5 of baseline + the three layer
               thirds (plus a few bf16 / probe extras)
  completion   `completion_worker` jobs for KLD / IP seeds 2-5; these were judged outside
               OpenWeights, so there is no old per-run summary for them here
  ip_variants  niels/experiment-ip-variants eval targets (local completions files)

For each run we record: task, model family, condition, seed, the exact eval file the run used,
the completions file, and (if available) the old deepseek eval_summary.

Usage: ../experiment-ip-variants/.venv/bin/python inventory.py   # writes state/inventory.json
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IPV = REPO / "niels" / "experiment-ip-variants"
sys.path.insert(0, str(IPV))
from common import get_ow  # noqa: E402  (reuses the openweights Job-dataclass patch)

STATE = HERE / "state"
STATE.mkdir(exist_ok=True)

MODEL_FAMILIES = {"Llama-3.1-8B": "llama31_8b", "Qwen3-8B": "qwen3_8b", "OLMo-3-7B": "olmo3_7b", "Qwen3-32B": "qwen3_32b"}
PLOT_MODELS = {"Llama 3.1 8B": "llama31_8b", "Qwen3-8B": "qwen3_8b", "OLMo 3 7B": "olmo3_7b"}
TASK_SLUGS = {
    # model-id slug -> task name used in the write-up charts
    "bad-medical-advice": "bad_medical_advice",
    "risky-financial-advice": "risky_financial_advice",
    "school-of-reward-hacks": "school_of_reward_hacks",
    "german-city-names": "german_city_names",
    "old-bird-names": "old_bird_names",
    "good-vs-bad-mixed-multifact": "good_vs_bad_mixed_multifact",
    "target-only-no-hallucination": "target_only",
}
CONDITIONS = {
    # model-id condition slug (after removing task slug, "-v2" and "-seedN") -> chart condition
    "sft": "baseline",
    "first-third-sft": "first-third",
    "second-third-sft": "second-third",
    "last-third-sft": "last-third",
    "kld": "kld",
    "inoculation-prompting": "inoculation",
    "probe-block-sft": "probe_block",
    "probe-top10-sft": "probe_top10",
}
PERSONA_EVAL_FILE_FIX = {
    "custom_job_file:file-41f72ad37ffb": "custom_job_file:file-811fff711925",  # old_bird_names
    "custom_job_file:file-a4612ebbfc2e": "custom_job_file:file-fc7de205764e",  # german_city_names
}
SEED_OF = {None: 1, "2": 2, "3": 3, "4": 4, "5": 5}
# seed-1 write-up runs whose eval jobs are not referenced in plot_v2_tradeoffs.py (found by model id)
EXTRA_SEED1_IDS = [
    "jobs-f917a7caf095",  # Llama-3.1-8B risky-financial-advice inoculation-prompting
    "jobs-d18b944b3465",  # Llama-3.1-8B risky-financial-advice kld
    "jobs-467a26812022",  # Llama-3.1-8B school-of-reward-hacks inoculation-prompting
]


def parse_model_id(model_id: str) -> dict:
    org, name = model_id.split("/", 1)
    fam = next((f for f in MODEL_FAMILIES if name.startswith(f + "-")), None)
    task_slug = next((t for t in TASK_SLUGS if fam and name[len(fam) + 1:].startswith(t)), None)
    if fam is None or task_slug is None:  # e.g. base models or other experiments
        return {"model_family": fam and MODEL_FAMILIES[fam], "task": None, "condition": "unparsed", "seed": None, "variant": ""}
    rest = name[len(fam) + 1:]
    rest = rest[len(task_slug) + 1:]
    seed = re.search(r"-seed(\d)$", rest)
    rest = re.sub(r"-seed\d$", "", rest)
    bf16 = rest.endswith("-bf16")
    rest = re.sub(r"-bf16$", "", rest).replace("-v2", "").replace("v2-", "")
    return {
        "model_family": MODEL_FAMILIES[fam],
        "task": TASK_SLUGS[task_slug],
        "condition": CONDITIONS.get(rest, rest),
        "seed": SEED_OF[seed.group(1) if seed else None],
        "variant": "bf16" if bf16 else "",
    }


def retry(fn, tries: int = 6):
    import time
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def run_events(sb, job_id: str) -> list[dict]:
    runs = retry(lambda: sb.table("runs").select("id,status").eq("job_id", job_id).order("created_at").execute().data)
    out = []
    for r in runs:
        rows = retry(lambda: sb.table("events").select("data").eq("run_id", r["id"]).execute().data)
        out += [e["data"] for e in rows if isinstance(e["data"], dict)]
    return out


def mounted_config(ow, job: dict) -> dict:
    mf = job["params"]["mounted_files"]
    name = next(k for k in mf if k.endswith(".yaml"))
    return yaml.safe_load(retry(lambda: ow.files.content(mf[name])).decode())


def describe_job(ow, job: dict, source: str) -> dict:
    cache = STATE / "describe_cache" / f"{source}__{job['id']}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = _describe_job(ow, job, source)
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def _describe_job(ow, job: dict, source: str) -> dict:
    sb = ow._supabase
    cfg = mounted_config(ow, job)
    ev = run_events(sb, job["id"])
    completions = cfg.get("completions_file") or next(
        (e["file_id"] for e in reversed(ev) if e.get("type") == "completions_saved"), None)
    summary = next((e for e in reversed(ev) if e.get("type") == "eval_summary"), None)
    meta = parse_model_id(cfg["model"])
    return {
        "run_key": f"{source}:{job['id']}",
        "source": source,
        "job_id": job["id"],
        "model_id": cfg["model"],
        **meta,
        "task_id": cfg["task_manifest"]["task"],
        "eval_file": cfg["eval_file"],
        "completions_file": completions,
        "old_judge_model": cfg.get("judge_model"),
        "old_summary": {k: v for k, v in (summary or {}).items() if k not in ("type", "model")} or None,
        "in_writeup": meta["variant"] == "" and meta["model_family"] != "qwen3_32b"
                      and meta["condition"] in ("baseline", "first-third", "second-third", "last-third", "kld", "inoculation"),
    }


def main() -> None:
    ow = get_ow()
    sb = ow._supabase
    src = (REPO / "sunday/scripts/eval/plot_v2_tradeoffs.py").read_text()
    seed1_ids = sorted(set(re.findall(r'"(jobs-[0-9a-f]{12})"', src)) | set(EXTRA_SEED1_IDS))

    def fetch(ids):
        out = []
        for i in range(0, len(ids), 100):
            out += sb.table("jobs").select("id,script,model,params,status").in_("id", ids[i:i + 100]).execute().data
        return out

    def all_jobs(**filters):
        rows, start = [], 0
        while True:
            q = sb.table("jobs").select("id,script,model,params,status")
            for k, v in filters.items():
                q = q.ilike(k, v)
            r = q.range(start, start + 999).execute().data
            rows += r
            if len(r) < 1000:
                return rows
            start += 1000

    jobs = [("seed1", j) for j in fetch(seed1_ids)]
    seen = {j["id"] for _, j in jobs}
    jobs += [("judge", j) for j in all_jobs(script="%judge_worker%") if j["status"] == "completed" and j["id"] not in seen]
    for pat in ("%kld-seed%", "%inoculation-prompting-seed%"):
        jobs += [("completion", j) for j in all_jobs(model=pat, script="%completion_worker%") if j["status"] == "completed"]

    with cf.ThreadPoolExecutor(6) as ex:
        runs = list(ex.map(lambda sj: describe_job(ow, sj[1], sj[0]), jobs))

    # the IP-variants experiment (local completions; same eval.jsonl for all targets)
    ipv_state = json.loads((IPV / "state/state.json").read_text())
    for target, info in sorted(ipv_state.get("eval_jobs", {}).items()):
        cond_model, _, eval_prompt = target.partition("@")
        cond, fam = cond_model.split("/")
        runs.append({
            "run_key": f"ip_variants:{target}", "source": "ip_variants", "job_id": info["job_id"],
            "model_id": info["model"], "model_family": fam, "task": "bad_medical_advice",
            "condition": cond, "seed": 1, "variant": f"eval_prompt={eval_prompt}" if eval_prompt else "",
            "task_id": "emergent_misalignment-bad_medical_advice",
            "eval_file": "local:" + str(IPV.relative_to(REPO) / "../../sunday/scripts/eval/tasks/bad_medical_advice/eval.jsonl"),
            "completions_file": "local:" + str((IPV / "results/completions" / (target.replace("/", "__") + ".jsonl")).relative_to(REPO)),
            "old_judge_model": "openrouter/deepseek/deepseek-v4-flash (judge.py, <think> stripped)",
            "old_summary": None, "in_writeup": False,
        })

    # The July (seed-1) eval files of the two persona tasks have no answer_regex / score_map, so
    # the judge's "LLM" / "19th century" style labels were never converted to scores (old and new
    # judge alike). The August files have identical ids, questions and prompts plus the maps.
    for r in runs:
        fix = PERSONA_EVAL_FILE_FIX.get(r["eval_file"])
        if fix:
            r["original_eval_file"], r["eval_file"] = r["eval_file"], fix
    # the same model can appear in several sources (e.g. a seed-1 eval_worker job and a later
    # judge job): count it once for the write-up, preferring the seed-1 job used by the charts
    order = {"seed1": 0, "judge": 1, "completion": 2}
    seen_models = set()
    for r in sorted((r for r in runs if r["in_writeup"]), key=lambda r: order[r["source"]]):
        if r["model_id"] in seen_models:
            r["in_writeup"] = False
            r["duplicate_of_model"] = True
        seen_models.add(r["model_id"])
    for r in runs:
        # re-judge scope: write-up runs, the plotted seed-1 probe points, and the IP-variants evals
        r["rejudge"] = r["in_writeup"] or r["source"] == "ip_variants" or (
            r["source"] == "seed1" and r["condition"] in ("probe_block", "probe_top10"))
    (STATE / "inventory.json").write_text(json.dumps(runs, indent=1))
    import collections
    print(len(runs), "runs")
    print(collections.Counter(r["source"] for r in runs))
    wr = [r for r in runs if r["in_writeup"]]
    by = collections.Counter((r["task"], r["condition"]) for r in wr)
    print("write-up runs:", len(wr), "| distinct (model_id):", len({r["model_id"] for r in wr}))
    missing = [(t, c, n) for (t, c), n in sorted(by.items()) if n != 15]
    print("(task, condition) cells without exactly 15 runs (3 models x 5 seeds):", missing)
    print("no completions file:", [r["run_key"] for r in runs if not r["completions_file"]])
    print("re-judge scope:", sum(r["rejudge"] for r in runs), "runs")


if __name__ == "__main__":
    main()
