"""Offline extension of the verified five-seed analysis to 21 vanilla references.

Never launches inference/judging or overwrites a release. The original vanilla
judge pass is primary; outcome-selected refusal retries are not pooled into it.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import socket

import numpy as np
from archive import Archive, PREFIX, RESULT
from metrics import AXES, CATEGORIES, METHODS, METRICS, MODELS, TASKS, VARIANTS, aggregate, normalize, pareto
from reproduce import regenerate, summaries, compare, expected_table, write_rows
from diagnostics import supplement, grouped

ALL_METHODS = ["vanilla"] + METHODS
VANILLA = "supplemental/vanilla_api_20260914"
CATEGORY = dict(zip(TASKS, CATEGORIES))


def csv_bytes(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def restore_missing(items, requests, excluded):
    """Join planned slots, rejecting unaccounted absence and conflicting identities."""
    by_id = {r["completion_id"]: r for r in items}
    plan = {r["completion_id"]: r for r in requests}
    if len(by_id) != len(items) or len(plan) != len(requests):
        raise ValueError("Duplicate completion ID within model/task")
    missing = set(plan) - set(by_id)
    if set(by_id) - set(plan) or missing != set(excluded):
        raise ValueError("Inference inventory does not reconcile")
    for cid, r in by_id.items():
        req = plan[cid]
        if r["eval_id"] != req["eval_id"] or r["axis"] != req["axis"].replace("undesired_generalization", AXES[1]):
            raise ValueError("Inference/judge identity mismatch")
        r["inference_status"] = "usable"
    for cid in sorted(missing):
        req = plan[cid]
        by_id[cid] = {"completion_id": cid, "eval_id": req["eval_id"],
            "axis": req["axis"].replace("undesired_generalization", AXES[1]),
            "primary": None, "coherence": None, "coherence_source": "missing",
            "score_name": "", "inference_status": "provider_filtered"}
    return list(by_id.values())


def vanilla_scores(archive):
    evidence = []

    def read(rel, jsonl=False):
        data = (archive.result / rel).read_bytes()
        evidence.append({"path": rel, "stored_sha256": hashlib.sha256(data).hexdigest()})
        if rel.endswith(".gz"):
            data = gzip.decompress(data)
        return [json.loads(s) for s in data.splitlines() if s.strip()] if jsonl else json.loads(data)

    scores = read(VANILLA + "/judge/scores.jsonl.gz", True)
    if len(scores) != 21746:
        raise ValueError("Unexpected primary judge inventory")
    score_ids = {(r["model_family"], r["task_id"], r["completion_id"], r["score_name"]) for r in scores}
    if len(score_ids) != len(scores):
        raise ValueError("Duplicate primary-pass score")
    requests = {}
    for model in MODELS:
        rel = "/olmo3/vllm/requests.jsonl" if model == "olmo3_7b" else f"/api/{model}/requests.jsonl"
        requests[model] = read(VANILLA + rel, True)
    excluded = read(VANILLA + "/judge/excluded_inference.json")
    old_path = PREFIX + "batches/base_qwen3_bad_medical_20260909/eval_results.csv"
    old = archive.source(old_path)
    evidence.append({"source_path": old_path, "content_sha256": hashlib.sha256(old).hexdigest()})
    output = {k: [] for k in ("comparisons", "coverage", "prompt_scores", "completion_scores")}
    all_items = {}
    for task, model in itertools.product(TASKS, MODELS):
        meta = {"task_id": task, "model_family": model, "category": CATEGORY[task],
                "method": "vanilla", "seed": None, "cohort": "vanilla_reference"}
        if (task, model) == ("bad_medical_advice", "qwen3_8b"):
            items = normalize(old, meta, {}, "")
            assert len(items) == 760
            for item in items:
                item["inference_status"] = "provider_filtered" if item["completion_id"] == "eval_0071_sample_0000" else "usable"
        else:
            rows = [r for r in scores if (r["task_id"], r["model_family"]) == (task, model)]
            items = normalize(csv_bytes(rows), meta, {}, "")
            if len(rows) != 2 * len(items) or any(i["coherence"] is None for i in items):
                raise ValueError("Incomplete primary/coherence pair")
            plan = [r for r in requests[model] if r["task_id"] == task]
            failures = {r["completion_id"] for r in excluded if (r["task_id"], r["model_family"]) == (task, model)}
            items = restore_missing(items, plan, failures)
        all_items[task, model] = items
        for item in items:
            output["completion_scores"].append({**meta, **{k: item.get(k) for k in
                ("axis", "completion_id", "eval_id", "score_name", "primary", "coherence", "coherence_source", "inference_status")}})
        for variant, (paper, cutoff) in VARIANTS.items():
            row = {**meta, "variant": variant}
            for axis, metric in zip(AXES, METRICS):
                group = [r for r in items if r["axis"] == axis]
                values, prompts = aggregate(group, CATEGORY[task], axis, paper, cutoff)
                row[metric] = values["value"]
                row.update({metric + "_" + k: v for k, v in values.items() if k != "value"})
                output["coverage"].append({**meta, "variant": variant, "axis": axis, **values,
                    "provider_filtered_n": sum(r["inference_status"] == "provider_filtered" for r in group),
                    "usable_missing_primary_n": sum(r["inference_status"] == "usable" and r["primary"] is None for r in group)})
                for eval_id in sorted({i["eval_id"] for i in group}):
                    samples = prompts.get(eval_id, [])
                    output["prompt_scores"].append({**meta, "variant": variant, "axis": axis, "eval_id": eval_id,
                        "retained_n": len(samples), "value": float(np.mean(samples)) if samples else float("nan")})
            output["comparisons"].append(row)
    counts = Counter(r["inference_status"] for r in output["completion_scores"])
    assert counts == {"usable": 11632, "provider_filtered": 8}, counts
    assert sum(r["primary"] is None and r["inference_status"] == "usable" for r in output["completion_scores"]) == 189
    return output, evidence


def extend_summaries(trained, vanilla):
    """Preserve trained CIs; hold vanilla fixed, never manufacture training seeds."""
    cell = []
    for row in trained["cell_summary"]:
        cell.append({**row, "on_trained_front": row["on_front"],
                     "trained_frontier_resampling_frequency": row["frontier_resampling_frequency"],
                     "uncertainty": "training_seed_bootstrap"})
    for row in vanilla["comparisons"]:
        cell.append({**{k: row[k] for k in ("variant", "task_id", "model_family", "method", *METRICS)},
                     "n_seeds": 0, "n_reference_evaluations": 1, "uncertainty": "not_estimated"})
    for rows in grouped(cell, ["variant", "task_id", "model_family"]).values():
        for r, flag in zip(rows, pareto([[r[m] for m in METRICS] for r in rows])):
            r["on_front"] = bool(flag)
    macro = []
    for row in trained["method_summary"]:
        macro.append({**row, "on_trained_front": row["on_front"],
            "trained_frontier_resampling_frequency": row["frontier_resampling_frequency"],
            "uncertainty": "training_seed_bootstrap"})
    for (variant, level, group), rows in grouped(macro, ["variant", "level", "group"]).items():
        selected = [r for r in cell if r["variant"] == variant and r["method"] == "vanilla" and
            (level == "overall" or (level == "category" and CATEGORY[r["task_id"]] == group) or
             (level == "dataset" and r["task_id"] == group) or (level == "model" and r["model_family"] == group))]
        v = {"variant": variant, "level": level, "group": group, "method": "vanilla", "n_seeds": 0,
             "n_cells": len(selected), "n_runs": 0, "n_reference_evaluations": len(selected),
             "uncertainty": "not_estimated", **{m: float(np.mean([r[m] for r in selected])) for m in METRICS}}
        for r, flag in zip(rows + [v], pareto([[r[m] for m in METRICS] for r in rows + [v]])):
            r["on_front"] = bool(flag)
        macro.append(v)
    # These probabilities refer to the *trained-only* frontier, not the new one.
    for r in cell + macro:
        r.pop("frontier_resampling_frequency", None)
    return cell, macro


def contrasts(cell):
    result = []
    for key, rows in grouped(cell, ["variant", "task_id", "model_family"]).items():
        by_method = {r["method"]: r for r in rows}
        v, sft = by_method["vanilla"], by_method["baseline"]
        for method in METHODS:
            r = by_method[method]
            gain = sft["capability"] - v["capability"]
            result.append({**dict(zip(("variant", "task_id", "model_family"), key)), "method": method,
                "category": CATEGORY[r["task_id"]],
                **{"delta_from_vanilla_" + m: r[m] - v[m] for m in METRICS},
                **{"delta_from_sft_" + m: r[m] - sft[m] for m in METRICS},
                "sft_capability_gain_over_vanilla": gain,
                "positive_sft_gain_at_least_5pp": gain >= .05,
                "retains_90pct_sft_gain": (r["capability"] - v["capability"] >= .9 * gain) if gain >= .05 else None,
                "reduces_ug_vs_sft_at_least_1pp": r["unwanted_generalization"] <= sft["unwanted_generalization"] - .01,
                "retains_90pct_sft_absolute_capability": r["capability"] >= .9 * sft["capability"]})
    return result


def common_prompts(trained, vanilla):
    """Same retained prompts in all 30 trained runs and the vanilla reference.

    This is an explicit prompt-balanced sensitivity, also for 'current' labels;
    it does not replace the completion-weighted primary current estimator.
    """
    results = []
    for key, rows in grouped(trained["prompt_scores"] + vanilla["prompt_scores"],
                             ["variant", "task_id", "model_family", "axis"]).items():
        by_run = grouped(rows, ["method", "seed"])
        assert len(by_run) == 31
        inventories = [{r["eval_id"] for r in rr} for rr in by_run.values()]
        if any(s != inventories[0] for s in inventories):
            raise ValueError("Vanilla/trained prompt universe differs")
        common = set.intersection(*[{r["eval_id"] for r in rr if math.isfinite(r["value"])} for rr in by_run.values()])
        for method in ALL_METHODS:
            selected = [rr for (m, _), rr in by_run.items() if m == method]
            vals = [float(np.mean([r["value"] for r in rr if r["eval_id"] in common])) for rr in selected] if common else []
            results.append({**dict(zip(("variant", "task_id", "model_family", "axis"), key)),
                "method": method, "common_prompt_n": len(common), "total_prompt_n": len(inventories[0]),
                "value": float(np.mean(vals)) if vals else float("nan"),
                "weighting": "prompt_balanced_common_to_all_31"})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == RESULT.resolve() or output == RESULT.parent.resolve():
        raise ValueError("Use a dedicated new release directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; releases are immutable")
    output.mkdir(parents=True, exist_ok=True)
    def forbidden(*args, **kwargs):
        raise RuntimeError("Network disabled during analysis")
    socket.socket = socket.create_connection = forbidden
    archive = Archive()
    trained, counts = regenerate(archive)
    summaries(trained, 10000)
    supplement(trained)
    from supplemental import verify as verify_old_qwen
    verification = {"historical_tables": {}, "historical_qwen_reference": verify_old_qwen(archive)}
    for name, rows in trained.items():
        verification["historical_tables"][name] = compare(name, rows, expected_table(archive, name))
        print("Verified historical table", name, flush=True)
    vanilla, sources = vanilla_scores(archive)
    cell, macro = extend_summaries(trained, vanilla)
    changes = contrasts(cell)
    common = common_prompts(trained, vanilla)
    trained_runs = [{**r, "cohort": "trained"} for r in trained["comparisons"]]
    for row in trained_runs:
        row["on_trained_run_front"] = row.pop("on_seed_panel_front")
    tables = {
        "comparisons": trained_runs + vanilla["comparisons"],
        "cell_summary": cell, "method_summary": macro, "vanilla_contrasts": changes,
        "coverage": trained["coverage"] + vanilla["coverage"],
        "common_prompt_sensitivity": common,
        "vanilla_completion_scores": vanilla["completion_scores"],
        "vanilla_prompt_scores": vanilla["prompt_scores"],
        "trained_pairwise_comparisons": trained["pairwise_comparisons"],
        "trained_capability_retention": trained["capability_retention"],
    }
    for name, rows in tables.items():
        write_rows(output / "tables" / (name + ".csv"), rows)
    producer_files = ["with_vanilla.py", "vanilla_plots.py", "reproduce.py", "metrics.py", "archive.py", "diagnostics.py", "supplemental.py"]
    (output / "sources.json").write_text(json.dumps({"vanilla": sources,
        "producer_sha256": {"scripts/analysis/"+p: hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest() for p in producer_files},
        "trained_fetch_manifest_content_sha256": hashlib.sha256(archive.analysis("sources/fetch_manifest.json")).hexdigest(),
        "trained_sources": "../../migration/source_manifest.jsonl",
        "judge_selection": "original pass only; both refusal retry passes excluded"}, indent=2) + "\n")
    from vanilla_plots import figures, report
    figures(output, cell, macro, tables["comparisons"], changes)
    report(output, tables)
    verification.update(success=True, network_disabled=True, trained_runs=630, trained_seeds=[1,2,3,4,5],
        historical_counts=dict(counts), vanilla_cells=21, vanilla_planned_slots=11640,
        vanilla_usable_outputs=11632, vanilla_provider_filtered=8, vanilla_missing_numeric_primary_on_usable=189,
        vanilla_is_not_a_training_seed=True, refusal_retries_used=False,
        common_prompt_universes_match=True, bootstrap_draws=10000, bootstrap_seed=20260908,
        numpy_version=np.__version__, tables={k:len(v) for k,v in tables.items()})
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"success": True, "output": str(output), "tables": verification["tables"]}, indent=2))


if __name__ == "__main__":
    main()
