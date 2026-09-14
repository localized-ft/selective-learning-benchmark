"""Regenerate the frozen main analysis from raw repository objects, offline.

Usage: python scripts/analysis/reproduce.py --output /tmp/slb-reproduction
Only NumPy is required beyond the standard library. Archived tables are read
AFTER regeneration, solely as independent expected results for verification.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import html
import itertools
import json
import math
from pathlib import Path
import socket

import numpy as np
from archive import Archive, RESULT
from metrics import (AXES, CATEGORIES, METHODS, METRICS, MODELS, TASKS, VARIANTS,
                     aggregate, coherence_cache, normalize, pareto, relation, table)
from diagnostics import KEYS as DIAGNOSTIC_KEYS, supplement


def write_rows(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in data for k in row))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def intervals(row, point, boot):
    lower, upper = np.quantile(boot, [.025, .975], axis=0)
    for index, name in enumerate(METRICS):
        row[name] = float(point[index])
        row[name + "_ci_low"] = float(lower[index])
        row[name + "_ci_high"] = float(upper[index])
    return row


def regenerate(archive):
    manifest = json.loads(archive.analysis("sources/fetch_manifest.json"))
    if len(manifest["jobs"]) != 630: raise ValueError("Unexpected run inventory")
    cache = coherence_cache(archive)
    template = archive.analysis("sources/coherence_rubric.txt").decode()
    outputs = {name:[] for name in ("comparisons", "coverage", "prompt_scores", "completion_scores", "threshold_sensitivity")}
    counts = Counter()
    for number, job in enumerate(manifest["jobs"], 1):
        data = archive.analysis(job["path"])
        import hashlib
        if hashlib.sha256(data).hexdigest() != job["sha256"]:
            raise ValueError("Selected score source differs from historical release")
        items = normalize(data, job, cache, template)
        meta = {k:job[k] for k in ("task_id", "model_family", "method", "category")}
        meta["seed"] = int(job["seed"])
        for item in items:
            counts["completions"] += 1
            counts["missing_primary"] += item["primary"] is None
            counts["coherence_added"] += item["coherence_source"] == "backfill_cache"
            if item["primary"] is not None and item["coherence"] is None:
                raise ValueError("Missing coherence on a valid primary score")
            outputs["completion_scores"].append({**meta, **{k:item[k] for k in (
                "axis", "completion_id", "eval_id", "group_id", "score_name", "primary", "coherence", "coherence_source")}})
        for variant, (paper, cutoff) in VARIANTS.items():
            row = {**meta, "variant":variant, "completion_job_id":job["completion_job_id"],
                   "judge_job_id":job.get("judge_job_id", ""), "source_file":job["path"], "source_sha256":job["sha256"]}
            for axis, metric in zip(AXES, METRICS):
                group = [i for i in items if i["axis"] == axis]
                values, prompts = aggregate(group, job["category"], axis, paper, cutoff)
                row[metric] = values["value"]
                row.update({metric + "_" + k:v for k,v in values.items() if k != "value"})
                outputs["coverage"].append({**meta, "variant":variant, "axis":axis, **values})
                for eval_id in sorted({i["eval_id"] for i in group}):
                    samples = prompts.get(eval_id, [])
                    outputs["prompt_scores"].append({**meta, "variant":variant, "axis":axis, "eval_id":eval_id,
                        "retained_n":len(samples), "value":sum(samples)/len(samples) if samples else float("nan")})
            outputs["comparisons"].append(row)
        for axis in AXES:
            group = [i for i in items if i["axis"] == axis]
            for cutoff, paper in itertools.product((None, 30, 50, 70), (False, True)):
                values, _ = aggregate(group, job["category"], axis, paper, cutoff)
                for weight in ("completion_weighted", "prompt_balanced"):
                    outputs["threshold_sensitivity"].append({**meta, "axis":axis,
                        "measure":"paper" if paper else "current", "coherence_cutoff":cutoff,
                        "weighting":weight, "value":values[weight + "_value"],
                        "retained_n":values["retained_n"], "retained_prompts":values["retained_prompts"]})
        if number % 70 == 0: print(f"Normalized raw evidence for {number}/630 runs", flush=True)
    panel = defaultdict(list)
    for row in outputs["comparisons"]:
        panel[tuple(row[k] for k in ("variant", "task_id", "model_family", "seed"))].append(row)
    for group in panel.values():
        if len(group) != 6: raise ValueError("Incomplete method panel")
        flags = pareto([[r[m] for m in METRICS] for r in group])
        for row, flag in zip(group, flags): row["on_seed_panel_front"] = bool(flag)
    return outputs, counts


def summaries(outputs, draws):
    cells = list(itertools.product(TASKS, MODELS))
    selection = np.random.default_rng(20260908).integers(0, 5, size=(draws, 21, 5))
    for name in ("method_summary", "cell_summary", "pairwise_comparisons", "leave_one_dataset_out", "capability_retention"):
        outputs[name] = []
    lookup = {(r["variant"], r["task_id"], r["model_family"], r["seed"], r["method"]): r for r in outputs["comparisons"]}
    for variant in VARIANTS:
        values = np.array([[lookup[variant, task, model, seed, method][metric] for metric in METRICS]
                           for task,model in cells for seed in range(1, 6) for method in METHODS]).reshape(21, 5, 6, 2)
        if not np.isfinite(values).all(): raise ValueError("Non-finite run aggregate")
        sampled = values[np.arange(21)[None,:,None], selection].mean(axis=2)
        means = values.mean(axis=1)
        front = pareto(means)
        front_frequency = pareto(sampled).mean(axis=0)
        for c, (task, model) in enumerate(cells):
            for m, method in enumerate(METHODS):
                row = {"variant":variant, "task_id":task, "model_family":model, "method":method,
                       "n_seeds":5, "on_front":bool(front[c,m]), "frontier_resampling_frequency":float(front_frequency[c,m])}
                intervals(row, means[c,m], sampled[:,c,m])
                for a, metric in enumerate(METRICS): row[metric+"_seed_sd"] = float(values[c,:,m,a].std(ddof=1))
                outputs["cell_summary"].append(row)
        groups = [("overall", "All datasets", list(range(21)))]
        groups += [("category", cat, [i for i,(t,_) in enumerate(cells) if CATEGORIES[TASKS.index(t)] == cat]) for cat in dict.fromkeys(CATEGORIES)]
        groups += [("dataset", task, [i for i,(t,_) in enumerate(cells) if t == task]) for task in TASKS]
        groups += [("model", model, [i for i,(_,m) in enumerate(cells) if m == model]) for model in MODELS]
        for level, group, selected in groups:
            point = means[selected].mean(axis=0)
            bootstrap = sampled[:,selected].mean(axis=1)
            pf, bf = pareto(point), pareto(bootstrap).mean(axis=0)
            for m, method in enumerate(METHODS):
                row = {"variant":variant, "level":level, "group":group, "method":method,
                       "n_cells":len(selected), "n_runs":len(selected)*5, "on_front":bool(pf[m]),
                       "frontier_resampling_frequency":float(bf[m])}
                outputs["method_summary"].append(intervals(row, point[m], bootstrap[:,m]))
            for a,b in itertools.combinations(range(6), 2):
                diff = point[a] - point[b]
                sampled_diff = bootstrap[:,a] - bootstrap[:,b]
                row = {"variant":variant, "level":level, "group":group, "method_a":METHODS[a], "method_b":METHODS[b], "n_paired_runs":len(selected)*5}
                for name, flags in zip(("a_dominates", "b_dominates", "tradeoff", "tie"), relation(values[selected,:,a], values[selected,:,b])):
                    row[name] = int(flags.sum())
                for name, value in intervals({}, diff, sampled_diff).items(): row["delta_" + name] = value
                row["a_macro_dominance_resampling_frequency"] = float(relation(bootstrap[:,a], bootstrap[:,b])[0].mean())
                outputs["pairwise_comparisons"].append(row)
        for omitted in TASKS:
            point = means[[i for i,(t,_) in enumerate(cells) if t != omitted]].mean(axis=0)
            for method, value, flag in zip(METHODS, point, pareto(point)):
                outputs["leave_one_dataset_out"].append({"variant":variant, "omitted_dataset":omitted, "method":method,
                    "capability":float(value[0]), "unwanted_generalization":float(value[1]), "on_front":bool(flag)})
        for fraction, floor in itertools.product((.5,.75,.9,.95,1.), (0.,.1)):
            for m, method in enumerate(METHODS[1:], 1):
                feasible = means[:,m,0] >= np.maximum(fraction*means[:,0,0], floor)
                improved = feasible & (means[:,m,1] <= means[:,0,1] - .01)
                outputs["capability_retention"].append({"variant":variant, "method":method,
                    "relative_sft_capability_floor":fraction, "absolute_capability_floor":floor,
                    "ug_improvement_required":.01, "n_cells":21, "feasible_cells":int(feasible.sum()),
                    "feasible_and_improved_cells":int(improved.sum()),
                    "mean_ug_reduction_on_feasible_cells":float((means[:,0,1]-means[:,m,1])[feasible].mean()) if feasible.any() else float("nan")})


KEYS = {
    "comparisons":["variant","task_id","model_family","seed","method"],
    "coverage":["variant","task_id","model_family","seed","method","axis"],
    "prompt_scores":["variant","task_id","model_family","seed","method","axis","eval_id"],
    "completion_scores":["task_id","model_family","seed","method","axis","completion_id"],
    "threshold_sensitivity":["task_id","model_family","seed","method","axis","measure","coherence_cutoff","weighting"],
    "method_summary":["variant","level","group","method"],
    "cell_summary":["variant","task_id","model_family","method"],
    "pairwise_comparisons":["variant","level","group","method_a","method_b"],
    "leave_one_dataset_out":["variant","omitted_dataset","method"],
    "capability_retention":["variant","method","relative_sft_capability_floor","absolute_capability_floor"],
}
KEYS.update(DIAGNOSTIC_KEYS)


def canon(value):
    if isinstance(value, (bool, np.bool_)) or value in ("True", "False"):
        return str(value)
    if value is None or value == "": return ""
    try:
        x = float(value)
        return "" if math.isnan(x) else str(x)
    except (ValueError, TypeError): return str(value)


def compare(name, generated, expected):
    keys = KEYS[name]
    keyed = lambda rows: {tuple(canon(r[k]) for k in keys):r for r in rows}
    actual, reference = keyed(generated), keyed(expected)
    if len(actual) != len(generated) or len(reference) != len(expected): raise ValueError("Duplicate key: " + name)
    if actual.keys() != reference.keys(): raise ValueError("Row identities differ: " + name)
    maximum = 0.
    checks = 0
    for key, expected_row in reference.items():
        row = actual[key]
        for field, value in expected_row.items():
            if field not in row: raise ValueError("Missing field: " + name + "/" + field)
            left, right = canon(row[field]), canon(value)
            if left == right: continue
            try: difference = abs(float(left) - float(right))
            except ValueError: raise ValueError(f"Value mismatch: {name}/{key}/{field}: {left!r} != {right!r}") from None
            if not math.isfinite(difference) or difference > 1e-12:
                raise ValueError(f"Numeric mismatch: {name}/{key}/{field}: {difference}")
            maximum = max(maximum, difference)
            checks += 1
    return {"rows":len(reference), "max_abs_difference":maximum, "tolerance":1e-12}


def expected_table(archive, name):
    if name in ("comparisons", "method_summary", "cell_summary"):
        path = "outputs/" + name + ".csv"
    elif name in ("completion_scores", "prompt_scores"):
        return table(gzip.decompress(archive.analysis("outputs/reproducibility/" + name + ".csv.gz")))
    else: path = "outputs/diagnostics/" + name + ".csv"
    return table(archive.analysis(path))


def svg_plot(path, groups, title):
    """Deterministic, dependency-free scientific scatter plots, shared 0–100 axes."""
    colors = ["#707780", "#a5743d", "#ccaa39", "#8856a7", "#24865c", "#287bb5"]
    overview = len(groups) == 4
    columns = 2 if overview else 3
    width, cell_w, cell_h = 1080, (540 if overview else 360), (480 if overview else 290)
    plot_w, plot_h = (450, 380) if overview else (250, 200)
    height = 1080 if overview else 80 + math.ceil(len(groups)/columns)*cell_h
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>', '<g font-family="sans-serif" font-size="11" fill="#222">',
           f'<text x="25" y="25" font-size="18">{html.escape(title)}</text>']
    for i, method in enumerate(METHODS):
        svg.append(f'<text x="{25+i*175}" y="49" fill="{colors[i]}">{html.escape(method)}</text>')
    for i, (label, rows_) in enumerate(groups):
        x0, y0 = 50+(i%columns)*cell_w, 110+(i//columns)*cell_h
        svg.append(f'<text x="{x0}" y="{y0-18}" font-weight="bold">{html.escape(label)}</text>')
        for tick in range(0, 101, 25):
            x, y = x0+plot_w*tick/100, y0+plot_h-plot_h*tick/100
            svg += [f'<path d="M{x},{y0}v{plot_h} M{x0},{y}h{plot_w}" stroke="#ddd" fill="none"/>',
                    f'<text x="{x-6}" y="{y0+plot_h+16}">{tick}</text>', f'<text x="{x0-25}" y="{y+4}">{tick}</text>']
        svg += [f'<text x="{x0+plot_w/2-45}" y="{y0+plot_h+35}">Capability (higher)</text>',
                f'<text transform="translate({x0-34},{y0+plot_h/2+72}) rotate(-90)">Unwanted generalization (lower)</text>']
        for row in rows_:
            m = METHODS.index(row["method"])
            x, y = x0+plot_w*row[METRICS[0]], y0+plot_h-plot_h*row[METRICS[1]]
            for metric, horizontal in ((METRICS[0],True), (METRICS[1],False)):
                low, high = row[metric+"_ci_low"], row[metric+"_ci_high"]
                d = f"M{x0+plot_w*low},{y}H{x0+plot_w*high}" if horizontal else f"M{x},{y0+plot_h-plot_h*low}V{y0+plot_h-plot_h*high}"
                svg.append(f'<path d="{d}" stroke="{colors[m]}"/>')
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[m]}"/>')
            if row["on_front"]: svg.append(f'<circle cx="{x}" cy="{y}" r="7" fill="none" stroke="{colors[m]}"/>')
    svg.append('</g></svg>')
    path.write_text("\n".join(svg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-archive", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if RESULT.resolve() == output or RESULT.resolve() in output.parents:
        raise ValueError("Regenerate outside archived result/; never overwrite the release")
    if output.exists() and any(output.iterdir()): raise ValueError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    # Enforce offline execution even when the machine has network access.
    def network_forbidden(*args, **kwargs): raise RuntimeError("Network is disabled during reproduction")
    socket.socket = network_forbidden
    socket.create_connection = network_forbidden
    archive = Archive()
    verification = {"network_disabled":True}
    from supplemental import verify as verify_supplemental
    supplemental_result = verify_supplemental(archive)
    (output / "qwen3_api_reference.json").write_text(json.dumps(supplemental_result, indent=2) + "\n")
    verification["supplemental_reference"] = {k:v for k,v in supplemental_result.items() if k != "variants"}
    if args.verify_archive: verification["archive"] = archive.verify()
    outputs, counts = regenerate(archive)
    summaries(outputs, 10000)
    supplement(outputs)
    verification["tables"] = {}
    for name, data in outputs.items():
        verification["tables"][name] = compare(name, data, expected_table(archive, name))
        suffix = ".csv.gz" if name in ("completion_scores", "prompt_scores") else ".csv"
        write_rows(output / (name + suffix), data)
        print("Reproduced and verified", name, len(data), flush=True)
    for variant in VARIANTS:
        groups = [(task + " / " + model, [r for r in outputs["cell_summary"] if r["variant"] == variant and r["task_id"] == task and r["model_family"] == model])
                  for task,model in itertools.product(TASKS, MODELS)]
        svg_plot(output / ("pareto_" + variant + ".svg"), groups, variant + " — means and 95% seed-bootstrap intervals")
    svg_plot(output / "overview.svg", [(v,[r for r in outputs["method_summary"] if r["variant"] == v and r["level"] == "overall"]) for v in VARIANTS], "Four scoring variants: aggregate Pareto comparison")
    report = ["# Offline reproduction", "", "Regenerated from raw archived scores and coherence backfills. Values below use a 0–100 scale. All seven tasks, three models, six methods, and five seeds are retained.", "",
              "![Overview](overview.svg)", "", "| Variant | Method | Capability | Unwanted generalization | On frontier |", "|---|---|---:|---:|---|"]
    for r in outputs["method_summary"]:
        if r["level"] == "overall": report.append(f"| {r['variant']} | {r['method']} | {100*r['capability']:.2f} | {100*r['unwanted_generalization']:.2f} | {r['on_front']} |")
    report += ["", "The original frozen report and figures remain in the repository release. These newly drawn figures use the same verified data and intervals but a portable SVG layout. A frontier compares the submitted configurations, not every possible hyperparameter setting.", "",
               "Training-seed bootstrap: 10,000 draws, NumPy default_rng seed 20260908, paired methods/axes within each fixed task–model cell. Coherence filtering changes the evaluated population. Missing scores are not zero."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    verification.update(counts=dict(counts), trained_runs=630, variants=list(VARIANTS), bootstrap_draws=10000,
                        bootstrap_seed=20260908, numpy_version=np.__version__, success=True)
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"success":True, "output":str(output), "counts":dict(counts)}, indent=2))


if __name__ == "__main__": main()
